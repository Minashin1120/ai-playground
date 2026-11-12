package com.minashin1120.aiplayground

import android.app.Application
import android.os.Build
import android.os.SystemClock
import com.minashin1120.aiplayground.data.AccountLock
import com.minashin1120.aiplayground.data.ApiException
import com.minashin1120.aiplayground.data.BotTelemetry
import com.minashin1120.aiplayground.data.PlaygroundApi
import com.minashin1120.aiplayground.data.SendSpamCounter
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import org.json.JSONObject

/**
 * Web `isBotDetectionActive` / `sendMessage` の連投ロック / `botTelemetry` / `vibrateHelper`。
 * 連投はサーバーへロックを要求してアカウントロックを表示し、疑わしいタップは
 * `/api/bot-telemetry` へ送る（BAN になり得る）。振動は送信の合図に使う。
 */
internal class ChatBotGuard(
    private val app: Application,
    private val state: StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val scope: CoroutineScope,
    private val api: PlaygroundApi,
    private val token: () -> String,
    private val hasSession: () -> Boolean,
    private val enterBannedState: (ApiException) -> Unit,
    private val notify: (String) -> Unit,
) {
    private val botTelemetry = BotTelemetry()
    private val sendSpam = SendSpamCounter()

    /** Web `isBotDetectionActive` (the server's `bot_detection_active`); the no-account profile has no server account. */
    private val botDetectionActive: Boolean get() {
        val current = state.value
        val preferences = current.preferences ?: return false
        return hasSession() && !current.localProfile && !current.banned && preferences.botDetectionActive && !preferences.isAdmin
    }

    /**
     * Web `sendMessage` + `applyBotLockFromServer`: 8 sends within 3 s ask the server to lock the
     * account for a while (repeated locks end in a BAN), then show the lock overlay.
     */
    fun spamLocked(): Boolean {
        if (!botDetectionActive) return false
        if (sendSpam.register(SystemClock.uptimeMillis()) < SendSpamCounter.LOCK_THRESHOLD) return false
        val reason = "送信操作が速すぎるため、一時的にロックしています。"
        scope.launch {
            var remaining = 600L
            try {
                val reply = api.post("/api/bot/lock", JSONObject().put("reason", reason), token())
                if (reply.optString("status") == "skipped" || reply.optBoolean("skipped")) return@launch
                if (reply.has("remaining_seconds")) remaining = reply.optLong("remaining_seconds", remaining)
            } catch (e: CancellationException) { throw e }
            catch (e: ApiException) { if (e.code == "banned") { enterBannedState(e); return@launch } }
            catch (e: Exception) {}
            sendSpam.reset()
            mutable.update { current ->
                if (current.accountLock != null) current
                else current.copy(accountLock = AccountLock(reason, System.currentTimeMillis() + remaining.coerceAtLeast(0) * 1000))
            }
        }
        return true
    }

    /**
     * Web `sendMessage` entry: a short vibration, then the waiting notices while an answer streams or files
     * upload. Returns false when the send must stop there.
     */
    fun beginSend(): Boolean {
        vibrate(50)
        when {
            state.value.streaming -> notify("回答生成中です。完了までお待ちいただくか、停止してください。")
            state.value.uploading -> notify("ファイルの送信・処理中です。しばらくお待ちください。")
            spamLocked() -> notify("送信操作が速すぎるため、確認後に再度お試しください。")
            else -> return true
        }
        return false
    }

    /**
     * Web `botTelemetry` for touch input, fed by `MainActivity.dispatchTouchEvent`: a suspicious
     * window of taps is reported to `/api/bot-telemetry`, which may BAN the account.
     */
    fun recordTouch(action: Int, eventTime: Long) {
        if (!botDetectionActive) return
        val force = when (action) {
            android.view.MotionEvent.ACTION_DOWN -> botTelemetry.recordTap(eventTime)
            android.view.MotionEvent.ACTION_MOVE -> { botTelemetry.recordMove(eventTime); false }
            else -> return
        }
        val payload = botTelemetry.takeReport(eventTime, force) ?: return
        scope.launch {
            try { api.post("/api/bot-telemetry", payload, token()) }
            catch (e: CancellationException) { throw e }
            catch (e: ApiException) { if (e.code == "banned") enterBannedState(e) }
            catch (e: Exception) {}
        }
    }

    /** Web `vibrateHelper` (`navigator.vibrate`): one pulse of [timings] ms, or on/off/on… pulses. */
    fun vibrate(vararg timings: Long) {
        runCatching {
            val context = app
            if (!isVibrationEnabled(context)) return
            val vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                context.getSystemService(android.os.VibratorManager::class.java)?.defaultVibrator
            } else {
                @Suppress("DEPRECATION")
                context.getSystemService(android.os.Vibrator::class.java)
            }
            if (vibrator?.hasVibrator() != true) return
            vibrator.vibrate(
                if (timings.size == 1) android.os.VibrationEffect.createOneShot(timings[0], android.os.VibrationEffect.DEFAULT_AMPLITUDE)
                else android.os.VibrationEffect.createWaveform(longArrayOf(0L) + timings, -1),
            )
        }
    }
}
