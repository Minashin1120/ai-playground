package com.minashin1120.aiplayground

import android.app.Application
import android.os.SystemClock
import com.minashin1120.aiplayground.data.backend.ChatBackend
import com.minashin1120.aiplayground.data.Diagnostics
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import org.json.JSONObject

/**
 * 前面／背面の切替と、一時チャットの期限更新（ハートビート）だけを持つ協作者
 * （Web の `visibilitychange` 相当）。復帰時の再結合・接続監視・低速回線の再判定は
 * 生成時に受け取った関数へ委ね、画面の状態は `mutable` へ書く。
 */
internal class ChatForeground(
    private val application: Application,
    private val state: StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val scope: CoroutineScope,
    private val generations: ChatGenerations,
    private val lowBandwidth: ChatLowBandwidth,
    private val syncRunner: ChatSyncRunner,
    private val backend: () -> ChatBackend,
    private val token: () -> String,
    private val answeringOnDevice: () -> Boolean,
    private val streamJob: () -> Job?,
    private val refreshInPlace: () -> Unit,
    private val discardIdleConnections: () -> Unit,
    private val stopRealtime: () -> Unit,
    private val stopLyria: () -> Unit,
    private val cancelBatchPoll: () -> Unit,
    private val startConnectionMonitor: () -> Unit,
    private val stopConnectionMonitor: () -> Unit,
    private val startBatchPolling: () -> Unit,
    private val startCacheSyncIfAllowed: () -> Unit,
    private val report: suspend (Throwable) -> Unit,
) {
    private var foreground = false
    private var backgroundedAt = 0L

    /** 一時チャットの期限更新（前面の間だけ動く）。 */
    private var heartbeatJob: Job? = null

    /** Whether the app is in front (the Web tab is visible). */
    val isForeground: Boolean get() = foreground

    /** The running temporary-chat heartbeat (BAN・ログアウトで止める対象). */
    val heartbeat: Job? get() = heartbeatJob

    fun setForeground(value: Boolean) {
        if (value != foreground) Diagnostics.log("app.foreground", "value" to value, "streaming" to state.value.streaming,
            "syncing" to state.value.syncing)
        val returning = value && !foreground
        foreground = value
        if (returning) discardIdleConnections()
        if (!value) backgroundedAt = SystemClock.elapsedRealtime()
        // Web keeps reading the answer while the tab is hidden, and the foreground service keeps the app running.
        // If Android still cut the connection without closing it (the service was refused or stopped, or no
        // event arrived lately), an answer the server took (it has a job id) is rejoined after a long absence.
        // A send the server has not taken yet keeps retrying.
        val now = SystemClock.elapsedRealtime()
        if (returning && state.value.streaming && !answeringOnDevice() && state.value.jobId != null &&
            shouldRejoinStream(awayMs = now - backgroundedAt, keptRunning = GenerationService.keptRunningSince(backgroundedAt),
                quietMs = now - (generations.current(state.value)?.lastEventAt ?: now), afterMs = ChatViewModel.STREAM_REJOIN_AFTER_MS)) {
            Diagnostics.log("send.rejoin", "away_ms" to now - backgroundedAt, "quiet_ms" to now - (generations.current(state.value)?.lastEventAt ?: now),
                "kept_running" to GenerationService.keptRunningSince(backgroundedAt))
            val id = state.value.selected?.id
            val job = streamJob()
            scope.launch {
                job?.cancelAndJoin()
                if (state.value.selected?.id == id && !state.value.streaming) refreshInPlace()
            }
        }
        if (!value && state.value.realtime.active) stopRealtime()
        if (!value && state.value.lyria.active) stopLyria()
        if (!value) heartbeatJob?.cancel()
        if (!value) cancelBatchPoll()
        if (value && !state.value.localProfile) startConnectionMonitor() else stopConnectionMonitor()
        if (returning) lowBandwidth.recompute(false)
        if (value && state.value.account != null && !state.value.localProfile) startBatchPolling()
        if (returning && state.value.account != null && state.value.selected != null && !state.value.busy && !state.value.streaming) refreshInPlace()
        if (returning && state.value.account != null) startCacheSyncIfAllowed()
        if (returning && state.value.serverless) syncRunner.schedule()
    }

    /** An answer ended while the app was away: a notification tells the user (in the app, the vibration does). */
    fun notifyAnswerFinishedIfAway(threadId: String, failed: Boolean) {
        if (foreground) return
        val title = state.value.threads.firstOrNull { it.id == threadId }?.title
            ?: state.value.selected?.takeIf { it.id == threadId }?.title.orEmpty()
        notifyAnswerFinished(application, title, failed)
    }

    fun dismissNotice() { mutable.update { it.copy(notice = null) } }

    /** Login screen "端末を連携" cancel: drops the device-code prompt. */
    fun cancelPairing() { mutable.update { it.copy(pairing = false, userCode = "") } }

    /** Web `temporary_chat_heartbeat`: keeps a temporary chat alive while the app is in front. */
    fun syncHeartbeat() {
        heartbeatJob?.cancel()
        val selected = state.value.selected ?: return
        if (!foreground || !selected.isTemporary) return
        heartbeatJob = scope.launch {
            while (isActive && foreground && state.value.selected?.id == selected.id && state.value.selected?.isTemporary == true) {
                try {
                    val reply = backend().post("/api/temporary_chat/heartbeat",
                        JSONObject().put("thread_id", selected.id).put("active", true), token())
                    mutable.update { it.copy(tempChatRemainingSeconds =
                        reply.optLong("temp_chat_remaining_seconds").takeIf { value -> !reply.isNull("temp_chat_remaining_seconds") && value >= 0 }) }
                } catch (e: CancellationException) { throw e }
                catch (e: Exception) { report(e) }
                delay(15_000)
            }
        }
    }
}
