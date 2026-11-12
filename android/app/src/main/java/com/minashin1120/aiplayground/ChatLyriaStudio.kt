package com.minashin1120.aiplayground

import android.media.AudioTrack
import com.minashin1120.aiplayground.data.*
import com.minashin1120.aiplayground.data.backend.ChatBackend
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import org.json.JSONArray
import org.json.JSONObject
import java.net.URLEncoder

/**
 * Web `LyriaRealtimeStudio`: weighted prompts, music settings, transport controls and saving the played
 * audio into the chat. State lives in the view model (`ChatState.lyria`); this class reports through `mutable`.
 */
internal class ChatLyriaStudio(
    private val state: StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val scope: CoroutineScope,
    private val api: () -> PlaygroundApi,
    private val backend: () -> ChatBackend,
    private val token: () -> String,
    private val notify: (String) -> Unit,
    private val openThreadId: (String) -> Unit,
) {
    private var lyriaStreamJob: Job? = null
    private var lyriaTrack: AudioTrack? = null

    /** Web `openLyriaStudio(promptText)`: the sent text becomes the studio's first prompt (while no session runs). */
    fun preparePrompt(text: String) {
        mutable.update { if (it.lyria.active) it else it.copy(lyria = it.lyria.copy(prompt = text)) }
    }

    private fun setLyriaStatus(text: String, kind: String) =
        mutable.update { it.copy(lyria = it.lyria.copy(status = text, kind = kind)) }

    /** Web `collectPrompts`: non-empty rows with their weights. */
    private fun lyriaPromptsJson(prompts: List<LyriaPrompt>): JSONArray = JSONArray().apply {
        prompts.filter { it.text.isNotBlank() }.forEach { put(JSONObject().put("text", it.text.trim().take(4000)).put("weight", it.weight.toDouble())) }
    }

    /** Web `startSession`: starts a Lyria RealTime session with the weighted prompts and music settings. */
    fun start(prompts: List<LyriaPrompt>, config: JSONObject) {
        if (state.value.lyria.busy) return
        val weighted = lyriaPromptsJson(prompts)
        if (weighted.length() == 0) { notify("プロンプトを入力してください"); return }
        mutable.update { it.copy(lyria = it.lyria.copy(busy = true, status = "接続中...", kind = "connecting")) }
        scope.launch {
            try {
                val started = backend().post("/api/gemini/music/start", JSONObject().put("weighted_prompts", weighted).put("config", config), token())
                val sid = started.getString("session_id")
                lyriaTrack?.let { track -> runCatching { track.stop(); track.release() } }
                lyriaTrack = newChatAudioTrack(48000, stereo = true)
                mutable.update { it.copy(lyria = it.lyria.copy(sessionId = sid, startedAt = 0L, status = "接続中...", kind = "connecting")) }
                openLyriaStream(sid)
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                val message = (e as? ApiException)?.payload?.optString("error")?.ifBlank { null } ?: e.message ?: "セッション開始に失敗しました"
                setLyriaStatus("エラー: $message", "error")
                notify("Lyria RealTime: $message")
            } finally { mutable.update { it.copy(lyria = it.lyria.copy(busy = false)) } }
        }
    }

    /** Web `openStream`: server events carry audio, status snapshots, errors and the end; a dropped stream reconnects. */
    private fun openLyriaStream(sid: String) {
        lyriaStreamJob?.cancel()
        lyriaStreamJob = scope.launch(Dispatchers.IO) {
            while (state.value.lyria.sessionId == sid) {
                try {
                    api().streamSse("/api/gemini/music/stream?session_id=${URLEncoder.encode(sid, "UTF-8")}", token()) { event -> handleLyriaEvent(event) }
                    break
                } catch (e: CancellationException) { throw e }
                catch (e: Exception) {
                    if (state.value.lyria.sessionId != sid) break
                    setLyriaStatus("ストリーム切断。再接続します…", "connecting")
                    delay(1200)
                }
            }
        }
    }

    private fun handleLyriaEvent(event: JSONObject) {
        if (event.optBoolean("snapshot")) {
            when (val status = event.optString("status")) {
                "error" -> setLyriaStatus("エラー", "error")
                "closed", "stopped" -> setLyriaStatus("終了", "closed")
                else -> if (status == "paused") setLyriaStatus("一時停止中", "paused") else setLyriaStatus("接続中...", "connecting")
            }
            return
        }
        val encoded = event.nullableString("audio")
        if (encoded.isNotBlank()) {
            val bytes = runCatching { android.util.Base64.decode(encoded, android.util.Base64.DEFAULT) }.getOrNull() ?: ByteArray(0)
            if (bytes.isNotEmpty()) lyriaTrack?.write(bytes, 0, bytes.size, AudioTrack.WRITE_NON_BLOCKING)
            mutable.update { it.copy(lyria = it.lyria.copy(status = "再生中...", kind = "streaming",
                startedAt = if (it.lyria.startedAt == 0L) System.currentTimeMillis() else it.lyria.startedAt)) }
            return
        }
        event.nullableString("error").takeIf { it.isNotBlank() }?.let { error -> setLyriaStatus("エラー: $error", "error"); return }
        if (event.optBoolean("final")) setLyriaStatus("終了", "closed")
    }

    private suspend fun lyriaCommand(type: String, fill: JSONObject.() -> Unit) {
        val sid = state.value.lyria.sessionId
        backend().post("/api/gemini/music/command", JSONObject().put("session_id", sid).put("type", type).apply(fill), token())
    }

    private fun lyriaErrorMessage(e: Exception, fallback: String) =
        (e as? ApiException)?.payload?.optString("error")?.ifBlank { null } ?: e.message ?: fallback

    /** Web `control`: PLAY / PAUSE / STOP / RESET_CONTEXT. */
    fun control(action: String) {
        if (!state.value.lyria.active) return
        mutable.update { it.copy(lyria = it.lyria.copy(busy = true)) }
        scope.launch {
            try {
                lyriaCommand("control") { put("action", action) }
                when (action) {
                    "PLAY" -> setLyriaStatus("再生中...", "streaming")
                    "PAUSE" -> setLyriaStatus("一時停止中", "paused")
                    "STOP" -> setLyriaStatus("停止中", "stopped")
                    "RESET_CONTEXT" -> setLyriaStatus("コンテキストをリセット...", "connecting")
                }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                val message = lyriaErrorMessage(e, "コマンド送信に失敗しました")
                notify("Lyria RealTime: $message")
                setLyriaStatus("エラー: $message", "error")
            } finally { mutable.update { it.copy(lyria = it.lyria.copy(busy = false)) } }
        }
    }

    /** Web `applyPrompts`. */
    fun applyPrompts(prompts: List<LyriaPrompt>) {
        if (!state.value.lyria.active) return
        val weighted = lyriaPromptsJson(prompts)
        if (weighted.length() == 0) { notify("プロンプトを入力してください"); return }
        mutable.update { it.copy(lyria = it.lyria.copy(busy = true)) }
        scope.launch {
            try {
                lyriaCommand("prompts") { put("weighted_prompts", weighted) }
                setLyriaStatus("プロンプトを適用しました", if (state.value.lyria.kind == "paused") "paused" else "streaming")
                notify("プロンプトを適用しました")
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) { notify("Lyria RealTime: " + lyriaErrorMessage(e, "コマンド送信に失敗しました")) }
            finally { mutable.update { it.copy(lyria = it.lyria.copy(busy = false)) } }
        }
    }

    /** Web `applyConfig`: BPM or scale changes reset the context. */
    fun applyConfig(config: JSONObject, resetContext: Boolean) {
        if (!state.value.lyria.active) return
        mutable.update { it.copy(lyria = it.lyria.copy(busy = true)) }
        scope.launch {
            try {
                lyriaCommand("config") { put("config", config).put("reset_context", resetContext) }
                val text = if (resetContext) "設定を適用しました（コンテキストをリセット）" else "設定を適用しました"
                setLyriaStatus(text, if (state.value.lyria.kind == "paused") "paused" else "streaming")
                notify(text)
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) { notify("Lyria RealTime: " + lyriaErrorMessage(e, "コマンド送信に失敗しました")) }
            finally { mutable.update { it.copy(lyria = it.lyria.copy(busy = false)) } }
        }
    }

    /** Web `saveSession`: stores the played audio in the chat as WAV, opens that chat and closes the studio. */
    fun saveToChat(onSaved: () -> Unit) {
        val sid = state.value.lyria.sessionId.takeIf { it.isNotBlank() } ?: return
        mutable.update { it.copy(lyria = it.lyria.copy(busy = true, status = "保存中...", kind = "connecting")) }
        scope.launch {
            try {
                val data = backend().post("/api/gemini/music/save", JSONObject().put("session_id", sid)
                    .put("thread_id", state.value.selected?.id ?: JSONObject.NULL), token())
                setLyriaStatus("保存しました", "closed")
                notify("チャットに保存しました")
                closeLyriaSession(cancel = false)
                val threadId = data.optString("thread_id").ifBlank { state.value.selected?.id.orEmpty() }
                if (threadId.isNotBlank()) openThreadId(threadId)
                onSaved()
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                val message = lyriaErrorMessage(e, "保存に失敗しました")
                setLyriaStatus("エラー: $message", "error")
                notify("Lyria RealTime: $message")
            } finally { mutable.update { it.copy(lyria = it.lyria.copy(busy = false)) } }
        }
    }

    /** Web `closeAndCleanup`: closing the studio cancels the session. */
    fun stop(save: Boolean = false) {
        if (save) return saveToChat {}
        closeLyriaSession(cancel = true)
    }

    private fun closeLyriaSession(cancel: Boolean) {
        val sid = state.value.lyria.sessionId
        lyriaStreamJob?.cancel()
        lyriaStreamJob = null
        if (cancel && sid.isNotBlank()) scope.launch {
            runCatching { backend().post("/api/gemini/music/cancel", JSONObject().put("session_id", sid), token()) }
        }
        lyriaTrack?.let { track -> runCatching { track.stop(); track.release() } }; lyriaTrack = null
        mutable.update { it.copy(lyria = LyriaState()) }
    }

    /**
     * Stops playback and forgets the session without asking the server (BAN screen and sign-out).
     * [except] is the caller's own job during sign-out, exactly as the view model cancelled its jobs before.
     */
    fun release(except: Job? = null) {
        if (lyriaStreamJob !== except) lyriaStreamJob?.cancel()
        lyriaStreamJob = null
        lyriaTrack?.let { track -> runCatching { track.stop(); track.release() } }; lyriaTrack = null
    }
}
