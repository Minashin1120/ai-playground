package com.minashin1120.aiplayground

import android.app.Application
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.os.Build
import com.minashin1120.aiplayground.data.*
import com.minashin1120.aiplayground.data.backend.ChatBackend
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.net.URLEncoder

/**
 * Web `RealtimeVoiceSession` plus the one-shot `/sts` recorder (`mic-btn` for transcription models):
 * starting and stopping a realtime session, capture and playback of PCM audio, and the `/sts` clip upload.
 * State lives in the view model (`ChatState.realtime`); this class only reads it and reports through `mutable`.
 */
internal class ChatRealtimeSession(
    private val state: StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val scope: CoroutineScope,
    private val application: Application,
    private val api: () -> PlaygroundApi,
    private val backend: () -> ChatBackend,
    private val token: () -> String,
    private val notify: (String) -> Unit,
    private val notifyOffline: (String) -> Boolean,
    private val ensureThread: (() -> Unit) -> Unit,
    private val openThreadId: (String) -> Unit,
    private val loadMessages: suspend (String) -> Unit,
) {
    private var realtimeStreamJob: Job? = null
    private var realtimeCaptureJob: Job? = null
    private var realtimeTrack: AudioTrack? = null

    fun start(
        modelId: String,
        voice: String = "alloy",
        targetLanguage: String = "ja",
        thinkingLevel: String = "minimal",
        transcriptionMode: String = "VERBATIM",
        customVocabulary: String = "",
        includeThoughts: Boolean = false,
        speed: Float? = null,
        rateIn: Int? = null,
        rateOut: Int? = null,
        autoPlay: Boolean = true,
        reasoningEffort: String? = null,
    ) {
        if (notifyOffline("オフライン中はRealtimeを開始できません。")) return
        rtAutoPlay = autoPlay
        rtResponseDone = 0
        rtLastAudioAt = 0L
        rtSpeechActive = false
        rtStreamError = null
        if (state.value.realtime.active || modelId.isBlank()) return
        rtStopping = false
        mutable.update { it.copy(realtime = RealtimeState(model = modelId, status = "接続中...")) }
        scope.launch {
            try {
                val started = backend().post("/api/realtime/start", JSONObject()
                    .put("model", modelId)
                    .put("voice", voice)
                    .put("target_lang", targetLanguage.trim().lowercase().take(16).ifBlank { "ja" })
                    .put("thinking_level", thinkingLevel.trim().lowercase().ifBlank { "minimal" })
                    .put("include_thoughts", includeThoughts)
                    .put("transcription_mode", transcriptionMode.trim().uppercase().ifBlank { "VERBATIM" })
                    .put("custom_vocabulary", JSONArray(customVocabulary.split(',', '、', '\n')
                        .map { it.trim() }.filter { it.isNotBlank() }.take(1000)))
                    .apply {
                        speed?.let { put("speed", it.toDouble()) }
                        rateIn?.let { put("rate_in", it) }
                        rateOut?.let { put("rate_out", it) }
                        reasoningEffort?.let { put("reasoning_effort", it) }
                    }, token())
                val sessionId = started.getString("session_id")
                val rateOut = started.optInt("rate_out", 24000).coerceIn(8000, 48000)
                realtimeTrack = newChatAudioTrack(rateOut, stereo = false)
                mutable.update { it.copy(realtime = RealtimeState(true, modelId, sessionId, "話してください...")) }
                realtimeCaptureJob = scope.launch(Dispatchers.IO) {
                    try { captureRealtimeAudio(sessionId, started.optInt("rate_in", rateOut).coerceIn(8000, 48000)) }
                    catch (e: CancellationException) { throw e }
                    catch (e: Exception) {
                        mutable.update { it.copy(realtime = it.realtime.copy(status = "マイクエラー", error = e.message)) }
                        notify("マイクを利用できません: " + (e.message ?: ""))
                    }
                }
                realtimeStreamJob = scope.launch(Dispatchers.IO) {
                    try {
                        api().streamSse("/api/realtime/stream?session_id=${URLEncoder.encode(sessionId, "UTF-8")}", token()) { event ->
                            handleRealtimeEvent(event, rateOut)
                        }
                    } catch (e: CancellationException) { throw e }
                    catch (e: Exception) {
                        if (state.value.realtime.sessionId == sessionId && !rtStopping) {
                            mutable.update { it.copy(realtime = it.realtime.copy(status = "ストリームエラー")) }
                            notify("リアルタイム接続が切断されました")
                        }
                    }
                }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                val message = (e as? ApiException)?.payload?.optString("error")?.ifBlank { null } ?: e.message ?: "セッション開始に失敗しました"
                mutable.update { it.copy(realtime = RealtimeState(status = "接続エラー")) }
                notify("リアルタイムセッションを開始できませんでした: $message")
            }
        }
    }

    private suspend fun captureRealtimeAudio(sessionId: String, rate: Int) {
        val min = AudioRecord.getMinBufferSize(rate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        if (min <= 0) throw IOException("マイクを初期化できません。")
        val recorder = try {
            AudioRecord(MediaRecorder.AudioSource.MIC, rate, AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT, (min * 2).coerceAtLeast(4096))
        } catch (e: SecurityException) { throw IOException("マイクの権限が必要です。", e) }
        try {
            recorder.startRecording()
            val buffer = ByteArray((rate / 5).coerceAtLeast(4096))
            while (currentCoroutineContext().isActive && state.value.realtime.active && state.value.realtime.sessionId == sessionId) {
                val read = recorder.read(buffer, 0, buffer.size)
                if (read > 0) api().postBytes("/api/realtime/audio?session_id=${URLEncoder.encode(sessionId, "UTF-8")}",
                    buffer.copyOf(read), "audio/pcm", token())
            }
        } finally { runCatching { recorder.stop() }; recorder.release() }
    }

    private var rtAutoPlay = true
    private var rtResponseDone = 0
    private var rtLastAudioAt = 0L
    private var rtSpeechActive = false
    private var rtStreamError: String? = null
    private var rtStopping = false

    private fun setRealtimeStatus(text: String) = mutable.update { it.copy(realtime = it.realtime.copy(status = text)) }

    /** Web `RealtimeVoiceSession._handleEvent`. */
    private fun handleRealtimeEvent(event: JSONObject, rateOut: Int) {
        when (event.optString("type")) {
            "status" -> if (event.optString("status") == "ready" && state.value.realtime.active && !rtStopping) setRealtimeStatus("話してください...")
            "audio" -> {
                rtLastAudioAt = System.currentTimeMillis()
                if (!rtAutoPlay) return
                val encoded = event.nullableString("data")
                val bytes = runCatching { android.util.Base64.decode(encoded, android.util.Base64.DEFAULT) }.getOrNull() ?: ByteArray(0)
                if (bytes.isNotEmpty()) {
                    realtimeTrack?.write(bytes, 0, bytes.size, AudioTrack.WRITE_NON_BLOCKING)
                    mutable.update { it.copy(realtime = it.realtime.copy(audioBytes = it.realtime.audioBytes + bytes.size,
                        status = if (rtStopping) it.realtime.status else "再生中...")) }
                }
            }
            "transcript" -> {
                val role = event.optString("role")
                val delta = event.nullableString("delta")
                mutable.update { current -> current.copy(realtime = current.realtime.copy(
                    userText = if (role == "user" && event.optBoolean("cumulative")) delta else if (role == "user") current.realtime.userText + delta else current.realtime.userText,
                    assistantText = if (role == "assistant") current.realtime.assistantText + delta else current.realtime.assistantText,
                    thoughtText = if (role == "thought") current.realtime.thoughtText + delta else current.realtime.thoughtText,
                )) }
            }
            "speech_started" -> {
                rtSpeechActive = true
                realtimeTrack?.let { track -> runCatching { track.pause(); track.flush(); track.play() } }
                if (!rtStopping) setRealtimeStatus("聞き取り中...")
            }
            "speech_stopped" -> { rtSpeechActive = false; if (!rtStopping) setRealtimeStatus("応答待ち...") }
            "interrupted" -> realtimeTrack?.let { track -> runCatching { track.pause(); track.flush(); track.play() } }
            "response_done", "turn_complete" -> rtResponseDone += 1
            // Recoverable provider error: the session stays open (Web shows a warning toast).
            "notice" -> event.nullableString("message").takeIf { it.isNotBlank() }?.let { notify("リアルタイム音声: $it") }
            "error" -> {
                rtStreamError = event.nullableString("message").ifBlank { "リアルタイムエラー" }
                mutable.update { it.copy(realtime = it.realtime.copy(status = "エラー", error = rtStreamError)) }
            }
            "final" -> if (state.value.realtime.active && !rtStopping) stop(save = true)
        }
    }

    /**
     * Web `RealtimeVoiceSession.stop` / `_cancel`: saving commits the last audio, waits for the reply (or a
     * quiet moment, up to 20s) and stores the conversation in the chat; cancelling discards it.
     */
    fun stop(save: Boolean) {
        if (stsRecorder != null) { stopOneShotSts(save); return }
        val current = state.value.realtime
        if (!current.active && current.sessionId.isBlank()) return
        if (rtStopping) return
        rtStopping = true
        scope.launch {
            val sid = current.sessionId
            realtimeCaptureJob?.cancelAndJoin()
            if (!save) {
                realtimeStreamJob?.cancelAndJoin()
                if (sid.isNotBlank()) runCatching { backend().post("/api/realtime/cancel", JSONObject().put("session_id", sid), token()) }
                finishRealtime("Canceled", 800)
                return@launch
            }
            setRealtimeStatus("応答を待っています...")
            runCatching { backend().post("/api/realtime/commit", JSONObject().put("session_id", sid), token()) }
            val startedAt = System.currentTimeMillis()
            val before = rtResponseDone
            var lastActivity = rtLastAudioAt
            while (System.currentTimeMillis() - startedAt < 20_000) {
                if (rtResponseDone > before) break
                if (rtLastAudioAt > lastActivity) lastActivity = rtLastAudioAt
                val now = System.currentTimeMillis()
                if (!rtSpeechActive && now - startedAt > 2_000 && now - lastActivity > 2_500) break
                delay(250)
            }
            realtimeStreamJob?.cancelAndJoin()
            try {
                val reply = backend().post("/api/realtime/save", JSONObject().put("session_id", sid)
                    .apply { state.value.selected?.id?.let { put("thread_id", it) } }, token())
                val error = rtStreamError
                if (error != null) {
                    finishRealtime("エラー", 0)
                    notify("リアルタイム会話でエラーが発生しました: $error")
                } else {
                    finishRealtime("保存しました", 1200)
                    val threadId = reply.optString("thread_id").ifBlank { state.value.selected?.id.orEmpty() }
                    if (threadId.isNotBlank()) {
                        if (state.value.selected?.id == threadId) runCatching { loadMessages(threadId) } else openThreadId(threadId)
                    }
                }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                finishRealtime("保存エラー", 0)
                notify("音声会話の保存に失敗しました: " + ((e as? ApiException)?.payload?.optString("error")?.ifBlank { null } ?: e.message ?: ""))
            }
        }
    }

    /** Releases the session and shows [status]; after [resetMillis] the dock returns to "Tap to speak". */
    // region: Web one-shot speech-to-speech (/sts) for transcription models

    private var stsRecorder: android.media.MediaRecorder? = null
    private var stsFile: File? = null
    private var stsRequest: Map<String, String> = emptyMap()
    private var stsAutoPlay = true
    private var stsAutoRestart = false

    /**
     * Web `mic-btn` for models that are not realtime sessions (gpt-transcribe, gpt-live-transcribe,
     * gpt-realtime-whisper, grok-voice-transcribe-2.0-file): records until the next tap, then sends the clip to `/sts`.
     */
    fun startOneShot(modelId: String, fields: Map<String, String>, autoPlay: Boolean, autoRestart: Boolean) {
        if (notifyOffline("オフライン中はRealtimeを開始できません。")) return
        if (stsRecorder != null || state.value.realtime.active) return
        stsRequest = fields + ("model" to modelId)
        stsAutoPlay = autoPlay
        stsAutoRestart = autoRestart
        val app = application
        val file = File(app.cacheDir, "sts_recording.m4a")
        try {
            val recorder = if (Build.VERSION.SDK_INT >= 31) android.media.MediaRecorder(app) else android.media.MediaRecorder()
            recorder.setAudioSource(android.media.MediaRecorder.AudioSource.MIC)
            recorder.setOutputFormat(android.media.MediaRecorder.OutputFormat.MPEG_4)
            recorder.setAudioEncoder(android.media.MediaRecorder.AudioEncoder.AAC)
            recorder.setAudioSamplingRate(44_100)
            recorder.setAudioEncodingBitRate(96_000)
            recorder.setOutputFile(file.absolutePath)
            recorder.prepare()
            recorder.start()
            stsRecorder = recorder
            stsFile = file
        } catch (e: Exception) {
            stsRecorder?.release()
            stsRecorder = null
            notify("Microphone access denied or not available.")
            return
        }
        mutable.update { it.copy(realtime = RealtimeState(active = true, model = modelId, status = "Recording... Tap to stop")) }
    }

    private fun stopOneShotSts(save: Boolean) {
        val recorder = stsRecorder ?: return
        stsRecorder = null
        runCatching { recorder.stop() }
        recorder.release()
        val file = stsFile
        val modelId = stsRequest["model"].orEmpty()
        if (!save || file == null || !file.exists() || file.length() == 0L) {
            file?.delete()
            scope.launch { finishRealtime("Canceled", 800) }
            return
        }
        mutable.update { it.copy(realtime = it.realtime.copy(active = false, status = "Sending audio...")) }
        ensureThread {
            scope.launch {
                val threadId = state.value.selected?.id.orEmpty()
                var track: AudioTrack? = null
                var firstAudio = true
                var saved = false
                try {
                    val transcription = modelId == "gpt-transcribe" || modelId == "gpt-live-transcribe" || modelId == "gpt-realtime-whisper" || modelId == "grok-voice-transcribe-2.0-file"
                    mutable.update { it.copy(realtime = it.realtime.copy(status = if (transcription) "Transcribing..." else "Processing audio...")) }
                    val rate = stsRequest["sts_rate_out"]?.toIntOrNull() ?: 24000
                    withContext(Dispatchers.IO) {
                        api().sts(file, stsRequest + ("thread_id" to threadId), token()) { chunk ->
                            chunk.optString("error").takeIf { it.isNotBlank() && !chunk.isNull("error") }?.let { throw IOException(it) }
                            val audio = chunk.optString("audio_delta")
                            if (audio.isNotEmpty() && stsAutoPlay) {
                                if (firstAudio) {
                                    firstAudio = false
                                    mutable.update { it.copy(realtime = it.realtime.copy(status = "Playing response...")) }
                                }
                                val pcm = android.util.Base64.decode(audio, android.util.Base64.DEFAULT)
                                val out = track ?: newChatAudioTrack(rate, stereo = false).also { track = it }
                                out.write(pcm, 0, pcm.size)
                            }
                            chunk.optString("input_delta").takeIf { it.isNotEmpty() }?.let { delta ->
                                mutable.update { it.copy(realtime = it.realtime.copy(userText = it.realtime.userText + delta)) }
                            }
                            chunk.optString("transcript_delta").takeIf { it.isNotEmpty() }?.let { delta ->
                                mutable.update { it.copy(realtime = it.realtime.copy(assistantText = it.realtime.assistantText + delta)) }
                            }
                            if (chunk.optBoolean("final") || chunk.has("audio_url")) saved = true
                        }
                    }
                    if (saved && threadId.isNotBlank()) runCatching { loadMessages(threadId) }
                } catch (e: CancellationException) { throw e }
                catch (e: Exception) {
                    notify("Audio processing error: " + ((e as? ApiException)?.payload?.optString("error")?.ifBlank { null } ?: e.message ?: ""))
                    saved = false
                } finally {
                    file.delete()
                    // stop() lets the written audio finish; release it once that has played.
                    track?.let { played -> runCatching { played.stop() }; scope.launch { delay(1500); runCatching { played.release() } } }
                }
                if (saved && stsAutoRestart && state.value.model == modelId) {
                    mutable.update { it.copy(realtime = RealtimeState(model = modelId, status = "Listening...")) }
                    delay(500)
                    startOneShot(modelId, stsRequest - "model" - "thread_id", stsAutoPlay, stsAutoRestart)
                } else mutable.update { it.copy(realtime = RealtimeState(model = modelId, status = "Tap to speak")) }
            }
        }
    }

    private suspend fun finishRealtime(status: String, resetMillis: Long) {
        realtimeTrack?.let { track -> runCatching { track.stop(); track.release() } }; realtimeTrack = null
        rtStopping = false
        mutable.update { it.copy(realtime = RealtimeState(status = status)) }
        if (resetMillis > 0) {
            delay(resetMillis)
            mutable.update { if (!it.realtime.active && it.realtime.status == status) it.copy(realtime = RealtimeState()) else it }
        }
    }

    /**
     * Drops the session and playback without saving (Web `_cancel`; BAN screen and sign-out).
     * [except] is the caller's own job during sign-out, exactly as the view model cancelled its jobs before.
     */
    fun release(except: Job? = null) {
        if (realtimeCaptureJob !== except) realtimeCaptureJob?.cancel()
        realtimeCaptureJob = null
        if (realtimeStreamJob !== except) realtimeStreamJob?.cancel()
        realtimeStreamJob = null
        realtimeTrack?.let { track -> runCatching { track.stop(); track.release() } }; realtimeTrack = null
        stsRecorder?.let { recorder -> runCatching { recorder.release() } }
        stsRecorder = null
        stsFile?.delete(); stsFile = null
    }
}
