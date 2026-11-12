package com.minashin1120.aiplayground

import android.app.Application
import android.media.AudioDeviceInfo
import android.media.AudioManager
import android.media.MediaRecorder
import android.os.Build
import com.minashin1120.aiplayground.data.PlaygroundApi
import com.minashin1120.aiplayground.data.backend.ChatBackend
import com.minashin1120.aiplayground.data.local.LocalSettingsStore
import com.minashin1120.aiplayground.data.modelMediaSupport
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.io.IOException

/**
 * チャット入力の音声録音と文字起こし（Web の `mic-btn`）。
 * 録音の機器・進捗・文字起こし要求の状態を持ち、画面の更新は生成時に受け取った関数へ委ねる。
 */
internal class ChatMicRecorder(
    private val scope: CoroutineScope,
    private val state: StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val app: Application,
    private val api: () -> PlaygroundApi,
    private val token: () -> String,
    private val localSettings: () -> LocalSettingsStore?,
    private val directHttp: () -> com.minashin1120.aiplayground.data.direct.DirectHttp,
    private val notify: (String) -> Unit,
) {
    private var recorder: MediaRecorder? = null
    private var file: File? = null
    private var levelJob: Job? = null

    fun cancel() {
        levelJob?.cancel()
        recorder?.let { runCatching { it.stop() }; it.release() }
        recorder = null
        file?.delete()
        file = null
    }

    /**
     * 最初のタップで録音、次のタップで停止して `/transcribe`（または端末の文字起こし）へ送り、結果を入力欄へ追記する。
     * 呼び出し元はマイクの権限を取得済みであること。
     */
    fun toggle() {
        when (state.value.micMode) {
            "recording" -> { stop(); return }
            "preparing", "transcribing" -> return
        }
        mutable.update { it.copy(micMode = "preparing", micLevels = emptyList()) }
        val out = File(app.cacheDir, "recording.m4a")
        try {
            // MediaRecorder can only be pinned to a device on API 28+; below that, refuse rather than risk a headset mic.
            if (Build.VERSION.SDK_INT < 28) throw IOException("cannot pin recording to the built-in microphone")
            val audioManager = app.getSystemService(AudioManager::class.java)
            val builtInMic = audioManager
                ?.getDevices(AudioManager.GET_DEVICES_INPUTS)
                ?.firstOrNull { it.type == AudioDeviceInfo.TYPE_BUILTIN_MIC }
                ?: throw IOException("built-in microphone not available")
            // Bluetooth / USB / wired headset mics are never used: pin the input to the built-in mic.
            val unprocessed = audioManager.getProperty(AudioManager.PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED) == "true"
            val mediaRecorder = if (Build.VERSION.SDK_INT >= 31) MediaRecorder(app) else @Suppress("DEPRECATION") MediaRecorder()
            recorder = mediaRecorder
            // UNPROCESSED turns off noise suppression / AGC so that quiet speech is kept; devices without it fall back to VOICE_RECOGNITION (least processing).
            mediaRecorder.setAudioSource(
                if (unprocessed) MediaRecorder.AudioSource.UNPROCESSED
                else MediaRecorder.AudioSource.VOICE_RECOGNITION,
            )
            if (!mediaRecorder.setPreferredDevice(builtInMic)) throw IOException("cannot route to built-in microphone")
            mediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
            mediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
            mediaRecorder.setAudioChannels(2)
            mediaRecorder.setAudioSamplingRate(44_100)
            mediaRecorder.setAudioEncodingBitRate(128_000)
            mediaRecorder.setOutputFile(out.absolutePath)
            mediaRecorder.prepare()
            mediaRecorder.start()
            val routed = mediaRecorder.routedDevice
            if (routed != null && routed.type != AudioDeviceInfo.TYPE_BUILTIN_MIC) {
                throw IOException("recording was routed to a non built-in microphone")
            }
            file = out
        } catch (e: Exception) {
            recorder?.let { runCatching { it.stop() }; it.release() }
            recorder = null
            mutable.update { it.copy(micMode = "", micLevels = emptyList()) }
            notify("Microphone access denied or not available.")
            return
        }
        mutable.update { it.copy(micMode = "recording") }
        levelJob?.cancel()
        levelJob = scope.launch {
            while (true) {
                val amplitude = runCatching { recorder?.maxAmplitude ?: 0 }.getOrDefault(0)
                val level = (amplitude / 32767f).coerceIn(0f, 1f)
                mutable.update { it.copy(micLevels = (it.micLevels + level).takeLast(24)) }
                delay(75)
            }
        }
    }

    private fun stop() {
        levelJob?.cancel()
        val mediaRecorder = recorder ?: return
        recorder = null
        val recorded = file
        val stopped = runCatching { mediaRecorder.stop() }.isSuccess
        mediaRecorder.release()
        if (!stopped || recorded == null || !recorded.exists()) {
            mutable.update { it.copy(micMode = "", micLevels = emptyList()) }
            notify("Audio processing error: 録音できませんでした")
            return
        }
        val modelId = state.value.model
        if (state.value.preferences?.micTranscribeMode == "llm" && !modelMediaSupport(modelId).first) {
            mutable.update { it.copy(micMode = "", micLevels = emptyList()) }
            notify("現在のモデルはLLM音声文字起こし（音声入力）に対応していません")
            recorded.delete()
            return
        }
        mutable.update { it.copy(micMode = "transcribing", micLevels = emptyList()) }
        scope.launch {
            try {
                val data = localSettings()?.let { settings -> transcribeOnDevice(settings, recorded) }
                    ?: api().transcribe(recorded, modelId, token())
                val transcript = data.optString("transcript")
                if (transcript.isNotEmpty()) mutable.update { it.copy(draft = if (it.draft.isEmpty()) transcript else it.draft + " " + transcript) }
                else notify(data.optString("error").ifBlank { "Transcription failed" })
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) { notify("Audio processing error: " + (e.message ?: "")) }
            finally {
                recorded.delete()
                mutable.update { it.copy(micMode = "", micLevels = emptyList()) }
            }
        }
    }

    /** Voice input while chats are local: the device's OpenAI key and speech-to-text model (`/transcribe` shape). */
    private suspend fun transcribeOnDevice(settings: LocalSettingsStore, recorded: File): JSONObject {
        val key = withContext(Dispatchers.IO) { settings.providerKey("openai_key") }
            ?: return JSONObject().put("error", "音声入力にはOpenAI APIキーが必要です（APIキータブで設定）")
        val model = state.value.preferences?.sttModel?.takeIf { it.startsWith("gpt") || it.startsWith("whisper") } ?: "gpt-4o-mini-transcribe"
        val bytes = withContext(Dispatchers.IO) { recorded.readBytes() }
        val text = TranscriptionDirect(directHttp()).transcribe(model, key, recorded.name, "audio/mp4", bytes)
        return JSONObject().put("transcript", text)
    }
}
