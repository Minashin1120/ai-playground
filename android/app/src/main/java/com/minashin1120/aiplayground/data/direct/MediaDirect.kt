package com.minashin1120.aiplayground.data.direct

import kotlinx.coroutines.delay
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.net.URLEncoder
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Base64

/** The text of the newest user turn (the prompt of image, speech and video models). */
internal fun DirectRequest.prompt(): String = turns.lastOrNull { it.role == "user" }?.text.orEmpty()

internal fun DirectRequest.inputImages(): List<DirectAttachment> =
    turns.lastOrNull { it.role == "user" }?.attachments?.filter { it.bytes != null && it.mime.startsWith("image/") }.orEmpty()

/**
 * GPT Image from the device (server "GPT Image Branch"): the Responses API `image_generation` tool,
 * editing when images are attached. Returns the image as a generated file.
 */
class OpenAiImageDirect(private val http: DirectHttp, private val baseUrl: String = "https://api.openai.com") : DirectEngine {
    override suspend fun run(request: DirectRequest, emit: (JSONObject) -> Unit, onProgress: (String, String) -> Unit): DirectResult {
        emit(event("status", "OpenAI API を呼び出し中... (これには時間がかかる場合があります)"))
        val reply = http.postJson("$baseUrl/v1/responses", mapOf("Authorization" to "Bearer ${request.apiKey}"), buildPayload(request))
        val output = reply.optJSONArray("output") ?: JSONArray()
        val encoded = (0 until output.length()).mapNotNull { output.optJSONObject(it) }
            .firstOrNull { it.optString("type") == "image_generation_call" }?.optString("result")?.takeIf { it.isNotBlank() }
            ?: throw DirectApiException(0, reply.optJSONObject("error")?.optString("message")?.ifBlank { null } ?: "No image data found in the response.")
        val format = request.options.optString("image_format").takeIf { it in setOf("png", "jpeg", "webp") } ?: "png"
        val bytes = Base64.getDecoder().decode(encoded)
        emit(event("status", "完了"))
        return DirectResult("Generated Image for: ${request.prompt()}",
            files = listOf(DirectOutputFile("gen_gpt_${System.currentTimeMillis()}.${if (format == "jpeg") "jpg" else format}", "image/$format", bytes)))
    }

    internal fun buildPayload(request: DirectRequest): JSONObject {
        val options = request.options
        fun pick(key: String, allowed: Set<String>, fallback: String) = options.optString(key).takeIf { it in allowed } ?: fallback
        val qualities = mutableSetOf("auto", "low", "medium", "high")
        if (request.model.lowercase() in setOf("gpt-image-2.5-sunburst", "gpt-image-2.5-flare")) qualities += setOf("xhigh", "max")
        val images = request.inputImages()
        val tool = JSONObject().put("type", "image_generation").put("model", request.model)
            .put("size", pick("image_size", setOf("auto", "1024x1024", "1536x1024", "1024x1536"), "auto"))
            .put("quality", pick("image_quality", qualities, "auto"))
            .put("output_format", pick("image_format", setOf("png", "jpeg", "webp"), "png"))
            .put("action", if (images.isEmpty()) "generate" else "edit")
        options.optString("image_compression").toIntOrNull()?.takeIf { it in 0..100 }?.let { tool.put("output_compression", it) }
        // Like the server, only the Gem / client system prompt goes in front of the image prompt.
        val prompt = listOf(options.optString("system_prompt"), request.prompt()).filter { it.isNotBlank() && it != "null" }.joinToString("\n\n")
        val content = JSONArray().put(JSONObject().put("type", "input_text").put("text", prompt))
        images.forEach { image ->
            content.put(JSONObject().put("type", "input_image")
                .put("image_url", "data:${image.mime};base64," + Base64.getEncoder().encodeToString(image.bytes)))
        }
        return JSONObject().put("model", "gpt-4o-mini").put("store", false)
            .put("input", JSONArray().put(JSONObject().put("role", "user").put("content", content)))
            .put("tools", JSONArray().put(tool))
    }
}

/** Grok Imagine images from the device (server "Grok Imagine image"): generations, or edits with up to 3 images. */
class XaiImageDirect(private val http: DirectHttp, private val baseUrl: String = "https://api.x.ai") : DirectEngine {
    override suspend fun run(request: DirectRequest, emit: (JSONObject) -> Unit, onProgress: (String, String) -> Unit): DirectResult {
        emit(event("status", "xAI API を呼び出し中..."))
        val images = request.inputImages()
        val endpoint = if (images.isEmpty()) "$baseUrl/v1/images/generations" else "$baseUrl/v1/images/edits"
        val reply = http.postJson(endpoint, mapOf("Authorization" to "Bearer ${request.apiKey}"), buildPayload(request))
        val data = reply.optJSONArray("data") ?: JSONArray()
        val encoded = (0 until data.length()).mapNotNull { data.optJSONObject(it)?.optString("b64_json")?.takeIf { v -> v.isNotBlank() } }
            .ifEmpty { listOfNotNull(reply.optString("image").takeIf { it.isNotBlank() }) }
        if (encoded.isEmpty()) throw DirectApiException(0, "No image data found in the response.")
        return DirectResult("Generated Image for: ${request.prompt()}", files = encoded.mapIndexed { index, value ->
            DirectOutputFile("gen_grok_${System.currentTimeMillis()}_$index.png", "image/png", Base64.getDecoder().decode(value))
        })
    }

    internal fun buildPayload(request: DirectRequest): JSONObject {
        val options = request.options
        val model = request.model
        val payload = JSONObject().put("model", model).put("prompt", request.prompt()).put("n", 1).put("response_format", "b64_json")
        payload.put("aspect_ratio", options.optString("grok_image_aspect").ifBlank { "1:1" })
        if (model in setOf("grok-imagine-image-2.0", "grok-imagine-image-quality")) {
            payload.put("resolution", options.optString("grok_image_resolution").lowercase().takeIf { it in setOf("1k", "2k") } ?: "1k")
        }
        val images = request.inputImages().take(3).map { image ->
            JSONObject().put("url", "data:${image.mime};base64," + Base64.getEncoder().encodeToString(image.bytes)).put("type", "image_url")
        }
        when (images.size) { 0 -> Unit; 1 -> payload.put("image", images[0]); else -> payload.put("images", JSONArray(images)) }
        return payload
    }
}

/**
 * Text to speech from the device (server "TTS" branches): OpenAI `audio/speech`, xAI `/v1/tts`, Google
 * Cloud Text-to-Speech (API key) and Gemini TTS models (PCM wrapped into WAV).
 */
class TtsDirect(private val http: DirectHttp, private val provider: String, private val baseUrl: String? = null) : DirectEngine {
    override suspend fun run(request: DirectRequest, emit: (JSONObject) -> Unit, onProgress: (String, String) -> Unit): DirectResult {
        val text = request.prompt()
        if (text.isBlank()) throw DirectApiException(0, "読み上げる文章を入力してください。")
        emit(event("status", "音声を生成中..."))
        val options = request.options
        val speed = options.optString("tts_speed").toDoubleOrNull()
        val (bytes, mime) = when (provider) {
            "openai" -> {
                val payload = JSONObject().put("model", request.model).put("input", text)
                    .put("voice", options.optString("tts_voice").lowercase().ifBlank { "alloy" }).put("response_format", "mp3")
                speed?.coerceIn(0.25, 4.0)?.let { payload.put("speed", it) }
                http.postBytes("${baseUrl ?: "https://api.openai.com"}/v1/audio/speech", mapOf("Authorization" to "Bearer ${request.apiKey}"),
                    http.jsonBody(payload)).first to "audio/mpeg"
            }
            "xai" -> {
                val voice = options.optString("tts_voice").ifBlank { "eve" }.lowercase().replaceFirstChar { it.uppercase() }
                val payload = JSONObject().put("text", text).put("voice_id", if (voice in setOf("Eve", "Ara", "Rex", "Sal", "Leo")) voice else "Eve")
                    .put("language", options.optString("tts_language").ifBlank { "ja" })
                speed?.coerceIn(0.7, 1.5)?.let { payload.put("speed", it) }
                http.postBytes("${baseUrl ?: "https://api.x.ai"}/v1/tts", mapOf("Authorization" to "Bearer ${request.apiKey}"), http.jsonBody(payload)).first to "audio/mpeg"
            }
            "google" -> {
                val voice = JSONObject().put("languageCode", options.optString("tts_language").ifBlank { "ja-JP" })
                options.optString("tts_voice_custom").takeIf { it.isNotBlank() }?.let { voice.put("name", it) }
                val audio = JSONObject().put("audioEncoding", "MP3")
                speed?.coerceIn(0.25, 2.0)?.let { audio.put("speakingRate", it) }
                val reply = http.postJson("${baseUrl ?: "https://texttospeech.googleapis.com"}/v1/text:synthesize",
                    mapOf("X-Goog-Api-Key" to request.apiKey),
                    JSONObject().put("input", JSONObject().put("text", text)).put("voice", voice).put("audioConfig", audio))
                Base64.getDecoder().decode(reply.optString("audioContent")) to "audio/mpeg"
            }
            else -> {
                val speech = JSONObject().put("voiceConfig", JSONObject().put("prebuiltVoiceConfig",
                    JSONObject().put("voiceName", options.optString("tts_voice").ifBlank { "Kore" })))
                options.optString("tts_language").takeIf { it.isNotBlank() }?.let { speech.put("languageCode", it) }
                val payload = JSONObject().put("contents", JSONArray().put(JSONObject().put("role", "user")
                    .put("parts", JSONArray().put(JSONObject().put("text", text)))))
                    .put("generationConfig", JSONObject().put("responseModalities", JSONArray().put("AUDIO")).put("speechConfig", speech))
                val reply = http.postJson("${baseUrl ?: "https://generativelanguage.googleapis.com"}/v1beta/models/" +
                    URLEncoder.encode(request.model, "UTF-8") + ":generateContent", mapOf("x-goog-api-key" to request.apiKey), payload)
                val data = reply.optJSONArray("candidates")?.optJSONObject(0)?.optJSONObject("content")?.optJSONArray("parts")
                    ?.optJSONObject(0)?.optJSONObject("inlineData") ?: throw DirectApiException(0, "Gemini TTS Error: No audio data returned.")
                val pcm = Base64.getDecoder().decode(data.optString("data"))
                val rate = Regex("rate=(\\d+)").find(data.optString("mimeType"))?.groupValues?.get(1)?.toIntOrNull() ?: 24000
                pcmToWav(pcm, rate) to "audio/wav"
            }
        }
        val ext = if (mime == "audio/wav") "wav" else "mp3"
        emit(event("status", "完了"))
        return DirectResult("", files = listOf(DirectOutputFile("speech_${System.currentTimeMillis()}.$ext", mime, bytes)))
    }
}

/** 16-bit mono PCM (Gemini TTS) as a WAV file. */
internal fun pcmToWav(pcm: ByteArray, sampleRate: Int, channels: Int = 1): ByteArray {
    val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
    header.put("RIFF".toByteArray()).putInt(36 + pcm.size).put("WAVE".toByteArray())
        .put("fmt ".toByteArray()).putInt(16).putShort(1).putShort(channels.toShort()).putInt(sampleRate)
        .putInt(sampleRate * channels * 2).putShort((channels * 2).toShort()).putShort(16)
        .put("data".toByteArray()).putInt(pcm.size)
    return ByteArrayOutputStream(44 + pcm.size).apply { write(header.array()); write(pcm) }.toByteArray()
}

/** Speech to text of an attached audio file (OpenAI `audio/transcriptions`), also used for voice input. */
class TranscriptionDirect(private val http: DirectHttp, private val baseUrl: String = "https://api.openai.com") : DirectEngine {
    override suspend fun run(request: DirectRequest, emit: (JSONObject) -> Unit, onProgress: (String, String) -> Unit): DirectResult {
        val audio = request.turns.lastOrNull { it.role == "user" }?.attachments?.firstOrNull { it.bytes != null && it.mime.startsWith("audio/") }
            ?: throw DirectApiException(0, "文字起こしする音声ファイルを添付してください。")
        emit(event("status", "文字起こし中..."))
        val text = transcribe(request.model, request.apiKey, audio.name, audio.mime, audio.bytes!!, request.prompt())
        emit(event("content", text))
        return DirectResult(text)
    }

    suspend fun transcribe(model: String, apiKey: String, name: String, mime: String, bytes: ByteArray, prompt: String = ""): String {
        val body = MultipartBody.Builder().setType(MultipartBody.FORM).addFormDataPart("model", model)
            .addFormDataPart("file", name, bytes.toRequestBody(mime.ifBlank { "audio/mp4" }.toMediaType()))
            .apply { if (prompt.isNotBlank()) addFormDataPart("prompt", prompt.take(2000)) }.build()
        val (reply, _) = http.postBytes("$baseUrl/v1/audio/transcriptions", mapOf("Authorization" to "Bearer $apiKey"), body, 8L * 1024 * 1024)
        val text = String(reply, Charsets.UTF_8)
        return runCatching { JSONObject(text).optString("text") }.getOrDefault(text).trim()
    }
}

/**
 * Video generation from the device: Veo (`predictLongRunning`, polled every 5 seconds for up to 10
 * minutes) and Grok Imagine video (`/v1/videos/generations`, polled every 2 seconds). The finished
 * video is downloaded and saved; the app must stay open while it is generated.
 */
class VideoDirect(private val http: DirectHttp, private val provider: String, private val baseUrl: String? = null) : DirectEngine {
    override suspend fun run(request: DirectRequest, emit: (JSONObject) -> Unit, onProgress: (String, String) -> Unit): DirectResult {
        emit(event("content", "**Generating Video...**\n"))
        onProgress("**Generating Video...**\n", "")
        val bytes = if (provider == "xai") xai(request, emit) else veo(request, emit)
        return DirectResult("Generated Video for: ${request.prompt()}",
            files = listOf(DirectOutputFile("gen_video_${System.currentTimeMillis()}.mp4", "video/mp4", bytes)))
    }

    private suspend fun veo(request: DirectRequest, emit: (JSONObject) -> Unit): ByteArray {
        val base = baseUrl ?: "https://generativelanguage.googleapis.com"
        val options = request.options
        val headers = mapOf("x-goog-api-key" to request.apiKey)
        val instance = JSONObject().put("prompt", request.prompt())
        request.inputImages().firstOrNull()?.let { image ->
            instance.put("image", JSONObject().put("bytesBase64Encoded", Base64.getEncoder().encodeToString(image.bytes)).put("mimeType", image.mime))
        }
        val parameters = JSONObject()
            .put("aspectRatio", options.optString("gemini_video_aspect").takeIf { it in setOf("16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3", "21:9") } ?: "16:9")
            .put("resolution", options.optString("gemini_video_resolution").takeIf { it in setOf("480p", "720p", "1080p", "4K") } ?: "720p")
            .put("durationSeconds", (options.optString("gemini_video_duration").toIntOrNull() ?: 8).coerceIn(1, 8))
        var operation = http.postJson("$base/v1beta/models/${URLEncoder.encode(request.model, "UTF-8")}:predictLongRunning", headers,
            JSONObject().put("instances", JSONArray().put(instance)).put("parameters", parameters))
        val name = operation.optString("name").ifBlank { throw DirectApiException(0, "No operation returned.") }
        emit(event("status", "生成中です。数分かかる場合があります..."))
        var polls = 0
        while (!operation.optBoolean("done")) {
            if (++polls > 120) throw DirectApiException(0, "Video generation timed out.")
            delay(5000)
            operation = http.getJson("$base/v1beta/$name", headers)
        }
        operation.optJSONObject("error")?.let { throw DirectApiException(it.optInt("code"), it.optString("message")) }
        val samples = operation.optJSONObject("response")?.optJSONObject("generateVideoResponse")?.optJSONArray("generatedSamples")
        val uri = samples?.optJSONObject(0)?.optJSONObject("video")?.optString("uri")?.takeIf { it.isNotBlank() }
            ?: throw DirectApiException(0, "No video URI in response.")
        emit(event("status", "動画を保存中..."))
        return http.download(uri, headers, 256L * 1024 * 1024)
    }

    private suspend fun xai(request: DirectRequest, emit: (JSONObject) -> Unit): ByteArray {
        val base = baseUrl ?: "https://api.x.ai"
        val options = request.options
        val headers = mapOf("Authorization" to "Bearer ${request.apiKey}")
        var resolution = options.optString("grok_video_resolution").takeIf { it in setOf("480p", "720p", "1080p") } ?: "720p"
        if (resolution == "1080p" && request.model != "grok-imagine-video-1.5") resolution = "720p"
        val payload = JSONObject().put("model", request.model).put("prompt", request.prompt())
            .put("duration", options.optString("grok_video_duration").toIntOrNull() ?: 5)
            .put("aspect_ratio", options.optString("grok_video_aspect").ifBlank { "16:9" }).put("resolution", resolution)
        request.inputImages().firstOrNull()?.let { image ->
            payload.put("image", JSONObject().put("url", "data:${image.mime};base64," + Base64.getEncoder().encodeToString(image.bytes)))
        }
        val started = http.postJson("$base/v1/videos/generations", headers, payload)
        val id = started.optString("request_id").ifBlank { throw DirectApiException(0, "No request_id returned.") }
        emit(event("status", "生成中です。数分かかる場合があります..."))
        var polls = 0
        while (true) {
            if (++polls > 300) throw DirectApiException(0, "Video generation timed out.")
            delay(2000)
            val status = runCatching { http.getJson("$base/v1/videos/$id", headers) }.getOrNull() ?: continue
            val url = status.optString("url").ifBlank { status.optJSONObject("video")?.optString("url").orEmpty() }
            if (url.isNotBlank()) return http.download(url, emptyMap(), 256L * 1024 * 1024)
            if (status.optString("status") == "failed") throw DirectApiException(0, status.optString("error").ifBlank { "Video generation failed." })
        }
    }
}
