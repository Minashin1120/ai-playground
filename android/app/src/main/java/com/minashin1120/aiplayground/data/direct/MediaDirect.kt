package com.minashin1120.aiplayground.data.direct

import com.minashin1120.aiplayground.data.GEMINI_INTERACTIONS_TTS_MODELS
import com.minashin1120.aiplayground.data.GEMINI_TTS_VOICES
import kotlinx.coroutines.delay
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody
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

/** One Ideogram request: the endpoint, its body (JSON or multipart) and whether it is a Precise Edit. */
internal class IdeogramCall(val url: String, val body: RequestBody, val edit: Boolean)

/**
 * Ideogram images from the device (server `server/ideogram.py`): Ideogram 4.5 generates, or runs Precise Edit
 * when an image is attached (or an earlier image is in the chat); 4.0 / 3.0 / 2a / 2.0 only generate.
 */
class IdeogramImageDirect(private val http: DirectHttp, private val baseUrl: String = "https://api.ideogram.ai") : DirectEngine {
    override suspend fun run(request: DirectRequest, emit: (JSONObject) -> Unit, onProgress: (String, String) -> Unit): DirectResult {
        val sources = sourceImages(request)
        emit(event("status", if (sources.isEmpty()) "Ideogram API を呼び出し中..." else "Ideogram API (Precise Edit) を呼び出し中..."))
        val call = buildCall(request, sources)
        val headers = mapOf("Api-Key" to request.apiKey, "Accept" to "application/json")
        val reply = http.execute(http.request(call.url, headers).post(call.body).build()) { response ->
            val text = response.body.string()
            if (!response.isSuccessful) throw ideogramError(response.code, text)
            runCatching { JSONObject(text) }.getOrElse { throw DirectApiException(0, "Ideogram の応答を読み取れませんでした。") }
        }
        val data = reply.optJSONArray("data") ?: JSONArray()
        val files = mutableListOf<DirectOutputFile>()
        var blocked = 0
        for (index in 0 until data.length()) {
            val item = data.optJSONObject(index) ?: continue
            val url = item.optString("url")
            if (!item.optBoolean("is_image_safe", true) || url.isBlank() || url == "null") {
                blocked++
                continue
            }
            val bytes = http.download(url, emptyMap(), 50L * 1024 * 1024)
            val (extension, mime) = ideogramImageKind(bytes)
            files += DirectOutputFile("gen_ideogram_${System.currentTimeMillis()}_$index.$extension", mime, bytes)
        }
        if (files.isEmpty()) {
            throw DirectApiException(0, if (blocked > 0) "安全性チェックにより画像が返されませんでした。プロンプトや入力画像の内容を変更して再度お試しください。"
            else "No data returned.")
        }
        emit(event("status", "完了"))
        return DirectResult(files.indices.joinToString("\n") { "Generated Image ${it + 1} for: ${request.prompt()}" }, files = files)
    }

    /** The attached images; Ideogram 4.5 without any continues from the newest image in the chat. */
    private fun sourceImages(request: DirectRequest): List<DirectAttachment> {
        val attached = request.inputImages()
        val chosen = when {
            attached.isNotEmpty() -> attached
            request.model.lowercase() == "ideogram-4.5" -> listOfNotNull(request.turns.asReversed().firstNotNullOfOrNull { turn ->
                turn.attachments.lastOrNull { it.bytes != null && it.mime.startsWith("image/") }
            })
            else -> emptyList()
        }
        chosen.forEach { image ->
            if (image.mime !in SOURCE_MIMES) throw DirectApiException(0, "入力画像を Ideogram が受け付ける形式（PNG / JPEG / WEBP）へ変換できませんでした。")
            if (image.bytes!!.size > 25 * 1024 * 1024) throw DirectApiException(0, "Ideogram に送れる入力画像は1枚あたり25MBまでです。")
        }
        return chosen
    }

    internal fun buildCall(request: DirectRequest, sources: List<DirectAttachment>): IdeogramCall {
        val model = request.model.lowercase()
        val (path, family) = MODELS[model] ?: throw DirectApiException(0, "未対応の Ideogram モデルです。")
        if (sources.isNotEmpty() && family != "v45") {
            throw DirectApiException(0, "画像の編集に対応しているのは Ideogram 4.5 だけです。Ideogram 4.5 を選択するか、添付画像を外してください。")
        }
        val options = request.options
        fun pick(key: String, allowed: Collection<String>) = options.optString(key).trim().lowercase().takeIf { it in allowed }
        val count = (options.optString("ideogram_count").trim().toIntOrNull() ?: 1).coerceIn(1, 8)
        val seed = options.optString("ideogram_seed").trim().toLongOrNull()?.coerceIn(0L, 2147483647L)
        val magic = pick("ideogram_magic_prompt", MAGIC_PROMPTS)
        val speed = pick("ideogram_speed", SPEEDS)
        val aspect = pick("ideogram_aspect", SIZE_PRESETS.keys) ?: "auto"
        val tier = if (options.optString("ideogram_resolution").trim().lowercase() == "2k") 1 else 0
        val negative = options.optString("ideogram_negative_prompt").trim().takeIf { it != "null" }.orEmpty().take(2000)
        val prompt = request.prompt()
        val generateUrl = "$baseUrl/v2/image/generate/$path"
        fun size(): String? = SIZE_PRESETS[aspect]?.let { if (tier == 1) it.second else it.first }
        fun legacyAspect(): String = if (aspect in SIZE_PRESETS) aspect.replace(':', 'x') else "auto"

        return when (family) {
            "v45" -> {
                val form = MultipartBody.Builder().setType(MultipartBody.FORM)
                    .addFormDataPart("prompt", prompt).addFormDataPart("num_images", count.toString())
                seed?.let { form.addFormDataPart("seed", it.toString()) }
                val quality = pick("ideogram_quality", QUALITIES)
                if (sources.isNotEmpty()) {
                    // Precise Edit keeps the source's own size, so no size or aspect is sent.
                    quality?.let { form.addFormDataPart("quality", it) }
                    sources.take(5).forEachIndexed { index, image ->
                        form.addFormDataPart(if (index == 0) "image" else "reference_images", image.name, image.bytes!!.toRequestBody(image.mime.toMediaType()))
                    }
                    return IdeogramCall("$baseUrl/v2/image/precise-edit/ideogram-4-5", form.build(), true)
                }
                magic?.let { form.addFormDataPart("magic_prompt", it) }
                if (quality != null && quality != "very_low") form.addFormDataPart("quality", quality)
                size()?.let { form.addFormDataPart("size", it) }
                IdeogramCall(generateUrl, form.build(), false)
            }
            "v4" -> {
                val body = JSONObject().put("prompt", prompt).put("num_images", count)
                seed?.let { body.put("seed", it) }
                magic?.let { body.put("magic_prompt", it) }
                speed?.let { body.put("rendering_speed", it) }
                size()?.let { body.put("resolution", it) }
                IdeogramCall(generateUrl, http.jsonBody(body), false)
            }
            "v3" -> {
                val form = MultipartBody.Builder().setType(MultipartBody.FORM)
                    .addFormDataPart("prompt", prompt).addFormDataPart("aspect_ratio", legacyAspect()).addFormDataPart("num_images", count.toString())
                seed?.let { form.addFormDataPart("seed", it.toString()) }
                magic?.let { form.addFormDataPart("magic_prompt", it) }
                speed?.let { form.addFormDataPart("rendering_speed", it) }
                pick("ideogram_style_type", STYLE_TYPES_V3)?.let { form.addFormDataPart("style_type", it) }
                if (negative.isNotEmpty()) form.addFormDataPart("negative_prompt", negative)
                IdeogramCall(generateUrl, form.build(), false)
            }
            else -> {
                val body = JSONObject().put("prompt", prompt).put("aspect_ratio", legacyAspect()).put("num_images", count)
                seed?.let { body.put("seed", it) }
                magic?.let { body.put("magic_prompt", it) }
                speed?.let { body.put("rendering_speed", it) }
                pick("ideogram_style_type", STYLE_TYPES_V2)?.let { body.put("style_type", it) }
                // Ideogram 2a has no negative_prompt field.
                if (negative.isNotEmpty() && model == "ideogram-2.0") body.put("negative_prompt", negative)
                IdeogramCall(generateUrl, http.jsonBody(body), false)
            }
        }
    }

    private companion object {
        /** App model id -> (API path segment, request family). */
        val MODELS = mapOf(
            "ideogram-4.5" to ("ideogram-4-5" to "v45"),
            "ideogram-4.0" to ("ideogram-4" to "v4"),
            "ideogram-3.0" to ("ideogram-3" to "v3"),
            "ideogram-2a" to ("ideogram-2a" to "v2"),
            "ideogram-2.0" to ("ideogram-2" to "v2"),
        )
        /** Aspect ratio -> (1K tier, 2K tier) exact sizes accepted by Ideogram 4.x. */
        val SIZE_PRESETS = mapOf(
            "1:1" to ("1024x1024" to "2048x2048"), "4:5" to ("896x1120" to "1792x2240"), "5:4" to ("1120x896" to "2240x1792"),
            "3:4" to ("864x1152" to "1728x2304"), "4:3" to ("1152x864" to "2304x1728"), "2:3" to ("832x1248" to "1664x2496"),
            "3:2" to ("1248x832" to "2496x1664"), "9:16" to ("720x1280" to "1440x2560"), "16:9" to ("1280x720" to "2560x1440"),
            "10:16" to ("800x1280" to "1600x2560"), "16:10" to ("1280x800" to "2560x1600"), "1:2" to ("720x1440" to "1440x2880"),
            "2:1" to ("1440x720" to "2880x1440"), "1:3" to ("512x1536" to "1024x3072"), "3:1" to ("1536x512" to "3072x1024"),
        )
        val QUALITIES = setOf("very_low", "low", "medium", "high")
        val SPEEDS = setOf("turbo", "default", "quality")
        val MAGIC_PROMPTS = setOf("auto", "on", "off")
        val STYLE_TYPES_V3 = setOf("auto", "general", "realistic", "design", "fiction", "stylized")
        val STYLE_TYPES_V2 = setOf("auto", "general", "realistic", "design", "render_3d", "anime")
        val SOURCE_MIMES = setOf("image/png", "image/jpeg", "image/webp")
    }
}

/** Ideogram's own error text by status (server `ideogram_error_message`). */
internal fun ideogramError(status: Int, body: String): DirectApiException {
    val json = runCatching { JSONObject(body) }.getOrNull()
    val reason = json?.optString("reject_reason").orEmpty()
    var detail = listOf("error", "detail", "message").map { json?.optString(it).orEmpty() }.firstOrNull { it.isNotBlank() }.orEmpty()
    if (reason.isNotBlank() && reason !in detail) detail = "$detail ($reason)".trim()
    detail = detail.take(400)
    val message = when (status) {
        401 -> "Ideogram のAPIキーが無効です。設定で Ideogram API Key を確認してください。"
        402 -> "Ideogram のクレジットまたは利用枠が不足しています。$detail".trim()
        422 -> "安全性チェックにより Ideogram が生成を拒否しました。プロンプトや入力画像の内容を変更して再度お試しください。"
        429 -> "Ideogram のリクエスト制限に達しました。しばらく待ってから再試行してください。$detail".trim()
        else -> "HTTP $status: ${detail.ifBlank { "Ideogram API request failed" }}"
    }
    return DirectApiException(status, message)
}

/** File extension and MIME type of a generated image from its first bytes (PNG when unknown). */
internal fun ideogramImageKind(bytes: ByteArray): Pair<String, String> = when {
    bytes.size > 3 && bytes[0] == 0xFF.toByte() && bytes[1] == 0xD8.toByte() -> "jpg" to "image/jpeg"
    bytes.size > 12 && String(bytes, 8, 4, Charsets.US_ASCII) == "WEBP" -> "webp" to "image/webp"
    else -> "png" to "image/png"
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
                val payload = JSONObject().put("text", text).put("voice_id", xaiTtsVoiceId(options.optString("tts_voice"), options.optString("tts_voice_custom")))
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
            else -> if (request.model in GEMINI_INTERACTIONS_TTS_MODELS) {
                // Gemini 3.8 TTS: verbatim text, style metadata and custom voice IDs (Interactions API).
                val reply = http.postJson("${baseUrl ?: "https://generativelanguage.googleapis.com"}/v1beta/interactions",
                    mapOf("x-goog-api-key" to request.apiKey), geminiInteractionsTtsPayload(request.model, text, options))
                (geminiInteractionsTtsWav(reply) ?: throw DirectApiException(0, "Gemini TTS Error: No audio data returned.")) to "audio/wav"
            } else {
                val speech = JSONObject().put("voiceConfig", JSONObject().put("prebuiltVoiceConfig",
                    JSONObject().put("voiceName", options.optString("tts_voice").ifBlank { "Kore" })))
                options.optString("tts_language").takeIf { it.isNotBlank() }?.let { speech.put("languageCode", it) }
                val payload = JSONObject().put("contents", JSONArray().put(JSONObject().put("role", "user")
                    .put("parts", JSONArray().put(JSONObject().put("text", text)))))
                    .put("generationConfig", JSONObject().put("responseModalities", JSONArray().put("AUDIO")).put("speechConfig", speech))
                val reply = http.postJson("${baseUrl ?: "https://generativelanguage.googleapis.com"}/v1beta/models/" +
                    URLEncoder.encode(request.model, "UTF-8") + ":generateContent", mapOf("x-goog-api-key" to request.apiKey), payload)
                val parts = reply.optJSONArray("candidates")?.optJSONObject(0)?.optJSONObject("content")?.optJSONArray("parts")
                (geminiTtsWav(parts) ?: throw DirectApiException(0, "Gemini TTS Error: No audio data returned.")) to "audio/wav"
            }
        }
        val ext = if (mime == "audio/wav") "wav" else "mp3"
        emit(event("status", "完了"))
        return DirectResult("", files = listOf(DirectOutputFile("speech_${System.currentTimeMillis()}.$ext", mime, bytes)))
    }
}

/**
 * xAI TTS `voice_id` (server `TTS Branch`): a custom voice ID from the Custom Voices API wins over the
 * preset; IDs are case-insensitive and the docs use lowercase. A leftover Google voice name is ignored.
 */
internal fun xaiTtsVoiceId(voice: String, custom: String): String {
    val c = custom.trim()
    if (c.isNotEmpty() && Regex("[A-Za-z0-9_\\-]{1,128}").matches(c) && !Regex("^[a-z]{2,3}-[A-Za-z]{2,4}-").containsMatchIn(c)) return c
    val v = voice.trim().lowercase()
    return if (v in setOf("eve", "ara", "rex", "sal", "leo")) v else "eve"
}

private val GEMINI_TTS_CUSTOM_VOICE = Regex("^[A-Za-z0-9][A-Za-z0-9_.\\-]{0,255}$")

/** Server `_gemini_tts_voice`: a designed / replicated voice ID, else a prebuilt voice (Kore). */
internal fun geminiTtsVoice(voice: String, custom: String): String = custom.trim().takeIf { GEMINI_TTS_CUSTOM_VOICE.matches(it) }
    ?: voice.trim().takeIf { it in GEMINI_TTS_VOICES } ?: "Kore"

/** Server `_gemini_tts_interactions_rest` request body. */
internal fun geminiInteractionsTtsPayload(model: String, text: String, options: JSONObject): JSONObject {
    val content = JSONObject().put("type", "text").put("text", text)
    options.optString("tts_style").trim().take(500).takeIf { it.isNotEmpty() }?.let { style ->
        content.put("annotations", JSONArray().put(JSONObject().put("type", "speech_metadata").put("style", style)))
    }
    return JSONObject().put("model", model).put("store", false)
        .put("input", JSONArray().put(JSONObject().put("type", "user_input").put("content", JSONArray().put(content))))
        .put("response_format", JSONObject().put("type", "audio"))
        .put("generation_config", JSONObject().put("speech_config", JSONArray().put(JSONObject()
            .put("voice", geminiTtsVoice(options.optString("tts_voice"), options.optString("tts_voice_custom"))))))
}

/** Audio blocks of an Interactions response (WAV by default) as one WAV file. */
internal fun geminiInteractionsTtsWav(reply: JSONObject): ByteArray? {
    val out = ByteArrayOutputStream()
    var mime = ""
    val steps = reply.optJSONArray("steps") ?: return null
    for (i in 0 until steps.length()) {
        val step = steps.optJSONObject(i) ?: continue
        if (step.optString("type", "model_output") != "model_output") continue
        val content = step.optJSONArray("content") ?: continue
        for (j in 0 until content.length()) {
            val block = content.optJSONObject(j) ?: continue
            if (block.optString("type") != "audio" || block.optString("data").isBlank()) continue
            var chunk = Base64.getDecoder().decode(block.optString("data"))
            val isWav = chunk.size > 44 && String(chunk, 0, 4, Charsets.US_ASCII) == "RIFF"
            if (out.size() > 0 && isWav) chunk = chunk.copyOfRange(44, chunk.size)
            out.write(chunk)
            if (mime.isEmpty()) mime = block.optString("mime_type")
        }
    }
    val audio = out.toByteArray()
    if (audio.isEmpty()) return null
    if (audio.size > 12 && String(audio, 0, 4, Charsets.US_ASCII) == "RIFF" && String(audio, 8, 4, Charsets.US_ASCII) == "WAVE") return audio
    val rate = Regex("rate=(\\d+)").find(mime)?.groupValues?.get(1)?.toIntOrNull() ?: 24000
    return pcmToWav(audio, rate)
}

/** Gemini TTS audio parts joined into one WAV (headerless 16-bit PCM, or a WAV the API already wrapped). */
internal fun geminiTtsWav(parts: JSONArray?): ByteArray? {
    val out = ByteArrayOutputStream()
    var mime = ""
    for (i in 0 until (parts?.length() ?: 0)) {
        val data = parts?.optJSONObject(i)?.optJSONObject("inlineData") ?: continue
        val encoded = data.optString("data")
        if (encoded.isBlank()) continue
        out.write(Base64.getDecoder().decode(encoded))
        if (mime.isEmpty()) mime = data.optString("mimeType")
    }
    val audio = out.toByteArray()
    if (audio.isEmpty()) return null
    if (audio.size > 12 && String(audio, 0, 4, Charsets.US_ASCII) == "RIFF" && String(audio, 8, 4, Charsets.US_ASCII) == "WAVE") return audio
    val rate = Regex("rate=(\\d+)").find(mime)?.groupValues?.get(1)?.toIntOrNull() ?: 24000
    return pcmToWav(audio, rate)
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
        // gpt-4o-transcribe-diarize (server `/transcribe`): speaker segments need diarized_json, audio over
        // 30 seconds needs a chunking strategy, and the model does not take a prompt.
        val diarize = model == DIARIZE_MODEL
        val body = MultipartBody.Builder().setType(MultipartBody.FORM).addFormDataPart("model", model)
            .apply {
                if (diarize) addFormDataPart("response_format", "diarized_json").addFormDataPart("chunking_strategy", "auto")
                else if (prompt.isNotBlank()) addFormDataPart("prompt", prompt.take(2000))
            }
            .addFormDataPart("file", name, bytes.toRequestBody(mime.ifBlank { "audio/mp4" }.toMediaType())).build()
        val (reply, _) = http.postBytes("$baseUrl/v1/audio/transcriptions", mapOf("Authorization" to "Bearer $apiKey"), body, 8L * 1024 * 1024)
        val text = String(reply, Charsets.UTF_8)
        val json = runCatching { JSONObject(text) }.getOrNull() ?: return text.trim()
        if (diarize) diarizedTranscript(json)?.let { return it }
        return json.optString("text").trim()
    }

    companion object {
        const val DIARIZE_MODEL = "gpt-4o-transcribe-diarize"

        /** `speaker: text` lines like the server's diarized transcript. */
        internal fun diarizedTranscript(json: JSONObject): String? {
            val segments = json.optJSONArray("segments") ?: return null
            val lines = (0 until segments.length()).mapNotNull { i ->
                val seg = segments.optJSONObject(i) ?: return@mapNotNull null
                val text = seg.optString("text").trim()
                if (text.isEmpty()) null else "${seg.optString("speaker").ifBlank { "Speaker" }}: $text"
            }
            return lines.takeIf { it.isNotEmpty() }?.joinToString("\n")
        }
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
