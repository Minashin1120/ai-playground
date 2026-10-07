package com.minashin1120.aiplayground.data.direct

import com.minashin1120.aiplayground.data.Diagnostics
import java.util.Base64
import java.util.IdentityHashMap
import kotlinx.coroutines.delay
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.net.URLEncoder

/**
 * Gemini `streamGenerateContent` (SSE) from the device, as the Web browser fast mode and the server's
 * Gemini branch do: history with inline images/PDF (videos and large files through the Files API),
 * `thinkingConfig`, Google Search / URL context / Maps / code execution tools, thought signatures, and
 * generated images.
 */
class GeminiDirect(
    private val http: DirectHttp,
    private val baseUrl: String = "https://generativelanguage.googleapis.com",
    /** Vertex AI instead of the Gemini API key (server `gemini_backend = vertex_ai`). */
    private val vertex: VertexTarget? = null,
) : DirectEngine {
    override suspend fun run(request: DirectRequest, emit: (JSONObject) -> Unit, onProgress: (String, String) -> Unit): DirectResult {
        if (request.model.equals(NANO_BANANA_21, ignoreCase = true)) {
            return runNanoBanana21(request, emit, onProgress)
        }
        val model = request.model
        val content = StringBuilder()
        val thought = StringBuilder()
        val signatures = mutableListOf<String>()
        val files = mutableListOf<DirectOutputFile>()
        var usage: JSONObject? = null
        var searching = false
        var pythonId: String? = null
        var pythonCode = ""
        val started = System.currentTimeMillis()
        val uploads = if (vertex == null) uploadLargeFiles(request, emit) else emptyMap()
        val payload = buildPayload(request, uploads)
        Diagnostics.log("gemini.payload", "uploaded" to uploads.size, "vertex" to (vertex != null),
            "inline_bytes" to request.turns.sumOf { turn -> turn.attachments.filter { it !in uploads.keys }.sumOf { it.bytes?.size?.toLong() ?: 0L } },
            "ms" to System.currentTimeMillis() - started, "memory" to Diagnostics.memory())
        emit(event("status", "Geminiへ直接送信中..."))
        var chunks = 0
        val url = vertex?.modelUrl(URLEncoder.encode(model, "UTF-8"), "streamGenerateContent?alt=sse")
            ?: "$baseUrl/v1beta/models/${URLEncoder.encode(model, "UTF-8")}:streamGenerateContent?alt=sse"
        val headers = if (vertex != null) mapOf("Authorization" to "Bearer ${vertex.token()}") else mapOf("x-goog-api-key" to request.apiKey)
        val requestStarted = System.currentTimeMillis()
        http.postSse(url, headers, payload) { sse ->
            if (chunks++ == 0) Diagnostics.log("gemini.first_chunk", "ms" to System.currentTimeMillis() - requestStarted)
            if (sse.data.isBlank() || sse.data == "[DONE]") return@postSse true
            val chunk = runCatching { JSONObject(sse.data) }.getOrNull() ?: return@postSse true
            chunk.optJSONObject("error")?.let { throw DirectApiException(it.optInt("code"), it.optString("message").ifBlank { "Gemini API error" }) }
            chunk.optJSONObject("usageMetadata")?.let { usage = it }
            val candidates = chunk.optJSONArray("candidates") ?: JSONArray()
            for (c in 0 until candidates.length()) {
                val candidate = candidates.optJSONObject(c) ?: continue
                if (candidate.has("groundingMetadata") && !searching) {
                    searching = true
                    emit(event("search_status", "searching"))
                }
                val parts = candidate.optJSONObject("content")?.optJSONArray("parts") ?: continue
                for (p in 0 until parts.length()) {
                    val part = parts.optJSONObject(p) ?: continue
                    part.optString("thoughtSignature").takeIf { it.isNotBlank() && it !in signatures }?.let { signatures += it }
                    part.optJSONObject("executableCode")?.let { code ->
                        pythonId = "py_${System.nanoTime()}"
                        pythonCode = code.optString("code")
                        content.append("\n```python\n").append(pythonCode).append("\n```\n")
                        emit(event("python", JSONObject().put("id", pythonId).put("code", pythonCode)))
                    }
                    part.optJSONObject("codeExecutionResult")?.let { result ->
                        val output = result.optString("output")
                        content.append("\n**Output:**\n```\n").append(output).append("\n```\n")
                        emit(event("python", JSONObject().put("id", pythonId ?: "py_${System.nanoTime()}").put("code", pythonCode).put("output", output)))
                    }
                    part.optJSONObject("inlineData")?.let { data ->
                        val mime = data.optString("mimeType").ifBlank { "image/png" }
                        val bytes = runCatching { Base64.getDecoder().decode(data.optString("data")) }.getOrNull()
                        if (bytes != null && bytes.isNotEmpty() && part.optBoolean("thought") != true) {
                            files += DirectOutputFile("gemini_${files.size + 1}.${extensionFor(mime)}", mime, bytes)
                        }
                    }
                    val text = part.optString("text")
                    if (text.isNotEmpty()) {
                        if (part.optBoolean("thought")) { thought.append(text); emit(event("thought", text)) }
                        else { content.append(text); emit(event("content", text)) }
                    }
                }
            }
            onProgress(content.toString(), thought.toString())
            true
        }
        Diagnostics.log("gemini.stream_end", "chunks" to chunks, "content_chars" to content.length,
            "ms" to System.currentTimeMillis() - requestStarted)
        if (searching) emit(event("search_status", "done"))
        val meta = usage
        return DirectResult(
            content = content.toString(), thought = thought.toString(),
            tokensIn = meta?.optInt("promptTokenCount")?.takeIf { it > 0 },
            tokensOut = meta?.let { it.optInt("candidatesTokenCount") + it.optInt("thoughtsTokenCount") }?.takeIf { it > 0 },
            tokensThought = meta?.optInt("thoughtsTokenCount")?.takeIf { it > 0 },
            files = files, thoughtSignatures = signatures,
        )
    }

    /** Gemini's latest Nano Banana model uses unary generateContent with its own image/thinking controls. */
    private suspend fun runNanoBanana21(
        request: DirectRequest,
        emit: (JSONObject) -> Unit,
        onProgress: (String, String) -> Unit,
    ): DirectResult {
        val model = request.model
        val started = System.currentTimeMillis()
        if (vertex != null) {
            val attachments = request.turns.flatMap { it.attachments }
            if (attachments.any { it.mime.startsWith("video/") }) {
                throw IOException("動画を使ったNano Banana 2.1生成にはGemini APIバックエンドが必要です。")
            }
            if (attachments.any { it.mime == "application/pdf" && (it.bytes?.size ?: 0) > INLINE_LIMIT_BYTES }) {
                throw IOException("大容量PDFを使うにはGemini APIバックエンドが必要です。")
            }
        }
        val uploads = if (vertex == null) uploadLargeFiles(request, emit) else emptyMap()
        val payload = buildPayload(request, uploads)
        val url = vertex?.modelUrl(URLEncoder.encode(model, "UTF-8"), "generateContent")
            ?: "$baseUrl/v1beta/models/${URLEncoder.encode(model, "UTF-8")}:generateContent"
        val headers = if (vertex != null) mapOf("Authorization" to "Bearer ${vertex.token()}") else mapOf("x-goog-api-key" to request.apiKey)
        emit(event("status", "Geminiへ直接送信中..."))
        if (request.options.flag("enable_search")) emit(event("search_status", "searching"))
        val response = http.postJson(url, headers, payload)
        response.optJSONObject("error")?.let {
            throw DirectApiException(it.optInt("code"), it.optString("message").ifBlank { "Gemini API error" })
        }
        val content = StringBuilder()
        val thought = StringBuilder()
        val files = mutableListOf<DirectOutputFile>()
        var usage: JSONObject? = null

        fun collect(result: JSONObject, intoContent: StringBuilder, intoFiles: MutableList<DirectOutputFile>) {
            result.optJSONObject("usageMetadata")?.let { usage = it }
            val candidates = result.optJSONArray("candidates") ?: JSONArray()
            for (c in 0 until candidates.length()) {
                val candidate = candidates.optJSONObject(c) ?: continue
                val parts = candidate.optJSONObject("content")?.optJSONArray("parts") ?: continue
                for (p in 0 until parts.length()) {
                    val part = parts.optJSONObject(p) ?: continue
                    if (part.optBoolean("thought")) continue
                    part.optJSONObject("inlineData")?.let { data ->
                        val mime = data.optString("mimeType").ifBlank { "image/png" }
                        val bytes = runCatching { Base64.getDecoder().decode(data.optString("data")) }.getOrNull()
                        if (bytes != null && bytes.isNotEmpty()) {
                            intoFiles += DirectOutputFile("gemini_${intoFiles.size + 1}.${extensionFor(mime)}", mime, bytes)
                        }
                    }
                    part.optString("text").takeIf { it.isNotEmpty() }?.let { intoContent.append(it) }
                }
            }
        }

        collect(response, content, files)
        if (files.isEmpty()) {
            val retry = JSONObject(payload.toString())
            val turns = retry.optJSONArray("contents")
            val lastTurn = turns?.optJSONObject(turns.length() - 1)
            val parts = lastTurn?.optJSONArray("parts")
            parts?.put(JSONObject().put("text", "Return an image for this request. Do not answer with text only."))
            val generation = retry.optJSONObject("generationConfig") ?: JSONObject().also { retry.put("generationConfig", it) }
            generation.put("responseModalities", JSONArray().put("IMAGE"))
            retry.remove("tools")
            val retryResponse = http.postJson(url, headers, retry)
            val retryContent = StringBuilder()
            val retryFiles = mutableListOf<DirectOutputFile>()
            collect(retryResponse, retryContent, retryFiles)
            if (retryFiles.isNotEmpty()) {
                content.clear().append(retryContent)
                files += retryFiles
            }
        }
        if (request.options.flag("enable_search")) emit(event("search_status", "done"))
        onProgress(content.toString(), thought.toString())
        content.toString().takeIf { it.isNotEmpty() }?.let { emit(event("content", it)) }
        Diagnostics.log("gemini.image_end", "files" to files.size, "content_chars" to content.length,
            "ms" to System.currentTimeMillis() - started)
        val meta = usage
        return DirectResult(
            content = content.toString(), thought = thought.toString(),
            tokensIn = meta?.optInt("promptTokenCount")?.takeIf { it > 0 },
            tokensOut = meta?.let { it.optInt("candidatesTokenCount") + it.optInt("thoughtsTokenCount") }?.takeIf { it > 0 },
            tokensThought = meta?.optInt("thoughtsTokenCount")?.takeIf { it > 0 },
            files = files,
        )
    }

    /**
     * Server Gemini branch: videos, and other files over [INLINE_LIMIT_BYTES], go through the Files API
     * (upload, then wait until processed) instead of inline base64, which a large video makes too big to
     * send. Vertex AI has no Files API, so its requests stay inline. Returns the `fileData` of each upload.
     */
    private suspend fun uploadLargeFiles(request: DirectRequest, emit: (JSONObject) -> Unit): Map<DirectAttachment, JSONObject> {
        val uploads = IdentityHashMap<DirectAttachment, JSONObject>()
        request.turns.flatMap { it.attachments }.forEach { attachment ->
            val bytes = attachment.bytes ?: return@forEach
            val video = attachment.mime.lowercase().startsWith("video/")
            if (!video && bytes.size <= INLINE_LIMIT_BYTES) return@forEach
            val label = if (video) "動画" else "ファイル"
            emit(event("status", "${label}をGeminiへアップロード中..."))
            val started = System.currentTimeMillis()
            Diagnostics.log("gemini.upload_start", "mime" to attachment.mime, "bytes" to bytes.size)
            val file = try {
                uploadFile(attachment, bytes, request.apiKey, label, emit)
            } catch (e: kotlinx.coroutines.CancellationException) {
                Diagnostics.log("gemini.upload_cancelled", "ms" to System.currentTimeMillis() - started)
                throw e
            } catch (e: Exception) {
                Diagnostics.failure("gemini.upload_error", e, "ms" to System.currentTimeMillis() - started)
                throw DirectApiException((e as? DirectApiException)?.status ?: 0,
                    "${label}(${attachment.name})のアップロードに失敗しました: ${e.message ?: e.javaClass.simpleName}")
            }
            Diagnostics.log("gemini.upload_done", "ms" to System.currentTimeMillis() - started)
            uploads[attachment] = file
        }
        return uploads
    }

    /** Files API resumable upload, then `files.get` every 2 seconds for up to 120 seconds until ACTIVE (server `_wait_gemini_file_active`). */
    private suspend fun uploadFile(attachment: DirectAttachment, bytes: ByteArray, apiKey: String, label: String, emit: (JSONObject) -> Unit): JSONObject {
        val key = mapOf("x-goog-api-key" to apiKey)
        val uploadUrl = http.postJsonForHeader("$baseUrl/upload/v1beta/files", key + mapOf(
            "X-Goog-Upload-Protocol" to "resumable", "X-Goog-Upload-Command" to "start",
            "X-Goog-Upload-Header-Content-Length" to bytes.size.toString(), "X-Goog-Upload-Header-Content-Type" to attachment.mime,
        ), JSONObject().put("file", JSONObject().put("display_name", attachment.name)), "X-Goog-Upload-URL")
        Diagnostics.log("gemini.upload_session")
        val (reply, _) = http.postBytes(uploadUrl, key + mapOf("X-Goog-Upload-Offset" to "0", "X-Goog-Upload-Command" to "upload, finalize"),
            bytes.toRequestBody(attachment.mime.toMediaTypeOrNull()), 1024 * 1024)
        var file = runCatching { JSONObject(String(reply, Charsets.UTF_8)).getJSONObject("file") }
            .getOrElse { throw IOException("アップロード結果を読み取れませんでした。") }
        var state = file.optString("state")
        Diagnostics.log("gemini.upload_sent", "state" to state)
        if (state == "PROCESSING") emit(event("status", "Geminiで${label}を処理中..."))
        val deadline = System.currentTimeMillis() + FILE_PROCESSING_TIMEOUT_MS
        while (state == "PROCESSING" && System.currentTimeMillis() < deadline) {
            delay(FILE_POLL_INTERVAL_MS)
            file = http.getJson("$baseUrl/v1beta/${file.optString("name")}", key)
            state = file.optString("state")
            Diagnostics.log("gemini.file_state", "state" to state)
        }
        if (state.isNotEmpty() && state != "ACTIVE") throw IOException(if (state == "PROCESSING") "処理が時間内に終わりませんでした。" else "state:$state")
        val uri = file.optString("uri").ifBlank { throw IOException("ファイルのURIを取得できませんでした。") }
        return JSONObject().put("mimeType", file.optString("mimeType").ifBlank { attachment.mime }).put("fileUri", uri)
    }

    internal fun buildPayload(request: DirectRequest, uploads: Map<DirectAttachment, JSONObject> = emptyMap()): JSONObject {
        val options = request.options
        val model = request.model.lowercase()
        val contents = JSONArray()
        request.turns.forEach { turn ->
            val parts = JSONArray()
            turn.attachments.forEach { attachment ->
                val bytes = attachment.bytes
                val uploaded = uploads[attachment]
                when {
                    uploaded != null -> parts.put(JSONObject().put("fileData", uploaded))
                    bytes != null -> parts.put(JSONObject().put("inlineData", JSONObject()
                        .put("mimeType", attachment.mime).put("data", Base64.getEncoder().encodeToString(bytes))))
                    attachment.text != null -> parts.put(JSONObject().put("text", attachmentText(attachment)))
                }
            }
            if (turn.text.isNotEmpty() || parts.length() == 0) {
                val text = JSONObject().put("text", turn.text)
                if (turn.role == "assistant") turn.thoughtSignatures.firstOrNull()?.let { text.put("thoughtSignature", it) }
                parts.put(text)
            }
            contents.put(JSONObject().put("role", if (turn.role == "assistant") "model" else "user").put("parts", parts))
        }
        val payload = JSONObject().put("contents", contents)
        if (request.system.isNotBlank()) payload.put("systemInstruction", JSONObject().put("parts", JSONArray().put(JSONObject().put("text", request.system))))
        val generation = JSONObject()
        thinkingConfig(model, options)?.let { generation.put("thinkingConfig", it) }
        if (isImageModel(model)) {
            generation.put("responseModalities", JSONArray().put("TEXT").put("IMAGE"))
            val imageConfig = JSONObject()
            val aspect = options.optString("gemini_image_aspect").takeIf { it in GEMINI_IMAGE_ASPECTS && it != "auto" }
            val lite = model.contains("gemini-3.1-flash-lite-image")
            val size = if (lite) "1K" else options.optString("gemini_image_size")
            val supportsSize = model == NANO_BANANA_21 || model == "gemini-3.1-flash-image" || model.contains("gemini-3-pro-image")
            aspect?.let { imageConfig.put("aspectRatio", it) }
            if (supportsSize) size.takeIf { it in setOf("1K", "2K", "4K") }?.let { imageConfig.put("imageSize", it) }
            if (imageConfig.length() > 0) generation.put("imageConfig", imageConfig)
        }
        if (generation.length() > 0) payload.put("generationConfig", generation)
        val tools = JSONArray()
        if (model == NANO_BANANA_21 && options.flag("enable_search")) {
            tools.put(JSONObject().put("google_search", JSONObject()))
        } else if (!isImageModel(model)) {
            if (options.flag("enable_search")) tools.put(JSONObject().put("google_search", JSONObject()))
            if (options.flag("enable_url_context")) tools.put(JSONObject().put("url_context", JSONObject()))
            if (options.flag("enable_maps")) tools.put(JSONObject().put("google_maps", JSONObject()))
            val hasNonImage = request.turns.lastOrNull()?.attachments?.any { !it.mime.startsWith("image/") && it.bytes != null } == true
            if (options.flag("enable_python") && !hasNonImage) tools.put(JSONObject().put("codeExecution", JSONObject()))
        }
        if (tools.length() > 0) payload.put("tools", tools)
        safetySettings(options.optString("safety_setting"))?.let { payload.put("safetySettings", it) }
        return payload
    }

    private fun isImageModel(model: String) = model.contains("image") || model == NANO_BANANA_21

    /** Web `browserFastThinkingConfig`: 2.5 uses a budget, newer models a level. */
    private fun thinkingConfig(model: String, options: JSONObject): JSONObject? {
        if (model == NANO_BANANA_21) {
            val level = options.optString("thinking_level").lowercase().takeIf { it in setOf("minimal", "medium", "high") } ?: "medium"
            return JSONObject().put("includeThoughts", options.flag("enable_thinking")).put("thinkingLevel", level.uppercase())
        }
        if (!options.flag("enable_thinking") || isImageModel(model)) return null
        if (model.contains("2.5")) {
            val budget = options.optString("thinking_budget").toIntOrNull()?.coerceIn(0, 32768) ?: 4096
            return JSONObject().put("includeThoughts", true).put("thinkingBudget", budget)
        }
        var level = options.optString("thinking_level").ifBlank { "high" }.uppercase()
        if (model.contains("3.6") && level !in setOf("MEDIUM", "HIGH")) level = "MEDIUM"
        if (model.contains("3.5") && level !in setOf("MINIMAL", "MEDIUM", "HIGH")) level = "MINIMAL"
        return JSONObject().put("includeThoughts", true).put("thinkingLevel", level)
    }

    private fun safetySettings(value: String): JSONArray? {
        // Composer safety select: "default" keeps the API defaults, "none" disables blocking (server Gemini branch).
        val threshold = when (value) { "none", "block_none" -> "BLOCK_NONE"; "off" -> "OFF"; else -> return null }
        return JSONArray().apply {
            listOf("HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")
                .forEach { put(JSONObject().put("category", it).put("threshold", threshold)) }
        }
    }

    private companion object {
        const val NANO_BANANA_21 = "gemini-nano-banana-2.1"
        val GEMINI_IMAGE_ASPECTS = setOf("1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9")
        /** Server `media_inline_limit`: larger files (and every video) go through the Files API. */
        const val INLINE_LIMIT_BYTES = 20 * 1024 * 1024
        const val FILE_POLL_INTERVAL_MS = 2_000L
        const val FILE_PROCESSING_TIMEOUT_MS = 120_000L
    }
}

internal fun attachmentText(attachment: DirectAttachment): String =
    "[添付ファイル: ${attachment.name}]\n${attachment.text.orEmpty()}"

internal fun extensionFor(mime: String): String = when (mime.lowercase()) {
    "image/png" -> "png"; "image/jpeg" -> "jpg"; "image/webp" -> "webp"; "image/gif" -> "gif"
    "audio/wav", "audio/x-wav" -> "wav"; "audio/mpeg" -> "mp3"; "video/mp4" -> "mp4"
    else -> mime.substringAfter('/', "bin").filter { it.isLetterOrDigit() }.take(8).ifBlank { "bin" }
}
