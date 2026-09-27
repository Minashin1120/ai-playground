package com.minashin1120.aiplayground.data.direct

import java.util.Base64
import org.json.JSONArray
import org.json.JSONObject
import java.net.URLEncoder

/**
 * Gemini `streamGenerateContent` (SSE) from the device, as the Web browser fast mode and the server's
 * Gemini branch do: history with inline images/PDF, `thinkingConfig`, Google Search / URL context /
 * Maps / code execution tools, thought signatures, and generated images.
 */
class GeminiDirect(
    private val http: DirectHttp,
    private val baseUrl: String = "https://generativelanguage.googleapis.com",
    /** Vertex AI instead of the Gemini API key (server `gemini_backend = vertex_ai`). */
    private val vertex: VertexTarget? = null,
) : DirectEngine {
    override suspend fun run(request: DirectRequest, emit: (JSONObject) -> Unit, onProgress: (String, String) -> Unit): DirectResult {
        val model = request.model
        val payload = buildPayload(request)
        val content = StringBuilder()
        val thought = StringBuilder()
        val signatures = mutableListOf<String>()
        val files = mutableListOf<DirectOutputFile>()
        var usage: JSONObject? = null
        var searching = false
        var pythonId: String? = null
        var pythonCode = ""
        emit(event("status", "Geminiへ直接送信中..."))
        val url = vertex?.modelUrl(URLEncoder.encode(model, "UTF-8"), "streamGenerateContent?alt=sse")
            ?: "$baseUrl/v1beta/models/${URLEncoder.encode(model, "UTF-8")}:streamGenerateContent?alt=sse"
        val headers = if (vertex != null) mapOf("Authorization" to "Bearer ${vertex.token()}") else mapOf("x-goog-api-key" to request.apiKey)
        http.postSse(url, headers, payload) { sse ->
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

    internal fun buildPayload(request: DirectRequest): JSONObject {
        val options = request.options
        val model = request.model.lowercase()
        val contents = JSONArray()
        request.turns.forEach { turn ->
            val parts = JSONArray()
            turn.attachments.forEach { attachment ->
                val bytes = attachment.bytes
                when {
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
        if (isImageModel(model)) generation.put("responseModalities", JSONArray().put("TEXT").put("IMAGE"))
        if (generation.length() > 0) payload.put("generationConfig", generation)
        val tools = JSONArray()
        if (!isImageModel(model)) {
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

    private fun isImageModel(model: String) = model.contains("image")

    /** Web `browserFastThinkingConfig`: 2.5 uses a budget, newer models a level. */
    private fun thinkingConfig(model: String, options: JSONObject): JSONObject? {
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
}

internal fun attachmentText(attachment: DirectAttachment): String =
    "[添付ファイル: ${attachment.name}]\n${attachment.text.orEmpty()}"

internal fun extensionFor(mime: String): String = when (mime.lowercase()) {
    "image/png" -> "png"; "image/jpeg" -> "jpg"; "image/webp" -> "webp"; "image/gif" -> "gif"
    "audio/wav", "audio/x-wav" -> "wav"; "audio/mpeg" -> "mp3"; "video/mp4" -> "mp4"
    else -> mime.substringAfter('/', "bin").filter { it.isLetterOrDigit() }.take(8).ifBlank { "bin" }
}
