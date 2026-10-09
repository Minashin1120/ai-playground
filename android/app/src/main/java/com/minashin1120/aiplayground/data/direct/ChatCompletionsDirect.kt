package com.minashin1120.aiplayground.data.direct

import org.json.JSONArray
import org.json.JSONObject
import java.util.Base64

/**
 * OpenAI-compatible Chat Completions streaming for DeepSeek, Kimi (Moonshot), Mistral and the OpenAI
 * search models (server `background.py` 5304-6031). Images go inline where the provider accepts them;
 * for the text-only providers the attachment text is sent and images are described as not viewable.
 */
class ChatCompletionsDirect(
    private val http: DirectHttp,
    private val baseUrl: String,
    private val acceptsImages: Boolean,
    private val openAiSearch: Boolean = false,
) : DirectEngine {
    override suspend fun run(request: DirectRequest, emit: (JSONObject) -> Unit, onProgress: (String, String) -> Unit): DirectResult {
        val payload = buildPayload(request)
        val content = StringBuilder()
        val thought = StringBuilder()
        var usage: JSONObject? = null
        emit(event("status", "APIへ送信完了。モデルが応答を生成中です..."))
        if (openAiSearch) emit(event("search_status", "searching"))
        http.postSse("$baseUrl/chat/completions", mapOf("Authorization" to "Bearer ${request.apiKey}"), payload) { sse ->
            if (sse.data == "[DONE]") return@postSse false
            val data = runCatching { JSONObject(sse.data) }.getOrNull() ?: return@postSse true
            data.optJSONObject("error")?.let { throw DirectApiException(0, it.optString("message").ifBlank { "API error" }) }
            data.optJSONObject("usage")?.let { usage = it }
            val choices = data.optJSONArray("choices") ?: JSONArray()
            for (i in 0 until choices.length()) {
                val delta = choices.optJSONObject(i)?.optJSONObject("delta") ?: continue
                delta.optString("reasoning_content").takeIf { it.isNotEmpty() && it != "null" }?.let { thought.append(it); emit(event("thought", it)) }
                val text = when (val raw = delta.opt("content")) {
                    is String -> raw
                    // Mistral reasoning models stream content as typed chunks.
                    is JSONArray -> (0 until raw.length()).mapNotNull { index ->
                        raw.optJSONObject(index)?.let { chunk ->
                            if (chunk.optString("type") == "thinking") {
                                val thinking = chunk.optJSONArray("thinking")?.let { parts ->
                                    (0 until parts.length()).joinToString("") { parts.optJSONObject(it)?.optString("text").orEmpty() }
                                }.orEmpty()
                                if (thinking.isNotEmpty()) { thought.append(thinking); emit(event("thought", thinking)) }
                                null
                            } else chunk.optString("text")
                        }
                    }.joinToString("")
                    else -> ""
                }
                if (text.isNotEmpty()) { content.append(text); emit(event("content", text)) }
            }
            onProgress(content.toString(), thought.toString())
            true
        }
        if (openAiSearch) emit(event("search_status", "done"))
        val meta = usage
        return DirectResult(content.toString(), thought.toString(),
            tokensIn = meta?.optInt("prompt_tokens")?.takeIf { it > 0 },
            tokensOut = meta?.optInt("completion_tokens")?.takeIf { it > 0 },
            tokensThought = meta?.optJSONObject("completion_tokens_details")?.optInt("reasoning_tokens")?.takeIf { it > 0 })
    }

    internal fun buildPayload(request: DirectRequest): JSONObject {
        val messages = JSONArray()
        if (request.system.isNotBlank()) messages.put(JSONObject().put("role", "system").put("content", request.system))
        request.turns.forEach { turn ->
            val assistant = turn.role == "assistant"
            if (assistant) {
                messages.put(JSONObject().put("role", "assistant").put("content", turn.text))
                return@forEach
            }
            val images = turn.attachments.filter { it.bytes != null && it.mime.startsWith("image/") }
            val textParts = turn.attachments.mapNotNull { attachment ->
                when {
                    attachment.text != null -> attachmentText(attachment)
                    attachment.bytes != null && (!acceptsImages || !attachment.mime.startsWith("image/")) ->
                        "[添付ファイル: ${attachment.name}]（このモデルでは内容を読み取れない形式です）"
                    else -> null
                }
            }
            val text = (textParts + turn.text).filter { it.isNotEmpty() }.joinToString("\n\n")
            if (acceptsImages && images.isNotEmpty()) {
                val parts = JSONArray()
                images.forEach { image ->
                    parts.put(JSONObject().put("type", "image_url").put("image_url", JSONObject()
                        .put("url", "data:${image.mime};base64," + Base64.getEncoder().encodeToString(image.bytes))))
                }
                parts.put(JSONObject().put("type", "text").put("text", text))
                messages.put(JSONObject().put("role", "user").put("content", parts))
            } else messages.put(JSONObject().put("role", "user").put("content", text))
        }
        val payload = JSONObject().put("model", apiModelId(request.model)).put("messages", messages).put("stream", true)
        if (!openAiSearch && !request.model.startsWith("glm-")) payload.put("stream_options", JSONObject().put("include_usage", true))
        if (openAiSearch) payload.put("web_search_options", JSONObject())
        return payload
    }

    companion object {
        /** App-facing DeepSeek release IDs sent as the official API alias (server `_deepseek_api_model_id`). */
        fun apiModelId(model: String): String = when (model.trim().lowercase()) {
            "deepseek-v4.1-flash" -> "deepseek-flash"
            "deepseek-v4-flash-0731" -> "deepseek-v4-flash"
            else -> model
        }
    }
}
