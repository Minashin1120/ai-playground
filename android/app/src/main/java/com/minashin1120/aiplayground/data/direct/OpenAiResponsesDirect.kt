package com.minashin1120.aiplayground.data.direct

import org.json.JSONArray
import org.json.JSONObject
import java.util.Base64

/**
 * OpenAI Responses API streaming from the device (server `background.py` Responses branch). Also used
 * for xAI Grok through its OpenAI-compatible Responses endpoint ([xai] = true). Python runs in the
 * provider's hosted code interpreter instead of the server sandbox.
 */
class OpenAiResponsesDirect(
    private val http: DirectHttp,
    private val xai: Boolean = false,
    private val baseUrl: String = if (xai) "https://api.x.ai" else "https://api.openai.com",
) : DirectEngine {
    override suspend fun run(request: DirectRequest, emit: (JSONObject) -> Unit, onProgress: (String, String) -> Unit): DirectResult {
        val payload = buildPayload(request)
        val content = StringBuilder()
        val thought = StringBuilder()
        val files = mutableListOf<DirectOutputFile>()
        var usage: JSONObject? = null
        var searching = false
        val codes = LinkedHashMap<String, StringBuilder>()
        emit(event("status", "APIへ送信完了。モデルが応答を生成中です..."))
        http.postSse("$baseUrl/v1/responses", mapOf("Authorization" to "Bearer ${request.apiKey}"), payload) { sse ->
            if (sse.data == "[DONE]") return@postSse false
            val data = runCatching { JSONObject(sse.data) }.getOrNull() ?: return@postSse true
            when (val type = data.optString("type").ifBlank { sse.event.orEmpty() }) {
                "response.output_text.delta" -> data.optString("delta").takeIf { it.isNotEmpty() }?.let {
                    content.append(it); emit(event("content", it))
                }
                "response.reasoning_summary_text.delta", "response.reasoning_text.delta" -> data.optString("delta").takeIf { it.isNotEmpty() }?.let {
                    thought.append(it); emit(event("thought", it))
                }
                "response.reasoning_summary_part.done" -> if (thought.isNotEmpty()) { thought.append("\n\n"); emit(event("thought", "\n\n")) }
                "response.web_search_call.in_progress", "response.web_search_call.searching" ->
                    if (!searching) { searching = true; emit(event("search_status", "searching")) }
                "response.web_search_call.completed" -> if (searching) emit(event("search_status", "done"))
                "response.code_interpreter_call_code.delta" -> {
                    val id = data.optString("item_id").ifBlank { "py" }
                    codes.getOrPut(id) { StringBuilder() }.append(data.optString("delta"))
                    emit(event("python", JSONObject().put("id", id).put("code", codes[id].toString())))
                }
                "response.output_item.done" -> data.optJSONObject("item")?.let { item -> outputItem(item, codes, content, files, emit) }
                "response.completed", "response.incomplete" -> usage = data.optJSONObject("response")?.optJSONObject("usage")
                "response.failed", "error" -> {
                    val error = data.optJSONObject("response")?.optJSONObject("error") ?: data.optJSONObject("error") ?: data
                    throw DirectApiException(0, error.optString("message").ifBlank { type })
                }
            }
            onProgress(content.toString(), thought.toString())
            true
        }
        val meta = usage
        val reasoning = meta?.optJSONObject("output_tokens_details")?.optInt("reasoning_tokens")?.takeIf { it > 0 }
        return DirectResult(content.toString(), thought.toString(),
            tokensIn = meta?.optInt("input_tokens")?.takeIf { it > 0 },
            tokensOut = meta?.optInt("output_tokens")?.takeIf { it > 0 },
            tokensThought = reasoning, files = files)
    }

    private fun outputItem(item: JSONObject, codes: Map<String, StringBuilder>, content: StringBuilder,
                           files: MutableList<DirectOutputFile>, emit: (JSONObject) -> Unit) {
        when (item.optString("type")) {
            "code_interpreter_call" -> {
                val id = item.optString("id").ifBlank { "py" }
                val code = item.optString("code").ifBlank { codes[id]?.toString().orEmpty() }
                val outputs = item.optJSONArray("outputs") ?: JSONArray()
                val output = (0 until outputs.length()).mapNotNull { outputs.optJSONObject(it)?.optString("logs")?.takeIf { log -> log.isNotBlank() } }
                    .joinToString("\n")
                content.append("\n```python\n").append(code).append("\n```\n")
                if (output.isNotBlank()) content.append("\n**Output:**\n```\n").append(output).append("\n```\n")
                emit(event("python", JSONObject().put("id", id).put("code", code).put("output", output.ifBlank { "(no output)" })))
            }
            "image_generation_call" -> item.optString("result").takeIf { it.isNotBlank() }?.let { encoded ->
                runCatching { Base64.getDecoder().decode(encoded) }.getOrNull()?.let { bytes ->
                    files += DirectOutputFile("openai_image_${files.size + 1}.png", "image/png", bytes)
                }
            }
        }
    }

    internal fun buildPayload(request: DirectRequest): JSONObject {
        val options = request.options
        val model = request.model.lowercase()
        val input = JSONArray()
        request.turns.forEach { turn ->
            val parts = JSONArray()
            val assistant = turn.role == "assistant"
            if (!assistant) turn.attachments.forEach { attachment ->
                val bytes = attachment.bytes
                when {
                    bytes != null && attachment.mime.startsWith("image/") -> parts.put(JSONObject().put("type", "input_image")
                        .put("image_url", "data:${attachment.mime};base64," + Base64.getEncoder().encodeToString(bytes)))
                    bytes != null && attachment.mime == "application/pdf" && !xai -> parts.put(JSONObject().put("type", "input_file")
                        .put("filename", attachment.name).put("file_data", "data:application/pdf;base64," + Base64.getEncoder().encodeToString(bytes)))
                    attachment.text != null -> parts.put(JSONObject().put("type", "input_text").put("text", attachmentText(attachment)))
                }
            }
            if (turn.text.isNotEmpty() || parts.length() == 0) {
                parts.put(JSONObject().put("type", if (assistant) "output_text" else "input_text").put("text", turn.text))
            }
            input.put(JSONObject().put("role", if (assistant) "assistant" else "user").put("content", parts))
        }
        val payload = JSONObject().put("model", request.model).put("input", input).put("stream", true).put("store", false)
        if (request.system.isNotBlank()) payload.put("instructions", request.system)
        val tools = JSONArray()
        if (options.flag("enable_search")) {
            tools.put(JSONObject().put("type", "web_search"))
            if (xai) tools.put(JSONObject().put("type", "x_search"))
        }
        if (options.flag("enable_python") && !xai) tools.put(JSONObject().put("type", "code_interpreter").put("container", JSONObject().put("type", "auto")))
        if (tools.length() > 0) payload.put("tools", tools)
        val effortOption = options.optString("reasoning_effort").lowercase().trim()
        val reasoningRequested = options.flag("enable_thinking") || (effortOption.isNotEmpty() && effortOption != "none")
        val reasoningModel = !xai && listOf("o1", "o3", "o4", "gpt-5", "reasoning").any { model.contains(it) }
        if (reasoningModel && reasoningRequested) {
            var effort = effortOption.ifBlank {
                when (options.optString("thinking_level").lowercase()) { "low" -> "low"; "high" -> "high"; else -> "medium" }
            }
            if (effort == "none" && listOf("gpt-5-mini", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.5-mini", "gpt-5.5-nano").any { model.contains(it) }) effort = "minimal"
            payload.put("reasoning", JSONObject().put("effort", effort).put("summary", "auto"))
        } else if (xai && reasoningRequested && effortOption.isNotEmpty() && effortOption != "none") {
            payload.put("reasoning", JSONObject().put("effort", effortOption))
        }
        return payload
    }
}
