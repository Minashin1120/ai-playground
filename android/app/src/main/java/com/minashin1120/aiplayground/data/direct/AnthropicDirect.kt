package com.minashin1120.aiplayground.data.direct

import org.json.JSONArray
import org.json.JSONObject
import java.util.Base64

/**
 * Claude Messages API streaming from the device (server `background.py` Anthropic branch): images and
 * PDFs inline, extended thinking with a budget, automatic prompt caching and, when search is on, the
 * provider-hosted web search tool.
 */
class AnthropicDirect(private val http: DirectHttp, private val baseUrl: String = "https://api.anthropic.com") : DirectEngine {
    override suspend fun run(request: DirectRequest, emit: (JSONObject) -> Unit, onProgress: (String, String) -> Unit): DirectResult {
        val payload = buildPayload(request)
        val messages = payload.optJSONArray("messages") ?: JSONArray()
        val skills = request.options.optJSONArray("skills")
        val content = StringBuilder()
        val thought = StringBuilder()
        var tokensIn: Int? = null
        var tokensOut: Int? = null
        var searching = false
        emit(event("status", "APIへ送信完了。モデルが応答を生成中です..."))
        val headers = mapOf(
            "x-api-key" to request.apiKey,
            "anthropic-version" to "2023-06-01",
            // Required by Anthropic for calls that do not come from a server.
            "anthropic-dangerous-direct-browser-access" to "true",
        )
        var round = 0
        while (true) {
            val roundContent = StringBuilder()
            val toolUses = mutableListOf<JSONObject>()
            var currentTool: JSONObject? = null
            http.postSse("$baseUrl/v1/messages", headers, payload) { sse ->
                val data = runCatching { JSONObject(sse.data) }.getOrNull() ?: return@postSse true
                when (data.optString("type")) {
                    "message_start" -> data.optJSONObject("message")?.optJSONObject("usage")?.let { usage ->
                        tokensIn = usage.optInt("input_tokens") + usage.optInt("cache_read_input_tokens") + usage.optInt("cache_creation_input_tokens")
                    }
                    "content_block_start" -> {
                        val block = data.optJSONObject("content_block")
                        if (block?.optString("type") == "server_tool_use" && !searching) { searching = true; emit(event("search_status", "searching")) }
                        if (block?.optString("type") == "web_search_tool_result" && searching) emit(event("search_status", "done"))
                        if (block?.optString("type") == "tool_use") {
                            currentTool = JSONObject().put("id", block.optString("id"))
                                .put("name", block.optString("name"))
                                .put("input", StringBuilder())
                            toolUses += currentTool!!
                        }
                    }
                    "content_block_delta" -> {
                        val delta = data.optJSONObject("delta") ?: return@postSse true
                        when (delta.optString("type")) {
                            "text_delta" -> delta.optString("text").takeIf { it.isNotEmpty() }?.let {
                                roundContent.append(it); content.append(it); emit(event("content", it))
                            }
                            "thinking_delta" -> delta.optString("thinking").takeIf { it.isNotEmpty() }?.let { thought.append(it); emit(event("thought", it)) }
                            "input_json_delta" -> delta.optString("partial_json").takeIf { it.isNotEmpty() }?.let {
                                (currentTool?.get("input") as? StringBuilder)?.append(it)
                            }
                        }
                    }
                    "content_block_stop" -> currentTool = null
                    "message_delta" -> data.optJSONObject("usage")?.optInt("output_tokens")?.takeIf { it > 0 }?.let { tokensOut = it }
                    "error" -> throw DirectApiException(0, data.optJSONObject("error")?.optString("message").orEmpty().ifBlank { "Anthropic API error" })
                    "message_stop" -> return@postSse false
                }
                onProgress(content.toString(), thought.toString())
                true
            }
            if (toolUses.isEmpty() || round >= 3) break
            round++
            val assistantBlocks = JSONArray()
            if (roundContent.isNotEmpty()) assistantBlocks.put(JSONObject().put("type", "text").put("text", roundContent.toString()))
            toolUses.forEach { tool ->
                val inputText = (tool.get("input") as? StringBuilder)?.toString().orEmpty()
                val input = runCatching { JSONObject(inputText.ifBlank { "{}" }) }.getOrElse { JSONObject() }
                assistantBlocks.put(JSONObject().put("type", "tool_use")
                    .put("id", tool.optString("id"))
                    .put("name", tool.optString("name"))
                    .put("input", input))
            }
            messages.put(JSONObject().put("role", "assistant").put("content", assistantBlocks))
            val results = JSONArray()
            toolUses.forEach { tool ->
                val name = tool.optString("name")
                val inputText = (tool.get("input") as? StringBuilder)?.toString().orEmpty()
                val input = runCatching { JSONObject(inputText.ifBlank { "{}" }) }.getOrElse { JSONObject() }
                val result = if (name == SkillTool.NAME) SkillTool.execute(skills, input.optString("name"))
                else "Error: Unsupported tool: $name"
                results.put(JSONObject().put("type", "tool_result")
                    .put("tool_use_id", tool.optString("id")).put("content", result))
            }
            messages.put(JSONObject().put("role", "user").put("content", results))
            payload.put("messages", messages)
        }
        return DirectResult(content.toString(), thought.toString(), tokensIn?.takeIf { it > 0 }, tokensOut)
    }

    internal fun buildPayload(request: DirectRequest): JSONObject {
        val options = request.options
        val messages = JSONArray()
        request.turns.forEach { turn ->
            val assistant = turn.role == "assistant"
            val blocks = JSONArray()
            if (!assistant) turn.attachments.forEach { attachment ->
                val bytes = attachment.bytes
                when {
                    bytes != null && attachment.mime.startsWith("image/") -> blocks.put(JSONObject().put("type", "image")
                        .put("source", JSONObject().put("type", "base64").put("media_type", attachment.mime)
                            .put("data", Base64.getEncoder().encodeToString(bytes))))
                    bytes != null && attachment.mime == "application/pdf" -> blocks.put(JSONObject().put("type", "document")
                        .put("source", JSONObject().put("type", "base64").put("media_type", "application/pdf")
                            .put("data", Base64.getEncoder().encodeToString(bytes))))
                    attachment.text != null -> blocks.put(JSONObject().put("type", "text").put("text", attachmentText(attachment)))
                }
            }
            val text = turn.text.ifEmpty { if (blocks.length() == 0) "(empty)" else "" }
            if (text.isNotEmpty()) blocks.put(JSONObject().put("type", "text").put("text", text))
            messages.put(JSONObject().put("role", if (assistant) "assistant" else "user").put("content", blocks))
        }
        val payload = JSONObject().put("model", request.model).put("messages", messages).put("max_tokens", 8192).put("stream", true)
        if (request.system.isNotBlank()) payload.put("system", request.system)
        if (options.flag("enable_prompt_caching")) payload.put("cache_control", JSONObject().put("type", "ephemeral"))
        if (options.flag("enable_thinking")) {
            val budget = (options.optString("thinking_budget").toIntOrNull() ?: 4096).coerceAtLeast(1024)
            payload.put("thinking", JSONObject().put("type", "enabled").put("budget_tokens", budget)).put("max_tokens", budget + 4096)
        }
        val tools = JSONArray()
        val skills = options.optJSONArray("skills")
        if (skills != null && skills.length() > 0) tools.put(SkillTool.anthropicDefinition())
        if (options.flag("enable_search")) {
            tools.put(JSONObject().put("type", "web_search_20250305").put("name", "web_search").put("max_uses", 5))
        }
        if (tools.length() > 0) payload.put("tools", tools)
        return payload
    }
}
