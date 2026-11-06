package com.minashin1120.aiplayground.data.direct

import kotlin.coroutines.coroutineContext
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
        val deepSeekSearch = if (request.model.lowercase().startsWith("deepseek") && request.options.flag("enable_search")) {
            searchDeepSeek(request, emit)
        } else null
        val payload = buildPayload(request, deepSeekSearch?.context.orEmpty())
        val messages = payload.optJSONArray("messages") ?: JSONArray()
        val skills = request.options.optJSONArray("skills")
        val hasSkills = skills != null && skills.length() > 0
        if (hasSkills && !openAiSearch) {
            payload.put("tools", JSONArray().put(SkillTool.openAiChatDefinition())).put("tool_choice", "auto")
        } else if (hasSkills) {
            // Search-only models keep their tool contract; instructions are embedded instead.
            val first = messages.optJSONObject(0)
            if (first != null && first.optString("role") == "system") {
                val catalog = SkillTool.catalogWithInstructions(skills)
                if (catalog.isNotBlank()) first.put("content", first.optString("content") + "\n\n" + catalog)
            }
        }
        val content = StringBuilder()
        val thought = StringBuilder()
        var usage: JSONObject? = null
        emit(event("status", "APIへ送信完了。モデルが応答を生成中です..."))
        if (openAiSearch) emit(event("search_status", "searching"))
        var round = 0
        while (true) {
            val toolCalls = LinkedHashMap<Int, JSONObject>()
            val roundContent = StringBuilder()
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
                    if (text.isNotEmpty()) { roundContent.append(text); content.append(text); emit(event("content", text)) }
                    val calls = delta.optJSONArray("tool_calls") ?: continue
                    for (index in 0 until calls.length()) {
                        val call = calls.optJSONObject(index) ?: continue
                        val slot = call.optInt("index", index)
                        val state = toolCalls.getOrPut(slot) {
                            JSONObject().put("id", "").put("name", "").put("arguments", StringBuilder())
                        }
                        call.optString("id").takeIf { it.isNotBlank() }?.let { state.put("id", it) }
                        call.optJSONObject("function")?.optString("name")?.takeIf { it.isNotBlank() }?.let { state.put("name", it) }
                        call.optJSONObject("function")?.optString("arguments")?.takeIf { it.isNotEmpty() }?.let {
                            (state.get("arguments") as StringBuilder).append(it)
                        }
                    }
                }
                onProgress(content.toString(), thought.toString())
                true
            }
            if (toolCalls.isEmpty() || round >= 3) break
            round++
            val assistantCalls = JSONArray()
            toolCalls.toSortedMap().forEach { (index, state) ->
                assistantCalls.put(JSONObject().put("id", state.optString("id").ifBlank { "skill_${round}_$index" })
                    .put("type", "function").put("function", JSONObject()
                        .put("name", state.optString("name"))
                        .put("arguments", state.get("arguments").toString())))
            }
            messages.put(JSONObject().put("role", "assistant").put("content", roundContent.toString()).put("tool_calls", assistantCalls))
            toolCalls.toSortedMap().forEach { (index, state) ->
                val name = state.optString("name")
                val arguments = state.get("arguments").toString()
                val result = if (name == SkillTool.NAME) {
                    val skillName = runCatching { JSONObject(arguments.ifBlank { "{}" }).optString("name") }.getOrDefault("")
                    SkillTool.execute(skills, skillName)
                } else "Error: Unsupported tool: $name"
                messages.put(JSONObject().put("role", "tool")
                    .put("tool_call_id", state.optString("id").ifBlank { "skill_${round}_$index" })
                    .put("content", result))
            }
            payload.put("messages", messages)
        }
        if (openAiSearch) emit(event("search_status", "done"))
        deepSeekSearch?.sources?.takeIf { it.isNotEmpty() }?.let { sources ->
            val suffix = "\n\n**Sources:**\n$sources"
            content.append(suffix)
            emit(event("content", suffix))
        }
        val meta = usage
        return DirectResult(content.toString(), thought.toString(),
            tokensIn = meta?.optInt("prompt_tokens")?.takeIf { it > 0 },
            tokensOut = meta?.optInt("completion_tokens")?.takeIf { it > 0 },
            tokensThought = meta?.optJSONObject("completion_tokens_details")?.optInt("reasoning_tokens")?.takeIf { it > 0 })
    }

    private data class SearchOutcome(val context: String, val sources: String)

    /**
     * DeepSeek Search (`POST /v1/tools/search`) for serverless mode. The Search API
     * works with every DeepSeek model, while the Responses API `web_search` tool is
     * ignored by deepseek-flash. Results are added to the system context as data.
     */
    private suspend fun searchDeepSeek(request: DirectRequest, emit: (JSONObject) -> Unit): SearchOutcome? {
        val query = request.turns.lastOrNull { it.role == "user" }?.text?.trim().orEmpty()
        if (query.isEmpty()) return null
        emit(event("search_status", "searching"))
        return try {
            val reply = http.postJson(
                "$baseUrl/v1/tools/search",
                mapOf("Authorization" to "Bearer ${request.apiKey}"),
                JSONObject().put("query", query.take(128)).put("count", 8).put("need_content", false),
            )
            val results = reply.optJSONArray("results") ?: JSONArray()
            if (results.length() == 0) return null
            val context = StringBuilder()
            val sources = StringBuilder()
            for (i in 0 until results.length()) {
                val item = results.optJSONObject(i) ?: continue
                val title = item.optString("title").ifBlank { "(untitled)" }
                val url = item.optString("url")
                val summary = item.optString("summary").ifBlank { item.optString("snippet") }
                context.append("[$title]($url)\n")
                if (summary.isNotBlank()) context.append(summary.take(1200)).append("\n\n")
                if (url.isNotBlank() && !sources.contains("($url)")) sources.append("- [").append(title).append("](").append(url).append(")\n")
            }
            // Search is billed per successful request, not by tokens. Keep it in
            // the same usage ledger so the cost estimate includes the tool fee.
            coroutineContext[UsageContext]?.ledger?.begin(request.model, "deepseek")?.apply {
                row.put("web_search_calls", 1)
                row.put("input_tokens", 0)
                row.put("output_tokens", 0)
                finished = true
            }
            SearchOutcome(context.toString().trim(), sources.toString().trim())
        } catch (e: kotlinx.coroutines.CancellationException) {
            throw e
        } catch (_: Exception) {
            null
        } finally {
            emit(event("search_status", "done"))
        }
    }

    internal fun buildPayload(request: DirectRequest, searchContext: String = ""): JSONObject {
        val messages = JSONArray()
        val system = listOf(request.system, searchContext.takeIf { it.isNotBlank() }?.let { "[Web search results from DeepSeek Search]\n$it" })
            .filter { !it.isNullOrBlank() }.joinToString("\n\n")
        if (system.isNotBlank()) messages.put(JSONObject().put("role", "system").put("content", system))
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
                    val original = image.bytes!!
                    val bytes = if (request.model.startsWith("deepseek")) com.minashin1120.aiplayground.data.compactVisionImage(original) else original
                    val mime = if (bytes !== original) "image/webp" else image.mime
                    parts.put(JSONObject().put("type", "image_url").put("image_url", JSONObject()
                        .put("url", "data:${mime};base64," + Base64.getEncoder().encodeToString(bytes))))
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
