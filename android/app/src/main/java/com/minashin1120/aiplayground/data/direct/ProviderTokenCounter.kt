package com.minashin1120.aiplayground.data.direct

import kotlinx.coroutines.withTimeoutOrNull
import org.json.JSONObject
import java.net.URLEncoder
import java.security.MessageDigest

/** Counts only the current composer input; never generates or uploads persistent files. */
class ProviderTokenCounter(private val http: DirectHttp, private val bases: Map<String, String> = emptyMap()) {
    private val cache = LinkedHashMap<String, Pair<Long, Int>>()

    suspend fun count(request: DirectRequest, vertex: VertexTarget? = null): Int? = withTimeoutOrNull(15_000) {
        val model = request.model.lowercase()
        val files = request.turns.flatMap { it.attachments }
        if (request.turns.sumOf { it.text.toByteArray().size.toLong() } + files.sumOf { it.bytes?.size?.toLong() ?: it.text?.toByteArray()?.size?.toLong() ?: 0L } > 16L * 1024 * 1024) return@withTimeoutOrNull null
        if (files.any { it.text in setOf("（この形式の内容は端末では読み取れません）", "（DOCXを読み取れませんでした）", "（XLSXを読み取れませんでした）") }) return@withTimeoutOrNull null
        val provider = when {
            model.startsWith("gemini") -> "gemini"
            model.startsWith("claude") -> "anthropic"
            model.startsWith("grok") -> "xai"
            model.startsWith("kimi") -> "kimi"
            model.startsWith("gpt") || model.startsWith("chatgpt") || Regex("o[134].*").matches(model) -> "openai"
            else -> return@withTimeoutOrNull null
        }
        if (listOf("image", "banana", "imagine", "tts", "transcribe", "search", "agent", "deep-research").any { it in model }) return@withTimeoutOrNull null
        // These two endpoints tokenize text only; never label omitted media as API-counted.
        if (provider in setOf("xai", "kimi") && files.any { it.text == null }) return@withTimeoutOrNull null
        if (provider != "gemini" && files.any { it.bytes != null && !it.mime.startsWith("image/") && it.mime != "application/pdf" }) return@withTimeoutOrNull null
        val key = MessageDigest.getInstance("SHA-256").run {
            fun field(bytes: ByteArray) { update(java.nio.ByteBuffer.allocate(4).putInt(bytes.size).array()); update(bytes) }
            field(request.model.toByteArray()); field(request.apiKey.toByteArray())
            field(request.system.toByteArray()); field(request.options.toString().toByteArray())
            field(vertex?.project.orEmpty().toByteArray())
            request.turns.forEach { turn ->
                field(turn.text.toByteArray())
                turn.attachments.forEach { f ->
                    field(f.name.toByteArray()); field(f.mime.toByteArray())
                    field(f.bytes ?: f.text.orEmpty().toByteArray())
                }
            }
            digest().joinToString("") { "%02x".format(it) }
        }
        val now = System.nanoTime()
        synchronized(cache) { cache[key]?.takeIf { now - it.first < 60_000_000_000L }?.second }?.let { return@withTimeoutOrNull it }
        val base = bases[provider] ?: when (provider) {
            "gemini" -> "https://generativelanguage.googleapis.com"
            "anthropic" -> "https://api.anthropic.com"
            "xai" -> "https://api.x.ai"
            "kimi" -> "https://api.moonshot.ai"
            else -> "https://api.openai.com"
        }
        val url: String
        val headers: Map<String, String>
        val payload: JSONObject
        when (provider) {
            "gemini" -> {
                val encoded = URLEncoder.encode(request.model, "UTF-8")
                url = vertex?.modelUrl(encoded, "countTokens") ?: "$base/v1beta/models/$encoded:countTokens"
                headers = if (vertex != null) mapOf("Authorization" to "Bearer ${vertex.token()}") else mapOf("x-goog-api-key" to request.apiKey)
                payload = GeminiDirect(http).buildPayload(request).apply { remove("generationConfig"); remove("safetySettings") }
            }
            "anthropic" -> {
                url = "$base/v1/messages/count_tokens"
                headers = mapOf("x-api-key" to request.apiKey, "anthropic-version" to "2023-06-01")
                payload = AnthropicDirect(http).buildPayload(request).apply { remove("stream"); remove("max_tokens") }
            }
            "openai" -> {
                url = "$base/v1/responses/input_tokens"
                headers = mapOf("Authorization" to "Bearer ${request.apiKey}")
                payload = OpenAiResponsesDirect(http).buildPayload(request).apply { remove("stream"); remove("store") }
            }
            else -> {
                val text = (request.turns.map { it.text } + files.map(::attachmentText)).filter { it.isNotEmpty() }.joinToString("\n\n")
                url = "$base/v1/" + if (provider == "xai") "tokenize-text" else "tokenizers/estimate-token-count"
                headers = mapOf("Authorization" to "Bearer ${request.apiKey}")
                payload = JSONObject().put("model", request.model)
                if (provider == "xai") payload.put("text", text)
                else payload.put("messages", org.json.JSONArray().put(JSONObject().put("role", "user").put("content", text)))
            }
        }
        val result = http.postJson(url, headers, payload)
        val value: Any? = when (provider) {
            "gemini" -> result.opt("totalTokens")
            "xai" -> result.optJSONArray("token_ids")?.length()
            "kimi" -> result.optJSONObject("data")?.opt("total_tokens")
            else -> result.opt("input_tokens")
        }
        val number = value as? Number ?: return@withTimeoutOrNull null
        val tokens = number.toLong().takeIf { it in 0..Int.MAX_VALUE && number.toDouble() == it.toDouble() }?.toInt() ?: return@withTimeoutOrNull null
        synchronized(cache) {
            if (cache.size >= 128) cache.remove(cache.keys.first())
            cache[key] = now to tokens
        }
        tokens
    }
}
