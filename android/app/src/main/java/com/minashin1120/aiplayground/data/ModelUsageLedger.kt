package com.minashin1120.aiplayground.data

import java.time.Instant
import org.json.JSONArray
import org.json.JSONObject

/** Per-request numeric usage only. Response IDs are held transiently for deduplication. */
class ModelUsageLedger {
    class Call(model: String, private val provider: String) {
        val row = JSONObject().put("model", model).put("created_at", Instant.now().toString())
        internal val priceSeen = mutableSetOf<String>()
        internal val querySeen = mutableSetOf<String>()
        init { if (provider == "openai" && model.contains("search-preview")) row.put("web_search_calls", 1).put("search_preview", true) }
        var finished = false
        var rejected = false
        var responseId: String? = null
        fun observe(payload: JSONObject) {
            val obj = payload.optJSONObject("response") ?: payload.optJSONObject("message") ?: payload
            ModelUsagePricing.payload(this, payload, obj, provider)
            obj.optString("id").takeIf { it.isNotBlank() && it != "null" }?.let { responseId = it }
            val usage = obj.optJSONObject("usageMetadata") ?: obj.optJSONObject("usage_metadata") ?: obj.optJSONObject("usage") ?: return
            ModelUsagePricing.usage(this, usage, provider)
            var input = usage.count("promptTokenCount", "prompt_token_count", "prompt_tokens", "input_tokens")
            if (provider == "anthropic" && input != null) input += (usage.count("cache_read_input_tokens") ?: 0) + (usage.count("cache_creation_input_tokens") ?: 0)
            var thought = usage.count("thoughtsTokenCount", "thoughts_token_count", "reasoning_tokens")
            (usage.optJSONObject("output_tokens_details") ?: usage.optJSONObject("completion_tokens_details"))?.let { thought = it.count("reasoning_tokens") }
            var output = usage.count("output_tokens", "completion_tokens")
            val candidates = usage.count("candidatesTokenCount", "candidates_token_count")
            if (candidates != null) output = candidates + (thought ?: 0)
            else {
                val total = usage.count("total_tokens")
                if (provider == "xai" && input != null && output != null && total != null && total >= input) output = total - input
            }
            for ((key, value) in listOf("input_tokens" to input, "output_tokens" to output, "thought_tokens" to thought)) {
                if (value != null) row.put(key, maxOf(value, row.count(key) ?: 0))
            }
        }
        fun snapshot(): JSONObject = JSONObject(row.toString()).put("complete", finished &&
            (row.has("cost_in_usd_ticks") || row.has("images") || (row.count("input_tokens") != null && row.count("output_tokens") != null)))
    }

    private val calls = mutableListOf<Call>()
    @Synchronized fun records(): JSONArray? = if (calls.isEmpty()) null else snapshot()
    @Synchronized fun begin(model: String, provider: String): Call = Call(model, provider).also { calls += it }
    @Synchronized fun snapshot(): JSONArray {
        val seen = mutableSetOf<String>()
        val containers = mutableSetOf<String>()
        return JSONArray().also { rows -> for (call in calls) {
            if (call.rejected) continue
            val id = call.responseId
            if (id != null && !seen.add(id)) continue
            val row = call.snapshot()
            val used = call.priceSeen.filter { it.startsWith("container:") }.toSet()
            if (used.isNotEmpty()) { row.put("container_sessions", (used - containers).size); containers.addAll(used) }
            rows.put(row)
        } }
    }
    fun total(key: String): Int? {
        val rows = snapshot()
        val counts = (0 until rows.length()).mapNotNull { rows.getJSONObject(it).count(key) }
        return if (counts.isEmpty()) null else counts.sum()
    }

    companion object {
        private fun JSONObject.count(vararg names: String): Int? {
            for (name in names) {
                val value = opt(name)
                if (value is Int && value >= 0) return value
                if (value is Long && value in 0..Int.MAX_VALUE.toLong()) return value.toInt()
            }
            return null
        }
    }
}
