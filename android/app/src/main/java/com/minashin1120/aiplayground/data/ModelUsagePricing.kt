package com.minashin1120.aiplayground.data

import org.json.JSONArray
import org.json.JSONObject

/** Retains numeric billing facts only; tool IDs and search text stay transient. */
internal object ModelUsagePricing {
    fun count(obj: JSONObject?, vararg keys: String): Int? {
        for (key in keys) {
            val n = obj?.opt(key)
            if (n is Number && n !is Double && n !is Float && n.toLong() in 0..Int.MAX_VALUE.toLong()) return n.toInt()
        }
        return null
    }
    private fun putMax(call: ModelUsageLedger.Call, key: String, value: Int?) {
        if (value != null) call.row.put(key, maxOf(value, count(call.row, key) ?: 0))
    }
    fun usage(call: ModelUsageLedger.Call, usage: JSONObject, provider: String) {
        val details = usage.optJSONObject("input_tokens_details") ?: usage.optJSONObject("prompt_tokens_details")
        val creation = usage.optJSONObject("cache_creation")
        val writes = details?.optJSONObject("cache_write_tokens_details")
        putMax(call, "cache_read_tokens", count(details, "cached_tokens") ?: count(usage, "cache_read_input_tokens", "cachedContentTokenCount", "cached_content_token_count", "prompt_cache_hit_tokens", "cached_prompt_tokens", "total_cached_tokens"))
        putMax(call, "cache_write_tokens", count(details, "cache_write_tokens") ?: count(usage, "cache_creation_input_tokens"))
        for ((ttl, obj, field) in listOf(Triple("5m", creation, "ephemeral_5m_input_tokens"), Triple("1h", creation, "ephemeral_1h_input_tokens"), Triple("30m", writes, "ttl_30m_tokens"), Triple("12h", writes, "ttl_12h_tokens"))) putMax(call, "cache_write_${ttl}_tokens", count(obj, field))
        val server = usage.optJSONObject("server_tool_use")
        putMax(call, "web_search_calls", count(server, "web_search_requests"))
        putMax(call, "code_execution_calls", count(server, "code_execution_requests"))
        val ticks = usage.opt("cost_in_usd_ticks")
        if (provider == "xai" && ticks is Number && ticks !is Double && ticks !is Float && ticks.toLong() in 0..9_000_000_000_000_000L) call.row.put("cost_in_usd_ticks", maxOf(ticks.toLong(), call.row.optLong("cost_in_usd_ticks", 0)))
    }
    fun configure(call: ModelUsageLedger.Call, obj: JSONObject) {
        if (!obj.isNull("cachedContent") || !obj.isNull("cached_content")) call.row.put("explicit_cache", true)
        val tools = obj.optJSONArray("tools") ?: return
        for (i in 0 until tools.length()) {
            val tool = tools.optJSONObject(i) ?: continue
            val type = tool.optString("type")
            if (type.startsWith("web_search_preview")) call.row.put("search_preview", true)
            if (type in setOf("web_search_20260209", "web_fetch_20260209", "web_search_20260515", "web_fetch_20260515")) call.row.put("code_execution_free", true)
            val memory = tool.optJSONObject("container")?.optString("memory_limit")
            if (type == "code_interpreter" && memory in setOf("1g", "4g", "16g", "64g")) call.row.put("container_memory_gb", memory!!.dropLast(1).toInt())
        }
    }
    fun payload(call: ModelUsageLedger.Call, payload: JSONObject, obj: JSONObject, provider: String) {
        configure(call, obj)
        val items = obj.optJSONArray("output") ?: JSONArray()
        val all = (0 until items.length()).mapNotNull { items.optJSONObject(it) } + listOfNotNull(payload.optJSONObject("item"))
        for (item in all) {
            val type = item.optString("type")
            if (type == "image_generation_call") call.row.put("pricing_incomplete", true)
            val field = when (type) { "web_search_call" -> "web_search_calls"; "file_search_call" -> "file_search_calls"; "code_interpreter_call" -> "code_execution_calls"; else -> continue }
            if (type == "web_search_call" && provider == "openai" && item.optJSONObject("action")?.optString("type") !in setOf("search", "image_search")) continue
            val id = item.optString("id")
            if (id.isBlank() || id == "null") continue
            if (item.optString("status") in setOf("failed", "incomplete")) { call.row.put("pricing_incomplete", true); continue }
            if (call.priceSeen.add("$type:$id")) call.row.put(field, (count(call.row, field) ?: 0) + 1)
            val container = item.optString("container_id")
            if (type == "code_interpreter_call" && container.isNotBlank() && container != "null" && call.priceSeen.add("container:$container")) call.row.put("container_sessions", (count(call.row, "container_sessions") ?: 0) + 1)
        }
        val candidates = obj.optJSONArray("candidates") ?: JSONArray()
        for (i in 0 until candidates.length()) {
            val candidate = candidates.optJSONObject(i) ?: continue
            val grounding = candidate.optJSONObject("groundingMetadata") ?: candidate.optJSONObject("grounding_metadata") ?: continue
            val queries = grounding.optJSONArray("webSearchQueries") ?: grounding.optJSONArray("web_search_queries") ?: continue
            for (j in 0 until queries.length()) { val q = queries.optString(j); if (q.isNotBlank() && q != "null") call.querySeen.add(q) }
        }
        if (call.querySeen.isNotEmpty()) call.row.put("grounding_queries", call.querySeen.size)
        if (obj.has("web_search") && call.row.optString("model").startsWith("glm-")) { call.row.put("web_search_calls", maxOf(1, count(call.row, "web_search_calls") ?: 0)); call.row.put("pricing_incomplete", true) }
    }
}
