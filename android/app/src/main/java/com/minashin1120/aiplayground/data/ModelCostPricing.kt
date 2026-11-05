package com.minashin1120.aiplayground.data

import org.json.JSONObject

/** Replaces cached input portions and adds tool fees independently of Batch token discounts. */
internal object ModelCostPricing {
    data class Amount(val low: Double, val high: Double, val incomplete: Boolean, val lowerBound: Boolean)
    fun amount(call: JSONObject, rate: JSONObject, long: Boolean, batch: Boolean): Amount {
        fun n(key: String) = ModelUsagePricing.count(call, key) ?: 0
        fun price(key: String, fallback: Double) = if (rate.isNull(key)) fallback else rate.optDouble(key, fallback)
        val input = n("input_tokens"); val output = n("output_tokens"); val read = n("cache_read_tokens")
        val lifetimes = listOf("5m", "1h", "30m", "12h")
        val portions = lifetimes.map { n("cache_write_${it}_tokens") }
        val written = maxOf(n("cache_write_tokens"), portions.sum())
        val ordinary = (input - read - written).coerceAtLeast(0)
        val inRate = rate.getDouble(if (long) "long_input" else "input")
        val outRate = rate.getDouble(if (long) "long_output" else "output")
        val maxInput = if (long) inRate else price("max_input", inRate)
        val cached = if (long) price("long_cached_input", price("cached_input", inRate)) else price("cached_input", inRate)
        val maxCached = if (long) cached else price("max_cached_input", cached)
        val writes = rate.optJSONObject("cache_write_multipliers") ?: JSONObject()
        var incomplete = call.optBoolean("pricing_incomplete") || call.optBoolean("explicit_cache") || read + written > input
        if (read > 0 && rate.isNull("cached_input")) incomplete = true
        val multiplier = if (batch) rate.optDouble("batch_multiplier", 1.0) else 1.0
        // Some providers keep cache-read pricing at the standard rate in Batch.
        val cachedMultiplier = if (batch) rate.optDouble("batch_cached_multiplier", multiplier) else 1.0
        val cachedFactor = if (batch && !rate.isNull("cached_input") && multiplier > 0.0) cachedMultiplier / multiplier else 1.0
        var lowInput = ordinary * inRate + read * cached * cachedFactor
        var highInput = ordinary * maxInput + read * maxCached * cachedFactor
        for ((i, ttl) in lifetimes.withIndex()) {
            if (portions[i] > 0 && writes.isNull(ttl)) incomplete = true
            lowInput += portions[i] * inRate * writes.optDouble(ttl, 1.0)
            highInput += portions[i] * maxInput * writes.optDouble(ttl, 1.0)
        }
        val unknown = (written - portions.sum()).coerceAtLeast(0)
        if (unknown > 0) {
            incomplete = true
            val possible = writes.keys().asSequence().map { writes.getDouble(it) }.toList()
            lowInput += unknown * inRate * (possible.minOrNull() ?: 1.0)
            highInput += unknown * maxInput * (possible.maxOrNull() ?: 1.0)
        }
        var low = (lowInput + output * outRate) * multiplier / 1e6
        var high = (highInput + output * price("max_output", outRate)) * multiplier / 1e6
        var lowerBound = false
        val tools = rate.optJSONObject("tool_rates") ?: JSONObject()
        fun add(count: Int, key: String) {
            if (count <= 0) return
            if (tools.isNull(key)) { incomplete = true; return }
            val cost = count * tools.getDouble(key)
            low += cost; high += cost
        }
        val searches = n("web_search_calls")
        add(searches, if (call.optBoolean("search_preview")) "web_search_preview" else "web_search")
        if (searches > 0 && !call.optBoolean("search_preview") && rate.has("search_content_tokens")) {
            val cost = searches * rate.getDouble("search_content_tokens") * inRate / 1e6
            low += cost; high += cost
        }
        add(n("file_search_calls"), "file_search")
        val queries = n("grounding_queries")
        add(if (tools.has("grounding_prompt")) minOf(1, queries) else queries, if (tools.has("grounding_prompt")) "grounding_prompt" else "grounding_query")
        val code = n("code_execution_calls")
        if (code > 0 && !call.optBoolean("code_execution_free")) {
            when (rate.optString("provider")) {
                "openai" -> { add((ModelUsagePricing.count(call, "container_sessions") ?: 1) * maxOf(1, n("container_memory_gb")), "container_session"); incomplete = true }
                "anthropic" -> { add(1, "code_execution_minimum"); incomplete = true; lowerBound = true }
                else -> add(code, "code_execution")
            }
        }
        return Amount(low, high, incomplete, lowerBound)
    }
}
