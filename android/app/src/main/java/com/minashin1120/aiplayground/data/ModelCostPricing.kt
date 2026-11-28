package com.minashin1120.aiplayground.data

import org.json.JSONObject

/** Replaces cached input portions and adds tool fees independently of Batch token discounts. */
internal object ModelCostPricing {
    data class Amount(val low: Double, val high: Double, val incomplete: Boolean, val lowerBound: Boolean, val derived: Boolean = false)
    fun amount(call: JSONObject, rate: JSONObject, long: Boolean, batch: Boolean, flex: Boolean = false): Amount {
        fun n(key: String) = ModelUsagePricing.count(call, key) ?: 0
        fun nOpt(key: String) = ModelUsagePricing.count(call, key)
        fun price(key: String, fallback: Double) = if (rate.isNull(key)) fallback else rate.optDouble(key, fallback)
        val rawIn = nOpt("input_tokens")
        val rawOut = nOpt("output_tokens")
        val input = rawIn ?: 0
        val output = rawOut ?: 0
        val read = n("cache_read_tokens")
        val lifetimes = listOf("5m", "1h", "30m", "12h")
        val portions = lifetimes.map { n("cache_write_${it}_tokens") }
        val written = maxOf(n("cache_write_tokens"), portions.sum())
        val ordinary = (input - read - written).coerceAtLeast(0)
        val baseIn = if (!rate.isNull("text_input")) rate.optDouble("text_input") else rate.optDouble("input", 0.0)
        val inRate = if (long) rate.optDouble("long_input", baseIn) else baseIn
        val outRate = if (long) rate.optDouble("long_output", rate.optDouble("output", 0.0)) else rate.optDouble("output", 0.0)
        val maxInput = if (long) inRate else price("max_input", inRate)
        val cached = if (long) price("long_cached_input", price("cached_input", inRate)) else price("cached_input", inRate)
        val maxCached = if (long) cached else price("max_cached_input", cached)
        val writes = rate.optJSONObject("cache_write_multipliers") ?: JSONObject()
        var incomplete = call.optBoolean("pricing_incomplete") || rate.optBoolean("pricing_unlisted") || call.optBoolean("explicit_cache") || read + written > input
        if (read > 0 && rate.isNull("cached_input")) incomplete = true
        val multiplier = if (batch) rate.optDouble("batch_multiplier", 1.0) else 1.0
        // Some providers keep cache-read pricing at the standard rate in Batch.
        val cachedMultiplier = if (batch) rate.optDouble("batch_cached_multiplier", multiplier) else 1.0
        val cachedFactor = if (batch && !rate.isNull("cached_input") && multiplier > 0.0) cachedMultiplier / multiplier else 1.0
        var derived = false
        val audioTps = if (rate.isNull("audio_tokens_per_second")) null else rate.optDouble("audio_tokens_per_second")
        var estAudioIn: Double? = null
        var estAudioOut: Double? = null
        if (audioTps != null) {
            val inMs = nOpt("audio_in_ms") ?: nOpt("duration_ms")
            if (rawIn == null && inMs != null) { estAudioIn = (inMs / 1000.0) * audioTps; derived = true }
            val outMs = nOpt("audio_out_ms")
            if (rawOut == null && outMs != null && !rate.isNull("audio_output")) { estAudioOut = (outMs / 1000.0) * audioTps; derived = true }
        }
        val audioInTok = nOpt("audio_input_tokens")
        val textInTok = nOpt("text_input_tokens")
        val imgInTok = nOpt("image_input_tokens")
        var lowInput = 0.0
        var highInput = 0.0
        if (audioInTok != null || textInTok != null || imgInTok != null) {
            val cText = n("cached_text_tokens"); val cImg = n("cached_image_tokens"); val cAudio = n("cached_audio_tokens")
            val hasModCache = !call.isNull("cached_text_tokens") || !call.isNull("cached_image_tokens") || !call.isNull("cached_audio_tokens")
            val ordText = ((textInTok ?: 0) - cText).coerceAtLeast(0)
            val ordImg = ((imgInTok ?: 0) - cImg).coerceAtLeast(0)
            val ordAudio = ((audioInTok ?: 0) - cAudio).coerceAtLeast(0)
            val remOrd = (ordinary - ordText - ordImg - ordAudio).coerceAtLeast(0)
            val imgInRate = price("image_input", inRate)
            val audInRate = price("audio_input", inRate)
            lowInput = ordText * inRate + ordImg * imgInRate + ordAudio * audInRate + remOrd * inRate
            highInput = ordText * maxInput + ordImg * imgInRate + ordAudio * audInRate + remOrd * maxInput
            if (hasModCache) {
                val modCache = (cText * cached + cImg * price("cached_image_input", price("image_input", cached)) + cAudio * price("cached_audio_input", price("audio_input", cached))) * cachedFactor
                lowInput += modCache; highInput += modCache
            } else {
                lowInput += read * cached * cachedFactor
                highInput += read * maxCached * cachedFactor
            }
        } else if (estAudioIn != null) {
            val v = estAudioIn * price("audio_input", inRate)
            lowInput = v; highInput = v
        } else {
            val modInMax = if (rate.optString("unit") == "image" && !rate.isNull("image_input")) maxOf(maxInput, rate.optDouble("image_input")) else maxInput
            lowInput = ordinary * inRate + read * cached * cachedFactor
            highInput = ordinary * (if (long) inRate else modInMax) + read * maxCached * cachedFactor
        }
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
        val imgOutTok = nOpt("image_output_tokens")
        val audOutTok = nOpt("audio_output_tokens")
        val maxOut = price("max_output", outRate)
        var lowOut = 0.0; var highOut = 0.0; var perImageExtra = 0.0
        val imgPerImage = rate.optJSONObject("image_per_image")
        when {
            imgOutTok != null && !rate.isNull("image_output") -> {
                val txtOut = (output - imgOutTok).coerceAtLeast(0)
                val imgRate = rate.getDouble("image_output")
                lowOut = txtOut * outRate + imgOutTok * imgRate
                highOut = txtOut * maxOut + imgOutTok * imgRate
            }
            audOutTok != null && !rate.isNull("audio_output") -> {
                val txtOut = (output - audOutTok).coerceAtLeast(0)
                val audRate = rate.getDouble("audio_output")
                lowOut = txtOut * outRate + audOutTok * audRate
                highOut = txtOut * maxOut + audOutTok * audRate
            }
            estAudioOut != null && !rate.isNull("audio_output") -> {
                val v = estAudioOut * rate.getDouble("audio_output")
                lowOut = v; highOut = v
            }
            imgPerImage != null && nOpt("images") != null -> {
                val cnt = n("images")
                val res = call.optString("resolution").lowercase()
                val vals = imgPerImage.keys().asSequence().mapNotNull { k -> imgPerImage.optDouble(k).takeIf { it.isFinite() } }.toList()
                val pVal = if (res.isNotEmpty() && !imgPerImage.isNull(res)) imgPerImage.optDouble(res) else if (!imgPerImage.isNull("default")) imgPerImage.optDouble("default") else null
                val loP = pVal ?: (vals.minOrNull() ?: 0.0)
                val hiP = pVal ?: (vals.maxOrNull() ?: 0.0)
                if (cnt > 0) {
                    derived = true
                    perImageExtra = cnt * loP
                    val txtOut = if (imgOutTok != null) (output - imgOutTok).coerceAtLeast(0) else if (rawOut != null && !rate.isNull("image_output")) 0 else output
                    lowOut = txtOut * outRate
                    highOut = txtOut * maxOut + cnt * (hiP - loP).coerceAtLeast(0.0) * 1e6
                }
            }
            !rate.isNull("image_output") && output > 0 -> {
                lowOut = output * outRate
                highOut = output * rate.getDouble("image_output")
                incomplete = true
            }
            else -> {
                lowOut = output * outRate
                highOut = output * maxOut
            }
        }
        val mul = multiplier * (if (flex) 0.5 else 1.0)
        var low = ((lowInput + lowOut) / 1e6 + perImageExtra) * mul
        var high = ((highInput + highOut) / 1e6 + perImageExtra) * mul
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
        return Amount(low, high, incomplete, lowerBound, derived)
    }
}
