package com.minashin1120.aiplayground.data

import java.util.Locale
import org.json.JSONObject

const val COST_ESTIMATE_NOTE = "有料APIの通常料金による概算（USD）。回答ごとの保存済みモデルで計算します。入力使用量がない回答は会話履歴から推定します。モデルAPIの呼び出しごとに入力・出力（Thinkingを含む）を合算します。Ideogramはモデル・品質・速度・処理別の単価と画像枚数で計算します。単価未公表の設定は公表最安値を下限として「以上・一部未算入」で表示します。使用量未取得の呼び出しは一部未算入です。API報告のキャッシュ読込・書込と検索などの追加料金を含みます。Batch APIは事業者ごとの割引率（割引のない事業者を含む）、Gemini Flexは標準料金の50%で計算し、割引対象外のキャッシュ読込は通常単価のままです。保存時間や実行時間など取得できない課金情報は一部未算入です。その他の画像・音声・動画生成、税、無料枠・個別契約は含みません。時間帯や入力種別で単価が異なる場合は範囲で表示します。過去の請求額を再現するものではありません。"

/** One saved API request in the token-detail breakdown, mirroring Web's `cost_estimate.items`. */
data class CostItem(val model: String = "", val date: String = "", val input: Int? = null, val output: Int? = null,
                    val cacheRead: Int? = null, val cacheWrite: Int? = null, val images: Int? = null, val tools: Int = 0,
                    val low: Double = 0.0, val high: Double = 0.0, val priced: Boolean = false, val reason: String = "",
                    val inferred: Boolean = false, val long: Boolean = false, val batch: Boolean = false,
                    val flex: Boolean = false, val apiKeyId: String = "", val apiKeyLabel: String = "",
                    val apiKeyInferred: Boolean = false, val pages: Int? = null, val characters: Int? = null,
                    val songs: Int? = null, val durationMs: Int? = null) {
    fun money(n: Double): String = if (n > 0 && n < 0.000001) "<\$0.000001" else String.format(Locale.US, "\$%.6f", n)

    /** Same wording as Web's `costAmountText`. */
    fun amountText(): String = when {
        !priced -> "算出不可"
        high - low > 1e-12 -> "${money(low)}〜${money(high)}"
        else -> money(low)
    }

    /** Same wording as Web's `costItemTokenText`. */
    fun tokenText(): String = buildList {
        images?.let { add("${it}枚") }
        songs?.let { add("${it}曲") }
        pages?.let { add("${it}ページ") }
        characters?.let { add("${it}文字") }
        durationMs?.let { add("${String.format(Locale.US, "%.1f", it / 1000.0)}秒") }
        if (input != null || output != null) add("入 ${input ?: "-"} / 出 ${output ?: "-"}")
        if ((cacheRead ?: 0) > 0 || (cacheWrite ?: 0) > 0) add("キャッシュ 読込 ${cacheRead ?: 0} / 書込 ${cacheWrite ?: 0}")
        if (tools > 0) add("ツール ${tools}回")
    }.joinToString(" · ")

    fun flagsText(): String = buildList {
        if (inferred) add("履歴から推定")
        if (long) add("長文料金")
        if (batch) add("Batch")
        if (flex) add("Flex")
        if (reason.isNotEmpty()) add(reason)
    }.joinToString(" / ")
}

data class CostEstimate(val low: Double = 0.0, val high: Double = 0.0, val priced: Int = 0,
                        val unpriced: Int = 0, val inferred: Int = 0, val long: Int = 0, val requests: Int = 0, val lowerBound: Boolean = false,
                        val items: List<CostItem> = emptyList()) {
    fun label(prefix: String = "費用予想"): String {
        if (priced == 0) return "$prefix: 算出不可"
        fun money(n: Double): String = if (n > 0 && n < 0.000001) "<\$0.000001" else String.format(Locale.US, "\$%.6f", n)
        val amount = if (lowerBound) "${money(low)}以上" else if (high - low > 1e-12) "${money(low)}〜${money(high)}" else money(low)
        return "$prefix: $amount${if (unpriced > 0) "（一部未算入）" else ""}"
    }
}

/** Matches Web's per-answer calculation; the selected composer model is deliberately absent. */
class ModelCostEstimator(json: String) {
    private val data = JSONObject(json)
    val checkedAt: String = data.getString("checked_at")
    val sources: List<Pair<String, String>> = data.getJSONObject("sources").let { obj ->
        obj.keys().asSequence().map { it to obj.getString(it) }.toList()
    }

    private class Total {
        var low = 0.0; var high = 0.0; var priced = 0; var unpriced = 0; var inferred = 0; var long = 0; var requests = 0; var lowerBound = false
        val items = mutableListOf<CostItem>()
        fun result() = CostEstimate(low, high, priced, unpriced, inferred, long, requests, lowerBound, items.toList())
    }

    private fun rate(model: String, date: String): JSONObject? {
        val key = data.getJSONObject("aliases").optString(model).ifBlank { model }
        val base = data.getJSONObject("models").optJSONObject(key) ?: return null
        val rate = JSONObject(base.toString())
        rate.optJSONArray("changes")?.let { changes ->
            for (i in 0 until changes.length()) {
                val change = changes.getJSONObject(i)
                if (date >= change.getString("from")) for (field in change.keys()) rate.put(field, change.get(field))
            }
        }
        return rate
    }

    private fun referenceDate(value: String, fallback: String = checkedAt) = Regex("^\\d{4}-\\d{2}-\\d{2}").find(value)?.value ?: fallback

    private fun add(total: Total, call: JSONObject, date: String, batch: Boolean, flex: Boolean = false) {
        total.requests++
        val ticks = call.opt("cost_in_usd_ticks")
        if (call.optString("model").startsWith("grok-") && ticks is Number && ticks.toLong() >= 0) {
            total.low += ticks.toDouble() / 1e10; total.high += ticks.toDouble() / 1e10; total.priced++
            if (call.has("complete") && !call.optBoolean("complete")) total.unpriced++
            return
        }
        val rate = rate(call.optString("model"), date)
        if (rate == null) { total.unpriced++; return }
        val unit = rate.optString("unit")
        if (unit.isNotEmpty() && unit != "token") {
            addUnit(total, call, rate, unit, batch, flex)
            return
        }
        val input = call.nullableInt("input_tokens")?.takeIf { it >= 0 }
        val output = call.nullableInt("output_tokens")?.takeIf { it >= 0 }
        val mediaFallback = (!rate.isNull("image_per_image") && call.nullableInt("images") != null) ||
            (!rate.isNull("audio_tokens_per_second") && listOf("audio_in_ms", "audio_out_ms", "duration_ms").any { call.nullableInt(it) != null })
        if (input == null && output == null && !mediaFallback) { total.unpriced++; return }
        val threshold = if (rate.isNull("threshold")) null else rate.getDouble("threshold")
        val long = threshold != null && if (rate.optBoolean("threshold_inclusive")) (input ?: 0) >= threshold else (input ?: 0) > threshold
        if (long) total.long++
        val amount = ModelCostPricing.amount(call, rate, long, batch, flex)
        total.low += amount.low; total.high += amount.high
        if (amount.lowerBound) total.lowerBound = true
        total.priced++
        val missingTok = (input == null || output == null) && !amount.derived
        if (missingTok || amount.incomplete || (call.has("complete") && !call.optBoolean("complete"))) total.unpriced++
    }

    private fun addImageUnit(total: Total, call: JSONObject, rate: JSONObject, batch: Boolean, flex: Boolean) {
        val flexMul = if (flex) 0.5 else 1.0
        val batchMul = if (batch) rate.optDouble("batch_multiplier", 1.0) else 1.0
        val inTok = call.nullableInt("input_tokens")?.takeIf { it >= 0 }
        val outTok = call.nullableInt("output_tokens")?.takeIf { it >= 0 }
        if ((inTok != null || outTok != null) && (!rate.isNull("image_output") || !rate.isNull("output"))) {
            val amount = ModelCostPricing.amount(call, rate, false, batch, flex)
            total.low += amount.low; total.high += amount.high
            if (amount.lowerBound) total.lowerBound = true
            total.priced++
            if (amount.incomplete || (call.has("complete") && !call.optBoolean("complete"))) total.unpriced++
            return
        }
        val count = call.nullableInt("images")?.takeIf { it >= 0 }
        if (count == null) { total.unpriced++; return }
        var low: Double? = null; var high: Double? = null; var lowerBound = false; var incomplete = false
        rate.optJSONObject("per_image_rates")?.let { qMap ->
            val q = call.optString("quality").takeIf { it.isNotBlank() && it != "auto" }
            val s = call.optString("size").takeIf { it.isNotBlank() && it != "auto" }
            val vals = mutableListOf<Double>()
            if (q != null) {
                val qObj = qMap.optJSONObject(q)
                if (qObj != null) {
                    if (s != null && !qObj.isNull(s)) vals.add(qObj.getDouble(s))
                    else qObj.keys().asSequence().forEach { k -> qObj.optDouble(k).takeIf { it.isFinite() }?.let(vals::add) }
                }
            } else {
                qMap.keys().asSequence().forEach { qk ->
                    val sub = qMap.optJSONObject(qk) ?: return@forEach
                    sub.keys().asSequence().forEach { sk ->
                        if (s == null || sk == s) sub.optDouble(sk).takeIf { it.isFinite() }?.let(vals::add)
                    }
                }
            }
            if (vals.isNotEmpty()) {
                low = count * vals.min() * batchMul * flexMul
                high = count * vals.max() * batchMul * flexMul
                if (vals.size > 1) incomplete = true
            }
        }
        if (low == null) {
            val res = call.optString("resolution").lowercase()
            val sub = if (res.isNotEmpty()) rate.optJSONObject("resolution_rates")?.opt(res) else null
            val u = when (sub) {
                is Number -> sub.toDouble()
                is JSONObject -> (sub.opt(call.optString("quality")) as? Number)?.toDouble() ?: (sub.opt("low") as? Number)?.toDouble()
                    ?: sub.keys().asSequence().mapNotNull { k -> (sub.opt(k) as? Number)?.toDouble() }.firstOrNull()
                else -> null
            }
            if (u != null && u.isFinite()) { low = count * u; high = count * u }
        }
        if (low == null) {
            val ir = rate.optJSONObject("image_rates")
            val prices = ir?.opt(call.optString("operation")) ?: ir?.opt("generate")
            val selected = (prices as? JSONObject)?.opt(call.optString(if (rate.optString("image_selector") == "speed") "speed" else "quality"))
            val resKey = call.optString("resolution").lowercase()
            val resPrice = (selected as? JSONObject)?.takeIf { resKey.isNotEmpty() && !it.isNull(resKey) }?.opt(resKey) as? Number
            if (prices is Number || selected is Number || resPrice != null) {
                val unitPrice = (prices as? Number)?.toDouble() ?: (selected as? Number)?.toDouble() ?: resPrice!!.toDouble()
                low = count * unitPrice * flexMul; high = count * unitPrice * flexMul
            } else {
                val vals = (selected as? JSONObject)?.keys()?.asSequence()
                    ?.filter { it != "minimum" }
                    ?.mapNotNull { (selected.opt(it) as? Number)?.toDouble() }
                    ?.toList().orEmpty()
                if (vals.isNotEmpty()) {
                    val minVal = vals.min()
                    val maxVal = vals.max()
                    low = count * minVal * flexMul; high = count * maxVal * flexMul
                    if (minVal != maxVal) incomplete = true
                } else {
                    val minimum = (selected as? JSONObject)?.optDouble("minimum", Double.NaN)
                        ?: (prices as? JSONObject)?.optDouble("minimum", Double.NaN) ?: Double.NaN
                    lowerBound = minimum.isFinite()
                    if (!minimum.isFinite()) { total.unpriced++; return }
                    low = count * minimum * flexMul; high = count * minimum * flexMul
                }
            }
        }
        val extraIn = (call.nullableInt("input_images")?.takeIf { it >= 0 } ?: 0) * rate.optDouble("input_image_rate", 0.0)
        total.low += (low ?: 0.0) + extraIn; total.high += (high ?: 0.0) + extraIn
        total.priced++
        if (lowerBound) total.lowerBound = true
        if (lowerBound || incomplete || (call.has("complete") && !call.optBoolean("complete"))) total.unpriced++
    }

    private fun addUnit(total: Total, call: JSONObject, rate: JSONObject, unit: String, batch: Boolean, flex: Boolean) {
        if (unit == "image") { addImageUnit(total, call, rate, batch, flex); return }
        val batchMul = if (batch) rate.optDouble("batch_multiplier", 1.0) else 1.0
        val flexMul = if (flex) 0.5 else 1.0
        var low = 0.0; var high = 0.0; var incomplete = call.has("complete") && !call.optBoolean("complete")
        when (unit) {
            "song" -> { val n = call.nullableInt("songs")?.takeIf { it >= 0 } ?: 1; val v = n * rate.optDouble("song_rate", 0.0); low = v; high = v }
            "page" -> {
                val p = call.nullableInt("pages")?.takeIf { it >= 0 } ?: run { total.unpriced++; return }
                val v = p * rate.optDouble("page_rate", 0.0) * batchMul; low = v; high = v
            }
            "character" -> {
                val c = call.nullableInt("characters")?.takeIf { it >= 0 } ?: run { total.unpriced++; return }
                val v = c * rate.optDouble("character_rate", 0.0) / 1e6; low = v; high = v
            }
            "duration" -> {
                val ms = call.nullableInt("duration_ms")?.takeIf { it >= 0 }
                val audioMs = (call.nullableInt("audio_in_ms")?.takeIf { it >= 0 } ?: 0) + (call.nullableInt("audio_out_ms")?.takeIf { it >= 0 } ?: 0)
                val mins = when {
                    ms != null -> ms / 60000.0
                    call.nullableInt("audio_in_ms") != null || call.nullableInt("audio_out_ms") != null -> audioMs / 60000.0
                    else -> null
                }
                val perMin = rate.optDouble("duration_rate_per_minute", 0.0)
                if (mins == null && perMin > 0.0) {
                    low = perMin / 60.0; high = perMin; incomplete = true
                } else {
                    val base = (mins ?: 0.0) * perMin
                    val standby = ((call.nullableInt("standby_ms")?.takeIf { it >= 0 } ?: 0) / 60000.0) * rate.optDouble("standby_rate_per_minute", 0.0)
                    val turns = (call.nullableInt("text_turns")?.takeIf { it >= 0 } ?: 0) * rate.optDouble("text_turn_rate", 0.0)
                    low = base + standby + turns; high = low
                }
            }
            "video" -> {
                val inTok = call.nullableInt("input_tokens")?.takeIf { it >= 0 }
                val vidTok = call.nullableInt("video_output_tokens")?.takeIf { it >= 0 }
                if (!rate.isNull("video_output") && inTok != null && vidTok != null) {
                    val outTok = call.nullableInt("output_tokens")?.takeIf { it >= 0 } ?: vidTok
                    val textOut = (outTok - vidTok).coerceAtLeast(0)
                    val v = (inTok * rate.optDouble("input", 0.0) + textOut * rate.optDouble("output", 0.0) + vidTok * rate.getDouble("video_output")) / 1e6
                    low = v; high = v
                } else {
                    val sec = call.nullableInt("duration_ms")?.takeIf { it >= 0 }?.let { it / 1000.0 }
                        ?: call.optDouble("duration_seconds", Double.NaN).takeIf { it.isFinite() && it >= 0.0 }
                    val tbl = if (call.optBoolean("without_audio") && !rate.isNull("video_only_rates")) rate.optJSONObject("video_only_rates") else rate.optJSONObject("video_rates")
                    val res = call.optString("resolution").lowercase()
                    val rVal = if (tbl != null && res.isNotEmpty() && !tbl.isNull(res)) tbl.optDouble(res).takeIf { it.isFinite() } else null
                    val vals = tbl?.keys()?.asSequence()?.mapNotNull { k -> tbl.optDouble(k).takeIf { it.isFinite() } }?.toList().orEmpty()
                    if (sec != null && vals.isNotEmpty()) {
                        val defVal = tbl?.optDouble("default", Double.NaN)?.takeIf { it.isFinite() }
                        val extra = (call.nullableInt("input_images")?.takeIf { it >= 0 } ?: 0) * rate.optDouble("input_image_rate", 0.0)
                        low = sec * (rVal ?: defVal ?: vals.min()) + extra
                        high = sec * (rVal ?: defVal ?: vals.max()) + extra
                        if (rVal == null && vals.size > 1) incomplete = true
                    } else if (vals.isNotEmpty()) {
                        val dSec = if (call.optString("model").startsWith("grok-")) 5.0 else 8.0
                        low = dSec * vals.min(); high = dSec * vals.max(); incomplete = true
                    } else { total.unpriced++; return }
                }
            }
            else -> { total.unpriced++; return }
        }
        total.low += low * flexMul; total.high += high * flexMul; total.priced++
        if (incomplete) total.unpriced++
    }

    /** Runs [add] and records the request's own share from the difference in the running totals. */
    private fun addDetailed(total: Total, call: JSONObject, date: String, batch: Boolean, flex: Boolean, inferred: Boolean) {
        val low = total.low; val high = total.high; val priced = total.priced
        val unpriced = total.unpriced; val long = total.long
        add(total, call, date, batch, flex)
        fun count(key: String) = call.nullableInt(key)?.takeIf { it >= 0 }
        var writes = count("cache_write_tokens") ?: 0
        var ttlWrites = 0
        for (ttl in listOf("5m", "1h", "30m", "12h")) {
            val value = count("cache_write_${ttl}_tokens") ?: 0
            writes = maxOf(writes, value)
            ttlWrites += value
        }
        val tools = listOf("web_search_calls", "file_search_calls", "grounding_queries", "code_execution_calls")
            .sumOf { count(it) ?: 0 }
        val wasPriced = total.priced > priced
        val reason = when {
            !wasPriced -> if (rate(call.optString("model"), date) != null) "使用量未取得" else "単価未登録"
            total.unpriced > unpriced -> "一部未算入"
            else -> ""
        }
        total.items.add(CostItem(
            model = call.optString("model"), date = date, input = count("input_tokens"), output = count("output_tokens"),
            cacheRead = count("cache_read_tokens"), cacheWrite = maxOf(writes, ttlWrites).takeIf { it > 0 },
            images = count("images"), tools = tools, low = total.low - low, high = total.high - high,
            priced = wasPriced, reason = reason, inferred = inferred, long = total.long > long, batch = batch, flex = flex,
            apiKeyId = call.optString("api_key_id"), apiKeyLabel = call.optString("api_key_label"),
            apiKeyInferred = call.optBoolean("api_key_inferred"),
            pages = count("pages"), characters = count("characters"), songs = count("songs"), durationMs = count("duration_ms"),
        ))
    }

    private fun legacyImageRequest(m: ChatMessage, rows: Map<String, ChatMessage>): JSONObject {
        val parent = rows[m.parentId?.toString()]
        val options = parent?.promptOptions ?: JSONObject()
        var source = parent?.files?.isNotEmpty() == true
        var node = parent
        val visited = mutableSetOf<String>()
        while (!source && node?.parentId != null && visited.add(node.id)) {
            node = rows[node.parentId.toString()]
            if (node?.role == "assistant" && node.files.isNotEmpty()) source = true
        }
        val modelName = m.model
        if (modelName.startsWith("gpt-image-")) {
            return JSONObject().put("model", m.model).put("images", maxOf(1, m.files.size))
                .put("operation", if (source) "edit" else "generate")
                .put("quality", options.optString("gpt_image_quality").ifBlank { "auto" })
                .put("size", options.optString("gpt_image_size").ifBlank { "auto" }).put("complete", parent != null)
        }
        if (modelName.startsWith("grok-imagine-image")) {
            return JSONObject().put("model", m.model).put("images", maxOf(1, m.files.size))
                .put("operation", if (source) "edit" else "generate")
                .put("quality", options.optString("grok_image_quality").ifBlank { "medium" })
                .put("resolution", options.optString("grok_image_resolution").ifBlank { "1k" })
                .put("input_images", if (source) 1 else 0).put("complete", parent != null)
        }
        val is46 = m.model == "ideogram-4.6"
        val action = options.optString("ideogram_operation")
        val operation = if (action in setOf("remove_object", "remove_background")) action else if (source && (is46 || m.model == "ideogram-4.5")) "edit" else "generate"
        var quality = options.optString("ideogram_quality")
        if (quality !in setOf("very_low", "low", "medium", "high") || (quality == "very_low" && operation == "generate" && !is46)) {
            quality = if (operation == "edit" || is46) "medium" else "high"
        }
        val speed = if (parent == null) null else options.optString("ideogram_speed").takeIf { it in setOf("turbo", "default", "quality") } ?: "default"
        val resolution = if (is46) {
            val rawRes = options.optString("ideogram_resolution").lowercase()
            if (operation == "generate") {
                when {
                    rawRes == "4k" && quality != "very_low" -> "4k"
                    rawRes == "2k" || rawRes == "4k" -> "2k"
                    rawRes == "1k" -> "1k"
                    else -> null
                }
            } else if (quality == "very_low") "2k" else null
        } else null
        return JSONObject().put("model", m.model).put("images", m.files.size.takeIf { it > 0 } ?: JSONObject.NULL)
            .put("operation", operation).put("quality", quality).put("speed", speed ?: JSONObject.NULL)
            .apply { if (resolution != null) put("resolution", resolution) }
            .put("complete", parent != null)
    }

    private fun legacyUnitRequest(m: ChatMessage, rows: Map<String, ChatMessage>, rate: JSONObject): JSONObject {
        val parent = rows[m.parentId?.toString()]
        return when (rate.optString("unit")) {
            "song" -> JSONObject().put("model", m.model).put("songs", 1).put("complete", true)
            "page" -> {
                val match = Regex("合計処理ページ数:\\s*(\\d+)").find(m.content)
                JSONObject().put("model", m.model).put("pages", match?.groupValues?.get(1)?.toIntOrNull() ?: 1).put("complete", match != null)
            }
            "character" -> {
                val chars = parent?.content?.length
                JSONObject().put("model", m.model).put("characters", chars ?: JSONObject.NULL).put("complete", chars != null)
            }
            "video" -> {
                val opts = parent?.promptOptions ?: JSONObject()
                val isGrok = m.model.startsWith("grok-")
                val dur = opts.optString(if (isGrok) "grok_video_duration" else "gemini_video_duration").toIntOrNull()
                val res = opts.optString(if (isGrok) "grok_video_resolution" else "gemini_video_resolution").lowercase()
                JSONObject().put("model", m.model)
                    .put("duration_ms", dur?.let { it * 1000 } ?: JSONObject.NULL)
                    .put("resolution", res.ifEmpty { JSONObject.NULL }).put("complete", false)
            }
            else -> JSONObject().put("model", m.model).put("complete", false)
        }
    }

    fun estimate(messages: List<ChatMessage>, context: List<ChatMessage> = messages): CostEstimate {
        val rows = context.associateBy { it.id }
        val total = Total()
        for (m in messages.distinctBy { it.id }) {
            if (m.role != "assistant") continue
            val metadata = runCatching { JSONObject(m.thought) }.getOrNull()
            val date = referenceDate(m.createdAt)
            val batch = m.batch?.state == "JOB_STATE_SUCCEEDED"
            val promptOptions = (rows[m.parentId?.toString()] ?: m).promptOptions ?: JSONObject()
            val flex = promptOptions.optBoolean("gemini_flex")
            val calls = metadata?.optJSONArray("pricing_calls")
            if (calls != null) {
                if (calls.length() == 0) total.priced++
                for (i in 0 until calls.length()) {
                    val call = calls.optJSONObject(i)
                    if (call == null) total.unpriced++ else {
                        if (rate(call.optString("model"), date)?.optString("image_selector") == "speed" &&
                            !call.has("speed") && call.optString("model") == m.model) {
                            call.put("speed", legacyImageRequest(m, rows).opt("speed"))
                        }
                        addDetailed(total, call, referenceDate(call.optString("created_at"), date), batch, flex, false)
                    }
                }
                continue
            }
            val mRate = rate(m.model, date)
            val mUnit = mRate?.optString("unit").orEmpty()
            if (mUnit == "image") {
                addDetailed(total, legacyImageRequest(m, rows), date, batch, flex, false)
                continue
            }
            if (mUnit.isNotEmpty() && mUnit != "token") {
                addDetailed(total, legacyUnitRequest(m, rows, mRate!!), date, batch, flex, true)
                continue
            }
            val out = m.tokensOut?.takeIf { it >= 0 }
            if (out == null && m.content.isEmpty() && (m.tokens ?: 0) == 0) continue
            if (out == null) { total.unpriced++; continue }
            var input = metadata?.nullableInt("pricing_input_tokens")?.takeIf { it >= 0 } ?: m.tokensIn?.takeIf { it >= 0 }
            var incomplete = false
            var inferred = false
            if (input == null) {
                total.inferred++; inferred = true; input = 0
                var parent = m.parentId?.toString()
                val ancestors = mutableSetOf(m.id)
                if (parent == null) incomplete = true
                while (parent != null) {
                    val ancestor = rows[parent]
                    if (!ancestors.add(parent) || ancestor == null) { incomplete = true; break }
                    val count = if (ancestor.role == "user") ancestor.tokensIn?.takeIf { it >= 0 }
                        else ancestor.tokensContent?.takeIf { it >= 0 }
                            ?: ancestor.tokensOut?.takeIf { it >= 0 }?.let { (it - (ancestor.tokensThought ?: 0)).coerceAtLeast(0) }
                    if (count == null) incomplete = true else input += count
                    parent = ancestor.parentId?.toString()
                }
            }
            val legReq = JSONObject().put("model", m.model).put("input_tokens", input).put("output_tokens", out).put("complete", !incomplete)
            if (mRate?.isNull("image_per_image") == false) {
                val nImg = if (m.files.isNotEmpty()) m.files.size else if (m.content.contains("![Image](")) 1 else 0
                if (nImg > 0) legReq.put("images", nImg)
            }
            addDetailed(total, legReq, date, batch, flex, inferred)
        }
        return total.result()
    }
}
