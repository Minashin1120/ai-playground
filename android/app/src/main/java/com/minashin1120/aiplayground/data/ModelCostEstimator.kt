package com.minashin1120.aiplayground.data

import java.util.Locale
import org.json.JSONObject

const val COST_ESTIMATE_NOTE = "有料APIの通常料金による概算（USD）。回答ごとの保存済みモデルで計算します。入力使用量がない回答は会話履歴から推定します。モデルAPIの呼び出しごとに入力・出力（Thinkingを含む）を合算します。Ideogramはモデル・品質・速度・処理別の単価と画像枚数で計算します。単価未公表の設定は公表最安値を下限として「以上・一部未算入」で表示します。使用量未取得の呼び出しは一部未算入です。API報告のキャッシュ読込・書込と検索などの追加料金を含みます。Batch APIは事業者ごとの割引率（割引のない事業者を含む）で計算し、割引対象外のキャッシュ読込は通常単価のままです。保存時間や実行時間など取得できない課金情報は一部未算入です。その他の画像・音声・動画生成、税、無料枠・個別契約は含みません。時間帯や入力種別で単価が異なる場合は範囲で表示します。過去の請求額を再現するものではありません。"

data class CostEstimate(val low: Double = 0.0, val high: Double = 0.0, val priced: Int = 0,
                        val unpriced: Int = 0, val inferred: Int = 0, val long: Int = 0, val requests: Int = 0, val lowerBound: Boolean = false) {
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
        fun result() = CostEstimate(low, high, priced, unpriced, inferred, long, requests, lowerBound)
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

    private fun add(total: Total, call: JSONObject, date: String, batch: Boolean) {
        total.requests++
        val ticks = call.opt("cost_in_usd_ticks")
        if (call.optString("model").startsWith("grok-") && ticks is Number && ticks.toLong() >= 0) {
            total.low += ticks.toDouble() / 1e10; total.high += ticks.toDouble() / 1e10; total.priced++
            if (call.has("complete") && !call.optBoolean("complete")) total.unpriced++
            return
        }
        val rate = rate(call.optString("model"), date)
        if (rate == null) { total.unpriced++; return }
        if (rate.optString("unit") == "image") {
            val count = call.nullableInt("images")?.takeIf { it >= 0 }
            val prices = rate.optJSONObject("image_rates")?.opt(call.optString("operation"))
            val selected = (prices as? JSONObject)?.opt(call.optString(if (rate.optString("image_selector") == "speed") "speed" else "quality"))
            val minimum = (selected as? JSONObject)?.optDouble("minimum", Double.NaN)
                ?: (prices as? JSONObject)?.optDouble("minimum", Double.NaN) ?: Double.NaN
            val unit = (prices as? Number)?.toDouble() ?: (selected as? Number)?.toDouble() ?: minimum
            val lowerBound = unit.isFinite() && prices !is Number && selected !is Number
            if (count == null || !unit.isFinite()) { total.unpriced++; return }
            total.low += count * unit; total.high += count * unit; total.priced++
            if (lowerBound) total.lowerBound = true
            if (lowerBound || (call.has("complete") && !call.optBoolean("complete"))) total.unpriced++
            return
        }
        val input = call.nullableInt("input_tokens")?.takeIf { it >= 0 }
        val output = call.nullableInt("output_tokens")?.takeIf { it >= 0 }
        if (input == null && output == null) { total.unpriced++; return }
        val threshold = if (rate.isNull("threshold")) null else rate.getDouble("threshold")
        val long = threshold != null && if (rate.optBoolean("threshold_inclusive")) (input ?: 0) >= threshold else (input ?: 0) > threshold
        if (long) total.long++
        val amount = ModelCostPricing.amount(call, rate, long, batch)
        total.low += amount.low; total.high += amount.high
        if (amount.lowerBound) total.lowerBound = true
        total.priced++
        if (input == null || output == null || amount.incomplete || (call.has("complete") && !call.optBoolean("complete"))) total.unpriced++

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
        val action = options.optString("ideogram_operation")
        val operation = if (action in setOf("remove_object", "remove_background")) action else if (source && m.model == "ideogram-4.5") "edit" else "generate"
        var quality = options.optString("ideogram_quality")
        if (quality !in setOf("very_low", "low", "medium", "high") || (quality == "very_low" && operation == "generate")) quality = if (operation == "edit") "medium" else "high"
        val speed = if (parent == null) null else options.optString("ideogram_speed").takeIf { it in setOf("turbo", "default", "quality") } ?: "default"
        return JSONObject().put("model", m.model).put("images", m.files.size.takeIf { it > 0 } ?: JSONObject.NULL)
            .put("operation", operation).put("quality", quality).put("speed", speed ?: JSONObject.NULL).put("complete", parent != null)
    }

    fun estimate(messages: List<ChatMessage>, context: List<ChatMessage> = messages): CostEstimate {
        val rows = context.associateBy { it.id }
        val total = Total()
        for (m in messages.distinctBy { it.id }) {
            if (m.role != "assistant") continue
            val metadata = runCatching { JSONObject(m.thought) }.getOrNull()
            val date = referenceDate(m.createdAt)
            val batch = m.batch?.state == "JOB_STATE_SUCCEEDED"
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
                        add(total, call, referenceDate(call.optString("created_at"), date), batch)
                    }
                }
                continue
            }
            if (rate(m.model, date)?.optString("unit") == "image") {
                add(total, legacyImageRequest(m, rows), date, batch)
                continue
            }
            val out = m.tokensOut?.takeIf { it >= 0 }
            if (out == null && m.content.isEmpty() && (m.tokens ?: 0) == 0) continue
            if (out == null) { total.unpriced++; continue }
            var input = metadata?.nullableInt("pricing_input_tokens")?.takeIf { it >= 0 } ?: m.tokensIn?.takeIf { it >= 0 }
            var incomplete = false
            if (input == null) {
                total.inferred++; input = 0
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
            add(total, JSONObject().put("model", m.model).put("input_tokens", input).put("output_tokens", out).put("complete", !incomplete), date, batch)
        }
        return total.result()
    }
}
