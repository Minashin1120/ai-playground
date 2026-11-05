package com.minashin1120.aiplayground.data

import java.util.Locale
import org.json.JSONObject

const val COST_ESTIMATE_NOTE = "有料APIの通常料金による概算（USD）。回答ごとの保存済みモデルで計算します。入力使用量がない回答は会話履歴から推定します。キャッシュ割引・保存料、検索・ツール・画像・音声・動画生成、税、無料枠・個別契約は含みません。時間帯や入力種別で単価が異なる場合は範囲で表示します。過去の請求額を再現するものではありません。"

data class CostEstimate(val low: Double = 0.0, val high: Double = 0.0, val priced: Int = 0,
                        val unpriced: Int = 0, val inferred: Int = 0, val long: Int = 0) {
    fun label(prefix: String = "費用予想"): String {
        if (priced == 0) return "$prefix: 算出不可"
        fun money(n: Double): String = if (n > 0 && n < 0.000001) "<\$0.000001" else String.format(Locale.US, "\$%.6f", n)
        val amount = if (high - low > 1e-12) "${money(low)}〜${money(high)}" else money(low)
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

    fun estimate(messages: List<ChatMessage>, context: List<ChatMessage> = messages): CostEstimate {
        val rows = context.associateBy { it.id }
        var low = 0.0; var high = 0.0; var priced = 0; var unpriced = 0; var inferred = 0; var longCount = 0
        for (m in messages.distinctBy { it.id }) {
            if (m.role != "assistant") continue
            val out = m.tokensOut?.takeIf { it >= 0 }
            if (out == null && m.content.isEmpty() && (m.tokens ?: 0) == 0) continue
            val key = data.getJSONObject("aliases").optString(m.model).ifBlank { m.model }
            val base = data.getJSONObject("models").optJSONObject(key)
            if (base == null || out == null) { unpriced++; continue }
            val rate = JSONObject(base.toString())
            val date = Regex("^\\d{4}-\\d{2}-\\d{2}").find(m.createdAt)?.value ?: checkedAt
            rate.optJSONArray("changes")?.let { changes ->
                for (i in 0 until changes.length()) {
                    val change = changes.getJSONObject(i)
                    if (date >= change.getString("from")) for (field in change.keys()) rate.put(field, change.get(field))
                }
            }
            var input = runCatching { JSONObject(m.thought).nullableInt("pricing_input_tokens") }.getOrNull()
                ?.takeIf { it >= 0 }?.toDouble() ?: m.tokensIn?.takeIf { it >= 0 }?.toDouble()
            var incomplete = false
            if (input == null) {
                inferred++
                input = 0.0
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
            val threshold = if (rate.isNull("threshold")) null else rate.getDouble("threshold")
            val long = threshold != null && if (rate.optBoolean("threshold_inclusive")) input >= threshold else input > threshold
            if (long) longCount++
            val inRate = rate.getDouble(if (long) "long_input" else "input")
            val outRate = rate.getDouble(if (long) "long_output" else "output")
            val multiplier = if (m.batch?.state == "JOB_STATE_SUCCEEDED") rate.optDouble("batch_multiplier", 1.0) else 1.0
            low += (input * inRate + out * outRate) * multiplier / 1e6
            high += (input * (if (long) inRate else rate.optDouble("max_input", inRate)) + out * rate.optDouble("max_output", outRate)) * multiplier / 1e6
            priced++
            if (incomplete) unpriced++
        }
        return CostEstimate(low, high, priced, unpriced, inferred, longCount)
    }
}
