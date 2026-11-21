package com.minashin1120.aiplayground.data

import org.json.JSONArray
import org.json.JSONObject

/** The account-wide cost estimate shown by the 費用 settings tab. */
data class UsageCostSummary(
    val period: String,
    val estimate: CostEstimate,
    val answers: Int,
    val unreadable: Int,
    val truncated: Boolean,
)

/** Period ids match the Web buttons: 1 day, 7 days, 2 weeks, 1 month, 3 months, 1 year. */
val USAGE_COST_PERIODS: List<Pair<String, String>> = listOf(
    "1d" to "1日以内",
    "7d" to "7日以内",
    "14d" to "2週間以内",
    "1m" to "1か月以内",
    "3m" to "3か月以内",
    "1y" to "1年間",
)

/**
 * Rebuilds the estimator input from `/api/usage/costs`. The server already resolved
 * legacy input inference and legacy image operations, so each answer carries the same
 * `pricing_calls` shape as a saved answer and the existing estimator can be reused.
 */
fun parseUsageCostSummary(payload: JSONObject, estimator: ModelCostEstimator): UsageCostSummary {
    val rows = payload.optJSONArray("messages")
    val messages = buildList {
        for (index in 0 until (rows?.length() ?: 0)) {
            val row = rows?.optJSONObject(index) ?: continue
            val calls = row.optJSONArray("pricing_calls") ?: JSONArray()
            add(
                ChatMessage(
                    id = row.opt("id").toString(),
                    role = "assistant",
                    content = "",
                    thought = JSONObject().put("pricing_calls", JSONArray(calls.toString())).toString(),
                    model = row.optString("model"),
                    createdAt = row.optString("created_at"),
                    batch = if (row.optBoolean("batch")) BatchInfo("JOB_STATE_SUCCEEDED") else null,
                    promptOptions = JSONObject().put("gemini_flex", row.optBoolean("flex")),
                )
            )
        }
    }
    return UsageCostSummary(
        period = payload.optString("period", "1m"),
        estimate = estimator.estimate(messages),
        answers = payload.optInt("answers", messages.size),
        unreadable = payload.optInt("unreadable"),
        truncated = payload.optBoolean("truncated"),
    )
}
