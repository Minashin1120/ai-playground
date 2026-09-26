package com.minashin1120.aiplayground.data

import org.json.JSONObject

data class BatchJob(
    val id: String,
    val threadId: String,
    val threadTitle: String,
    val model: String,
    val provider: String,
    val state: String,
    val status: String,
    val error: String,
    val active: Boolean,
    val canCancel: Boolean,
    val createdAt: String = "",
    val threadExists: Boolean = true,
)

fun parseBatchJobs(payload: JSONObject): List<BatchJob> {
    val rows = payload.optJSONArray("jobs") ?: return emptyList()
    return (0 until rows.length()).map { index ->
        val row = rows.getJSONObject(index)
        BatchJob(
            id = row.nullableString("job_id"),
            threadId = row.nullableString("thread_id"),
            threadTitle = row.nullableString("thread_title").ifBlank { "無題のチャット" },
            model = row.nullableString("model"),
            provider = row.nullableString("provider"),
            state = row.nullableString("state"),
            status = row.nullableString("status_text"),
            error = row.nullableString("error"),
            active = row.optBoolean("is_active"),
            canCancel = row.optBoolean("can_cancel"),
            createdAt = row.nullableString("created_at"),
            threadExists = row.optBoolean("thread_exists", true),
        )
    }
}

fun batchStateLabel(job: BatchJob): String = job.status.ifBlank {
    when (job.state) {
        "JOB_STATE_SUCCEEDED" -> "完了"
        "JOB_STATE_FAILED" -> "失敗"
        "JOB_STATE_CANCELLED" -> "キャンセル済み"
        "JOB_STATE_EXPIRED" -> "期限切れ"
        "JOB_STATE_RUNNING" -> "実行中"
        else -> "待機中"
    }
}

/** Web `batchStateLabelShort`: the badge text. */
fun batchStateShort(state: String): String = when (state.uppercase()) {
    "JOB_STATE_QUEUED" -> "送信待ち"
    "JOB_STATE_VALIDATING" -> "検証中"
    "JOB_STATE_PENDING" -> "待機中"
    "JOB_STATE_RUNNING" -> "実行中"
    "JOB_STATE_FINALIZING" -> "結果取得中"
    "JOB_STATE_SUCCEEDED" -> "完了"
    "JOB_STATE_FAILED" -> "失敗"
    "JOB_STATE_CANCELLING" -> "停止中"
    "JOB_STATE_CANCELLED" -> "停止"
    "JOB_STATE_EXPIRED" -> "期限切れ"
    else -> "確認中"
}

/** Web `batchProviderLabel`. */
fun batchProviderLabel(provider: String): String = when (provider.lowercase()) {
    "gemini" -> "Gemini"
    "openai" -> "OpenAI"
    "xai" -> "xAI"
    else -> provider.ifBlank { "Batch" }
}

/** Web `batchFormatTime`: naive times are UTC; shown as `MM/dd HH:mm` in local time (ja-JP). */
fun batchFormatTime(value: String, zone: java.time.ZoneId = java.time.ZoneId.systemDefault()): String {
    if (value.isBlank()) return ""
    val hasZone = value.endsWith("Z", ignoreCase = true) || Regex("[+-]\\d\\d:?\\d\\d$").containsMatchIn(value)
    val raw = if (hasZone) value else value + "Z"
    return runCatching {
        java.time.OffsetDateTime.parse(raw).atZoneSameInstant(zone)
            .format(java.time.format.DateTimeFormatter.ofPattern("MM/dd HH:mm"))
    }.getOrDefault(value)
}
