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
