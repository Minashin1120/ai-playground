package com.minashin1120.aiplayground.data

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.io.IOException

/** Saves the server acknowledgement before uploading attachments, so retries resume at that stage. */
internal class PendingFeedbackSender(
    private val pendingFeedback: PendingFeedbackStore,
    private val cacheDir: File,
    private val client: PlaygroundApi,
    private val auth: String,
) {
    suspend fun send(record: PendingFeedbackStore.Record): String {
        ActivityLog.log("change", "tab" to "feedback", "state" to "start", "retry" to true)
        val data = record.data
        if (data.optString("feedback_id").isBlank()) {
            val reply = client.postLong("/api/feedback", data.getJSONObject("payload"), auth)
            val id = reply.optString("public_id").ifBlank { reply.optInt("feedback_id").takeIf { it > 0 }?.toString().orEmpty() }
            if (id.isBlank()) throw IOException("フィードバックの送信結果を確認できません")
            data.put("feedback_id", id).put("result", reply)
            withContext(Dispatchers.IO) { pendingFeedback.save(record) }
        }
        val files = data.getJSONArray("files")
        for (index in data.optInt("next_file") until files.length()) {
            val metadata = files.getJSONObject(index)
            val temporary = File(cacheDir, "feedback-" + java.util.UUID.randomUUID())
            try {
                withContext(Dispatchers.IO) { pendingFeedback.materialize(record, index, temporary) }
                val reference = metadata.optString("reference")
                if (reference.isBlank()) {
                    val bytes = withContext(Dispatchers.IO) { temporary.readBytes() }
                    client.uploadFeedbackImages(data.getString("feedback_id"),
                        listOf(FeedbackImage(metadata.getString("name"), metadata.getString("mime"), bytes)), auth)
                } else if (data.optJSONObject("result")?.optBoolean("chat_copy_saved") == true) {
                    val reply = client.uploadFeedbackChatFile(data.getString("feedback_id"), reference,
                        metadata.getString("name"), temporary, metadata.getString("mime"), auth)
                    if (!reply.optBoolean("saved")) throw IOException("添付ファイルを送信できません")
                }
                data.put("next_file", index + 1)
                withContext(Dispatchers.IO) { pendingFeedback.save(record) }
            } finally { temporary.delete() }
        }
        withContext(Dispatchers.IO) { pendingFeedback.remove(record) }
        ActivityLog.log("change", "tab" to "feedback", "state" to "done", "ok" to true)
        val payload = data.getJSONObject("payload")
        val reply = data.optJSONObject("result") ?: JSONObject()
        return feedbackSentText(payload.optJSONObject("client_logs")?.optJSONArray("entries")?.let { reply.optInt("logs_count", it.length()) },
            reply.optBoolean("logs_saved", true), payload.has("chat_copy"), reply.optBoolean("chat_copy_saved"),
            (0 until files.length()).count { files.getJSONObject(it).optString("reference").isBlank() }, true)
    }

}
