package com.minashin1120.aiplayground.data.sync

import com.minashin1120.aiplayground.data.ApiException
import com.minashin1120.aiplayground.data.Diagnostics
import com.minashin1120.aiplayground.data.PlaygroundApi
import com.minashin1120.aiplayground.data.local.LocalChatStore
import com.minashin1120.aiplayground.data.local.textOf
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException

/** Outcome of one sync run, shown in the settings "接続" card. */
data class SyncReport(
    val uploadedMessages: Int = 0,
    val skippedAttachments: Int = 0,
    val rejected: Int = 0,
    /** Messages still waiting on this device after the run. */
    val pending: Int = 0,
)

/**
 * Whether a failed sync is tried again later by itself: the server or the network was out of reach
 * (connection errors, 5xx, 408, 429) or the account was switching its encryption. Other refusals wait
 * for the next answer, the app's return or "今すぐ同期".
 */
fun isRetryableSyncFailure(e: Throwable): Boolean = when (e) {
    is ApiException -> e.status >= 500 || e.status == 408 || e.status == 429 || e.code == "e2ee_migration_in_progress"
    is IOException -> true
    else -> false
}

/** Wait before automatic retry number [attempt] (from 1): 30 seconds, doubling up to 10 minutes. */
fun syncRetryDelayMillis(attempt: Int): Long =
    (SYNC_RETRY_FIRST_DELAY_MS shl (attempt - 1).coerceIn(0, 5)).coerceAtMost(SYNC_RETRY_MAX_DELAY_MS)

private const val SYNC_RETRY_FIRST_DELAY_MS = 30_000L
private const val SYNC_RETRY_MAX_DELAY_MS = 10L * 60 * 1000

/**
 * Sends the outbox of serverless mode to the server account (`/api/mobile/v1/sync/push`, server
 * `routes_mobile_sync.py`). Chats are read from the server itself, so nothing is downloaded here.
 *
 * Attachments go first (each upload's server reference is recorded at once, so a retry does not
 * upload it again), then messages in batches of at most 100; accepted records leave the device store.
 * Titles of new server chats and deletions left by older versions follow.
 */
class SyncEngine(
    private val store: LocalChatStore,
    private val api: PlaygroundApi,
    private val token: () -> String,
) {
    suspend fun run(): SyncReport = withContext(Dispatchers.IO) {
        Diagnostics.log("sync.engine_start")
        store.repairSyncIdentities()
        store.migrateToOutbox()
        Diagnostics.log("sync.prepared")
        push().copy(pending = store.pendingCount())
    }

    private suspend fun uploadAttachment(reference: String): String? {
        store.fileInfo(reference)?.optString("server_ref")?.takeIf { it.isNotBlank() && it != "null" }?.let { return it }
        val info = store.fileInfo(reference) ?: return null
        val loadStarted = System.currentTimeMillis()
        val job = currentCoroutineContext()[Job]
        val bytes = store.loadFile(reference, MAX_UPLOAD_BYTES) { job?.isActive == false }
        Diagnostics.log("sync.attachment_loaded", "mime" to info.optString("mime"), "bytes" to info.optLong("size"),
            "loaded" to (bytes != null), "ms" to System.currentTimeMillis() - loadStarted, "memory" to Diagnostics.memory())
        if (bytes == null) return null
        val uploadStarted = System.currentTimeMillis()
        val name = info.optString("name").ifBlank { reference.substringAfterLast('/') }
        val serverRef = try {
            if (bytes.size > CHUNK_BYTES) {
                val init = api.uploadInit(name, bytes.size.toLong(), token())
                val uploadId = init.getString("upload_id")
                val chunk = init.optLong("chunk_size", CHUNK_BYTES.toLong()).toInt().coerceAtLeast(1)
                val total = (bytes.size + chunk - 1) / chunk
                for (index in 0 until total) {
                    val start = index * chunk
                    api.uploadChunk(uploadId, index, total, bytes.copyOfRange(start, minOf(bytes.size, start + chunk)), token())
                }
                api.uploadComplete(uploadId, token()).getString("filename")
            } else {
                api.upload(name, bytes.toRequestBody(info.optString("mime").toMediaTypeOrNull()), token()).getString("filename")
            }
        } catch (e: ApiException) {
            // Storage quota or a rejected type: the text syncs without this attachment.
            if (e.status == 413 || e.status == 400 || e.status == 415) return null
            throw e
        }
        Diagnostics.log("sync.attachment_uploaded", "bytes" to bytes.size, "ms" to System.currentTimeMillis() - uploadStarted)
        store.setServerReference(reference, serverRef)
        return serverRef
    }

    private suspend fun push(): SyncReport {
        var uploaded = 0
        var skipped = 0
        var rejected = 0
        val pending = store.pendingPush()
        Diagnostics.log("sync.pending", "threads" to pending.size, "messages" to pending.sumOf { it.messageIds.size },
            "files" to pending.sumOf { it.localFiles.size })
        for (thread in pending) {
            // Local attachment references become server references before the chat is sent.
            val serverRefs = HashMap<String, String?>()
            thread.localFiles.forEach { ref -> serverRefs[ref] = uploadAttachment(ref) }
            val messages = thread.entry.optJSONArray("messages") ?: JSONArray()
            for (i in 0 until messages.length()) {
                val message = messages.getJSONObject(i)
                val files = message.optJSONArray("files") ?: JSONArray()
                val mapped = JSONArray()
                for (j in 0 until files.length()) {
                    val ref = files.getString(j)
                    val serverRef = if (LocalChatStore.isLocalReference(ref)) serverRefs[ref] else ref
                    if (serverRef != null) mapped.put(serverRef) else skipped++
                }
                message.put("files", mapped)
            }
            // At most 100 messages per request; parents are always sent before their children.
            val all = (0 until messages.length()).map { messages.getJSONObject(it) }
            val batches: List<List<JSONObject>> = if (all.isEmpty()) listOf(emptyList()) else all.chunked(MAX_MESSAGES_PER_PUSH)
            var serverThreadId = thread.entry.optString("id").ifBlank { null }
            var allAccepted = true
            for (batch in batches) {
                val entry = JSONObject(thread.entry.toString()).put("messages", JSONArray(batch))
                serverThreadId?.let { entry.put("id", it) }
                val pushStarted = System.currentTimeMillis()
                val reply = api.post("/api/mobile/v1/sync/push", JSONObject().put("threads", JSONArray().put(entry)), token())
                Diagnostics.log("sync.pushed", "messages" to batch.size, "ms" to System.currentTimeMillis() - pushStarted)
                val result = reply.optJSONArray("threads")?.optJSONObject(0) ?: continue
                if (result.has("error")) {
                    when (result.textOf("error")) {
                        // The server chat was deleted on the Web: the unsent messages go up again as a new chat.
                        "thread_not_found" -> store.detachFromServer(thread.localId)
                        // A temporary chat on the server does not keep messages sent this way.
                        "temporary_chat" -> serverThreadId?.let(store::dropOutbox)
                    }
                    allAccepted = false
                    break
                }
                serverThreadId = result.optString("id").ifBlank { serverThreadId }
                val accepted = HashMap<String, Int>()
                result.optJSONArray("messages")?.let { rows ->
                    for (k in 0 until rows.length()) rows.optJSONObject(k)?.let { accepted[it.textOf("client_uuid")] = it.optInt("id") }
                }
                val rejectedRows = result.optJSONArray("rejected")?.let { rows -> (0 until rows.length()).mapNotNull { rows.optJSONObject(it) } }.orEmpty()
                // A server parent that no longer exists was deleted there: those messages cannot be sent any more.
                val orphans = rejectedRows.filter { it.textOf("reason") == "parent_missing" }.map { it.textOf("client_uuid") }
                    .filter { uuid -> thread.serverParented.contains(uuid) }
                rejected += rejectedRows.size - orphans.size
                if (rejectedRows.size > orphans.size) allAccepted = false
                uploaded += accepted.size
                serverThreadId?.let { store.markPushed(thread.localId, it, accepted, allAccepted && batch === batches.last()) }
                store.dropMessages(thread.localId, orphans)
            }
        }
        for ((localId, serverId, title) in store.pendingTitles()) {
            try {
                api.put("/api/threads/$serverId/title", JSONObject().put("title", title), token())
            } catch (e: ApiException) {
                if (e.status != 403 && e.status != 404) throw e
            }
            store.clearPendingTitle(localId)
        }
        val (deletedThreads, deletedMessages) = store.pendingDeletions()
        if (deletedThreads.isNotEmpty() || deletedMessages.isNotEmpty()) {
            val reply = api.post("/api/mobile/v1/sync/push", JSONObject().put("threads", JSONArray())
                .put("deleted_threads", JSONArray(deletedThreads)).put("deleted_messages", JSONArray(deletedMessages)), token())
            val doneThreads = reply.optJSONArray("deleted_threads")?.let { rows -> (0 until rows.length()).map { rows.optString(it) } }.orEmpty()
            val doneMessages = reply.optJSONArray("deleted_messages")?.let { rows -> (0 until rows.length()).map { rows.optInt(it) } }.orEmpty()
            store.clearDeletions(doneThreads, doneMessages)
        }
        return SyncReport(uploadedMessages = uploaded, skippedAttachments = skipped, rejected = rejected)
    }

    companion object {
        const val MAX_MESSAGES_PER_PUSH = 100
        private const val CHUNK_BYTES = 8 * 1024 * 1024
        private const val MAX_UPLOAD_BYTES = 64L * 1024 * 1024
    }
}
