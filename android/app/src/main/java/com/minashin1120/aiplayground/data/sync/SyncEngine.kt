package com.minashin1120.aiplayground.data.sync

import com.minashin1120.aiplayground.data.ApiException
import com.minashin1120.aiplayground.data.PlaygroundApi
import com.minashin1120.aiplayground.data.local.LocalChatStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject

/** Outcome of one sync run, shown in the settings "接続" card. */
data class SyncReport(
    val uploadedMessages: Int = 0,
    val downloadedChats: Int = 0,
    val skippedAttachments: Int = 0,
    val rejected: Int = 0,
)

/**
 * Two-way sync between the device store of serverless mode and the server account
 * (`/api/mobile/v1/sync/push` and `/api/mobile/v1/sync/changes`, server `routes_mobile_sync.py`).
 *
 * 1. Push: attachments first (each upload's server reference is recorded at once, so a retry does
 *    not upload it again), then chats and messages in batches of at most 100, then deletions.
 * 2. Pull: changed chats since the last sync; each one is read again in full and merged, and
 *    deleted chats are removed.
 */
class SyncEngine(
    private val store: LocalChatStore,
    private val api: PlaygroundApi,
    private val token: () -> String,
) {
    suspend fun run(): SyncReport = withContext(Dispatchers.IO) {
        val pushed = push()
        val downloaded = pull()
        pushed.copy(downloadedChats = downloaded)
    }

    private suspend fun uploadAttachment(reference: String): String? {
        store.fileInfo(reference)?.optString("server_ref")?.takeIf { it.isNotBlank() && it != "null" }?.let { return it }
        val info = store.fileInfo(reference) ?: return null
        val bytes = store.loadFile(reference, MAX_UPLOAD_BYTES) ?: return null
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
        store.setServerReference(reference, serverRef)
        return serverRef
    }

    private suspend fun push(): SyncReport {
        var uploaded = 0
        var skipped = 0
        var rejected = 0
        val pending = store.pendingPush()
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
                val reply = api.post("/api/mobile/v1/sync/push", JSONObject().put("threads", JSONArray().put(entry)), token())
                val result = reply.optJSONArray("threads")?.optJSONObject(0) ?: continue
                if (result.has("error")) { allAccepted = false; break }
                serverThreadId = result.optString("id").ifBlank { serverThreadId }
                val accepted = HashMap<String, Int>()
                result.optJSONArray("messages")?.let { rows ->
                    for (k in 0 until rows.length()) rows.optJSONObject(k)?.let { accepted[it.optString("client_uuid")] = it.optInt("id") }
                }
                val rejectedRows = result.optJSONArray("rejected")?.length() ?: 0
                rejected += rejectedRows
                if (rejectedRows > 0) allAccepted = false
                uploaded += accepted.size
                serverThreadId?.let { store.markPushed(thread.localId, it, accepted, allAccepted && batch === batches.last()) }
            }
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

    private suspend fun pull(): Int {
        val since = store.syncSince()
        var cursor: Int? = null
        var serverTime = 0L
        var downloaded = 0
        do {
            val query = buildString {
                append("/api/mobile/v1/sync/changes?")
                if (since != null) append("since=").append(since).append('&')
                cursor?.let { append("cursor=").append(it) }
            }.trimEnd('&', '?')
            val reply = api.get(query, token())
            serverTime = reply.optLong("server_time_ms", serverTime)
            val rows = reply.optJSONArray("threads") ?: JSONArray()
            for (i in 0 until rows.length()) {
                val row = rows.getJSONObject(i)
                val localId = store.upsertServerThread(row)
                store.mergeServerMessages(localId, fetchMessages(row.getString("id")))
                downloaded++
            }
            val tombstones = reply.optJSONArray("tombstones") ?: JSONArray()
            for (i in 0 until tombstones.length()) {
                val row = tombstones.getJSONObject(i)
                store.applyTombstone(row.optString("id").ifBlank { null }, row.optString("client_uuid").ifBlank { null })
            }
            cursor = if (reply.optBoolean("has_more")) reply.optInt("next_cursor").takeIf { it > 0 } else null
        } while (cursor != null)
        // Changes are stamped before their transaction commits, so the next run looks two minutes back.
        if (serverTime > 0) store.setSyncSince(serverTime - OVERLAP_MS)
        return downloaded
    }

    /** Every message of a server chat, paging with `before_id` like the offline full sync. */
    private suspend fun fetchMessages(serverId: String): List<JSONObject> {
        val out = ArrayList<JSONObject>()
        var before: String? = null
        while (true) {
            val suffix = before?.let { "&before_id=$it" }.orEmpty()
            val reply = api.get("/api/threads/$serverId?limit=200&include_meta=0$suffix", token())
            val rows = reply.optJSONArray("messages") ?: JSONArray()
            for (i in 0 until rows.length()) out += rows.getJSONObject(i)
            val oldest = reply.opt("oldest_loaded_id")?.takeIf { it != JSONObject.NULL }?.toString()
            if (!reply.optBoolean("has_older_messages") || oldest == null || oldest == before) break
            before = oldest
        }
        return out
    }

    companion object {
        const val MAX_MESSAGES_PER_PUSH = 100
        const val OVERLAP_MS = 2 * 60 * 1000L
        private const val CHUNK_BYTES = 8 * 1024 * 1024
        private const val MAX_UPLOAD_BYTES = 64L * 1024 * 1024
    }
}
