package com.minashin1120.aiplayground.data.local

import com.minashin1120.aiplayground.data.secure.EncryptedFileStore
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.time.Instant
import java.util.UUID

/**
 * Chats kept on the device for the no-account profile and serverless mode. Answers the same JSON shapes
 * as the server's thread endpoints (`/api/threads`, `/api/threads/<id>`), so the chat screen parses them
 * unchanged. Message ids are a per-profile counter (the branch UI works on integer ids); each record also
 * carries a UUID and the server id it is synced to.
 *
 * Layout under [root]: `index.enc` (threads, counters, pending deletions), `thread-<digest>.enc`
 * (messages) and `files/<digest>.enc` with `files.enc` (attachment bytes).
 */
class LocalChatStore(private val root: File, private val crypto: EncryptedFileStore) {
    private val indexFile get() = File(root, "index.enc")

    // --- index ---

    private fun loadIndex(): JSONObject = crypto.readJson(indexFile)
        ?: JSONObject().put("next_message_id", 1).put("threads", JSONArray()).put("deleted_threads", JSONArray())
            .put("deleted_messages", JSONArray())

    private fun saveIndex(index: JSONObject) = crypto.writeJson(indexFile, index)

    private fun threads(index: JSONObject): MutableList<JSONObject> {
        val rows = index.optJSONArray("threads") ?: JSONArray()
        return (0 until rows.length()).mapNotNull { rows.optJSONObject(it) }.toMutableList()
    }

    private fun putThreads(index: JSONObject, rows: List<JSONObject>) { index.put("threads", JSONArray(rows)) }

    private fun threadFile(id: String) = File(root, "thread-${EncryptedFileStore.sha256Hex(id).take(40)}.enc")

    private fun loadMessages(id: String): MutableList<JSONObject> {
        val rows = crypto.readJson(threadFile(id))?.optJSONArray("messages") ?: JSONArray()
        return (0 until rows.length()).mapNotNull { rows.optJSONObject(it) }.toMutableList()
    }

    private fun saveMessages(id: String, rows: List<JSONObject>) =
        crypto.writeJson(threadFile(id), JSONObject().put("messages", JSONArray(rows)))

    private fun now(): Long = System.currentTimeMillis()

    private fun iso(millis: Long): String = Instant.ofEpochMilli(millis).toString()

    private fun findThread(index: JSONObject, id: String): JSONObject? = threads(index).firstOrNull { it.optString("id") == id }

    // --- threads ---

    /** `GET /api/threads`: bookmarked first, then most recently active; 20 per page like the server. */
    @Synchronized
    fun listThreads(page: Int, query: String): JSONObject {
        val needle = query.trim()
        val visible = threads(loadIndex()).filter { !it.optBoolean("deleted") && !it.optString("title").startsWith("[LIBRARY]") }
            .filter { row ->
                needle.isEmpty() || row.optString("title").contains(needle, ignoreCase = true) ||
                    loadMessages(row.getString("id")).any { it.optString("content").contains(needle, ignoreCase = true) }
            }
            .sortedWith(compareByDescending<JSONObject> { it.optBoolean("is_bookmarked") }
                .thenByDescending { it.optLong("bookmarked_at") }
                .thenByDescending { it.optLong("updated_at") })
        val start = ((page.coerceAtLeast(1) - 1) * PAGE_SIZE).coerceAtMost(visible.size)
        val slice = visible.subList(start, (start + PAGE_SIZE).coerceAtMost(visible.size))
        val rows = JSONArray()
        slice.forEach { row ->
            rows.put(JSONObject().put("id", row.getString("id")).put("title", row.optString("title", "New Chat"))
                .put("is_bookmarked", row.optBoolean("is_bookmarked")).put("last_model", row.opt("last_model") ?: JSONObject.NULL)
                .put("is_temporary", row.optBoolean("is_temporary")))
        }
        val hasNext = start + PAGE_SIZE < visible.size
        return JSONObject().put("threads", rows).put("has_next", hasNext).put("next_page", if (hasNext) page + 1 else JSONObject.NULL)
    }

    @Synchronized
    fun createThread(temporary: Boolean, title: String = "New Chat"): JSONObject {
        val index = loadIndex()
        val id = "l_" + UUID.randomUUID().toString().replace("-", "")
        val stamp = now()
        val row = JSONObject().put("id", id).put("uuid", UUID.randomUUID().toString()).put("title", title)
            .put("is_temporary", temporary).put("is_bookmarked", false).put("bookmarked_at", 0L)
            .put("custom_instruction", "").put("include_global_instruction", true).put("last_model", JSONObject.NULL)
            .put("last_gem_uuid", JSONObject.NULL).put("enable_prompt_caching", false)
            .put("created_at", stamp).put("updated_at", stamp).put("meta_changed_at", stamp)
            .put("server_id", JSONObject.NULL).put("dirty", true)
        putThreads(index, threads(index) + row)
        saveIndex(index)
        return JSONObject().put("id", id).put("title", title).put("is_temporary", temporary)
    }

    @Synchronized
    fun threadExists(id: String): Boolean = findThread(loadIndex(), id)?.optBoolean("deleted") == false

    /** `GET /api/threads/<id>` with `limit` / `before_id` paging by message id, oldest first. */
    @Synchronized
    fun getThread(id: String, limit: Int?, beforeId: Int?): JSONObject? {
        val thread = findThread(loadIndex(), id)?.takeIf { !it.optBoolean("deleted") } ?: return null
        val all = loadMessages(id).filter { beforeId == null || it.optInt("id") < beforeId }
        val ordered = all.sortedByDescending { it.optInt("id") }
        val page = if (limit != null) ordered.take(limit) else ordered
        val hasOlder = limit != null && ordered.size > limit
        val messages = page.sortedWith(compareBy<JSONObject> { it.optLong("created_ms") }.thenBy { it.optInt("id") })
        val rows = JSONArray()
        messages.forEach { rows.put(publicMessage(it)) }
        return JSONObject().put("messages", rows).put("has_older_messages", hasOlder)
            .put("oldest_loaded_id", messages.firstOrNull()?.optInt("id") ?: JSONObject.NULL)
            .put("loaded_count", messages.size).put("total_messages", JSONObject.NULL)
            .put("title", thread.optString("title", "New Chat"))
            .put("custom_instruction", thread.optString("custom_instruction"))
            .put("include_global_instruction", thread.optBoolean("include_global_instruction", true))
            .put("last_model", thread.opt("last_model") ?: JSONObject.NULL)
            .put("last_gem_uuid", thread.opt("last_gem_uuid") ?: JSONObject.NULL)
            .put("enable_prompt_caching", thread.optBoolean("enable_prompt_caching"))
            .put("is_temporary", thread.optBoolean("is_temporary"))
            .put("pending_job", JSONObject.NULL)
    }

    private fun publicMessage(row: JSONObject): JSONObject = JSONObject()
        .put("id", row.getInt("id")).put("role", row.optString("role")).put("content", row.optString("content"))
        .put("image_url", row.opt("image_url") ?: JSONObject.NULL).put("model", row.opt("model") ?: JSONObject.NULL)
        .put("thought_data", row.opt("thought_data") ?: JSONObject.NULL)
        .put("tokens", row.opt("tokens") ?: JSONObject.NULL).put("tokens_in", row.opt("tokens_in") ?: JSONObject.NULL)
        .put("tokens_out", row.opt("tokens_out") ?: JSONObject.NULL).put("tokens_content", JSONObject.NULL)
        .put("tokens_thought", row.opt("tokens_thought") ?: JSONObject.NULL)
        .put("is_encrypted", false).put("quote_text", row.opt("quote_text") ?: JSONObject.NULL)
        .put("parent_id", row.opt("parent_id") ?: JSONObject.NULL)
        .put("gem_uuid", row.opt("gem_uuid") ?: JSONObject.NULL).put("gem_name", row.opt("gem_name") ?: JSONObject.NULL)
        .put("created_at", iso(row.optLong("created_ms"))).put("batch_job", JSONObject.NULL)

    /** Applies [change] to the thread row; returns the updated row or null when it does not exist. */
    @Synchronized
    fun updateThread(id: String, touch: Boolean = false, change: (JSONObject) -> Unit): JSONObject? {
        val index = loadIndex()
        val rows = threads(index)
        val row = rows.firstOrNull { it.optString("id") == id && !it.optBoolean("deleted") } ?: return null
        change(row)
        val stamp = now()
        row.put("meta_changed_at", stamp).put("dirty", true)
        if (touch) row.put("updated_at", stamp)
        putThreads(index, rows)
        saveIndex(index)
        return row
    }

    /** Deletes a chat; one already on the server is remembered so the next sync deletes it there too. */
    @Synchronized
    fun deleteThread(id: String): Boolean {
        val index = loadIndex()
        val rows = threads(index)
        val row = rows.firstOrNull { it.optString("id") == id } ?: return false
        val serverId = row.opt("server_id")?.takeIf { it != JSONObject.NULL }?.toString()
        if (serverId != null) index.getJSONArray("deleted_threads").put(JSONObject().put("server_id", serverId).put("uuid", row.optString("uuid")))
        putThreads(index, rows.filterNot { it.optString("id") == id })
        saveIndex(index)
        loadMessages(id).forEach { message -> fileRefs(message).forEach(::deleteFile) }
        threadFile(id).delete()
        return true
    }

    // --- messages ---

    data class NewMessage(
        val role: String,
        val content: String,
        val parentId: Int?,
        val model: String = "",
        val thought: String = "",
        val files: List<String> = emptyList(),
        val quote: String = "",
        val gemUuid: String = "",
        val gemName: String = "",
        val tokensIn: Int? = null,
        val tokensOut: Int? = null,
        val tokensThought: Int? = null,
        val createdMs: Long? = null,
        val uuid: String? = null,
        val serverId: Int? = null,
        /** An answer still being generated: not uploaded until it is finished. */
        val generating: Boolean = false,
    )

    /** Appends a message and returns its local id; the thread moves to the top of the list. */
    @Synchronized
    fun appendMessage(threadId: String, message: NewMessage, synced: Boolean = false): Int {
        val index = loadIndex()
        val id = index.optInt("next_message_id", 1)
        index.put("next_message_id", id + 1)
        val rows = loadMessages(threadId)
        val stamp = message.createdMs ?: maxOf(now(), (rows.maxOfOrNull { it.optLong("created_ms") } ?: 0L) + 1)
        rows += JSONObject().put("id", id).put("uuid", message.uuid ?: UUID.randomUUID().toString())
            .put("role", message.role).put("content", message.content)
            .put("thought_data", if (message.thought.isBlank()) JSONObject.NULL else JSONObject().put("text", message.thought).toString())
            .put("model", message.model.ifBlank { null } ?: JSONObject.NULL)
            .put("image_url", if (message.files.isEmpty()) JSONObject.NULL else JSONArray(message.files).toString())
            .put("parent_id", message.parentId ?: JSONObject.NULL)
            .put("quote_text", message.quote.ifBlank { null } ?: JSONObject.NULL)
            .put("gem_uuid", message.gemUuid.ifBlank { null } ?: JSONObject.NULL)
            .put("gem_name", message.gemName.ifBlank { null } ?: JSONObject.NULL)
            .put("tokens_in", message.tokensIn ?: JSONObject.NULL).put("tokens_out", message.tokensOut ?: JSONObject.NULL)
            .put("tokens_thought", message.tokensThought ?: JSONObject.NULL)
            .put("tokens", listOfNotNull(message.tokensIn, message.tokensOut).takeIf { it.isNotEmpty() }?.sum() ?: JSONObject.NULL)
            .put("created_ms", stamp).put("server_id", message.serverId ?: JSONObject.NULL).put("synced", synced)
            .put("generating", message.generating)
        saveMessages(threadId, rows)
        val threadRows = threads(index)
        threadRows.firstOrNull { it.optString("id") == threadId }?.let { thread ->
            thread.put("updated_at", maxOf(thread.optLong("updated_at"), stamp))
            if (message.role == "assistant" && message.model.isNotBlank()) thread.put("last_model", message.model)
            if (!synced) thread.put("dirty", true)
        }
        putThreads(index, threadRows)
        saveIndex(index)
        return id
    }

    /** Replaces the text of a message being generated (saved every few seconds and at the end). */
    @Synchronized
    fun updateMessage(threadId: String, id: Int, content: String, thought: String, tokensIn: Int? = null, tokensOut: Int? = null,
                      files: List<String>? = null, done: Boolean = false) {
        val rows = loadMessages(threadId)
        val row = rows.firstOrNull { it.optInt("id") == id } ?: return
        row.put("content", content)
            .put("thought_data", if (thought.isBlank()) JSONObject.NULL else JSONObject().put("text", thought).toString())
            .put("synced", false)
        if (tokensIn != null) row.put("tokens_in", tokensIn)
        if (tokensOut != null) row.put("tokens_out", tokensOut)
        if (tokensIn != null || tokensOut != null) row.put("tokens", (tokensIn ?: 0) + (tokensOut ?: 0))
        if (files != null) row.put("image_url", if (files.isEmpty()) JSONObject.NULL else JSONArray(files).toString())
        if (done) row.put("generating", false)
        saveMessages(threadId, rows)
        markDirty(threadId)
    }

    private fun markDirty(threadId: String) {
        val index = loadIndex()
        val rows = threads(index)
        rows.firstOrNull { it.optString("id") == threadId }?.put("dirty", true) ?: return
        putThreads(index, rows)
        saveIndex(index)
    }

    /** Raw message rows (for building the request history and for sync). */
    @Synchronized
    fun messages(threadId: String): List<JSONObject> = loadMessages(threadId).map { JSONObject(it.toString()) }

    @Synchronized
    fun threadRow(id: String): JSONObject? = findThread(loadIndex(), id)?.let { JSONObject(it.toString()) }

    /**
     * Server `delete_message`: removes the message and every message created at or after it in the same chat
     * (all branches). Returns the thread id, or null when the message is unknown.
     */
    @Synchronized
    fun deleteMessage(id: Int): String? {
        val index = loadIndex()
        for (thread in threads(index)) {
            val threadId = thread.getString("id")
            val rows = loadMessages(threadId)
            val target = rows.firstOrNull { it.optInt("id") == id } ?: continue
            val cutoff = target.optLong("created_ms")
            val (removed, kept) = rows.partition { it.optLong("created_ms") >= cutoff }
            val deleted = index.getJSONArray("deleted_messages")
            removed.forEach { row ->
                row.opt("server_id")?.takeIf { it != JSONObject.NULL }?.let {
                    deleted.put(JSONObject().put("server_id", it).put("thread", threadId))
                }
                fileRefs(row).forEach(::deleteFile)
            }
            saveMessages(threadId, kept)
            thread.put("dirty", true)
            saveIndex(index)
            return threadId
        }
        return null
    }

    // --- sync with the server account (serverless mode) ---

    /** One chat with the records the server does not have yet (`/api/mobile/v1/sync/push` entry). */
    data class PendingThread(val localId: String, val entry: JSONObject, val messageIds: Map<String, Int>, val localFiles: Set<String>)

    @Synchronized
    fun syncSince(): Long? = loadIndex().optLong("sync_since", 0L).takeIf { it > 0 }

    @Synchronized
    fun setSyncSince(value: Long) { val index = loadIndex(); index.put("sync_since", value); saveIndex(index) }

    /**
     * Chats with unsent settings or messages. Messages are listed oldest first so parents come before
     * children; a parent is referenced by its server id when known, otherwise by its UUID. Answers still
     * being generated (for less than 10 minutes) wait for the next sync.
     */
    @Synchronized
    fun pendingPush(now: Long = System.currentTimeMillis()): List<PendingThread> {
        val out = ArrayList<PendingThread>()
        for (thread in threads(loadIndex())) {
            if (thread.optBoolean("deleted") || thread.optBoolean("is_temporary")) continue
            val id = thread.getString("id")
            val rows = loadMessages(id).sortedWith(compareBy<JSONObject> { it.optLong("created_ms") }.thenBy { it.optInt("id") })
            val byId = rows.associateBy { it.optInt("id") }
            val unsent = rows.filter { row ->
                row.isNull("server_id") && !(row.optBoolean("generating") && now - row.optLong("created_ms") < 10 * 60 * 1000)
            }
            if (unsent.isEmpty() && !thread.optBoolean("dirty")) continue
            val messages = JSONArray()
            val ids = HashMap<String, Int>()
            val files = LinkedHashSet<String>()
            unsent.forEach { row ->
                val item = JSONObject().put("client_uuid", row.getString("uuid")).put("role", row.optString("role"))
                    .put("content", row.optString("content")).put("model", row.opt("model") ?: JSONObject.NULL)
                    .put("thought", row.opt("thought_data")?.takeIf { it != JSONObject.NULL }?.toString()
                        ?.let { raw -> runCatching { JSONObject(raw).optString("text") }.getOrDefault(raw) }.orEmpty())
                    .put("quote_text", row.opt("quote_text") ?: JSONObject.NULL)
                    .put("gem_uuid", row.opt("gem_uuid") ?: JSONObject.NULL).put("gem_name", row.opt("gem_name") ?: JSONObject.NULL)
                    .put("tokens_in", row.opt("tokens_in") ?: JSONObject.NULL).put("tokens_out", row.opt("tokens_out") ?: JSONObject.NULL)
                    .put("created_at_ms", row.optLong("created_ms"))
                if (!row.isNull("parent_id")) {
                    val parent = byId[row.optInt("parent_id")]
                    if (parent != null) item.put("parent", if (!parent.isNull("server_id")) JSONObject().put("id", parent.optInt("server_id"))
                        else JSONObject().put("client_uuid", parent.getString("uuid")))
                }
                val refs = fileRefs(row)
                files += refs
                item.put("files", JSONArray(refs))
                messages.put(item)
                ids[row.getString("uuid")] = row.getInt("id")
            }
            val entry = JSONObject().put("client_uuid", thread.getString("uuid")).put("title", thread.optString("title", "New Chat"))
                .put("is_bookmarked", thread.optBoolean("is_bookmarked")).put("custom_instruction", thread.optString("custom_instruction"))
                .put("include_global_instruction", thread.optBoolean("include_global_instruction", true))
                .put("last_gem_uuid", thread.opt("last_gem_uuid") ?: JSONObject.NULL)
                .put("meta_changed_at_ms", thread.optLong("meta_changed_at")).put("messages", messages)
            thread.opt("server_id")?.takeIf { it != JSONObject.NULL }?.let { entry.put("id", it.toString()) }
            out += PendingThread(id, entry, ids, files)
        }
        return out
    }

    /** Records what the server accepted for [localId]: its public id and the server ids of the messages. */
    @Synchronized
    fun markPushed(localId: String, serverThreadId: String, accepted: Map<String, Int>, allAccepted: Boolean) {
        val rows = loadMessages(localId)
        rows.forEach { row -> accepted[row.optString("uuid")]?.let { row.put("server_id", it).put("synced", true) } }
        saveMessages(localId, rows)
        val index = loadIndex()
        val threadRows = threads(index)
        threadRows.firstOrNull { it.optString("id") == localId }?.let { thread ->
            thread.put("server_id", serverThreadId)
            if (allAccepted) thread.put("dirty", false)
        }
        putThreads(index, threadRows)
        saveIndex(index)
    }

    /** Deletions waiting for the server: (thread public ids, message server ids). */
    @Synchronized
    fun pendingDeletions(): Pair<List<String>, List<Int>> {
        val index = loadIndex()
        val threadsDeleted = index.optJSONArray("deleted_threads") ?: JSONArray()
        val messagesDeleted = index.optJSONArray("deleted_messages") ?: JSONArray()
        return (0 until threadsDeleted.length()).mapNotNull { threadsDeleted.optJSONObject(it)?.optString("server_id")?.takeIf { v -> v.isNotBlank() } } to
            (0 until messagesDeleted.length()).mapNotNull { messagesDeleted.optJSONObject(it)?.optInt("server_id")?.takeIf { v -> v > 0 } }
    }

    @Synchronized
    fun clearDeletions(threadIds: Collection<String>, messageIds: Collection<Int>) {
        val index = loadIndex()
        fun filter(name: String, keep: (JSONObject) -> Boolean) {
            val rows = index.optJSONArray(name) ?: JSONArray()
            index.put(name, JSONArray((0 until rows.length()).mapNotNull { rows.optJSONObject(it) }.filter(keep)))
        }
        filter("deleted_threads") { it.optString("server_id") !in threadIds }
        filter("deleted_messages") { it.optInt("server_id") !in messageIds }
        saveIndex(index)
    }

    /** The local chat synced with server chat [serverId] (or created on this device as [uuid]). */
    @Synchronized
    fun localIdForServer(serverId: String, uuid: String?): String? = threads(loadIndex()).firstOrNull { row ->
        row.opt("server_id")?.toString() == serverId || (!uuid.isNullOrBlank() && row.optString("uuid") == uuid)
    }?.optString("id")

    /**
     * Applies a server chat from `/api/mobile/v1/sync/changes`: creates it locally when new, and takes the
     * server's settings unless the device changed them later. Returns the local id.
     */
    @Synchronized
    fun upsertServerThread(row: JSONObject): String {
        val serverId = row.getString("id")
        val index = loadIndex()
        val rows = threads(index)
        val existing = rows.firstOrNull { it.opt("server_id")?.toString() == serverId }
            ?: row.optString("client_uuid").takeIf { it.isNotBlank() }?.let { uuid -> rows.firstOrNull { it.optString("uuid") == uuid } }
        val serverChanged = row.optLong("changed_at_ms")
        val stamp = row.optLong("updated_at_ms").takeIf { it > 0 } ?: now()
        val thread: JSONObject = existing ?: JSONObject().put("id", "l_" + UUID.randomUUID().toString().replace("-", ""))
            .put("uuid", row.optString("client_uuid").ifBlank { UUID.randomUUID().toString() })
            .put("is_temporary", false).put("bookmarked_at", 0L).put("enable_prompt_caching", false)
            .put("created_at", stamp).put("updated_at", stamp).put("meta_changed_at", 0L).put("dirty", false)
            .also { rows += it }
        thread.put("server_id", serverId)
        if (!thread.optBoolean("dirty") || thread.optLong("meta_changed_at") <= serverChanged) {
            thread.put("title", row.optString("title", "New Chat")).put("is_bookmarked", row.optBoolean("is_bookmarked"))
                .put("custom_instruction", row.optString("custom_instruction"))
                .put("include_global_instruction", row.optBoolean("include_global_instruction", true))
                .put("last_model", row.opt("last_model") ?: JSONObject.NULL).put("last_gem_uuid", row.opt("last_gem_uuid") ?: JSONObject.NULL)
            if (row.optBoolean("is_bookmarked") && thread.optLong("bookmarked_at") == 0L) thread.put("bookmarked_at", serverChanged)
        }
        row.optLong("updated_at_ms").takeIf { it > thread.optLong("updated_at") }?.let { thread.put("updated_at", it) }
        putThreads(index, rows)
        saveIndex(index)
        return thread.getString("id")
    }

    /**
     * Merges the server's full message list of a chat (`GET /api/threads/<id>` rows): adds messages the
     * device does not have, takes the server's text for synced ones, and removes synced messages the
     * server no longer has. Unsent device messages under a removed message move to its nearest surviving
     * ancestor so nothing typed on the device is lost.
     */
    @Synchronized
    fun mergeServerMessages(localId: String, server: List<JSONObject>) {
        val index = loadIndex()
        var nextId = index.optInt("next_message_id", 1)
        val rows = loadMessages(localId)
        val byServer = rows.filter { !it.isNull("server_id") }.associateBy { it.optInt("server_id") }
        val serverIds = server.map { it.optInt("id") }.toSet()
        val localByServerId = HashMap<Int, Int>()
        byServer.forEach { (serverId, row) -> localByServerId[serverId] = row.optInt("id") }
        val ordered = server.sortedWith(compareBy<JSONObject> { it.optString("created_at") }.thenBy { it.optInt("id") })
        for (message in ordered) {
            val serverId = message.optInt("id")
            val thought = message.opt("thought_data")?.takeIf { it != JSONObject.NULL }?.toString()
            val existing = byServer[serverId]
            if (existing != null) {
                existing.put("content", message.optString("content")).put("thought_data", thought ?: JSONObject.NULL).put("synced", true)
                continue
            }
            val localIdForMessage = nextId++
            localByServerId[serverId] = localIdForMessage
            val parentServer = if (message.isNull("parent_id")) null else message.optInt("parent_id")
            val created = runCatching { Instant.parse(message.optString("created_at")).toEpochMilli() }.getOrElse { now() }
            rows += JSONObject().put("id", localIdForMessage).put("uuid", UUID.randomUUID().toString())
                .put("role", message.optString("role")).put("content", message.optString("content"))
                .put("thought_data", thought ?: JSONObject.NULL).put("model", message.opt("model") ?: JSONObject.NULL)
                .put("image_url", message.opt("image_url") ?: JSONObject.NULL)
                .put("parent_id", parentServer?.let { localByServerId[it] } ?: JSONObject.NULL)
                .put("quote_text", message.opt("quote_text") ?: JSONObject.NULL)
                .put("gem_uuid", message.opt("gem_uuid") ?: JSONObject.NULL).put("gem_name", message.opt("gem_name") ?: JSONObject.NULL)
                .put("tokens_in", message.opt("tokens_in") ?: JSONObject.NULL).put("tokens_out", message.opt("tokens_out") ?: JSONObject.NULL)
                .put("tokens", message.opt("tokens") ?: JSONObject.NULL).put("tokens_thought", message.opt("tokens_thought") ?: JSONObject.NULL)
                .put("created_ms", created).put("server_id", serverId).put("synced", true).put("generating", false)
        }
        // Synced messages the server no longer has were deleted there.
        val removed = rows.filter { !it.isNull("server_id") && it.optInt("server_id") !in serverIds }.map { it.optInt("id") }.toSet()
        if (removed.isNotEmpty()) {
            val byLocal = rows.associateBy { it.optInt("id") }
            fun survivor(start: Int?): Int? {
                var cursor = start
                val seen = HashSet<Int>()
                while (cursor != null && cursor in removed && seen.add(cursor)) {
                    val row = byLocal[cursor] ?: return null
                    cursor = if (row.isNull("parent_id")) null else row.optInt("parent_id")
                }
                return cursor
            }
            rows.removeAll { it.optInt("id") in removed }
            rows.forEach { row ->
                if (!row.isNull("parent_id") && row.optInt("parent_id") in removed) row.put("parent_id", survivor(row.optInt("parent_id")) ?: JSONObject.NULL)
            }
        }
        saveMessages(localId, rows)
        index.put("next_message_id", nextId)
        saveIndex(index)
    }

    /** A chat deleted on the server: removed here, unless it holds unsent messages (then it is uploaded again as a new chat). */
    @Synchronized
    fun applyTombstone(serverId: String?, uuid: String?) {
        val index = loadIndex()
        val rows = threads(index)
        val thread = rows.firstOrNull { (serverId != null && it.opt("server_id")?.toString() == serverId) ||
            (!uuid.isNullOrBlank() && it.optString("uuid") == uuid) } ?: return
        val id = thread.getString("id")
        val messages = loadMessages(id)
        if (messages.any { it.isNull("server_id") }) {
            messages.forEach { it.put("server_id", JSONObject.NULL).put("synced", false) }
            saveMessages(id, messages)
            thread.put("server_id", JSONObject.NULL).put("uuid", UUID.randomUUID().toString()).put("dirty", true)
            putThreads(index, rows)
            saveIndex(index)
            return
        }
        putThreads(index, rows.filterNot { it.optString("id") == id })
        saveIndex(index)
        messages.forEach { message -> fileRefs(message).forEach(::deleteFile) }
        threadFile(id).delete()
    }

    /**
     * Copies the chats of [other] (the no-account profile) into this store, keeping their UUIDs so a
     * second import adds nothing. The copies are unsent and upload with the next sync.
     */
    @Synchronized
    fun importFrom(other: LocalChatStore): Int {
        var imported = 0
        val existingUuids = threads(loadIndex()).map { it.optString("uuid") }.toSet()
        for (source in other.threadRowsForImport()) {
            if (source.optString("uuid") in existingUuids) continue
            val created = createThread(false, source.optString("title", "New Chat"))
            val newId = created.getString("id")
            updateThread(newId) { row ->
                row.put("uuid", source.optString("uuid")).put("is_bookmarked", source.optBoolean("is_bookmarked"))
                    .put("custom_instruction", source.optString("custom_instruction"))
                    .put("include_global_instruction", source.optBoolean("include_global_instruction", true))
                    .put("last_model", source.opt("last_model") ?: JSONObject.NULL)
            }
            val mapping = HashMap<Int, Int>()
            other.messages(source.getString("id")).sortedWith(compareBy<JSONObject> { it.optLong("created_ms") }.thenBy { it.optInt("id") })
                .forEach { message ->
                    val files = fileRefs(message).mapNotNull { ref -> copyFileFrom(other, ref) }
                    val thought = message.opt("thought_data")?.takeIf { it != JSONObject.NULL }?.toString()
                        ?.let { raw -> runCatching { JSONObject(raw).optString("text") }.getOrDefault(raw) }.orEmpty()
                    val parent = if (message.isNull("parent_id")) null else mapping[message.optInt("parent_id")]
                    mapping[message.optInt("id")] = appendMessage(newId, NewMessage(message.optString("role"), message.optString("content"), parent,
                        model = message.optString("model").takeIf { it != "null" }.orEmpty(), thought = thought, files = files,
                        quote = message.optString("quote_text").takeIf { it != "null" }.orEmpty(),
                        tokensIn = message.opt("tokens_in") as? Int, tokensOut = message.opt("tokens_out") as? Int,
                        createdMs = message.optLong("created_ms"), uuid = message.optString("uuid").ifBlank { null }))
                }
            imported++
        }
        return imported
    }

    @Synchronized
    fun threadRowsForImport(): List<JSONObject> = threads(loadIndex()).filter { !it.optBoolean("deleted") && !it.optBoolean("is_temporary") }
        .map { JSONObject(it.toString()) }

    private fun copyFileFrom(other: LocalChatStore, reference: String): String? {
        val info = other.fileInfo(reference) ?: return null
        val bytes = other.loadFile(reference, 64L * 1024 * 1024) ?: return null
        crypto.writeBytes(fileTarget(reference), bytes)
        val index = loadFileIndex()
        index.put(reference, JSONObject(info.toString()).put("server_ref", JSONObject.NULL))
        crypto.writeJson(File(root, "files.enc"), index)
        return reference
    }

    // --- attachments ---

    private val filesDir get() = File(root, "files")
    private fun fileTarget(reference: String) = File(filesDir, "${EncryptedFileStore.sha256Hex(reference).take(40)}.enc")

    private fun loadFileIndex(): JSONObject = crypto.readJson(File(root, "files.enc")) ?: JSONObject()

    private fun fileRefs(row: JSONObject): List<String> {
        val raw = row.opt("image_url")?.takeIf { it != JSONObject.NULL }?.toString().orEmpty()
        if (raw.isBlank()) return emptyList()
        return runCatching { JSONArray(raw).let { a -> (0 until a.length()).map { a.getString(it) } } }.getOrElse { listOf(raw) }
            .filter { isLocalReference(it) }
    }

    /** Stores attachment bytes; returns the `local/<uuid>.<ext>` reference used in messages. */
    @Synchronized
    fun saveFile(name: String, mime: String, input: java.io.InputStream, size: Long): String {
        val ext = name.substringAfterLast('.', "").lowercase().filter { it.isLetterOrDigit() }.take(8).ifBlank { "bin" }
        val reference = "$LOCAL_PREFIX${UUID.randomUUID().toString().replace("-", "")}.$ext"
        crypto.writeStream(fileTarget(reference), input)
        val index = loadFileIndex()
        index.put(reference, JSONObject().put("name", name).put("mime", mime.ifBlank { "application/octet-stream" })
            .put("size", size).put("server_ref", JSONObject.NULL))
        crypto.writeJson(File(root, "files.enc"), index)
        return reference
    }

    @Synchronized
    fun fileInfo(reference: String): JSONObject? = loadFileIndex().optJSONObject(reference)

    @Synchronized
    fun loadFile(reference: String, limit: Long): ByteArray? {
        val target = fileTarget(reference)
        if (!isLocalReference(reference) || !target.isFile) return null
        return runCatching { crypto.readBytes(target, limit) }.getOrNull()
    }

    @Synchronized
    fun materializeFile(reference: String, destination: File): String? {
        val target = fileTarget(reference)
        if (!isLocalReference(reference) || !target.isFile) return null
        return runCatching {
            destination.parentFile?.mkdirs()
            crypto.decryptToFile(target, destination)
            fileInfo(reference)?.optString("mime") ?: "application/octet-stream"
        }.getOrElse { destination.delete(); null }
    }

    @Synchronized
    fun setServerReference(reference: String, serverRef: String) {
        val index = loadFileIndex()
        index.optJSONObject(reference)?.put("server_ref", serverRef) ?: return
        crypto.writeJson(File(root, "files.enc"), index)
    }

    @Synchronized
    private fun deleteFile(reference: String) {
        fileTarget(reference).delete()
        val index = loadFileIndex()
        if (index.has(reference)) { index.remove(reference); crypto.writeJson(File(root, "files.enc"), index) }
    }

    /** Bytes used by chats and attachments of this profile. */
    @Synchronized
    fun usageBytes(): Long = root.walkTopDown().filter { it.isFile }.sumOf { it.length() }

    /** Removes everything of this profile. */
    @Synchronized
    fun clearAll() { root.deleteRecursively() }

    companion object {
        const val PAGE_SIZE = 20
        const val LOCAL_PREFIX = "local/"

        /** References of attachments kept on the device (never a server `uid/filename` reference). */
        fun isLocalReference(reference: String): Boolean =
            reference.startsWith(LOCAL_PREFIX) && reference.length in 10..80 && !reference.contains("..") &&
                reference.removePrefix(LOCAL_PREFIX).none { it == '/' || it == '?' || it == '#' || it == ':' }
    }
}
