package com.minashin1120.aiplayground.data.local

import com.minashin1120.aiplayground.data.Diagnostics
import com.minashin1120.aiplayground.data.secure.EncryptedFileStore
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.time.Instant
import java.util.UUID

/**
 * Chats kept on the device: every chat of the no-account profile, and the outbox of serverless mode
 * (see "outbox" below). Answers the same JSON shapes as the server's thread endpoints (`/api/threads`,
 * `/api/threads/<id>`), so the chat screen parses them unchanged. Message ids are a per-profile counter
 * (the branch UI works on integer ids); each record also carries a UUID.
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

    /**
     * `GET /api/threads/<id>` with `limit` / `before_id` paging by message id, oldest first. Message ids
     * shown to the screen are local ids + [idOffset] (serverless mode keeps them in the pending range).
     */
    @Synchronized
    fun getThread(id: String, limit: Int?, beforeId: Int?, idOffset: Int = 0): JSONObject? {
        val thread = findThread(loadIndex(), id)?.takeIf { !it.optBoolean("deleted") } ?: return null
        val all = loadMessages(id).filter { beforeId == null || it.optInt("id") < beforeId }
        val ordered = all.sortedByDescending { it.optInt("id") }
        val page = if (limit != null) ordered.take(limit) else ordered
        val hasOlder = limit != null && ordered.size > limit
        val messages = page.sortedWith(compareBy<JSONObject> { it.optLong("created_ms") }.thenBy { it.optInt("id") })
        val rows = JSONArray()
        messages.forEach { row ->
            rows.put(publicMessage(if (idOffset == 0) row else JSONObject(row.toString()).put("id", row.getInt("id") + idOffset)
                .put("parent_id", if (row.isNull("parent_id")) JSONObject.NULL else row.optInt("parent_id") + idOffset)))
        }
        return JSONObject().put("messages", rows).put("has_older_messages", hasOlder)
            .put("oldest_loaded_id", messages.firstOrNull()?.optInt("id")?.plus(idOffset) ?: JSONObject.NULL)
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
        /** Parent on the server (serverless mode: a message sent under a server message). */
        val parentServerId: Int? = null,
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
            .put("parent_server_id", message.parentServerId ?: JSONObject.NULL)
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
            pruneOutboxRow(threadId)
            return threadId
        }
        return null
    }

    // --- outbox of serverless mode (a signed-in account) ---
    //
    // Chats of the account are read from the server; this store keeps only what the server does not
    // have yet: unsent messages of server chats (a thread row with `server_id` and `outbox`, parents on
    // the server in `parent_server_id`) and device chats (no `server_id`: created while the server was
    // out of reach, temporary chats, chats imported from the no-account profile). Once the server
    // accepts a message it is removed here. In a server chat, the screen sees unsent messages under
    // [PENDING_ID_BASE] + local id, above every server message id, so they never collide with server
    // ids and the newest unsent answer is the latest leaf.

    /** One chat with the records the server does not have yet (`/api/mobile/v1/sync/push` entry). */
    data class PendingThread(
        val localId: String,
        val entry: JSONObject,
        val messageIds: Map<String, Int>,
        val localFiles: Set<String>,
        /** UUIDs of messages whose parent is a server message. */
        val serverParented: Set<String> = emptySet(),
    )

    /** A chat that exists only on this device (not an outbox row of a server chat). */
    @Synchronized
    fun isDeviceThread(id: String): Boolean = findThread(loadIndex(), id)?.let { !it.optBoolean("deleted") && it.isNull("server_id") } == true

    /** Device chats for the top of the chat list (server `/api/threads` row shape), most recent first. */
    @Synchronized
    fun deviceThreads(query: String): JSONArray {
        val needle = query.trim()
        val rows = JSONArray()
        threads(loadIndex()).filter { !it.optBoolean("deleted") && it.isNull("server_id") }
            .filter { row ->
                needle.isEmpty() || row.optString("title").contains(needle, ignoreCase = true) ||
                    loadMessages(row.getString("id")).any { it.optString("content").contains(needle, ignoreCase = true) }
            }
            .sortedByDescending { it.optLong("updated_at") }
            .forEach { row ->
                rows.put(JSONObject().put("id", row.getString("id")).put("title", row.optString("title", "New Chat"))
                    .put("is_bookmarked", row.optBoolean("is_bookmarked")).put("last_model", row.opt("last_model") ?: JSONObject.NULL)
                    .put("is_temporary", row.optBoolean("is_temporary")))
            }
        return rows
    }

    /** Local id of the outbox row of server chat [serverId], or null when nothing is waiting for it. */
    @Synchronized
    fun outboxIdFor(serverId: String): String? = threads(loadIndex()).firstOrNull { it.opt("server_id")?.toString() == serverId }?.optString("id")

    /** The outbox row of server chat [serverId], created when missing; returns its local id. */
    @Synchronized
    fun ensureOutboxThread(serverId: String): String {
        outboxIdFor(serverId)?.let { return it }
        val index = loadIndex()
        val id = "l_" + UUID.randomUUID().toString().replace("-", "")
        val stamp = now()
        val row = JSONObject().put("id", id).put("uuid", UUID.randomUUID().toString()).put("title", "New Chat")
            .put("is_temporary", false).put("is_bookmarked", false).put("bookmarked_at", 0L)
            .put("custom_instruction", "").put("include_global_instruction", true).put("last_model", JSONObject.NULL)
            .put("last_gem_uuid", JSONObject.NULL).put("enable_prompt_caching", false)
            .put("created_at", stamp).put("updated_at", stamp).put("meta_changed_at", stamp)
            .put("server_id", serverId).put("outbox", true).put("dirty", false)
        putThreads(index, threads(index) + row)
        saveIndex(index)
        return id
    }

    /** Remembers a title to set on server chat [serverId] with the next sync (the first message of a new chat). */
    @Synchronized
    fun setPendingTitle(serverId: String, title: String) {
        val localId = ensureOutboxThread(serverId)
        val index = loadIndex()
        val rows = threads(index)
        val row = rows.firstOrNull { it.optString("id") == localId } ?: return
        if (row.textOf("title_pending").isNotBlank()) return
        row.put("title_pending", title)
        putThreads(index, rows)
        saveIndex(index)
    }

    /** Titles waiting for their server chats: (local id, server chat id, title). */
    @Synchronized
    fun pendingTitles(): List<Triple<String, String, String>> = threads(loadIndex()).mapNotNull { row ->
        val serverId = row.opt("server_id")?.takeIf { it != JSONObject.NULL }?.toString() ?: return@mapNotNull null
        val title = row.textOf("title_pending").takeIf { it.isNotBlank() } ?: return@mapNotNull null
        Triple(row.getString("id"), serverId, title)
    }

    @Synchronized
    fun clearPendingTitle(localId: String) {
        val index = loadIndex()
        val rows = threads(index)
        val row = rows.firstOrNull { it.optString("id") == localId } ?: return
        row.remove("title_pending")
        putThreads(index, rows)
        saveIndex(index)
        pruneOutboxRow(localId)
    }

    /**
     * Adds the unsent messages of server chat [serverId] to a `GET /api/threads/<id>` payload (from the
     * server or the offline cache), under pending ids so they keep their place in the branch tree.
     */
    @Synchronized
    fun overlayPending(serverId: String, payload: JSONObject): JSONObject {
        val localId = outboxIdFor(serverId) ?: return payload
        val rows = loadMessages(localId)
        if (rows.isEmpty()) return payload
        val messages = payload.optJSONArray("messages") ?: JSONArray().also { payload.put("messages", it) }
        rows.sortedWith(compareBy<JSONObject> { it.optLong("created_ms") }.thenBy { it.optInt("id") })
            .forEach { row -> messages.put(publicMessage(pendingView(row))) }
        return payload
    }

    /** Unsent messages of server chat [serverId] as raw rows with the screen's ids (pending ids, parents mapped). */
    @Synchronized
    fun pendingRows(serverId: String): List<JSONObject> {
        val localId = outboxIdFor(serverId) ?: return emptyList()
        return loadMessages(localId).map { pendingView(it) }
    }

    private fun pendingView(row: JSONObject): JSONObject = JSONObject(row.toString()).put("id", PENDING_ID_BASE + row.getInt("id"))
        .put("parent_id", when {
            !row.isNull("parent_server_id") -> row.optInt("parent_server_id")
            !row.isNull("parent_id") -> PENDING_ID_BASE + row.optInt("parent_id")
            else -> JSONObject.NULL
        })

    /** Messages still waiting for the server (temporary chats are never uploaded). */
    @Synchronized
    fun pendingCount(): Int = threads(loadIndex()).filter { !it.optBoolean("deleted") && !it.optBoolean("is_temporary") }
        .sumOf { thread -> loadMessages(thread.getString("id")).count { it.isNull("server_id") } }

    /** Server chat id of device chat [id] after it was uploaded, or [id] itself. */
    @Synchronized
    fun resolveAlias(id: String): String {
        val aliases = loadIndex().optJSONArray("aliases") ?: return id
        for (i in 0 until aliases.length()) {
            val row = aliases.optJSONObject(i) ?: continue
            if (row.optString("id") == id) return row.optString("server_id").ifBlank { id }
        }
        return id
    }

    /**
     * Repairs stores written by 1.38.0–1.40.1, whose pull saved a server chat without a device UUID
     * as UUID "null" (Android's `optString` turns JSON null into "null"), so every such chat was
     * merged into one. Gives those chats fresh UUIDs. Returns true when something was repaired.
     */
    @Synchronized
    fun repairSyncIdentities(): Boolean {
        val index = loadIndex()
        val rows = threads(index)
        val broken = rows.filter { it.optString("uuid").let { uuid -> uuid.isBlank() || uuid == "null" } }
        if (broken.isEmpty()) return false
        broken.forEach { it.put("uuid", UUID.randomUUID().toString()) }
        putThreads(index, rows)
        saveIndex(index)
        return true
    }

    /**
     * Turns a store of 1.38.0–1.40.x, which held a full copy of the account's chats, into the outbox:
     * synced messages and their files are removed, unsent messages under a synced parent point at the
     * parent's server id, and server chats with nothing left to send are dropped. Works offline; runs once.
     */
    @Synchronized
    fun migrateToOutbox(): Boolean {
        val index = loadIndex()
        if (index.optInt("outbox_version") >= OUTBOX_VERSION) return false
        val keep = ArrayList<JSONObject>()
        for (thread in threads(index)) {
            val id = thread.getString("id")
            if (thread.isNull("server_id")) { keep += thread; continue }
            val messages = loadMessages(id)
            val byId = messages.associateBy { it.optInt("id") }
            val (synced, unsent) = messages.partition { !it.isNull("server_id") }
            unsent.forEach { row ->
                if (row.isNull("parent_id")) return@forEach
                byId[row.optInt("parent_id")]?.takeIf { !it.isNull("server_id") }?.let { parent ->
                    row.put("parent_server_id", parent.optInt("server_id")).put("parent_id", JSONObject.NULL)
                }
            }
            val unsentRefs = unsent.flatMap(::fileRefs).toSet()
            synced.flatMap(::fileRefs).filter { it !in unsentRefs }.forEach(::deleteFile)
            if (unsent.isEmpty() && !thread.optBoolean("dirty")) { threadFile(id).delete(); continue }
            saveMessages(id, unsent)
            keep += thread
        }
        putThreads(index, keep)
        index.remove("sync_since")
        index.put("outbox_version", OUTBOX_VERSION)
        saveIndex(index)
        return true
    }

    /**
     * Chats with unsent settings or messages. Messages are listed oldest first so parents come before
     * children; a parent is referenced by its server id when known, otherwise by its UUID. Answers still
     * being generated (for less than 10 minutes) wait for the next sync. Outbox rows of server chats send
     * no settings (the server's settings are the ones shown).
     */
    @Synchronized
    fun pendingPush(now: Long = System.currentTimeMillis()): List<PendingThread> {
        val out = ArrayList<PendingThread>()
        for (thread in threads(loadIndex())) {
            if (thread.optBoolean("deleted") || thread.optBoolean("is_temporary")) continue
            val id = thread.getString("id")
            val outbox = thread.optBoolean("outbox")
            val rows = loadMessages(id).sortedWith(compareBy<JSONObject> { it.optLong("created_ms") }.thenBy { it.optInt("id") })
            val byId = rows.associateBy { it.optInt("id") }
            val unsent = rows.filter { row ->
                row.isNull("server_id") && !(row.optBoolean("generating") && now - row.optLong("created_ms") < 10 * 60 * 1000)
            }
            if (unsent.isEmpty() && (outbox || !thread.optBoolean("dirty"))) continue
            val messages = JSONArray()
            val ids = HashMap<String, Int>()
            val files = LinkedHashSet<String>()
            val serverParented = HashSet<String>()
            unsent.forEach { row ->
                val item = JSONObject().put("client_uuid", row.getString("uuid")).put("role", row.optString("role"))
                    .put("content", row.optString("content")).put("model", row.opt("model") ?: JSONObject.NULL)
                    .put("thought", row.opt("thought_data")?.takeIf { it != JSONObject.NULL }?.toString()
                        ?.let { raw -> runCatching { JSONObject(raw).optString("text") }.getOrDefault(raw) }.orEmpty())
                    .put("quote_text", row.opt("quote_text") ?: JSONObject.NULL)
                    .put("gem_uuid", row.opt("gem_uuid") ?: JSONObject.NULL).put("gem_name", row.opt("gem_name") ?: JSONObject.NULL)
                    .put("tokens_in", row.opt("tokens_in") ?: JSONObject.NULL).put("tokens_out", row.opt("tokens_out") ?: JSONObject.NULL)
                    .put("created_at_ms", row.optLong("created_ms"))
                if (!row.isNull("parent_server_id")) {
                    item.put("parent", JSONObject().put("id", row.optInt("parent_server_id")))
                    serverParented += row.getString("uuid")
                } else if (!row.isNull("parent_id")) {
                    val parent = byId[row.optInt("parent_id")]
                    if (parent != null) item.put("parent", if (!parent.isNull("server_id")) JSONObject().put("id", parent.optInt("server_id"))
                        else JSONObject().put("client_uuid", parent.getString("uuid")))
                }
                val refs = attachmentRefs(row)
                files += refs.filter { isLocalReference(it) }
                item.put("files", JSONArray(refs))
                messages.put(item)
                ids[row.getString("uuid")] = row.getInt("id")
            }
            val entry = JSONObject().put("messages", messages)
            if (!outbox) {
                entry.put("client_uuid", thread.getString("uuid")).put("title", thread.optString("title", "New Chat"))
                    .put("is_bookmarked", thread.optBoolean("is_bookmarked")).put("custom_instruction", thread.optString("custom_instruction"))
                    .put("include_global_instruction", thread.optBoolean("include_global_instruction", true))
                    .put("last_gem_uuid", thread.opt("last_gem_uuid") ?: JSONObject.NULL)
                    .put("meta_changed_at_ms", thread.optLong("meta_changed_at"))
            }
            thread.opt("server_id")?.takeIf { it != JSONObject.NULL }?.let { entry.put("id", it.toString()) }
            out += PendingThread(id, entry, ids, files, serverParented)
        }
        return out
    }

    /**
     * Records what the server accepted for [localId] and removes it here: accepted messages (and their
     * device files) are dropped, unsent children of an accepted message point at its server id, and a
     * device chat now on the server is remembered as an alias of its server id. The row goes away when
     * nothing is left to send.
     */
    @Synchronized
    fun markPushed(localId: String, serverThreadId: String, accepted: Map<String, Int>, allAccepted: Boolean) {
        val rows = loadMessages(localId)
        val serverOf = HashMap<Int, Int>()
        rows.forEach { row -> accepted[row.optString("uuid")]?.let { serverOf[row.getInt("id")] = it } }
        val kept = rows.filter { it.getInt("id") !in serverOf }
        kept.forEach { row ->
            if (!row.isNull("parent_id")) serverOf[row.optInt("parent_id")]?.let { row.put("parent_server_id", it).put("parent_id", JSONObject.NULL) }
        }
        val keptRefs = kept.flatMap(::fileRefs).toSet()
        rows.filter { it.getInt("id") in serverOf }.flatMap(::fileRefs).filter { it !in keptRefs }.forEach(::deleteFile)
        saveMessages(localId, kept)
        val index = loadIndex()
        val threadRows = threads(index)
        threadRows.firstOrNull { it.optString("id") == localId }?.let { thread ->
            if (thread.isNull("server_id")) addAlias(index, localId, serverThreadId)
            thread.put("server_id", serverThreadId)
            if (allAccepted) thread.put("dirty", false).put("outbox", true)
        }
        putThreads(index, threadRows)
        saveIndex(index)
        pruneOutboxRow(localId)
    }

    /** Drops unsent messages the server can no longer take (their parent was deleted there), with their descendants. */
    @Synchronized
    fun dropMessages(localId: String, uuids: Collection<String>) {
        if (uuids.isEmpty()) return
        val rows = loadMessages(localId)
        val dropped = rows.filter { it.optString("uuid") in uuids }.map { it.getInt("id") }.toMutableSet()
        var grew = true
        while (grew) {
            grew = rows.any { row -> row.getInt("id") !in dropped && !row.isNull("parent_id") && row.optInt("parent_id") in dropped && dropped.add(row.getInt("id")) }
        }
        val kept = rows.filter { it.getInt("id") !in dropped }
        val keptRefs = kept.flatMap(::fileRefs).toSet()
        rows.filter { it.getInt("id") in dropped }.flatMap(::fileRefs).filter { it !in keptRefs }.forEach(::deleteFile)
        saveMessages(localId, kept)
        pruneOutboxRow(localId)
    }

    /**
     * Server chat of [localId] is gone (deleted on the Web): its unsent messages go up again as a new chat.
     */
    @Synchronized
    fun detachFromServer(localId: String) {
        val rows = loadMessages(localId)
        rows.forEach { row -> if (!row.isNull("parent_server_id")) row.put("parent_server_id", JSONObject.NULL) }
        saveMessages(localId, rows)
        val index = loadIndex()
        val threadRows = threads(index)
        val thread = threadRows.firstOrNull { it.optString("id") == localId } ?: return
        thread.textOf("title_pending").takeIf { it.isNotBlank() }?.let { thread.put("title", it) }
        thread.remove("title_pending")
        thread.put("server_id", JSONObject.NULL).put("outbox", false).put("uuid", UUID.randomUUID().toString())
            .put("dirty", true).put("meta_changed_at", now())
        putThreads(index, threadRows)
        saveIndex(index)
    }

    /** Server chat [serverId] was deleted from this device: nothing more is sent for it. */
    @Synchronized
    fun dropOutbox(serverId: String) {
        val localId = outboxIdFor(serverId) ?: return
        loadMessages(localId).flatMap(::fileRefs).forEach(::deleteFile)
        val index = loadIndex()
        putThreads(index, threads(index).filterNot { it.optString("id") == localId })
        saveIndex(index)
        threadFile(localId).delete()
    }

    private fun pruneOutboxRow(localId: String) {
        val index = loadIndex()
        val rows = threads(index)
        val thread = rows.firstOrNull { it.optString("id") == localId } ?: return
        if (!thread.optBoolean("outbox") || thread.textOf("title_pending").isNotBlank() || loadMessages(localId).isNotEmpty()) return
        putThreads(index, rows.filterNot { it.optString("id") == localId })
        saveIndex(index)
        threadFile(localId).delete()
    }

    private fun addAlias(index: JSONObject, localId: String, serverId: String) {
        val current = index.optJSONArray("aliases") ?: JSONArray()
        val rows = (0 until current.length()).mapNotNull { current.optJSONObject(it) }.filterNot { it.optString("id") == localId }
        index.put("aliases", JSONArray((rows + JSONObject().put("id", localId).put("server_id", serverId)).takeLast(MAX_ALIASES)))
    }

    /** Deletions waiting for the server (left by 1.38.0–1.40.x): (thread public ids, message server ids). */
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

    /** Every attachment reference of a message (device `local/…` and server `uid/filename`). */
    private fun attachmentRefs(row: JSONObject): List<String> {
        val raw = row.opt("image_url")?.takeIf { it != JSONObject.NULL }?.toString().orEmpty()
        if (raw.isBlank()) return emptyList()
        return runCatching { JSONArray(raw).let { a -> (0 until a.length()).map { a.getString(it) } } }.getOrElse { listOf(raw) }
    }

    private fun fileRefs(row: JSONObject): List<String> = attachmentRefs(row).filter { isLocalReference(it) }

    // Attachment bodies (a video can be tens of MB) are encrypted and decrypted outside the store lock, so
    // reading one for an answer or a sync never holds up opening, sending, deleting or syncing chats. Each
    // body has its own file, written to `.part` and renamed into place, so a reader never sees half of one;
    // only the `files.enc` index is changed under the lock.

    /** Stores attachment bytes; returns the `local/<uuid>.<ext>` reference used in messages. */
    fun saveFile(name: String, mime: String, input: java.io.InputStream, size: Long): String {
        val ext = name.substringAfterLast('.', "").lowercase().filter { it.isLetterOrDigit() }.take(8).ifBlank { "bin" }
        val reference = "$LOCAL_PREFIX${UUID.randomUUID().toString().replace("-", "")}.$ext"
        val started = System.currentTimeMillis()
        crypto.writeStream(fileTarget(reference), input)
        Diagnostics.log("store.file_written", "mime" to mime, "bytes" to size, "ms" to System.currentTimeMillis() - started)
        val locked = System.currentTimeMillis()
        synchronized(this) {
            Diagnostics.log("store.lock_wait", "op" to "saveFile", "ms" to System.currentTimeMillis() - locked)
            val index = loadFileIndex()
            index.put(reference, JSONObject().put("name", name).put("mime", mime.ifBlank { "application/octet-stream" })
                .put("size", size).put("server_ref", JSONObject.NULL))
            crypto.writeJson(File(root, "files.enc"), index)
        }
        return reference
    }

    @Synchronized
    fun fileInfo(reference: String): JSONObject? = loadFileIndex().optJSONObject(reference)

    fun loadFile(reference: String, limit: Long): ByteArray? {
        val target = fileTarget(reference)
        if (!isLocalReference(reference) || !target.isFile) return null
        val started = System.currentTimeMillis()
        val result = runCatching { crypto.readBytes(target, limit) }
        val ms = System.currentTimeMillis() - started
        // Only large or slow reads (not every thumbnail).
        if (target.length() > 1024 * 1024 || ms > 1000 || result.isFailure) {
            Diagnostics.log("store.file_read", "stored_bytes" to target.length(), "ok" to result.isSuccess,
                "error" to result.exceptionOrNull()?.javaClass?.name, "ms" to ms)
        }
        return result.getOrNull()
    }

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
        private const val OUTBOX_VERSION = 1
        private const val MAX_ALIASES = 200

        /** Screen ids of unsent messages in a server chat start here (server message ids stay below). */
        const val PENDING_ID_BASE = 1_900_000_000

        fun isPendingId(id: Int): Boolean = id >= PENDING_ID_BASE

        /** Ids of chats created on the device (`l_` + 32 hex digits; server ids are 43-character tokens). */
        fun isDeviceThreadId(id: String): Boolean =
            id.length == 34 && id.startsWith("l_") && id.substring(2).all { it in '0'..'9' || it in 'a'..'f' }

        /** References of attachments kept on the device (never a server `uid/filename` reference). */
        fun isLocalReference(reference: String): Boolean =
            reference.startsWith(LOCAL_PREFIX) && reference.length in 10..80 && !reference.contains("..") &&
                reference.removePrefix(LOCAL_PREFIX).none { it == '/' || it == '?' || it == '#' || it == ':' }
    }
}

/** A string field of a server row, with JSON null read as [fallback] (Android's `optString` returns "null" for it). */
internal fun JSONObject.textOf(key: String, fallback: String = ""): String = if (isNull(key)) fallback else optString(key, fallback)
