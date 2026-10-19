package com.minashin1120.aiplayground.data.local

import com.minashin1120.aiplayground.data.parseMessages
import com.minashin1120.aiplayground.data.parseThreadItem
import com.minashin1120.aiplayground.data.secure.EncryptedFileStore
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import javax.crypto.KeyGenerator

class LocalChatStoreTest {
    @get:Rule val folder = TemporaryFolder()
    private val key = KeyGenerator.getInstance("AES").apply { init(256) }.generateKey()
    private val crypto = EncryptedFileStore(byteArrayOf('T'.code.toByte(), 1)) { key }
    private val store by lazy { LocalChatStore(folder.root.resolve("chats"), crypto) }

    @Test fun branchesShareFilesAndSurviveSourceDeletion() {
        val source = store.createThread(false, "Source").getString("id")
        val file = store.saveFile("a.txt", "text/plain", "hello".byteInputStream(), 5)
        val first = store.appendMessage(source, LocalChatStore.NewMessage("user", "root", null, files = listOf(file), createdMs = 100))
        val left = store.appendMessage(source, LocalChatStore.NewMessage("assistant", "left", first, createdMs = 200))
        store.appendMessage(source, LocalChatStore.NewMessage("assistant", "other", first, createdMs = 300))
        val selected = store.appendMessage(source, LocalChatStore.NewMessage("user", "selected", left, createdMs = 400))
        store.appendMessage(source, LocalChatStore.NewMessage("assistant", "later", selected, createdMs = 500))
        val branch = store.branchThread(source, selected, "request-1234567890", true)
        val id = branch.getString("id")
        assertEquals(4, store.messages(id).size)
        assertEquals(id, store.branchThread(source, selected, "request-1234567890", true).getString("id"))
        assertTrue(store.getThread(id, null, null)!!.getJSONObject("branch_origin").getBoolean("available"))
        store.deleteThread(source)
        assertArrayEquals("hello".toByteArray(), store.loadFile(file, 100))
        assertFalse(store.getThread(id, null, null)!!.getJSONObject("branch_origin").getBoolean("available"))
        store.deleteThread(id)
        assertNull(store.loadFile(file, 100))
    }

    @Test fun temporaryBranchNeedsExplicitSaveConfirmation() {
        val source = store.createThread(true).getString("id")
        val message = store.appendMessage(source, LocalChatStore.NewMessage("user", "hi", null))
        assertTrue(runCatching { store.branchThread(source, message, "request-temporary", false) }.isFailure)
        val id = store.branchThread(source, message, "request-temporary", true).getString("id")
        assertFalse(store.getThread(id, null, null)!!.getBoolean("is_temporary"))
    }

    @Test fun threadsAndMessagesUseTheServerShapes() {
        val id = store.createThread(temporary = false).getString("id")
        assertTrue(id.startsWith("l_"))
        val first = store.appendMessage(id, LocalChatStore.NewMessage("user", "こんにちは", null, model = "gemini-3.6-flash"))
        val answer = store.appendMessage(id, LocalChatStore.NewMessage("assistant", "回答", first, model = "gemini-3.6-flash", thought = "考え中"))
        val list = store.listThreads(1, "")
        val row = list.getJSONArray("threads").getJSONObject(0)
        assertEquals(id, parseThreadItem(row).id)
        assertEquals("gemini-3.6-flash", row.getString("last_model"))
        val payload = store.getThread(id, limit = 50, beforeId = null)!!
        val messages = parseMessages(payload)
        assertEquals(listOf(first.toString(), answer.toString()), messages.map { it.id })
        assertEquals(first, messages[1].parentId)
        assertEquals("考え中", JSONObject(messages[1].thought).getString("text"))
        assertFalse(payload.getBoolean("has_older_messages"))
        assertEquals("考え中", JSONObject(store.messages(id)[1].getString("thought_data")).getString("text"))
    }

    @Test fun pagingSearchBookmarksAndDeletion() {
        val ids = (1..25).map { n -> store.createThread(false, "Chat $n").getString("id") }
        assertEquals(20, store.listThreads(1, "").getJSONArray("threads").length())
        assertTrue(store.listThreads(1, "").getBoolean("has_next"))
        assertEquals(5, store.listThreads(2, "").getJSONArray("threads").length())
        store.appendMessage(ids[3], LocalChatStore.NewMessage("user", "needle text", null))
        assertEquals(ids[3], store.listThreads(1, "needle").getJSONArray("threads").getJSONObject(0).getString("id"))
        store.updateThread(ids[0]) { it.put("is_bookmarked", true).put("bookmarked_at", 1L) }
        assertEquals(ids[0], store.listThreads(1, "").getJSONArray("threads").getJSONObject(0).getString("id"))
        assertTrue(store.deleteThread(ids[0]))
        assertNull(store.getThread(ids[0], null, null))
        assertFalse(store.threadExists(ids[0]))
    }

    @Test fun deletingAMessageRemovesEverythingAfterItLikeTheServer() {
        val id = store.createThread(false).getString("id")
        val a = store.appendMessage(id, LocalChatStore.NewMessage("user", "a", null, createdMs = 1000))
        val b = store.appendMessage(id, LocalChatStore.NewMessage("assistant", "b", a, createdMs = 2000))
        store.appendMessage(id, LocalChatStore.NewMessage("user", "c", b, createdMs = 3000))
        store.appendMessage(id, LocalChatStore.NewMessage("user", "branch", a, createdMs = 4000))
        assertEquals(id, store.deleteMessage(b))
        assertEquals(listOf(a), store.messages(id).map { it.getInt("id") })
        assertNull(store.deleteMessage(9999))
    }

    @Test fun attachmentsStayEncryptedAndAreRemovedWithTheChat() {
        val reference = store.saveFile("photo.PNG", "image/png", byteArrayOf(9, 8, 7).inputStream(), 3)
        assertTrue(LocalChatStore.isLocalReference(reference))
        assertTrue(reference.endsWith(".png"))
        assertArrayEquals(byteArrayOf(9, 8, 7), store.loadFile(reference, 10))
        assertEquals("image/png", store.fileInfo(reference)!!.getString("mime"))
        val id = store.createThread(false).getString("id")
        store.appendMessage(id, LocalChatStore.NewMessage("user", "見て", null, files = listOf(reference)))
        store.deleteThread(id)
        assertNull(store.loadFile(reference, 10))
        assertFalse(LocalChatStore.isLocalReference("12/1700000000_abc.png"))
        assertFalse(LocalChatStore.isLocalReference("local/../secret"))
        assertFalse(LocalChatStore.isLocalReference("local/a/b.png"))
    }

    @Test fun serverIdsOfDeletedChatsAreKeptForTheNextSync() {
        val id = store.createThread(false).getString("id")
        store.updateThread(id) { it.put("server_id", "srv123") }
        store.deleteThread(id)
        val index = crypto.readJson(folder.root.resolve("chats/index.enc"))!!
        assertEquals("srv123", index.getJSONArray("deleted_threads").getJSONObject(0).getString("server_id"))
    }

    @Test fun unsentMessagesOfAServerChatAreShownInThePendingRange() {
        val outbox = store.ensureOutboxThread("srvA")
        assertEquals(outbox, store.ensureOutboxThread("srvA"))
        assertFalse(store.isDeviceThread(outbox))
        assertEquals(0, store.deviceThreads("").length())
        val user = store.appendMessage(outbox, LocalChatStore.NewMessage("user", "質問", null, parentServerId = 42))
        store.appendMessage(outbox, LocalChatStore.NewMessage("assistant", "回答", user))
        val payload = JSONObject().put("messages", org.json.JSONArray().put(JSONObject().put("id", 42).put("role", "assistant").put("content", "前の回答")))
        val rows = store.overlayPending("srvA", payload).getJSONArray("messages")
        assertEquals(3, rows.length())
        val shownUser = rows.getJSONObject(1)
        assertEquals(LocalChatStore.PENDING_ID_BASE + user, shownUser.getInt("id"))
        assertEquals(42, shownUser.getInt("parent_id"))
        assertEquals(shownUser.getInt("id"), rows.getJSONObject(2).getInt("parent_id"))
        assertTrue(LocalChatStore.isPendingId(rows.getJSONObject(2).getInt("id")))
        // The lock icon shows how the account will store them, not "Plain" until the sync.
        assertTrue(rows.getJSONObject(2).isNull("is_encrypted"))
        val encrypted = store.overlayPending("srvA", JSONObject().put("messages", org.json.JSONArray()), encrypted = true).getJSONArray("messages")
        assertTrue(encrypted.getJSONObject(1).getBoolean("is_encrypted"))
        assertEquals(2, store.pendingCount())
        val pending = store.pendingPush().single()
        assertEquals("srvA", pending.entry.getString("id"))
        // The server's own settings stay: an outbox row sends none.
        assertFalse(pending.entry.has("title"))
        assertEquals(42, pending.entry.getJSONArray("messages").getJSONObject(0).getJSONObject("parent").getInt("id"))
        assertEquals(1, pending.serverParented.size)
    }

    @Test fun acceptedMessagesLeaveTheDeviceAndUploadedChatsBecomeAliases() {
        val device = store.createThread(false, "端末").getString("id")
        assertTrue(LocalChatStore.isDeviceThreadId(device))
        assertTrue(store.isDeviceThread(device))
        assertEquals(1, store.deviceThreads("").length())
        val file = store.saveFile("a.png", "image/png", byteArrayOf(1).inputStream(), 1)
        val user = store.appendMessage(device, LocalChatStore.NewMessage("user", "質問", null, files = listOf(file)))
        val answer = store.appendMessage(device, LocalChatStore.NewMessage("assistant", "回答", user))
        val uuids = store.messages(device).associate { it.getInt("id") to it.getString("uuid") }
        // Only the question is accepted: the answer now points at the question's server id.
        store.markPushed(device, "srvB", mapOf(uuids.getValue(user) to 7), allAccepted = false)
        assertEquals("srvB", store.resolveAlias(device))
        assertFalse(store.isDeviceThread(device))
        assertNotNull(store.loadFile(file, 10))
        assertEquals(7, store.messages(device).first { it.getInt("id") == answer }.getInt("parent_server_id"))
        store.markPushed(device, "srvB", mapOf(uuids.getValue(answer) to 8), allAccepted = true)
        assertNotNull(store.threadRow(device))
        assertEquals(0, store.pendingCount())
        val stale = JSONObject().put("messages", org.json.JSONArray())
        val visible = store.overlayPending("srvB", stale).getJSONArray("messages")
        assertEquals(listOf(7, 8), (0 until visible.length()).map { visible.getJSONObject(it).getInt("id") })
        assertEquals(7, visible.getJSONObject(1).getInt("parent_id"))
        assertNotNull(store.loadFile(file, 10))
        assertEquals(8, store.resolveMessageId("srvB", LocalChatStore.PENDING_ID_BASE + answer))
        val observed = JSONObject().put("messages", visible)
        store.overlayPending("srvB", JSONObject(observed.toString()), observedOnServer = false)
        assertNotNull(store.loadFile(file, 10))
        store.overlayPending("srvB", observed)
        assertEquals(2, observed.getJSONArray("messages").length())
        assertNull(store.threadRow(device))
        assertNull(store.loadFile(file, 10))
        // The selected branch and a draft can still carry the previous pending id after cleanup.
        assertEquals(8, store.resolveMessageId("srvB", LocalChatStore.PENDING_ID_BASE + answer))
        assertEquals("srvB", store.resolveAlias(device))
        assertFalse(LocalChatStore.isDeviceThreadId("srvB"))
    }

    @Test fun aChildAppendedAfterItsParentIsAcceptedStillUsesTheServerParent() {
        val outbox = store.ensureOutboxThread("srvRace")
        val question = store.appendMessage(outbox, LocalChatStore.NewMessage("user", "q", null, parentServerId = 41))
        val uuid = store.messages(outbox).single().getString("uuid")
        store.markPushed(outbox, "srvRace", mapOf(uuid to 42), allAccepted = true)
        val answer = store.appendMessage(outbox, LocalChatStore.NewMessage("assistant", "a", question))
        val row = store.pendingRows("srvRace").first { it.getInt("id") == LocalChatStore.PENDING_ID_BASE + answer }
        assertEquals(42, row.getInt("parent_id"))
        val pushed = store.pendingPush().single().entry.getJSONArray("messages").getJSONObject(0)
        assertEquals(42, pushed.getJSONObject("parent").getInt("id"))
    }

    @Test fun deletingAServerChatDoesNotReuploadAcceptedMessagesStillWaitingForObservation() {
        val device = store.createThread(false).getString("id")
        val question = store.appendMessage(device, LocalChatStore.NewMessage("user", "q", null))
        val uuid = store.messages(device).single().getString("uuid")
        store.markPushed(device, "srvGone", mapOf(uuid to 42), allAccepted = true)
        store.appendMessage(device, LocalChatStore.NewMessage("assistant", "a", question))
        store.detachFromServer(device)
        assertEquals(device, store.resolveAlias(device))
        assertEquals(listOf("a"), store.messages(device).map { it.getString("content") })
        val messages = store.pendingPush().single().entry.getJSONArray("messages")
        assertEquals(1, messages.length())
        assertFalse(messages.getJSONObject(0).has("parent"))
    }

    @Test fun deviceChatsOfServerlessModeUseThePendingRange() {
        val device = store.createThread(false).getString("id")
        val first = store.appendMessage(device, LocalChatStore.NewMessage("user", "a", null))
        store.appendMessage(device, LocalChatStore.NewMessage("assistant", "b", first))
        val payload = store.getThread(device, null, null, LocalChatStore.PENDING_ID_BASE)!!
        val rows = payload.getJSONArray("messages")
        assertEquals(LocalChatStore.PENDING_ID_BASE + first, rows.getJSONObject(0).getInt("id"))
        assertEquals(LocalChatStore.PENDING_ID_BASE + first, rows.getJSONObject(1).getInt("parent_id"))
        assertEquals(LocalChatStore.PENDING_ID_BASE + first, payload.getInt("oldest_loaded_id"))
    }

    @Test fun fullCopiesOfOlderVersionsShrinkToTheOutbox() {
        val copy = store.createThread(false, "Web").getString("id")
        val question = store.appendMessage(copy, LocalChatStore.NewMessage("user", "a", null, serverId = 11), synced = true)
        val answer = store.appendMessage(copy, LocalChatStore.NewMessage("assistant", "b", question, serverId = 12), synced = true)
        store.appendMessage(copy, LocalChatStore.NewMessage("user", "端末で追加", answer))
        val clean = store.createThread(false, "同期済み").getString("id")
        store.appendMessage(clean, LocalChatStore.NewMessage("user", "c", null, serverId = 13), synced = true)
        val device = store.createThread(false, "端末だけ").getString("id")
        val indexFile = folder.root.resolve("chats/index.enc")
        val index = crypto.readJson(indexFile)!!
        val threads = index.getJSONArray("threads")
        for (i in 0 until threads.length()) {
            val row = threads.getJSONObject(i)
            when (row.getString("id")) {
                copy -> row.put("server_id", "srvC").put("dirty", false)
                clean -> row.put("server_id", "srvD").put("dirty", false)
            }
        }
        index.put("sync_since", 5L)
        crypto.writeJson(indexFile, index)
        assertTrue(store.migrateToOutbox())
        assertFalse(store.migrateToOutbox())
        val left = store.messages(copy).single()
        assertEquals("端末で追加", left.getString("content"))
        assertEquals(12, left.getInt("parent_server_id"))
        assertNull(store.threadRow(clean))
        assertTrue(store.isDeviceThread(device))
        assertFalse(crypto.readJson(indexFile)!!.has("sync_since"))
    }

    @Test fun unsentMessagesUnderADeletedServerParentAreDropped() {
        val outbox = store.ensureOutboxThread("srvE")
        val user = store.appendMessage(outbox, LocalChatStore.NewMessage("user", "q", null, parentServerId = 3))
        store.appendMessage(outbox, LocalChatStore.NewMessage("assistant", "a", user))
        store.dropMessages(outbox, listOf(store.messages(outbox).first { it.getInt("id") == user }.getString("uuid")))
        assertNull(store.outboxIdFor("srvE"))
        assertEquals(0, store.pendingCount())
    }
}
