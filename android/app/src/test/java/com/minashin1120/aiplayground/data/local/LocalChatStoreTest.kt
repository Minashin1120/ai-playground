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
}
