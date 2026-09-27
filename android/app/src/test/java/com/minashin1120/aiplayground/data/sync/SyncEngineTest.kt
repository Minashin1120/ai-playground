package com.minashin1120.aiplayground.data.sync

import com.minashin1120.aiplayground.data.PlaygroundApi
import com.minashin1120.aiplayground.data.local.LocalChatStore
import com.minashin1120.aiplayground.data.local.textOf
import com.minashin1120.aiplayground.data.secure.EncryptedFileStore
import kotlinx.coroutines.runBlocking
import mockwebserver3.MockResponse
import mockwebserver3.MockWebServer
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import javax.crypto.KeyGenerator

class SyncEngineTest {
    @get:Rule val folder = TemporaryFolder()
    private val key = KeyGenerator.getInstance("AES").apply { init(256) }.generateKey()
    private val store by lazy { LocalChatStore(folder.root.resolve("chats"), EncryptedFileStore(byteArrayOf(7)) { key }) }
    private fun json(body: String) = MockResponse.Builder().addHeader("Content-Type", "application/json").body(body).build()

    @Test fun pushesDeviceChatsThenPullsServerChats() = runBlocking {
        val thread = store.createThread(false, "端末のチャット").getString("id")
        val file = store.saveFile("a.png", "image/png", byteArrayOf(1, 2, 3).inputStream(), 3)
        val first = store.appendMessage(thread, LocalChatStore.NewMessage("user", "こんにちは", null, files = listOf(file)))
        store.appendMessage(thread, LocalChatStore.NewMessage("assistant", "回答", first, model = "gemini-3.6-flash"))
        store.appendMessage(thread, LocalChatStore.NewMessage("assistant", "", first, generating = true))
        val uuids = store.messages(thread).map { it.getString("uuid") }
        MockWebServer().use { server ->
            server.start()
            server.enqueue(json("""{"filename":"5/1_a.png"}"""))
            server.enqueue(json("""{"status":"ok","threads":[{"id":"srv1","messages":[{"client_uuid":"${uuids[0]}","id":11},{"client_uuid":"${uuids[1]}","id":12}],"rejected":[]}]}"""))
            server.enqueue(json("""{"threads":[{"id":"srv2","title":"Webのチャット","changed_at_ms":5,"updated_at_ms":5}],"tombstones":[],"has_more":false,"server_time_ms":1000000}"""))
            server.enqueue(json("""{"messages":[{"id":21,"role":"user","content":"Webから","parent_id":null,"created_at":"2026-09-27T00:00:00Z"},{"id":22,"role":"assistant","content":"返事","parent_id":21,"created_at":"2026-09-27T00:00:01Z"}],"has_older_messages":false}"""))
            val report = SyncEngine(store, PlaygroundApi(server.url("/"))) { "token" }.run()
            assertEquals(2, report.uploadedMessages)
            assertEquals(1, report.downloadedChats)
            assertEquals("/upload", server.takeRequest().target)
            assertEquals("/api/mobile/v1/sync/push", server.takeRequest().target)
            assertEquals("/api/mobile/v1/sync/changes", server.takeRequest().target)
            assertTrue(server.takeRequest().target.startsWith("/api/threads/srv2?limit=200"))
        }
        val rows = store.messages(thread)
        assertEquals(listOf(11, 12), rows.take(2).map { it.getInt("server_id") })
        assertTrue(rows[2].isNull("server_id"))
        assertEquals("5/1_a.png", store.fileInfo(file)!!.getString("server_ref"))
        assertEquals("srv1", store.threadRow(thread)!!.getString("server_id"))
        assertEquals(1000000L - SyncEngine.OVERLAP_MS, store.syncSince())
        val pulled = store.localIdForServer("srv2", null)!!
        val pulledRows = store.messages(pulled)
        assertEquals(listOf("Webから", "返事"), pulledRows.map { it.getString("content") })
        assertEquals(pulledRows[0].getInt("id"), pulledRows[1].getInt("parent_id"))
        assertEquals("Webのチャット", store.threadRow(pulled)!!.getString("title"))
    }

    @Test fun serverChatsWithoutDeviceUuidStaySeparate() = runBlocking {
        MockWebServer().use { server ->
            server.start()
            server.enqueue(json("""{"threads":[{"id":"a1","client_uuid":null,"title":"古いチャット","updated_at_ms":10},{"id":"b2","client_uuid":null,"title":"新しいチャット","updated_at_ms":20}],"tombstones":[],"has_more":false,"server_time_ms":1000000}"""))
            server.enqueue(json("""{"messages":[{"id":1,"role":"user","content":"古い","parent_id":null,"created_at":"2026-09-27T00:00:00Z"}],"has_older_messages":false}"""))
            server.enqueue(json("""{"messages":[{"id":2,"role":"user","content":null,"parent_id":null,"created_at":null}],"has_older_messages":false}"""))
            assertEquals(2, SyncEngine(store, PlaygroundApi(server.url("/"))) { "token" }.run().downloadedChats)
        }
        val titles = store.listThreads(1, "").getJSONArray("threads").let { rows -> (0 until rows.length()).map { rows.getJSONObject(it).getString("title") } }
        assertEquals(listOf("新しいチャット", "古いチャット"), titles)
        val newer = store.localIdForServer("b2", null)!!
        assertNotEquals(store.localIdForServer("a1", null), newer)
        assertNotEquals("null", store.threadRow(newer)!!.getString("uuid"))
        assertEquals("", store.messages(newer).single().getString("content"))
    }

    @Test fun storesMergedByOlderVersionsAreRepairedAndPulledAgain() {
        val merged = store.upsertServerThread(org.json.JSONObject().put("id", "b2").put("client_uuid", "null").put("title", "T"))
        store.setSyncSince(123)
        assertTrue(store.repairSyncIdentities())
        assertNull(store.syncSince())
        assertNotEquals("null", store.threadRow(merged)!!.getString("uuid"))
        assertEquals("b2", store.threadRow(merged)!!.getString("server_id"))
        assertFalse(store.repairSyncIdentities())
    }

    @Test fun jsonNullReadsAsTheFallback() {
        val row = org.json.JSONObject().put("a", org.json.JSONObject.NULL).put("b", "x")
        assertEquals("", row.textOf("a"))
        assertEquals("New Chat", row.textOf("a", "New Chat"))
        assertEquals("New Chat", row.textOf("missing", "New Chat"))
        assertEquals("x", row.textOf("b"))
    }

    @Test fun serverDeletionsRemoveSyncedMessagesAndKeepUnsentOnes() {
        val thread = store.upsertServerThread(org.json.JSONObject().put("id", "srv9").put("title", "T").put("changed_at_ms", 1))
        store.mergeServerMessages(thread, listOf(
            org.json.JSONObject().put("id", 1).put("role", "user").put("content", "a").put("created_at", "2026-01-01T00:00:00Z"),
            org.json.JSONObject().put("id", 2).put("role", "assistant").put("content", "b").put("parent_id", 1).put("created_at", "2026-01-01T00:00:01Z"),
        ))
        val answer = store.messages(thread).first { it.optInt("server_id") == 2 }.getInt("id")
        val unsent = store.appendMessage(thread, LocalChatStore.NewMessage("user", "端末で追加", answer))
        store.mergeServerMessages(thread, listOf(
            org.json.JSONObject().put("id", 1).put("role", "user").put("content", "a（編集）").put("created_at", "2026-01-01T00:00:00Z"),
        ))
        val rows = store.messages(thread)
        assertEquals(listOf("a（編集）", "端末で追加"), rows.map { it.getString("content") })
        assertEquals(rows[0].getInt("id"), rows.first { it.getInt("id") == unsent }.getInt("parent_id"))
        store.applyTombstone("srv9", null)
        assertNotNull(store.threadRow(thread))
        assertTrue(store.threadRow(thread)!!.isNull("server_id"))
    }

    @Test fun localProfileChatsImportOnceWithTheirFiles() {
        val other = LocalChatStore(folder.root.resolve("other"), EncryptedFileStore(byteArrayOf(7)) { key })
        val source = other.createThread(false, "ローカル").getString("id")
        val file = other.saveFile("n.txt", "text/plain", "x".byteInputStream(), 1)
        val parent = other.appendMessage(source, LocalChatStore.NewMessage("user", "質問", null, files = listOf(file)))
        other.appendMessage(source, LocalChatStore.NewMessage("assistant", "答え", parent))
        assertEquals(1, store.importFrom(other))
        assertEquals(0, store.importFrom(other))
        val imported = store.listThreads(1, "").getJSONArray("threads").getJSONObject(0).getString("id")
        val rows = store.messages(imported)
        assertEquals(2, rows.size)
        assertArrayEquals("x".toByteArray(), store.loadFile(file, 10))
        assertEquals(1, store.pendingPush().size)
    }
}
