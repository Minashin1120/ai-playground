package com.minashin1120.aiplayground.data.sync

import com.minashin1120.aiplayground.data.ApiException
import com.minashin1120.aiplayground.data.PlaygroundApi
import com.minashin1120.aiplayground.data.local.LocalChatStore
import com.minashin1120.aiplayground.data.local.textOf
import com.minashin1120.aiplayground.data.secure.EncryptedFileStore
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.runBlocking
import mockwebserver3.MockResponse
import mockwebserver3.MockWebServer
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import org.json.JSONObject
import java.io.IOException
import java.net.SocketTimeoutException
import javax.crypto.KeyGenerator

class SyncEngineTest {
    @get:Rule val folder = TemporaryFolder()
    private val key = KeyGenerator.getInstance("AES").apply { init(256) }.generateKey()
    private val store by lazy { LocalChatStore(folder.root.resolve("chats"), EncryptedFileStore(byteArrayOf(7)) { key }) }
    private fun json(body: String) = MockResponse.Builder().addHeader("Content-Type", "application/json").body(body).build()

    @Test fun concurrentUploadsShareTheRecordedReferenceAndReportBytes() = runBlocking {
        val bytes = ByteArray(192 * 1024) { 42 }
        val file = store.saveFile("attachment.bin", "application/octet-stream", bytes.inputStream(), bytes.size.toLong())
        MockWebServer().use { server ->
            server.start()
            // A duplicate request must fail the assertion rather than wait for a response forever.
            repeat(2) { server.enqueue(json("""{"filename":"5/attachment.bin"}""")) }
            val api = PlaygroundApi(server.url("/"))
            val progress = java.util.Collections.synchronizedList(mutableListOf<Long>())
            val refs = withTimeout(10_000) {
                (1..2).map { async {
                    SyncEngine(store, api) { "token" }.uploadAttachment(file) { sent, total ->
                        assertEquals(bytes.size.toLong(), total)
                        progress.add(sent)
                    }
                } }.awaitAll()
            }
            assertEquals(listOf("5/attachment.bin", "5/attachment.bin"), refs)
            assertEquals(1, server.requestCount)
            assertEquals(0L, progress.first())
            assertEquals(bytes.size.toLong(), progress.last())
            assertTrue(progress.zipWithNext().all { (a, b) -> b >= a })
            assertTrue(progress.any { it > 0 && it < bytes.size })
        }
    }

    @Test fun failedUploadReleasesTheLockAndDoesNotRecordAReference() = runBlocking {
        val file = store.saveFile("a.txt", "text/plain", byteArrayOf(1).inputStream(), 1)
        MockWebServer().use { server ->
            server.start()
            server.enqueue(MockResponse.Builder().code(500).body("{}").build())
            server.enqueue(json("""{"filename":"5/a.txt"}"""))
            val engine = SyncEngine(store, PlaygroundApi(server.url("/"))) { "token" }
            try { engine.uploadAttachment(file); fail("Expected a failed upload") }
            catch (e: ApiException) { assertEquals(500, e.status) }
            assertTrue(store.fileInfo(file)!!.isNull("server_ref"))
            assertEquals("5/a.txt", withTimeout(10_000) { engine.uploadAttachment(file) })
            assertEquals("5/a.txt", engine.uploadAttachment(file))
            assertEquals(2, server.requestCount)
        }
    }

    @Test fun reusesMissingServerReferencesWithoutRejectingTheBranch() = runBlocking {
        val thread = store.createThread(false).getString("id")
        val file = store.saveFile("gone.pdf", "application/pdf", byteArrayOf(1).inputStream(), 1)
        store.setServerReference(file, "5/gone.pdf")
        store.appendMessage(thread, LocalChatStore.NewMessage("user", "question", null, files = listOf(file)))
        val uuid = store.messages(thread).single().getString("uuid")
        MockWebServer().use { server ->
            server.start()
            server.enqueue(json("""{"status":"ok","threads":[{"id":"srv-missing","messages":[{"client_uuid":"$uuid","id":11}],"rejected":[]}]}"""))
            val report = SyncEngine(store, PlaygroundApi(server.url("/"))) { "token" }.run()
            assertEquals(1, report.uploadedMessages)
            assertEquals(0, report.pending)
            val request = server.takeRequest()
            assertEquals("/api/mobile/v1/sync/push", request.target)
            val message = JSONObject(request.body!!.utf8()).getJSONArray("threads").getJSONObject(0)
                .getJSONArray("messages").getJSONObject(0)
            assertEquals("5/gone.pdf", message.getJSONArray("files").getString(0))
            assertEquals("5/gone.pdf", message.getJSONArray("missing_files").getString(0))
        }
    }

    @Test fun pushesDeviceChatsAndKeepsOnlyWhatTheServerLacks() = runBlocking {
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
            val report = SyncEngine(store, PlaygroundApi(server.url("/"))) { "token" }.run()
            assertEquals(2, report.uploadedMessages)
            assertEquals(1, report.pending)
            assertEquals("/upload", server.takeRequest().target)
            assertEquals("/api/mobile/v1/sync/push", server.takeRequest().target)
            // Nothing is downloaded: chats are read from the server itself.
            assertEquals(2, server.requestCount)
        }
        // Accepted messages remain until a server fetch observes them. The generating answer
        // stays unsent, and still points at the question when the accepted rows are pruned.
        val left = store.messages(thread).single { it.optBoolean("generating") }
        assertEquals(11, left.getInt("parent_server_id"))
        assertNotNull(store.loadFile(file, 10))
        val payload = JSONObject("""{"messages":[{"id":11},{"id":12}]}""")
        store.overlayPending("srv1", payload)
        assertEquals(left.getInt("id"), store.messages(thread).single().getInt("id"))
        assertEquals(3, payload.getJSONArray("messages").length())
        assertNull(store.loadFile(file, 10))
        assertEquals("srv1", store.resolveAlias(thread))
    }

    @Test fun unsentMessagesOfADeletedServerChatGoUpAsANewChat() = runBlocking {
        val outbox = store.ensureOutboxThread("gone")
        store.appendMessage(outbox, LocalChatStore.NewMessage("user", "q", null, parentServerId = 5))
        store.setPendingTitle("gone", "最初の質問")
        MockWebServer().use { server ->
            server.start()
            server.enqueue(json("""{"status":"ok","threads":[{"error":"thread_not_found","messages":[],"rejected":[]}]}"""))
            val report = SyncEngine(store, PlaygroundApi(server.url("/"))) { "token" }.run()
            assertEquals(1, report.pending)
            assertEquals(1, server.requestCount)
        }
        assertTrue(store.isDeviceThread(outbox))
        assertEquals("最初の質問", store.threadRow(outbox)!!.getString("title"))
        assertTrue(store.messages(outbox).single().isNull("parent_server_id"))
        assertTrue(store.pendingTitles().isEmpty())
    }

    @Test fun titleOfANewServerChatIsSetAfterItsMessages() = runBlocking {
        val outbox = store.ensureOutboxThread("srvT")
        store.appendMessage(outbox, LocalChatStore.NewMessage("user", "最初の質問", null))
        store.setPendingTitle("srvT", "最初の質問")
        val uuid = store.messages(outbox).single().getString("uuid")
        MockWebServer().use { server ->
            server.start()
            server.enqueue(json("""{"status":"ok","threads":[{"id":"srvT","messages":[{"client_uuid":"$uuid","id":30}],"rejected":[]}]}"""))
            server.enqueue(json("""{"status":"ok","title":"最初の質問"}"""))
            SyncEngine(store, PlaygroundApi(server.url("/"))) { "token" }.run()
            assertEquals("/api/mobile/v1/sync/push", server.takeRequest().target)
            val title = server.takeRequest()
            assertEquals("PUT", title.method)
            assertEquals("/api/threads/srvT/title", title.target)
        }
        assertNotNull(store.outboxIdFor("srvT"))
        assertEquals(0, store.pendingCount())
        assertTrue(store.pendingPush().isEmpty())
        store.overlayPending("srvT", JSONObject("""{"messages":[{"id":30}]}"""))
        assertNull(store.outboxIdFor("srvT"))
    }

    @Test fun messagesUnderADeletedServerMessageAreDropped() = runBlocking {
        val outbox = store.ensureOutboxThread("srvF")
        store.appendMessage(outbox, LocalChatStore.NewMessage("user", "q", null, parentServerId = 9))
        val uuid = store.messages(outbox).single().getString("uuid")
        MockWebServer().use { server ->
            server.start()
            server.enqueue(json("""{"status":"ok","threads":[{"id":"srvF","messages":[],"rejected":[{"client_uuid":"$uuid","reason":"parent_missing"}]}]}"""))
            val report = SyncEngine(store, PlaygroundApi(server.url("/"))) { "token" }.run()
            assertEquals(0, report.rejected)
            assertEquals(0, report.pending)
        }
        assertNull(store.outboxIdFor("srvF"))
    }

    @Test fun storesMergedByOlderVersionsAreRepaired() {
        val merged = store.createThread(false, "T").getString("id")
        store.updateThread(merged) { it.put("uuid", "null") }
        assertTrue(store.repairSyncIdentities())
        assertNotEquals("null", store.threadRow(merged)!!.getString("uuid"))
        assertFalse(store.repairSyncIdentities())
    }

    @Test fun jsonNullReadsAsTheFallback() {
        val row = org.json.JSONObject().put("a", org.json.JSONObject.NULL).put("b", "x")
        assertEquals("", row.textOf("a"))
        assertEquals("New Chat", row.textOf("a", "New Chat"))
        assertEquals("New Chat", row.textOf("missing", "New Chat"))
        assertEquals("x", row.textOf("b"))
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

    @Test fun unreachableServerIsRetriedLaterWithGrowingWaits() {
        assertTrue(isRetryableSyncFailure(SocketTimeoutException()))
        assertTrue(isRetryableSyncFailure(IOException("connect failed")))
        assertTrue(isRetryableSyncFailure(ApiException(502, JSONObject())))
        assertTrue(isRetryableSyncFailure(ApiException(503, JSONObject())))
        assertTrue(isRetryableSyncFailure(ApiException(429, JSONObject())))
        assertTrue(isRetryableSyncFailure(ApiException(409, JSONObject().put("code", "e2ee_migration_in_progress"))))
        assertFalse(isRetryableSyncFailure(ApiException(401, JSONObject())))
        assertFalse(isRetryableSyncFailure(ApiException(403, JSONObject().put("code", "turnstile_required"))))
        assertFalse(isRetryableSyncFailure(IllegalStateException()))
        assertEquals(30_000L, syncRetryDelayMillis(1))
        assertEquals(60_000L, syncRetryDelayMillis(2))
        assertEquals(480_000L, syncRetryDelayMillis(5))
        assertEquals(600_000L, syncRetryDelayMillis(6))
        assertEquals(600_000L, syncRetryDelayMillis(100))
    }
}
