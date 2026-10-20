package com.minashin1120.aiplayground.data.local

import com.minashin1120.aiplayground.data.ApiException
import com.minashin1120.aiplayground.data.backend.ChatBackend
import com.minashin1120.aiplayground.data.direct.DirectHttp
import com.minashin1120.aiplayground.data.direct.DirectRouter
import com.minashin1120.aiplayground.data.direct.ServerlessDefaults
import com.minashin1120.aiplayground.data.secure.EncryptedFileStore
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import javax.crypto.KeyGenerator

class LocalChatBackendPythonTest {
    @get:Rule val folder = TemporaryFolder()

    private class RecordingServer : ChatBackend {
        val requests = mutableListOf<JSONObject>()
        override suspend fun get(path: String, token: String?): JSONObject = error("Unexpected GET")
        override suspend fun getArray(path: String, token: String?): JSONArray = error("Unexpected GET array")
        override suspend fun post(path: String, payload: JSONObject, token: String?): JSONObject = error("Unexpected POST")
        override suspend fun put(path: String, payload: JSONObject, token: String): JSONObject = error("Unexpected PUT")
        override suspend fun delete(path: String, token: String): JSONObject = error("Unexpected DELETE")
        override suspend fun stream(path: String, payload: JSONObject, token: String, onAccepted: () -> Unit, onEvent: (JSONObject) -> Unit) {
            assertEquals("/chat_stream", path)
            assertEquals("session", token)
            requests += payload
            onAccepted()
            onEvent(JSONObject().put("type", "python").put("content", JSONObject().put("code", "print(6 * 7)").put("output", "42")))
            onEvent(JSONObject().put("type", "done"))
        }
    }

    private fun fixture(server: RecordingServer?): Pair<LocalChatBackend, LocalChatStore> {
        fun crypto(id: Byte): EncryptedFileStore {
            val key = KeyGenerator.getInstance("AES").apply { init(256) }.generateKey()
            return EncryptedFileStore(byteArrayOf(id)) { key }
        }
        val root = folder.newFolder()
        val encryption = crypto(1)
        val store = LocalChatStore(root, encryption)
        val remote = server?.let { ServerHistory(it, { "session" }, { _, _ -> error("Unexpected file request") },
            { null }, { _, _ -> null }, { false }, {}) }
        return LocalChatBackend(store, LocalSettingsStore(root, encryption, crypto(2)), ServerlessDefaults(JSONObject()),
            DirectRouter(DirectHttp()), "この端末", server, { "chat" }, remote = remote) to store
    }

    private fun request(model: String = "deepseek-v4.1-flash", thread: String = "server-thread") = JSONObject()
        .put("model", model).put("thread_id", thread).put("message", "Pythonで何か計算して")
        .put("enable_python", true).put("enable_file_creation", false).put("enable_mcp", false)

    @Test fun pythonOnlySendUsesServerAndForwardsExecutionEvents() = runBlocking {
        val server = RecordingServer()
        val (backend, _) = fixture(server)
        for (model in listOf("deepseek-v4.1-flash", "deepseek-v4-pro", "deepseek-v4-flash-0731")) {
            var accepted = false
            val events = mutableListOf<JSONObject>()
            val body = request(model)
            backend.stream("/chat_stream", body, "session", { accepted = true }, { events += it })
            assertTrue(accepted)
            assertSame(body, server.requests.last())
            assertEquals(listOf("python", "done"), events.map { it.getString("type") })
            assertEquals("42", events.first().getJSONObject("content").getString("output"))
        }
        assertEquals(3, server.requests.size)
    }

    @Test fun newThreadCanUseServerForPython() = runBlocking {
        val server = RecordingServer()
        val (backend, _) = fixture(server)
        backend.stream("/chat_stream", request(thread = ""), "session", {}, {})
        assertEquals(1, server.requests.size)
    }

    private suspend fun assertPythonUnavailable(backend: LocalChatBackend, body: JSONObject) {
        var accepted = false
        val events = mutableListOf<JSONObject>()
        try {
            backend.stream("/chat_stream", body, "session", { accepted = true }, { events += it })
            fail("Python must not be silently sent to a provider without a runtime")
        } catch (error: ApiException) {
            assertEquals("serverless_python_unavailable", error.payload.getString("code"))
        }
        assertFalse(accepted)
        assertTrue(events.isEmpty())
    }

    @Test fun noAccountDoesNotPretendPythonCanExecute() = runBlocking {
        val (backend, _) = fixture(null)
        assertPythonUnavailable(backend, request())
    }

    @Test fun deviceOnlyChatDoesNotLoseHistoryBySendingToServer() = runBlocking {
        val server = RecordingServer()
        val (backend, store) = fixture(server)
        val thread = store.createThread(false).getString("id")
        assertPythonUnavailable(backend, request(thread = thread))
        assertTrue(server.requests.isEmpty())
    }

    @Test fun unsyncedMessagesArePreservedAndPythonSendWaitsForSync() = runBlocking {
        val server = RecordingServer()
        val (backend, store) = fixture(server)
        val outbox = store.ensureOutboxThread("server-thread")
        store.appendMessage(outbox, LocalChatStore.NewMessage("assistant", "Pending answer", null))
        assertPythonUnavailable(backend, request())
        assertEquals(1, store.pendingRows("server-thread").size)
        assertTrue(server.requests.isEmpty())
    }

    @Test fun nativeExecutionAndPythonOffKeepDirectRouting() {
        val (backend, _) = fixture(null)
        for (model in listOf("gemini-3.5-flash", "gpt-5.4", "o3")) {
            assertFalse(model, backend.requiresServerPython(request(model)))
        }
        assertTrue(backend.requiresServerPython(request()))
        assertFalse(backend.requiresServerPython(request().put("enable_python", false)))
    }

    private fun codingRequest(thread: String = "server-thread", candidates: Boolean = false) =
        request(model = "gpt-5.4", thread = thread).put("enable_python", false)
            .put("coding_mode", true).put("coding_target", JSONObject().put("code", "print(1)"))
            .apply { if (candidates) put("coding_candidates", JSONArray().put(JSONObject().put("code", "print(2)"))) }

    @Test fun codingOnlySendUsesServerWithoutDeviceApiKey() = runBlocking {
        val server = RecordingServer()
        val (backend, _) = fixture(server)
        for (thread in listOf("server-thread", "")) {
            for (candidates in listOf(false, true)) {
                val body = codingRequest(thread, candidates)
                var accepted = false
                val events = mutableListOf<JSONObject>()
                backend.stream("/chat_stream", body, "session", { accepted = true }, { events += it })
                assertTrue(accepted)
                assertSame(body, server.requests.last())
                assertEquals("done", events.last().getString("type"))
            }
        }
        assertEquals(4, server.requests.size)
    }

    private suspend fun assertCodingUnavailable(backend: LocalChatBackend, body: JSONObject) {
        var accepted = false
        val events = mutableListOf<JSONObject>()
        try {
            backend.stream("/chat_stream", body, "session", { accepted = true }, { events += it })
            fail("Coding Mode requires a server with the complete chat history")
        } catch (error: ApiException) {
            assertEquals("serverless_coding_unavailable", error.payload.getString("code"))
        }
        assertFalse(accepted)
        assertTrue(events.isEmpty())
    }

    @Test fun codingWithoutAccountRequiresServerBeforeCheckingDeviceApiKey() = runBlocking {
        val (backend, _) = fixture(null)
        assertCodingUnavailable(backend, codingRequest(candidates = true))
    }

    @Test fun codingPreservesDeviceOnlyChatAndPendingAnswers() = runBlocking {
        val server = RecordingServer()
        val (backend, store) = fixture(server)
        val localThread = store.createThread(false).getString("id")
        assertCodingUnavailable(backend, codingRequest(localThread))
        val outbox = store.ensureOutboxThread("server-thread")
        store.appendMessage(outbox, LocalChatStore.NewMessage("assistant", "Pending answer", null))
        assertCodingUnavailable(backend, codingRequest(candidates = true))
        assertEquals(1, store.pendingRows("server-thread").size)
        assertTrue(server.requests.isEmpty())
    }
}
