package com.minashin1120.aiplayground.data.local

import com.minashin1120.aiplayground.data.direct.DirectHttp
import com.minashin1120.aiplayground.data.direct.DirectRouter
import com.minashin1120.aiplayground.data.direct.ServerlessDefaults
import com.minashin1120.aiplayground.data.secure.EncryptedFileStore
import kotlinx.coroutines.runBlocking
import mockwebserver3.MockResponse
import mockwebserver3.MockWebServer
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import javax.crypto.KeyGenerator
import javax.net.ssl.SSLException

class LocalChatBackendSslTest {
    @get:Rule val folder = TemporaryFolder()

    @Test fun interruptedAnswerRetainsPartialContentAndThoughtUntilManualRegeneration() = runBlocking {
        fun crypto(id: Byte): EncryptedFileStore {
            val key = KeyGenerator.getInstance("AES").apply { init(256) }.generateKey()
            return EncryptedFileStore(byteArrayOf(id)) { key }
        }
        // Keep each fixture's encryption key stable across reads.
        val key = KeyGenerator.getInstance("AES").apply { init(256) }.generateKey()
        val encryption = EncryptedFileStore(byteArrayOf(1)) { key }
        val store = LocalChatStore(folder.root, encryption)
        val settings = LocalSettingsStore(folder.root, encryption, crypto(2))
        settings.savePreferences(JSONObject().put("openai_key", "fake-key"))
        MockWebServer().use { server ->
            server.start()
            val stream = listOf(
                """{"type":"response.reasoning_summary_text.delta","delta":"partial thought"}""",
                """{"type":"response.output_text.delta","delta":"partial answer"}""",
                """{"type":"response.output_text.delta","delta":"later"}""",
            ).joinToString("") { "data: $it\n\n" }
            repeat(2) { server.enqueue(MockResponse.Builder().addHeader("Content-Type", "text/event-stream").body(stream).build()) }
            val backend = LocalChatBackend(store, settings, ServerlessDefaults(JSONObject()),
                DirectRouter(DirectHttp(emptySet(), true), mapOf("openai" to server.url("/").toString().trimEnd('/'))),
                "test", null, { "chat" })
            val thread = store.createThread(false, "test").getString("id")
            val body = JSONObject().put("model", "gpt-6-luna").put("thread_id", thread).put("message", "question")
                .put("enable_file_creation", false).put("enable_mcp", false)
                .put("enable_thinking", true).put("reasoning_effort", "high").put("enable_prompt_caching", true)
                .put("api_key", "must-not-be-saved")
            val events = mutableListOf<JSONObject>()
            backend.stream("/chat_stream", body, "", {}, { event ->
                if (event.optString("type") == "content" && event.optString("content") == "later")
                    throw SSLException("fake sensitive TLS detail")
                events += event
            })
            val savedOptions = store.messages(thread).first().getJSONObject("prompt_options")
            assertTrue(savedOptions.getBoolean("enable_thinking"))
            assertTrue(savedOptions.getBoolean("enable_prompt_caching"))
            assertEquals("high", savedOptions.getString("reasoning_effort"))
            assertFalse(savedOptions.has("api_key"))
            assertFalse(savedOptions.has("message"))
            val answer = store.messages(thread).last()
            assertTrue(answer.getString("content").startsWith("partial answer"))
            assertTrue(answer.getString("content").contains("安全な通信が中断されました。回答を再生成してください。"))
            assertFalse(answer.getString("content").contains("fake sensitive"))
            assertTrue(answer.getString("thought_data").contains("partial thought"))
            assertFalse(answer.getBoolean("generating"))
            assertEquals("error", events.last().getString("type"))
            assertEquals(1, server.requestCount)
            backend.stream("/chat_stream", JSONObject(body.toString()).put("parent_id_explicit", true), "", {}, {})
            assertEquals(2, server.requestCount)
            assertEquals("partial answerlater", store.messages(thread).last().getString("content"))
        }
    }
}
