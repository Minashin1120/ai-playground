package com.minashin1120.aiplayground.data

import com.minashin1120.aiplayground.data.secure.EncryptedFileStore
import kotlinx.coroutines.runBlocking
import mockwebserver3.MockResponse
import mockwebserver3.MockWebServer
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.IOException
import javax.crypto.KeyGenerator

class PendingFeedbackSenderTest {
    @get:Rule val folder = TemporaryFolder()
    private val key = KeyGenerator.getInstance("AES").apply { init(256) }.generateKey()
    private fun store() = PendingFeedbackStore(folder.root.resolve("queue"), EncryptedFileStore("TESTFB".toByteArray()) { key })
    private fun response(body: String, code: Int = 200) = MockResponse.Builder().code(code)
        .addHeader("Content-Type", "application/json").body(body).build()

    @Test fun serverOutageKeepsFeedbackUntilSuccessfulRetry() = runBlocking {
        MockWebServer().use { server ->
            server.start()
            val origin = server.url("/")
            val record = store().enqueue(origin.toString(), 1, JSONObject().put("message", "test"), emptyList())
            val sender = PendingFeedbackSender(store(), folder.root, PlaygroundApi(origin), "token")
            server.enqueue(response("{}", 503))
            try { sender.send(record); fail("Expected server failure") } catch (_: IOException) { }
            assertEquals(1, store().records(origin.toString(), 1).size)
            server.enqueue(response("""{"public_id":"accepted"}"""))
            sender.send(store().records(origin.toString(), 1).single())
            assertTrue(store().records(origin.toString(), 1).isEmpty())
            assertEquals("/api/feedback", server.takeRequest().target)
            assertEquals("/api/feedback", server.takeRequest().target)
        }
    }

    @Test fun failedImageUploadResumesWithoutPostingFeedbackAgain() = runBlocking {
        MockWebServer().use { server ->
            server.start()
            val origin = server.url("/")
            val record = store().enqueue(origin.toString(), 1, JSONObject().put("message", "test"),
                listOf(PendingFeedbackStore.Attachment("image/png", "image-1") { "image".byteInputStream() }))
            server.enqueue(response("""{"public_id":"accepted"}"""))
            server.enqueue(response("{}", 503))
            try { PendingFeedbackSender(store(), folder.root, PlaygroundApi(origin), "token").send(record)
                fail("Expected upload failure") } catch (_: IOException) { }
            assertEquals("accepted", store().records(origin.toString(), 1).single().data.getString("feedback_id"))
            server.enqueue(response("""{"saved":1,"images":1}"""))
            PendingFeedbackSender(store(), folder.root, PlaygroundApi(origin), "token")
                .send(store().records(origin.toString(), 1).single())
            assertEquals("/api/feedback", server.takeRequest().target)
            repeat(2) { assertEquals("/api/feedback/accepted/images", server.takeRequest().target) }
            assertTrue(store().records(origin.toString(), 1).isEmpty())
            assertFalse(folder.root.listFiles()!!.any { it.name.startsWith("feedback-") })
        }
    }
}
