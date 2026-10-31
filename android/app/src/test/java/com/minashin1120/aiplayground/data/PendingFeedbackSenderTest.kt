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

    @Test fun missingChatCopyPausesUploadsAcrossRestartWithoutLosingTheAttachment() = runBlocking {
        MockWebServer().use { server ->
            server.start()
            val origin = server.url("/")
            val record = store().enqueue(origin.toString(), 1, JSONObject().put("message", "test"),
                listOf(PendingFeedbackStore.Attachment("image/png", "image", "local-ref") { "image".byteInputStream() }))
            server.enqueue(response("{\"public_id\":\"accepted\",\"chat_copy_saved\":true}"))
            server.enqueue(response("{\"error\":\"not_found\"}", 404))
            try { PendingFeedbackSender(store(), folder.root, PlaygroundApi(origin), "token").send(record)
                fail("Expected rejected attachment") } catch (_: ApiException) { }
            val restored = store().records(origin.toString(), 1).single()
            assertFalse(PendingFeedbackSender.canRetry(restored, Long.MAX_VALUE))
            assertEquals(0, restored.data.getInt("next_file"))
            val copy = folder.root.resolve("preserved")
            store().materialize(restored, 0, copy)
            assertEquals("image", copy.readText())
            assertEquals(2, server.requestCount)
            val next = store().enqueue(origin.toString(), 1, JSONObject().put("message", "next"), emptyList())
            assertTrue(PendingFeedbackSender.canRetry(next))
            server.enqueue(response("{\"public_id\":\"next\"}"))
            PendingFeedbackSender(store(), folder.root, PlaygroundApi(origin), "token").send(next)
            assertEquals(1, store().records(origin.toString(), 1).size)
        }
    }

    @Test fun temporaryFailuresPersistIncreasingRetryDelays() = runBlocking {
        MockWebServer().use { server ->
            server.start()
            val origin = server.url("/")
            var record = store().enqueue(origin.toString(), 1, JSONObject().put("message", "test"), emptyList())
            repeat(7) { attempt ->
                server.enqueue(response("{}", 503))
                val before = System.currentTimeMillis()
                try { PendingFeedbackSender(store(), folder.root, PlaygroundApi(origin), "token").send(record)
                    fail("Expected outage") } catch (_: ApiException) { }
                record = store().records(origin.toString(), 1).single()
                val due = record.data.getLong("retry_after")
                val expected = (30_000L * (1L shl attempt)).coerceAtMost(600_000L)
                assertTrue(due >= before + expected)
                assertFalse(PendingFeedbackSender.canRetry(record, due - 1))
                assertTrue(PendingFeedbackSender.canRetry(record, due))
            }
        }
    }

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
