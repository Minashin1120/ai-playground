package com.minashin1120.aiplayground.data

import com.minashin1120.aiplayground.data.secure.EncryptedFileStore
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.IOException
import javax.crypto.KeyGenerator

class PendingFeedbackStoreTest {
    @get:Rule val folder = TemporaryFolder()
    private val key = KeyGenerator.getInstance("AES").apply { init(256) }.generateKey()
    private val cipher = EncryptedFileStore("TESTFB".toByteArray()) { key }
    private fun store() = PendingFeedbackStore(folder.root.resolve("queue"), cipher)

    @Test fun restartPreservesPayloadAttachmentsAndAcknowledgedProgress() {
        val queue = store()
        val source = folder.newFile().apply { writeText("private-image") }
        val record = queue.enqueue("https://one.example/", 1,
            JSONObject().put("message", "private-feedback").put("client_logs", JSONObject().put("entries", "selected-logs")),
            listOf(PendingFeedbackStore.Attachment("image/png", "private-name") { source.inputStream() }))
        source.delete()
        record.data.put("feedback_id", "accepted-id").put("next_file", 1)
        queue.save(record)
        val restored = store().records("https://one.example/", 1).single()
        assertEquals("private-feedback", restored.data.getJSONObject("payload").getString("message"))
        assertEquals("accepted-id", restored.data.getString("feedback_id"))
        assertEquals(1, restored.data.getInt("next_file"))
        val output = folder.newFile()
        queue.materialize(restored, 0, output)
        assertEquals("private-image", output.readText())
        record.directory.listFiles()!!.forEach {
            val disk = String(it.readBytes(), Charsets.ISO_8859_1)
            listOf("private-feedback", "private-image", "private-name", "selected-logs").forEach { secret -> assertFalse(disk.contains(secret)) }
        }
        queue.remove(restored)
        assertTrue(store().records("https://one.example/", 1).isEmpty())
    }

    @Test fun doesNotSendToAnotherAccountOrServer() {
        val queue = store()
        queue.enqueue("https://one.example/", 1, JSONObject().put("message", "test"), emptyList())
        assertTrue(queue.records("https://one.example/", 2).isEmpty())
        assertTrue(queue.records("https://two.example/", 1).isEmpty())
        assertEquals(1, queue.records("https://one.example/", 1).size)
    }

    @Test fun failedAttachmentCopyNeverPublishesPartialFeedback() {
        val queue = store()
        try {
            queue.enqueue("https://one.example/", 1, JSONObject(), listOf(
                PendingFeedbackStore.Attachment("image/png", "first") { "first".byteInputStream() },
                PendingFeedbackStore.Attachment("image/png", "missing") { throw IOException("missing") }))
            fail("Expected attachment read failure")
        } catch (_: IOException) { }
        assertTrue(store().records("https://one.example/", 1).isEmpty())
        assertFalse(folder.root.resolve("queue").walkTopDown().any { it.isFile })
    }
}
