package com.minashin1120.aiplayground

import com.minashin1120.aiplayground.data.Attachment
import org.junit.Assert.*
import org.junit.Test

class AttachmentUploadStateTest {
    @Test fun readyAttachmentReplacesTemporaryRowWhileSyncIsStillRunning() {
        val earlier = Attachment("earlier.png", "local:earlier", "image/png")
        val added = Attachment("new.png", "local:new", "image/png")
        val before = ChatState(attachments = listOf(earlier), uploading = true,
            uploadName = "new.png", uploadSent = 200, uploadTotal = 200,
            uploadCompleted = 1, uploadCount = 3)

        val after = before.withUploadedAttachment(added)

        assertEquals(listOf(earlier, added), after.attachments)
        assertEquals("", after.uploadName)
        assertEquals(0L, after.uploadSent)
        assertEquals(0L, after.uploadTotal)
        assertTrue(after.uploading) // Sending remains blocked until sync and the queued files finish.
        assertEquals(1, after.uploadCompleted)
        assertEquals(3, after.uploadCount)
    }
}
