package com.minashin1120.aiplayground.ui

import com.minashin1120.aiplayground.data.AttachmentKind
import com.minashin1120.aiplayground.data.attachmentKind
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class FileViewerTest {
    @Test fun titlesPreferDisplayNameThenPath() {
        assertEquals("note.txt", fileViewerTitle("1/abc.txt", "note.txt"))
        assertEquals("photo.png", fileViewerTitle("/files/9/photo.png", "  "))
        assertEquals("ファイル", fileViewerTitle("", ""))
    }

    @Test fun previewKindsMatchWebViewer() {
        assertEquals(AttachmentKind.IMAGE, attachmentKind("shot.WEBP"))
        assertEquals(AttachmentKind.TEXT, attachmentKind("readme.md"))
        assertEquals(AttachmentKind.PDF, attachmentKind("doc.pdf"))
        assertEquals(AttachmentKind.AUDIO, attachmentKind("voice.m4a"))
        assertEquals(AttachmentKind.VIDEO, attachmentKind("clip.webm"))
        assertEquals(AttachmentKind.FILE, attachmentKind("pack.docx"))
        assertEquals(AttachmentKind.FILE, attachmentKind("archive.zip"))
        assertEquals(AttachmentKind.IMAGE, fileViewerKind(FileViewRequest("1/abc", "写真", "png", "image/")))
        assertEquals(AttachmentKind.PDF, fileViewerKind(FileViewRequest("1/report", "報告書", "pdf")))
    }

    @Test fun textPreviewAcceptsUtf8AndRejectsBinary() {
        assertEquals("hello", decodePreviewText("hello".toByteArray()))
        val bom = byteArrayOf(0xEF.toByte(), 0xBB.toByte(), 0xBF.toByte()) + "あ".toByteArray(Charsets.UTF_8)
        assertEquals("あ", decodePreviewText(bom))
        assertEquals("", decodePreviewText(ByteArray(0)))
        assertNull(decodePreviewText(byteArrayOf(0, 1, 2, 3, 0, 4)))
        assertTrue(decodePreviewText("{ \"ok\": true }".toByteArray())!!.startsWith("{"))
    }
}
