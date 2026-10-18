package com.minashin1120.aiplayground.ui

import com.minashin1120.aiplayground.data.AttachmentKind
import com.minashin1120.aiplayground.data.attachmentKind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.unit.IntSize
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
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

    @Test fun backdropTapsAreThoseOutsideTheFittedImage() {
        val viewport = IntSize(1000, 2000)
        // A 2:1 landscape image fits the width: 1000×500 centered, from y=750 to y=1250.
        val image = IntSize(400, 200)
        assertTrue(isOnFittedImage(Offset(500f, 1000f), viewport, image, 1f, Offset.Zero))
        assertFalse(isOnFittedImage(Offset(500f, 300f), viewport, image, 1f, Offset.Zero))
        assertFalse(isOnFittedImage(Offset(500f, 1300f), viewport, image, 1f, Offset.Zero))
        // Zoomed 2× the image reaches y=500..1500; moved down 400 it covers y=900..1900.
        assertTrue(isOnFittedImage(Offset(500f, 600f), viewport, image, 2f, Offset.Zero))
        assertFalse(isOnFittedImage(Offset(500f, 600f), viewport, image, 2f, Offset(0f, 400f)))
        // Until the image has loaded the whole viewer is backdrop, as Web's empty viewer.
        assertFalse(isOnFittedImage(Offset(500f, 1000f), viewport, null, 1f, Offset.Zero))
    }

    @Test fun tapsDoNotCloseTheViewerWhileTheImageLoads() {
        val viewport = IntSize(1000, 2000)
        val image = IntSize(400, 200)
        // The full image can take seconds: a tap before it arrives must not dismiss the empty viewer.
        assertFalse(viewerTapCloses(Offset(500f, 300f), viewport, null, 1f, Offset.Zero, ready = false, failed = false))
        assertFalse(viewerTapCloses(Offset(500f, 300f), viewport, image, 1f, Offset.Zero, ready = false, failed = false))
        // Once loaded, the backdrop still closes and the image itself does not.
        assertTrue(viewerTapCloses(Offset(500f, 300f), viewport, image, 1f, Offset.Zero, ready = true, failed = false))
        assertFalse(viewerTapCloses(Offset(500f, 1000f), viewport, image, 1f, Offset.Zero, ready = true, failed = false))
        // A failed load must not trap the user in the viewer.
        assertTrue(viewerTapCloses(Offset(500f, 300f), viewport, null, 1f, Offset.Zero, ready = false, failed = true))
    }
}
