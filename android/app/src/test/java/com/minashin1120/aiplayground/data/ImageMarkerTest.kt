package com.minashin1120.aiplayground.data

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ImageMarkerTest {
    @Test fun colorsAndOpacityFollowWeb() {
        assertEquals("#aabbcc", normalizeMarkerHex("#ABC"))
        assertEquals("#ef4444", normalizeMarkerHex(" #EF4444 "))
        assertEquals("#facc15", normalizeMarkerHex("red"))
        assertEquals(0.1, clampMarkerOpacityPct(0.0), 0.0)
        assertEquals(100.0, clampMarkerOpacityPct(150.0), 0.0)
        assertEquals(60.0, clampMarkerOpacityPct(null), 0.0)
        assertEquals(60.0, clampMarkerOpacityPct(Double.NaN), 0.0)
        assertEquals("60", formatMarkerOpacityPct(60.0))
        assertEquals("12.3", formatMarkerOpacityPct(12.34))
        assertEquals("0.1", formatMarkerOpacityPct(0.05))
    }

    @Test fun strokePointsSkipTinyMovesAndFillLongOnes() {
        val points = mutableListOf<MarkerPoint>()
        assertTrue(appendStrokePoint(points, MarkerPoint(0f, 0f), 16f))
        assertFalse(appendStrokePoint(points, MarkerPoint(0.1f, 0f), 16f))
        assertTrue(appendStrokePoint(points, MarkerPoint(8f, 0f), 16f))
        assertEquals(listOf(MarkerPoint(0f, 0f), MarkerPoint(4f, 0f), MarkerPoint(8f, 0f)), points)
    }

    @Test fun mosaicAreasAndCells() {
        assertEquals(MarkerRect(2f, 2f, 16f, 16f), mosaicRectAt(MarkerPoint(10f, 10f), 16f))
        assertEquals(MarkerRect(7f, 7f, 6f, 6f), mosaicRectAt(MarkerPoint(10f, 10f), 3f))
        assertEquals(MarkerRect(1f, 2f, 4f, 6f), normalizeMosaicRect(MarkerPoint(5f, 8f), MarkerPoint(1f, 2f)))
        assertEquals(8, mosaicBlockSize(16f))
        assertEquals(4, mosaicBlockSize(5f))
        val cells = mosaicCells(MarkerRect(0f, 0f, 10f, 5f), 4, 100, 100)
        assertEquals(6, cells.size)
        assertArrayEquals(intArrayOf(0, 0, 4, 4), cells.first())
        assertArrayEquals(intArrayOf(8, 4, 2, 1), cells.last())
        val clipped = mosaicCells(MarkerRect(-5f, -5f, 10f, 10f), 4, 3, 3)
        assertEquals(1, clipped.size)
        assertArrayEquals(intArrayOf(0, 0, 3, 3), clipped[0])
    }

    @Test fun cropHandlesMatchWebHitTest() {
        val rect = MarkerRect(10f, 10f, 100f, 100f)
        assertEquals("nw", cropHitTest(MarkerPoint(12f, 12f), rect, 14f))
        assertEquals("move", cropHitTest(MarkerPoint(60f, 60f), rect, 14f))
        assertEquals("n", cropHitTest(MarkerPoint(60f, 11f), rect, 14f))
        assertEquals("w", cropHitTest(MarkerPoint(0f, 60f), rect, 14f))
        assertEquals("se", cropHitTest(MarkerPoint(200f, 200f), rect, 14f))
        assertEquals("w", cropHitTest(MarkerPoint(-50f, 60f), rect, 14f))
        assertEquals("n", cropHitTest(MarkerPoint(60f, -50f), rect, 14f))
        assertEquals("ne", cropHitTest(MarkerPoint(200f, -50f), rect, 14f))
    }

    @Test fun cropDragStaysInsideTheCanvas() {
        val start = MarkerRect(10f, 10f, 100f, 100f)
        assertEquals(MarkerRect(100f, 10f, 100f, 100f),
            dragCropRect("move", start, MarkerPoint(50f, 50f), MarkerPoint(500f, 50f), 200f, 200f, 8f))
        assertEquals(MarkerRect(10f, 10f, 140f, 140f),
            dragCropRect("se", start, MarkerPoint(110f, 110f), MarkerPoint(150f, 150f), 200f, 200f, 8f))
        assertEquals(MarkerRect(102f, 102f, 8f, 8f),
            dragCropRect("nw", start, MarkerPoint(10f, 10f), MarkerPoint(105f, 105f), 200f, 200f, 8f))
        assertTrue(isFullCrop(MarkerRect(0f, 0f, 100f, 50f), 100f, 50f))
        assertFalse(isFullCrop(MarkerRect(0f, 0f, 99f, 50f), 100f, 50f))
        assertArrayEquals(intArrayOf(100, 100, 300, 200), cropToImage(MarkerRect(10f, 20f, 30f, 40f), 100f, 100f, 1000, 500))
    }

    @Test fun zoomOffsetKeepsPartOfTheImageVisible() {
        assertEquals(0f to 0f, clampMarkerOffset(1f, 50f, 50f, 300f, 200f, 300f, 200f))
        assertEquals(264f to -376f, clampMarkerOffset(2f, 10000f, -10000f, 300f, 200f, 300f, 200f))
    }

    @Test fun editedFileNameAndSendItems() {
        assertEquals("photo_marked.png", markedFileName("photo.jpg"))
        assertEquals("a.b_marked.png", markedFileName("a.b.png"))
        assertEquals("noext_marked.png", markedFileName("noext"))
        val original = Attachment("o.png", "1/o", source = "upload")
        val items = attachmentItemsForSend(listOf(
            Attachment("a.png", "1/a"),
            Attachment("b.png", "1/b", source = "library", original = original, attachOriginal = true),
            Attachment("b2.png", "1/b", source = "upload"),
        ))
        assertEquals(listOf("1/a", "1/b", "1/o"), items.map { it.reference })
        assertEquals("upload", items[1].source)
        assertEquals("b.png", items[1].name)
        assertEquals(listOf("1/a"), attachmentItemsForSend(listOf(Attachment("a.png", "1/a", original = original))).map { it.reference })
    }
}
