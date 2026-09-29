package com.minashin1120.aiplayground.data

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ImageSplitTest {
    @Test fun countsAndOverlapAreClamped() {
        assertEquals(IMAGE_SPLIT_DEFAULT_PIECES, clampSplitPieces(null))
        assertEquals(IMAGE_SPLIT_MIN_PIECES, clampSplitPieces(1))
        assertEquals(IMAGE_SPLIT_MAX_PIECES, clampSplitPieces(99))
        assertEquals(IMAGE_SPLIT_DEFAULT_OVERLAP_PCT, clampSplitOverlap(null))
        assertEquals(0, clampSplitOverlap(-5))
        assertEquals(IMAGE_SPLIT_MAX_OVERLAP_PCT, clampSplitOverlap(80))
    }

    @Test fun gridUsesTheExactCountWithNearSquareCells() {
        assertEquals(SplitGrid(2, 2), chooseSplitGrid(4, 1000, 1000))
        assertEquals(SplitGrid(3, 2), chooseSplitGrid(6, 1600, 1000))
        assertEquals(SplitGrid(2, 3), chooseSplitGrid(6, 1000, 1600))
        assertEquals(SplitGrid(4, 1), chooseSplitGrid(4, 4000, 1000))
        // A prime count becomes strips along the longer side.
        assertEquals(SplitGrid(5, 1), chooseSplitGrid(5, 1600, 1000))
        assertEquals(SplitGrid(1, 5), chooseSplitGrid(5, 1000, 1600))
        // Ties: columns for landscape/square, rows for portrait.
        assertEquals(SplitGrid(2, 1), chooseSplitGrid(2, 1000, 1000))
        for (pieces in IMAGE_SPLIT_MIN_PIECES..IMAGE_SPLIT_MAX_PIECES) {
            assertEquals(pieces, chooseSplitGrid(pieces, 4032, 3024).count)
        }
    }

    @Test fun tilesOverlapAtInnerCutsOnlyAndCoverTheImage() {
        val grid = SplitGrid(2, 2)
        val tiles = splitTiles(1000, 800, grid, 10)
        assertEquals(4, tiles.size)
        val first = tiles[0]
        // Cell 500×400, 10 % overlap → each side extends by 25 px horizontally and 20 px vertically.
        assertEquals(listOf(0, 0, 525, 420), listOf(first.left, first.top, first.right, first.bottom))
        assertEquals(listOf(0, 0, 500, 400), listOf(first.cellLeft, first.cellTop, first.cellRight, first.cellBottom))
        val last = tiles[3]
        assertEquals(listOf(475, 380, 1000, 800), listOf(last.left, last.top, last.right, last.bottom))
        assertEquals(listOf(1, 1, 3), listOf(last.row, last.column, last.index))
        // Neighbours share 50 px (10 % of the cell width).
        assertEquals(50, tiles[0].right - tiles[1].left)
        assertEquals(1000, tiles.maxOf { it.right })
        assertEquals(0, tiles.minOf { it.left })
    }

    @Test fun zeroOverlapTilesMeetAtTheCuts() {
        val tiles = splitTiles(1001, 10, SplitGrid(3, 1), 0)
        assertEquals(listOf(0, 333, 667, 1001), listOf(tiles[0].left, tiles[1].left, tiles[2].left, tiles[2].right))
        tiles.forEach { assertEquals(it.cellLeft, it.left); assertEquals(it.cellRight, it.right) }
        assertArrayEquals(intArrayOf(0, 333, 667, 1001), splitCuts(1001, 3))
    }

    @Test fun smallImagesDoNotFit() {
        assertTrue(splitFits(SplitGrid(2, 2), 32, 32))
        assertFalse(splitFits(SplitGrid(4, 1), 60, 200))
    }

    @Test fun namesSortInReadingOrderAndCaptionsGiveThePosition() {
        val grid = SplitGrid(4, 3)
        val tile = splitTiles(4000, 3000, grid, 10)[5]
        assertEquals("photo_part06of12_r2c2.jpg", splitTileName("photo", tile, grid, "jpg"))
        assertEquals("photo_split_grid.png", splitOriginalName("photo", true, "png"))
        assertEquals("IMG_01", splitBaseName("IMG 01.HEIC"))
        assertEquals("image", splitBaseName(".."))
        assertEquals("Tile 6/12  row 2/3, col 2/4  x 950-2050, y 950-2050 of 4000x3000",
            splitTileCaption(tile, grid, 4000, 3000))
    }

    @Test fun photosStayJpegAndOthersBecomePng() {
        assertFalse(splitOutputIsPng("image/jpeg", "a.jpg"))
        assertFalse(splitOutputIsPng("", "a.HEIC"))
        assertTrue(splitOutputIsPng("image/png", "a.png"))
        assertTrue(splitOutputIsPng("image/webp", "a.webp"))
    }

    @Test fun orientedRectanglesMapBackToStoredPixels() {
        // Stored 400×300; rotated 90° clockwise it is displayed as 300×400.
        assertArrayEquals(intArrayOf(0, 200, 100, 300), orientedRectToRaw(0, 0, 100, 100, 400, 300, 90, false))
        assertArrayEquals(intArrayOf(300, 200, 400, 300), orientedRectToRaw(0, 0, 100, 100, 400, 300, 180, false))
        assertArrayEquals(intArrayOf(300, 0, 400, 100), orientedRectToRaw(0, 0, 100, 100, 400, 300, 270, false))
        assertArrayEquals(intArrayOf(300, 0, 400, 100), orientedRectToRaw(0, 0, 100, 100, 400, 300, 0, true))
        // EXIF transpose (mirror, then 270°) swaps the axes.
        assertArrayEquals(intArrayOf(10, 20, 50, 30), orientedRectToRaw(20, 10, 30, 50, 400, 300, 270, true))
        assertArrayEquals(intArrayOf(5, 6, 7, 8), orientedRectToRaw(5, 6, 7, 8, 400, 300, 0, false))
    }

    @Test fun sampleSizeKeepsDecodesUnderTheLimit() {
        assertEquals(1, splitSampleSize(4000, 3000, 16_000_000L))
        assertEquals(2, splitSampleSize(8000, 6000, 16_000_000L))
        assertEquals(4, splitSampleSize(16000, 12000, 16_000_000L))
    }
}
