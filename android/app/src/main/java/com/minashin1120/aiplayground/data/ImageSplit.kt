package com.minashin1120.aiplayground.data

import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.roundToInt

/** 画像分割: pieces per image. The upper bound matches the attachment limit of one message. */
internal const val IMAGE_SPLIT_MIN_PIECES = 2
internal const val IMAGE_SPLIT_MAX_PIECES = 30
internal const val IMAGE_SPLIT_DEFAULT_PIECES = 4
internal const val IMAGE_SPLIT_MAX_OVERLAP_PCT = 50
internal const val IMAGE_SPLIT_DEFAULT_OVERLAP_PCT = 10
/** Images shared or picked at once. */
internal const val IMAGE_SPLIT_MAX_IMAGES = 10
/** Each nominal cell must keep at least this many pixels per side. */
internal const val IMAGE_SPLIT_MIN_CELL_PX = 16

internal data class ImageSplitOptions(
    val pieces: Int = IMAGE_SPLIT_DEFAULT_PIECES,
    val overlapPct: Int = IMAGE_SPLIT_DEFAULT_OVERLAP_PCT,
    /** Attach or save the whole image together with the pieces. */
    val includeOriginal: Boolean = true,
    /** Draw the cut lines, overlap bands and piece numbers on that whole image. */
    val markOriginal: Boolean = true,
    /** Add a caption strip (number, row/column, source coordinates) and dashed cut lines to each piece. */
    val markPieces: Boolean = true,
)

internal data class SplitGrid(val columns: Int, val rows: Int) {
    val count: Int get() = columns * rows
}

/**
 * One piece. [left]..[bottom] is the cropped area including the overlap with its neighbours;
 * [cellLeft]..[cellBottom] is the nominal cell between the cut lines. Right/bottom are exclusive.
 */
internal data class SplitTile(
    val index: Int,
    val row: Int,
    val column: Int,
    val left: Int,
    val top: Int,
    val right: Int,
    val bottom: Int,
    val cellLeft: Int,
    val cellTop: Int,
    val cellRight: Int,
    val cellBottom: Int,
) {
    val width: Int get() = right - left
    val height: Int get() = bottom - top
}

internal fun clampSplitPieces(value: Int?): Int =
    (value ?: IMAGE_SPLIT_DEFAULT_PIECES).coerceIn(IMAGE_SPLIT_MIN_PIECES, IMAGE_SPLIT_MAX_PIECES)

internal fun clampSplitOverlap(value: Int?): Int =
    (value ?: IMAGE_SPLIT_DEFAULT_OVERLAP_PCT).coerceIn(0, IMAGE_SPLIT_MAX_OVERLAP_PCT)

/**
 * The columns × rows whose product is exactly [pieces] and whose cells are closest to square for a
 * [width] × [height] image. A prime count therefore becomes strips along the longer side.
 */
internal fun chooseSplitGrid(pieces: Int, width: Int, height: Int): SplitGrid {
    val count = clampSplitPieces(pieces)
    val w = width.coerceAtLeast(1).toDouble()
    val h = height.coerceAtLeast(1).toDouble()
    var best = SplitGrid(count, 1)
    var bestScore = Double.MAX_VALUE
    for (columns in 1..count) {
        if (count % columns != 0) continue
        val rows = count / columns
        val score = abs(ln((w / columns) / (h / rows)))
        // Ties (e.g. a square image split in 2) prefer columns for landscape and rows for portrait.
        val better = score < bestScore - 1e-9 ||
            (abs(score - bestScore) <= 1e-9 && (if (width >= height) columns > best.columns else rows > best.rows))
        if (better) { best = SplitGrid(columns, rows); bestScore = score }
    }
    return best
}

/** Whether every nominal cell keeps at least [IMAGE_SPLIT_MIN_CELL_PX] pixels per side. */
internal fun splitFits(grid: SplitGrid, width: Int, height: Int): Boolean =
    width / grid.columns >= IMAGE_SPLIT_MIN_CELL_PX && height / grid.rows >= IMAGE_SPLIT_MIN_CELL_PX

/** Cut positions 0 = c0 < c1 < … < cN = [size], spread as evenly as whole pixels allow. */
internal fun splitCuts(size: Int, parts: Int): IntArray =
    IntArray(parts + 1) { i -> (i.toLong() * size / parts).toInt() }

/**
 * Pieces in reading order (left to right, top to bottom). Neighbouring pieces share [overlapPct] % of a
 * cell: each piece extends by half of that past every inner cut line, never past the image edge.
 */
internal fun splitTiles(width: Int, height: Int, grid: SplitGrid, overlapPct: Int): List<SplitTile> {
    val xs = splitCuts(width, grid.columns)
    val ys = splitCuts(height, grid.rows)
    val overlap = clampSplitOverlap(overlapPct) / 100.0
    val padX = (width.toDouble() / grid.columns * overlap / 2).roundToInt()
    val padY = (height.toDouble() / grid.rows * overlap / 2).roundToInt()
    return buildList {
        for (row in 0 until grid.rows) for (column in 0 until grid.columns) {
            add(SplitTile(
                index = row * grid.columns + column,
                row = row,
                column = column,
                left = (xs[column] - padX).coerceAtLeast(0),
                top = (ys[row] - padY).coerceAtLeast(0),
                right = (xs[column + 1] + padX).coerceAtMost(width),
                bottom = (ys[row + 1] + padY).coerceAtMost(height),
                cellLeft = xs[column],
                cellTop = ys[row],
                cellRight = xs[column + 1],
                cellBottom = ys[row + 1],
            ))
        }
    }
}

/**
 * Caption drawn above a piece so the model can place it: number, row/column and the covered area
 * in the source image's pixels. ASCII only so that every model reads it the same way.
 */
internal fun splitTileCaption(tile: SplitTile, grid: SplitGrid, width: Int, height: Int): String =
    "Tile ${tile.index + 1}/${grid.count}  row ${tile.row + 1}/${grid.rows}, col ${tile.column + 1}/${grid.columns}  " +
        "x ${tile.left}-${tile.right}, y ${tile.top}-${tile.bottom} of ${width}x$height"

/** File-name stem without the extension and characters that are awkward in file names. */
internal fun splitBaseName(name: String): String {
    val stem = name.substringAfterLast('/').let { if (it.contains('.')) it.substringBeforeLast('.') else it }
    return stem.replace(Regex("[\\\\/:*?\"<>|\\s]+"), "_").trim('_', '.').take(80).ifBlank { "image" }
}

/** `photo_part02of06_r1c2.jpg`: zero-padded so the pieces sort in reading order. */
internal fun splitTileName(base: String, tile: SplitTile, grid: SplitGrid, extension: String): String {
    val digits = grid.count.toString().length
    val number = (tile.index + 1).toString().padStart(digits, '0')
    return "${base}_part${number}of${grid.count}_r${tile.row + 1}c${tile.column + 1}.$extension"
}

internal fun splitOriginalName(base: String, marked: Boolean, extension: String): String =
    if (marked) "${base}_split_grid.$extension" else "${base}.$extension"

/** PNG keeps lossless sources lossless; photos stay JPEG. */
internal fun splitOutputIsPng(sourceMime: String, sourceName: String): Boolean {
    val mime = sourceMime.lowercase()
    val ext = sourceName.substringAfterLast('.', "").lowercase()
    val photo = mime == "image/jpeg" || mime == "image/jpg" || mime == "image/heic" || mime == "image/heif" ||
        ext in setOf("jpg", "jpeg", "heic", "heif")
    return !photo
}

/**
 * Maps a rectangle of the upright image back to the stored pixels of an image whose EXIF says it is
 * mirrored horizontally ([flipped], applied first) and then rotated clockwise by [rotation] degrees.
 * Returns left, top, right, bottom in stored-pixel coordinates.
 */
internal fun orientedRectToRaw(
    left: Int, top: Int, right: Int, bottom: Int,
    rawWidth: Int, rawHeight: Int, rotation: Int, flipped: Boolean,
): IntArray {
    val degrees = ((rotation % 360) + 360) % 360
    fun unrotate(u: Int, v: Int): Pair<Int, Int> = when (degrees) {
        90 -> v to rawHeight - u
        180 -> rawWidth - u to rawHeight - v
        270 -> rawWidth - v to u
        else -> u to v
    }
    val a = unrotate(left, top)
    val b = unrotate(right, bottom)
    var l = minOf(a.first, b.first)
    var r = maxOf(a.first, b.first)
    val t = minOf(a.second, b.second)
    val bt = maxOf(a.second, b.second)
    if (flipped) { val nl = rawWidth - r; r = rawWidth - l; l = nl }
    return intArrayOf(l.coerceIn(0, rawWidth), t.coerceIn(0, rawHeight), r.coerceIn(0, rawWidth), bt.coerceIn(0, rawHeight))
}

/** Power-of-two sample size that keeps a [width] × [height] decode at or under [maxPixels]. */
internal fun splitSampleSize(width: Int, height: Int, maxPixels: Long): Int {
    var sample = 1
    while (width.toLong() * height / (sample.toLong() * sample) > maxPixels) sample *= 2
    return sample
}
