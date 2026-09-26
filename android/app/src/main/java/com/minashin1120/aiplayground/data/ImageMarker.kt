package com.minashin1120.aiplayground.data

import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * The Web image marker editor (`#marker-modal`) in canvas coordinates: the canvas has the size the
 * image is shown at (Web `img.clientWidth` / `clientHeight`), and the result is scaled up to the
 * image's own size when saved.
 */
data class MarkerPoint(val x: Float, val y: Float)

data class MarkerRect(val x: Float, val y: Float, val w: Float, val h: Float) {
    val right get() = x + w
    val bottom get() = y + h
}

enum class MarkerMode { DRAW, MOSAIC, CROP }

/** Web `markerToolHints`. */
val MARKER_TOOL_HINTS = mapOf(
    MarkerMode.DRAW to "マーカー（色・透明度変更可） / 二本指で拡大",
    MarkerMode.MOSAIC to "ドラッグで範囲モザイク（複数追加可） / 二本指で拡大",
    MarkerMode.CROP to "外側をドラッグして切り取り / 二本指で拡大",
)

/** Web marker color chips (`data-marker-color`) with their titles; yellow is the default. */
val MARKER_COLORS = listOf("#ef4444" to "赤", "#3b82f6" to "青", "#111827" to "黒", "#facc15" to "黄色", "#ec4899" to "ピンク")
const val MARKER_DEFAULT_COLOR = "#facc15"
const val MARKER_OPACITY_MIN_PCT = 0.1
const val MARKER_OPACITY_MAX_PCT = 100.0
const val MARKER_HISTORY_LIMIT = 40
const val MARKER_MIN_CROP_SIZE = 8f
const val MARKER_HANDLE_RADIUS = 14f
const val MARKER_MAX_SCALE = 4f

/** Web `normalizeMarkerHexColor`. */
fun normalizeMarkerHex(color: String?): String {
    val value = color.orEmpty().trim().lowercase()
    if (Regex("^#[0-9a-f]{6}$").matches(value)) return value
    if (Regex("^#[0-9a-f]{3}$").matches(value)) return "#" + value.substring(1).map { "$it$it" }.joinToString("")
    return MARKER_DEFAULT_COLOR
}

/** Web `clampMarkerOpacityPct`. */
fun clampMarkerOpacityPct(value: Double?, fallback: Double = 60.0): Double {
    val pct = value?.takeIf { it.isFinite() } ?: fallback
    return pct.coerceIn(MARKER_OPACITY_MIN_PCT, MARKER_OPACITY_MAX_PCT)
}

/** Web `formatMarkerOpacityPct`: one decimal, without a trailing ".0". */
fun formatMarkerOpacityPct(pct: Double): String {
    val rounded = (clampMarkerOpacityPct(pct) * 10).roundToInt() / 10.0
    return if (rounded % 1.0 == 0.0) rounded.toLong().toString() else rounded.toString()
}

/**
 * Web `appendStrokePoint`: points closer than a small step are skipped and long moves are filled in,
 * so fast strokes stay smooth. Returns whether the stroke changed.
 */
fun appendStrokePoint(points: MutableList<MarkerPoint>, point: MarkerPoint, strokeSize: Float): Boolean {
    if (points.isEmpty()) { points += point; return true }
    val from = points.last()
    val dx = point.x - from.x
    val dy = point.y - from.y
    val dist = hypot(dx, dy)
    if (dist < max(0.35f, strokeSize * 0.04f)) return false
    val steps = max(1, ceil(dist / max(1f, strokeSize * 0.25f)).toInt())
    for (i in 1..steps) {
        val t = i.toFloat() / steps
        points += MarkerPoint(from.x + dx * t, from.y + dy * t)
    }
    return true
}

/** Web `normalizeMosaicRect`. */
fun normalizeMosaicRect(a: MarkerPoint, b: MarkerPoint) =
    MarkerRect(min(a.x, b.x), min(a.y, b.y), abs(a.x - b.x), abs(a.y - b.y))

/** Web `buildMosaicRectFromPoint`: a square the brush size (at least 6) around the tap. */
fun mosaicRectAt(point: MarkerPoint, size: Float): MarkerRect {
    val side = max(6f, floor(size))
    val half = floor(side / 2)
    return MarkerRect(point.x - half, point.y - half, side, side)
}

/** Web `applyMosaicRect` block size. */
fun mosaicBlockSize(size: Float): Int = max(4, floor(size / 2).toInt())

/** The mosaic cells of [rect] clipped to the canvas: (x, y, width, height) with the cell's sample point at its centre. */
fun mosaicCells(rect: MarkerRect, block: Int, width: Int, height: Int): List<IntArray> {
    val x1 = max(0, floor(rect.x).toInt())
    val y1 = max(0, floor(rect.y).toInt())
    val x2 = min(width, ceil(rect.right).toInt())
    val y2 = min(height, ceil(rect.bottom).toInt())
    if (x2 <= x1 || y2 <= y1) return emptyList()
    val cells = mutableListOf<IntArray>()
    var y = y1
    while (y < y2) {
        var x = x1
        while (x < x2) {
            cells += intArrayOf(x, y, min(block, x2 - x), min(block, y2 - y))
            x += block
        }
        y += block
    }
    return cells
}

/** Web crop `hitTest`: which edge or corner a press grabs; outside the rectangle picks the nearest side. */
fun cropHitTest(p: MarkerPoint, rect: MarkerRect?, handle: Float): String {
    if (rect == null) return "move"
    val nearLeft = abs(p.x - rect.x) <= handle
    val nearRight = abs(p.x - rect.right) <= handle
    val nearTop = abs(p.y - rect.y) <= handle
    val nearBottom = abs(p.y - rect.bottom) <= handle
    when {
        nearLeft && nearTop -> return "nw"
        nearRight && nearTop -> return "ne"
        nearLeft && nearBottom -> return "sw"
        nearRight && nearBottom -> return "se"
        nearTop -> return "n"
        nearBottom -> return "s"
        nearLeft -> return "w"
        nearRight -> return "e"
    }
    if (p.x > rect.x + handle && p.x < rect.right - handle && p.y > rect.y + handle && p.y < rect.bottom - handle) return "move"
    val outsideX = if (p.x < rect.x) "w" else if (p.x > rect.right) "e" else null
    val outsideY = if (p.y < rect.y) "n" else if (p.y > rect.bottom) "s" else null
    if (outsideX != null && outsideY != null) return outsideY + outsideX
    return outsideX ?: outsideY ?: "move"
}

/** Web crop `move`: the rectangle after dragging [mode] from [dragStart] to [p]. */
fun dragCropRect(mode: String, start: MarkerRect, dragStart: MarkerPoint, p: MarkerPoint, maxW: Float, maxH: Float, minSize: Float): MarkerRect {
    var x = start.x
    var y = start.y
    var w = start.w
    var h = start.h
    fun clamp(value: Float, low: Float, high: Float) = min(high, max(low, value))
    if (mode == "move") {
        x = clamp(start.x + p.x - dragStart.x, 0f, maxW - start.w)
        y = clamp(start.y + p.y - dragStart.y, 0f, maxH - start.h)
    } else {
        if ('w' in mode) { val nx = clamp(p.x, 0f, start.right - minSize); x = nx; w = start.right - nx }
        if ('e' in mode) w = clamp(p.x - start.x, minSize, maxW - start.x)
        if ('n' in mode) { val ny = clamp(p.y, 0f, start.bottom - minSize); y = ny; h = start.bottom - ny }
        if ('s' in mode) h = clamp(p.y - start.y, minSize, maxH - start.y)
    }
    return MarkerRect(clamp(x, 0f, maxW - w), clamp(y, 0f, maxH - h), w, h)
}

/** Whether [rect] covers the whole canvas (Web `renderCropOverlay` `isFull`). */
fun isFullCrop(rect: MarkerRect, width: Float, height: Float) =
    rect.x == 0f && rect.y == 0f && abs(rect.w - width) < 1 && abs(rect.h - height) < 1

/** The crop in image pixels (Web `saveMarkerToRow`). */
fun cropToImage(rect: MarkerRect, canvasW: Float, canvasH: Float, imageW: Int, imageH: Int): IntArray {
    val scaleX = imageW / canvasW
    val scaleY = imageH / canvasH
    val cx = max(0, floor(rect.x * scaleX).toInt())
    val cy = max(0, floor(rect.y * scaleY).toInt())
    val cw = min(imageW - cx, max(1, floor(rect.w * scaleX).toInt()))
    val ch = min(imageH - cy, max(1, floor(rect.h * scaleY).toInt()))
    return intArrayOf(cx, cy, max(1, cw), max(1, ch))
}

/**
 * Web `clampMarkerViewOffset`: at the minimum scale the view is centred; zoomed in, at least a margin of
 * the image stays on screen. [base] is the image's top-left inside the stage at scale 1.
 */
fun clampMarkerOffset(
    scale: Float, offsetX: Float, offsetY: Float,
    stageW: Float, stageH: Float, baseW: Float, baseH: Float,
): Pair<Float, Float> {
    if (scale <= 1.0001f || stageW <= 1 || stageH <= 1 || baseW <= 1 || baseH <= 1) return 0f to 0f
    val baseLeft = (stageW - baseW) / 2
    val baseTop = (stageH - baseH) / 2
    val minVisibleX = min(stageW * 0.45f, max(24f, stageW * 0.12f))
    val minVisibleY = min(stageH * 0.45f, max(24f, stageH * 0.12f))
    fun clamp(value: Float, low: Float, high: Float) = if (low > high) (low + high) / 2 else min(high, max(low, value))
    return clamp(offsetX, minVisibleX - baseLeft - baseW * scale, stageW - minVisibleX - baseLeft) to
        clamp(offsetY, minVisibleY - baseTop - baseH * scale, stageH - minVisibleY - baseTop)
}

/** Web `saveMarkerToRow` file name: `<name without extension>_marked.png`. */
fun markedFileName(name: String): String = "${name.replace(Regex("\\.[^/.]+$"), "")}_marked.png"
