package com.minashin1120.aiplayground.data

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.BitmapRegionDecoder
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.Typeface
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.DocumentsContract
import android.provider.MediaStore
import android.provider.OpenableColumns
import androidx.core.content.FileProvider
import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream

/** Largest decode for one piece or for the marked whole image. */
private const val SPLIT_MAX_DECODE_PIXELS = 16_000_000L
/** Folder under Pictures used by 保存のみ on Android 10 and later. */
internal const val IMAGE_SPLIT_SAVE_FOLDER = "AI Playground"

/** Images waiting in the 画像分割 dialog; [busy] while they are rendered or saved, [error] after a failed run. */
internal data class ImageSplitRequest(val uris: List<Uri>, val busy: Boolean = false, val error: String? = null)

/** An image picked or shared for splitting, measured upright (after EXIF orientation). */
internal data class SplitSource(
    val uri: Uri,
    val name: String,
    val mime: String,
    val width: Int,
    val height: Int,
    val rawWidth: Int,
    val rawHeight: Int,
    val rotation: Int,
    val flipped: Boolean,
)

/** One file ready to attach ([uploadUri]) or to save ([open]). */
internal class SplitOutput(val name: String, val mime: String, val uploadUri: Uri, val open: () -> InputStream)

internal class SplitPreview(val bitmap: Bitmap, val source: SplitSource)

internal fun readSplitSource(context: Context, uri: Uri): SplitSource {
    val resolver = context.contentResolver
    var name = "image"
    resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
        if (cursor.moveToFirst()) {
            val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (index >= 0) cursor.getString(index)?.takeIf { it.isNotBlank() }?.let { name = it }
        }
    }
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    (resolver.openInputStream(uri) ?: throw IOException("画像を開けません。")).use { BitmapFactory.decodeStream(it, null, bounds) }
    if (bounds.outWidth <= 0 || bounds.outHeight <= 0) throw IOException("画像として読み込めません: $name")
    val mime = resolver.getType(uri)?.takeIf { it.startsWith("image/") } ?: bounds.outMimeType.orEmpty()
    val exif = runCatching { resolver.openInputStream(uri)?.use { ExifInterface(it) } }.getOrNull()
    val rotation = exif?.rotationDegrees ?: 0
    val flipped = exif?.isFlipped ?: false
    val swap = rotation == 90 || rotation == 270
    return SplitSource(
        uri = uri, name = name, mime = mime,
        width = if (swap) bounds.outHeight else bounds.outWidth,
        height = if (swap) bounds.outWidth else bounds.outHeight,
        rawWidth = bounds.outWidth, rawHeight = bounds.outHeight,
        rotation = rotation, flipped = flipped,
    )
}

/** A small upright copy for the dialog preview. */
internal fun loadSplitPreview(context: Context, uri: Uri, maxSide: Int = 1024): SplitPreview {
    val source = readSplitSource(context, uri)
    var sample = 1
    while (maxOf(source.rawWidth, source.rawHeight) / (sample * 2) >= maxSide) sample *= 2
    val decoded = context.contentResolver.openInputStream(uri)?.use {
        BitmapFactory.decodeStream(it, null, BitmapFactory.Options().apply { inSampleSize = sample })
    } ?: throw IOException("画像として読み込めません: ${source.name}")
    return SplitPreview(orient(decoded, source), source)
}

/**
 * Renders the whole image (optional) and the pieces of [uri] into [outDir] and returns them in the order
 * they should be attached: the whole image first, then the pieces in reading order.
 */
internal fun renderImageSplit(context: Context, uri: Uri, options: ImageSplitOptions, outDir: File): List<SplitOutput> {
    val source = readSplitSource(context, uri)
    val grid = chooseSplitGrid(options.pieces, source.width, source.height)
    if (!splitFits(grid, source.width, source.height)) {
        throw IOException("画像が小さすぎるため${grid.count}枚に分割できません: ${source.name}")
    }
    val tiles = splitTiles(source.width, source.height, grid, options.overlapPct)
    val png = splitOutputIsPng(source.mime, source.name)
    val extension = if (png) "png" else "jpg"
    val mime = if (png) "image/png" else "image/jpeg"
    val base = splitBaseName(source.name)
    outDir.mkdirs()
    val outputs = mutableListOf<SplitOutput>()
    if (options.includeOriginal) {
        if (options.markOriginal) {
            val whole = decodeWhole(context, source)
            val marked = try { drawGridOverview(whole, source, grid, tiles) } finally { whole.recycle() }
            outputs += writeOutput(context, marked, File(outDir, splitOriginalName(base, true, extension)), png, mime)
        } else {
            val originalName = source.name.takeIf { it.contains('.') } ?: splitOriginalName(base, false, extension)
            outputs += SplitOutput(originalName, source.mime.ifBlank { mime }, uri) {
                context.contentResolver.openInputStream(uri) ?: throw IOException("画像を開けません。")
            }
        }
    }
    val decoder = openRegionDecoder(context, uri)
    var fallback: Bitmap? = null
    try {
        for (tile in tiles) {
            val piece = if (decoder != null) {
                decodeRegion(decoder, source, tile)
            } else {
                val whole = fallback ?: decodeWhole(context, source).also { fallback = it }
                cropScaled(whole, source, tile)
            }
            val output = if (options.markPieces) {
                try { drawPieceMarks(piece, tile, grid, source) } finally { piece.recycle() }
            } else piece
            outputs += writeOutput(context, output, File(outDir, splitTileName(base, tile, grid, extension)), png, mime)
        }
    } finally {
        decoder?.recycle()
        fallback?.recycle()
    }
    return outputs
}

/** 保存のみ: Pictures/AI Playground on Android 10+, or the folder the user picked on older versions. */
internal fun saveSplitOutputs(context: Context, outputs: List<SplitOutput>, tree: Uri?): Int {
    val resolver = context.contentResolver
    var saved = 0
    for (output in outputs) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, output.name)
                put(MediaStore.Images.Media.MIME_TYPE, output.mime)
                put(MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/$IMAGE_SPLIT_SAVE_FOLDER")
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
            val target = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                ?: throw IOException("端末に保存できません。")
            try {
                copyTo(resolver.openOutputStream(target), output)
                resolver.update(target, ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) }, null, null)
            } catch (e: Exception) {
                runCatching { resolver.delete(target, null, null) }
                throw e
            }
        } else {
            val folder = tree ?: throw IOException("保存先のフォルダーが選択されていません。")
            val parent = DocumentsContract.buildDocumentUriUsingTree(folder, DocumentsContract.getTreeDocumentId(folder))
            val target = DocumentsContract.createDocument(resolver, parent, output.mime, output.name)
                ?: throw IOException("端末に保存できません。")
            copyTo(resolver.openOutputStream(target), output)
        }
        saved++
    }
    return saved
}

/** FileProvider URI so the upload reads the same display name and type as a picked file. */
internal fun splitShareUri(context: Context, file: File): Uri =
    FileProvider.getUriForFile(context, "${context.packageName}.files", file)

private fun copyTo(target: java.io.OutputStream?, output: SplitOutput) {
    (target ?: throw IOException("端末に保存できません。")).use { out -> output.open().use { it.copyTo(out) } }
}

private fun writeOutput(context: Context, bitmap: Bitmap, file: File, png: Boolean, mime: String): SplitOutput {
    try {
        FileOutputStream(file).use { out ->
            val ok = if (png) bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            else bitmap.compress(Bitmap.CompressFormat.JPEG, 95, out)
            if (!ok) throw IOException("分割画像を書き出せません。")
        }
    } finally {
        bitmap.recycle()
    }
    return SplitOutput(file.name, mime, splitShareUri(context, file)) { file.inputStream() }
}

@Suppress("DEPRECATION")
private fun openRegionDecoder(context: Context, uri: Uri): BitmapRegionDecoder? = runCatching {
    context.contentResolver.openInputStream(uri)?.use { input ->
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) BitmapRegionDecoder.newInstance(input)
        else BitmapRegionDecoder.newInstance(input, false)
    }
}.getOrNull()

private fun decodeRegion(decoder: BitmapRegionDecoder, source: SplitSource, tile: SplitTile): Bitmap {
    val raw = orientedRectToRaw(tile.left, tile.top, tile.right, tile.bottom,
        source.rawWidth, source.rawHeight, source.rotation, source.flipped)
    val rect = Rect(raw[0], raw[1], raw[2], raw[3])
    val sample = splitSampleSize(rect.width(), rect.height(), SPLIT_MAX_DECODE_PIXELS)
    val decoded = decoder.decodeRegion(rect, BitmapFactory.Options().apply { inSampleSize = sample })
        ?: throw IOException("画像の一部を読み込めません: ${source.name}")
    return orient(decoded, source)
}

/** Upright whole image, reduced to at most [SPLIT_MAX_DECODE_PIXELS]. */
private fun decodeWhole(context: Context, source: SplitSource): Bitmap {
    val sample = splitSampleSize(source.rawWidth, source.rawHeight, SPLIT_MAX_DECODE_PIXELS)
    val decoded = context.contentResolver.openInputStream(source.uri)?.use {
        BitmapFactory.decodeStream(it, null, BitmapFactory.Options().apply { inSampleSize = sample })
    } ?: throw IOException("画像として読み込めません: ${source.name}")
    return orient(decoded, source)
}

/** Fallback for formats without region decoding (e.g. GIF): crop from the reduced whole image. */
private fun cropScaled(whole: Bitmap, source: SplitSource, tile: SplitTile): Bitmap {
    val sx = whole.width.toDouble() / source.width
    val sy = whole.height.toDouble() / source.height
    val l = (tile.left * sx).toInt().coerceIn(0, whole.width - 1)
    val t = (tile.top * sy).toInt().coerceIn(0, whole.height - 1)
    val r = (tile.right * sx).toInt().coerceIn(l + 1, whole.width)
    val b = (tile.bottom * sy).toInt().coerceIn(t + 1, whole.height)
    return Bitmap.createBitmap(whole, l, t, r - l, b - t).let { if (it === whole) it.copy(Bitmap.Config.ARGB_8888, false) else it }
}

/** Applies the EXIF mirror and rotation (same order as [normalizeCapturedPhotoOrientation]). */
private fun orient(bitmap: Bitmap, source: SplitSource): Bitmap {
    if (source.rotation == 0 && !source.flipped) return bitmap
    val matrix = Matrix().apply {
        if (source.flipped) postScale(-1f, 1f)
        if (source.rotation != 0) postRotate(source.rotation.toFloat())
    }
    val oriented = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    if (oriented !== bitmap) bitmap.recycle()
    return oriented
}

private const val MARK_COLOR = 0xFFFACC15.toInt()
private const val MARK_SHADOW = 0xB3000000.toInt()

// Cut lines are semi-transparent so the pixels under them stay readable.
private const val MARK_LINE_COLOR = 0x99FACC15.toInt()
private const val MARK_LINE_SHADOW = 0x4D000000

/**
 * A piece with a caption strip above the pixels (nothing of the image is covered) and dashed lines at
 * the cut positions, so the parts beyond a line are the overlap shared with the neighbouring piece.
 */
private fun drawPieceMarks(piece: Bitmap, tile: SplitTile, grid: SplitGrid, source: SplitSource): Bitmap {
    val scale = piece.width.toFloat() / tile.width
    val caption = splitTileCaption(tile, grid, source.width, source.height)
    val text = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        typeface = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
        textSize = (maxOf(piece.width, piece.height) / 45f).coerceIn(12f, 40f)
    }
    val available = piece.width * 0.96f
    val measured = text.measureText(caption)
    if (measured > available) text.textSize = (text.textSize * available / measured).coerceAtLeast(6f)
    val header = (text.textSize * 1.8f).toInt().coerceAtLeast(12)
    val out = Bitmap.createBitmap(piece.width, piece.height + header, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(out)
    canvas.drawColor(0xFF111827.toInt())
    canvas.drawBitmap(piece, 0f, header.toFloat(), null)
    val metrics = text.fontMetrics
    canvas.drawText(caption, piece.width * 0.02f, header / 2f - (metrics.ascent + metrics.descent) / 2f, text)
    val stroke = (maxOf(piece.width, piece.height) / 500f).coerceIn(1.5f, 4f)
    val lines = mutableListOf<FloatArray>()
    val top = header.toFloat()
    val bottom = top + piece.height
    val right = piece.width.toFloat()
    fun vertical(x: Int) { val px = (x - tile.left) * scale; lines += floatArrayOf(px, top, px, bottom) }
    fun horizontal(y: Int) { val py = top + (y - tile.top) * scale; lines += floatArrayOf(0f, py, right, py) }
    if (tile.cellLeft > tile.left) vertical(tile.cellLeft)
    if (tile.cellRight < tile.right) vertical(tile.cellRight)
    if (tile.cellTop > tile.top) horizontal(tile.cellTop)
    if (tile.cellBottom < tile.bottom) horizontal(tile.cellBottom)
    drawMarkLines(canvas, lines, stroke, dashed = true)
    return out
}

/** The whole image with the overlap bands tinted, the cut lines drawn and each cell numbered. */
private fun drawGridOverview(whole: Bitmap, source: SplitSource, grid: SplitGrid, tiles: List<SplitTile>): Bitmap {
    val out = whole.copy(Bitmap.Config.ARGB_8888, true) ?: throw IOException("画像を加工できません。")
    val canvas = Canvas(out)
    val sx = out.width.toFloat() / source.width
    val sy = out.height.toFloat() / source.height
    val band = Paint().apply { color = MARK_COLOR; alpha = 46 }
    val xs = splitCuts(source.width, grid.columns)
    val ys = splitCuts(source.height, grid.rows)
    val first = tiles.first()
    val padX = if (grid.columns > 1) (first.right - first.cellRight) else 0
    val padY = if (grid.rows > 1) (first.bottom - first.cellBottom) else 0
    val lines = mutableListOf<FloatArray>()
    for (i in 1 until grid.columns) {
        canvas.drawRect(RectF((xs[i] - padX) * sx, 0f, (xs[i] + padX) * sx, out.height.toFloat()), band)
        lines += floatArrayOf(xs[i] * sx, 0f, xs[i] * sx, out.height.toFloat())
    }
    for (i in 1 until grid.rows) {
        canvas.drawRect(RectF(0f, (ys[i] - padY) * sy, out.width.toFloat(), (ys[i] + padY) * sy), band)
        lines += floatArrayOf(0f, ys[i] * sy, out.width.toFloat(), ys[i] * sy)
    }
    val stroke = (maxOf(out.width, out.height) / 400f).coerceIn(2f, 8f)
    drawMarkLines(canvas, lines, stroke, dashed = false)
    val cellSide = minOf(out.width.toFloat() / grid.columns, out.height.toFloat() / grid.rows)
    val text = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        textSize = (cellSide / 6f).coerceIn(12f, 96f)
        textAlign = Paint.Align.CENTER
    }
    val badge = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = MARK_SHADOW }
    val ring = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = MARK_COLOR; style = Paint.Style.STROKE; strokeWidth = stroke }
    val metrics = text.fontMetrics
    for (tile in tiles) {
        val label = (tile.index + 1).toString()
        val radius = maxOf(text.textSize * 0.9f, text.measureText(label) / 2f + text.textSize * 0.35f)
        val cx = tile.cellLeft * sx + radius + stroke * 2
        val cy = tile.cellTop * sy + radius + stroke * 2
        canvas.drawCircle(cx, cy, radius, badge)
        canvas.drawCircle(cx, cy, radius, ring)
        canvas.drawText(label, cx, cy - (metrics.ascent + metrics.descent) / 2f, text)
    }
    return out
}

/** Translucent yellow lines over a faint dark outline: visible on light and dark images without hiding them. */
private fun drawMarkLines(canvas: Canvas, lines: List<FloatArray>, stroke: Float, dashed: Boolean) {
    if (lines.isEmpty()) return
    val effect = if (dashed) DashPathEffect(floatArrayOf(stroke * 4, stroke * 5), 0f) else null
    val shadow = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = MARK_LINE_SHADOW; strokeWidth = stroke * 1.8f; style = Paint.Style.STROKE; pathEffect = effect
    }
    val line = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = MARK_LINE_COLOR; strokeWidth = stroke; style = Paint.Style.STROKE; pathEffect = effect
    }
    val path = Path()
    for (l in lines) { path.moveTo(l[0], l[1]); path.lineTo(l[2], l[3]) }
    canvas.drawPath(path, shadow)
    canvas.drawPath(path, line)
}
