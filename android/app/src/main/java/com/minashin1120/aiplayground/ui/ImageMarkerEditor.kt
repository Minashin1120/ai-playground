package com.minashin1120.aiplayground.ui

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas as AndroidCanvas
import android.graphics.Paint as AndroidPaint
import android.graphics.Rect as AndroidRect
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.calculatePan
import androidx.compose.foundation.gestures.calculateZoom
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.TransformOrigin
import androidx.compose.ui.graphics.asAndroidPath
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import kotlin.math.max
import kotlin.math.min

/** One undoable edit, replayed onto the marker layer (Web `saveMarkerHistory` snapshots). */
private sealed interface MarkerOp {
    data class StrokeOp(val points: List<MarkerPoint>, val width: Float, val color: Color) : MarkerOp
    data class MosaicOp(val rect: MarkerRect, val block: Int) : MarkerOp
    data object ClearOp : MarkerOp
}

/** Web `ctx.quadraticCurveTo` smoothing through the stroke points. */
private fun strokePath(points: List<MarkerPoint>): Path = Path().apply {
    if (points.isEmpty()) return@apply
    moveTo(points[0].x, points[0].y)
    when {
        points.size == 1 -> Unit
        points.size == 2 -> lineTo(points[1].x, points[1].y)
        else -> {
            for (i in 1 until points.size - 2) {
                val p = points[i]
                val n = points[i + 1]
                quadraticTo(p.x, p.y, (p.x + n.x) / 2, (p.y + n.y) / 2)
            }
            val secondLast = points[points.size - 2]
            val last = points.last()
            quadraticTo(secondLast.x, secondLast.y, last.x, last.y)
        }
    }
}

private fun colorOf(hex: String, alpha: Float): Color =
    Color(android.graphics.Color.parseColor(normalizeMarkerHex(hex))).copy(alpha = alpha.coerceIn(0.001f, 1f))

/** Draws [op] onto the marker layer; a mosaic samples the image with the layer on top (Web `getMosaicSourceImageData`). */
private fun drawOp(layer: Bitmap, base: Bitmap, op: MarkerOp) {
    val canvas = AndroidCanvas(layer)
    when (op) {
        is MarkerOp.StrokeOp -> {
            val paint = AndroidPaint(AndroidPaint.ANTI_ALIAS_FLAG).apply {
                color = android.graphics.Color.argb((op.color.alpha * 255).toInt(), (op.color.red * 255).toInt(),
                    (op.color.green * 255).toInt(), (op.color.blue * 255).toInt())
                strokeWidth = op.width
                strokeCap = AndroidPaint.Cap.ROUND
                strokeJoin = AndroidPaint.Join.ROUND
            }
            if (op.points.size == 1) {
                paint.style = AndroidPaint.Style.FILL
                canvas.drawCircle(op.points[0].x, op.points[0].y, op.width / 2, paint)
            } else {
                paint.style = AndroidPaint.Style.STROKE
                canvas.drawPath(strokePath(op.points).asAndroidPath(), paint)
            }
        }
        is MarkerOp.MosaicOp -> {
            val source = base.copy(Bitmap.Config.ARGB_8888, true)
            AndroidCanvas(source).drawBitmap(layer, 0f, 0f, null)
            val paint = AndroidPaint()
            mosaicCells(op.rect, op.block, layer.width, layer.height).forEach { (x, y, w, h) ->
                val cx = (x + w / 2).coerceIn(0, layer.width - 1)
                val cy = (y + h / 2).coerceIn(0, layer.height - 1)
                paint.color = source.getPixel(cx, cy) or (0xFF shl 24)
                canvas.drawRect(x.toFloat(), y.toFloat(), (x + w).toFloat(), (y + h).toFloat(), paint)
            }
            source.recycle()
        }
        MarkerOp.ClearOp -> canvas.drawColor(android.graphics.Color.TRANSPARENT, android.graphics.PorterDuff.Mode.CLEAR)
    }
}

/**
 * Web `#marker-modal` (`openMarkerModalForRow` / `saveMarkerToRow`): marker, mosaic and crop on an upload
 * row's image with two-finger zoom. "保存して反映" hands back the edited PNG at the image's own size.
 */
@Composable
internal fun ImageMarkerEditor(
    attachment: Attachment,
    loader: FileBytesLoader?,
    onDismiss: () -> Unit,
    onSave: (png: ByteArray, attachOriginal: Boolean) -> Unit,
    onError: (String) -> Unit,
) {
    val web = LocalWebPalette.current
    val density = LocalDensity.current.density
    var image by remember { mutableStateOf<Bitmap?>(null) }
    LaunchedEffect(attachment.reference) {
        val bytes = runCatching { loader?.invoke(attachment.reference, false, 40L * 1024 * 1024) }.getOrNull()
        val decoded = bytes?.let { data -> withContext(Dispatchers.Default) { decodeForEditing(data) } }
        if (decoded == null) { onError("画像が読み込めませんでした"); onDismiss() } else image = decoded
    }
    var mode by remember { mutableStateOf(MarkerMode.DRAW) }
    var brushSize by remember { mutableFloatStateOf(16f) }
    var colorHex by remember { mutableStateOf(MARKER_DEFAULT_COLOR) }
    var opacityPct by remember { mutableDoubleStateOf(60.0) }
    var attachOriginal by remember { mutableStateOf(attachment.attachOriginal) }
    var pickingColor by remember { mutableStateOf(false) }
    var saving by remember { mutableStateOf(false) }
    val history = remember { mutableStateListOf<MarkerOp>() }
    // Web keeps 40 snapshots: edits older than the last 39 can no longer be undone.
    var undoFloor by remember { mutableIntStateOf(0) }
    val push: (MarkerOp) -> Unit = { op ->
        history += op
        undoFloor = max(undoFloor, history.size - (MARKER_HISTORY_LIMIT - 1))
    }
    var cropRect by remember { mutableStateOf<MarkerRect?>(null) }
    var scale by remember { mutableFloatStateOf(1f) }
    var offset by remember { mutableStateOf(Offset.Zero) }
    val resetView = { scale = 1f; offset = Offset.Zero }
    var canvasSize by remember { mutableStateOf<Pair<Int, Int>?>(null) }

    TwModalFrame(onDismiss, maxPanelWidth = 768.dp) { phone ->
        Row(Modifier.fillMaxWidth().padding(bottom = 8.dp), verticalAlignment = Alignment.CenterVertically) {
            FaIcon(R.drawable.fa_solid_highlighter, null, size = 16.dp, tint = web.twText(Tw.yellow300))
            Text("画像編集", fontSize = 18.sp, fontWeight = FontWeight.Bold, color = web.text, modifier = Modifier.padding(start = 8.dp).weight(1f))
            Box(Modifier.size(32.dp).clip(CircleShape).clickable(role = Role.Button, onClick = onDismiss), contentAlignment = Alignment.Center) {
                FaIcon(R.drawable.fa_solid_times, "閉じる", size = 14.dp, tint = Tw.gray400)
            }
        }
        Text("マーカー・モザイク・トリミングが可能です。送信時に編集済み画像の確認を促します。", fontSize = 12.sp, color = Tw.gray400,
            modifier = Modifier.padding(bottom = 12.dp))
        val source = image
        val stageShape = RoundedCornerShape(14.dp)
        val screenH = LocalConfiguration.current.screenHeightDp.dp
        BoxWithConstraints(
            Modifier.fillMaxWidth().padding(bottom = 16.dp).clip(stageShape).background(if (web.isLight) Color(0xFFF1F5F9) else Color(0xFF080D1D))
                .border(1.dp, web.line, stageShape).padding(if (phone) 8.dp else 10.dp),
            contentAlignment = Alignment.Center,
        ) {
            val areaW = constraints.maxWidth.toFloat()
            val areaMaxH = screenH.value * density * (if (phone) 0.52f else 0.58f)
            if (source == null) {
                Box(Modifier.fillMaxWidth().height(200.dp), contentAlignment = Alignment.Center) {
                    Text("読み込み中...", fontSize = 12.sp, color = Tw.gray400)
                }
                return@BoxWithConstraints
            }
            val fit = min(min(1f, areaW / (source.width * density)), areaMaxH / (source.height * density)) * density
            val canvasW = max(1, (source.width * fit).toInt())
            val canvasH = max(1, (source.height * fit).toInt())
            val base = remember(source, canvasW, canvasH) { Bitmap.createScaledBitmap(source, canvasW, canvasH, true) }
            val layer = remember(canvasW, canvasH) { Bitmap.createBitmap(canvasW, canvasH, Bitmap.Config.ARGB_8888) }
            SideEffect { canvasSize = canvasW to canvasH }
            var layerVersion by remember { mutableIntStateOf(0) }
            // Replays the edits since the last 消去 whenever the history changes (undo included).
            LaunchedEffect(history.size, history.lastOrNull(), layer) {
                layer.eraseColor(android.graphics.Color.TRANSPARENT)
                val start = history.indexOfLast { it == MarkerOp.ClearOp } + 1
                history.drop(start).forEach { drawOp(layer, base, it) }
                layerVersion += 1
            }
            LaunchedEffect(mode, canvasW, canvasH) {
                val rect = cropRect
                if (mode == MarkerMode.CROP && (rect == null || rect.w <= 1 || rect.h <= 1)) cropRect = MarkerRect(0f, 0f, canvasW.toFloat(), canvasH.toFloat())
            }
            val livePoints = remember { mutableStateListOf<MarkerPoint>() }
            var liveMosaic by remember { mutableStateOf<MarkerRect?>(null) }
            val stageW = areaW
            val stageH = canvasH.toFloat()
            val originX = (stageW - canvasW) / 2
            fun toCanvas(p: Offset) = MarkerPoint((p.x - originX - offset.x) / scale, (p.y - offset.y) / scale)
            fun clampOffset(next: Offset): Offset {
                val (x, y) = clampMarkerOffset(scale, next.x, next.y, stageW, stageH, canvasW.toFloat(), canvasH.toFloat())
                return Offset(x, y)
            }
            val brushPx = brushSize * density
            Box(
                Modifier.fillMaxWidth().height(with(LocalDensity.current) { canvasH.toDp() })
                    .pointerInput(mode, brushPx, colorHex, opacityPct, canvasW, canvasH) {
                        awaitEachGesture {
                            val down = awaitFirstDown(requireUnconsumed = false)
                            var pinching = false
                            var cropMode = "move"
                            var cropStart: MarkerRect? = null
                            val start = toCanvas(down.position)
                            var last = start
                            when (mode) {
                                MarkerMode.DRAW -> { livePoints.clear(); appendStrokePoint(livePoints, start, brushPx) }
                                MarkerMode.MOSAIC -> liveMosaic = mosaicRectAt(start, brushPx)
                                MarkerMode.CROP -> {
                                    val rect = cropRect ?: MarkerRect(0f, 0f, canvasW.toFloat(), canvasH.toFloat())
                                    cropStart = rect
                                    cropMode = cropHitTest(start, rect, MARKER_HANDLE_RADIUS * density)
                                }
                            }
                            down.consume()
                            while (true) {
                                val event = awaitPointerEvent()
                                val pressed = event.changes.filter { it.pressed }
                                if (pressed.isEmpty()) break
                                if (pressed.size >= 2) {
                                    if (!pinching) { pinching = true; livePoints.clear(); liveMosaic = null }
                                    scale = (scale * event.calculateZoom()).coerceIn(1f, MARKER_MAX_SCALE)
                                    offset = clampOffset(offset + event.calculatePan())
                                } else if (!pinching) {
                                    val p = toCanvas(pressed[0].position)
                                    last = p
                                    when (mode) {
                                        MarkerMode.DRAW -> appendStrokePoint(livePoints, p, brushPx)
                                        MarkerMode.MOSAIC -> liveMosaic = normalizeMosaicRect(start, p)
                                        MarkerMode.CROP -> cropStart?.let { rect ->
                                            cropRect = dragCropRect(cropMode, rect, start, p, canvasW.toFloat(), canvasH.toFloat(),
                                                MARKER_MIN_CROP_SIZE * density)
                                        }
                                    }
                                }
                                event.changes.forEach { it.consume() }
                            }
                            if (!pinching) when (mode) {
                                MarkerMode.DRAW -> if (livePoints.isNotEmpty()) {
                                    push(MarkerOp.StrokeOp(livePoints.toList(), brushPx, colorOf(colorHex, (opacityPct / 100).toFloat())))
                                }
                                MarkerMode.MOSAIC -> {
                                    var rect = normalizeMosaicRect(start, last)
                                    if (rect.w < 2 || rect.h < 2) rect = mosaicRectAt(start, brushPx)
                                    push(MarkerOp.MosaicOp(rect, mosaicBlockSize(brushPx)))
                                }
                                MarkerMode.CROP -> Unit
                            }
                            livePoints.clear()
                            liveMosaic = null
                        }
                    },
            ) {
                Box(
                    Modifier.offset { androidx.compose.ui.unit.IntOffset(originX.toInt(), 0) }
                        .size(with(LocalDensity.current) { canvasW.toDp() }, with(LocalDensity.current) { canvasH.toDp() })
                        .graphicsLayer {
                            transformOrigin = TransformOrigin(0f, 0f)
                            scaleX = scale; scaleY = scale
                            translationX = offset.x; translationY = offset.y
                        },
                ) {
                    Image(base.asImageBitmap(), null, Modifier.fillMaxSize().clip(RoundedCornerShape(12.dp)), contentScale = ContentScale.FillBounds)
                    Canvas(Modifier.fillMaxSize()) {
                        layerVersion.let { drawImage(layer.asImageBitmap()) }
                        if (livePoints.isNotEmpty()) {
                            val color = colorOf(colorHex, (opacityPct / 100).toFloat())
                            if (livePoints.size == 1) drawCircle(color, brushPx / 2, Offset(livePoints[0].x, livePoints[0].y))
                            else drawPath(strokePath(livePoints), color, style = Stroke(brushPx, cap = StrokeCap.Round, join = StrokeJoin.Round))
                        }
                        val visible = history.drop(history.indexOfLast { it == MarkerOp.ClearOp } + 1)
                        drawMarkerOverlay(mode, cropRect, visible.filterIsInstance<MarkerOp.MosaicOp>().map { it.rect }, liveMosaic)
                    }
                }
            }
        }
        MarkerToolbar(
            mode = mode, onMode = { mode = it },
            size = brushSize, onSize = { brushSize = it },
            colorHex = colorHex, onColor = { colorHex = normalizeMarkerHex(it) }, onPickColor = { pickingColor = true },
            opacityPct = opacityPct, onOpacity = { opacityPct = clampMarkerOpacityPct(it) },
            attachOriginal = attachOriginal, onAttachOriginal = { attachOriginal = it },
            onResetView = resetView,
            onResetCrop = { canvasSize?.let { (w, h) -> cropRect = MarkerRect(0f, 0f, w.toFloat(), h.toFloat()) } },
            onUndo = { if (history.size > undoFloor) history.removeAt(history.lastIndex) },
            onClear = { push(MarkerOp.ClearOp) },
            saving = saving,
            onSave = save@{
                val original = image ?: return@save
                val edited = canvasSize ?: return@save
                if (saving) return@save
                saving = true
                val ops = history.toList()
                val crop = cropRect
                onSaveRequest(original, ops, crop, edited) { png ->
                    saving = false
                    if (png == null) onError("編集画像の生成に失敗しました") else onSave(png, attachOriginal)
                }
            },
        )
    }
    if (pickingColor) ColorPickerDialog(colorHex, onDismiss = { pickingColor = false }) { picked ->
        colorHex = normalizeMarkerHex(picked)
        pickingColor = false
    }
}

private fun onSaveRequest(
    original: Bitmap, ops: List<MarkerOp>, crop: MarkerRect?, canvasSize: Pair<Int, Int>, done: (ByteArray?) -> Unit,
) {
    // Rendering runs on a plain thread so the dialog stays responsive while a large image is encoded.
    Thread {
        val result = runCatching { renderEdited(original, ops, crop, canvasSize) }.getOrNull()
        android.os.Handler(android.os.Looper.getMainLooper()).post { done(result) }
    }.start()
}

/**
 * Web `saveMarkerToRow`: the image at its own size with the marker layer scaled over it, then the crop.
 * The layer is rebuilt at the size it was edited at from the recorded edits.
 */
private fun renderEdited(original: Bitmap, ops: List<MarkerOp>, crop: MarkerRect?, canvasSize: Pair<Int, Int>): ByteArray {
    val (cw, ch) = canvasSize
    val base = Bitmap.createScaledBitmap(original, cw, ch, true)
    val layer = Bitmap.createBitmap(cw, ch, Bitmap.Config.ARGB_8888)
    val start = ops.indexOfLast { it == MarkerOp.ClearOp } + 1
    ops.drop(start).forEach { drawOp(layer, base, it) }
    var out = original.copy(Bitmap.Config.ARGB_8888, true)
    val paint = AndroidPaint(AndroidPaint.FILTER_BITMAP_FLAG)
    AndroidCanvas(out).drawBitmap(layer, null, AndroidRect(0, 0, out.width, out.height), paint)
    if (crop != null && !isFullCrop(crop, cw.toFloat(), ch.toFloat())) {
        val (cx, cy, w, h) = cropToImage(crop, cw.toFloat(), ch.toFloat(), out.width, out.height)
        out = Bitmap.createBitmap(out, cx, cy, w, h)
    }
    return ByteArrayOutputStream().use { stream ->
        out.compress(Bitmap.CompressFormat.PNG, 100, stream)
        stream.toByteArray()
    }
}

/** Decodes at most 4096px on the long side so a large photo cannot exhaust memory. */
private fun decodeForEditing(bytes: ByteArray): Bitmap? {
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
    if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null
    var sample = 1
    while (max(bounds.outWidth, bounds.outHeight) / sample > 4096) sample *= 2
    return BitmapFactory.decodeByteArray(bytes, 0, bytes.size, BitmapFactory.Options().apply { inSampleSize = sample })
}

/** Web `renderCropOverlay`: the dimmed outside of the crop, its frame, and the mosaic areas in mosaic mode. */
private fun DrawScope.drawMarkerOverlay(mode: MarkerMode, crop: MarkerRect?, mosaics: List<MarkerRect>, preview: MarkerRect?) {
    fun frame(rect: MarkerRect, stroke: Color, fill: Color? = null, dashed: Boolean = false) {
        val topLeft = Offset(max(0f, rect.x), max(0f, rect.y))
        val area = Size(max(1f, rect.w), max(1f, rect.h))
        if (fill != null) drawRect(fill, topLeft, area)
        drawRect(stroke, topLeft, area, style = Stroke(2.dp.toPx(),
            pathEffect = if (dashed) PathEffect.dashPathEffect(floatArrayOf(6.dp.toPx(), 4.dp.toPx())) else null))
    }
    if (crop != null && (mode == MarkerMode.CROP || !isFullCrop(crop, size.width, size.height))) {
        val dim = Color.Black.copy(alpha = 0.35f)
        drawRect(dim, Offset.Zero, Size(size.width, crop.y))
        drawRect(dim, Offset(0f, crop.bottom), Size(size.width, size.height - crop.bottom))
        drawRect(dim, Offset(0f, crop.y), Size(crop.x, crop.h))
        drawRect(dim, Offset(crop.right, crop.y), Size(size.width - crop.right, crop.h))
        frame(crop, Color(250, 204, 21).copy(alpha = if (mode == MarkerMode.CROP) 0.9f else 0.4f))
    }
    if (mode != MarkerMode.MOSAIC) return
    mosaics.forEach { frame(it, Color(250, 204, 21).copy(alpha = 0.9f), Color(250, 204, 21).copy(alpha = 0.10f)) }
    preview?.let { frame(it, Color(56, 189, 248).copy(alpha = 0.95f), Color(56, 189, 248).copy(alpha = 0.14f), dashed = true) }
}

/** `#marker-toolbar`. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun MarkerToolbar(
    mode: MarkerMode, onMode: (MarkerMode) -> Unit,
    size: Float, onSize: (Float) -> Unit,
    colorHex: String, onColor: (String) -> Unit, onPickColor: () -> Unit,
    opacityPct: Double, onOpacity: (Double) -> Unit,
    attachOriginal: Boolean, onAttachOriginal: (Boolean) -> Unit,
    onResetView: () -> Unit, onResetCrop: () -> Unit, onUndo: () -> Unit, onClear: () -> Unit,
    saving: Boolean, onSave: () -> Unit,
) {
    val web = LocalWebPalette.current
    val yellow = Color(0xFFFACC15)
    FlowRow(
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
        itemVerticalAlignment = Alignment.CenterVertically,
    ) {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            listOf(MarkerMode.DRAW to "マーカー", MarkerMode.MOSAIC to "モザイク", MarkerMode.CROP to "トリミング").forEach { (tool, label) ->
                val active = tool == mode
                val shape = RoundedCornerShape(4.dp)
                Text(label, fontSize = 11.sp, color = if (active) Color(0xFFFDE68A) else web.twText(Color(0xFFE5E7EB)),
                    modifier = Modifier.clip(shape)
                        .background(if (active) Color(250, 204, 21).copy(alpha = 0.2f) else web.twBg(Color(55, 65, 81), 0.85f))
                        .border(1.dp, if (active) Color(250, 204, 21).copy(alpha = 0.5f) else Color(107, 114, 128).copy(alpha = 0.5f), shape)
                        .clickable(role = Role.Button) { onMode(tool) }.padding(horizontal = 12.dp, vertical = 4.dp))
            }
        }
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("太さ", fontSize = 12.sp, color = Tw.gray400)
            YellowSlider(size, 1f..40f, 1f, Modifier.width(128.dp)) { onSize(it) }
        }
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
            Text("色", fontSize = 12.sp, color = Tw.gray400)
            MARKER_COLORS.forEach { (hex, title) ->
                val active = normalizeMarkerHex(hex) == colorHex
                Box(
                    Modifier.padding(2.dp).size(18.dp).clip(CircleShape)
                        .background(Color(android.graphics.Color.parseColor(hex)))
                        .border(2.dp, Color.White.copy(alpha = if (active) 0.9f else 0.3f), CircleShape)
                        .clickable(role = Role.Button, onClickLabel = title) { onColor(hex) },
                )
            }
        }
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
            Text("任意", fontSize = 12.sp, color = Tw.gray400)
            val shape = RoundedCornerShape(6.dp)
            Box(Modifier.size(24.dp, 20.dp).clip(shape).background(Color(android.graphics.Color.parseColor(colorHex)))
                .border(1.dp, Color(148, 163, 184).copy(alpha = 0.6f), shape).clickable(role = Role.Button, onClickLabel = "マーカー色", onClick = onPickColor))
        }
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("透明度", fontSize = 12.sp, color = Tw.gray400)
            YellowSlider(opacityPct.toFloat(), 0.1f..100f, 0.1f, Modifier.width(90.dp)) { onOpacity(it.toDouble()) }
            OpacityNumber(opacityPct, onOpacity)
            Text("${formatMarkerOpacityPct(opacityPct)}%", fontSize = 12.sp, color = web.twText(Tw.slate300), modifier = Modifier.widthIn(min = 34.dp))
        }
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp),
            modifier = Modifier.clickable(role = Role.Checkbox) { onAttachOriginal(!attachOriginal) }) {
            WebCheckbox(attachOriginal, onAttachOriginal, accent = yellow, size = 12.dp)
            Text("元画像も添付", fontSize = 10.sp, color = Tw.gray400)
        }
        Text(MARKER_TOOL_HINTS[mode].orEmpty(), fontSize = 10.sp, color = Tw.gray500)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
            MarkerActionButton("位置/倍率リセット", onClick = onResetView)
            if (mode == MarkerMode.CROP) MarkerActionButton("トリミングリセット", onClick = onResetCrop)
            MarkerActionButton("戻す", onClick = onUndo)
            MarkerActionButton("消去", onClick = onClear)
            val shape = RoundedCornerShape(4.dp)
            Text("保存して反映", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = Color.Black,
                modifier = Modifier.clip(shape).background(Color(0xFFEAB308)).clickable(enabled = !saving, role = Role.Button, onClick = onSave)
                    .padding(horizontal = 16.dp, vertical = 4.dp))
        }
    }
}

@Composable
private fun MarkerActionButton(label: String, onClick: () -> Unit) {
    val shape = RoundedCornerShape(4.dp)
    Text(label, fontSize = 12.sp, color = Color.White,
        modifier = Modifier.clip(shape).background(Tw.gray700).clickable(role = Role.Button, onClick = onClick)
            .padding(horizontal = 12.dp, vertical = 4.dp))
}

/** `#marker-toolbar input[type=range] { accent-color: #facc15 }`. */
@Composable
private fun YellowSlider(value: Float, range: ClosedFloatingPointRange<Float>, step: Float, modifier: Modifier, onChange: (Float) -> Unit) {
    val yellow = Color(0xFFFACC15)
    Slider(
        value.coerceIn(range), { onChange((Math.round(it / step) * step).coerceIn(range)) }, valueRange = range,
        colors = SliderDefaults.colors(thumbColor = yellow, activeTrackColor = yellow, inactiveTrackColor = LocalWebPalette.current.twBg(Tw.gray600)),
        modifier = modifier.height(24.dp),
    )
}

/** `#marker-opacity-number` (0.1–100, step 0.1); applied when the text is a number. */
@Composable
private fun OpacityNumber(value: Double, onChange: (Double) -> Unit) {
    val web = LocalWebPalette.current
    var text by remember { mutableStateOf(formatMarkerOpacityPct(value)) }
    LaunchedEffect(value) { if (text.toDoubleOrNull() != value) text = formatMarkerOpacityPct(value) }
    val shape = RoundedCornerShape(4.dp)
    BasicTextField(
        text, { next -> text = next; next.toDoubleOrNull()?.let(onChange) },
        singleLine = true,
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
        textStyle = TextStyle(fontSize = 12.sp, color = web.text),
        cursorBrush = SolidColor(web.text),
        modifier = Modifier.width(56.dp).clip(shape).background(web.twBg(Tw.gray900, 0.6f))
            .border(1.dp, web.twBorder(Tw.gray600), shape).padding(horizontal = 6.dp, vertical = 3.dp),
    )
}

/** A Web Tailwind modal panel (`bg-gray-800 rounded-lg p-6 m-4`, `p-3 m-2` on phones). */
@Composable
private fun TwModalFrame(onDismiss: () -> Unit, maxPanelWidth: androidx.compose.ui.unit.Dp, content: @Composable ColumnScope.(Boolean) -> Unit) {
    val web = LocalWebPalette.current
    WebOverlayModal(onDismiss, grayOverlay(), 4.dp) { phone ->
        Column(
            Modifier.fillMaxSize().verticalScroll(rememberScrollState()),
            verticalArrangement = if (phone) Arrangement.Top else Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            val shape = RoundedCornerShape(8.dp)
            Column(
                Modifier.padding(if (phone) 8.dp else 16.dp).widthIn(max = maxPanelWidth).fillMaxWidth().clip(shape)
                    .background(web.twBg(Tw.gray800)).border(1.dp, web.twBorder(Tw.gray700), shape).padding(if (phone) 12.dp else 24.dp),
            ) { content(phone) }
        }
    }
}
