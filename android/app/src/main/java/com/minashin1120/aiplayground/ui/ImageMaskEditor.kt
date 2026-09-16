package com.minashin1120.aiplayground.ui

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas as AndroidCanvas
import android.graphics.Paint as AndroidPaint
import android.graphics.PorterDuff
import android.net.Uri
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream

/** Native equivalent of Web's GPT-Image mask marker. The saved PNG contains
 * transparent pixels except for the painted regions, which the server sends as
 * the provider's input_image_mask. */
@Composable
fun ImageMaskEditor(uri: Uri, onDismiss: () -> Unit, onSave: (ByteArray) -> Unit) {
    val context = LocalContext.current
    val bitmap by produceState<Bitmap?>(null, uri) {
        value = withContext(Dispatchers.IO) {
            runCatching { context.contentResolver.openInputStream(uri)?.use { input -> BitmapFactory.decodeStream(input) } }.getOrNull()
        }
    }
    var paths by remember(uri) { mutableStateOf<List<List<Offset>>>(emptyList()) }
    var active by remember(uri) { mutableStateOf<List<Offset>>(emptyList()) }
    var saving by remember { mutableStateOf(false) }
    var canvasSize by remember(uri) { mutableStateOf(IntSize.Zero) }
    val scope = rememberCoroutineScope()
    PlaygroundDialog(
        onDismissRequest = { if (!saving) onDismiss() },
        title = { Text("画像マスク") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("編集したい部分を指で塗りつぶしてください。透明な部分が生成対象になります。",
                    style = MaterialTheme.typography.bodySmall)
                if (bitmap == null) {
                    Box(Modifier.fillMaxWidth().height(300.dp)) { CircularProgressIndicator() }
                } else {
                    val image = bitmap!!.asImageBitmap()
                    val aspect = image.width.toFloat() / image.height.coerceAtLeast(1)
                    Canvas(Modifier.fillMaxWidth().heightIn(max = 420.dp).aspectRatio(aspect.coerceIn(.45f, 2.2f))
                        .onSizeChanged { canvasSize = it }
                        .pointerInput(uri) {
                            detectDragGestures(
                                onDragStart = { active = listOf(it) },
                                onDrag = { change, drag ->
                                    change.consume()
                                    active = active + (active.lastOrNull()?.plus(drag) ?: change.position)
                                },
                                onDragEnd = { if (active.isNotEmpty()) paths = paths + listOf(active); active = emptyList() },
                                onDragCancel = { active = emptyList() },
                            )
                        }) {
                        drawImage(image, dstSize = androidx.compose.ui.unit.IntSize(size.width.toInt(), size.height.toInt()))
                        val brush = Stroke(width = 42f, cap = androidx.compose.ui.graphics.StrokeCap.Round,
                            join = androidx.compose.ui.graphics.StrokeJoin.Round)
                        (paths + listOf(active).filter { it.isNotEmpty() }).forEach { points ->
                            if (points.isNotEmpty()) {
                                val path = Path().apply {
                                    moveTo(points.first().x, points.first().y)
                                    points.drop(1).forEach { lineTo(it.x, it.y) }
                                }
                                drawPath(path, Color(0xAAFFCC00), style = brush)
                            }
                        }
                    }
                }
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    TextButton(onClick = { paths = emptyList(); active = emptyList() }, enabled = paths.isNotEmpty()) { Text("クリア") }
                    Text("${paths.size}ストローク", style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
        },
        confirmButton = {
            TextButton(onClick = {
                val source = bitmap
                if (source != null) {
                    saving = true
                    // Canvas/Bitmap work is small after the user has already loaded
                    // the image; still perform it off the Compose frame thread.
                    val width = source.width
                    val height = source.height
                    val snapshot = paths
                    scope.launch {
                        onSave(buildMaskPng(width, height, snapshot, canvasSize.width.coerceAtLeast(1), canvasSize.height.coerceAtLeast(1)))
                        saving = false
                    }
                }
            }, enabled = bitmap != null && paths.isNotEmpty() && !saving) { Text("マスクを適用") }
        },
        dismissButton = { TextButton(onClick = { if (!saving) onDismiss() }) { Text("キャンセル") } },
    )
}

private suspend fun buildMaskPng(width: Int, height: Int, paths: List<List<Offset>>, displayWidth: Int, displayHeight: Int): ByteArray = withContext(Dispatchers.Default) {
            val out = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            val canvas = AndroidCanvas(out)
            canvas.drawColor(android.graphics.Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
            val paint = AndroidPaint(AndroidPaint.ANTI_ALIAS_FLAG).apply {
                color = android.graphics.Color.WHITE
                style = AndroidPaint.Style.STROKE
                strokeWidth = 42f * (width.toFloat() / displayWidth).coerceIn(.25f, 8f)
                strokeCap = AndroidPaint.Cap.ROUND
                strokeJoin = AndroidPaint.Join.ROUND
            }
            paths.forEach { points ->
                if (points.isEmpty()) return@forEach
                val path = android.graphics.Path().apply {
                    moveTo(points.first().x * width / displayWidth, points.first().y * height / displayHeight)
                    points.drop(1).forEach { lineTo(it.x * width / displayWidth, it.y * height / displayHeight) }
                }
                canvas.drawPath(path, paint)
            }
            ByteArrayOutputStream().use { stream -> out.compress(Bitmap.CompressFormat.PNG, 100, stream); stream.toByteArray() }
}
