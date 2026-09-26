package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * The picker behind the Web `<input type="color">`: a saturation/value square and a hue bar.
 * Returns `#rrggbb` in lower case like the Web input.
 */
@Composable
internal fun ColorPickerDialog(initial: String, onDismiss: () -> Unit, onPick: (String) -> Unit) {
    val hsv = remember(initial) {
        FloatArray(3).also { android.graphics.Color.colorToHSV(android.graphics.Color.parseColor(initial), it) }
    }
    var hue by remember(initial) { mutableFloatStateOf(hsv[0]) }
    var sat by remember(initial) { mutableFloatStateOf(hsv[1]) }
    var value by remember(initial) { mutableFloatStateOf(hsv[2]) }
    val hex = String.format("#%06x", android.graphics.Color.HSVToColor(floatArrayOf(hue, sat, value)) and 0xFFFFFF)
    PlaygroundDialog(
        onDismissRequest = onDismiss,
        title = { Text("テーマ") },
        panelMaxWidth = 360.dp,
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                val pure = Color.hsv(hue, 1f, 1f)
                Canvas(
                    Modifier.fillMaxWidth().aspectRatio(1.4f).clip(RoundedCornerShape(12.dp))
                        .pointerInput(Unit) {
                            fun set(o: Offset) {
                                sat = (o.x / size.width).coerceIn(0f, 1f)
                                value = 1f - (o.y / size.height).coerceIn(0f, 1f)
                            }
                            detectTapGestures { set(it) }
                        }
                        .pointerInput(Unit) {
                            detectDragGestures { change, _ ->
                                sat = (change.position.x / size.width).coerceIn(0f, 1f)
                                value = 1f - (change.position.y / size.height).coerceIn(0f, 1f)
                            }
                        },
                ) {
                    drawRect(Brush.horizontalGradient(listOf(Color.White, pure)))
                    drawRect(Brush.verticalGradient(listOf(Color.Transparent, Color.Black)))
                    val point = Offset(sat * size.width, (1f - value) * size.height)
                    drawCircle(Color.White, radius = 8.dp.toPx(), center = point, style = Stroke(2.dp.toPx()))
                }
                Canvas(
                    Modifier.fillMaxWidth().height(22.dp).clip(RoundedCornerShape(11.dp))
                        .pointerInput(Unit) { detectTapGestures { hue = (it.x / size.width).coerceIn(0f, 1f) * 360f } }
                        .pointerInput(Unit) { detectDragGestures { change, _ -> hue = (change.position.x / size.width).coerceIn(0f, 1f) * 360f } },
                ) {
                    drawRect(Brush.horizontalGradient((0..6).map { Color.hsv(it * 60f % 360f, 1f, 1f) }))
                    val x = hue / 360f * size.width
                    drawCircle(Color.White, radius = 9.dp.toPx(), center = Offset(x, size.height / 2), style = Stroke(2.dp.toPx()))
                }
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    Box(Modifier.size(32.dp).clip(RoundedCornerShape(8.dp)).background(parseHexColor(hex))
                        .border(1.dp, LocalWebPalette.current.line, RoundedCornerShape(8.dp)))
                    Text(hex, fontSize = 13.sp, color = LocalWebPalette.current.text)
                }
            }
        },
        dismissButton = {
            WebButton(onClick = onDismiss, variant = WebButtonVariant.Ghost) { Text("キャンセル") }
        },
        confirmButton = {
            WebButton(onClick = { onPick(hex) }, variant = WebButtonVariant.Primary) { Text("OK") }
        },
    )
}
