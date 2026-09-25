package com.minashin1120.aiplayground.ui

import android.os.Build
import android.view.WindowManager
import androidx.annotation.DrawableRes
import androidx.compose.animation.core.Animatable
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.min
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.compose.ui.window.DialogWindowProvider

/**
 * Web modal (`.modal-overlay` + `.settings-modal-panel`): a blurred, tinted overlay and a rounded
 * gradient panel with the icon/title/subtitle header, a scrolling body and a right-aligned footer.
 * Phones (< 768dp) inset the panel by 8dp with 18dp corners; wider screens center it up to
 * [panelMaxWidth] with 22dp corners.
 */
@Composable
internal fun PlaygroundDialog(
    onDismissRequest: () -> Unit,
    title: @Composable () -> Unit,
    text: @Composable () -> Unit,
    confirmButton: @Composable () -> Unit,
    dismissButton: @Composable (() -> Unit)? = null,
    @DrawableRes icon: Int? = null,
    subtitle: String? = null,
    panelMaxWidth: Dp = 920.dp,
    fillHeight: Boolean = false,
) {
    Dialog(onDismissRequest, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        WebModalWindow()
        Box(Modifier.fillMaxSize()) {
        WebModalScrim()
        BoxWithConstraints(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            val phone = maxWidth < PlaygroundDimens.breakpoint
            val panelModifier = if (phone) {
                Modifier.safeDrawingPadding().padding(8.dp).fillMaxWidth()
                    .then(if (fillHeight) Modifier.fillMaxHeight() else Modifier)
            } else {
                val height = min(maxHeight * 0.9f, 880.dp)
                Modifier.imePadding().padding(16.dp).widthIn(max = panelMaxWidth).fillMaxWidth()
                    .then(if (fillHeight) Modifier.height(height) else Modifier.heightIn(max = height))
            }
            ModalPanelMotion(fullScreen = phone, onDismissRequest = onDismissRequest) {
                WebModalPanel(panelModifier, phone) {
                    WebModalHeader(title = title, onClose = onDismissRequest, phone = phone, icon = icon, subtitle = subtitle)
                    Box(
                        Modifier.weight(1f, fill = fillHeight).fillMaxWidth()
                            .padding(start = if (phone) 12.dp else 18.dp, end = if (phone) 12.dp else 18.dp, top = if (phone) 12.dp else 16.dp, bottom = if (phone) 14.dp else 18.dp),
                    ) { text() }
                    WebModalFooter(phone) {
                        dismissButton?.invoke()
                        confirmButton()
                    }
                }
            }
        }
        }
    }
}

/** `.settings-modal-panel` surface: vertical gradient, hairline border, rounded corners. */
@Composable
internal fun WebModalPanel(modifier: Modifier, phone: Boolean, content: @Composable ColumnScope.() -> Unit) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(if (phone) 18.dp else 22.dp)
    val background = if (web.isLight) {
        Brush.verticalGradient(listOf(Color.White, Color(0xFFF7F9FC)))
    } else {
        Brush.verticalGradient(listOf(Color(10, 14, 28).copy(alpha = 0.98f), Color(6, 9, 18).copy(alpha = 0.99f)))
    }
    Column(
        modifier
            .clip(shape)
            .background(background)
            .border(1.dp, if (web.isLight) Color(15, 23, 42).copy(alpha = 0.10f) else Color.White.copy(alpha = 0.08f), shape),
        content = content,
    )
}

/** Web overlay tint (`rgba(3,7,16,.72)`, light `rgba(15,23,42,.35)` unless [color] is given) fading with the modal (320ms). */
@Composable
internal fun WebModalScrim(color: Color? = null) {
    val web = LocalWebPalette.current
    val visible = LocalModalVisible.current
    val reduce = LocalReduceMotion.current
    val alpha = remember { Animatable(0f) }
    LaunchedEffect(visible) {
        alpha.animateTo(if (visible) 1f else 0f, motionTween(reduce, 320, PlaygroundMotion.WebStandard))
    }
    val tint = color ?: if (web.isLight) Color(15, 23, 42).copy(alpha = 0.35f) else Color(3, 7, 16).copy(alpha = 0.72f)
    Box(Modifier.fillMaxSize().background(tint.copy(alpha = tint.alpha * alpha.value)))
}

/**
 * Replaces the platform dialog dim with the Web overlay and, where the device supports cross-window
 * blur (Android 12+), blurs what is behind like `backdrop-filter: blur(10px)`.
 */
@Composable
internal fun WebModalWindow(blur: Dp = 10.dp) {
    val view = LocalView.current
    val blurPx = with(LocalDensity.current) { blur.roundToPx() }
    DisposableEffect(view) {
        val window = (view.parent as? DialogWindowProvider)?.window
        window?.setDimAmount(0f)
        if (window != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            runCatching {
                window.addFlags(WindowManager.LayoutParams.FLAG_BLUR_BEHIND)
                window.attributes = window.attributes.apply { blurBehindRadius = blurPx }
            }
        }
        onDispose { }
    }
}
