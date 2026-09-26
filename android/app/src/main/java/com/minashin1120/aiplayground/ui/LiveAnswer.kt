package com.minashin1120.aiplayground.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.ui.semantics.Role
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.StatusCard
import com.minashin1120.aiplayground.data.CardKind
import com.minashin1120.aiplayground.data.pendingSkeletonKind
import kotlinx.coroutines.delay

/** `.skeleton-line` fill: a moving highlight (`skeleton-shimmer`, 1.55s). */
@Composable
private fun skeletonBrush(): Brush {
    val web = LocalWebPalette.current
    val reduce = LocalReduceMotion.current
    val shift = if (reduce) 0.5f else rememberInfiniteTransition(label = "skeleton").animateFloat(
        1f, -1f, infiniteRepeatable(tween(1550, easing = LinearEasing), RepeatMode.Restart), label = "skeleton shift",
    ).value
    val base = Color(148, 163, 184)
    return object : androidx.compose.ui.graphics.ShaderBrush() {
        override fun createShader(size: androidx.compose.ui.geometry.Size): androidx.compose.ui.graphics.Shader {
            val width = size.width * 2.2f
            val start = shift * width
            return androidx.compose.ui.graphics.LinearGradientShader(
                from = androidx.compose.ui.geometry.Offset(start, 0f),
                to = androidx.compose.ui.geometry.Offset(start + width, 0f),
                colors = listOf(base.copy(alpha = 0.10f), base.copy(alpha = 0.18f), web.theme.rgb(0.28f), base.copy(alpha = 0.18f), base.copy(alpha = 0.10f)),
                colorStops = listOf(0f, 0.35f, 0.5f, 0.65f, 1f),
                tileMode = androidx.compose.ui.graphics.TileMode.Clamp,
            )
        }
    }
}

/**
 * Web `buildPendingSkeletonHtml`: a model-aware placeholder (text lines, image, video, audio wave or
 * code window) with the `.skeleton-status` line and its sub line.
 */
@Composable
internal fun PendingSkeleton(model: String, status: String, sub: String) {
    val kind = remember(model) { pendingSkeletonKind(model) }
    val brush = skeletonBrush()
    Column(Modifier.widthIn(min = 280.dp).padding(vertical = 2.dp)) {
        when (kind) {
            "image" -> SkeletonMedia(brush, 280.dp, 1f, R.drawable.fa_solid_image, 44.dp, 16.dp, progress = false)
            "video" -> SkeletonMedia(brush, 360.dp, 16f / 9f, R.drawable.fa_solid_play, 48.dp, 15.dp, progress = true)
            "audio" -> SkeletonAudio(brush)
            "code" -> SkeletonCode(brush)
            else -> SkeletonLines(brush, listOf(0.92f, 0.78f, 0.86f, 0.64f, 0.48f), 11.5.dp, 8.8.dp, 420.dp, CircleShape)
        }
        val statusColor = when (kind) {
            "image" -> Color(244, 114, 182).copy(alpha = 0.8f)
            "video" -> Color(96, 165, 250).copy(alpha = 0.85f)
            "audio" -> Color(248, 113, 113).copy(alpha = 0.8f)
            "code" -> Color(52, 211, 153).copy(alpha = 0.8f)
            else -> if (LocalWebPalette.current.isLight) LocalWebPalette.current.muted else Color(203, 213, 225).copy(alpha = 0.72f)
        }
        Text(status, fontSize = 11.2.sp, lineHeight = 15.7.sp, letterSpacing = 0.11.sp, color = statusColor, modifier = Modifier.padding(top = 12.dp))
        if (sub.isNotEmpty()) Text(sub, fontSize = 10.4.sp, lineHeight = 14.6.sp, color = Color(148, 163, 184).copy(alpha = 0.65f),
            modifier = Modifier.padding(top = 3.2.dp))
    }
}

@Composable
private fun SkeletonLines(brush: Brush, widths: List<Float>, height: Dp, gap: Dp, maxWidth: Dp, shape: androidx.compose.ui.graphics.Shape) {
    Column(Modifier.widthIn(max = maxWidth).fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(gap)) {
        widths.forEach { width -> Box(Modifier.fillMaxWidth(width).height(height).clip(shape).background(brush)) }
    }
}

@Composable
private fun SkeletonMedia(brush: Brush, maxWidth: Dp, ratio: Float, icon: Int, iconSize: Dp, glyph: Dp, progress: Boolean) {
    val shape = RoundedCornerShape(14.dp)
    Box(
        Modifier.widthIn(max = maxWidth).fillMaxWidth().aspectRatio(ratio).clip(shape).background(brush)
            .border(1.dp, Color(148, 163, 184).copy(alpha = 0.14f), shape),
        contentAlignment = Alignment.Center,
    ) {
        Box(Modifier.size(iconSize).clip(CircleShape).background(Color(15, 23, 42).copy(alpha = 0.45f)), contentAlignment = Alignment.Center) {
            FaIcon(icon, null, size = glyph, tint = Color(226, 232, 240).copy(alpha = 0.55f))
        }
        if (progress) Box(Modifier.align(Alignment.BottomCenter).padding(12.dp).fillMaxWidth().height(4.dp).clip(CircleShape)
            .alpha(0.85f).background(brush))
    }
}

@Composable
private fun SkeletonAudio(brush: Brush) {
    val web = LocalWebPalette.current
    val reduce = LocalReduceMotion.current
    val transition = rememberInfiniteTransition(label = "wave")
    Row(Modifier.widthIn(max = 320.dp).fillMaxWidth().padding(horizontal = 2.4.dp, vertical = 5.6.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(13.6.dp)) {
        Box(Modifier.size(44.dp).clip(CircleShape).background(brush).border(1.dp, Color(148, 163, 184).copy(alpha = 0.16f), CircleShape),
            contentAlignment = Alignment.Center) {
            FaIcon(R.drawable.fa_solid_volume_up, null, size = 15.dp, tint = Color(226, 232, 240).copy(alpha = 0.55f))
        }
        Row(Modifier.height(36.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
            listOf(0.55f, 0.8f, 1f, 0.7f, 0.9f, 0.6f, 0.85f, 0.5f).forEachIndexed { index, height ->
                val scale = if (reduce) 1f else transition.animateFloat(
                    0.45f, 1f, infiniteRepeatable(tween(550, delayMillis = 0), RepeatMode.Reverse,
                        initialStartOffset = androidx.compose.animation.core.StartOffset(index * 80)), label = "bar",
                ).value
                Box(Modifier.width(6.dp).fillMaxHeight(height).graphicsLayer { scaleY = scale; alpha = 0.55f + 0.45f * scale }
                    .clip(CircleShape).background(Brush.verticalGradient(listOf(web.theme.rgb(0.55f), Color(148, 163, 184).copy(alpha = 0.22f)))))
            }
        }
    }
}

@Composable
private fun SkeletonCode(brush: Brush) {
    val shape = RoundedCornerShape(12.dp)
    Column(Modifier.widthIn(max = 380.dp).fillMaxWidth().clip(shape).background(Color(8, 12, 24).copy(alpha = 0.35f))
        .border(1.dp, Color(148, 163, 184).copy(alpha = 0.14f), shape)) {
        Row(
            Modifier.fillMaxWidth().background(Color(15, 23, 42).copy(alpha = 0.35f)).padding(horizontal = 12.dp, vertical = 8.8.dp),
            verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.4.dp),
        ) {
            listOf(Color(248, 113, 113), Color(251, 191, 36), Color(74, 222, 128)).forEach {
                Box(Modifier.size(8.dp).alpha(0.7f).clip(CircleShape).background(it.copy(alpha = 0.55f)))
            }
            Box(Modifier.padding(start = 5.6.dp).width(72.dp).height(8.dp).clip(CircleShape).background(brush))
        }
        Box(Modifier.padding(start = 14.4.dp, end = 14.4.dp, top = 13.6.dp, bottom = 16.dp)) {
            SkeletonLines(brush, listOf(0.72f, 0.88f, 0.54f, 0.76f, 0.41f), 9.3.dp, 8.dp, 380.dp, RoundedCornerShape(4.dp))
        }
    }
}

/** `.search-box`: "Searching web..." (pulsing) and then "Search complete" for two seconds. */
@Composable
internal fun LiveSearchBox(state: String) {
    val web = LocalWebPalette.current
    val reduce = LocalReduceMotion.current
    val pulse = if (state == "searching" && !reduce) rememberInfiniteTransition(label = "search pulse").animateFloat(
        1f, 0.5f, infiniteRepeatable(tween(1000), RepeatMode.Reverse), label = "search alpha",
    ).value else 1f
    val shape = RoundedCornerShape(11.2.dp)
    Row(
        Modifier.padding(bottom = 8.dp).fillMaxWidth().alpha(pulse).clip(shape)
            .background(if (web.isLight) Color.White else Color(8, 14, 28).copy(alpha = 0.8f)).border(1.dp, web.line, shape)
            .padding(horizontal = 12.8.dp, vertical = 8.8.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        if (state == "done") FaIcon(R.drawable.fa_solid_check_circle, null, size = 12.dp, tint = Tw.green400, fixedWidth = true)
        else FaIcon(R.drawable.fa_solid_globe, null, size = 12.dp, tint = Color(0xFF94A3B8), fixedWidth = true)
        Text(if (state == "done") "Search complete" else "Searching web...", fontSize = 12.sp, lineHeight = 16.sp, color = Color(0xFF94A3B8))
    }
}

/** `.image-analysis-box`. */
@Composable
internal fun LiveImageAnalysis(text: String) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(4.dp)
    Column(
        Modifier.padding(bottom = 8.dp).fillMaxWidth().clip(shape).background(web.twBg(Tw.blue900, 0.2f))
            .border(1.dp, web.twBorder(Tw.blue500, 0.3f), shape).padding(8.dp),
    ) {
        Row(Modifier.padding(bottom = 4.dp), verticalAlignment = Alignment.CenterVertically) {
            FaIcon(R.drawable.fa_solid_image, null, size = 10.dp, tint = web.twText(Tw.blue300), modifier = Modifier.padding(end = 4.dp))
            Text("Image Analysis", fontSize = 10.sp, lineHeight = 15.sp, fontWeight = FontWeight.Medium, color = web.twText(Tw.blue300))
        }
        Text(text, fontSize = 11.sp, lineHeight = 16.sp, color = web.twText(Tw.gray300))
    }
}

/**
 * `.python-box`: "Python Execution" (collapsed), expand, copy code, copy output; Code and Output sections.
 * [detail] is the box in the "Python 実行結果" modal (`buildPythonExecDetailBoxHtml`): open, with the Coding
 * target and download buttons instead of expand.
 */
@Composable
internal fun PythonExecutionBox(card: StatusCard, label: String = "Python Execution", detail: Boolean = false) {
    val colors = markdownColors()
    val actions = LocalMarkdownCodeActions.current
    var collapsed by remember(card.id) { mutableStateOf(!detail) }
    val clipboard = LocalClipboardManager.current
    val shape = RoundedCornerShape(16.dp)
    Column(
        Modifier.padding(top = 9.6.dp, bottom = 16.dp).fillMaxWidth().clip(shape).background(colors.codeWrapper)
            .border(1.dp, colors.codeWrapperBorder, shape),
    ) {
        Row(
            Modifier.fillMaxWidth().heightIn(min = 28.dp).background(colors.codeHeader).background(Color(245, 158, 11).copy(alpha = 0.12f))
                .padding(start = 10.4.dp, end = 6.4.dp, top = 3.2.dp, bottom = 3.2.dp),
            verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.4.dp),
        ) {
            Row(Modifier.weight(1f), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(5.6.dp)) {
                FaIcon(R.drawable.fa_solid_terminal, null, size = 10.88.dp, tint = Color(0xFFFDE68A))
                Text(label, color = Color(0xFFFDE68A), fontSize = 10.88.sp, lineHeight = 13.06.sp, fontWeight = FontWeight.SemiBold)
            }
            val actionColors = colors.copy(codeHeaderText = Color(0xFFCBD5F5))
            if (detail) {
                if (actions.onCodingTarget != null) {
                    val active = actions.selectedCodingKey == com.minashin1120.aiplayground.data.codingTargetKey("python", card.code)
                    CodeActionButton(if (active) R.drawable.fa_solid_thumbtack else R.drawable.fa_solid_quote_right,
                        if (active) "編集対象に設定済み" else "Coding Modeの編集対象に指定", actionColors) { actions.onCodingTarget.invoke(card.code, "python") }
                }
                CodeActionButton(R.drawable.fa_solid_download, "コードをダウンロード", actionColors) { actions.onDownload(card.code, "python") }
            } else CodeActionButton(if (collapsed) R.drawable.fa_solid_chevron_down else R.drawable.fa_solid_chevron_up,
                if (collapsed) "展開" else "折りたたむ", actionColors) { collapsed = !collapsed }
            CodeActionButton(R.drawable.fa_solid_copy, "コードをコピー", actionColors) { clipboard.setText(AnnotatedString(card.code)) }
            CodeActionButton(R.drawable.fa_solid_align_left, "出力をコピー", actionColors) { clipboard.setText(AnnotatedString(card.output)) }
        }
        if (!collapsed) Column(Modifier.padding(horizontal = 11.2.dp, vertical = 8.8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            PythonSection("Code", card.code, colors)
            PythonSection("Output", card.output, colors)
        }
    }
}

@Composable
private fun PythonSection(label: String, text: String, colors: MarkdownColors) {
    Column(verticalArrangement = Arrangement.spacedBy(5.6.dp)) {
        Text(label.uppercase(), fontSize = 11.2.sp, letterSpacing = 0.45.sp, color = Color(0xFFA3B3C5))
        val shape = RoundedCornerShape(8.dp)
        SelectionContainer {
            Text(
                text, fontFamily = WebFonts.mono, fontSize = 12.6.sp, lineHeight = 19.sp, color = colors.codeText, softWrap = false,
                modifier = Modifier.fillMaxWidth().clip(shape).background(Color(0xFF060D1D)).border(1.dp, Color(148, 163, 184).copy(alpha = 0.2f), shape)
                    .horizontalScroll(rememberScrollState()).padding(12.6.dp),
            )
        }
    }
}

/**
 * Web `#global-progress-spinner`: bottom-right pill that appears 400ms after a tracked request starts
 * and shows the newest request's label (`progress_spinner.js`). It never takes touches.
 */
@Composable
internal fun GlobalProgressSpinner(label: String?, modifier: Modifier = Modifier) {
    var shown by remember { mutableStateOf(false) }
    var text by remember { mutableStateOf("通信中...") }
    LaunchedEffect(label != null) {
        if (label != null) { delay(400); shown = true } else shown = false
    }
    if (label != null) text = label
    val reduce = LocalReduceMotion.current
    AnimatedVisibility(
        shown, modifier = modifier,
        enter = if (reduce) fadeIn(tween(0)) else fadeIn(tween(180)) + slideInVertically(tween(180)) { 12 },
        exit = if (reduce) fadeOut(tween(0)) else fadeOut(tween(180)) + slideOutVertically(tween(180)) { 12 },
    ) {
        Row(
            Modifier.clip(CircleShape).background(Color(2, 6, 23).copy(alpha = 0.92f))
                .border(1.dp, Color(148, 163, 184).copy(alpha = 0.35f), CircleShape).padding(horizontal = 12.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            CircularProgressIndicator(Modifier.size(16.dp), color = Color(0xFF5EEAD4), strokeWidth = 2.dp,
                trackColor = Color(148, 163, 184).copy(alpha = 0.35f))
            Text(text, fontSize = 12.sp, fontWeight = FontWeight.SemiBold, letterSpacing = 0.24.sp, color = Color(0xFFE2E8F0))
        }
    }
}

/**
 * A temporary `/settings` bubble (Web `runAiSettingsCommand`): the user's command, the pending
 * "設定リクエストを確認しています..." skeleton, or the result with rows that open the matching settings.
 */
@Composable
internal fun SettingsBubbleView(
    bubble: com.minashin1120.aiplayground.data.SettingsBubble,
    currentModel: String,
    onFile: (String) -> Unit,
    loader: FileBytesLoader?,
    onJump: (String) -> Unit,
) {
    val message = com.minashin1120.aiplayground.data.ChatMessage(bubble.id, bubble.role, bubble.text, model = bubble.model)
    val skeleton: (@Composable () -> Unit)? = if (bubble.pending) {
        { PendingSkeleton(bubble.model.ifBlank { currentModel }, "設定リクエストを確認しています...", "") }
    } else null
    val rows: (@Composable () -> Unit)? = if (bubble.entries.isEmpty()) null else {
        { SettingsResultRows(bubble.entries, onJump) }
    }
    MessageBubble(message, onFile, loader, MessageActions(), controlsVisible = false, onToggleControls = {},
        liveSkeleton = skeleton, liveBottom = rows)
}

/** `.ai-settings-result-list`: label, value and the external-link glyph for each changed or inspected setting. */
@Composable
private fun SettingsResultRows(entries: List<Pair<String, String>>, onJump: (String) -> Unit) {
    val web = LocalWebPalette.current
    Column(Modifier.padding(top = 12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        entries.forEach { (key, value) ->
            val shape = RoundedCornerShape(12.dp)
            Row(
                Modifier.fillMaxWidth().clip(shape).background(Color.Black.copy(alpha = 0.2f)).border(1.dp, Color.White.copy(alpha = 0.1f), shape)
                    .clickable(role = androidx.compose.ui.semantics.Role.Button) { onJump(key) }.padding(horizontal = 12.dp, vertical = 10.dp),
                verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Column(Modifier.weight(1f)) {
                    Text(com.minashin1120.aiplayground.data.AI_SETTING_JUMP_TARGETS[key]?.label ?: key, fontSize = 12.sp, lineHeight = 16.sp,
                        fontWeight = FontWeight.Bold, color = web.twText(Tw.blue200))
                    Text(value, fontSize = 11.sp, lineHeight = 16.sp, color = web.twText(Tw.gray300), modifier = Modifier.padding(top = 2.dp))
                }
                FaIcon(R.drawable.fa_solid_arrow_up_right_from_square, null, size = 10.dp, tint = web.twText(Tw.blue300))
            }
        }
    }
}

/** Web `#python-exec-modal` (`openPythonExecDetail`): every Python run saved in the answer. */
@Composable
internal fun PythonExecutionDialog(runs: List<com.minashin1120.aiplayground.data.PythonExecution>, onDismiss: () -> Unit) {
    val web = LocalWebPalette.current
    WebOverlayModal(onDismiss, grayOverlay(), 4.dp) { _ ->
        val shape = RoundedCornerShape(12.dp)
        Box(
            Modifier.padding(16.dp).widthIn(max = 672.dp).fillMaxWidth().clip(shape).background(web.twBg(Tw.gray900))
                .border(1.dp, web.twBorder(Tw.gray700), shape).padding(16.dp),
        ) {
            Column {
                Row(Modifier.padding(end = 32.dp), verticalAlignment = Alignment.CenterVertically) {
                    FaIcon(R.drawable.fa_solid_terminal, null, size = 13.dp, tint = web.twText(Tw.amber300), modifier = Modifier.padding(end = 6.dp))
                    Text("Python 実行結果" + if (runs.size > 1) "（${runs.size}件）" else "", fontSize = 14.sp, fontWeight = FontWeight.Bold,
                        color = web.twText(Tw.amber300))
                }
                Column(Modifier.padding(top = 12.dp).heightIn(max = 640.dp).verticalScroll(rememberScrollState())) {
                    runs.forEachIndexed { index, run ->
                        PythonExecutionBox(
                            StatusCard("pyexec-detail-$index", CardKind.PYTHON, "", code = run.code, output = run.output, done = true),
                            label = if (runs.size > 1) "Python Execution ${index + 1}/${runs.size}" else "Python Execution",
                            detail = true,
                        )
                    }
                }
            }
            Box(Modifier.align(Alignment.TopEnd).size(24.dp).clip(CircleShape).clickable(role = Role.Button, onClick = onDismiss),
                contentAlignment = Alignment.Center) { FaIcon(R.drawable.fa_solid_times, "閉じる", size = 13.dp, tint = Tw.gray400) }
        }
    }
}
