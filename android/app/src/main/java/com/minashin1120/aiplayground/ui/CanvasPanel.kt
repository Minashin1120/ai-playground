package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.data.CanvasBlock
import com.minashin1120.aiplayground.data.CanvasSelection
import com.minashin1120.aiplayground.data.canvasBlockPreview
import com.minashin1120.aiplayground.data.canvasStatus
import com.minashin1120.aiplayground.data.canvasTitle
import com.minashin1120.aiplayground.data.nextCanvasSelection
import com.minashin1120.aiplayground.data.parseCanvasMarkdown

/** Whether answers replace their fences with the Canvas placeholder (Web `canvasModeEnabled`). */
internal val LocalCanvasMode = compositionLocalOf { false }

/** Web: the Canvas follows the answer rendered last, or the one streaming now. */
internal fun canvasSourceText(state: ChatState): String =
    state.liveContent.takeIf { state.streaming && it.isNotBlank() }
        ?: state.messages.lastOrNull { it.role == "assistant" }?.content.orEmpty()

/** Web `#canvas-panel` switches to the full-screen layout below 1024px. */
internal val CANVAS_SIDE_PANEL_MIN_WIDTH = 1024.dp

/**
 * Web `#canvas-panel`: the latest answer's code blocks with Copy / Clear / ×. [fullScreen] is the
 * mobile layout with the view tabs; otherwise the side panel shows both sections. Live Preview is
 * not offered (ANDROID_ONLY.md), so the phone opens on Blocks and picking a block shows its Source.
 */
@Composable
internal fun CanvasPanel(
    source: String,
    fullScreen: Boolean,
    notify: (String) -> Unit,
    onClose: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val web = LocalWebPalette.current
    val clipboard = LocalClipboardManager.current
    var clearedSource by remember { mutableStateOf<String?>(null) }
    val parsed = remember(source) { parseCanvasMarkdown(source).blocks }
    val blocks = if (clearedSource == source) emptyList() else parsed
    var selection by remember { mutableStateOf(CanvasSelection()) }
    LaunchedEffect(blocks.isEmpty()) { if (blocks.isEmpty()) selection = CanvasSelection() }
    val current = nextCanvasSelection(blocks, selection)
    val block = blocks.getOrNull(current.index)
    var view by remember { mutableStateOf("blocks") }
    val select: (Int, String) -> Unit = { index, next -> selection = CanvasSelection(index, manual = true); view = next }

    Column(
        modifier.background(if (web.isLight) Color(248, 250, 252).copy(alpha = 0.96f) else Color(3, 10, 28).copy(alpha = if (fullScreen) 0.97f else 0.85f))
            .then(if (fullScreen) Modifier else Modifier.drawBehind {
                drawRect(Color(148, 163, 184).copy(alpha = 0.15f), Offset.Zero, Size(1.dp.toPx(), size.height))
            }),
    ) {
        Column(
            Modifier.fillMaxWidth().background(if (web.isLight) Color.White else Color(4, 8, 24).copy(alpha = 0.72f))
                .drawBehind { drawRect(web.twBorder(Tw.gray700, 0.7f), Offset(0f, size.height - 1.dp.toPx()), Size(size.width, 1.dp.toPx())) }
                .padding(12.dp),
        ) {
            Row(verticalAlignment = Alignment.Top, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Column(Modifier.weight(1f)) {
                    Text("CANVAS", fontSize = 10.sp, letterSpacing = 0.24.em, color = web.twText(Color(0xFFA5F3FC)))
                    Text(canvasTitle(blocks, current.index), fontSize = 14.sp, fontWeight = FontWeight.Bold, color = web.text,
                        maxLines = 1, overflow = TextOverflow.Ellipsis)
                }
                Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                    CanvasHeaderButton("Copy") {
                        val code = block?.code.orEmpty()
                        if (code.isBlank()) notify("コピーするコードがありません")
                        else { clipboard.setText(AnnotatedString(code)); notify("Canvasコードをコピーしました") }
                    }
                    CanvasHeaderButton("Clear") {
                        clearedSource = source
                        selection = CanvasSelection()
                        view = "blocks"
                        notify("Canvasプレビューをクリアしました")
                    }
                    CanvasHeaderButton("×", close = true, onClick = onClose)
                }
            }
            Text(canvasStatus(block), fontSize = 10.sp, color = Tw.gray400, modifier = Modifier.padding(top = 4.dp))
            if (fullScreen) Row(Modifier.padding(top = 12.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                listOf("blocks" to "Blocks", "source" to "Source").forEach { (id, label) ->
                    val active = view == id
                    val shape = RoundedCornerShape(8.dp)
                    Box(
                        Modifier.weight(1f).clip(shape)
                            .background(if (active) web.theme.rgb(0.18f) else web.twBg(Tw.gray800))
                            .border(1.dp, if (active) web.theme.rgb(0.55f) else web.twBorder(Tw.gray700), shape)
                            .clickable(role = Role.Tab) { view = id }.padding(horizontal = 12.dp, vertical = 8.dp),
                        contentAlignment = Alignment.Center,
                    ) {
                        Text(label, fontSize = 11.sp, fontWeight = FontWeight.SemiBold,
                            color = if (active) web.twText(Color(0xFFECFEFF)) else web.twText(Tw.gray300))
                    }
                }
            }
        }
        Column(Modifier.weight(1f).fillMaxWidth().padding(12.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            if (!fullScreen || view == "blocks") {
                CanvasSection("Blocks", Modifier.then(if (fullScreen) Modifier.weight(1f) else Modifier), trailing = {
                    Text(blocks.size.toString(), fontSize = 10.sp, color = web.twText(Color(0xFFA5F3FC)))
                }) {
                    Column(
                        Modifier.fillMaxWidth().then(if (fullScreen) Modifier.fillMaxHeight() else Modifier.heightIn(max = 160.dp))
                            .verticalScroll(rememberScrollState()).padding(8.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        if (blocks.isEmpty()) Text("コードブロックを待機中", fontSize = 12.sp, color = Tw.gray500,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 12.dp))
                        blocks.forEach { item ->
                            CanvasBlockChip(item, selected = item.index == current.index) { select(item.index, "source") }
                        }
                    }
                }
            }
            if (!fullScreen || view == "source") {
                CanvasSection("Source", Modifier.weight(1f), trailing = {
                    WebSelect(
                        value = if (current.index >= 0) current.index.toString() else "",
                        options = if (blocks.isEmpty()) listOf(WebOption("", "-"))
                            else blocks.map { WebOption(it.index.toString(), "#${it.index + 1} ${it.lang.ifEmpty { "text" }}") },
                        onSelect = { value -> value.toIntOrNull()?.let { select(it, "source") } },
                        modifier = Modifier.widthIn(max = 200.dp),
                        enabled = blocks.isNotEmpty(),
                        fontSize = 10.sp,
                        background = web.twBg(Tw.gray900),
                        borderColor = web.twBorder(Tw.gray700),
                        textColor = web.twText(Color(0xFFA5F3FC)),
                        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 4.dp),
                        contentDescription = "Canvasのソースコードブロック",
                    )
                }) {
                    SelectionContainer(Modifier.fillMaxSize()) {
                        Text(
                            block?.code.orEmpty(), fontFamily = WebFonts.mono, fontSize = 11.84.sp, lineHeight = 18.35.sp,
                            color = web.twText(Tw.gray200),
                            modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp),
                        )
                    }
                }
            }
        }
    }
}

/** Web `#canvas-block-shell` / `#canvas-source-shell`: a rounded section with an uppercase header. */
@Composable
private fun CanvasSection(title: String, modifier: Modifier, trailing: @Composable () -> Unit, content: @Composable ColumnScope.() -> Unit) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(12.dp)
    Column(
        modifier.fillMaxWidth().clip(shape).background(if (web.isLight) Color.White else Color(4, 8, 24).copy(alpha = 0.72f))
            .border(1.dp, web.twBorder(Tw.gray700), shape),
    ) {
        Row(
            Modifier.fillMaxWidth().background(if (web.isLight) Color(241, 245, 249) else Color(10, 16, 40).copy(alpha = 0.72f))
                .drawBehind { drawRect(web.twBorder(Tw.gray700, 0.7f), Offset(0f, size.height - 1.dp.toPx()), Size(size.width, 1.dp.toPx())) }
                .padding(horizontal = 12.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Text(title.uppercase(), fontSize = 10.sp, letterSpacing = 0.18.em, color = Tw.gray400)
            trailing()
        }
        content()
    }
}

/** `.canvas-block-chip`. */
@Composable
private fun CanvasBlockChip(block: CanvasBlock, selected: Boolean, onClick: () -> Unit) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(10.4.dp)
    val cyan = Color(0xFF67E8F9)
    Row(
        Modifier.fillMaxWidth().heightIn(min = 48.dp).clip(shape)
            .background(if (selected) web.theme.rgb(0.2f) else if (web.isLight) Color(241, 245, 249) else Color(15, 23, 42).copy(alpha = 0.72f))
            .border(1.dp, if (selected) web.theme.rgb(0.55f) else Color(148, 163, 184).copy(alpha = 0.18f), shape)
            .drawBehind {
                if (selected) drawRect(cyan, Offset(0f, 7.2.dp.toPx()), Size(3.dp.toPx(), size.height - 14.4.dp.toPx()))
            }
            .clickable(role = Role.Button, onClick = onClick)
            .padding(horizontal = 10.4.dp, vertical = 8.8.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.4.dp),
    ) {
        val indexShape = RoundedCornerShape(8.dp)
        Box(
            Modifier.size(32.dp).clip(indexShape).background(Color(8, 145, 178).copy(alpha = 0.12f))
                .border(1.dp, Color(103, 232, 249).copy(alpha = 0.2f), indexShape),
            contentAlignment = Alignment.Center,
        ) { Text("#${block.index + 1}", fontSize = 11.sp, fontWeight = FontWeight.SemiBold, color = web.twText(cyan)) }
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.4.dp)) {
            Text(block.lang.ifEmpty { "text" }, fontSize = 11.sp, fontWeight = FontWeight.SemiBold,
                color = web.twText(Color(0xFFE2E8F0)), maxLines = 1, overflow = TextOverflow.Ellipsis)
            Text(canvasBlockPreview(block), fontSize = 9.sp, fontFamily = WebFonts.mono, color = Color(0xFF94A3B8),
                maxLines = 1, overflow = TextOverflow.Ellipsis)
        }
        Text(
            if (selected) "表示中" else if (block.open) "生成中" else "表示", fontSize = 9.sp, fontWeight = FontWeight.Medium,
            color = if (selected) web.twText(Color(0xFFCFFAFE)) else Color(0xFF94A3B8),
            modifier = Modifier.clip(CircleShape)
                .background(if (selected) Color(103, 232, 249).copy(alpha = 0.12f) else Color(148, 163, 184).copy(alpha = 0.09f))
                .padding(horizontal = 7.2.dp, vertical = 3.2.dp),
        )
    }
}

@Composable
private fun CanvasHeaderButton(label: String, close: Boolean = false, onClick: () -> Unit) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(4.dp)
    Text(
        label, fontSize = 10.sp, color = if (close) Color.White else web.twText(Tw.gray200),
        modifier = Modifier.clip(shape)
            .background(if (close) Tw.red700.copy(alpha = 0.8f) else web.twBg(Tw.gray800))
            .border(1.dp, if (close) Tw.red500.copy(alpha = 0.4f) else web.twBorder(Tw.gray700), shape)
            .clickable(role = Role.Button, onClickLabel = if (close) "Canvasを閉じる" else null, onClick = onClick)
            .padding(horizontal = 8.dp, vertical = 4.dp),
    )
}

/** Side panel width (Web `clamp(340px, 36vw, 640px)`). */
internal fun canvasSidePanelWidth(screenWidth: Dp): Dp = (screenWidth * 0.36f).coerceIn(340.dp, 640.dp)
