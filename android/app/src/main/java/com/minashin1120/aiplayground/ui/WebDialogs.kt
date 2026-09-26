package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.R

/** Host shown by Chrome in the title of `alert()` / `confirm()` / `prompt()` dialogs. */
internal const val WEB_DIALOG_HOST = "ai.minashin1120.com"

/**
 * Web `confirm(message)` as Chrome for Android shows it: the site host as the title, the message,
 * and キャンセル / OK. Used wherever Web calls `confirm()` so the wording and the choice match.
 */
@Composable
internal fun BrowserConfirmDialog(message: String, onResult: (Boolean) -> Unit) {
    AlertDialog(
        onDismissRequest = { onResult(false) },
        title = { Text("$WEB_DIALOG_HOST の内容", fontSize = 18.sp) },
        text = { Text(message, fontSize = 15.sp) },
        confirmButton = { TextButton(onClick = { onResult(true) }) { Text("OK") } },
        dismissButton = { TextButton(onClick = { onResult(false) }) { Text("キャンセル") } },
    )
}

/** Web `prompt(message)`: returns the entered text, or null when cancelled (Web ignores empty input). */
@Composable
internal fun BrowserPromptDialog(message: String, onResult: (String?) -> Unit) {
    var value by remember { mutableStateOf("") }
    AlertDialog(
        onDismissRequest = { onResult(null) },
        title = { Text("$WEB_DIALOG_HOST の内容", fontSize = 18.sp) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(message, fontSize = 15.sp)
                OutlinedTextField(value, { value = it }, singleLine = true, modifier = Modifier.fillMaxWidth())
            }
        },
        confirmButton = { TextButton(onClick = { onResult(value) }) { Text("OK") } },
        dismissButton = { TextButton(onClick = { onResult(null) }) { Text("キャンセル") } },
    )
}

/**
 * A Web `.modal-overlay` with a custom `.modal-panel`: tinted, blurred backdrop and the Web
 * open/close motion. [content] draws the panel itself.
 */
@Composable
internal fun WebOverlayModal(
    onDismissRequest: () -> Unit,
    overlay: Color,
    blur: Dp,
    alignment: Alignment = Alignment.Center,
    content: @Composable BoxScope.(phone: Boolean) -> Unit,
) {
    Dialog(onDismissRequest, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        WebModalWindow(blur)
        Box(Modifier.fillMaxSize()) {
            WebModalScrim(overlay)
            BoxWithConstraints(Modifier.fillMaxSize().safeDrawingPadding(), contentAlignment = alignment) {
                val phone = maxWidth < PlaygroundDimens.breakpoint
                ModalPanelMotion(fullScreen = phone, onDismissRequest = onDismissRequest) {
                    Box(contentAlignment = alignment) { content(phone) }
                }
            }
        }
    }
}

/** `bg-gray-900/95` overlays used by the history, legal and alpha modals. */
@Composable
internal fun grayOverlay(): Color {
    val web = LocalWebPalette.current
    return if (web.isLight) Color(0xFFF7F9FC) else Tw.gray900.copy(alpha = 0.95f)
}

/** `#history-modal`: the thread list moved into a modal with its own search box. */
@Composable
internal fun HistoryDialog(state: ChatState, actions: SidebarActions, onDismiss: () -> Unit) {
    val web = LocalWebPalette.current
    val colors = sidebarColors()
    val panel = web.twBg(Tw.gray800)
    val border = web.twBorder(Tw.gray700)
    WebOverlayModal(onDismiss, grayOverlay(), 4.dp) { _ ->
        val shape = RoundedCornerShape(8.dp)
        Column(
            Modifier
                .padding(horizontal = 16.dp, vertical = 60.dp)
                .widthIn(max = 720.dp)
                .fillMaxSize()
                .clip(shape)
                .background(panel)
                .border(1.dp, border, shape),
        ) {
            Row(Modifier.fillMaxWidth().padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                FaIcon(R.drawable.fa_solid_history, null, size = 18.dp, tint = web.theme300)
                Spacer(Modifier.width(8.dp))
                Text("チャット履歴", color = web.twText(Tw.white), fontSize = 18.sp, lineHeight = 28.sp,
                    fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
                Box(Modifier.clickable(onClickLabel = "閉じる", role = Role.Button, onClick = onDismiss).padding(4.dp)) {
                    FaIcon(R.drawable.fa_solid_times, "閉じる", size = 14.dp, tint = web.twText(Tw.gray400))
                }
            }
            Box(Modifier.fillMaxWidth().height(1.dp).background(border))
            Box(Modifier.fillMaxWidth().padding(12.dp)) {
                val fieldShape = RoundedCornerShape(4.dp)
                Box(
                    Modifier
                        .fillMaxWidth()
                        .height(38.dp)
                        .clip(fieldShape)
                        .background(web.twBg(Tw.gray900))
                        .border(1.dp, web.twBorder(Tw.gray600), fieldShape)
                        .padding(horizontal = 16.dp),
                    contentAlignment = Alignment.CenterStart,
                ) {
                    val text = web.twText(Tw.white)
                    BasicTextField(
                        state.search, actions.onSearch, singleLine = true,
                        textStyle = TextStyle(color = text, fontSize = 14.sp, lineHeight = 20.sp, fontFamily = WebFonts.sans),
                        cursorBrush = SolidColor(text),
                        modifier = Modifier.fillMaxWidth().padding(end = 20.dp),
                        decorationBox = { inner ->
                            if (state.search.isEmpty()) Text("履歴を検索...", color = Tw.gray400, fontSize = 14.sp, lineHeight = 20.sp)
                            inner()
                        },
                    )
                    FaIcon(R.drawable.fa_solid_search, null, size = 14.dp, tint = web.twText(Tw.gray500),
                        modifier = Modifier.align(Alignment.CenterEnd))
                }
            }
            Box(Modifier.fillMaxWidth().height(1.dp).background(border))
            ThreadList(state, colors, actions, Modifier.weight(1f), PaddingValues(8.dp))
        }
    }
}

/** `#alpha-info-modal` (`showAlphaInfo()`), opened from the version in the sidebar footer. */
@Composable
internal fun AlphaInfoDialog(onDismiss: () -> Unit) {
    val web = LocalWebPalette.current
    WebOverlayModal(onDismiss, grayOverlay(), 12.dp) { _ ->
        val shape = RoundedCornerShape(20.dp)
        Column(
            Modifier
                .padding(16.dp)
                .widthIn(max = 448.dp)
                .fillMaxWidth()
                .clip(shape)
                .background(if (web.isLight) Color.White.copy(alpha = 0.95f) else Color(14, 20, 40).copy(alpha = 0.9f))
                .border(1.dp, Color(249, 115, 22).copy(alpha = 0.6f), shape)
                .padding(24.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(bottom = 16.dp)) {
                FaIcon(R.drawable.fa_solid_exclamation_triangle, null, size = 20.dp, tint = Tw.orange500)
                Spacer(Modifier.width(8.dp))
                Text("アルファ版に関するご注意", color = Tw.orange500, fontSize = 20.sp, lineHeight = 28.sp, fontWeight = FontWeight.Bold)
            }
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text(
                    buildAnnotatedString {
                        append("このアプリケーションは開発中の")
                        withStyle(SpanStyle(fontWeight = FontWeight.Bold)) { append("アルファ版") }
                        append("です。")
                    },
                    color = web.twText(Tw.gray300), fontSize = 14.sp, lineHeight = 20.sp,
                )
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    listOf("予告なくデータが削除される可能性があります。", "予期せぬ動作やエラーが発生する場合があります。").forEach { item ->
                        Text("•  $item", color = web.twText(Tw.gray400), fontSize = 14.sp, lineHeight = 20.sp)
                    }
                }
            }
            Row(Modifier.fillMaxWidth().padding(top = 24.dp), horizontalArrangement = Arrangement.End) {
                val buttonShape = RoundedCornerShape(4.dp)
                Box(
                    Modifier
                        .clip(buttonShape)
                        .background(Tw.orange600)
                        .border(1.dp, Color.White.copy(alpha = 0.06f), buttonShape)
                        .clickable(role = Role.Button, onClick = onDismiss)
                        .padding(horizontal = 16.dp, vertical = 8.dp),
                ) { Text("理解しました", color = Color.White, fontSize = 14.sp, lineHeight = 20.sp, fontWeight = FontWeight.Bold) }
            }
        }
    }
}

/** `#legal-modal` (`showLegal('terms' | 'privacy')`): Markdown from `/static/legal/<kind>.md`. */
@Composable
internal fun LegalDialog(kind: String, load: suspend (String) -> String, onDismiss: () -> Unit) {
    val web = LocalWebPalette.current
    var content by remember(kind) { mutableStateOf<String?>(null) }
    var failed by remember(kind) { mutableStateOf(false) }
    LaunchedEffect(kind) {
        try { content = load(kind) } catch (e: kotlinx.coroutines.CancellationException) { throw e } catch (_: Exception) { failed = true }
    }
    val title = if (kind == "privacy") "プライバシーポリシー" else "利用規約"
    WebOverlayModal(onDismiss, grayOverlay(), 4.dp, alignment = Alignment.TopCenter) { _ ->
        val shape = RoundedCornerShape(8.dp)
        val border = web.twBorder(Tw.gray700)
        Column(
            Modifier
                .padding(16.dp)
                .widthIn(max = 768.dp)
                .fillMaxWidth()
                .heightIn(max = 2000.dp)
                .clip(shape)
                .background(web.twBg(Tw.gray800))
                .border(1.dp, border, shape)
                .padding(24.dp),
        ) {
            Text(title, color = web.twText(Tw.white), fontSize = 20.sp, lineHeight = 28.sp, fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(bottom = 8.dp))
            Box(Modifier.fillMaxWidth().height(1.dp).background(border))
            Spacer(Modifier.height(16.dp))
            Column(Modifier.weight(1f, fill = false).verticalScroll(rememberScrollState()).padding(end = 8.dp)) {
                // Web leaves the body empty until the document arrives and when the request fails.
                content?.takeIf { !failed }?.let { LegalMarkdown(it) }
            }
            Row(Modifier.fillMaxWidth().padding(top = 16.dp), horizontalArrangement = Arrangement.End) {
                val buttonShape = RoundedCornerShape(4.dp)
                Box(
                    Modifier
                        .clip(buttonShape)
                        .background(web.twBg(Tw.gray700))
                        .clickable(role = Role.Button, onClick = onDismiss)
                        .padding(horizontal = 16.dp, vertical = 8.dp),
                ) { Text("閉じる", color = web.twText(Tw.white), fontSize = 14.sp, lineHeight = 20.sp) }
            }
        }
    }
}

/**
 * The legal documents render through the chat `prose` styles, where headings, paragraphs and list
 * items all share the 15.2px body text with 16px gaps.
 */
@Composable
private fun LegalMarkdown(source: String) {
    val web = LocalWebPalette.current
    val color = if (web.isLight) web.text else Color(0xFFE6E9F0)
    val style = TextStyle(color = color, fontSize = 15.2.sp, lineHeight = 26.14.sp, letterSpacing = 0.01.em, fontFamily = WebFonts.sans)
    val blocks = remember(source) { source.trim().split(Regex("\\n\\s*\\n")) }
    Column {
        blocks.forEachIndexed { index, block ->
            val lines = block.lines().map { it.trim() }.filter { it.isNotEmpty() }
            val top = if (index == 0) 0.dp else 16.dp
            Column(Modifier.padding(top = top)) {
                lines.forEachIndexed { lineIndex, line ->
                    val bullet = line.startsWith("* ") || line.startsWith("- ")
                    val heading = line.startsWith("#")
                    val text = line.trimStart('#', ' ').removePrefix("* ").removePrefix("- ")
                    when {
                        bullet -> Row(Modifier.padding(start = 8.dp, bottom = 4.dp)) {
                            Text("•", style = style, modifier = Modifier.width(16.dp))
                            Text(text, style = style)
                        }
                        heading && lineIndex > 0 -> Text(text, style = style, modifier = Modifier.padding(top = 16.dp))
                        else -> Text(text, style = style)
                    }
                }
            }
        }
    }
}
