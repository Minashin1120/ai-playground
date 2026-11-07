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

/** Host shown by Chrome in the title of `alert()` / `confirm()` / `prompt()` dialogs (the active server). */
internal val WEB_DIALOG_HOST: String get() = com.minashin1120.aiplayground.data.ServerOrigin.host

/**
 * Web `confirm(message)` as Chrome for Android shows it: the site host as the title, the message,
 * and キャンセル / OK. Used wherever Web calls `confirm()` so the wording and the choice match.
 */
@Composable
internal fun BrowserConfirmDialog(message: String, onResult: (Boolean) -> Unit) {
    NoticeAlertDialog(
        onDismissRequest = { onResult(false) },
        properties = BrowserDialogProperties,
        title = { Text("AI Playground", fontSize = 18.sp) },
        text = { Text(message, fontSize = 15.sp) },
        confirmButton = { TextButton(onClick = { onResult(true) }) { Text("OK") } },
        dismissButton = { TextButton(onClick = { onResult(false) }) { Text("キャンセル") } },
    )
}

/** Browser dialogs close only through their buttons (or back), never an outside tap. */
private val BrowserDialogProperties = DialogProperties(dismissOnClickOutside = false)

/** Web `alert(message)` as Chrome for Android shows it: the site host as the title, the message and OK. */
@Composable
internal fun BrowserAlertDialog(message: String, onDismiss: () -> Unit) {
    NoticeAlertDialog(
        onDismissRequest = onDismiss,
        properties = BrowserDialogProperties,
        title = { Text("AI Playground", fontSize = 18.sp) },
        text = { Text(message, fontSize = 15.sp) },
        confirmButton = { TextButton(onClick = onDismiss) { Text("OK") } },
    )
}

/** Web `prompt(message)`: returns the entered text, or null when cancelled (Web ignores empty input). */
@Composable
internal fun BrowserPromptDialog(message: String, initial: String = "", onResult: (String?) -> Unit) {
    var value by remember { mutableStateOf(initial) }
    NoticeAlertDialog(
        onDismissRequest = { onResult(null) },
        properties = BrowserDialogProperties,
        title = { Text("AI Playground", fontSize = 18.sp) },
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

/** The chat rename form starts with the current title, matching Web. */
@Composable
internal fun ThreadRenameDialog(initial: String, onResult: (String?) -> Unit) {
    var value by remember { mutableStateOf(initial) }
    PlaygroundDialog(
        onDismissRequest = { onResult(null) },
        title = { Text("チャット名を変更") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("チャット名", color = LocalWebPalette.current.twText(Tw.gray300), fontSize = 14.sp)
                TwInput(value, { value = it }, "チャット名")
            }
        },
        confirmButton = {
            WebButton(onClick = { onResult(value) }, variant = WebButtonVariant.Primary, enabled = value.isNotBlank()) { Text("保存") }
        },
        dismissButton = { WebButton(onClick = { onResult(null) }, variant = WebButtonVariant.Ghost) { Text("キャンセル") } },
        icon = R.drawable.fa_solid_pen,
        subtitle = "このチャットの表示名を変更します",
        panelMaxWidth = 440.dp,
    )
}

/**
 * A Web `.modal-overlay` with a custom `.modal-panel`: tinted, blurred backdrop and the Web
 * open/close motion. [content] draws the panel itself and must mark it with [modalPanelTaps];
 * any other tap (including on a full-size scroll wrapper) closes the modal like Web's overlay click.
 */
@Composable
internal fun WebOverlayModal(
    onDismissRequest: () -> Unit,
    overlay: Color,
    blur: Dp,
    alignment: Alignment = Alignment.Center,
    /** Web `.modal-overlay`: a tap outside the panel closes it. False where Web's panel covers the overlay. */
    dismissOnOutsideTap: Boolean = true,
    content: @Composable BoxScope.(phone: Boolean) -> Unit,
) {
    Dialog(onDismissRequest, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        WebModalWindow(blur)
        Box(Modifier.fillMaxSize().modalBackdropDismiss(dismissOnOutsideTap, onDismissRequest)) {
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
    val border = web.line
    PlaygroundDialog(
        onDismissRequest = onDismiss,
        title = { Text("チャット履歴") },
        text = {
            Column(Modifier.fillMaxSize()) {
                Box(Modifier.fillMaxWidth().padding(12.dp)) {
                    val fieldShape = RoundedCornerShape(12.dp)
                    Box(
                        Modifier
                            .fillMaxWidth()
                            .height(38.dp)
                            .clip(fieldShape)
                            .background(if (web.isLight) Color.White else Tw.white.copy(alpha = 0.04f))
                            .border(1.dp, border, fieldShape)
                            .padding(horizontal = 12.dp),
                        contentAlignment = Alignment.CenterStart,
                    ) {
                        val text = web.text
                        BasicTextField(
                            state.search, actions.onSearch, singleLine = true,
                            textStyle = TextStyle(color = text, fontSize = 13.sp, lineHeight = 20.sp, fontFamily = WebFonts.sans),
                            cursorBrush = SolidColor(text),
                            modifier = Modifier.fillMaxWidth().padding(start = 18.dp, end = 20.dp),
                            decorationBox = { inner ->
                                if (state.search.isEmpty()) Text("履歴を検索...", color = web.muted, fontSize = 13.sp, lineHeight = 20.sp)
                                inner()
                            },
                        )
                        FaIcon(R.drawable.fa_solid_search, null, size = 12.dp, tint = web.muted,
                            modifier = Modifier.align(Alignment.CenterStart))
                        FaIcon(R.drawable.fa_solid_times, null, size = 12.dp, tint = web.muted,
                            modifier = Modifier.align(Alignment.CenterEnd))
                    }
                }
                Box(Modifier.fillMaxWidth().height(1.dp).background(border))
                ThreadList(state, colors, actions, Modifier.weight(1f), PaddingValues(8.dp))
            }
        },
        confirmButton = {},
        showFooter = false,
        fillHeight = true,
        padBody = false,
        icon = R.drawable.fa_solid_history,
        subtitle = "過去のチャットを検索して再開",
        panelMaxWidth = 760.dp,
        tone = WebModalTone.Info,
    )
}

/** `#alpha-info-modal` (`showAlphaInfo()`), opened from the version in the sidebar footer. */
@Composable
internal fun AlphaInfoDialog(onDismiss: () -> Unit) {
    val web = LocalWebPalette.current
    PlaygroundDialog(
        onDismissRequest = onDismiss,
        title = { Text("アルファ版に関するご注意") },
        text = {
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
        },
        confirmButton = { WebButton(onClick = onDismiss, variant = WebButtonVariant.Primary) { Text("理解しました") } },
        icon = R.drawable.fa_solid_exclamation_triangle,
        subtitle = "開発中の機能をご利用いただく前に",
        panelMaxWidth = 560.dp,
        tone = WebModalTone.Warn,
    )
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
    PlaygroundDialog(
        onDismissRequest = onDismiss,
        title = { Text(title) },
        text = {
            Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
                // Web leaves the body empty until the document arrives and when the request fails.
                content?.takeIf { !failed }?.let { LegalMarkdown(it) }
            }
        },
        confirmButton = { WebButton(onClick = onDismiss, variant = WebButtonVariant.Ghost) { Text("閉じる") } },
        fillHeight = true,
        icon = R.drawable.fa_regular_file_lines,
        subtitle = "サービスのご利用条件",
        panelMaxWidth = 760.dp,
        tone = WebModalTone.Info,
    )
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
