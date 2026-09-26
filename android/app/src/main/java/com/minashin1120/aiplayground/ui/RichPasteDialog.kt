package com.minashin1120.aiplayground.ui

import android.content.ClipDescription
import android.content.Context
import android.text.Html
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.Preferences
import kotlinx.coroutines.delay

/** Web `RICH_PASTE_DEFAULT_PROMPT`. */
internal const val RICH_PASTE_DEFAULT_PROMPT = "このPDFをMarkdown形式に変換し、コードブロックに書き出してください。"

/** Web `getRichPasteEffectivePrompt`: the saved prompt only while "既定値として保存" is on. */
internal fun richPasteEffectivePrompt(prefs: Preferences?): String =
    prefs?.takeIf { it.richPastePromptUseCustomDefault }?.richPastePromptDefault?.trim()?.ifEmpty { null } ?: RICH_PASTE_DEFAULT_PROMPT

/** The imported clipboard as text, with the counts Web shows in `updateRichPasteStatus`. */
private data class RichPasteContent(val text: String, val images: Int = 0, val tables: Int = 0, val links: Int = 0, val headings: Int = 0)

private fun readRichClipboard(context: Context): RichPasteContent? {
    val manager = context.getSystemService(Context.CLIPBOARD_SERVICE) as? android.content.ClipboardManager ?: return null
    val clip = manager.primaryClip ?: return null
    if (clip.itemCount == 0) return null
    val html = if (clip.description.hasMimeType(ClipDescription.MIMETYPE_TEXT_HTML)) clip.getItemAt(0).htmlText else null
    if (!html.isNullOrBlank()) {
        fun count(pattern: String) = Regex(pattern, RegexOption.IGNORE_CASE).findAll(html).count()
        val text = Html.fromHtml(html, Html.FROM_HTML_MODE_COMPACT).toString().replace("￼", "").trim()
        return RichPasteContent(text, count("<img\\b"), count("<table\\b"), count("<a\\b"), count("<h[1-6]\\b"))
    }
    return RichPasteContent(clip.getItemAt(0).coerceToText(context).toString().trim())
}

/** Web `updateRichPasteStatus`. */
private fun richPasteStatus(content: RichPasteContent?): String =
    if (content == null || content.text.isEmpty()) "まだ内容がありません。"
    else "${content.text.length} 文字 / 画像 ${content.images} / 表 ${content.tables} / リンク ${content.links} / 見出し ${content.headings}"

/**
 * Web `#rich-paste-modal`. Android turns the clipboard HTML into text and adds it to the input with the
 * instruction instead of making a PDF (ANDROID_ONLY.md); the frame, wording and the saved default prompt
 * follow Web.
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun RichPasteDialog(
    preferences: Preferences?,
    onDismiss: () -> Unit,
    onSavePrompt: (prompt: String, useCustomDefault: Boolean) -> Unit,
    notify: (String) -> Unit,
    onInsert: (String) -> Unit,
) {
    val web = LocalWebPalette.current
    val context = LocalContext.current
    var content by remember { mutableStateOf<RichPasteContent?>(null) }
    var prompt by remember { mutableStateOf(richPasteEffectivePrompt(preferences)) }
    var useDefault by remember { mutableStateOf(preferences?.richPastePromptUseCustomDefault == true) }
    var edited by remember { mutableStateOf(false) }
    // Web `queueRichPastePromptPreferenceSave`: saved 500ms after the last change.
    LaunchedEffect(prompt, useDefault) {
        if (!edited) return@LaunchedEffect
        delay(500)
        onSavePrompt(prompt, useDefault)
    }
    TwModalFrame(onDismiss, maxPanelWidth = 896.dp) { phone ->
        Row(Modifier.fillMaxWidth().padding(bottom = 12.dp), verticalAlignment = Alignment.CenterVertically) {
            FaIcon(R.drawable.fa_solid_paste, null, size = 16.dp, tint = web.twText(Tw.amber400))
            Text("リッチ貼り付け → テキスト", fontSize = 18.sp, fontWeight = FontWeight.Bold, color = web.text,
                modifier = Modifier.padding(start = 8.dp).weight(1f))
            Box(Modifier.size(32.dp).clip(androidx.compose.foundation.shape.CircleShape).clickable(role = Role.Button, onClick = onDismiss),
                contentAlignment = Alignment.Center) { FaIcon(R.drawable.fa_solid_times, "閉じる", size = 14.dp, tint = Tw.gray400) }
        }
        val left: @Composable ColumnScope.() -> Unit = {
            val noteShape = RoundedCornerShape(8.dp)
            Text(
                "Word、記事、ブラウザのリッチテキストをコピーしてから取り込むと、書式を外したテキストとして内部保持します。指示文と一緒に入力欄へ追加できます。",
                fontSize = 12.sp, lineHeight = 19.5.sp, color = web.twText(Color(0xFFFEF3C7)).copy(alpha = 0.9f),
                modifier = Modifier.fillMaxWidth().clip(noteShape).background(Color(69, 26, 3).copy(alpha = 0.2f))
                    .border(1.dp, Tw.amber500.copy(alpha = 0.3f), noteShape).padding(horizontal = 16.dp, vertical = 12.dp),
            )
            FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                RichPasteButton("クリップボードを取り込み", Tw.amber600) {
                    val read = readRichClipboard(context)
                    if (read == null || read.text.isEmpty()) notify("クリップボードに貼り付け可能な内容がありませんでした")
                    else content = read
                }
                RichPasteButton("内容をクリア", Tw.gray700) { content = null }
            }
            Text(richPasteStatus(content), fontSize = 11.sp, color = Tw.gray400)
            content?.let { imported ->
                val shape = RoundedCornerShape(8.dp)
                Text(imported.text, fontSize = 12.sp, lineHeight = 18.sp, color = web.twText(Tw.gray300),
                    modifier = Modifier.fillMaxWidth().heightIn(max = 220.dp).clip(shape).background(web.twBg(Tw.gray900, 0.6f))
                        .border(1.dp, web.twBorder(Tw.gray700), shape).verticalScroll(rememberScrollState()).padding(12.dp))
            }
        }
        val right: @Composable ColumnScope.() -> Unit = {
            val cardShape = RoundedCornerShape(8.dp)
            Column(
                Modifier.fillMaxWidth().clip(cardShape).background(web.twBg(Tw.gray900, 0.6f))
                    .border(1.dp, web.twBorder(Tw.gray700), cardShape).padding(16.dp),
            ) {
                Text("モデルへの指示", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = web.twText(Tw.gray200),
                    modifier = Modifier.padding(bottom = 8.dp))
                val fieldShape = RoundedCornerShape(8.dp)
                BasicTextField(
                    prompt, { prompt = it; edited = true },
                    textStyle = TextStyle(fontSize = 14.sp, lineHeight = 24.sp, color = web.text),
                    cursorBrush = SolidColor(web.text),
                    modifier = Modifier.fillMaxWidth().heightIn(min = 200.dp).clip(fieldShape).background(web.twBg(Tw.gray800))
                        .border(1.dp, web.twBorder(Tw.gray600), fieldShape).padding(12.dp),
                    decorationBox = { inner ->
                        if (prompt.isEmpty()) Text("モデルへの指示を入力...", fontSize = 14.sp, color = Tw.gray500)
                        inner()
                    },
                )
                Row(
                    Modifier.padding(top = 12.dp).clickable(role = Role.Checkbox) { useDefault = !useDefault; edited = true },
                    verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    WebCheckbox(useDefault, { useDefault = it; edited = true }, accent = Tw.amber500, size = 16.dp)
                    Text("このプロンプトをこのユーザーの既定値として保存する", fontSize = 12.sp, color = web.twText(Tw.gray300))
                }
            }
            Column(
                Modifier.fillMaxWidth().clip(cardShape).background(web.twBg(Tw.gray900, 0.6f))
                    .border(1.dp, web.twBorder(Tw.gray700), cardShape).padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text("使い方", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = web.twText(Tw.gray200))
                listOf(
                    "1. コピーした内容を「クリップボードを取り込み」で読み込みます。",
                    "2. 必要なら指示文を調整します。",
                    "3. 「入力欄へ追加」を押すと、指示文とテキスト化した内容を入力欄へ追加します。",
                ).forEach { Text(it, fontSize = 12.sp, lineHeight = 24.sp, color = Tw.gray400) }
            }
            Column(Modifier.padding(top = 8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                val imported = content?.text.orEmpty()
                RichPasteButton("入力欄へ追加", Tw.emerald600, Modifier.fillMaxWidth(), vertical = 12.dp, enabled = imported.isNotEmpty()) {
                    val instruction = prompt.trim()
                    onInsert(if (instruction.isEmpty()) imported else "$instruction\n\n$imported")
                    onDismiss()
                }
                RichPasteButton("閉じる", Tw.gray700, Modifier.fillMaxWidth(), vertical = 10.dp, onClick = onDismiss)
            }
        }
        if (phone) Column(verticalArrangement = Arrangement.spacedBy(16.dp)) { left(); right() }
        else Row(horizontalArrangement = Arrangement.spacedBy(24.dp)) {
            Column(Modifier.weight(1.7f), verticalArrangement = Arrangement.spacedBy(16.dp)) { left() }
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(16.dp)) { right() }
        }
    }
}

@Composable
private fun RichPasteButton(
    label: String, color: Color, modifier: Modifier = Modifier, vertical: androidx.compose.ui.unit.Dp = 8.dp,
    enabled: Boolean = true, onClick: () -> Unit,
) {
    val shape = RoundedCornerShape(if (vertical > 8.dp) 8.dp else 4.dp)
    Box(
        modifier.clip(shape).background(if (enabled) color else color.copy(alpha = 0.5f))
            .clickable(enabled = enabled, role = Role.Button, onClick = onClick).padding(horizontal = 12.dp, vertical = vertical),
        contentAlignment = Alignment.Center,
    ) { Text(label, fontSize = 14.sp, fontWeight = FontWeight.Bold, color = Color.White) }
}
