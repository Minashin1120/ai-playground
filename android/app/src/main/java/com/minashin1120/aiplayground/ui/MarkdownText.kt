package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.InlineTextContent
import androidx.compose.foundation.text.appendInlineContent
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.compositionLocalOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.Layout
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.LinkAnnotation
import androidx.compose.ui.text.Placeholder
import androidx.compose.ui.text.PlaceholderVerticalAlign
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextLayoutResult
import androidx.compose.ui.text.TextLinkStyles
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.text.withLink
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.Constraints
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.codingTargetKey
import com.minashin1120.aiplayground.data.fileReferencePath
import kotlinx.coroutines.delay
import java.net.URI

/*
 * Markdown as the Web chat renders it: `marked` (GFM, `breaks: true`) with the custom code renderer
 * of chat_core part07, inside a `white-space: pre-wrap` `.content-area`. Because of pre-wrap, the
 * newline marked writes after each block (not after code blocks) shows as an empty line, which is
 * why blocks are separated by a margin plus one line box.
 */

internal sealed interface MarkdownBlock {
    data class Paragraph(val text: String) : MarkdownBlock
    data class Heading(val level: Int, val text: String) : MarkdownBlock
    data class Quote(val blocks: List<MarkdownBlock>) : MarkdownBlock
    data class ListBlock(val ordered: Boolean, val start: Int, val items: List<ListItem>, val loose: Boolean) : MarkdownBlock
    data class Code(val language: String, val text: String) : MarkdownBlock
    data class Math(val tex: String, val display: Boolean) : MarkdownBlock
    data class Table(val headers: List<String>, val rows: List<List<String>>) : MarkdownBlock
    data class Image(val reference: String, val alt: String) : MarkdownBlock
    /** ```` ```chat_error ```` fence: a persisted generation error (Web `buildChatErrorBubbleHtml`). */
    data class ChatError(val text: String) : MarkdownBlock
    data object Rule : MarkdownBlock
}

/** One list item; [checked] is non-null for GFM task items. */
internal data class ListItem(val blocks: List<MarkdownBlock>, val checked: Boolean? = null)

private val FENCE = Regex("^( {0,3})(`{3,}|~{3,})(.*)$")
private val HEADING = Regex("^ {0,3}(#{1,6})(?:[ \\t]+(.*?))?(?:[ \\t]+#+)?[ \\t]*$")
private val RULE = Regex("^ {0,3}([-*_])(?:[ \\t]*\\1){2,}[ \\t]*$")
private val QUOTE = Regex("^ {0,3}> ?(.*)$")
private val LIST_ITEM = Regex("^( {0,3})([*+-]|\\d{1,9}[.)])([ \\t]+(.*)|$)")
private val TABLE_SEPARATOR = Regex("^\\|?\\s*:?-+:?\\s*(\\|\\s*:?-+:?\\s*)*\\|?\\s*$")
private val SETEXT = Regex("^ {0,3}(=+|-+)[ \\t]*$")
private val TASK = Regex("^\\[([ xX])][ \\t]+(.*)$", RegexOption.DOT_MATCHES_ALL)
private val IMAGE_TOKEN = Regex("!\\[([^\\]\\n]*)]\\(([^)\\n]+)\\)")

private fun isTableStart(lines: List<String>, index: Int): Boolean {
    if (index + 1 >= lines.size) return false
    if (!lines[index].contains('|')) return false
    val separator = lines[index + 1].trim()
    return separator.contains('-') && TABLE_SEPARATOR.matches(separator)
}

private fun splitTableRow(line: String): List<String> {
    val trimmed = line.trim().removePrefix("|").let { if (it.endsWith("|") && !it.endsWith("\\|")) it.dropLast(1) else it }
    val cells = mutableListOf<String>()
    val current = StringBuilder()
    var i = 0
    while (i < trimmed.length) {
        val c = trimmed[i]
        if (c == '\\' && i + 1 < trimmed.length && trimmed[i + 1] == '|') { current.append('|'); i += 2; continue }
        if (c == '|') { cells += current.toString().trim(); current.clear() } else current.append(c)
        i++
    }
    cells += current.toString().trim()
    return cells
}

/** Lines that end a paragraph (marked's "interrupting" block starts). */
private fun interruptsParagraph(line: String): Boolean {
    if (FENCE.matches(line) || RULE.matches(line) || QUOTE.matches(line)) return true
    if (HEADING.matches(line)) return true
    if (line.trimStart().startsWith("$$") || line.trimStart().startsWith("\\[")) return true
    val item = LIST_ITEM.matchEntire(line) ?: return false
    val marker = item.groupValues[2]
    val content = item.groupValues[4]
    // GFM: an ordered list may interrupt a paragraph only when it starts at 1; no empty items.
    if (content.isBlank()) return false
    return !marker[0].isDigit() || marker.dropLast(1) == "1"
}

internal fun parseMarkdownBlocks(source: String): List<MarkdownBlock> =
    parseBlocks(source.replace("\r\n", "\n").replace('\t', ' ').split('\n'))

private fun parseBlocks(lines: List<String>): List<MarkdownBlock> {
    val result = mutableListOf<MarkdownBlock>()
    var index = 0
    while (index < lines.size) {
        val line = lines[index]
        val trimmed = line.trimStart()
        val fence = FENCE.matchEntire(line)
        when {
            line.isBlank() -> index++

            fence != null && !(fence.groupValues[2][0] == '`' && fence.groupValues[3].contains('`')) -> {
                val indent = fence.groupValues[1].length
                val marker = fence.groupValues[2]
                val language = fence.groupValues[3].trim().substringBefore(' ').take(32)
                val body = mutableListOf<String>()
                index++
                while (index < lines.size) {
                    val candidate = lines[index].trimStart()
                    if (candidate.startsWith(marker) && candidate.trimEnd().all { it == marker[0] }) { index++; break }
                    body += lines[index].let { l -> var drop = 0; while (drop < indent && drop < l.length && l[drop] == ' ') drop++; l.substring(drop) }
                    index++
                }
                val text = body.joinToString("\n")
                when (language) {
                    // Web keeps Python runs out of the answer body (they open from the footer button).
                    "pyexec" -> Unit
                    "chat_error" -> result += MarkdownBlock.ChatError(text)
                    else -> result += MarkdownBlock.Code(language, text)
                }
            }

            trimmed.startsWith("$$") -> {
                val body = mutableListOf<String>()
                val rest = trimmed.removePrefix("$$")
                if (rest.contains("$$")) {
                    body += rest.substringBefore("$$")
                    index++
                } else {
                    if (rest.isNotBlank()) body += rest
                    index++
                    while (index < lines.size && !lines[index].contains("$$")) body += lines[index++]
                    if (index < lines.size) { body += lines[index].substringBefore("$$"); index++ }
                }
                result += MarkdownBlock.Math(body.joinToString("\n").trim(), true)
            }

            trimmed.startsWith("\\[") -> {
                val body = mutableListOf<String>()
                val rest = trimmed.removePrefix("\\[")
                if (rest.contains("\\]")) {
                    body += rest.substringBefore("\\]")
                    index++
                } else {
                    if (rest.isNotBlank()) body += rest
                    index++
                    while (index < lines.size && !lines[index].contains("\\]")) body += lines[index++]
                    if (index < lines.size) { body += lines[index].substringBefore("\\]"); index++ }
                }
                result += MarkdownBlock.Math(body.joinToString("\n").trim(), true)
            }

            HEADING.matches(line) -> {
                val match = HEADING.matchEntire(line)!!
                result += MarkdownBlock.Heading(match.groupValues[1].length, match.groupValues[2].trim())
                index++
            }

            RULE.matches(line) -> { result += MarkdownBlock.Rule; index++ }

            QUOTE.matches(line) -> {
                val inner = mutableListOf<String>()
                while (index < lines.size) {
                    val quote = QUOTE.matchEntire(lines[index])
                    when {
                        quote != null -> inner += quote.groupValues[1]
                        // Lazy continuation of a quoted paragraph.
                        lines[index].isNotBlank() && inner.lastOrNull()?.isNotBlank() == true && !interruptsParagraph(lines[index]) ->
                            inner += lines[index]
                        else -> break
                    }
                    index++
                }
                result += MarkdownBlock.Quote(parseBlocks(inner))
            }

            LIST_ITEM.matches(line) -> {
                val (block, next) = parseList(lines, index)
                result += block
                index = next
            }

            isTableStart(lines, index) -> {
                val headers = splitTableRow(lines[index])
                val columnCount = headers.size
                var rowIndex = index + 2
                val rows = mutableListOf<List<String>>()
                while (rowIndex < lines.size && lines[rowIndex].isNotBlank() && lines[rowIndex].contains('|')) {
                    val cells = splitTableRow(lines[rowIndex])
                    rows += (0 until columnCount).map { cells.getOrNull(it).orEmpty() }
                    rowIndex++
                }
                result += MarkdownBlock.Table(headers, rows)
                index = rowIndex
            }

            else -> {
                val paragraph = mutableListOf(line.trim())
                index++
                var heading: Int? = null
                while (index < lines.size && lines[index].isNotBlank()) {
                    val setext = SETEXT.matchEntire(lines[index])
                    if (setext != null) { heading = if (setext.groupValues[1][0] == '=') 1 else 2; index++; break }
                    if (interruptsParagraph(lines[index]) || isTableStart(lines, index)) break
                    paragraph += lines[index++].trim()
                }
                val text = paragraph.joinToString("\n")
                if (heading != null) result += MarkdownBlock.Heading(heading, text) else appendParagraph(result, text)
            }
        }
    }
    return result
}

/** A run of list items of the same kind, with nested content taken from indented lines. */
private fun parseList(lines: List<String>, start: Int): Pair<MarkdownBlock.ListBlock, Int> {
    val first = LIST_ITEM.matchEntire(lines[start])!!
    val ordered = first.groupValues[2][0].isDigit()
    val delimiter = first.groupValues[2].last()
    val startNumber = if (ordered) first.groupValues[2].dropLast(1).toIntOrNull() ?: 1 else 1
    val items = mutableListOf<ListItem>()
    var loose = false
    var index = start
    while (index < lines.size) {
        val match = LIST_ITEM.matchEntire(lines[index]) ?: break
        val marker = match.groupValues[2]
        val sameKind = marker[0].isDigit() == ordered && (if (ordered) marker.last() == delimiter else marker == first.groupValues[2])
        if (!sameKind) break
        val indent = match.groupValues[1].length + marker.length +
            (match.groupValues[3].length - match.groupValues[4].length).coerceIn(1, 4)
        val body = mutableListOf(match.groupValues[4])
        index++
        var sawBlank = false
        while (index < lines.size) {
            val candidate = lines[index]
            if (candidate.isBlank()) { body += ""; sawBlank = true; index++; continue }
            val leading = candidate.length - candidate.trimStart().length
            if (leading >= indent) {
                if (sawBlank) loose = loose || body.dropLast(1).lastOrNull()?.isNotBlank() == true && leading == indent && !LIST_ITEM.matches(candidate.drop(indent))
                body += candidate.drop(indent); sawBlank = false; index++; continue
            }
            if (sawBlank) break
            if (LIST_ITEM.matches(candidate) || interruptsParagraph(candidate)) break
            body += candidate.trim(); index++
        }
        // A blank line between two items makes the whole list loose (marked wraps items in <p>).
        if (sawBlank && index < lines.size && LIST_ITEM.matchEntire(lines[index])?.let { next ->
                val m = next.groupValues[2]; m[0].isDigit() == ordered } == true) loose = true
        while (body.isNotEmpty() && body.last().isBlank()) body.removeAt(body.lastIndex)
        val joined = body.joinToString("\n")
        val task = TASK.matchEntire(joined)
        val checked = task?.let { it.groupValues[1] != " " }
        val content = task?.groupValues?.get(2) ?: joined
        items += ListItem(parseBlocks(content.split('\n')), checked)
        if (sawBlank && (index >= lines.size || !LIST_ITEM.matches(lines[index]))) break
    }
    return MarkdownBlock.ListBlock(ordered, startNumber, items, loose) to index
}

private fun appendParagraph(result: MutableList<MarkdownBlock>, text: String) {
    var cursor = 0
    IMAGE_TOKEN.findAll(text).forEach { match ->
        val reference = fileReferencePath(match.groupValues[2]) ?: return@forEach
        val before = text.substring(cursor, match.range.first)
        if (before.isNotBlank()) result += MarkdownBlock.Paragraph(before.trim())
        result += MarkdownBlock.Image(reference, match.groupValues[1])
        cursor = match.range.last + 1
    }
    val tail = text.substring(cursor)
    if (tail.isNotBlank()) result += MarkdownBlock.Paragraph(tail.trim())
}

internal fun safeWebUrl(value: String): String? = runCatching {
    URI(value.trim()).takeIf {
        val scheme = it.scheme?.lowercase()
        (scheme == "http" || scheme == "https") && !it.host.isNullOrBlank()
    }?.toString()
}.getOrNull()

// ---------------------------------------------------------------------------------------------
// Colors and metrics of `.content-area` (chat.custom.v*.css, theme-light-manual.css)
// ---------------------------------------------------------------------------------------------

internal data class MarkdownColors(
    val text: Color,
    val quote: Color,
    val quoteBar: Color,
    val link: Color,
    val inlineCode: Color,
    val inlineCodeBackground: Color,
    val codeWrapper: Color,
    val codeWrapperBorder: Color,
    val codeHeader: Color,
    val codeHeaderText: Color,
    val codeBody: Color,
    val codeText: Color,
    val table: Color,
    val tableBorder: Color,
    val tableHead: Color,
    val tableHeadText: Color,
    val rule: Color,
    val syntax: SyntaxColors,
)

/** highlight.js theme colors (atom-one-dark on the dark theme, atom-one-light on the light theme). */
internal data class SyntaxColors(
    val keyword: Color,
    val title: Color,
    val string: Color,
    val number: Color,
    val comment: Color,
    val builtIn: Color,
    val literal: Color,
)

@Composable
internal fun markdownColors(): MarkdownColors {
    val web = LocalWebPalette.current
    return if (web.isLight) MarkdownColors(
        text = web.text,
        quote = web.muted,
        quoteBar = web.theme.t500,
        link = web.theme300,
        inlineCode = web.theme300,
        inlineCodeBackground = web.theme.rgb(0.12f),
        codeWrapper = Color.White,
        codeWrapperBorder = Color(15, 23, 42).copy(alpha = 0.10f),
        codeHeader = Color(0xFFF1F4F9),
        codeHeaderText = web.muted,
        codeBody = Color(0xFFF8FAFC),
        codeText = Color(0xFF383A42),
        table = Color(0xFFF7F9FC),
        tableBorder = web.line,
        tableHead = web.theme.rgb(0.10f),
        tableHeadText = web.theme300,
        rule = web.line,
        syntax = SyntaxColors(
            keyword = Color(0xFFA626A4), title = Color(0xFF4078F2), string = Color(0xFF50A14F),
            number = Color(0xFF986801), comment = Color(0xFFA0A1A7), builtIn = Color(0xFFC18401), literal = Color(0xFF0184BB),
        ),
    ) else MarkdownColors(
        text = Color(0xFFE6E9F0),
        quote = web.muted,
        quoteBar = web.theme.t500,
        link = web.theme300,
        inlineCode = Color(0xFFCCFBF1),
        inlineCodeBackground = web.theme.rgb(0.14f),
        codeWrapper = Color(8, 12, 22).copy(alpha = 0.72f),
        codeWrapperBorder = Color.White.copy(alpha = 0.08f),
        codeHeader = Color(15, 23, 42).copy(alpha = 0.6f),
        codeHeaderText = Color(0xFF9AA3B2),
        codeBody = Color(0xFF060D1D),
        codeText = Color(0xFFABB2BF),
        table = Color(0xFF090E1C),
        tableBorder = web.line,
        tableHead = web.theme.rgb(0.12f),
        tableHeadText = web.theme200,
        rule = web.line,
        syntax = SyntaxColors(
            keyword = Color(0xFFC678DD), title = Color(0xFF61AEEE), string = Color(0xFF98C379),
            number = Color(0xFFD19A66), comment = Color(0xFF5C6370), builtIn = Color(0xFFE6C07B), literal = Color(0xFF56B6C2),
        ),
    )
}

/** Body text of an AI answer: 15.2px / 26.144px, `letter-spacing: .01em`. */
internal val MarkdownBodySize = 15.2.sp
internal val MarkdownLineHeight = 26.144.sp
private val InlineCodeSize = 13.68.sp
private val ParagraphGap = 16.dp
private val CodeGap = 11.4.dp
private val ListItemGap = 4.dp
private val ListIndent = 24.dp

/** Actions offered by rendered code blocks (download is saved through the system file picker). */
internal class MarkdownCodeActions(
    val onDownload: (code: String, language: String) -> Unit = { _, _ -> },
    val onCodingTarget: ((code: String, language: String) -> Unit)? = null,
    /** Key of the explicit Coding target (`codingTargetKey`); its button shows the pin. */
    val selectedCodingKey: String? = null,
)

internal val LocalMarkdownCodeActions = compositionLocalOf { MarkdownCodeActions() }

/**
 * Markdown message body. [startCollapsed] mirrors Web: stored answers show code blocks collapsed,
 * while an answer that is still streaming keeps them open.
 */
@Composable
fun MarkdownText(text: String, loader: FileBytesLoader? = null, onOpen: (String) -> Unit = {}, startCollapsed: Boolean = true) {
    val blocks = remember(text) { parseMarkdownBlocks(text) }
    val colors = markdownColors()
    val base = TextStyle(
        color = colors.text, fontSize = MarkdownBodySize, lineHeight = MarkdownLineHeight,
        letterSpacing = 0.01.em, fontFamily = WebFonts.sans,
    )
    PreWrapBlocks(blocks, base, colors, loader, onOpen, startCollapsed, leadingLine = false)
}

/** Height of one pre-wrap line box (the empty line produced by marked's `\n` between blocks). */
@Composable
private fun lineBox(style: TextStyle): Dp = with(LocalDensity.current) { style.lineHeight.toDp() }

private fun MarkdownBlock.marginTop(): Dp = when (this) {
    is MarkdownBlock.Quote, is MarkdownBlock.Table -> ParagraphGap
    is MarkdownBlock.Code -> CodeGap
    is MarkdownBlock.ChatError -> 8.dp
    MarkdownBlock.Rule -> 24.dp
    else -> 0.dp
}

private fun MarkdownBlock.marginBottom(): Dp = when (this) {
    is MarkdownBlock.Paragraph, is MarkdownBlock.ListBlock, is MarkdownBlock.Quote, is MarkdownBlock.Table,
    is MarkdownBlock.Image, is MarkdownBlock.Math -> ParagraphGap
    is MarkdownBlock.Code -> CodeGap
    MarkdownBlock.Rule -> 24.dp
    is MarkdownBlock.Heading, is MarkdownBlock.ChatError -> 0.dp
}

/** Code blocks come from a custom renderer that does not end with a newline. */
private fun MarkdownBlock.followedByNewline(): Boolean = this !is MarkdownBlock.Code && this !is MarkdownBlock.ChatError

@Composable
private fun PreWrapBlocks(
    blocks: List<MarkdownBlock>,
    style: TextStyle,
    colors: MarkdownColors,
    loader: FileBytesLoader?,
    onOpen: (String) -> Unit,
    startCollapsed: Boolean,
    leadingLine: Boolean,
) {
    val line = lineBox(style)
    Column(Modifier.fillMaxWidth()) {
        if (leadingLine) Spacer(Modifier.height(line))
        var pendingMargin = 0.dp
        var previousHadLine = true
        blocks.forEachIndexed { index, block ->
            // Adjacent block margins collapse; a line box between them keeps both.
            val top = if (previousHadLine) pendingMargin + block.marginTop() else maxOf(pendingMargin, block.marginTop())
            if (top > 0.dp) Spacer(Modifier.height(top))
            key(index) { BlockContent(block, style, colors, loader, onOpen, startCollapsed) }
            pendingMargin = block.marginBottom()
            previousHadLine = block.followedByNewline()
            if (previousHadLine) {
                Spacer(Modifier.height(pendingMargin + line))
                pendingMargin = 0.dp
            }
        }
        if (pendingMargin > 0.dp) Spacer(Modifier.height(pendingMargin))
    }
}

@Composable
private fun BlockContent(
    block: MarkdownBlock,
    style: TextStyle,
    colors: MarkdownColors,
    loader: FileBytesLoader?,
    onOpen: (String) -> Unit,
    startCollapsed: Boolean,
) {
    when (block) {
        is MarkdownBlock.Paragraph -> InlineText(block.text, style, colors)
        // Headings keep the paragraph typography in the chat (no prose heading styles apply).
        is MarkdownBlock.Heading -> InlineText(block.text, style, colors)
        // `border-left: 4px solid` in the theme color with 16px padding; the bar is drawn so it follows the content height.
        is MarkdownBlock.Quote -> Box(
            Modifier.fillMaxWidth().drawBehind { drawRect(colors.quoteBar, size = Size(4.dp.toPx(), size.height)) }.padding(start = 20.dp),
        ) {
            PreWrapBlocks(block.blocks, style.copy(color = colors.quote, fontStyle = FontStyle.Italic), colors, loader, onOpen,
                startCollapsed, leadingLine = true)
        }
        is MarkdownBlock.ListBlock -> ListContent(block, style, colors, loader, onOpen, startCollapsed, depth = 0)
        is MarkdownBlock.Code -> CodeBlock(block, colors, startCollapsed)
        is MarkdownBlock.ChatError -> ChatErrorBlock(block.text)
        is MarkdownBlock.Math -> MathBlock(block, style)
        is MarkdownBlock.Table -> TableBlock(block, colors)
        is MarkdownBlock.Image -> ProtectedImage(
            block.reference, loader, onOpen,
            modifier = Modifier.fillMaxWidth(), thumbnail = true,
            contentDescription = block.alt.ifBlank { null },
        )
        MarkdownBlock.Rule -> Box(Modifier.fillMaxWidth().height(1.dp).background(colors.rule))
    }
}

@Composable
private fun ListContent(
    block: MarkdownBlock.ListBlock,
    style: TextStyle,
    colors: MarkdownColors,
    loader: FileBytesLoader?,
    onOpen: (String) -> Unit,
    startCollapsed: Boolean,
    depth: Int,
) {
    Column(Modifier.fillMaxWidth()) {
        block.items.forEachIndexed { index, item ->
            Row(Modifier.fillMaxWidth().padding(bottom = ListItemGap)) {
                Box(Modifier.width(ListIndent), contentAlignment = Alignment.TopEnd) {
                    val marker = when {
                        block.ordered -> "${block.start + index}."
                        depth == 0 -> "•"
                        depth == 1 -> "◦"
                        else -> "▪"
                    }
                    Text(marker, style = style, modifier = Modifier.padding(end = 6.dp), textAlign = TextAlign.End)
                }
                Column(Modifier.weight(1f)) {
                    item.blocks.forEachIndexed { blockIndex, child ->
                        val firstLine = blockIndex == 0
                        when (child) {
                            is MarkdownBlock.ListBlock -> ListContent(child, style, colors, loader, onOpen, startCollapsed, depth + 1)
                            is MarkdownBlock.Paragraph -> {
                                if (firstLine && item.checked != null) {
                                    Row(verticalAlignment = Alignment.CenterVertically) {
                                        WebCheckbox(item.checked, onCheckedChange = null, accent = colors.link, enabled = false,
                                            modifier = Modifier.padding(end = 6.dp))
                                        InlineText(child.text, style, colors)
                                    }
                                } else InlineText(child.text, style, colors)
                                if (block.loose) Spacer(Modifier.height(ParagraphGap))
                            }
                            else -> {
                                if (child.marginTop() > 0.dp) Spacer(Modifier.height(child.marginTop()))
                                BlockContent(child, style, colors, loader, onOpen, startCollapsed)
                                if (child.marginBottom() > 0.dp) Spacer(Modifier.height(child.marginBottom()))
                            }
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Inline Markdown
// ---------------------------------------------------------------------------------------------

private const val CODE_PAD = "codepad"
private const val MATH_TAG = "math"

/** Inline code spans carry a start/end placeholder so the background gets the Web 6.4px padding. */
internal data class InlineMarkdown(val text: AnnotatedString, val codeRanges: List<IntRange>)

private val AUTOLINK = Regex("^<(https?://[^>\\s]+)>")
private val BARE_URL = Regex("^(https?://|www\\.)[^\\s<]*[^\\s<?!.,:*_~)\\]]")
private val ESCAPABLE = "\\`*_{}[]()#+-.!|~<>$"

internal fun parseInlineMarkdown(source: String, colors: MarkdownColors): InlineMarkdown {
    val codeRanges = mutableListOf<IntRange>()
    val text = buildAnnotatedString { appendInline(source, colors, codeRanges) }
    return InlineMarkdown(text, codeRanges)
}

private fun AnnotatedString.Builder.appendInline(source: String, colors: MarkdownColors, codeRanges: MutableList<IntRange>) {
    var i = 0
    val plain = StringBuilder()
    fun flush() { if (plain.isNotEmpty()) { append(plain.toString()); plain.clear() } }
    while (i < source.length) {
        val c = source[i]
        val rest = source.substring(i)
        // Escapes
        if (c == '\\' && i + 1 < source.length && source[i + 1] in ESCAPABLE && !rest.startsWith("\\(") && !rest.startsWith("\\[")) {
            plain.append(source[i + 1]); i += 2; continue
        }
        // Inline math \( … \)
        if (rest.startsWith("\\(")) {
            val end = source.indexOf("\\)", i + 2)
            if (end > 0) {
                flush()
                withStyle(SpanStyle(fontFamily = FontFamily.Serif, fontStyle = FontStyle.Italic)) { append(latexToDisplay(source.substring(i + 2, end))) }
                i = end + 2; continue
            }
        }
        if (c == '$' && i + 1 < source.length && source[i + 1] != ' ' && source[i + 1] != '$') {
            val end = source.indexOf('$', i + 1)
            if (end > i + 1 && source[end - 1] != ' ' && !source.substring(i + 1, end).contains('\n')) {
                flush()
                withStyle(SpanStyle(fontFamily = FontFamily.Serif, fontStyle = FontStyle.Italic)) { append(latexToDisplay(source.substring(i + 1, end))) }
                i = end + 1; continue
            }
        }
        // Code spans (one or more backticks)
        if (c == '`') {
            var ticks = 0
            while (i + ticks < source.length && source[i + ticks] == '`') ticks++
            val fence = "`".repeat(ticks)
            val end = source.indexOf(fence, i + ticks)
            if (end > 0) {
                flush()
                var code = source.substring(i + ticks, end).replace('\n', ' ')
                if (code.length > 2 && code.startsWith(" ") && code.endsWith(" ")) code = code.substring(1, code.length - 1)
                val startIndex = length
                appendInlineContent(CODE_PAD, " ")
                withStyle(SpanStyle(fontFamily = WebFonts.mono, fontSize = InlineCodeSize, color = colors.inlineCode)) { append(code) }
                appendInlineContent(CODE_PAD, " ")
                codeRanges += startIndex until length
                i = end + ticks; continue
            }
        }
        // Images (non-attachment images render as their alt text)
        if (rest.startsWith("![")) {
            val match = Regex("^!\\[([^\\]]*)]\\(([^)\\s]+)(?:\\s+\"[^\"]*\")?\\)").find(rest)
            if (match != null) { flush(); append(match.groupValues[1]); i += match.value.length; continue }
        }
        // Links
        if (c == '[') {
            val match = Regex("^\\[((?:[^\\[\\]]|\\[[^\\]]*])*)]\\(\\s*<?([^)\\s>]+)>?(?:\\s+\"[^\"]*\")?\\s*\\)").find(rest)
            if (match != null) {
                flush()
                val url = safeWebUrl(match.groupValues[2])
                if (url == null) appendInline(match.groupValues[1], colors, codeRanges)
                else withLink(LinkAnnotation.Url(url, TextLinkStyles(SpanStyle(color = colors.link, textDecoration = TextDecoration.Underline)))) {
                    appendInline(match.groupValues[1], colors, codeRanges)
                }
                i += match.value.length; continue
            }
        }
        val autolink = AUTOLINK.find(rest)
        if (autolink != null) {
            flush()
            val url = safeWebUrl(autolink.groupValues[1])
            if (url == null) append(autolink.groupValues[1])
            else withLink(LinkAnnotation.Url(url, TextLinkStyles(SpanStyle(color = colors.link, textDecoration = TextDecoration.Underline)))) { append(autolink.groupValues[1]) }
            i += autolink.value.length
            continue
        }
        val previous = if (i > 0) source[i - 1] else ' '
        if ((c == 'h' || c == 'w') && !previous.isLetterOrDigit()) {
            val match = BARE_URL.find(rest)
            if (match != null) {
                flush()
                val raw = match.value
                val url = safeWebUrl(if (raw.startsWith("www.")) "http://$raw" else raw)
                if (url == null) append(raw)
                else withLink(LinkAnnotation.Url(url, TextLinkStyles(SpanStyle(color = colors.link, textDecoration = TextDecoration.Underline)))) { append(raw) }
                i += raw.length; continue
            }
        }
        // Strong / emphasis / strikethrough
        if (rest.startsWith("**") || rest.startsWith("__")) {
            val marker = rest.substring(0, 2)
            val end = source.indexOf(marker, i + 2)
            if (end > i + 2 && !source[i + 2].isWhitespace() && !source[end - 1].isWhitespace()) {
                flush()
                withStyle(SpanStyle(fontWeight = FontWeight.Bold)) { appendInline(source.substring(i + 2, end), colors, codeRanges) }
                i = end + 2; continue
            }
        }
        if (rest.startsWith("~~")) {
            val end = source.indexOf("~~", i + 2)
            if (end > i + 2) {
                flush()
                withStyle(SpanStyle(textDecoration = TextDecoration.LineThrough)) { appendInline(source.substring(i + 2, end), colors, codeRanges) }
                i = end + 2; continue
            }
        }
        if ((c == '*' || (c == '_' && !previous.isLetterOrDigit())) && i + 1 < source.length && !source[i + 1].isWhitespace()) {
            var end = source.indexOf(c, i + 1)
            while (end > 0 && end + 1 < source.length && source[end + 1] == c) end = source.indexOf(c, end + 2)
            if (end > i + 1 && !source[end - 1].isWhitespace() && (c == '*' || end + 1 >= source.length || !source[end + 1].isLetterOrDigit())) {
                flush()
                withStyle(SpanStyle(fontStyle = FontStyle.Italic)) { appendInline(source.substring(i + 1, end), colors, codeRanges) }
                i = end + 1; continue
            }
        }
        plain.append(c)
        i++
    }
    flush()
}

@Composable
private fun InlineText(text: String, style: TextStyle, colors: MarkdownColors) {
    val parsed = remember(text, colors) { parseInlineMarkdown(text, colors) }
    InlineRichText(parsed, style, colors.inlineCodeBackground)
}

/** Text with Web inline-code chips: rounded 6.4px background, 6.4px side padding, 24px tall. */
@Composable
internal fun InlineRichText(parsed: InlineMarkdown, style: TextStyle, codeBackground: Color, modifier: Modifier = Modifier) {
    var layout by remember { mutableStateOf<TextLayoutResult?>(null) }
    val inline = remember {
        mapOf(CODE_PAD to InlineTextContent(Placeholder(6.4.sp, 1.sp, PlaceholderVerticalAlign.TextCenter)) {})
    }
    val density = LocalDensity.current
    val chipHeight = with(density) { 24.dp.toPx() }
    val radius = with(density) { 6.4.dp.toPx() }
    SelectionContainer {
        Text(
            parsed.text,
            style = style,
            inlineContent = inline,
            onTextLayout = { layout = it },
            modifier = modifier.drawBehind {
                val result = layout ?: return@drawBehind
                parsed.codeRanges.forEach { range ->
                    if (range.first >= result.layoutInput.text.length) return@forEach
                    val last = range.last.coerceAtMost(result.layoutInput.text.length - 1)
                    var lineIndex = result.getLineForOffset(range.first)
                    var segmentStart = range.first
                    for (offset in range.first..last + 1) {
                        val lineAt = if (offset <= last) result.getLineForOffset(offset) else -1
                        if (lineAt != lineIndex) {
                            val left = result.getBoundingBox(segmentStart).left
                            val right = result.getBoundingBox(offset - 1).right
                            val center = (result.getLineTop(lineIndex) + result.getLineBottom(lineIndex)) / 2f
                            drawRoundRect(
                                codeBackground,
                                topLeft = Offset(left, center - chipHeight / 2f),
                                size = Size(right - left, chipHeight),
                                cornerRadius = CornerRadius(radius, radius),
                            )
                            lineIndex = lineAt
                            segmentStart = offset
                        }
                    }
                }
            },
        )
    }
}

// ---------------------------------------------------------------------------------------------
// Code blocks
// ---------------------------------------------------------------------------------------------

@Composable
private fun CodeBlock(block: MarkdownBlock.Code, colors: MarkdownColors, startCollapsed: Boolean) {
    var collapsed by remember(block.text, block.language) { mutableStateOf(startCollapsed) }
    var copied by remember { mutableStateOf<Boolean?>(null) }
    LaunchedEffect(copied) { if (copied != null) { delay(2000); copied = null } }
    val clipboard = LocalClipboardManager.current
    val actions = LocalMarkdownCodeActions.current
    val shape = RoundedCornerShape(16.dp)
    Column(
        Modifier
            .fillMaxWidth()
            .clip(shape)
            .background(colors.codeWrapper)
            .border(1.dp, colors.codeWrapperBorder, shape),
    ) {
        Row(
            Modifier
                .fillMaxWidth()
                .heightIn(min = 28.dp)
                .background(colors.codeHeader)
                .padding(start = 10.4.dp, end = 6.4.dp, top = 3.2.dp, bottom = 3.2.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(6.4.dp),
        ) {
            Text(
                block.language.ifBlank { "TEXT" }, color = colors.codeHeaderText, fontSize = 10.88.sp, lineHeight = 13.06.sp,
                fontWeight = FontWeight.Medium, letterSpacing = 0.01.em, modifier = Modifier.weight(1f).graphicsAlpha(0.75f),
            )
            CodeActionButton(if (collapsed) R.drawable.fa_solid_chevron_down else R.drawable.fa_solid_chevron_up,
                if (collapsed) "展開" else "折りたたむ", colors) { collapsed = !collapsed }
            if (block.language.lowercase() != "diff" && actions.onCodingTarget != null) {
                val active = actions.selectedCodingKey == codingTargetKey(block.language.ifBlank { "text" }, block.text)
                CodeActionButton(if (active) R.drawable.fa_solid_thumbtack else R.drawable.fa_solid_quote_right,
                    if (active) "編集対象に設定済み" else "編集対象に指定", colors) { actions.onCodingTarget.invoke(block.text, block.language) }
            }
            CodeActionButton(R.drawable.fa_solid_download, "ダウンロード", colors) { actions.onDownload(block.text, block.language) }
            CodeActionButton(
                when (copied) { true -> R.drawable.fa_solid_check; false -> R.drawable.fa_solid_times; null -> R.drawable.fa_solid_copy },
                "コピー", colors,
            ) {
                copied = runCatching { clipboard.setText(AnnotatedString(block.text)) }.isSuccess
            }
        }
        if (!collapsed) {
            Box(Modifier.fillMaxWidth().height(1.dp).background(colors.codeWrapperBorder))
            val highlighted = remember(block.text, block.language, colors.syntax) { highlightCode(block.text, block.language, colors.syntax) }
            SelectionContainer {
                Text(
                    highlighted,
                    color = colors.codeText,
                    fontFamily = WebFonts.mono, fontSize = 13.68.sp, lineHeight = 21.2.sp, letterSpacing = 0.01.em,
                    softWrap = false,
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(colors.codeBody)
                        .horizontalScroll(rememberScrollState())
                        .padding(13.68.dp),
                )
            }
        }
    }
}

private fun Modifier.graphicsAlpha(alpha: Float): Modifier = this.alpha(alpha)

/** `.code-actions button`: 24.8px, 5.6px corners, 11.2px glyph. */
@Composable
internal fun CodeActionButton(@androidx.annotation.DrawableRes icon: Int, label: String, colors: MarkdownColors, onClick: () -> Unit) {
    Box(
        Modifier
            .size(24.8.dp)
            .clip(RoundedCornerShape(5.6.dp))
            .clickable(onClickLabel = label, role = Role.Button, onClick = onClick),
        contentAlignment = Alignment.Center,
    ) { FaIcon(icon, label, size = 11.2.dp, tint = colors.codeHeaderText) }
}

private val KEYWORDS = setOf(
    "fun", "val", "var", "class", "object", "interface", "if", "else", "when", "for", "while", "do", "return", "break", "continue",
    "import", "package", "private", "public", "protected", "internal", "override", "open", "data", "sealed", "enum", "companion",
    "suspend", "inline", "try", "catch", "finally", "throw", "is", "in", "as", "by", "this", "super", "typealias", "const", "lateinit",
    "def", "from", "with", "lambda", "yield", "pass", "raise", "global", "nonlocal", "async", "await", "and", "or", "not", "elif", "except",
    "function", "let", "new", "delete", "typeof", "instanceof", "extends", "implements", "static", "switch", "case", "default", "export",
    "void", "int", "long", "float", "double", "char", "boolean", "byte", "short", "struct", "union", "unsigned", "signed", "sizeof",
    "func", "go", "defer", "chan", "map", "range", "select", "type", "fn", "mut", "impl", "trait", "pub", "use", "mod", "match", "where", "loop",
    "SELECT", "FROM", "WHERE", "INSERT", "UPDATE", "DELETE", "JOIN", "ON", "GROUP", "ORDER", "BY", "AS", "AND", "OR",
)
private val LITERALS = setOf("true", "false", "null", "None", "True", "False", "nil", "undefined", "NaN")
private val TITLE_AFTER = setOf("fun", "def", "function", "func", "fn", "class", "interface", "object", "struct", "enum", "trait")

/** A small language-agnostic highlighter using the highlight.js theme colors. */
internal fun highlightCode(code: String, language: String, colors: SyntaxColors): AnnotatedString = buildAnnotatedString {
    val lang = language.lowercase()
    val hashComments = lang in setOf("python", "py", "bash", "sh", "shell", "zsh", "ruby", "rb", "yaml", "yml", "toml", "r", "perl", "pl", "")
    val slashComments = lang !in setOf("python", "py", "bash", "sh", "shell", "zsh", "yaml", "yml", "toml", "")
    if (lang in setOf("text", "plaintext", "txt", "")) { append(code); return@buildAnnotatedString }
    var i = 0
    var previousWord = ""
    while (i < code.length) {
        val c = code[i]
        val rest = code.substring(i)
        when {
            slashComments && rest.startsWith("//") -> {
                val end = code.indexOf('\n', i).let { if (it < 0) code.length else it }
                withStyle(SpanStyle(color = colors.comment, fontStyle = FontStyle.Italic)) { append(code.substring(i, end)) }
                i = end
            }
            slashComments && rest.startsWith("/*") -> {
                val end = code.indexOf("*/", i + 2).let { if (it < 0) code.length else it + 2 }
                withStyle(SpanStyle(color = colors.comment, fontStyle = FontStyle.Italic)) { append(code.substring(i, end)) }
                i = end
            }
            hashComments && c == '#' -> {
                val end = code.indexOf('\n', i).let { if (it < 0) code.length else it }
                withStyle(SpanStyle(color = colors.comment, fontStyle = FontStyle.Italic)) { append(code.substring(i, end)) }
                i = end
            }
            c == '"' || c == '\'' || c == '`' -> {
                var end = i + 1
                while (end < code.length && code[end] != c && code[end] != '\n') { if (code[end] == '\\') end++; end++ }
                end = (end + 1).coerceAtMost(code.length)
                withStyle(SpanStyle(color = colors.string)) { append(code.substring(i, end)) }
                i = end
            }
            c.isDigit() && (i == 0 || !code[i - 1].isLetterOrDigit() && code[i - 1] != '_') -> {
                var end = i
                while (end < code.length && (code[end].isLetterOrDigit() || code[end] == '.' || code[end] == '_')) end++
                withStyle(SpanStyle(color = colors.number)) { append(code.substring(i, end)) }
                i = end
            }
            c.isLetter() || c == '_' -> {
                var end = i
                while (end < code.length && (code[end].isLetterOrDigit() || code[end] == '_')) end++
                val word = code.substring(i, end)
                when {
                    word in KEYWORDS -> withStyle(SpanStyle(color = colors.keyword)) { append(word) }
                    word in LITERALS -> withStyle(SpanStyle(color = colors.literal)) { append(word) }
                    previousWord in TITLE_AFTER -> withStyle(SpanStyle(color = if (previousWord == "fun" || previousWord == "def" ||
                        previousWord == "function" || previousWord == "func" || previousWord == "fn") colors.title else colors.builtIn)) { append(word) }
                    else -> append(word)
                }
                previousWord = word
                i = end
                continue
            }
            else -> { append(c); i++ }
        }
        if (!c.isWhitespace()) previousWord = ""
    }
}

/** Web `buildChatErrorBubbleHtml`: `text-red-400 text-xs border border-red-500 p-2 rounded` with a warning glyph. */
/** `.chat-error-box`: also drawn for a stream `error` event (`buildChatErrorBubbleHtml`). */
@Composable
internal fun ChatErrorBlock(text: String) {
    val shape = RoundedCornerShape(4.dp)
    Row(
        Modifier
            .fillMaxWidth()
            .border(1.dp, Tw.red500, shape)
            .padding(8.dp),
    ) {
        FaIcon(R.drawable.fa_solid_triangle_exclamation, null, size = 12.dp, tint = Tw.red400, modifier = Modifier.padding(top = 2.dp, end = 4.dp))
        Text("Error: ${text.trim().ifBlank { "Unknown error" }}", color = Tw.red400, fontSize = 12.sp, lineHeight = 16.sp)
    }
}

@Composable
private fun MathBlock(block: MarkdownBlock.Math, style: TextStyle) {
    val rendered = remember(block.tex) { latexToDisplay(block.tex) }
    Box(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()).padding(vertical = 15.2.dp), contentAlignment = Alignment.Center) {
        Text(rendered, style = style.copy(fontFamily = FontFamily.Serif, fontStyle = FontStyle.Italic), textAlign = TextAlign.Center)
    }
}

// ---------------------------------------------------------------------------------------------
// Tables (auto layout like a 100%-wide HTML table)
// ---------------------------------------------------------------------------------------------

@Composable
private fun TableBlock(block: MarkdownBlock.Table, colors: MarkdownColors) {
    val columnCount = maxOf(block.headers.size, block.rows.maxOfOrNull { it.size } ?: 0)
    if (columnCount == 0) return
    val cellStyle = TextStyle(
        color = colors.text, fontSize = 13.68.sp, lineHeight = 23.53.sp, letterSpacing = 0.01.em, fontFamily = WebFonts.sans,
    )
    val headStyle = cellStyle.copy(color = colors.tableHeadText, fontWeight = FontWeight.SemiBold)
    val rows = listOf(block.headers) + block.rows
    val geometry = remember { TableGeometry() }
    val density = LocalDensity.current
    val border = with(density) { 1.dp.toPx() }
    BoxWithConstraints(Modifier.fillMaxWidth()) {
        val available = with(density) { maxWidth.toPx() }.toInt()
        val scroll = rememberScrollState()
        Box(Modifier.horizontalScroll(scroll)) {
            Layout(
                content = {
                    rows.forEachIndexed { rowIndex, row ->
                        for (column in 0 until columnCount) {
                            val cell = row.getOrNull(column).orEmpty()
                            Box(Modifier.padding(horizontal = 13.6.dp, vertical = 9.6.dp)) {
                                InlineText(cell, if (rowIndex == 0) headStyle else cellStyle, colors)
                            }
                        }
                    }
                },
                modifier = Modifier.drawBehind {
                    drawRect(colors.table, size = Size(geometry.width.toFloat(), geometry.height.toFloat()))
                    if (geometry.rowTops.size > 1) {
                        drawRect(colors.tableHead, size = Size(geometry.width.toFloat(), geometry.rowTops[1].toFloat()))
                    }
                    geometry.rowTops.forEach { y -> drawRect(colors.tableBorder, Offset(0f, y.toFloat()), Size(geometry.width.toFloat(), border)) }
                    drawRect(colors.tableBorder, Offset(0f, geometry.height - border), Size(geometry.width.toFloat(), border))
                    geometry.columnLefts.forEach { x -> drawRect(colors.tableBorder, Offset(x.toFloat(), 0f), Size(border, geometry.height.toFloat())) }
                    drawRect(colors.tableBorder, Offset(geometry.width - border, 0f), Size(border, geometry.height.toFloat()))
                },
            ) { measurables, _ ->
                val maxWidths = IntArray(columnCount)
                val minWidths = IntArray(columnCount)
                measurables.forEachIndexed { index, measurable ->
                    val column = index % columnCount
                    maxWidths[column] = maxOf(maxWidths[column], measurable.maxIntrinsicWidth(Constraints.Infinity))
                    minWidths[column] = maxOf(minWidths[column], measurable.minIntrinsicWidth(Constraints.Infinity))
                }
                val widths = tableColumnWidths(minWidths, maxWidths, available - border.toInt())
                val placeables = measurables.mapIndexed { index, measurable ->
                    val w = widths[index % columnCount]
                    measurable.measure(Constraints(minWidth = w, maxWidth = w))
                }
                val rowHeights = (0 until rows.size).map { row ->
                    (0 until columnCount).maxOf { column -> placeables[row * columnCount + column].height }
                }
                val totalWidth = widths.sum() + border.toInt()
                val totalHeight = rowHeights.sum() + border.toInt()
                geometry.width = totalWidth
                geometry.height = totalHeight
                geometry.rowTops = rowHeights.runningFold(0) { acc, h -> acc + h }.dropLast(1)
                geometry.columnLefts = widths.toList().runningFold(0) { acc, w -> acc + w }.dropLast(1)
                layout(totalWidth, totalHeight) {
                    var y = 0
                    rowHeights.forEachIndexed { row, height ->
                        var x = 0
                        for (column in 0 until columnCount) {
                            placeables[row * columnCount + column].placeRelative(x, y)
                            x += widths[column]
                        }
                        y += height
                    }
                }
            }
        }
    }
}

private class TableGeometry {
    var width = 0
    var height = 0
    var rowTops: List<Int> = emptyList()
    var columnLefts: List<Int> = emptyList()
}

/**
 * Column widths of an auto-layout table stretched to [available]: extra space follows the
 * max-content widths; when content does not fit, columns shrink toward their min-content width.
 */
internal fun tableColumnWidths(minWidths: IntArray, maxWidths: IntArray, available: Int): IntArray {
    val sumMax = maxWidths.sum()
    val sumMin = minWidths.sum()
    if (sumMax <= 0) return IntArray(maxWidths.size) { available / maxOf(1, maxWidths.size) }
    return when {
        sumMax <= available -> IntArray(maxWidths.size) { i -> maxWidths[i] * available / sumMax }
            .also { widths -> widths[widths.lastIndex] += available - widths.sum() }
        sumMin < available -> {
            val extra = available - sumMin
            val flexible = (sumMax - sumMin).coerceAtLeast(1)
            IntArray(maxWidths.size) { i -> minWidths[i] + (maxWidths[i] - minWidths[i]) * extra / flexible }
                .also { widths -> widths[widths.lastIndex] += available - widths.sum() }
        }
        else -> minWidths.copyOf()
    }
}

/** File name of the code-block download (Web `.download-btn` handler: `code.<ext>`, Dockerfile, Makefile). */
internal fun codeDownloadName(language: String): String {
    val lang = language.lowercase().ifBlank { "txt" }
    if (lang == "dockerfile") return "Dockerfile"
    if (lang == "makefile") return "Makefile"
    val map = mapOf(
        "python" to "py", "javascript" to "js", "typescript" to "ts", "markdown" to "md",
        "html" to "html", "css" to "css", "json" to "json", "xml" to "xml", "sql" to "sql",
        "bash" to "sh", "sh" to "sh", "shell" to "sh", "zsh" to "sh",
        "c" to "c", "cpp" to "cpp", "csharp" to "cs", "cs" to "cs",
        "java" to "java", "kotlin" to "kt", "swift" to "swift",
        "go" to "go", "rust" to "rs", "ruby" to "rb", "php" to "php",
        "perl" to "pl", "lua" to "lua", "r" to "r", "matlab" to "m",
        "yaml" to "yaml", "yml" to "yaml", "toml" to "toml", "ini" to "ini",
        "plaintext" to "txt", "text" to "txt",
    )
    var ext = map[lang] ?: lang
    if (lang.length > 8 || Regex("[^a-z0-9]").containsMatchIn(lang)) ext = "txt"
    return "code.$ext"
}

/** Provides the code-block actions to every [MarkdownText] below. */
@Composable
internal fun ProvideMarkdownCodeActions(actions: MarkdownCodeActions, content: @Composable () -> Unit) {
    CompositionLocalProvider(LocalMarkdownCodeActions provides actions, content = content)
}

/** Kept for callers that only need the plain body color. */
@Composable
internal fun markdownTextColor(): Color = markdownColors().text

@Suppress("unused")
private fun widthCap(): Modifier = Modifier.widthIn(max = 832.dp)
