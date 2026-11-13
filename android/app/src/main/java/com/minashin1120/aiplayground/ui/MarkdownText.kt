package com.minashin1120.aiplayground.ui

import androidx.compose.animation.AnimatedVisibility
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
import androidx.compose.foundation.shape.CircleShape
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
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.saveable.rememberSaveable
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
import androidx.compose.ui.layout.layout
import androidx.compose.ui.unit.takeOrElse
import androidx.compose.ui.graphics.takeOrElse
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.layout.positionInWindow
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.platform.UriHandler
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.LinkAnnotation
import androidx.compose.ui.text.Placeholder
import androidx.compose.ui.text.PlaceholderVerticalAlign
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextLayoutResult
import androidx.compose.ui.text.ParagraphStyle
import androidx.compose.ui.text.TextLinkStyles
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.withLink
import androidx.compose.ui.text.style.BaselineShift
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.Constraints
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.codingTargetKey
import com.minashin1120.aiplayground.data.CANVAS_PLACEHOLDER_LINE
import com.minashin1120.aiplayground.data.CANVAS_PLACEHOLDER_TEXT
import com.minashin1120.aiplayground.data.parseCanvasMarkdown
import com.minashin1120.aiplayground.data.fileReferencePath
import kotlinx.coroutines.delay

/*
 * Markdown as the Web chat renders it: `marked` (GFM, `breaks: true`) with the custom code renderer
 * of chat_core part08, inside a `white-space: pre-wrap` `.content-area`. Because of pre-wrap, the
 * newline marked writes after each block (not after code blocks) shows as an empty line, which is
 * why blocks are separated by a margin plus one line box.
 */

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
 * Text that should flash after jumping to a quote: the selection to find, the current pulse alpha and
 * the callback that reports where the selection was laid out (window Y), so the jump can put it under
 * the top bar instead of wherever the previous reader left it.
 */
internal data class QuoteFlash(
    val query: String,
    val alpha: Float,
    val onLocated: ((Float) -> Unit)? = null,
)

internal val LocalQuoteFlash = compositionLocalOf<QuoteFlash?> { null }

/** The translucent yellow used for the quote's flash; the bubble background uses the same color. */
internal val QuoteFlashColor = Color(0xFFFDE047)

/** [text] with the part matching [quote] flashing; a no-op AnnotatedString when there is no match. */
internal fun quoteFlashText(text: String, quote: String, alpha: Float): AnnotatedString {
    if (alpha <= 0f) return AnnotatedString(text)
    val range = findQuoteRange(text, quote) ?: return AnnotatedString(text)
    return AnnotatedString(
        text,
        spanStyles = listOf(AnnotatedString.Range(
            SpanStyle(background = QuoteFlashColor.copy(alpha = 0.5f * alpha)),
            range.first,
            range.last + 1,
        )),
    )
}

/**
 * Markdown message body. [startCollapsed] mirrors Web: stored answers show code blocks collapsed,
 * while an answer that is still streaming keeps them open.
 */
@Composable
fun MarkdownText(
    text: String,
    loader: FileBytesLoader? = null,
    onOpen: (String) -> Unit = {},
    startCollapsed: Boolean = true,
    /** Web Canvas mode: fenced blocks are shown in the Canvas panel and replaced here. */
    canvasMode: Boolean = false,
    /** The quoted selection to flash after a quote jump, and the pulse alpha (0 when idle). */
    flashQuote: String? = null,
    flashAlpha: Float = 0f,
    /** Reports the quoted range's window Y once it is laid out, so the jump can align it. */
    onQuoteFlashLocated: ((Float) -> Unit)? = null,
) {
    val blocks = remember(text, canvasMode) { parseMarkdownBlocks(if (canvasMode) parseCanvasMarkdown(text).renderText else text) }
    val colors = markdownColors()
    val base = TextStyle(
        color = colors.text, fontSize = MarkdownBodySize, lineHeight = MarkdownLineHeight,
        letterSpacing = 0.01.em, fontFamily = WebFonts.sans,
    )
    // Web opens every link in a new tab; here a link that no installed app can open is ignored.
    val uriHandler = LocalUriHandler.current
    val safeHandler = remember(uriHandler) { SafeUriHandler(uriHandler) }
    val flash = if (flashAlpha > 0f && !flashQuote.isNullOrBlank())
        QuoteFlash(flashQuote, flashAlpha, onQuoteFlashLocated) else null
    CompositionLocalProvider(LocalUriHandler provides safeHandler, LocalQuoteFlash provides flash) {
        PreWrapBlocks(blocks, base, colors, loader, onOpen, startCollapsed, leadingLine = false)
    }
}

private class SafeUriHandler(private val base: UriHandler) : UriHandler {
    override fun openUri(uri: String) { runCatching { base.openUri(uri) } }
}

/** Height of one pre-wrap line box (the empty line produced by marked's `\n` between blocks). */
@Composable
private fun lineBox(style: TextStyle): Dp = with(LocalDensity.current) { style.lineHeight.toDp() }

private fun MarkdownBlock.marginTop(): Dp = when (this) {
    is MarkdownBlock.Quote, is MarkdownBlock.Table -> ParagraphGap
    is MarkdownBlock.Code -> CodeGap
    is MarkdownBlock.ChatError -> 8.dp
    MarkdownBlock.Rule -> 24.dp
    MarkdownBlock.CanvasPlaceholder -> 14.4.dp
    is MarkdownBlock.Svg -> 11.2.dp
    is MarkdownBlock.Details -> 0.dp
    else -> 0.dp
}

private fun MarkdownBlock.marginBottom(): Dp = when (this) {
    is MarkdownBlock.Paragraph, is MarkdownBlock.ListBlock, is MarkdownBlock.Quote, is MarkdownBlock.Table,
    is MarkdownBlock.Image, is MarkdownBlock.Math, is MarkdownBlock.Audio -> ParagraphGap
    is MarkdownBlock.Code -> CodeGap
    MarkdownBlock.Rule -> 24.dp
    is MarkdownBlock.Heading, is MarkdownBlock.ChatError -> 0.dp
    MarkdownBlock.CanvasPlaceholder -> 14.4.dp
    is MarkdownBlock.Svg -> 11.2.dp
    is MarkdownBlock.Details -> ParagraphGap
}

/** Code blocks come from a custom renderer that does not end with a newline. */
private fun MarkdownBlock.followedByNewline(): Boolean =
    this !is MarkdownBlock.Code && this !is MarkdownBlock.ChatError && !(this is MarkdownBlock.Svg && fromCode)

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
            // `.chat-image-frame`: up to 320px wide with the image's own height.
            fitToImage = true, fitMaxWidth = 320.dp,
        )
        MarkdownBlock.Rule -> Box(Modifier.fillMaxWidth().height(1.dp).background(colors.rule))
        MarkdownBlock.CanvasPlaceholder -> CanvasPlaceholderPill(style)
        is MarkdownBlock.Svg -> SvgRenderBox(block.markup, style, block.fromCode)
        is MarkdownBlock.Details -> DetailsBlock(block, style, colors, loader, onOpen, startCollapsed)
        is MarkdownBlock.Audio -> InlineAudioPlayer(block.src, onOpen)
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
internal data class InlineMarkdown(val text: AnnotatedString, val codeRanges: List<IntRange>, val math: List<InlineMath> = emptyList())

/** A formula placeholder at [start] in the text; its alternate text is the [latexToDisplay] approximation ([alternate]). */
internal data class InlineMath(val id: String, val tex: String, val display: Boolean, val start: Int, val alternate: String = "")

/**
 * [InlineMarkdown] with each typeset formula ([ready] ids) reduced to its [MathCopyText] token, so a long press
 * selects the formula whole. Formulas still loading keep their approximation as text.
 */
internal fun InlineMarkdown.withMathTokens(ready: Set<String>): InlineMarkdown {
    if (math.none { it.id in ready && it.alternate.isNotEmpty() }) return this
    val source = text
    val formulas = math
    var cursor = 0
    var shift = 0
    // Original end offset of each swapped formula and the total shift from there on.
    val shifts = mutableListOf<Pair<Int, Int>>()
    val moved = mutableListOf<InlineMath>()
    val built = buildAnnotatedString {
        formulas.forEach { item ->
            val placed = item.copy(start = item.start + shift)
            if (item.id !in ready || item.alternate.isEmpty()) { moved += placed; return@forEach }
            val end = item.start + item.alternate.length
            append(source.subSequence(cursor, item.start))
            // The styles over the whole formula (its serif italic, an enclosing bold or heading size) size the placeholder.
            val piece = source.subSequence(item.start, end)
            val styles = piece.spanStyles.filter { it.start == 0 && it.end == piece.length }
            styles.forEach { pushStyle(it.item) }
            val token = MathCopyText.token(item.alternate)
            appendInlineContent(item.id, token)
            repeat(styles.size) { pop() }
            moved += placed
            shift += token.length - item.alternate.length
            shifts += end to shift
            cursor = end
        }
        append(source.subSequence(cursor, source.length))
    }
    fun shiftAt(offset: Int) = shifts.lastOrNull { it.first <= offset }?.second ?: 0
    return InlineMarkdown(built, codeRanges.map { (it.first + shiftAt(it.first))..(it.last + shiftAt(it.first)) }, moved)
}

private class InlineParts {
    val codeRanges = mutableListOf<IntRange>()
    val math = mutableListOf<InlineMath>()
}

private const val MATH_PREFIX = "math:"
private val BEGIN_ENV = Regex("^\\\\begin\\{([a-zA-Z*]+)\\}")

private val AUTOLINK = Regex("^<(https?://[^>\\s]+)>")
private val BARE_URL = Regex("^(https?://|www\\.)[^\\s<]*[^\\s<?!.,:*_~)\\]]")
private val ESCAPABLE = "\\`*_{}[]()#+-.!|~<>$"

internal fun parseInlineMarkdown(source: String, colors: MarkdownColors): InlineMarkdown {
    val parts = InlineParts()
    val text = buildAnnotatedString { appendInline(source, colors, parts) }
    return InlineMarkdown(text, parts.codeRanges, parts.math)
}

private fun AnnotatedString.Builder.appendInline(source: String, colors: MarkdownColors, parts: InlineParts) {
    var i = 0
    val plain = StringBuilder()
    fun flush() { if (plain.isNotEmpty()) { append(plain.toString()); plain.clear() } }
    // MathJax draws the formula in place of its text approximation (shown while loading or if it fails).
    fun math(tex: String, display: Boolean) {
        flush()
        val id = MATH_PREFIX + parts.math.size
        val alternate = latexToDisplay(tex).ifEmpty { " " }
        parts.math += InlineMath(id, tex, display, length, alternate)
        withStyle(SpanStyle(fontFamily = FontFamily.Serif, fontStyle = FontStyle.Italic)) {
            // Add the placeholder only after typesetting in withMathTokens. Copying slices
            // with existing inline annotations leaves zero-length spans at formula boundaries;
            // Android never measures those spans and Compose crashes reading their width.
            append(alternate)
        }
    }
    while (i < source.length) {
        if (source.startsWith("\\\\(", i) || source.startsWith("\\\\[", i)) {
            val display = source[i + 2] == '['
            val closer = if (display) "\\\\]" else "\\\\)"
            val end = source.indexOf(closer, i + 3)
            if (end > i + 3) {
                math(source.substring(i + 3, end), display)
                i = end + 3
                continue
            }
        }
        val c = source[i]
        val rest = source.substring(i)
        // Escapes
        if (c == '\\' && i + 1 < source.length && source[i + 1] in ESCAPABLE && !rest.startsWith("\\(") && !rest.startsWith("\\[")) {
            plain.append(source[i + 1]); i += 2; continue
        }
        // Display math $$ … $$ / \[ … \] inside a paragraph: MathJax sets it on its own centered line.
        val closer = when {
            rest.startsWith("$$") -> "$$"
            rest.startsWith("\\[") -> "\\]"
            else -> null
        }
        if (closer != null) {
            val end = source.indexOf(closer, i + 2)
            if (end > i + 2 && source.substring(i + 2, end).isNotBlank()) {
                flush()
                withStyle(ParagraphStyle(textAlign = TextAlign.Center)) { math(source.substring(i + 2, end), true) }
                i = end + 2; continue
            }
        }
        // Environments outside delimiters (Web protects `\begin{env}…\end{env}` and MathJax sets them as display math).
        if (rest.startsWith("\\begin{")) {
            val env = BEGIN_ENV.find(rest)
            if (env != null) {
                val close = "\\end{${env.groupValues[1]}}"
                val end = source.indexOf(close, i + env.value.length)
                if (end > i + env.value.length) {
                    flush()
                    withStyle(ParagraphStyle(textAlign = TextAlign.Center)) { math(source.substring(i, end + close.length), true) }
                    i = end + close.length; continue
                }
            }
        }
        // Inline math \( … \); like Web, a full-width \（ or \） from the model also delimits it.
        if (rest.startsWith("\\(") || rest.startsWith("\\（")) {
            val end = listOf(source.indexOf("\\)", i + 2), source.indexOf("\\）", i + 2)).filter { it >= 0 }.minOrNull() ?: -1
            if (end > 0) {
                math(source.substring(i + 2, end), false)
                i = end + 2; continue
            }
        }
        if (c == '$' && i + 1 < source.length && source[i + 1] != ' ' && source[i + 1] != '$') {
            val end = source.indexOf('$', i + 1)
            if (end > i + 1 && source[end - 1] != ' ' && !source.substring(i + 1, end).contains('\n')) {
                math(source.substring(i + 1, end), false)
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
                parts.codeRanges += startIndex until length
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
                val url = markdownLinkTarget(match.groupValues[2])
                if (url == null) appendInline(match.groupValues[1], colors, parts)
                else withLink(LinkAnnotation.Url(url, TextLinkStyles(SpanStyle(color = colors.link, textDecoration = TextDecoration.Underline)))) {
                    appendInline(match.groupValues[1], colors, parts)
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
        if (c == '<') {
            val consumed = appendInlineHtml(rest, colors, parts, ::flush)
            if (consumed > 0) { i += consumed; continue }
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
                withStyle(SpanStyle(fontWeight = FontWeight.Bold)) { appendInline(source.substring(i + 2, end), colors, parts) }
                i = end + 2; continue
            }
        }
        if (rest.startsWith("~~")) {
            val end = source.indexOf("~~", i + 2)
            if (end > i + 2) {
                flush()
                withStyle(SpanStyle(textDecoration = TextDecoration.LineThrough)) { appendInline(source.substring(i + 2, end), colors, parts) }
                i = end + 2; continue
            }
        }
        if ((c == '*' || (c == '_' && !previous.isLetterOrDigit())) && i + 1 < source.length && !source[i + 1].isWhitespace()) {
            var end = source.indexOf(c, i + 1)
            while (end > 0 && end + 1 < source.length && source[end + 1] == c) end = source.indexOf(c, end + 2)
            if (end > i + 1 && !source[end - 1].isWhitespace() && (c == '*' || end + 1 >= source.length || !source[end + 1].isLetterOrDigit())) {
                flush()
                withStyle(SpanStyle(fontStyle = FontStyle.Italic)) { appendInline(source.substring(i + 1, end), colors, parts) }
                i = end + 1; continue
            }
        }
        plain.append(c)
        i++
    }
    flush()
}

private val HTML_BREAK = Regex("^<br\\s*/?>", RegexOption.IGNORE_CASE)
private val HTML_COMMENT = Regex("^<!--[\\s\\S]*?-->")
private val HTML_DROPPED = Regex("^<(script|style|iframe|object|embed|noscript|template)\\b[^>]*>[\\s\\S]*?</\\1\\s*>", RegexOption.IGNORE_CASE)
private val HTML_OPEN = Regex("^<([A-Za-z][A-Za-z0-9-]*)((?:\\s+[^<>]*?)?)\\s*(/?)>")
private val HTML_CLOSE = Regex("^</[A-Za-z][A-Za-z0-9-]*\\s*>")
private val HTML_HREF = Regex("href\\s*=\\s*(\"([^\"]*)\"|'([^']*)'|([^\\s>]+))", RegexOption.IGNORE_CASE)

/**
 * Raw HTML inside an answer, as Web shows it after DOMPurify: `<br>`, the text styles, links and code are
 * drawn; script-like elements vanish with their content; other tags are dropped and their text kept.
 * Returns the characters consumed, or 0 when [rest] does not start with a tag.
 */
private fun AnnotatedString.Builder.appendInlineHtml(rest: String, colors: MarkdownColors, parts: InlineParts, flush: () -> Unit): Int {
    HTML_BREAK.find(rest)?.let { flush(); append("\n"); return it.value.length }
    HTML_COMMENT.find(rest)?.let { flush(); return it.value.length }
    HTML_DROPPED.find(rest)?.let { flush(); return it.value.length }
    val open = HTML_OPEN.find(rest)
    if (open != null) {
        val name = open.groupValues[1].lowercase()
        val closing = Regex("</${Regex.escape(name)}\\s*>", RegexOption.IGNORE_CASE).find(rest, open.value.length)
        val style: SpanStyle? = when (name) {
            "b", "strong" -> SpanStyle(fontWeight = FontWeight.Bold)
            "i", "em", "cite", "var" -> SpanStyle(fontStyle = FontStyle.Italic)
            "u", "ins" -> SpanStyle(textDecoration = TextDecoration.Underline)
            "s", "del", "strike" -> SpanStyle(textDecoration = TextDecoration.LineThrough)
            "sup" -> SpanStyle(baselineShift = BaselineShift.Superscript, fontSize = 0.75.em)
            "sub" -> SpanStyle(baselineShift = BaselineShift.Subscript, fontSize = 0.75.em)
            "small" -> SpanStyle(fontSize = 0.83.em)
            "mark" -> SpanStyle(background = Color(0xFFFEF08A), color = Color(0xFF111827))
            "code", "kbd", "samp", "tt" -> SpanStyle(fontFamily = WebFonts.mono, fontSize = InlineCodeSize, color = colors.inlineCode)
            else -> null
        }
        if (open.groupValues[3].isEmpty() && closing != null && (style != null || name == "a")) {
            flush()
            val inner = rest.substring(open.value.length, closing.range.first)
            val href = if (name == "a") HTML_HREF.find(open.groupValues[2])?.let { m -> m.groupValues.drop(2).firstOrNull { it.isNotEmpty() } }
                ?.let(::markdownLinkTarget) else null
            when {
                href != null -> withLink(LinkAnnotation.Url(href, TextLinkStyles(SpanStyle(color = colors.link, textDecoration = TextDecoration.Underline)))) {
                    appendInline(inner, colors, parts)
                }
                style != null -> withStyle(style) { appendInline(inner, colors, parts) }
                else -> appendInline(inner, colors, parts)
            }
            return closing.range.last + 1
        }
        flush()
        return open.value.length
    }
    HTML_CLOSE.find(rest)?.let { flush(); return it.value.length }
    return 0
}

@Composable
private fun InlineText(text: String, style: TextStyle, colors: MarkdownColors) {
    val parsed = remember(text, colors) { parseInlineMarkdown(text, colors) }
    val flash = LocalQuoteFlash.current
    val range = remember(parsed, flash?.query) {
        flash?.let { findQuoteRange(parsed.text.text, it.query) }
    }
    val flashed = remember(parsed, flash?.alpha, range) {
        if (flash == null || range == null) parsed
        else InlineMarkdown(
            text = AnnotatedString(
                parsed.text.text,
                spanStyles = parsed.text.spanStyles + AnnotatedString.Range(
                    SpanStyle(background = QuoteFlashColor.copy(alpha = 0.5f * flash.alpha)),
                    range.first,
                    range.last + 1,
                ),
            ),
            codeRanges = parsed.codeRanges,
            math = parsed.math,
        )
    }
    InlineRichText(
        flashed,
        style,
        colors.inlineCodeBackground,
        quoteRange = range,
        onQuoteLocated = flash?.onLocated,
    )
}

/** True inside a bubble-wide selection, so nested text joins it instead of starting its own. */
private val LocalMessageSelection = staticCompositionLocalOf { false }

/** Joins the bubble selection, except for text in an independently scrolling viewport. */
@Composable
internal fun MessageSelection(independent: Boolean = false, content: @Composable () -> Unit) {
    if (LocalMessageSelection.current && !independent) content()
    else {
        val trace = LocalInteractionTrace.current ?: rememberInteractionTrace("message")
        ProvideMathCopy {
            SelectionContainer(modifier = Modifier.observeInteraction(trace)) {
                CompositionLocalProvider(LocalMessageSelection provides true, LocalInteractionTrace provides trace, content = content)
            }
        }
    }
}

/** Text with Web inline-code chips: rounded 6.4px background, 6.4px side padding, 24px tall. */
@Composable
internal fun InlineRichText(
    source: InlineMarkdown,
    style: TextStyle,
    codeBackground: Color,
    modifier: Modifier = Modifier,
    maxLines: Int = Int.MAX_VALUE,
    overflow: TextOverflow = TextOverflow.Clip,
    // One-line controls such as the quote bar keep their own tap handling instead
    // of starting a text selection inside the strip.
    selectable: Boolean = true,
    /** The quoted range to report the window position of, and where to report it (quote jump). */
    quoteRange: IntRange? = null,
    onQuoteLocated: ((Float) -> Unit)? = null,
) {
    var textLayout by remember { mutableStateOf<TextLayoutResult?>(null) }
    var windowTop by remember { mutableFloatStateOf(Float.NaN) }
    val density = LocalDensity.current
    val hasMath = source.math.isNotEmpty()
    var availableWidth by remember { mutableIntStateOf(Constraints.Infinity) }
    val mathStates = source.math.map { item -> key(item.id, item.tex) { rememberMathState(item.tex, item.display) } }
    val ready = source.math.filterIndexed { index, _ -> mathStates[index] is MathState.Ready }.map { it.id }.toSet()
    val parsed = remember(source, ready) { source.withMathTokens(ready) }
    val mathColor = style.color.takeOrElse { LocalContentColor.current }
    val fontPx = with(density) { style.fontSize.takeOrElse { 16.sp }.toPx() }
    val inline = remember(parsed.math, mathStates, mathColor, availableWidth, fontPx) {
        val map = HashMap<String, InlineTextContent>()
        map[CODE_PAD] = InlineTextContent(Placeholder(6.4.sp, 1.sp, PlaceholderVerticalAlign.TextCenter)) {}
        parsed.math.forEachIndexed { index, item ->
            val svg = (mathStates[index] as? MathState.Ready)?.svg ?: return@forEachIndexed
            map[item.id] = inlineMathContent(item, svg, mathColor, availableWidth, fontPx) { textLayout }
        }
        map
    }
    val chipHeight = with(density) { 24.dp.toPx() }
    val radius = with(density) { 6.4.dp.toPx() }
    // Quote jump: once this block is laid out, tell the conversation where its quoted range starts.
    LaunchedEffect(textLayout, windowTop, quoteRange, onQuoteLocated) {
        val layout = textLayout ?: return@LaunchedEffect
        val range = quoteRange ?: return@LaunchedEffect
        val report = onQuoteLocated ?: return@LaunchedEffect
        if (!windowTop.isFinite() || range.first < 0 || range.first >= layout.layoutInput.text.length) return@LaunchedEffect
        report(windowTop + layout.getBoundingBox(range.first).top)
    }
    val body: @Composable () -> Unit = {
        Text(
            parsed.text,
            style = style,
            inlineContent = inline,
            onTextLayout = { textLayout = it },
            maxLines = maxLines,
            overflow = overflow,
            modifier = modifier
                .onGloballyPositioned { windowTop = it.positionInWindow().y }
                // Screen readers get the approximations, not the formula tokens.
                .then(if (parsed === source) Modifier else Modifier.semantics { contentDescription = source.text.text })
                .then(
                    if (!hasMath) Modifier
                    // Formulas wider than the bubble shrink to fit, so remember the width they get.
                    else Modifier.layout { measurable, constraints ->
                        if (constraints.maxWidth != availableWidth) availableWidth = constraints.maxWidth
                        val placeable = measurable.measure(constraints)
                        layout(placeable.width, placeable.height) { placeable.place(0, 0) }
                    },
                )
                .drawBehind {
                val result = textLayout ?: return@drawBehind
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
    if (selectable) MessageSelection { body() } else body()
}

// ---------------------------------------------------------------------------------------------
// Code blocks
// ---------------------------------------------------------------------------------------------

@Composable
private fun CodeBlock(block: MarkdownBlock.Code, colors: MarkdownColors, startCollapsed: Boolean) {
    // Web renders new code blocks collapsed and reapplies their prior state by block position on each stream update.
    // The parent keys blocks by position, so keep this state while the code text changes and the live row is handed off.
    var collapsed by rememberSaveable { mutableStateOf(startCollapsed) }
    var previewOpen by rememberSaveable { mutableStateOf(false) }
    var copied by remember { mutableStateOf<Boolean?>(null) }
    LaunchedEffect(copied) { if (copied != null) { delay(2000); copied = null } }
    val clipboard = LocalClipboardManager.current
    val actions = LocalMarkdownCodeActions.current
    val shape = RoundedCornerShape(16.dp)
    if (previewOpen) HtmlCodePreviewDialog(block.text) { previewOpen = false }
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
            if (isHtmlPreviewCandidate(block.language, block.text)) {
                CodeActionButton(R.drawable.fa_solid_up_right_from_square, "プレビュー", colors) { previewOpen = true }
            }
            CodeActionButton(R.drawable.fa_solid_download, "ダウンロード", colors) { actions.onDownload(block.text, block.language) }
            CodeActionButton(
                when (copied) { true -> R.drawable.fa_solid_check; false -> R.drawable.fa_solid_times; null -> R.drawable.fa_solid_copy },
                "コピー", colors,
            ) {
                copied = runCatching { clipboard.setText(AnnotatedString(block.text)) }.isSuccess
            }
        }
        // Web `.code-body`: max-height/opacity/padding animate on the collapse toggle.
        AnimatedVisibility(
            visible = !collapsed,
            enter = expandFadeIn(LocalReduceMotion.current),
            exit = shrinkFadeOut(LocalReduceMotion.current),
        ) {
            Column {
                Box(Modifier.fillMaxWidth().height(1.dp).background(colors.codeWrapperBorder))
                val highlighted = remember(block.text, block.language, colors.syntax) { highlightCode(block.text, block.language, colors.syntax) }
                MessageSelection {
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
    val state = rememberMathState(block.tex, true)
    val color = style.color.takeOrElse { LocalContentColor.current }
    val svg = (state as? MathState.Ready)?.svg
    val drawing = svg?.let { rememberMathDrawing(it, color) }
    // The readable text approximation: shown while MathJax loads or if it cannot set the formula, and copied otherwise.
    val rendered = remember(block.tex) { latexToDisplay(block.tex) }
    val scroll = rememberScrollState()
    val trace = rememberInteractionTrace("math_block")
    val naturalWidth = with(LocalDensity.current) { svg?.let { (it.widthEm * WEB_MATH_SCALE * style.fontSize.takeOrElse { 16.sp }.toPx()).toInt() } }
    TraceMathRendering(trace, state, scroll, naturalWidth, WEB_MATH_SCALE)
    Box(Modifier.fillMaxWidth().observeInteraction(trace, scroll).horizontalScroll(scroll).padding(vertical = 15.2.dp), contentAlignment = Alignment.Center) {
        if (svg != null && drawing != null) {
            MathDisplayText(svg, drawing, rendered, color, style)
        } else {
            Text(rendered, style = style.copy(fontFamily = FontFamily.Serif, fontStyle = FontStyle.Italic), textAlign = TextAlign.Center)
        }
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
    val hasHeader = block.headers.isNotEmpty()
    val rows = if (hasHeader) listOf(block.headers) + block.rows else block.rows
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
                                InlineText(cell, if (rowIndex == 0 && hasHeader) headStyle else cellStyle, colors)
                            }
                        }
                    }
                },
                modifier = Modifier.drawBehind {
                    drawRect(colors.table, size = Size(geometry.width.toFloat(), geometry.height.toFloat()))
                    if (hasHeader && geometry.rowTops.size > 1) {
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

/** `.canvas-code-placeholder`: a dashed theme pill in place of a fence while Canvas mode is on. */
@Composable
private fun CanvasPlaceholderPill(style: TextStyle) {
    val web = LocalWebPalette.current
    Box(
        Modifier.fillMaxWidth().clip(CircleShape).background(web.theme.rgb(0.12f))
            .drawBehind {
                drawRoundRect(web.theme.rgb(0.5f), cornerRadius = androidx.compose.ui.geometry.CornerRadius(size.height / 2),
                    style = androidx.compose.ui.graphics.drawscope.Stroke(1.dp.toPx(),
                        pathEffect = androidx.compose.ui.graphics.PathEffect.dashPathEffect(floatArrayOf(4.dp.toPx(), 3.dp.toPx()))))
            }
            .padding(horizontal = 16.dp, vertical = 12.8.dp),
        contentAlignment = Alignment.Center,
    ) { Text(CANVAS_PLACEHOLDER_TEXT, style = style.copy(color = web.theme200)) }
}

/** Web `.svg-render-box`: the drawing on a light, checkered card; it never loads external references. */
@Composable
private fun SvgRenderBox(markup: String, style: TextStyle, fromCode: Boolean = false) {
    val svg = remember(markup) {
        if (markup.length > 300_000) null else runCatching { com.caverock.androidsvg.SVG.getFromString(markup) }.getOrNull()
    }
    if (svg == null && !fromCode) { Text(markup, style = style); return }
    val shape = RoundedCornerShape(12.dp)
    val density = LocalDensity.current.density
    BoxWithConstraints(
        Modifier.fillMaxWidth().clip(shape).background(Color(0xFFF8FAFC)).border(1.dp, Color(148, 163, 184).copy(alpha = 0.25f), shape).padding(12.dp),
        contentAlignment = Alignment.Center,
    ) {
        val available = constraints.maxWidth.coerceAtLeast(1)
        // The frame is the size Web gives the drawing: scaled down to the box (and, for a fence
        // without a fixed width, stretched to it). It is shared by the still and animated paths.
        val frame = remember(markup, available, density) {
            if (svg == null) return@remember null
            runCatching {
                val box = svg.documentViewBox
                val docW = svg.documentWidth.takeIf { it > 0 } ?: box?.width() ?: 300f
                val docH = svg.documentHeight.takeIf { it > 0 } ?: box?.height() ?: 150f
                // Web `.svg-code-fill`: a fence's drawing without a fixed width spans the box.
                val fill = fromCode && svg.documentWidth <= 0f
                val scale = (available / (docW * density)).let { if (fill) it else it.coerceAtMost(1f) } * density
                SvgFrame((docW * scale).toInt().coerceIn(1, 4096), (docH * scale).toInt().coerceIn(1, 4096))
            }.getOrNull()
        }
        val animated = remember(markup) { svgHasAnimation(markup) && svgAnimationSafe(markup) }
        val animate = LocalSvgAnimation.current && !LocalReduceMotion.current
        var webViewFailed by remember(markup) { mutableStateOf(false) }
        if (animated && animate && frame != null && !webViewFailed) {
            // AndroidSVG freezes SMIL/CSS animation; WebView plays it like the browser on Web.
            AnimatedSvg(markup, frame.width, frame.height, onError = { webViewFailed = true })
        } else {
            val bitmap = remember(markup, available, density, frame) {
                if (svg == null || frame == null) return@remember null
                runCatching {
                    svg.setDocumentWidth(frame.width.toFloat()); svg.setDocumentHeight(frame.height.toFloat())
                    android.graphics.Bitmap.createBitmap(frame.width, frame.height, android.graphics.Bitmap.Config.ARGB_8888)
                        .also { svg.renderToCanvas(android.graphics.Canvas(it)) }
                }.getOrNull()
            }
            if (bitmap != null) androidx.compose.foundation.Image(bitmap.asImageBitmap(), contentDescription = null)
            // Web `.svg-code-render[data-svg-state="error"]`: the code block below still shows the source.
            else if (fromCode) Text("SVGを描画できませんでした", style = style.copy(fontSize = 12.sp, color = Color(0xFF64748B)), textAlign = TextAlign.Center, modifier = Modifier.fillMaxWidth())
            else Text(markup, style = style)
        }
    }
}

private data class SvgFrame(val width: Int, val height: Int)

/** Browser `<details>`: "▶ summary" toggles the content ("▼" while open). */
@Composable
private fun DetailsBlock(
    block: MarkdownBlock.Details,
    style: TextStyle,
    colors: MarkdownColors,
    loader: FileBytesLoader?,
    onOpen: (String) -> Unit,
    startCollapsed: Boolean,
) {
    var open by remember(block) { mutableStateOf(block.open) }
    Column(Modifier.fillMaxWidth()) {
        Row(
            Modifier.clickable(role = androidx.compose.ui.semantics.Role.Button, onClickLabel = if (open) "閉じる" else "開く") { open = !open },
            verticalAlignment = Alignment.Top,
        ) {
            Text(if (open) "▼ " else "▶ ", style = style.copy(fontSize = style.fontSize * 0.75f))
            InlineText(block.summary, style, colors)
        }
        if (open) PreWrapBlocks(block.blocks, style, colors, loader, onOpen, startCollapsed, leadingLine = false)
    }
}
