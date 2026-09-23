package com.minashin1120.aiplayground.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.*
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.minashin1120.aiplayground.data.fileReferencePath
import java.net.URI

internal sealed interface MarkdownBlock {
    data class Paragraph(val text: String) : MarkdownBlock
    data class Heading(val level: Int, val text: String) : MarkdownBlock
    data class Quote(val text: String) : MarkdownBlock
    data class ListItem(val text: String, val ordered: Boolean) : MarkdownBlock
    data class Code(val language: String, val text: String) : MarkdownBlock
    data class Math(val tex: String, val display: Boolean) : MarkdownBlock
    data class Table(val headers: List<String>, val rows: List<List<String>>) : MarkdownBlock
    data class Image(val reference: String, val alt: String) : MarkdownBlock
    data object Rule : MarkdownBlock
}

private val HEADING = Regex("^(#{1,6})\\s+(.+)$")
private val ORDERED_ITEM = Regex("^\\s*\\d+[.)]\\s+(.+)$")
private val RULE = Regex("^\\s*([-*_])(?:\\s*\\1){2,}\\s*$")
private val TABLE_SEPARATOR = Regex("^\\|?\\s*:?-{2,}:?\\s*(\\|\\s*:?-{2,}:?\\s*)*\\|?$")
private val IMAGE_TOKEN = Regex("!\\[([^\\]\\n]*)]\\(([^)\\n]+)\\)")

private fun startsBlock(lines: List<String>, index: Int): Boolean {
    val line = lines[index]
    if (line.startsWith("```")) return true
    if (line.trimStart().startsWith("$$")) return true
    if (line.trimStart().startsWith("\\[")) return true
    if (HEADING.matches(line) || RULE.matches(line)) return true
    val trimmed = line.trimStart()
    if (trimmed.startsWith("> ") || trimmed.startsWith("- ") || trimmed.startsWith("* ")) return true
    if (ORDERED_ITEM.matches(line)) return true
    return isTableStart(lines, index)
}

private fun isTableStart(lines: List<String>, index: Int): Boolean {
    if (index + 1 >= lines.size) return false
    if (!lines[index].contains('|')) return false
    val separator = lines[index + 1].trim()
    if (separator.isEmpty() || !separator.contains('-')) return false
    return TABLE_SEPARATOR.matches(separator)
}

private fun splitTableRow(line: String): List<String> =
    line.trim().removePrefix("|").removeSuffix("|").split('|').map { it.trim() }

internal fun parseMarkdownBlocks(source: String): List<MarkdownBlock> {
    val lines = source.replace("\r\n", "\n").split('\n')
    val result = mutableListOf<MarkdownBlock>()
    var index = 0
    while (index < lines.size) {
        val line = lines[index]
        val trimmed = line.trimStart()
        when {
            line.isBlank() -> index++

            line.startsWith("```") -> {
                val language = line.removePrefix("```").trim().take(32)
                val code = mutableListOf<String>()
                index++
                while (index < lines.size && !lines[index].startsWith("```")) code += lines[index++]
                if (index < lines.size) index++
                result += MarkdownBlock.Code(language, code.joinToString("\n"))
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
                    while (index < lines.size && !lines[index].trimStart().startsWith("$$")) body += lines[index++]
                    if (index < lines.size) index++
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
                    if (index < lines.size) {
                        body += lines[index].substringBefore("\\]")
                        index++
                    }
                }
                result += MarkdownBlock.Math(body.joinToString("\n").trim(), true)
            }

            HEADING.matches(line) -> {
                val match = HEADING.matchEntire(line)!!
                result += MarkdownBlock.Heading(match.groupValues[1].length, match.groupValues[2])
                index++
            }

            RULE.matches(line) -> { result += MarkdownBlock.Rule; index++ }

            trimmed.startsWith("> ") -> { result += MarkdownBlock.Quote(trimmed.removePrefix("> ")); index++ }

            trimmed.startsWith("- ") || trimmed.startsWith("* ") -> {
                result += MarkdownBlock.ListItem(trimmed.drop(2), false); index++
            }

            ORDERED_ITEM.matches(line) -> {
                result += MarkdownBlock.ListItem(ORDERED_ITEM.matchEntire(line)!!.groupValues[1], true); index++
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
                val paragraph = mutableListOf(line)
                index++
                while (index < lines.size && lines[index].isNotBlank() && !startsBlock(lines, index)) {
                    paragraph += lines[index++]
                }
                appendParagraph(result, paragraph.joinToString("\n"))
            }
        }
    }
    return result
}

private fun appendParagraph(result: MutableList<MarkdownBlock>, text: String) {
    var cursor = 0
    IMAGE_TOKEN.findAll(text).forEach { match ->
        val before = text.substring(cursor, match.range.first)
        if (before.isNotBlank()) result += MarkdownBlock.Paragraph(before.trim())
        val reference = fileReferencePath(match.groupValues[2])
        if (reference != null) result += MarkdownBlock.Image(reference, match.groupValues[1])
        else result += MarkdownBlock.Paragraph(match.value)
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

private val MATH_STYLE = SpanStyle(fontFamily = FontFamily.Serif, fontStyle = FontStyle.Italic)

private fun inlineMarkdown(text: String, linkColor: Color): AnnotatedString = buildAnnotatedString {
    val token = Regex(
        """\\\((.+?)\\\)""" + """|\$\$(.+?)\$\$""" + """|\$([^\s$][^$\n]*?)\$""" +
            """|(`[^`\n]+`|\*\*[^*\n]+\*\*|(?<!\*)\*[^*\n]+\*|\[[^\]\n]+\]\([^)\n]+\))"""
    )
    var cursor = 0
    token.findAll(text).forEach { match ->
        append(text.substring(cursor, match.range.first))
        val value = match.value
        when {
            value.startsWith("\\(") && value.endsWith("\\)") ->
                withStyle(MATH_STYLE) { append(latexToDisplay(value.substring(2, value.length - 2))) }
            value.startsWith("$$") && value.endsWith("$$") ->
                withStyle(MATH_STYLE) { append(latexToDisplay(value.substring(2, value.length - 2))) }
            value.startsWith("$") && value.endsWith("$") ->
                withStyle(MATH_STYLE) { append(latexToDisplay(value.substring(1, value.length - 1))) }
            value.startsWith('`') ->
                withStyle(SpanStyle(fontFamily = FontFamily.Monospace, background = Color(0x2214B8A6))) {
                    append(value.drop(1).dropLast(1))
                }
            value.startsWith("**") -> withStyle(SpanStyle(fontWeight = FontWeight.Bold)) { append(value.drop(2).dropLast(2)) }
            value.startsWith('*') -> withStyle(SpanStyle(fontStyle = FontStyle.Italic)) { append(value.drop(1).dropLast(1)) }
            value.startsWith('[') -> {
                val split = value.indexOf("](")
                val label = value.substring(1, split)
                val url = safeWebUrl(value.substring(split + 2, value.length - 1))
                if (url == null) append(label) else withLink(
                    LinkAnnotation.Url(url, TextLinkStyles(style = SpanStyle(
                        color = linkColor,
                        textDecoration = androidx.compose.ui.text.style.TextDecoration.Underline,
                    )))
                ) { append(label) }
            }
        }
        cursor = match.range.last + 1
    }
    append(text.substring(cursor))
}

@Composable
private fun InlineText(text: String, style: androidx.compose.ui.text.TextStyle = LocalTextStyle.current) {
    val linkColor = MaterialTheme.colorScheme.primary
    val annotated = remember(text, linkColor) { inlineMarkdown(text, linkColor) }
    Text(text = annotated, style = style.copy(color = LocalContentColor.current))
}

@Composable
private fun CodeBlock(block: MarkdownBlock.Code) {
    var expanded by remember(block.text) { mutableStateOf(block.text.lineSequence().count() <= 18) }
    val clipboard = LocalClipboardManager.current
    Surface(color = MaterialTheme.colorScheme.surfaceVariant, shape = RoundedCornerShape(12.dp)) {
        Column {
            Row(Modifier.fillMaxWidth().padding(start = 12.dp), horizontalArrangement = Arrangement.SpaceBetween) {
                Text(block.language.ifBlank { "code" }, style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 10.dp))
                Row {
                    if (block.text.lineSequence().count() > 18) TextButton(onClick = { expanded = !expanded }) { Text(if (expanded) "折り畳む" else "展開") }
                    TextButton(onClick = { clipboard.setText(AnnotatedString(block.text)) }) { Text("コピー") }
                }
            }
            AnimatedVisibility(
                visible = expanded,
                enter = expandFadeIn(LocalReduceMotion.current),
                exit = shrinkFadeOut(LocalReduceMotion.current),
            ) {
                SelectionContainer {
                    Text(block.text, fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier.horizontalScroll(rememberScrollState()).padding(12.dp))
                }
            }
        }
    }
}

@Composable
private fun MathBlock(block: MarkdownBlock.Math) {
    val rendered = remember(block.tex) { latexToDisplay(block.tex) }
    Surface(color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f), shape = RoundedCornerShape(10.dp)) {
        Box(
            Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()).padding(horizontal = 12.dp, vertical = 10.dp),
            contentAlignment = Alignment.CenterStart,
        ) {
            Text(
                rendered,
                fontFamily = FontFamily.Serif,
                style = if (block.display) MaterialTheme.typography.titleMedium else MaterialTheme.typography.bodyMedium,
            )
        }
    }
}

@Composable
private fun TableRow(cells: List<String>, cellWidth: Dp, header: Boolean, columnCount: Int) {
    Row(Modifier.background(if (header) MaterialTheme.colorScheme.surfaceVariant else Color.Transparent)) {
        for (column in 0 until columnCount) {
            Box(
                Modifier.width(cellWidth)
                    .border(0.5.dp, MaterialTheme.colorScheme.outline)
                    .padding(horizontal = 10.dp, vertical = 8.dp)
            ) {
                InlineText(
                    cells.getOrNull(column).orEmpty(),
                    if (header) MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                    else MaterialTheme.typography.bodySmall,
                )
            }
        }
    }
}

@Composable
private fun TableBlock(block: MarkdownBlock.Table) {
    val columnCount = maxOf(block.headers.size, block.rows.maxOfOrNull { it.size } ?: 0)
    if (columnCount == 0) return
    val scroll = rememberScrollState()
    BoxWithConstraints {
        val cellWidth = maxOf(112.dp, (maxWidth - 2.dp) / columnCount)
        Column(
            Modifier.horizontalScroll(scroll)
                .border(0.5.dp, MaterialTheme.colorScheme.outline, RoundedCornerShape(10.dp))
        ) {
            TableRow(block.headers, cellWidth, header = true, columnCount = columnCount)
            block.rows.forEach { row -> TableRow(row, cellWidth, header = false, columnCount = columnCount) }
        }
    }
}

@Composable
private fun BlockContent(block: MarkdownBlock, loader: FileBytesLoader?, onOpen: (String) -> Unit) {
    when (block) {
        is MarkdownBlock.Paragraph -> InlineText(block.text)
        is MarkdownBlock.Heading -> InlineText(block.text, when (block.level) {
            1 -> MaterialTheme.typography.headlineSmall
            2 -> MaterialTheme.typography.titleLarge
            else -> MaterialTheme.typography.titleMedium
        }.copy(fontWeight = FontWeight.Bold))
        is MarkdownBlock.Quote -> Surface(color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.45f), shape = RoundedCornerShape(8.dp)) {
            Box(Modifier.padding(horizontal = 12.dp, vertical = 10.dp)) { InlineText(block.text) }
        }
        is MarkdownBlock.ListItem -> Row { Text("•  "); Box(Modifier.weight(1f)) { InlineText(block.text) } }
        is MarkdownBlock.Code -> CodeBlock(block)
        is MarkdownBlock.Math -> MathBlock(block)
        is MarkdownBlock.Table -> TableBlock(block)
        is MarkdownBlock.Image -> ProtectedImage(
            block.reference, loader, onOpen,
            modifier = Modifier.fillMaxWidth(), thumbnail = true,
            contentDescription = block.alt.ifBlank { null },
        )
        MarkdownBlock.Rule -> HorizontalDivider()
    }
}

@Composable
fun MarkdownText(text: String, loader: FileBytesLoader? = null, onOpen: (String) -> Unit = {}) {
    val blocks = remember(text) { parseMarkdownBlocks(text) }
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        blocks.forEachIndexed { index, block -> key(index) {
            if (block is MarkdownBlock.Image) BlockContent(block, loader, onOpen)
            else SelectionContainer { BlockContent(block, loader, onOpen) }
        } }
    }
}
