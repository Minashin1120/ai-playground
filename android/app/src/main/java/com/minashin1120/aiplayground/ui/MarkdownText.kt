package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.ClickableText
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.saveable.rememberSaveable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.text.*
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import java.net.URI

internal sealed interface MarkdownBlock {
    data class Paragraph(val text: String) : MarkdownBlock
    data class Heading(val level: Int, val text: String) : MarkdownBlock
    data class Quote(val text: String) : MarkdownBlock
    data class ListItem(val text: String, val ordered: Boolean) : MarkdownBlock
    data class Code(val language: String, val text: String) : MarkdownBlock
    data object Rule : MarkdownBlock
}

internal fun parseMarkdownBlocks(source: String): List<MarkdownBlock> {
    val lines = source.replace("\r\n", "\n").split('\n')
    val result = mutableListOf<MarkdownBlock>()
    var index = 0
    while (index < lines.size) {
        val line = lines[index]
        if (line.startsWith("```")) {
            val language = line.removePrefix("```").trim().take(32)
            val code = mutableListOf<String>()
            index++
            while (index < lines.size && !lines[index].startsWith("```")) code += lines[index++]
            if (index < lines.size) index++
            result += MarkdownBlock.Code(language, code.joinToString("\n"))
            continue
        }
        val heading = Regex("^(#{1,6})\\s+(.+)$").matchEntire(line)
        val ordered = Regex("^\\s*\\d+[.)]\\s+(.+)$").matchEntire(line)
        when {
            line.isBlank() -> index++
            heading != null -> { result += MarkdownBlock.Heading(heading.groupValues[1].length, heading.groupValues[2]); index++ }
            line.matches(Regex("^\\s*([-*_])(?:\\s*\\1){2,}\\s*$")) -> { result += MarkdownBlock.Rule; index++ }
            line.trimStart().startsWith("> ") -> { result += MarkdownBlock.Quote(line.trimStart().removePrefix("> ")); index++ }
            line.trimStart().startsWith("- ") || line.trimStart().startsWith("* ") -> { result += MarkdownBlock.ListItem(line.trimStart().drop(2), false); index++ }
            ordered != null -> { result += MarkdownBlock.ListItem(ordered.groupValues[1], true); index++ }
            else -> {
                val paragraph = mutableListOf(line)
                index++
                while (index < lines.size && lines[index].isNotBlank() && !lines[index].startsWith("```") &&
                    !lines[index].matches(Regex("^(#{1,6})\\s+.+$")) && !lines[index].trimStart().startsWith("> ") &&
                    !lines[index].trimStart().startsWith("- ") && !lines[index].trimStart().startsWith("* ") &&
                    !lines[index].matches(Regex("^\\s*\\d+[.)]\\s+.+$"))) paragraph += lines[index++]
                result += MarkdownBlock.Paragraph(paragraph.joinToString("\n"))
            }
        }
    }
    return result
}

private fun safeWebUrl(value: String): String? = runCatching {
    URI(value.trim()).takeIf { it.scheme?.lowercase() in setOf("http", "https") && !it.host.isNullOrBlank() }?.toString()
}.getOrNull()

private fun inlineMarkdown(text: String): AnnotatedString = buildAnnotatedString {
    val token = Regex("(`[^`\\n]+`|\\*\\*[^*\\n]+\\*\\*|(?<!\\*)\\*[^*\\n]+\\*|\\[[^]\\n]+]\\([^)\\n]+\\))")
    var cursor = 0
    token.findAll(text).forEach { match ->
        append(text.substring(cursor, match.range.first))
        val value = match.value
        when {
            value.startsWith('`') -> withStyle(SpanStyle(fontFamily = FontFamily.Monospace, background = androidx.compose.ui.graphics.Color(0x2214B8A6))) { append(value.drop(1).dropLast(1)) }
            value.startsWith("**") -> withStyle(SpanStyle(fontWeight = FontWeight.Bold)) { append(value.drop(2).dropLast(2)) }
            value.startsWith('*') -> withStyle(SpanStyle(fontStyle = FontStyle.Italic)) { append(value.drop(1).dropLast(1)) }
            value.startsWith('[') -> {
                val split = value.indexOf("](")
                val label = value.substring(1, split)
                val url = safeWebUrl(value.substring(split + 2, value.length - 1))
                if (url == null) append(label) else pushStringAnnotation("URL", url).also {
                    withStyle(SpanStyle(color = androidx.compose.ui.graphics.Color(0xFF0A8F84), textDecoration = androidx.compose.ui.text.style.TextDecoration.Underline)) { append(label) }
                    pop()
                }
            }
        }
        cursor = match.range.last + 1
    }
    append(text.substring(cursor))
}

@Composable
private fun InlineText(text: String, style: androidx.compose.ui.text.TextStyle = LocalTextStyle.current) {
    val annotated = remember(text) { inlineMarkdown(text) }
    val uri = LocalUriHandler.current
    ClickableText(text = annotated, style = style.copy(color = LocalContentColor.current), onClick = { offset ->
        annotated.getStringAnnotations("URL", offset, offset).firstOrNull()?.item?.let { runCatching { uri.openUri(it) } }
    })
}

@Composable
private fun CodeBlock(block: MarkdownBlock.Code) {
    var expanded by rememberSaveable(block.text) { mutableStateOf(block.text.lineSequence().count() <= 18) }
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
            if (expanded) SelectionContainer {
                Text(block.text, fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.horizontalScroll(rememberScrollState()).padding(12.dp))
            }
        }
    }
}

@Composable
fun MarkdownText(text: String) {
    val blocks = remember(text) { parseMarkdownBlocks(text) }
    SelectionContainer {
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            blocks.forEachIndexed { index, block -> key(index) {
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
                    is MarkdownBlock.ListItem -> Row { Text(if (block.ordered) "•  " else "•  "); Box(Modifier.weight(1f)) { InlineText(block.text) } }
                    is MarkdownBlock.Code -> CodeBlock(block)
                    MarkdownBlock.Rule -> HorizontalDivider()
                }
            } }
        }
    }
}
