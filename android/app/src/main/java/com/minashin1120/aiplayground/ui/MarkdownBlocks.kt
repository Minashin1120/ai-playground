package com.minashin1120.aiplayground.ui

import com.minashin1120.aiplayground.data.CANVAS_PLACEHOLDER_LINE
import com.minashin1120.aiplayground.data.fileReferencePath
import java.net.URI

/*
 * Block parser of `MarkdownText.kt`: the GFM blocks `marked` produces (headings, quotes, lists,
 * tables, fences, raw HTML/SVG) split out so the renderer stays under the Kotlin size budget.
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
    /** A fence moved to the Canvas panel (Web `.canvas-code-placeholder`). */
    data object CanvasPlaceholder : MarkdownBlock
    /** Raw `<svg>` markup in the answer, drawn in Web `.svg-render-box`; [fromCode] is the drawing above a closed SVG fence. */
    data class Svg(val markup: String, val fromCode: Boolean = false) : MarkdownBlock
    /** Raw `<details>`: the browser's disclosure (▶ summary), closed unless it has `open`. */
    data class Details(val summary: String, val blocks: List<MarkdownBlock>, val open: Boolean) : MarkdownBlock
    /** Raw `<audio controls src>` written by the server for generated speech / music: an inline player. */
    data class Audio(val src: String) : MarkdownBlock
}

/** One list item; [checked] is non-null for GFM task items. */
internal data class ListItem(val blocks: List<MarkdownBlock>, val checked: Boolean? = null)

private val FENCE = Regex("^( {0,3})(`{3,}|~{3,})(.*)$")
private val SVG_START = Regex("^ {0,3}<svg[\\s>]", RegexOption.IGNORE_CASE)
private val SVG_CODE_PROLOG = Regex("^(?:<\\?xml[\\s\\S]*?\\?>\\s*|<!--[\\s\\S]*?-->\\s*|<!DOCTYPE[^>]*>\\s*)*", RegexOption.IGNORE_CASE)
private val SVG_CODE_ROOT = Regex("^<svg[\\s>]", RegexOption.IGNORE_CASE)
private val SVG_CODE_END = Regex("</svg\\s*>$", RegexOption.IGNORE_CASE)

/** Web `getRenderableSvgCode`: a closed ```svg fence (or ```xml holding one `<svg>` document) is also drawn. */
internal fun renderableSvgCode(language: String, text: String): String? {
    if (language.lowercase() !in setOf("svg", "xml", "image/svg+xml", "mathsvg")) return null
    val code = text.trim()
    if (code.isEmpty() || code.length > 300_000) return null
    val body = SVG_CODE_PROLOG.replaceFirst(code, "")
    return code.takeIf { SVG_CODE_ROOT.containsMatchIn(body) && SVG_CODE_END.containsMatchIn(body) }
}
private val HTML_RULE = Regex("^ {0,3}<hr\\s*/?>\\s*$", RegexOption.IGNORE_CASE)
private val DETAILS_START = Regex("^ {0,3}<details(\\s[^>]*)?>", RegexOption.IGNORE_CASE)
private val AUDIO_START = Regex("^ {0,3}<audio[\\s>]", RegexOption.IGNORE_CASE)
private val HTML_SRC = Regex("\\bsrc\\s*=\\s*(?:\"([^\"]*)\"|'([^']*)')", RegexOption.IGNORE_CASE)
private val TABLE_START = Regex("^ {0,3}<table(\\s[^>]*)?>", RegexOption.IGNORE_CASE)
private val HTML_SUMMARY = Regex("<summary[^>]*>([\\s\\S]*?)</summary\\s*>", RegexOption.IGNORE_CASE)
private val HTML_ROW = Regex("<tr[^>]*>([\\s\\S]*?)</tr\\s*>", RegexOption.IGNORE_CASE)
private val HTML_CELL = Regex("<(t[hd])(\\s[^>]*)?>([\\s\\S]*?)</\\1\\s*>", RegexOption.IGNORE_CASE)

/** Lines of an HTML block from [start] through the line that closes [tag] (same-name nesting counted). */
private fun htmlBlockLines(lines: List<String>, start: Int, tag: String): List<String> {
    val open = Regex("<$tag(\\s[^>]*)?>", RegexOption.IGNORE_CASE)
    val close = Regex("</$tag\\s*>", RegexOption.IGNORE_CASE)
    var depth = 0
    var index = start
    val body = mutableListOf<String>()
    while (index < lines.size) {
        val line = lines[index]
        body += line
        depth += open.findAll(line).count() - close.findAll(line).count()
        index++
        if (depth <= 0) break
    }
    return body
}

/** Web (DOMPurify + browser) `<details>`: summary text, inner Markdown, `open` attribute. */
internal fun parseHtmlDetails(markup: String): MarkdownBlock.Details {
    val openTag = DETAILS_START.find(markup.trimStart())
    val attributes = openTag?.groupValues?.get(1).orEmpty()
    var inner = markup.trimStart().substring(openTag?.value?.length ?: 0)
    inner = inner.replace(Regex("</details\\s*>\\s*$", RegexOption.IGNORE_CASE), "")
    val summary = HTML_SUMMARY.find(inner)
    val title = summary?.groupValues?.get(1)?.trim()?.replace(Regex("\\s+"), " ").orEmpty().ifEmpty { "詳細" }
    if (summary != null) inner = inner.removeRange(summary.range)
    return MarkdownBlock.Details(title, parseMarkdownBlocks(inner.trim('\n')), Regex("\\bopen\\b", RegexOption.IGNORE_CASE).containsMatchIn(attributes))
}

/** The file an `<audio>` element plays: its own `src`, else the first `<source src>`; null when it has none. */
internal fun parseHtmlAudioSource(markup: String): String? =
    HTML_SRC.find(markup)?.let { m -> m.groupValues.drop(1).firstOrNull { it.isNotEmpty() } }
        ?.replace("&amp;", "&")?.trim()?.takeIf { it.isNotEmpty() }

/** Web `.prose table` for a raw HTML table: `<th>` cells of the first row become the header. */
internal fun parseHtmlTable(markup: String): MarkdownBlock.Table {
    val rows = HTML_ROW.findAll(markup).map { row ->
        HTML_CELL.findAll(row.groupValues[1]).map { it.groupValues[1].lowercase() to it.groupValues[3].trim().replace(Regex("\\s*\\n\\s*"), " ") }.toList()
    }.filter { it.isNotEmpty() }.toList()
    val header = rows.firstOrNull()?.takeIf { cells -> cells.all { it.first == "th" } }
    return MarkdownBlock.Table(header?.map { it.second }.orEmpty(), rows.drop(if (header != null) 1 else 0).map { cells -> cells.map { it.second } })
}
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
    if (line == CANVAS_PLACEHOLDER_LINE) return true
    if (SVG_START.containsMatchIn(line) || HTML_RULE.matches(line) || DETAILS_START.containsMatchIn(line) || TABLE_START.containsMatchIn(line) || AUDIO_START.containsMatchIn(line)) return true
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

            line == CANVAS_PLACEHOLDER_LINE -> { result += MarkdownBlock.CanvasPlaceholder; index++ }

            // Raw SVG (an HTML block for marked): kept whole until `</svg>`.
            SVG_START.containsMatchIn(line) -> {
                val body = mutableListOf<String>()
                while (index < lines.size) {
                    body += lines[index]
                    index++
                    if (body.last().contains("</svg>", ignoreCase = true)) break
                }
                result += MarkdownBlock.Svg(body.joinToString("\n"))
            }

            HTML_RULE.matches(line) -> { result += MarkdownBlock.Rule; index++ }

            DETAILS_START.containsMatchIn(line) -> {
                val body = htmlBlockLines(lines, index, "details")
                index += body.size
                result += parseHtmlDetails(body.joinToString("\n"))
            }

            AUDIO_START.containsMatchIn(line) -> {
                // `<audio …></audio>` on one line (the server's form); an unclosed tag takes only its own line.
                val body = htmlBlockLines(lines, index, "audio").takeIf { b -> b.any { it.contains("</audio", ignoreCase = true) } } ?: listOf(line)
                index += body.size
                parseHtmlAudioSource(body.joinToString("\n"))?.let { result += MarkdownBlock.Audio(it) }
            }

            TABLE_START.containsMatchIn(line) -> {
                val body = htmlBlockLines(lines, index, "table")
                index += body.size
                result += parseHtmlTable(body.joinToString("\n"))
            }

            fence != null && !(fence.groupValues[2][0] == '`' && fence.groupValues[3].contains('`')) -> {
                val indent = fence.groupValues[1].length
                val marker = fence.groupValues[2]
                val language = fence.groupValues[3].trim().substringBefore(' ').take(32)
                val body = mutableListOf<String>()
                var closed = false
                index++
                while (index < lines.size) {
                    val candidate = lines[index].trimStart()
                    if (candidate.startsWith(marker) && candidate.trimEnd().all { it == marker[0] }) { index++; closed = true; break }
                    body += lines[index].let { l -> var drop = 0; while (drop < indent && drop < l.length && l[drop] == ' ') drop++; l.substring(drop) }
                    index++
                }
                val text = body.joinToString("\n")
                when (language) {
                    // Web keeps Python runs out of the answer body (they open from the footer button).
                    "pyexec" -> Unit
                    "chat_error" -> result += MarkdownBlock.ChatError(text)
                    // Web `mathsvg`: the drawing only, never the usual code header/preview/download UI.
                    "mathsvg" -> if (closed) renderableSvgCode(language, text)?.let { result += MarkdownBlock.Svg(it, fromCode = true) }
                    else -> {
                        // A fence still open while the answer streams stays code until it closes.
                        if (closed) renderableSvgCode(language, text)?.let { result += MarkdownBlock.Svg(it, fromCode = true) }
                        result += MarkdownBlock.Code(language, text)
                    }
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

/** Web `openImageViewer('.chat-image')`: the images a reply shows, in order (site files and https images). */
internal fun markdownImageReferences(text: String): List<String> = IMAGE_TOKEN.findAll(text).mapNotNull { match ->
    val target = match.groupValues[2].trim().substringBefore(' ')
    fileReferencePath(target) ?: target.takeIf { it.startsWith("https://") && safeWebUrl(it) != null }
}.toList()

private fun appendParagraph(result: MutableList<MarkdownBlock>, text: String) {
    var cursor = 0
    IMAGE_TOKEN.findAll(text).forEach { match ->
        // Web keeps a Markdown title ("…") out of the source; images on other sites load over https.
        val target = match.groupValues[2].trim().substringBefore(' ')
        val reference = fileReferencePath(target)
            ?: target.takeIf { it.startsWith("https://") && safeWebUrl(it) != null } ?: return@forEach
        val before = text.substring(cursor, match.range.first)
        if (before.isNotBlank()) result += MarkdownBlock.Paragraph(before.trim())
        result += MarkdownBlock.Image(reference, match.groupValues[1])
        cursor = match.range.last + 1
    }
    val tail = text.substring(cursor)
    if (tail.isNotBlank()) result += MarkdownBlock.Paragraph(tail.trim())
}

/**
 * Web renders every Markdown / `<a>` href as a link: web URLs, `mailto:` / `tel:`, and paths on this site
 * (`/…`, opened on the server's origin). Other relative forms and script schemes stay plain text.
 */
internal fun markdownLinkTarget(value: String): String? {
    val trimmed = value.trim()
    safeWebUrl(trimmed)?.let { return it }
    val lower = trimmed.lowercase()
    if ((lower.startsWith("mailto:") || lower.startsWith("tel:")) && trimmed.substringAfter(':').isNotBlank()) return trimmed
    if (trimmed.startsWith("//")) return safeWebUrl("https:$trimmed")
    if (trimmed.startsWith("/")) return safeWebUrl(com.minashin1120.aiplayground.data.ServerOrigin.base + trimmed)
    return null
}

internal fun safeWebUrl(value: String): String? = runCatching {
    URI(value.trim()).takeIf {
        val scheme = it.scheme?.lowercase()
        (scheme == "http" || scheme == "https") && !it.host.isNullOrBlank()
    }?.toString()
}.getOrNull()
