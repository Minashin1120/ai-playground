package com.minashin1120.aiplayground.ui

import com.minashin1120.aiplayground.data.ChatMessage
import com.minashin1120.aiplayground.data.MathStepView
import com.minashin1120.aiplayground.data.numericId

/** Space between the token bar and the first line of a quoted range after a jump. */
internal const val QUOTE_JUMP_TOP_MARGIN_DP = 12

/**
 * How far the conversation must scroll so the quoted range's top edge sits [margin] below the list's
 * [viewportTop]. Positive values scroll forward (the content moves up).
 */
internal fun quoteScrollDelta(quoteTop: Float, viewportTop: Float, margin: Float): Float =
    quoteTop - viewportTop - margin

/**
 * Where the quote bar's "引用元へ移動" action should scroll.
 *
 * The quote remembers the stable row key rather than the raw message id: a background sync can replace
 * a pending id with the server id while the same row keeps its [ChatMessage.conversationKey]. The numeric
 * id is only a fallback; a quote taken from the still-streaming answer targets that live row instead of a
 * stored message.
 */
internal sealed interface QuoteTarget {
    data class Message(val index: Int) : QuoteTarget
    data object Live : QuoteTarget
    data object Missing : QuoteTarget
}

internal fun resolveQuoteTarget(
    messages: List<ChatMessage>,
    sourceKey: String?,
    sourceMessageId: Int?,
    liveKey: String?,
    liveVisible: Boolean,
    keyOf: (ChatMessage) -> String = { it.conversationKey ?: it.id },
): QuoteTarget {
    if (sourceKey != null) {
        val index = messages.indexOfFirst { it.id == sourceKey || keyOf(it) == sourceKey }
        if (index >= 0) return QuoteTarget.Message(index)
        if (liveVisible && liveKey != null && sourceKey == liveKey) return QuoteTarget.Live
    }
    if (sourceMessageId != null) {
        val index = messages.indexOfFirst { numericId(it) == sourceMessageId }
        if (index >= 0) return QuoteTarget.Message(index)
    }
    return QuoteTarget.Missing
}

/**
 * The row a sent message's quote strip came from. The server keeps the quote text but not the quoted
 * message id, so find the nearest preceding message whose rendered text contains it. A selection copies
 * what the reader saw rather than the stored Markdown, so the full-text match is followed by a tolerant
 * comparison that ignores punctuation, list markers, Markdown and the approximations used for formulas.
 * Quotes from other branches (or a quote too different to place) have no target and stay inert.
 */
internal fun findQuotedMessageIndex(messages: List<ChatMessage>, beforeIndex: Int, quote: String): Int? =
    findQuoteTextIndex(messages, beforeIndex, quote)

/**
 * The same lookup for any quote text: exact on the normalized text, then the closest preceding row by
 * [compactQuoteScore]. The composer uses this after a draft edit restored a quote without its source id.
 * Math steps quote the formula the reader saw, and an easier step lives beside the original Markdown, so
 * both the readable formula text and those saved step bodies are searched.
 */
internal fun findQuoteTextIndex(messages: List<ChatMessage>, beforeIndex: Int, quote: String): Int? {
    val needle = normalizeQuoteMatch(quote)
    if (needle.isEmpty()) return null
    val last = (beforeIndex - 1).coerceAtMost(messages.lastIndex)
    if (last < 0) return null
    for (index in last downTo 0) {
        if (quoteCandidates(messages[index]).any { textContainsQuote(it, quote) }) return index
    }
    val compactNeedle = compactQuoteText(quote)
    if (compactNeedle.length < FUZZY_QUOTE_MIN_CHARS) return null
    var best = -1
    var bestScore = 0.0
    for (index in last downTo 0) {
        val score = quoteCandidates(messages[index]).maxOf { fuzzyQuoteScore(it, compactNeedle) }
        if (score >= FUZZY_QUOTE_MIN_SCORE && score > bestScore) {
            best = index
            bestScore = score
        }
    }
    return best.takeIf { it >= 0 }
}

/** A math-step row to open so the quoted text is on screen. [revealOriginal] hides a saved easier body. */
internal data class MathQuoteTarget(val index: Int, val revealOriginal: Boolean)

/**
 * The step view that contains [quote]. The step on screen wins when it already shows the quote; otherwise
 * the nearest earlier view. An easier replacement that hides the quoted original is opened back.
 */
internal fun findMathQuoteTarget(
    views: List<MathStepView>,
    simplified: Map<Int, String>,
    quote: String,
    current: Int = 0,
): MathQuoteTarget? {
    if (views.isEmpty() || normalizeQuoteMatch(quote).isEmpty()) return null
    fun shown(view: MathStepView) = if (view.expanded) view.body else simplified[view.sourceIndex] ?: view.body
    fun covered(view: MathStepView) = textContainsQuote(shown(view), quote)
    views.getOrNull(current)?.takeIf { covered(it) }?.let { return MathQuoteTarget(current, revealOriginal = false) }
    val shownHit = views.indexOfFirst { covered(it) }
    if (shownHit >= 0) return MathQuoteTarget(shownHit, revealOriginal = false)
    fun originalHidden(view: MathStepView) =
        !view.expanded && simplified[view.sourceIndex] != null && textContainsQuote(view.body, quote)
    views.getOrNull(current)?.takeIf { originalHidden(it) }?.let { return MathQuoteTarget(current, revealOriginal = true) }
    val hiddenHit = views.indexOfFirst { originalHidden(it) }
    return hiddenHit.takeIf { it >= 0 }?.let { MathQuoteTarget(it, revealOriginal = true) }
}

/** True when a typeset formula's approximation is the quote, or the quote is the text around that formula. */
internal fun formulaHoldsQuote(approximation: String, quote: String): Boolean {
    val formula = normalizeQuoteMatch(approximation)
    val needle = normalizeQuoteMatch(quote)
    if (formula.length < 4 || needle.length < 4) return false
    return needle.contains(formula) || formula.contains(needle)
}

private fun quoteCandidates(message: ChatMessage): List<String> {
    val state = message.mathStepsState ?: return listOf(message.content)
    val texts = ArrayList<String>(4)
    texts += message.content
    state.simplified.values.forEach { texts += it }
    state.expanded.values.forEach { parts -> parts.forEach { texts += it.body } }
    return texts
}

/** Exact match on the stored Markdown, then on the formula text a selection actually copies. */
private fun textContainsQuote(text: String, quote: String): Boolean {
    val needle = normalizeQuoteMatch(quote)
    if (needle.isEmpty()) return false
    if (normalizeQuoteMatch(text).contains(needle)) return true
    val reader = readerQuoteText(text)
    return reader != text && normalizeQuoteMatch(reader).contains(needle)
}

private fun fuzzyQuoteScore(text: String, compactNeedle: String): Double {
    val raw = compactQuoteScore(compactNeedle, compactQuoteText(text))
    val reader = readerQuoteText(text)
    if (reader == text) return raw
    return maxOf(raw, compactQuoteScore(compactNeedle, compactQuoteText(reader)))
}

/**
 * Markdown with formulas replaced by the approximation a selection copies ([latexToDisplay]), and mathsvg
 * drawings replaced by ［図］. Code fences stay literal. Display formulas are one line, as the copy is.
 */
internal fun readerQuoteText(source: String): String {
    if (source.isEmpty()) return source
    if (!source.contains('\\') && !source.contains('$') && !source.contains('`')) return source
    val text = source.replace("\r\n", "\n").replace('\r', '\n')
    val out = StringBuilder(text.length)
    var i = 0
    var fence: String? = null
    while (i < text.length) {
        val atLine = i == 0 || text[i - 1] == '\n'
        if (fence != null || atLine) {
            val lineEnd = text.indexOf('\n', i).let { if (it < 0) text.length else it }
            val line = text.substring(i, lineEnd)
            val marker = CODE_FENCE_LINE.find(line)?.groupValues?.get(1)
            if (fence != null) {
                out.append(line)
                if (marker != null && line.trim().isNotEmpty() && line.trim().all { it == fence!![0] }) fence = null
                i = if (lineEnd < text.length) lineEnd + 1 else text.length
                if (i <= text.length && lineEnd < text.length) out.append('\n')
                continue
            }
            if (marker != null) {
                val lang = line.trim().removePrefix(marker).trim().substringBefore(' ').substringBefore('\t').lowercase()
                if (lang == "mathsvg") {
                    var j = if (lineEnd < text.length) lineEnd + 1 else text.length
                    while (j < text.length) {
                        val end = text.indexOf('\n', j).let { if (it < 0) text.length else it }
                        val candidate = text.substring(j, end).trim()
                        val closed = candidate.isNotEmpty() && candidate.all { it == marker[0] }
                        j = if (end < text.length) end + 1 else text.length
                        if (closed) break
                    }
                    out.append('［').append('図').append('］')
                    if (j < text.length) out.append('\n')
                    i = j
                    continue
                }
                fence = marker.take(3)
                out.append(line)
                i = if (lineEnd < text.length) lineEnd + 1 else text.length
                if (lineEnd < text.length) out.append('\n')
                continue
            }
        }
        val doubled = when {
            text.startsWith("\\\\(", i) -> "\\\\)" to 3
            text.startsWith("\\\\[", i) -> "\\\\]" to 3
            else -> null
        }
        if (doubled != null) {
            val (closer, openLen) = doubled
            val end = text.indexOf(closer, i + openLen)
            if (end > i + openLen) {
                out.append(oneLineFormula(text.substring(i + openLen, end)))
                i = end + closer.length
                continue
            }
        }
        val display = when {
            text.startsWith("$$", i) -> "$$" to 2
            text.startsWith("\\[", i) -> "\\]" to 2
            else -> null
        }
        if (display != null) {
            val (closer, openLen) = display
            val end = text.indexOf(closer, i + openLen)
            if (end > i + openLen && text.substring(i + openLen, end).isNotBlank()) {
                out.append(oneLineFormula(text.substring(i + openLen, end)))
                i = end + closer.length
                continue
            }
        }
        if (text.startsWith("\\begin{", i)) {
            val env = BEGIN_ENV.find(text.substring(i, minOf(text.length, i + 40)))
            if (env != null) {
                val close = "\\end{${env.groupValues[1]}}"
                val end = text.indexOf(close, i + env.value.length)
                if (end > i) {
                    out.append(oneLineFormula(text.substring(i, end + close.length)))
                    i = end + close.length
                    continue
                }
            }
        }
        if (text.startsWith("\\(", i) || text.startsWith("\\（", i)) {
            val end = listOf(text.indexOf("\\)", i + 2), text.indexOf("\\）", i + 2)).filter { it >= 0 }.minOrNull() ?: -1
            if (end > i) {
                out.append(latexToDisplay(text.substring(i + 2, end)))
                i = end + 2
                continue
            }
        }
        if (text[i] == '$' && i + 1 < text.length && text[i + 1] != ' ' && text[i + 1] != '$') {
            val end = text.indexOf('$', i + 1)
            if (end > i + 1 && text[end - 1] != ' ' && !text.substring(i + 1, end).contains('\n')) {
                out.append(latexToDisplay(text.substring(i + 1, end)))
                i = end + 1
                continue
            }
        }
        out.append(text[i])
        i++
    }
    return out.toString()
}

private fun oneLineFormula(tex: String): String =
    latexToDisplay(tex).replace(FORMULA_LINE_BREAKS, " ").trim()

private val CODE_FENCE_LINE = Regex("""^ {0,3}(`{3,}|~{3,})""")
private val BEGIN_ENV = Regex("""^\\begin\{([A-Za-z*]+)\}""")
private val FORMULA_LINE_BREAKS = Regex("""\s*\n\s*""")

/** Letters and digits only, so symbols, Markdown markers and formula approximations do not block a match. */
private fun compactQuoteText(text: String): String = buildString(text.length) {
    text.forEach { c -> if (c.isLetterOrDigit()) append(c.lowercaseChar()) }
}

/** Share of [needle]'s character pairs that also appear in [candidate]; repeated pairs count once each. */
private fun compactQuoteScore(needle: String, candidate: String): Double {
    if (candidate.length < 2) return 0.0
    val available = HashMap<String, Int>()
    for (index in 0 until candidate.length - 1) {
        val pair = candidate.substring(index, index + 2)
        available[pair] = (available[pair] ?: 0) + 1
    }
    var matched = 0
    for (index in 0 until needle.length - 1) {
        val pair = needle.substring(index, index + 2)
        val count = available[pair] ?: 0
        if (count > 0) {
            matched++
            available[pair] = count - 1
        }
    }
    return matched.toDouble() / (needle.length - 1)
}

private const val FUZZY_QUOTE_MIN_CHARS = 12
private const val FUZZY_QUOTE_MIN_SCORE = 0.65

private val MARKDOWN_LINK = Regex("!?\\[([^\\]]*)]\\([^)]*\\)")
private val MARKDOWN_MARKERS = Regex("[*_`~#>]+")
private val WHITESPACE = Regex("\\s+")
private const val MARKDOWN_MARKER_CHARS = "*_`~#>"

/**
 * The range in a bubble's rendered [text] that [quote] came from, used to flash the quoted text
 * after the jump. The exact selection usually matches a stored user message; an answer goes through
 * Markdown, so fall back to [normalizeQuoteMatch] and map the match back to the original offsets.
 */
internal fun findQuoteRange(text: String, quote: String): IntRange? {
    val needle = quote.trim()
    if (needle.isEmpty() || text.isEmpty()) return null
    val exact = text.indexOf(needle)
    if (exact >= 0) return exact until exact + needle.length
    val normalized = normalizeQuoteMatch(needle)
    if (normalized.isEmpty()) return null
    val builder = StringBuilder(text.length)
    val origin = ArrayList<Int>(text.length)
    text.forEachIndexed { index, c ->
        when {
            c.isWhitespace() -> if (builder.isNotEmpty() && builder.last() != ' ') {
                builder.append(' '); origin += index
            }
            c in MARKDOWN_MARKER_CHARS -> Unit
            else -> {
                builder.append(c); origin += index
            }
        }
    }
    val flat = builder.toString()
    val lead = flat.indexOfFirst { !it.isWhitespace() }
    if (lead < 0) return null
    val trailing = flat.indexOfLast { !it.isWhitespace() }
    val found = flat.indexOf(normalized, lead)
    if (found < 0 || found + normalized.length - 1 > trailing) return null
    val start = origin[found]
    val last = origin[found + normalized.length - 1]
    return start until last + 1
}

/** Compare what the reader selected with the stored Markdown, ignoring markers and line wrapping. */
internal fun normalizeQuoteMatch(text: String): String =
    text.replace(MARKDOWN_LINK, "$1")
        .replace(MARKDOWN_MARKERS, "")
        .replace(WHITESPACE, " ")
        .trim()
