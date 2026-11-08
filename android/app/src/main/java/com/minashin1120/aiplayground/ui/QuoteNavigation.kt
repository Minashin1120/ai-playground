package com.minashin1120.aiplayground.ui

import com.minashin1120.aiplayground.data.ChatMessage
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
 */
internal fun findQuoteTextIndex(messages: List<ChatMessage>, beforeIndex: Int, quote: String): Int? {
    val needle = normalizeQuoteMatch(quote)
    if (needle.isEmpty()) return null
    val last = (beforeIndex - 1).coerceAtMost(messages.lastIndex)
    if (last < 0) return null
    for (index in last downTo 0) {
        if (normalizeQuoteMatch(messages[index].content).contains(needle)) return index
    }
    val compactNeedle = compactQuoteText(quote)
    if (compactNeedle.length < FUZZY_QUOTE_MIN_CHARS) return null
    var best = -1
    var bestScore = 0.0
    for (index in last downTo 0) {
        val score = compactQuoteScore(compactNeedle, compactQuoteText(messages[index].content))
        if (score >= FUZZY_QUOTE_MIN_SCORE && score > bestScore) {
            best = index
            bestScore = score
        }
    }
    return best.takeIf { it >= 0 }
}

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
