package com.minashin1120.aiplayground.ui

import com.minashin1120.aiplayground.data.ChatMessage
import com.minashin1120.aiplayground.data.numericId

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
 * message id, so find the nearest preceding message whose rendered text contains it. Quotes from other
 * branches (or formulas whose rendered form differs from the stored LaTeX) have no target and stay inert.
 */
internal fun findQuotedMessageIndex(messages: List<ChatMessage>, beforeIndex: Int, quote: String): Int? {
    val needle = normalizeQuoteMatch(quote)
    if (needle.isEmpty()) return null
    val last = (beforeIndex - 1).coerceAtMost(messages.lastIndex)
    if (last < 0) return null
    for (index in last downTo 0) {
        if (normalizeQuoteMatch(messages[index].content).contains(needle)) return index
    }
    return null
}

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
