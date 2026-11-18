package com.minashin1120.aiplayground.ui

import android.icu.text.BreakIterator
import android.icu.util.ULocale
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.buildAnnotatedString

/**
 * Compose's long press selects one platform word. With a non-Japanese locale that word runs through
 * hiragana, kanji and Japanese punctuation, and a press on the first character also takes the previous
 * word, so 「そのため」 becomes 「となる。そのため、」. A zero-width space at each boundary is not a letter, so the
 * platform word stops there. Copying removes the spaces ([stripSelectionBreaks]).
 */
internal const val SELECTION_BREAK = '\u200B'

internal fun stripSelectionBreaks(text: String): String =
    if (SELECTION_BREAK !in text) text else text.replace(SELECTION_BREAK.toString(), "")

/** Offsets in [text] before which a [SELECTION_BREAK] is inserted. Empty when [text] needs none. */
internal fun selectionBreakOffsets(text: String): IntArray {
    if (text.length < 2) return IntArray(0)
    val icu = icuWordBoundaries(text)
    val points = ArrayList<Int>()
    var index = 0
    var previous = Kind.Other
    while (index < text.length) {
        val cp = text.codePointAt(index)
        val kind = kindOf(cp)
        if (index > 0 && shouldBreak(index, previous, kind, icu)) points += index
        if (kind != Kind.Mark) previous = kind
        index += Character.charCount(cp)
    }
    return points.toIntArray()
}

internal fun String.withSelectionBreaks(breaks: IntArray = selectionBreakOffsets(this)): String {
    if (breaks.isEmpty()) return this
    val out = StringBuilder(length + breaks.size)
    var cursor = 0
    for (at in breaks) {
        out.append(this, cursor, at)
        out.append(SELECTION_BREAK)
        cursor = at
    }
    out.append(this, cursor, length)
    return out.toString()
}

internal fun AnnotatedString.withSelectionBreaks(breaks: IntArray = selectionBreakOffsets(text)): AnnotatedString {
    if (breaks.isEmpty()) return this
    return buildAnnotatedString {
        var cursor = 0
        for (at in breaks) {
            append(this@withSelectionBreaks.subSequence(cursor, at))
            append(SELECTION_BREAK)
            cursor = at
        }
        append(this@withSelectionBreaks.subSequence(cursor, this@withSelectionBreaks.length))
    }
}

/** [range] of the original string after [breaks] from [selectionBreakOffsets] have been inserted. */
internal fun mapSelectionRange(breaks: IntArray, range: IntRange): IntRange {
    if (breaks.isEmpty()) return range
    return mapOffset(breaks, range.first, exclusiveEnd = false) until mapOffset(breaks, range.last + 1, exclusiveEnd = true)
}

private fun mapOffset(breaks: IntArray, offset: Int, exclusiveEnd: Boolean): Int {
    var extra = 0
    for (at in breaks) {
        if (if (exclusiveEnd) at < offset else at <= offset) extra++ else break
    }
    return offset + extra
}

private enum class Kind { Hira, Kata, Han, CjkPunct, Mark, Space, Other }

private fun shouldBreak(index: Int, previous: Kind, next: Kind, icu: Set<Int>): Boolean {
    if (previous == Kind.Space || next == Kind.Space || previous == Kind.Mark || next == Kind.Mark) return false
    val touchesJapanese = previous != Kind.Other || next != Kind.Other
    if (!touchesJapanese) return false
    if (previous != next) return true
    // Same script: split only where the Japanese dictionary does, so 「そのため」 stays one word
    // and 「するためです」 does not stay one long press.
    return index in icu && (previous == Kind.Hira || previous == Kind.Kata || previous == Kind.Han)
}

private fun kindOf(cp: Int): Kind = when {
    Character.isWhitespace(cp) || cp == SELECTION_BREAK.code || cp == 0xFEFF -> Kind.Space
    Character.getType(cp) == Character.NON_SPACING_MARK.toInt() -> Kind.Mark
    isCjkPunct(cp) -> Kind.CjkPunct
    cp in 0x3040..0x309F -> Kind.Hira
    cp in 0x30A0..0x30FF -> Kind.Kata
    cp in 0x4E00..0x9FFF || cp in 0x3400..0x4DBF || cp == 0x3005 || cp == 0x3006 || cp == 0x3007 -> Kind.Han
    cp in 0xFF66..0xFF9D -> Kind.Kata
    else -> Kind.Other
}

private fun isCjkPunct(cp: Int): Boolean = when (cp) {
    0x3001, 0x3002, 0x30FB, 0xFF0C, 0xFF0E, 0xFF01, 0xFF1F, 0xFF1A, 0xFF1B, 0x00B7, 0x2025, 0x2026 -> true
    in 0x3008..0x3011 -> true
    in 0x3014..0x301F -> true
    in 0xFF08..0xFF09 -> true
    in 0xFF3B..0xFF3D -> true
    in 0xFF5B..0xFF60 -> true
    in 0xFF61..0xFF65 -> true
    else -> false
}

private fun icuWordBoundaries(text: String): Set<Int> = try {
    val iterator = BreakIterator.getWordInstance(ULocale.JAPANESE)
    iterator.setText(text)
    val bounds = HashSet<Int>()
    var at = iterator.next()
    while (at != BreakIterator.DONE) {
        if (at in 1 until text.length) bounds += at
        at = iterator.next()
    }
    bounds
} catch (_: RuntimeException) {
    emptySet()
}
