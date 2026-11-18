package com.minashin1120.aiplayground.ui

import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.font.FontWeight
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** Long-press selection stays inside a Japanese word instead of crossing 。 and 、. */
class SelectionWordsTest {
    @Test fun pressAfterAPeriodStaysInsideTheNextWord() {
        val broken = "前の文となる。そのため、次".withSelectionBreaks()
        assertEquals("そのため", wordAt(broken, "た"))
        assertFalse(wordAt(broken, "た").contains('。'))
        assertFalse(wordAt(broken, "る").contains('そ'))
    }

    @Test fun pressOnToiuDoesNotTakeTheSurroundingSentence() {
        val broken = "「それぞれ1枚以上」という条件を扱いやすくするためです。".withSelectionBreaks()
        assertEquals("という", wordAt(broken, "い"))
        assertFalse(wordAt(broken, "い").contains('」'))
        assertFalse(wordAt(broken, "い").contains('条'))
    }

    @Test fun latinTextIsLeftWhole() {
        assertEquals("Hello, world", "Hello, world".withSelectionBreaks())
        assertEquals("Answerword", "Answerword".withSelectionBreaks())
        assertEquals("a $$$$ b", "a $$$$ b".withSelectionBreaks())
    }

    @Test fun breaksAreRemovedWhenTheSelectionIsCopied() {
        val broken = "となる。そのため、".withSelectionBreaks()
        assertTrue(broken.contains(SELECTION_BREAK))
        assertEquals("となる。そのため、", stripSelectionBreaks(broken))
        assertEquals("となる。そのため、", MathCopyText.restore(broken))
        assertEquals("plain", MathCopyText.restore("plain"))
    }

    @Test fun anOffsetAfterJapaneseMovesPastTheInsertedBreaks() {
        val text = "そのため、次の式"
        val at = text.indexOf('次')
        val breaks = selectionBreakOffsets(text)
        val mapped = mapSelectionRange(breaks, at..at).first
        assertEquals('次', text.withSelectionBreaks(breaks)[mapped])
    }

    @Test fun stylesStayOnTheSameCharacters() {
        val source = AnnotatedString("そのため", listOf(AnnotatedString.Range(SpanStyle(fontWeight = FontWeight.Bold), 0, 4)))
        val shown = "前となる。".let { AnnotatedString(it + source) }.withSelectionBreaks()
        val start = shown.text.indexOf("そのため")
        assertTrue(start > 0)
        assertEquals(SELECTION_BREAK, shown.text[start - 1])
        val bold = shown.spanStyles.single { it.item.fontWeight == FontWeight.Bold }
        assertEquals("そのため", shown.text.substring(bold.start, bold.end))
    }

    private fun wordAt(broken: String, needle: String): String {
        val at = broken.indexOf(needle)
        assertTrue(at >= 0)
        var start = at
        var end = at + needle.length
        while (start > 0 && broken[start - 1] != SELECTION_BREAK) start--
        while (end < broken.length && broken[end] != SELECTION_BREAK) end++
        return broken.substring(start, end)
    }
}
