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
        // Concatenate annotated strings. AnnotatedString(string + annotated) keeps only the characters.
        val shown = (AnnotatedString("前となる。") + source).withSelectionBreaks()
        val start = shown.text.indexOf("そのため")
        assertTrue("そのため stays in [${shown.text}]", start > 0)
        assertEquals("a break precedes そのため", SELECTION_BREAK, shown.text[start - 1])
        val bolds = shown.spanStyles.filter { it.item.fontWeight == FontWeight.Bold }
        assertEquals("one bold span in $bolds", 1, bolds.size)
        assertEquals("bold text", "そのため", shown.text.substring(bolds[0].start, bolds[0].end))
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
