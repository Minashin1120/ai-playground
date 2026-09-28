package com.minashin1120.aiplayground.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ChatNavigationHistoryTest {
    private val first = ThreadItem("first", "First", "model")
    private val second = ThreadItem("second", "Second", "model")

    @Test fun backReturnsToNewChatThenStops() {
        val history = ChatNavigationHistory()
        history.record(ChatLocation(null), ChatLocation(first))
        assertTrue(history.canGoBack)
        assertEquals(ChatLocation(null), history.pop())
        assertFalse(history.canGoBack)
        assertNull(history.pop())
    }

    @Test fun backVisitsPreviouslyOpenedChatsInReverseOrder() {
        val history = ChatNavigationHistory()
        history.record(ChatLocation(null), ChatLocation(first))
        history.record(ChatLocation(first), ChatLocation(second))
        assertEquals(ChatLocation(first), history.pop())
        assertEquals(ChatLocation(null), history.pop())
    }

    @Test fun newChatReturnsToTheThreadItWasOpenedFrom() {
        val history = ChatNavigationHistory()
        history.record(ChatLocation(first), ChatLocation(null))
        assertEquals(ChatLocation(first), history.pop())
    }

    @Test fun reopeningCurrentChatDoesNotAddAHistoryEntry() {
        val history = ChatNavigationHistory()
        history.record(ChatLocation(first), ChatLocation(first.copy(title = "Renamed")))
        history.record(ChatLocation(null), ChatLocation(null))
        assertFalse(history.canGoBack)
    }

    @Test fun deletedChatCannotBeRestoredByBack() {
        val history = ChatNavigationHistory()
        history.record(ChatLocation(first), ChatLocation(second))
        history.removeThread(first.id)
        assertFalse(history.canGoBack)
    }

    @Test fun temporaryNewChatIsRestored() {
        val history = ChatNavigationHistory()
        history.record(ChatLocation(null, temporary = true), ChatLocation(first))
        assertEquals(ChatLocation(null, temporary = true), history.pop())
    }
}
