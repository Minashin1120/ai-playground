package com.minashin1120.aiplayground.data

import org.junit.Assert.*
import org.junit.Test

class SlashCommandsTest {
    @Test fun matchingFiltersByPrefix() {
        val matches = matchingSlashCommands("/th")
        assertTrue(matches.any { it.id == "thinking" })
        assertFalse(matches.any { it.id == "search" })
    }

    @Test fun parseRequiresArgumentForThinking() {
        assertNull(parseSlashAction("/thinking"))
        val action = parseSlashAction("/thinking high")
        assertEquals("thinking", action?.id)
        assertEquals("high", action?.argument)
    }

    @Test fun newlineIsNotASlashCommand() {
        assertNull(slashToken("/search\nhello"))
        assertTrue(matchingSlashCommands("/search\nhello").isEmpty())
    }
}
