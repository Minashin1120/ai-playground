package com.minashin1120.aiplayground.data

import org.junit.Assert.*
import org.junit.Test

class WebModelCatalogTest {
    private fun model(id: String, selectable: Boolean = true) = ModelInfo(id, id, "test", "Test", "chat", emptySet(), false, selectable)

    @Test fun displayCatalogCannotAddModelsOrEnableUnavailableModels() {
        val result = applyWebModelCatalog(listOf(model("a", false)), """[
            {"id":"a","name":"Web name","description":"About a","price":"Price","tags":["fast"]},
            {"id":"unapproved","name":"Not allowed"}
        ]""")
        assertEquals(1, result.size)
        assertEquals("Web name", result.single().name)
        assertEquals(setOf("fast"), result.single().tags)
        assertFalse(result.single().selectable)
    }

    @Test fun webDeprecationDisablesSelectionAndRecentModelsUseDateThenRank() {
        val result = applyWebModelCatalog(listOf(model("old"), model("a"), model("b")), """[
            {"id":"old","deprecated":true,"implementedAt":"2030-01-01"},
            {"id":"a","implementedAt":"2026-09-01","implementedRank":1},
            {"id":"b","implementedAt":"2026-09-01","implementedRank":2}
        ]""")
        assertFalse(result.first().selectable)
        assertEquals(listOf("b", "a"), recentWebModels(result).map { it.id })
    }
}
