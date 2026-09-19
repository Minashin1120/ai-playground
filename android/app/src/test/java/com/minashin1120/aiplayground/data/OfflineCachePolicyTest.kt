package com.minashin1120.aiplayground.data

import org.junit.Assert.assertEquals
import org.junit.Test

class OfflineCachePolicyTest {
    @Test fun unknownHistoryCacheModeUsesSafeViewedDefault() {
        assertEquals(HistoryCacheMode.VIEWED, HistoryCacheMode.from(null))
        assertEquals(HistoryCacheMode.VIEWED, HistoryCacheMode.from("unknown"))
    }

    @Test fun persistedHistoryCacheModesRoundTrip() {
        assertEquals(HistoryCacheMode.VIEWED, HistoryCacheMode.from(HistoryCacheMode.VIEWED.value))
        assertEquals(HistoryCacheMode.FULL, HistoryCacheMode.from(HistoryCacheMode.FULL.value))
    }

    @Test fun cachedHistoryPayloadKeepsAttachmentsAndBranches() {
        val payload = org.json.JSONObject("""
            {"messages":[{"id":1,"role":"user","content":"質問","image_url":"[\"1/a.png\"]","parent_id":null},
            {"id":2,"role":"assistant","content":"回答","thought_data":"考察","image_url":"[]","parent_id":1}]}
        """.trimIndent())
        val messages = parseMessages(payload)
        assertEquals(listOf("1/a.png"), messages[0].files)
        assertEquals("考察", messages[1].thought)
        assertEquals(1, messages[1].parentId)
    }
}
