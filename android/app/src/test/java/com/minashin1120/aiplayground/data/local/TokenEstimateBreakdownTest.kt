package com.minashin1120.aiplayground.data.local

import com.minashin1120.aiplayground.data.direct.DirectAttachment
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class TokenEstimateBreakdownTest {
    private val result get() = JSONObject().put("tokens_total", 108).put("tokens_prompt", 8)
    private val files = listOf(DirectAttachment("a.png", "image/png"), DirectAttachment("b.png", "image/png"), DirectAttachment("c.png", "image/png"))

    @Test fun quoteAndFileMarginalsAddUpToCombinedCount() = runBlocking {
        val detail = providerTokenBreakdown(result, true, listOf("a", "b", "c"), files,
            countMessage = { 3 }, countFiles = { if (it.size == 1) 28 else 68 })
        assertEquals(3, detail.getInt("message_tokens"))
        assertEquals(5, detail.getInt("quote_tokens"))
        val rows = detail.getJSONArray("files")
        assertEquals(listOf(20, 40, 40), (0 until rows.length()).map { rows.getJSONObject(it).getInt("tokens") })
        assertEquals(0, detail.getInt("adjustment_tokens"))
    }

    @Test fun unknownSplitsRemainNullAndCombinedCountIsPreserved() = runBlocking {
        val detail = providerTokenBreakdown(result, false, listOf("a", "b", "c"), files,
            countMessage = { error("Should not count a missing quote") }, countFiles = { null })
        assertEquals(8, detail.getInt("message_tokens"))
        assertTrue(detail.getJSONArray("files").getJSONObject(0).isNull("tokens"))
        assertEquals(100, detail.getInt("adjustment_tokens"))
    }
}
