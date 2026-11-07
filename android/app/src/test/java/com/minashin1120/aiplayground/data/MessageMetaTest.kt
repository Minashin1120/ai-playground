package com.minashin1120.aiplayground.data

import com.minashin1120.aiplayground.ui.tokenDetail
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class MessageMetaTest {
    @Test
    fun parsesWebMessageMeta() {
        val messages = parseMessages(JSONObject("""{"messages":[
            {"id":1,"role":"user","content":"q","quote_text":"引用","skill_name":"翻訳","is_encrypted":false,"tokens":null,"tokens_in":null,"tokens_out":null},
            {"id":2,"role":"assistant","content":"a","parent_id":1,"model":"m","tokens":1523,"tokens_in":210,"tokens_out":1313,"tokens_content":900,"tokens_thought":413,"is_encrypted":true}
        ]}"""))
        assertEquals("引用", messages[0].quote)
        assertEquals("翻訳", messages[0].skillName)
        assertEquals(false, messages[0].encrypted)
        assertNull(messages[0].tokens)
        assertEquals(1523, messages[1].tokens)
        assertEquals(413, messages[1].tokensThought)
        assertEquals(true, messages[1].encrypted)
    }

    @Test
    fun footerTokenLabelMatchesWeb() {
        val full = ChatMessage("2", "assistant", "a", tokens = 1523, tokensIn = 210, tokensOut = 1313, tokensThought = 413)
        assertEquals("In 210 / Out 1313 (Thought 413)", messageTokenLabel(full))
        assertEquals("In 1600 / Out 120", messageTokenLabel(full.copy(tokensIn = 1600, tokensOut = 120, tokensThought = 0)))
        assertEquals("42 tokens", messageTokenLabel(ChatMessage("3", "assistant", "a", tokens = 42)))
        assertNull(messageTokenLabel(ChatMessage("4", "user", "q")))
    }

    @Test
    fun totalsFollowBuildTokenTotals() {
        val totals = buildTokenTotals(listOf(
            ChatMessage("1", "user", "q"),
            ChatMessage("2", "assistant", "a", tokens = 1523, tokensIn = 210, tokensOut = 1313, tokensContent = 900, tokensThought = 413),
            ChatMessage("3", "assistant", "b", tokensIn = 100, tokensOut = 20),
        ))
        assertEquals(1643, totals.total)
        assertEquals(310, totals.tokensIn)
        assertEquals(1333, totals.tokensOut)
        assertEquals(900, totals.tokensContent)
        assertEquals(413, totals.tokensThought)
        assertEquals(TokenTotals(0, null, null, null, null), buildTokenTotals(emptyList()))
    }

    @Test
    fun cacheUsageFeedsTokenDetailAndTotals() {
        val thought = """{"pricing_calls":[
            {"model":"gpt-6.1-sol","cache_read_tokens":4000,"cache_write_tokens":7000,"cache_write_30m_tokens":3000,"cache_write_12h_tokens":4000},
            {"model":"gpt-6.1-sol","cache_read_tokens":1000,"cache_write_tokens":500}
        ]}"""
        val answer = ChatMessage("2", "assistant", "a", thought = thought, tokensIn = 12000, tokensOut = 1000)
        val detail = answer.tokenDetail()
        assertEquals(5000, detail.cacheRead)
        assertEquals(7500, detail.cacheWrite)
        assertEquals(mapOf("30m" to 3000, "12h" to 4000), detail.cacheWriteTtls)
        val totals = buildTokenTotals(listOf(answer))
        assertEquals(5000, totals.cacheRead)
        assertEquals(7500, totals.cacheWrite)
        assertEquals(mapOf("30m" to 3000, "12h" to 4000), totals.cacheWriteTtls)
        assertNull(ChatMessage("3", "assistant", "b", thought = """{"pricing_calls":[{"input_tokens":100}]}""").tokenDetail().cacheRead)
    }
}
