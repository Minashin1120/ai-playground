package com.minashin1120.aiplayground.data

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class FixedPromptsTest {
    @Test fun readsWebSerializedPromptsAndDefaultModel() {
        val prompts = JSONArray().put(JSONObject().put("name", "要約").put("content", "日本語で要約してください。"))
        val gem = parseGems(JSONArray().put(JSONObject().put("uuid", "gem-1")
            .put("name", "編集者").put("default_model", "model-1").put("fixed_prompts", prompts.toString()))).single()
        assertEquals("model-1", gem.defaultModel)
        assertEquals(listOf(FixedPrompt("要約", "日本語で要約してください。")), gem.fixedPrompts)
    }

    @Test fun missingOrCorruptPromptsDoNotBreakTheGemList() {
        for (value in listOf(null, JSONObject.NULL, "not json", "{}", "[]")) {
            assertTrue(parseFixedPrompts(value).isEmpty())
        }
    }

    @Test fun ignoresInvalidRowsWhilePreservingOrderAndMultilineContent() {
        val rows = JSONArray().put(JSONObject().put("name", "one").put("content", "first\nsecond"))
            .put("invalid").put(JSONObject().put("name", "empty").put("content", ""))
            .put(JSONObject().put("name", "two").put("content", "last"))
        assertEquals(listOf(FixedPrompt("one", "first\nsecond"), FixedPrompt("two", "last")), parseFixedPrompts(rows))
    }
}
