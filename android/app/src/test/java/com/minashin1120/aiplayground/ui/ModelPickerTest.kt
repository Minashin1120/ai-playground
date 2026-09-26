package com.minashin1120.aiplayground.ui

import com.minashin1120.aiplayground.data.FixedPrompt
import com.minashin1120.aiplayground.data.ModelInfo
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelPickerTest {
    private fun model(id: String, tags: Set<String> = emptySet(), apiId: String = id, agentic: Boolean = false, terms: List<String> = emptyList()) =
        ModelInfo(id, id.uppercase(), "", "", "chat", emptySet(), deprecated = false, selectable = true,
            category = "Featured", tags = tags, apiId = apiId, agenticView = agentic, searchTerms = terms)

    @Test
    fun searchMatchesApiIdCapabilityTermsAndAgenticView() {
        val gpt = model("gpt-5.5", tags = setOf("openai"), apiId = "gpt-5.5-2026-08", agentic = true, terms = listOf("vision"))
        assertTrue(modelVisible(gpt, "2026-08", "all", null))
        assertTrue(modelVisible(gpt, "VISION", "all", null))
        assertTrue(modelVisible(gpt, "agentic view", "all", null))
        assertTrue(modelVisible(gpt, "featured", "all", null))
        assertFalse(modelVisible(gpt, "lyria", "all", null))
    }

    @Test
    fun tagAndPromptCacheProviderFilter() {
        val gemini = model("gemini-3.6-flash", tags = setOf("gemini", "fast"))
        assertTrue(modelVisible(gemini, "", "fast", null))
        assertFalse(modelVisible(gemini, "", "openai", null))
        assertTrue(modelVisible(gemini, "", "all", "gemini"))
        assertFalse(modelVisible(gemini, "", "all", "openai"))
    }

    @Test
    fun tagBarKeepsWebOrder() {
        assertEquals(18, MODEL_TAGS.size)
        assertEquals("All", MODEL_TAGS.first())
        assertEquals("Agentic View", MODEL_TAGS.last())
    }

    @Test
    fun gemFixedPromptsSkipIncompleteRows() {
        val rows = listOf(FixedPrompt(" a ", " b "), FixedPrompt("", "x"), FixedPrompt("y", "  "))
        assertEquals(listOf(FixedPrompt("a", "b")), collectGemFixedPrompts(rows))
    }

    @Test
    fun gemDefaultModelOffersTheChatModelsAndKeepsASavedOne() {
        val models = listOf(model("gpt-5.5"), model("old").copy(deprecated = true))
        val options = gemDefaultModelOptions(models, "")
        assertEquals(listOf("", "gpt-5.5"), options.map { it.value })
        assertEquals("Use current model", options.first().label)
        assertEquals("Featured", options[1].group)
        assertEquals(listOf("", "gpt-5.5", "retired-1"), gemDefaultModelOptions(models, "retired-1").map { it.value })
    }
}

