package com.minashin1120.aiplayground.data

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

/** Expected values follow chat_core part07 `toggleOptions()` on the Web. */
class ComposerRulesTest {
    @Test fun gemini3FlashShowsUrlsForcesThinkingAndHidesMaps() {
        val rules = composerRules("gemini-3.6-flash", mcpEnabledServer = false)
        assertTrue(rules.urls.visible)
        assertTrue(rules.thinking.visible)
        assertEquals(true, rules.thinking.forced)
        assertTrue(rules.thinking.disabled)
        assertEquals(setOf("medium", "high"), rules.thinkingLevels)
        assertEquals("medium", rules.thinkingFallback)
        // The effort step hides Maps again for every model that is not DeepSeek / reasoning-effort.
        assertFalse(rules.maps.visible)
        assertFalse(rules.effort.visible)
        assertFalse(rules.budgetEnabled)
        assertFalse(rules.mcp.visible)
        assertTrue(rules.python.visible)
    }

    @Test fun gemini25EnablesBudgetAndKeepsThinkingOptional() {
        val rules = composerRules("gemini-2.5-pro", mcpEnabledServer = true)
        assertTrue(rules.budgetEnabled)
        assertNull(rules.thinking.forced)
        assertEquals(setOf("low", "high"), rules.thinkingLevels)
        assertTrue(rules.mcp.visible)
    }

    @Test fun gptModelsShowEffortWithModelSpecificOptions() {
        val rules = composerRules("gpt-5.6", mcpEnabledServer = false)
        assertTrue(rules.effort.visible)
        assertEquals(listOf("none", "low", "medium", "high", "xhigh", "max"), rules.effortOptions)
        assertFalse(rules.thinking.visible)
        assertTrue(rules.batch.visible)
        val grok = composerRules("grok-4.5", mcpEnabledServer = false)
        assertEquals(listOf("low", "medium", "high"), grok.effortOptions)
    }

    @Test fun deepSeekForcesSearchOffAndFallsBackToHighEffort() {
        val rules = composerRules("deepseek-v4-pro", mcpEnabledServer = false)
        assertTrue(rules.effort.visible)
        assertEquals(false, rules.search.forced)
        assertTrue(rules.search.disabled)
        assertEquals("high", rules.effortFallback)
        assertFalse(rules.effortOptions.contains("low"))
        assertTrue(rules.promptCache.disabled)
        assertEquals(false, rules.promptCache.forced)
    }

    @Test fun ocrAndImageModelsTurnSystemPromptOffAndDimModeChips() {
        val ocr = composerRules("mistral-ocr-4-0", mcpEnabledServer = true)
        assertEquals(false, ocr.sysPrompt.forced)
        assertTrue(ocr.canvas.dimmed)
        assertTrue(ocr.coding.dimmed)
        assertFalse(ocr.python.visible)
        assertFalse(ocr.mcp.visible)
        val image = composerRules("gpt-image-2", mcpEnabledServer = false)
        assertTrue(image.mask)
        assertEquals(false, image.sysPrompt.forced)
        assertFalse(image.batch.visible)
    }

    @Test fun searchModelForcesSearchOn() {
        val rules = composerRules("gpt-5-search-api", mcpEnabledServer = false)
        assertEquals(true, rules.search.forced)
        assertEquals(false, rules.python.forced)
    }

    @Test fun claudeUsesBudgetWithoutLevels() {
        val rules = composerRules("claude-opus-5-5", mcpEnabledServer = true)
        assertTrue(rules.thinking.visible)
        assertTrue(rules.budgetEnabled)
        assertTrue(rules.thinkingLevels.isEmpty())
        assertTrue(rules.mcp.visible)
    }

    @Test fun providerLockUsesWebProviderNames() {
        assertEquals("anthropic", modelApiProvider("claude-opus-5-5"))
        assertEquals("xai", modelApiProvider("grok-4.5"))
        assertEquals("gemini", modelApiProvider("veo-3.1-generate-preview"))
        assertEquals("openai", modelApiProvider("gpt-5.6"))
        assertEquals("Anthropic (Claude)", PROVIDER_LABELS["anthropic"])
    }

    @Test fun tokenEstimateLineMatchesWebTexts() {
        assertNull(tokenEstimateLine(hasInput = false, pending = false, data = null))
        assertEquals("入力トークンを計算中...", tokenEstimateLine(true, true, null)!!.text)
        assertEquals("入力トークンを計算できませんでした", tokenEstimateLine(true, false, null)!!.text)
        assertEquals("このモデルは入力トークン表示対象外です",
            tokenEstimateLine(true, false, JSONObject("""{"countable":false}"""))!!.text)
        val line = tokenEstimateLine(true, false, JSONObject(
            """{"countable":true,"tokens_total":42,"tokens_prompt":30,"tokens_files":12,"files_non_text":1,"files_error":2}"""))!!
        assertEquals("入力見積: 42 tokens (本文 30 / ファイル 12) ・ 非テキスト1件は0換算 / 失敗2件", line.text)
        assertEquals("count", line.tone)
    }
}
