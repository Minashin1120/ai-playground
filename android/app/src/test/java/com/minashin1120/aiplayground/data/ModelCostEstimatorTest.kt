package com.minashin1120.aiplayground.data

import java.io.File
import org.junit.Assert.*
import org.junit.Test

class ModelCostEstimatorTest {
    @Test fun eachCallKeepsItsOwnLongContextTier() {
        val calls = """{"pricing_calls":[{"model":"grok-4.6","input_tokens":150000,"output_tokens":1000,"complete":true},{"model":"grok-4.6","input_tokens":150000,"output_tokens":1000,"complete":true}]}"""
        val row = ChatMessage("1", "assistant", "answer", model = "gpt-6-astra", thought = calls)
        val cost = estimator.estimate(listOf(row))
        assertEquals(2 * (150000 * 2.0 + 1000 * 6) / 1e6, cost.low, 1e-12)
        assertEquals(0, cost.long)
        assertEquals(2, cost.requests)
    }

    @Test fun ideogram45UsesImageQuantityQualityAndOperationInsteadOfTokens() {
        for ((operation, quality, unit) in listOf(Triple("generate", "high", .10), Triple("edit", "high", .22), Triple("edit", "very_low", .008))) {
            val metadata = """{"pricing_calls":[{"model":"ideogram-4.5","images":3,"operation":"$operation","quality":"$quality","complete":true}]}"""
            val row = ChatMessage("1", "assistant", "answer", model = "ideogram-4.5", thought = metadata)
            assertEquals(3 * unit, estimator.estimate(listOf(row)).low, 1e-12)
        }
    }

    @Test fun ideogram46UsesImageQuantityQualityOperationAndResolutionInsteadOfTokens() {
        val cases = listOf(
            listOf("generate", "very_low", "1k", .008),
            listOf("generate", "high", "4k", .20),
            listOf("edit", "medium", "4k", .20),
            listOf("edit", "high", "4k", .60),
        )
        for (rowCase in cases) {
            val metadata = """{"pricing_calls":[{"model":"ideogram-4.6","images":2,"operation":"${rowCase[0]}","quality":"${rowCase[1]}","resolution":"${rowCase[2]}","complete":true}]}"""
            val row = ChatMessage("1", "assistant", "answer", model = "ideogram-4.6", thought = metadata)
            assertEquals(2.0 * (rowCase[3] as Double), estimator.estimate(listOf(row)).low, 1e-12)
        }
    }
    private val estimator = ModelCostEstimator(File("src/main/assets/model-pricing.json").readText())
    private fun answer(model: String, input: Int?, output: Int = 1000) =
        ChatMessage("2", "assistant", "test", model = model, tokensIn = input, tokensOut = output)

    @Test fun thresholdsUseTheWholeRequestAndTheCorrectBoundary() {
        assertEquals((272000 * 2.0 + 1000 * 10) / 1e6, estimator.estimate(listOf(answer("gpt-6.1-sol", 272000))).low, 1e-12)
        assertEquals((272001 * 4.0 + 1000 * 15) / 1e6, estimator.estimate(listOf(answer("gpt-6.1-sol", 272001))).low, 1e-12)
        assertEquals((200000 * 2.0 + 1000 * 12) / 1e6, estimator.estimate(listOf(answer("gemini-3.1-pro-preview", 200000))).low, 1e-12)
        assertEquals((200001 * 4.0 + 1000 * 18) / 1e6, estimator.estimate(listOf(answer("gemini-3.1-pro-preview", 200001))).low, 1e-12)
        assertEquals((200000 * 4.0 + 1000 * 12) / 1e6, estimator.estimate(listOf(answer("grok-4.6", 200000))).low, 1e-12)
    }

    @Test fun modelSwitchAndBranchesDoNotRepriceEarlierAnswers() {
        val user = ChatMessage("1", "user", "test", tokensIn = 100)
        val cheap = answer("gpt-4o-mini", null, 20).copy(parentId = 1)
        val expensive = answer("gpt-6-astra", null, 30).copy(id = "3", parentId = 1)
        val all = listOf(user, cheap, expensive)
        val path = estimator.estimate(listOf(user, cheap), all)
        assertEquals((100 * .15 + 20 * .6) / 1e6, path.low, 1e-12)
        assertEquals((100 * 10.0 + 30 * 50) / 1e6, estimator.estimate(all).low - path.low, 1e-12)
        assertEquals(0, path.unpriced)
        assertEquals(1, estimator.estimate(listOf(cheap)).unpriced)
    }

    @Test fun unknownIsNotFreeAndDeepseekUsesARange() {
        assertEquals("費用予想: 算出不可", estimator.estimate(listOf(answer("unknown", 100))).label())
        val cost = estimator.estimate(listOf(answer("deepseek-v4.1-flash", 1000000, 1000000)))
        assertEquals(.75, cost.low, 1e-12)
        assertEquals(1.5, cost.high, 1e-12)
        assertEquals("費用予想: $0.750000〜$1.500000", cost.label())
    }

    @Test fun officialUsageSelectsLongContextEvenWhenDisplayInputIsSmall() {
        val row = answer("gpt-6.1-sol", 100).copy(thought = """{"pricing_input_tokens":272001}""")
        val cost = estimator.estimate(listOf(row))
        assertEquals((272001 * 4.0 + 1000 * 15) / 1e6, cost.low, 1e-12)
        assertEquals(1, cost.long)
        assertEquals(0, cost.inferred)
    }

    @Test fun batchDiscountsAndDateChangesFollowEachSavedAnswer() {
        for ((model, multiplier) in listOf("gpt-6.1-sol" to .5, "grok-4.6" to .8, "claude-sonnet-4-6" to .5, "kimi-k3" to .6)) {
            val row = answer(model, 300000)
            assertEquals(estimator.estimate(listOf(row)).low * multiplier,
                estimator.estimate(listOf(row.copy(batch = BatchInfo("JOB_STATE_SUCCEEDED")))).low, 1e-12)
        }
        val row = answer("gemini-4-argon", 1000000, 0)
        assertEquals(2.0, estimator.estimate(listOf(row)).low, 1e-12)
        assertEquals(4.0, estimator.estimate(listOf(row.copy(createdAt = "2026-12-01T00:00:00Z"))).low, 1e-12)
    }

    @Test fun batchCacheReadPricingFollowsEachProvider() {
        fun batchCost(model: String, input: Int, read: Int, output: Int): CostEstimate {
            val call = """{"model":"$model","input_tokens":$input,"cache_read_tokens":$read,"output_tokens":$output,"complete":true}"""
            val row = answer(model, input, output).copy(thought = """{"pricing_calls":[$call]}""", batch = BatchInfo("JOB_STATE_SUCCEEDED"))
            return estimator.estimate(listOf(row))
        }
        assertEquals((50000 * 1.0 + 50000 * .2 + 1000 * 6) / 1e6,
            batchCost("gemini-3.1-pro-preview", 100000, 50000, 1000).low, 1e-12)
        assertEquals((50000 * 1.0 + 50000 * .05 + 1000 * 5) / 1e6,
            batchCost("gpt-6.1-sol", 100000, 50000, 1000).low, 1e-12)
    }

    @Test fun breakdownItemsCarryEachRequestsOwnCostAndUsage() {
        val calls = """{"pricing_calls":[
            {"model":"gpt-6.1-sol","input_tokens":300000,"output_tokens":1000,"cache_read_tokens":20,"web_search_calls":2,"complete":true},
            {"model":"gpt-4o-mini","input_tokens":100,"output_tokens":20,"complete":true}]}"""
        val cost = estimator.estimate(listOf(answer("gpt-6.1-sol", 300000).copy(thought = calls)))
        assertEquals(listOf("gpt-6.1-sol", "gpt-4o-mini"), cost.items.map { it.model })
        assertEquals(cost.low, cost.items.sumOf { it.low }, 1e-12)
        assertEquals(cost.high, cost.items.sumOf { it.high }, 1e-12)
        assertTrue(cost.items[0].long)
        assertFalse(cost.items[1].long)
        assertEquals(300000, cost.items[0].input)
        assertEquals(1000, cost.items[0].output)
        assertEquals(20, cost.items[0].cacheRead)
        assertEquals(2, cost.items[0].tools)
        assertTrue(cost.items.all { it.priced && it.reason.isEmpty() })
    }

    @Test fun breakdownExplainsWhichRequestsAreUnpriced() {
        val calls = """{"pricing_calls":[
            {"model":"unknown-model-x","input_tokens":10,"output_tokens":10,"complete":true},
            {"model":"gpt-6.1-sol","complete":false},
            {"model":"gpt-6.1-sol","input_tokens":100,"output_tokens":1000,"complete":true}]}"""
        val cost = estimator.estimate(listOf(answer("gpt-6.1-sol", 100).copy(thought = calls)))
        assertEquals(listOf(false, false, true), cost.items.map { it.priced })
        assertEquals("単価未登録", cost.items[0].reason)
        assertEquals("使用量未取得", cost.items[1].reason)
        assertEquals("", cost.items[2].reason)
        assertEquals(2, cost.unpriced)
    }

    @Test fun breakdownFlagsUsageReconstructedFromHistory() {
        val user = ChatMessage("1", "user", "test", tokensIn = 200000)
        val row = ChatMessage("2", "assistant", "test", model = "gpt-6.1-sol", tokensOut = 1000, parentId = 1)
        val item = estimator.estimate(listOf(user, row)).items.last()
        assertTrue(item.inferred)
        assertEquals(200000, item.input)
        assertEquals("", item.reason)
    }

    @Test fun allUnitAndModalityModelsEstimateCorrectly() {
        fun callEst(callJson: String, date: String = "2026-11-28") = estimator.estimate(listOf(
            ChatMessage("2", "assistant", "ok", model = "test", createdAt = "${date}T00:00:00Z", thought = """{"pricing_calls":[$callJson]}""")
        ))
        assertEquals((1000 * 1.5 + 100 * 7.5 + 1120 * 30.0) / 1e6,
            callEst("""{"model":"gemini-nano-banana-2.1","input_tokens":1000,"output_tokens":1220,"image_output_tokens":1120,"images":1,"complete":true}""").low, 1e-12)
        assertEquals(1000 * 0.3 / 1e6 + 0.039,
            callEst("""{"model":"gemini-2.5-flash-image","input_tokens":1000,"output_tokens":1290,"image_output_tokens":1290,"images":1,"complete":true}""").low, 1e-12)
        assertEquals(0.034,
            callEst("""{"model":"gpt-image-1.5","images":1,"quality":"medium","size":"1024x1024","complete":true}""").low, 1e-12)
        assertEquals(0.96,
            callEst("""{"model":"veo-3.1-fast-generate-preview","duration_ms":8000,"resolution":"1080p","complete":true}""").low, 1e-12)
        assertEquals(0.09,
            callEst("""{"model":"grok-imagine-image-2.0","images":1,"resolution":"2k","quality":"medium","input_images":1,"complete":true}""").low, 1e-12)
        assertEquals(1.26,
            callEst("""{"model":"grok-imagine-video-1.5","duration_ms":5000,"resolution":"1080p","input_images":1,"complete":true}""").low, 1e-12)
        assertEquals(0.08, callEst("""{"model":"lyria-3.5","songs":1,"complete":true}""").low, 1e-12)
        assertEquals(0.04, callEst("""{"model":"mistral-ocr-4-0","pages":10,"complete":true}""").low, 1e-12)
        assertEquals(0.16, callEst("""{"model":"google-tts-studio","characters":1000,"complete":true}""").low, 1e-12)
        assertEquals(0.009, callEst("""{"model":"gpt-transcribe","duration_ms":120000,"complete":true}""").low, 1e-12)
        assertEquals(0.088, callEst("""{"model":"grok-voice-3.0","duration_ms":60000,"text_turns":2,"complete":true}""").low, 1e-12)
        val unlisted = callEst("""{"model":"gpt-5.5-mini","input_tokens":1000,"output_tokens":500,"complete":true}""")
        assertEquals(1, unlisted.priced)
        assertEquals(1, unlisted.unpriced)
        assertTrue(unlisted.high > unlisted.low && unlisted.low > 0.0)
        assertEquals(0.02, callEst("""{"model":"grok-4-1-fast-reasoning","input_tokens":100000,"output_tokens":0,"complete":true}""", "2026-05-14").low, 1e-12)
        assertEquals(0.125, callEst("""{"model":"grok-4-1-fast-reasoning","input_tokens":100000,"output_tokens":0,"complete":true}""", "2026-05-15").low, 1e-12)
    }
}
