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
}
