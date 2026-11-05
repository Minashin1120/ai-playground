package com.minashin1120.aiplayground.data

import java.io.File
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class IdeogramCostTest {
    private val estimator = ModelCostEstimator(File("src/main/assets/model-pricing.json").readText())
    private fun estimate(model: String, speed: String = "default", operation: String = "generate"): CostEstimate {
        val call = JSONObject().put("model", model).put("speed", speed).put("operation", operation).put("images", 3).put("complete", true)
        val metadata = JSONObject().put("pricing_calls", JSONArray().put(call)).toString()
        return estimator.estimate(listOf(ChatMessage("2", "assistant", "answer", model = model, thought = metadata)))
    }

    @Test fun allPublishedSpeedsAndCounts() {
        val rates = mapOf("ideogram-4.0" to listOf(.03, .06, .10), "ideogram-3.0" to listOf(.03, .06, .09),
            "ideogram-2a" to listOf(.025, .04), "ideogram-2.0" to listOf(.05, .08))
        for ((model, prices) in rates) for ((index, unit) in prices.withIndex()) {
            val cost = estimate(model, listOf("turbo", "default", "quality")[index])
            assertEquals(3 * unit, cost.low, 1e-12)
            assertEquals(0, cost.unpriced)
        }
    }

    @Test fun endpointPricesIgnoreSelectedGenerationModel() {
        for (model in listOf("ideogram-4.5", "ideogram-4.0", "ideogram-3.0", "ideogram-2a", "ideogram-2.0")) {
            assertEquals(.03, estimate(model, operation = "remove_background").low, 1e-12)
            val cost = estimate(model, operation = "remove_object")
            assertEquals(.09, cost.low, 1e-12)
            assertTrue(cost.lowerBound)
            assertTrue(cost.label().contains("以上"))
            assertTrue(cost.label().contains("一部未算入"))
        }
    }

    @Test fun unpublishedLegacyQualityIsNotAnExactPrice() {
        for ((model, minimum) in listOf("ideogram-2a" to .025, "ideogram-2.0" to .05)) {
            val cost = estimate(model, "quality")
            assertEquals(3 * minimum, cost.low, 1e-12)
            assertTrue(cost.lowerBound)
            assertEquals(1, cost.unpriced)
        }
    }

    @Test fun oldRecordsRecoverSpeedFromTheSavedQuestion() {
        val parent = ChatMessage("1", "user", "prompt", promptOptions = JSONObject().put("ideogram_speed", "quality"))
        val old = ChatMessage("2", "assistant", "answer", parentId = 1, model = "ideogram-3.0",
            thought = """{"pricing_calls":[{"model":"ideogram-3.0","operation":"generate","images":2,"complete":true}]}""")
        assertEquals(.18, estimator.estimate(listOf(parent, old)).low, 1e-12)
    }
}
