package com.minashin1120.aiplayground.data

import java.io.File
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class CacheToolCostTest {
    private val estimator = ModelCostEstimator(File("src/main/assets/model-pricing.json").readText())
    private fun cost(model: String, fields: JSONObject): CostEstimate {
        val call = JSONObject().put("model", model).put("input_tokens", 100).put("output_tokens", 1000).put("complete", true)
        for (key in fields.keys()) call.put(key, fields.get(key))
        val row = ChatMessage("1", "assistant", "answer", model = model, thought = JSONObject().put("pricing_calls", JSONArray().put(call)).toString())
        return estimator.estimate(listOf(row))
    }
    @Test fun cacheReadAndTwoWriteLifetimesReplaceInput() {
        val fields = JSONObject().put("input_tokens", 12000).put("cache_read_tokens", 4000).put("cache_write_tokens", 7000)
            .put("cache_write_30m_tokens", 3000).put("cache_write_12h_tokens", 4000)
        val cost = cost("gpt-6.1-sol", fields)
        assertEquals(.0399, cost.low, 1e-12)
        assertEquals(0, cost.unpriced)
    }
    @Test fun toolChargesAndAuthoritativeXaiCharge() {
        assertEquals(.0302, cost("gpt-6.1-sol", JSONObject().put("web_search_calls", 2)).low, 1e-12)
        assertEquals(.04, cost("grok-4.7", JSONObject().put("cost_in_usd_ticks", 400000000L).put("web_search_calls", 100)).low, 1e-12)
        assertEquals(.0542, cost("gemini-3.1-pro-preview", JSONObject().put("grounding_queries", 3)).low, 1e-12)
    }
    @Test fun unknownRuntimeAndCacheLifetimeRemainVisible() {
        val unknown = cost("gpt-6.1-sol", JSONObject().put("cache_write_tokens", 100))
        assertTrue(unknown.high > unknown.low)
        assertEquals(1, unknown.unpriced)
        assertEquals(1, cost("gpt-6.1-sol", JSONObject().put("code_execution_calls", 1)).unpriced)
        assertEquals(0, cost("claude-sonnet-4-6", JSONObject().put("code_execution_calls", 1).put("code_execution_free", true)).unpriced)
    }
    @Test fun sharedContainerIsOnlyChargedOnceWithinAGeneration() {
        val ledger = ModelUsageLedger()
        repeat(2) { index ->
            val call = ledger.begin("gpt-6.1-sol", "openai")
            call.observe(JSONObject("""{"output":[{"id":"code_$index","type":"code_interpreter_call","status":"completed","container_id":"PRIVATE_CONTAINER"}],"usage":{"input_tokens":100,"output_tokens":1000}}"""))
            call.finished = true
        }
        val rows = ledger.snapshot()
        assertEquals(1, rows.getJSONObject(0).getInt("container_sessions"))
        assertEquals(0, rows.getJSONObject(1).getInt("container_sessions"))
        assertFalse(rows.toString().contains("PRIVATE_CONTAINER"))
    }
    @Test fun repeatedToolEventsAndCachePartitionsHaveNoPrivateText() {
        val call = ModelUsageLedger().begin("gpt-6.1-sol", "openai")
        val event = JSONObject("""{"item":{"id":"SECRET_ID","type":"web_search_call","status":"completed","action":{"type":"search","query":"SECRET_QUERY"}}}""")
        call.observe(event); call.observe(event)
        call.observe(JSONObject("""{"usage":{"input_tokens":12000,"output_tokens":1000,"input_tokens_details":{"cached_tokens":4000,"cache_write_tokens":7000,"cache_write_tokens_details":{"ttl_30m_tokens":3000,"ttl_12h_tokens":4000}}}}"""))
        val row = call.snapshot()
        assertEquals(1, row.getInt("web_search_calls"))
        assertEquals(4000, row.getInt("cache_write_12h_tokens"))
        assertFalse(row.toString().contains("SECRET"))
    }
}
