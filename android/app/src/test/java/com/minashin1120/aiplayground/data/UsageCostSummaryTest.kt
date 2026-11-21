package com.minashin1120.aiplayground.data

import java.io.File
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test

class UsageCostSummaryTest {
    private val estimator = ModelCostEstimator(File("src/main/assets/model-pricing.json").readText())

    @Test fun savedCallsReuseThePerAnswerEstimator() {
        val payload = JSONObject(
            """{"period":"1m","answers":1,"unreadable":0,"truncated":false,"messages":[
               {"id":7,"model":"gpt-6.1-sol","created_at":"2026-11-21T00:00:00Z","batch":false,"flex":false,
                "pricing_calls":[{"model":"gpt-6.1-sol","input_tokens":100,"output_tokens":10,"complete":true}]}]}"""
        )
        val summary = parseUsageCostSummary(payload, estimator)
        assertEquals("1m", summary.period)
        assertEquals(1, summary.answers)
        assertEquals(0, summary.unreadable)
        assertEquals(1, summary.estimate.requests)
        assertEquals((100 * 2.0 + 10 * 10) / 1e6, summary.estimate.low, 1e-12)
    }

    @Test fun batchAndUnpricedCallsStayVisible() {
        val payload = JSONObject(
            """{"period":"7d","answers":1,"unreadable":1,"truncated":true,"messages":[
               {"id":8,"model":"gpt-6.1-sol","created_at":"2026-11-21T00:00:00Z","batch":true,"flex":false,
                "pricing_calls":[{"model":"unknown","input_tokens":100,"output_tokens":10,"complete":true}]}]}"""
        )
        val summary = parseUsageCostSummary(payload, estimator)
        assertEquals(1, summary.unreadable)
        assertEquals(true, summary.truncated)
        assertEquals(1, summary.estimate.unpriced)
        assertEquals(0, summary.estimate.priced)
    }
}
