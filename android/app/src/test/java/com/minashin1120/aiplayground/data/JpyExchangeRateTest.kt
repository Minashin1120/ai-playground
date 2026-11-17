package com.minashin1120.aiplayground.data

import org.junit.Assert.assertEquals
import org.junit.Test

class JpyExchangeRateTest {
    private val rate = JpyExchangeRate(159.6, "2026-11-16", "ECB")

    @Test fun convertsTheUsdEstimateIntoYen() {
        val cost = CostEstimate(0.75, 0.75, priced = 1)
        assertEquals("日本円換算: 約119.70円（参考レート 1ドル=159.60円・2026-11-16）", jpyCostLabel(cost, rate))
    }

    @Test fun keepsRangesLowerBoundsAndUnpricedMarks() {
        assertEquals("日本円換算: 約1.60円〜3.19円（参考レート 1ドル=159.60円・2026-11-16）",
            jpyCostLabel(CostEstimate(0.01, 0.02, priced = 1), rate))
        assertEquals("日本円換算: 約1.60円以上（参考レート 1ドル=159.60円・2026-11-16）",
            jpyCostLabel(CostEstimate(0.01, 0.01, priced = 1, lowerBound = true), rate))
        assertEquals("日本円換算: 約1.60円（一部未算入）（参考レート 1ドル=159.60円・2026-11-16）",
            jpyCostLabel(CostEstimate(0.01, 0.01, priced = 1, unpriced = 1), rate))
    }

    @Test fun tinyAndUnknownAmountsKeepTheUsdRules() {
        assertEquals("日本円換算: 約0.01円未満（参考レート 1ドル=159.60円）",
            jpyCostLabel(CostEstimate(0.0000001, 0.0000001, priced = 1), JpyExchangeRate(159.6, "", "ECB")))
        assertEquals("日本円換算: 算出不可", jpyCostLabel(CostEstimate(0.0, 0.0), rate))
    }

    @Test fun formatsYenAmountsForEachMagnitude() {
        assertEquals("0円", jpyAmountText(0.0))
        assertEquals("0.01円未満", jpyAmountText(0.005))
        assertEquals("12.34円", jpyAmountText(12.34))
        assertEquals("1,596円", jpyAmountText(1596.4))
    }
}
