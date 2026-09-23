package com.minashin1120.aiplayground.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class BrowserLoginPkceTest {
    @Test fun challengeMatchesRfc7636Example() {
        assertEquals(
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM",
            BrowserLoginPkce.challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),
        )
    }

    @Test fun verifierIsUnpaddedBase64UrlAndUnique() {
        val first = BrowserLoginPkce.newVerifier()
        assertTrue(first.matches(Regex("[A-Za-z0-9_-]{43}")))
        assertNotEquals(first, BrowserLoginPkce.newVerifier())
    }
}
