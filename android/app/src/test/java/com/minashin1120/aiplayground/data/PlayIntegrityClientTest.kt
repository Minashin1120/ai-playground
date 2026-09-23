package com.minashin1120.aiplayground.data

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Test

class PlayIntegrityClientTest {
    @Test fun requestHashIsStableAcrossObjectKeyOrder() {
        val first = JSONObject("""{"username":"alice","nested":{"b":2,"a":1}}""")
        val reordered = JSONObject("""{"nested":{"a":1,"b":2},"username":"alice"}""")
        assertEquals(
            PlayIntegrityClient.requestHash("/api/mobile/v1/auth/login", "request-id", first),
            PlayIntegrityClient.requestHash("/api/mobile/v1/auth/login", "request-id", reordered),
        )
    }

    @Test fun requestHashChangesForDifferentActionOrBody() {
        val body = JSONObject("""{"username":"alice","password":"secret"}""")
        val original = PlayIntegrityClient.requestHash("/api/mobile/v1/auth/login", "request-id", body)
        assertNotEquals(original, PlayIntegrityClient.requestHash("/api/mobile/v1/auth/signup", "request-id", body))
        assertNotEquals(original, PlayIntegrityClient.requestHash("/api/mobile/v1/auth/login", "other-request", body))
        assertNotEquals(original, PlayIntegrityClient.requestHash(
            "/api/mobile/v1/auth/login", "request-id", JSONObject("""{"username":"alice","password":"changed"}"""),
        ))
    }
}
