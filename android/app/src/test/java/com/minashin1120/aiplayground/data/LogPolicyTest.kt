package com.minashin1120.aiplayground.data

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class LogPolicyTest {
    @Test fun excludesContentFromEveryBoundary() {
        val secret = "PRIVATE_PROMPT_ANSWER_TITLE_FILENAME_KEY"
        val raw = JSONObject().put("ev", "send.error").put("t", 123L).put("ms", 7).put("status", 500)
            .put("message", secret).put("title", secret).put("filename", secret).put("model", secret)
            .put("path", "/files/$secret").put("detail", JSONObject().put("bytes", secret))
        val safe = LogPolicy.entry(raw)!!
        assertFalse(safe.toString().contains(secret))
        assertEquals(7, safe.getInt("ms"))
        assertEquals(500, safe.getInt("status"))
        assertNull(LogPolicy.entry(JSONObject().put("ev", secret)))
    }
    @Test fun routesNeverRetainDynamicSegmentsOrQueries() {
        assertEquals("/api/threads/<thread_id>", LogPolicy.route("/api/threads/private-title?key=private-key"))
        assertNull(LogPolicy.route("/unknown/private-filename.txt"))
        assertEquals("/api/threads/<thread_id>", LogPolicy.route(LogPolicy.route("/api/threads/private-id")!!))
    }
}
