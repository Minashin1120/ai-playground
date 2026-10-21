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
    @Test fun interactionStateSurvivesButSelectionAndFormulaContentsDoNot() {
        val secret = "PRIVATE_PROMPT_ANSWER_TITLE_FILENAME_KEY"
        val id = "12345678-1234-1234-1234-123456789abc"
        val raw = JSONObject().put("ev", "ui.interaction").put("op", id).put("parent_op", id).put("gesture", id)
            .put("ui_element", "math_block").put("interaction_stage", "settled").put("phase", "done")
            .put("scroll_x", 42).put("scroll_max", 120).put("focused", true).put("action_mode_active", false)
            .put("tex", secret).put("selection_text", secret).put("message", secret)
        val safe = LogPolicy.entry(raw)!!
        assertEquals(id, safe.getString("gesture"))
        assertEquals(id, safe.getString("parent_op"))
        assertEquals(42, safe.getInt("scroll_x"))
        assertTrue(safe.getBoolean("focused"))
        assertFalse(safe.getBoolean("action_mode_active"))
        assertFalse(safe.toString().contains(secret))
        val invalid = LogPolicy.entry(JSONObject().put("ev", "ui.interaction")
            .put("parent_op", secret).put("gesture", secret).put("interaction_stage", secret))!!
        assertEquals(1, invalid.length())
    }
}
