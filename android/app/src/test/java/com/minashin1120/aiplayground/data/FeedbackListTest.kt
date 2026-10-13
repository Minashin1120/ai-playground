package com.minashin1120.aiplayground.data

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class FeedbackListTest {
    @Test
    fun parsesAdministratorFields() {
        val list = parseFeedbackList(JSONObject("""{"is_admin":true,"items":[
            {"id":2,"public_id":"0123456789ab","user_id":7,"title":"","message":"m","status":"replied","admin_reply":"ok",
             "handled_by":"admin","agent_name":"","created_at":"2026-10-13T00:00:00","image_count":2,
             "log_file":"feedback/0123456789ab/logs/activity.jsonl","chat_file":null,"image_dir":"feedback/0123456789ab/images"}
        ]}"""))
        assertTrue(list.isAdmin)
        val item = list.items.single()
        assertEquals(7, item.userId)
        assertEquals("admin", item.handledBy)
        assertEquals("feedback/0123456789ab/logs/activity.jsonl", item.logFile)
        assertEquals("", item.chatFile)
        assertEquals("feedback/0123456789ab/images", item.imageDir)
    }

    @Test
    fun ownListOfAnOlderServerHasNoAdministratorFields() {
        val list = parseFeedbackList(JSONObject("""{"items":[{"id":1,"title":"t","message":"m","status":"new","admin_reply":"","created_at":"c"}]}"""))
        assertFalse(list.isAdmin)
        assertNull(list.items.single().userId)
        assertEquals("", list.items.single().logFile)
    }
}
