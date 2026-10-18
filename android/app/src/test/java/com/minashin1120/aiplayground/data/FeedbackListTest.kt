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
    fun parsesTheReplyHistory() {
        val list = parseFeedbackList(JSONObject("""{"is_admin":true,"items":[
            {"id":3,"public_id":"0123456789ab","title":"","message":"m","status":"new","admin_reply":"ok","created_at":"c",
             "replies":[
               {"id":1,"role":"admin","author":"admin","message":"ok","created_at":"r1","log_file":null},
               {"id":2,"role":"user","author":"","message":"still broken","created_at":"r2","log_file":"feedback/0123456789ab/logs/reply-2.jsonl",
                "image_count":2,"image_dir":"feedback/0123456789ab/images/reply-2"}
             ]}
        ]}"""))
        val replies = list.items.single().replies
        assertEquals(listOf("admin", "user"), replies.map { it.role })
        assertEquals("admin", replies[0].author)
        assertEquals("", replies[0].logFile)
        assertEquals("still broken", replies[1].message)
        assertEquals("feedback/0123456789ab/logs/reply-2.jsonl", replies[1].logFile)
        assertEquals(0, replies[0].imageCount)
        assertEquals(2, replies[1].imageCount)
        assertEquals("feedback/0123456789ab/images/reply-2", replies[1].imageDir)
    }

    @Test
    fun anOlderServerShowsTheLatestAdministratorReplyAsTheHistory() {
        val item = parseFeedbackList(JSONObject("""{"items":[{"id":1,"title":"t","message":"m","status":"replied","admin_reply":"thanks","created_at":"c","updated_at":"u"}]}""")).items.single()
        assertEquals(listOf(FeedbackReply(0, "admin", "thanks", "u")), item.replies)
        val empty = parseFeedbackList(JSONObject("""{"items":[{"id":1,"title":"t","message":"m","status":"new","admin_reply":"","created_at":"c"}]}""")).items.single()
        assertTrue(empty.replies.isEmpty())
    }

    @Test
    fun ownListOfAnOlderServerHasNoAdministratorFields() {
        val list = parseFeedbackList(JSONObject("""{"items":[{"id":1,"title":"t","message":"m","status":"new","admin_reply":"","created_at":"c"}]}"""))
        assertFalse(list.isAdmin)
        assertNull(list.items.single().userId)
        assertEquals("", list.items.single().logFile)
    }
}
