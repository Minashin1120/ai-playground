package com.minashin1120.aiplayground.data.local

import com.minashin1120.aiplayground.data.backend.ChatBackend
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class ServerHistoryAttachmentTest {
    private val server = object : ChatBackend {
        override suspend fun get(path: String, token: String?): JSONObject = error("Unexpected history request")
        override suspend fun getArray(path: String, token: String?): JSONArray = error("Unexpected history request")
        override suspend fun post(path: String, payload: JSONObject, token: String?): JSONObject = error("Unexpected history request")
        override suspend fun put(path: String, payload: JSONObject, token: String): JSONObject = error("Unexpected history request")
        override suspend fun delete(path: String, token: String): JSONObject = error("Unexpected history request")
        override suspend fun stream(path: String, payload: JSONObject, token: String, onAccepted: () -> Unit, onEvent: (JSONObject) -> Unit) = error("Unexpected history request")
    }

    @Test fun cachedFullResolutionAttachmentAvoidsAnotherDownload() = runBlocking {
        val bytes = byteArrayOf(1, 2, 3)
        var downloads = 0
        val history = ServerHistory(server, { "token" }, { _, _ -> downloads++; error("Cache should be reused") },
            { null }, { ref, limit -> assertEquals("5/image.jpg", ref); assertEquals(10L, limit); bytes }, { false }, {})
        assertArrayEquals(bytes, history.file("5/image.jpg", 10))
        assertArrayEquals(bytes, history.file("5/image.jpg", 10))
        assertEquals(0, downloads)
    }

    @Test fun cacheMissDownloadsWhenOnlineAndKeepsOfflineBehavior() = runBlocking {
        val bytes = byteArrayOf(4, 5)
        var downloads = 0
        var offline = false
        val history = ServerHistory(server, { "token" }, { _, _ -> downloads++; bytes },
            { null }, { _, _ -> null }, { offline }, {})
        assertArrayEquals(bytes, history.file("5/image.jpg", 10))
        offline = true
        assertNull(history.file("5/image.jpg", 10))
        assertEquals(1, downloads)
    }
    private fun cachedConversation(vararg rows: JSONObject) = JSONObject()
        .put("messages", JSONArray(rows.toList()))

    private fun row(id: Int, parent: Int?) = JSONObject().put("id", id)
        .put("parent_id", parent ?: JSONObject.NULL)

    @Test fun completeExplicitParentUsesCacheWithoutWaitingForServer() = runBlocking {
        val cached = cachedConversation(row(1, null), row(2, 1))
        val history = ServerHistory(server, { "token" }, { _, _ -> error("Unexpected file") },
            { cached }, { _, _ -> null }, { false }, {})
        val snapshot = history.thread("chat", 2)
        assertSame(cached, snapshot.thread)
        assertEquals(2, snapshot.messages.size)
        // Cache reuse must never acknowledge pending outbox messages as observed on the server.
        assertFalse(snapshot.observedOnServer)
    }

    @Test fun missingAncestorsAndUnknownLatestParentStillFetchServer() = runBlocking {
        var requests = 0
        val remote = cachedConversation(row(1, null), row(2, 1))
        val backend = object : ChatBackend by server {
            override suspend fun get(path: String, token: String?): JSONObject {
                requests++
                return remote
            }
        }
        for (parent in listOf(2, null)) {
            val history = ServerHistory(backend, { "token" }, { _, _ -> byteArrayOf() },
                { cachedConversation(row(2, 1)) }, { _, _ -> null }, { false }, {})
            val snapshot = history.thread("chat", parent)
            assertSame(remote, snapshot.thread)
            assertTrue(snapshot.observedOnServer)
        }
        assertEquals(2, requests)
    }

    @Test fun absentCacheStillFetchesServer() = runBlocking {
        var requests = 0
        val backend = object : ChatBackend by server {
            override suspend fun get(path: String, token: String?): JSONObject {
                requests++
                return cachedConversation(row(1, null))
            }
        }
        val history = ServerHistory(backend, { "token" }, { _, _ -> byteArrayOf() },
            { null }, { _, _ -> null }, { false }, {})
        assertTrue(history.thread("chat", 1).observedOnServer)
        assertEquals(1, requests)
    }

    @Test fun cyclicCachedAncestryIsIncomplete() {
        assertFalse(ServerHistory.complete(listOf(row(1, 2), row(2, 1)), 2))
    }

}
