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
}
