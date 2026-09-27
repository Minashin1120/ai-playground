package com.minashin1120.aiplayground.data

import okhttp3.HttpUrl.Companion.toHttpUrl
import org.json.JSONArray
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.*
import org.junit.Test

class ServerOriginTest {
    @After fun resetOrigin() = ServerOrigin.reset()

    @Test fun parsesBareHostsAndHttpsUrls() {
        assertEquals("https://example.com/", parseServerOrigin("example.com").toString())
        assertEquals("https://example.com/", parseServerOrigin(" https://Example.com/ ").toString())
        assertEquals("https://chat.example.com:8443/", parseServerOrigin("chat.example.com:8443").toString())
        assertEquals("https://example.com/", parseServerOrigin("example.com:443").toString())
    }

    @Test fun rejectsUnsafeOrAmbiguousInput() {
        listOf(
            "", "http://example.com", "ftp://example.com", "https://user:pass@example.com",
            "https://example.com/path", "https://example.com/?q=1", "https://example.com/#x",
            "localhost", "exa mple.com", "https://[::1]/", ".example.com",
        ).forEach { assertNull(it, parseServerOrigin(it)) }
    }

    @Test fun officialOriginIsTheBuildDefault() {
        assertTrue(ServerOrigin.isOfficial)
        assertTrue(ServerOrigin.isOfficial("https://ai.minashin1120.com/".toHttpUrl()))
        assertFalse(ServerOrigin.isOfficial("https://ai.minashin1120.com:8443/".toHttpUrl()))
        assertFalse(ServerOrigin.isOfficial("https://example.com/".toHttpUrl()))
    }

    @Test fun sameOriginChecksFollowTheActiveServer() {
        assertEquals("123/pic.png", fileReferencePath("https://ai.minashin1120.com/files/123/pic.png"))
        ServerOrigin.set("https://example.com/".toHttpUrl())
        assertEquals("https://example.com", ServerOrigin.base)
        assertEquals("123/pic.png", fileReferencePath("https://example.com/files/123/pic.png"))
        assertNull(fileReferencePath("https://ai.minashin1120.com/files/123/pic.png"))
    }

    @Test fun offlineFoldersAreSeparatedPerServer() {
        assertEquals("7", offlineAccountScope(7, ServerOrigin.DEFAULT))
        assertEquals("example.com#7", offlineAccountScope(7, "https://example.com/".toHttpUrl()))
        assertEquals("example.com:8443#7", offlineAccountScope(7, "https://example.com:8443/".toHttpUrl()))
        assertEquals("example.com:8443", originLabel("https://example.com:8443/".toHttpUrl()))
    }

    private fun config(extra: JSONObject.() -> Unit = {}) = JSONObject().put("api_version", 1).put("client_id", "official-android")
        .put("google_server_client_id", "cid").put("play_integrity_cloud_project_number", "123").apply(extra)

    @Test fun serverInfoKeepsOfficialLoginMethodsForOlderConfigs() {
        val info = parseServerInfo(ServerOrigin.DEFAULT, config())!!
        assertTrue(info.googleNative && info.googleBrowser && info.minashin && info.passkey && info.playIntegrity)
        assertEquals("cid", info.googleServerClientId)
        assertEquals("123", info.integrityProjectNumber)
    }

    @Test fun selfHostedServersNeverUseOfficialOnlyMethods() {
        val origin = "https://example.com/".toHttpUrl()
        val old = parseServerInfo(origin, config())!!
        assertFalse(old.googleNative || old.playIntegrity || old.googleBrowser || old.minashin || old.browserReturn)
        assertEquals("", old.googleServerClientId)
        val current = parseServerInfo(origin, config {
            put("server_name", "Example AI")
            put("auth_callback_modes", JSONArray().put("app_link").put("app_scheme"))
            put("auth_methods", JSONObject().put("passkey", true).put("google_browser", true).put("minashin", true).put("google_native", true))
            put("sync_api_version", 1).put("secrets_export", true)
        })!!
        assertEquals("Example AI", current.name)
        assertTrue(current.googleBrowser && current.minashin && current.passkey && current.browserReturn)
        assertFalse(current.googleNative || current.playIntegrity)
        assertEquals(1, current.syncApiVersion)
        assertTrue(current.secretsExport)
        assertNull(parseServerInfo(origin, JSONObject().put("api_version", 2).put("client_id", "official-android")))
        assertNull(parseServerInfo(origin, JSONObject().put("api_version", 1).put("client_id", "other")))
    }

    @Test fun browserFlowsAskForTheAppSchemeOnlyOffTheOfficialHost() {
        assertEquals("/android/auth/google/start?a=1", withAppReturn("/android/auth/google/start?a=1"))
        ServerOrigin.set("https://example.com/".toHttpUrl())
        assertEquals("/android/auth/google/start?a=1&return=app", withAppReturn("/android/auth/google/start?a=1"))
        assertEquals("/x?return=app", withAppReturn("/x"))
    }
}
