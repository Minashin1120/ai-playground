package com.minashin1120.aiplayground.data.local

import com.minashin1120.aiplayground.data.backend.ChatBackend
import com.minashin1120.aiplayground.data.direct.DirectHttp
import com.minashin1120.aiplayground.data.direct.DirectRouter
import com.minashin1120.aiplayground.data.direct.ServerlessDefaults
import com.minashin1120.aiplayground.data.secure.EncryptedFileStore
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import javax.crypto.KeyGenerator

/** Serverless mode must not hide realtime models that the account server starts on the app's behalf. */
class LocalChatBackendModelSelectableTest {
    @get:Rule val folder = TemporaryFolder()

    private fun encryptedStore(id: Byte): EncryptedFileStore {
        val key = KeyGenerator.getInstance("AES").apply { init(256) }.generateKey()
        return EncryptedFileStore(byteArrayOf(id)) { key }
    }

    private fun backend(fallback: ChatBackend?, defaults: ServerlessDefaults = ServerlessDefaults(JSONObject())) =
        LocalChatBackend(
            LocalChatStore(folder.root, encryptedStore(1)),
            LocalSettingsStore(folder.root, encryptedStore(1), encryptedStore(2)),
            defaults, DirectRouter(DirectHttp()), "この端末", fallback, { "chat" },
        )

    private fun selectable(models: JSONArray, id: String): Boolean {
        for (index in 0 until models.length()) {
            val row = models.getJSONObject(index)
            if (row.optString("id") == id) return row.optBoolean("selectable", true)
        }
        error("model not found: $id")
    }

    private class CatalogBackend(private val body: JSONObject) : ChatBackend {
        override suspend fun get(path: String, token: String?): JSONObject = body
        override suspend fun getArray(path: String, token: String?): JSONArray = JSONArray()
        override suspend fun post(path: String, payload: JSONObject, token: String?): JSONObject = JSONObject()
        override suspend fun put(path: String, payload: JSONObject, token: String): JSONObject = JSONObject()
        override suspend fun delete(path: String, token: String): JSONObject = JSONObject()
        override suspend fun stream(path: String, payload: JSONObject, token: String, onAccepted: () -> Unit, onEvent: (JSONObject) -> Unit) = Unit
    }

    @Test fun signedInServerlessModeKeepsRealtimeModelsSelectable() = runBlocking {
        val live = JSONObject().put("id", "gemini-3.8-live").put("mode", "realtime_audio").put("selectable", true)
        val oneShot = JSONObject().put("id", "gpt-live-transcribe").put("mode", "transcription").put("selectable", true)
        val unroutableChat = JSONObject().put("id", "unknown-chat-model").put("mode", "chat").put("selectable", true)
        val server = CatalogBackend(JSONObject().put("models", JSONArray().put(live).put(oneShot).put(unroutableChat)))

        val models = backend(server).get("/api/mobile/v1/me").getJSONArray("models")

        assertEquals(true, selectable(models, "gemini-3.8-live"))
        assertEquals(true, selectable(models, "gpt-live-transcribe"))
        assertEquals(false, selectable(models, "unknown-chat-model"))
    }

    @Test fun noAccountProfileStillHidesRealtimeModels() = runBlocking {
        val live = JSONObject().put("id", "gemini-3.8-live").put("mode", "realtime_audio").put("selectable", true)
        val defaults = ServerlessDefaults(JSONObject().put("models", JSONArray().put(live)))

        val models = backend(null, defaults).get("/api/mobile/v1/me").getJSONArray("models")

        assertEquals(false, selectable(models, "gemini-3.8-live"))
    }
}
