package com.minashin1120.aiplayground.data.backend

import com.minashin1120.aiplayground.data.PlaygroundApi
import org.json.JSONArray
import org.json.JSONObject

/**
 * The JSON endpoints the chat screen calls. [ServerChatBackend] sends them to the Playground server
 * unchanged; the local backend (no-account profile and serverless mode) answers the same paths with
 * the same JSON shapes from the device, so the screen code does not need to know which one is active.
 */
interface ChatBackend {
    suspend fun get(path: String, token: String? = null): JSONObject
    suspend fun getArray(path: String, token: String? = null): JSONArray
    suspend fun post(path: String, payload: JSONObject, token: String? = null): JSONObject
    suspend fun put(path: String, payload: JSONObject, token: String): JSONObject
    suspend fun delete(path: String, token: String): JSONObject
    /** NDJSON-style chat events (`thread_id`, `content`, `thought`, `error`, `done`, …). */
    suspend fun stream(path: String, payload: JSONObject, token: String, onAccepted: () -> Unit = {}, onEvent: (JSONObject) -> Unit)
}

class ServerChatBackend(private val api: PlaygroundApi) : ChatBackend {
    override suspend fun get(path: String, token: String?): JSONObject = api.get(path, token)
    override suspend fun getArray(path: String, token: String?): JSONArray = api.getArray(path, token)
    override suspend fun post(path: String, payload: JSONObject, token: String?): JSONObject = api.post(path, payload, token)
    override suspend fun put(path: String, payload: JSONObject, token: String): JSONObject = api.put(path, payload, token)
    override suspend fun delete(path: String, token: String): JSONObject = api.delete(path, token)
    override suspend fun stream(path: String, payload: JSONObject, token: String, onAccepted: () -> Unit, onEvent: (JSONObject) -> Unit) =
        api.stream(path, payload, token, onAccepted, onEvent)
}
