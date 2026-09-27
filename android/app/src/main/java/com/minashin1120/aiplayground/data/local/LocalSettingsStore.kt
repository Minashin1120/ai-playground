package com.minashin1120.aiplayground.data.local

import com.minashin1120.aiplayground.data.PROVIDER_KEY_FIELDS
import com.minashin1120.aiplayground.data.SECRET_MASK
import com.minashin1120.aiplayground.data.apiKeyInfoFor
import com.minashin1120.aiplayground.data.secure.EncryptedFileStore
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.UUID

/**
 * Settings, Gems and API keys of a device profile. API keys live in their own file under a separate
 * Keystore key ([secrets]) and are only ever returned to the screen masked, like the server does.
 */
class LocalSettingsStore(
    private val root: File,
    private val crypto: EncryptedFileStore,
    private val secrets: EncryptedFileStore,
) {
    private val prefsFile get() = File(root, "preferences.enc")
    private val secretsFile get() = File(root, "secrets.enc")
    private val gemsFile get() = File(root, "gems.enc")

    @Synchronized
    fun preferences(): JSONObject = crypto.readJson(prefsFile) ?: JSONObject()

    /** Stored (non-secret) settings plus masked key fields, in the `/api/mobile/v1/preferences` shape. */
    @Synchronized
    fun preferencesPayload(username: String): JSONObject {
        val payload = JSONObject(preferences().toString())
        payload.put("username", username)
        val keys = loadSecrets()
        PROVIDER_KEY_FIELDS.forEach { field -> payload.put(field, if (keys.optString(field).isNotBlank()) SECRET_MASK else "") }
        val models = JSONObject()
        keys.optJSONObject(MODEL_KEYS)?.keys()?.forEach { models.put(it, SECRET_MASK) }
        payload.put(MODEL_KEYS, models)
        payload.put(VERTEX_JSON, if (keys.optString(VERTEX_JSON).isNotBlank()) SECRET_MASK else "")
        return payload
    }

    /** Merges a settings save: key fields go to the secret file (the mask keeps the stored value). */
    @Synchronized
    fun savePreferences(payload: JSONObject) {
        val stored = preferences()
        val keys = loadSecrets()
        var keysChanged = false
        payload.keys().forEach { key ->
            val value = payload.opt(key)
            when {
                key in PROVIDER_KEY_FIELDS || key == VERTEX_JSON -> {
                    val text = value?.takeIf { it != JSONObject.NULL }?.toString().orEmpty()
                    if (text != SECRET_MASK) { keys.put(key, text.trim()); keysChanged = true }
                }
                key == MODEL_KEYS -> {
                    val submitted = value as? JSONObject ?: JSONObject()
                    val existing = keys.optJSONObject(MODEL_KEYS) ?: JSONObject()
                    val merged = JSONObject()
                    submitted.keys().forEach { model ->
                        val text = submitted.optString(model).trim()
                        val kept = if (text == SECRET_MASK) existing.optString(model) else text
                        if (kept.isNotBlank()) merged.put(model.trim().take(64), kept)
                    }
                    keys.put(MODEL_KEYS, merged); keysChanged = true
                }
                key in NEVER_STORED -> Unit
                else -> stored.put(key, value)
            }
        }
        crypto.writeJson(prefsFile, stored)
        if (keysChanged) secrets.writeJson(secretsFile, keys)
    }

    /** Server key choice: the model-specific key first (case-insensitive), then the provider key. */
    @Synchronized
    fun apiKeyFor(model: String): String? {
        val keys = loadSecrets()
        val modelKeys = keys.optJSONObject(MODEL_KEYS)
        if (modelKeys != null) {
            modelKeys.optString(model).takeIf { it.isNotBlank() }?.let { return it }
            modelKeys.keys().asSequence().firstOrNull { it.equals(model, ignoreCase = true) }
                ?.let { modelKeys.optString(it).takeIf { value -> value.isNotBlank() } }?.let { return it }
        }
        val field = apiKeyInfoFor(model)?.keyField ?: return null
        return keys.optString(field).takeIf { it.isNotBlank() }
    }

    @Synchronized
    fun providerKey(field: String): String? = loadSecrets().optString(field).takeIf { it.isNotBlank() }

    @Synchronized
    fun vertexCredentials(): String? = loadSecrets().optString(VERTEX_JSON).takeIf { it.isNotBlank() }

    /** Stores keys imported from the server account (`/api/mobile/v1/secrets/export`). */
    @Synchronized
    fun importSecrets(values: JSONObject, overwrite: Boolean) {
        val keys = loadSecrets()
        (PROVIDER_KEY_FIELDS + VERTEX_JSON).forEach { field ->
            val incoming = values.optString(field).trim()
            if (incoming.isNotBlank() && (overwrite || keys.optString(field).isBlank())) keys.put(field, incoming)
        }
        values.optJSONObject(MODEL_KEYS)?.let { incoming ->
            val merged = keys.optJSONObject(MODEL_KEYS) ?: JSONObject()
            incoming.keys().forEach { model ->
                val value = incoming.optString(model).trim()
                if (value.isNotBlank() && (overwrite || merged.optString(model).isBlank())) merged.put(model, value)
            }
            keys.put(MODEL_KEYS, merged)
        }
        secrets.writeJson(secretsFile, keys)
    }

    @Synchronized
    fun hasAnyKey(): Boolean {
        val keys = loadSecrets()
        return (PROVIDER_KEY_FIELDS + VERTEX_JSON).any { keys.optString(it).isNotBlank() } ||
            (keys.optJSONObject(MODEL_KEYS)?.length() ?: 0) > 0
    }

    private fun loadSecrets(): JSONObject = secrets.readJson(secretsFile) ?: JSONObject()

    // --- account settings cached for the system prompt in serverless mode ---

    private val serverPrefsFile get() = File(root, "server-preferences.enc")

    /** Keeps the account's settings (without key masks) so answers can be built while offline. */
    @Synchronized
    fun cacheServerPreferences(payload: JSONObject) {
        val copy = JSONObject(payload.toString())
        (PROVIDER_KEY_FIELDS + VERTEX_JSON + MODEL_KEYS).forEach { copy.remove(it) }
        crypto.writeJson(serverPrefsFile, copy)
    }

    @Synchronized
    fun cachedServerPreferences(): JSONObject? = crypto.readJson(serverPrefsFile)

    // --- Gems (`/api/gems` shapes) ---

    @Synchronized
    fun gems(): JSONArray = crypto.readJson(gemsFile)?.optJSONArray("gems") ?: JSONArray()

    @Synchronized
    fun saveGem(uuid: String?, payload: JSONObject): JSONObject {
        val rows = gems()
        val list = (0 until rows.length()).mapNotNull { rows.optJSONObject(it) }.toMutableList()
        val id = uuid ?: UUID.randomUUID().toString()
        val existing = list.firstOrNull { it.optString("uuid") == id }
        val gem = existing ?: JSONObject().put("uuid", id).also { list += it }
        payload.keys().forEach { key -> if (key != "uuid") gem.put(key, payload.opt(key)) }
        crypto.writeJson(gemsFile, JSONObject().put("gems", JSONArray(list)))
        return JSONObject().put("status", "ok").put("uuid", id).put("gem", gem)
    }

    @Synchronized
    fun deleteGem(uuid: String) {
        val rows = gems()
        val kept = (0 until rows.length()).mapNotNull { rows.optJSONObject(it) }.filterNot { it.optString("uuid") == uuid }
        crypto.writeJson(gemsFile, JSONObject().put("gems", JSONArray(kept)))
    }

    @Synchronized
    fun clearAll() { root.deleteRecursively() }

    companion object {
        const val MODEL_KEYS = "model_api_keys"
        const val VERTEX_JSON = "gemini_vertex_credentials_json"
        /** Account-level values the device never stores as a local setting. */
        val NEVER_STORED = setOf("username", "password", "current_password", "new_password")
    }
}
