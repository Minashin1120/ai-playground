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
 * Settings, Skills and API keys of a device profile. API keys live in their own file under a separate
 * Keystore key ([secrets]) and are only ever returned to the screen masked, like the server does.
 */
class LocalSettingsStore(
    private val root: File,
    private val crypto: EncryptedFileStore,
    private val secrets: EncryptedFileStore,
) {
    private val prefsFile get() = File(root, "preferences.enc")
    private val secretsFile get() = File(root, "secrets.enc")
    private val skillsFile get() = File(root, "skills.enc")
    private val projectsFile get() = File(root, "projects.enc")
    private val serverProjectsFile get() = File(root, "server-projects.enc")

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
    fun apiKeyFor(model: String, geminiFreeTier: Boolean = false): String? {
        val keys = loadSecrets()
        // The free-tier switch uses only the free key; model-specific and paid keys are not fallbacks.
        if (geminiFreeTier && apiKeyInfoFor(model)?.provider == "gemini") {
            return keys.optString("gemini_free_key").takeIf { it.isNotBlank() }
        }
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

    // --- Skills (`/api/skills` shapes) ---

    /** Reads the skill list, migrating the older `gems.enc` file once. */
    @Synchronized
    private fun readSkillsFile(): JSONObject? {
        crypto.readJson(skillsFile)?.let { return it }
        val legacy = File(root, "gems.enc")
        if (!legacy.isFile) return null
        val migrated = crypto.readJson(legacy) ?: return null
        crypto.writeJson(skillsFile, migrated)
        legacy.delete()
        return migrated
    }

    @Synchronized
    fun skills(): JSONArray = readSkillsFile()?.optJSONArray("skills") ?: JSONArray()

    @Synchronized
    fun saveSkill(uuid: String?, payload: JSONObject): JSONObject {
        val rows = skills()
        val list = (0 until rows.length()).mapNotNull { rows.optJSONObject(it) }.toMutableList()
        val id = uuid ?: UUID.randomUUID().toString()
        val existing = list.firstOrNull { it.optString("uuid") == id }
        val skill = existing ?: JSONObject().put("uuid", id).also { list += it }
        payload.keys().forEach { key -> if (key != "uuid") skill.put(key, payload.opt(key)) }
        crypto.writeJson(skillsFile, JSONObject().put("skills", JSONArray(list)))
        return JSONObject().put("status", "ok").put("uuid", id).put("skill", skill)
    }

    @Synchronized
    fun deleteSkill(uuid: String) {
        val rows = skills()
        val kept = (0 until rows.length()).mapNotNull { rows.optJSONObject(it) }.filterNot { it.optString("uuid") == uuid }
        crypto.writeJson(skillsFile, JSONObject().put("skills", JSONArray(kept)))
    }

    // --- Projects (`/api/projects` shapes) ---

    @Synchronized
    fun projects(): JSONArray = crypto.readJson(projectsFile)?.optJSONArray("projects") ?: JSONArray()

    @Synchronized
    fun getProject(uuid: String): JSONObject? {
        val rows = projects()
        for (i in 0 until rows.length()) {
            val row = rows.optJSONObject(i) ?: continue
            if (row.optString("uuid") == uuid) return JSONObject(row.toString())
        }
        return null
    }

    @Synchronized
    fun saveProject(uuid: String?, payload: JSONObject): JSONObject {
        val rows = projects()
        val list = (0 until rows.length()).mapNotNull { rows.optJSONObject(it) }.toMutableList()
        val id = uuid?.takeIf { it.isNotBlank() } ?: UUID.randomUUID().toString()
        val existing = list.firstOrNull { it.optString("uuid") == id }
        val nowIso = java.time.Instant.now().toString()
        val project = existing ?: JSONObject()
            .put("uuid", id)
            .put("name", "")
            .put("description", "")
            .put("instruction", "")
            .put("use_vision_for_images", false)
            .put("vision_model", JSONObject.NULL)
            .put("files", JSONArray())
            .put("next_file_id", 1)
            .put("created_at", nowIso)
            .also { list += it }
        if (payload.has("name")) project.put("name", payload.optString("name").trim().take(120))
        if (payload.has("description")) project.put("description", payload.optString("description").trim().take(1000))
        if (payload.has("instruction")) project.put("instruction", payload.optString("instruction").take(100_000))
        if (payload.has("use_vision_for_images")) project.put("use_vision_for_images", payload.optBoolean("use_vision_for_images"))
        if (payload.has("vision_model")) {
            val vm = payload.optString("vision_model").takeIf { !payload.isNull("vision_model") && it.isNotBlank() }?.trim()?.take(120)
            project.put("vision_model", vm ?: JSONObject.NULL)
        }
        project.put("updated_at", nowIso)
        val files = project.optJSONArray("files") ?: JSONArray()
        project.put("file_count", files.length())
        crypto.writeJson(projectsFile, JSONObject().put("projects", JSONArray(list)))
        return JSONObject(project.toString())
    }

    @Synchronized
    fun deleteProject(uuid: String) {
        val rows = projects()
        val kept = (0 until rows.length()).mapNotNull { rows.optJSONObject(it) }.filterNot { it.optString("uuid") == uuid }
        crypto.writeJson(projectsFile, JSONObject().put("projects", JSONArray(kept)))
    }

    @Synchronized
    fun addProjectFiles(uuid: String, payload: JSONObject, fileInfoLookup: (String) -> JSONObject?): JSONObject {
        val rows = projects()
        val list = (0 until rows.length()).mapNotNull { rows.optJSONObject(it) }.toMutableList()
        val project = list.firstOrNull { it.optString("uuid") == uuid }
            ?: return JSONObject().put("error", "Project not found")
        val files = project.optJSONArray("files") ?: JSONArray().also { project.put("files", it) }
        val existing = (0 until files.length()).mapNotNull { files.optJSONObject(it) }.toMutableList()
        val existingPaths = existing.map { it.optString("filepath") }.toMutableSet()
        var nextId = project.optInt("next_file_id", existing.maxOfOrNull { it.optInt("id") }?.plus(1) ?: 1)
        val rawPaths = mutableListOf<String>()
        payload.optJSONArray("rel_paths")?.let { arr ->
            for (i in 0 until arr.length()) arr.optString(i).trim().takeIf { it.isNotBlank() }?.let(rawPaths::add)
        }
        payload.optJSONArray("filepaths")?.let { arr ->
            for (i in 0 until arr.length()) arr.optString(i).trim().takeIf { it.isNotBlank() }?.let(rawPaths::add)
        }
        payload.optString("rel_path").trim().takeIf { it.isNotBlank() }?.let(rawPaths::add)
        payload.optString("filepath").trim().takeIf { it.isNotBlank() }?.let(rawPaths::add)
        val note = payload.optString("note").trim().take(200)
        val displayNameOverride = payload.optString("display_name").trim().take(255)
        val added = JSONArray()
        val nowIso = java.time.Instant.now().toString()
        for (relPath in rawPaths.distinct()) {
            if (relPath in existingPaths) continue
            if (existing.size >= 20) break
            val info = fileInfoLookup(relPath)
            val name = displayNameOverride.ifBlank { info?.optString("name")?.takeIf { it.isNotBlank() } ?: relPath.substringAfterLast('/') }
            val mime = info?.optString("mime")?.takeIf { it.isNotBlank() }
                ?: java.net.URLConnection.guessContentTypeFromName(name)
                ?: "application/octet-stream"
            val isImage = mime.startsWith("image/")
            val size = info?.optLong("size", 0L) ?: 0L
            val item = JSONObject()
                .put("id", nextId++)
                .put("filepath", relPath)
                .put("filename", name)
                .put("url", if (relPath.startsWith("local/")) "/local-file/$relPath" else "/files/$relPath")
                .put("type", if (isImage) "image" else "file")
                .put("mime", mime)
                .put("size", size)
                .put("is_image", isImage)
                .put("note", note)
                .put("created_at", nowIso)
            existing += item
            existingPaths += relPath
            added.put(item)
        }
        project.put("files", JSONArray(existing))
        project.put("next_file_id", nextId)
        project.put("file_count", existing.size)
        project.put("updated_at", nowIso)
        crypto.writeJson(projectsFile, JSONObject().put("projects", JSONArray(list)))
        return JSONObject().put("status", "ok").put("added", added).put("files", JSONArray(existing)).put("file_count", existing.size)
    }

    @Synchronized
    fun updateProjectFile(uuid: String, fileId: Int, payload: JSONObject): JSONObject {
        val rows = projects()
        val list = (0 until rows.length()).mapNotNull { rows.optJSONObject(it) }.toMutableList()
        val project = list.firstOrNull { it.optString("uuid") == uuid }
            ?: return JSONObject().put("error", "Project not found")
        val files = project.optJSONArray("files") ?: JSONArray()
        var updated: JSONObject? = null
        for (i in 0 until files.length()) {
            val row = files.optJSONObject(i) ?: continue
            if (row.optInt("id") == fileId) {
                if (payload.has("note")) row.put("note", payload.optString("note").trim().take(200))
                if (payload.has("display_name")) {
                    val dn = payload.optString("display_name").trim().take(255)
                    if (dn.isNotBlank()) row.put("filename", dn)
                }
                updated = row
                break
            }
        }
        project.put("updated_at", java.time.Instant.now().toString())
        crypto.writeJson(projectsFile, JSONObject().put("projects", JSONArray(list)))
        return JSONObject().put("status", "ok").put("file", updated ?: JSONObject.NULL)
    }

    @Synchronized
    fun deleteProjectFile(uuid: String, fileId: Int): JSONObject {
        val rows = projects()
        val list = (0 until rows.length()).mapNotNull { rows.optJSONObject(it) }.toMutableList()
        val project = list.firstOrNull { it.optString("uuid") == uuid }
            ?: return JSONObject().put("error", "Project not found")
        val files = project.optJSONArray("files") ?: JSONArray()
        val kept = (0 until files.length()).mapNotNull { files.optJSONObject(it) }.filterNot { it.optInt("id") == fileId }
        project.put("files", JSONArray(kept))
        project.put("file_count", kept.size)
        project.put("updated_at", java.time.Instant.now().toString())
        crypto.writeJson(projectsFile, JSONObject().put("projects", JSONArray(list)))
        return JSONObject().put("status", "ok").put("file_count", kept.size)
    }

    @Synchronized
    fun isFileReferencedByProject(relPath: String): Boolean {
        val rows = projects()
        for (i in 0 until rows.length()) {
            val files = rows.optJSONObject(i)?.optJSONArray("files") ?: continue
            for (j in 0 until files.length()) {
                if (files.optJSONObject(j)?.optString("filepath") == relPath) return true
            }
        }
        return false
    }

    @Synchronized
    fun cacheServerProject(project: JSONObject) {
        val uuid = project.optString("uuid").trim()
        if (uuid.isEmpty()) return
        val existing = crypto.readJson(serverProjectsFile) ?: JSONObject()
        existing.put(uuid, JSONObject(project.toString()))
        crypto.writeJson(serverProjectsFile, existing)
    }

    @Synchronized
    fun cachedServerProject(uuid: String): JSONObject? =
        crypto.readJson(serverProjectsFile)?.optJSONObject(uuid)?.let { JSONObject(it.toString()) }

    @Synchronized
    fun clearAll() { root.deleteRecursively() }

    companion object {
        const val MODEL_KEYS = "model_api_keys"
        const val VERTEX_JSON = "gemini_vertex_credentials_json"
        /** Account-level values the device never stores as a local setting. */
        val NEVER_STORED = setOf("username", "password", "current_password", "new_password")
    }
}
