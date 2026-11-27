package com.minashin1120.aiplayground.data.local

import com.minashin1120.aiplayground.data.backend.ChatBackend
import com.minashin1120.aiplayground.data.direct.DirectAttachment
import com.minashin1120.aiplayground.data.direct.DirectRequest
import com.minashin1120.aiplayground.data.direct.DirectRouter
import com.minashin1120.aiplayground.data.direct.DirectTurn
import com.minashin1120.aiplayground.data.direct.GeminiDirect
import com.minashin1120.aiplayground.data.direct.ServerlessDefaults
import com.minashin1120.aiplayground.data.direct.VertexTarget
import org.json.JSONArray
import org.json.JSONObject

internal class LocalChatProjects(
    private val store: LocalChatStore,
    private val settings: LocalSettingsStore,
    private val fallback: ChatBackend?,
    private val router: DirectRouter,
    private val defaults: ServerlessDefaults,
    private val vertexTarget: () -> VertexTarget?,
) {
    fun names(): Map<String, String> {
        val map = HashMap<String, String>()
        val rows = settings.projects()
        for (i in 0 until rows.length()) {
            val row = rows.optJSONObject(i) ?: continue
            val uuid = row.optString("uuid")
            if (uuid.isNotBlank()) map[uuid] = row.optString("name")
        }
        return map
    }

    fun enriched(): JSONArray {
        val counts = store.projectThreadCounts()
        val rows = settings.projects()
        val out = JSONArray()
        for (i in 0 until rows.length()) {
            val row = rows.optJSONObject(i) ?: continue
            val copy = JSONObject(row.toString())
            val uuid = copy.optString("uuid")
            copy.put("thread_count", counts[uuid] ?: 0)
            copy.put("file_count", copy.optJSONArray("files")?.length() ?: 0)
            out.put(copy)
        }
        return out
    }

    suspend fun resolve(projectUuid: String?, token: String): JSONObject? {
        val uuid = projectUuid?.trim()?.takeIf { it.isNotEmpty() && it != "null" } ?: return null
        val fb = fallback
        if (fb != null) {
            val fetched = runCatching {
                val raw = fb.get("/api/projects/$uuid", token)
                raw.optJSONObject("project") ?: raw.takeIf { it.optString("uuid").isNotBlank() }
            }.getOrNull()
            if (fetched != null) {
                settings.cacheServerProject(fetched)
                return fetched
            }
            settings.cachedServerProject(uuid)?.let { return it }
        }
        return settings.getProject(uuid)
    }

    suspend fun analyzeImage(
        project: JSONObject,
        preferences: JSONObject,
        bytes: ByteArray,
        mime: String,
        filename: String,
    ): String? {
        if (!project.optBoolean("use_vision_for_images", false)) return null
        val visionModel = project.optString("vision_model").takeIf { !project.isNull("vision_model") && it.isNotBlank() && it != "null" }
            ?: preferences.optString("default_vision_model").takeIf { it.isNotBlank() && it != "null" }
            ?: "gemini-3.6-flash"
        val found = router.route(visionModel, "chat") ?: return null
        val vertex = if (found.provider == "gemini" && found.engine is GeminiDirect) runCatching { vertexTarget() }.getOrNull() else null
        val route = if (vertex != null) DirectRouter.Route("gemini", router.vertexGemini(vertex)) else found
        val vKey = if (vertex != null) "" else (settings.apiKeyFor(visionModel) ?: return null)
        val prompt = defaults.json.optString("image_analysis_prompt").ifBlank { "Describe this image in detail, including any visible text." }
        val vBody = JSONObject().put("model", visionModel).put("enable_system_prompt", false)
        val result = runCatching {
            route.engine.run(
                DirectRequest(visionModel, vKey, "", listOf(DirectTurn("user", prompt, listOf(DirectAttachment(filename, mime, bytes)))), vBody),
                {},
            ) { _, _ -> }
        }.getOrNull() ?: return null
        return result.content.trim().takeIf { it.isNotEmpty() }
    }
}
