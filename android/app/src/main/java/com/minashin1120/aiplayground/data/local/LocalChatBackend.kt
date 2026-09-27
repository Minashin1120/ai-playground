package com.minashin1120.aiplayground.data.local

import com.minashin1120.aiplayground.data.ApiException
import com.minashin1120.aiplayground.data.backend.ChatBackend
import com.minashin1120.aiplayground.data.direct.AttachmentExtractor
import com.minashin1120.aiplayground.data.direct.DirectApiException
import com.minashin1120.aiplayground.data.direct.DirectRequest
import com.minashin1120.aiplayground.data.direct.DirectRouter
import com.minashin1120.aiplayground.data.direct.DirectTurn
import com.minashin1120.aiplayground.data.direct.ServerlessDefaults
import com.minashin1120.aiplayground.data.direct.SystemPromptBuilder
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.net.URLDecoder

/**
 * Answers the chat screen's endpoints on the device: chats come from [store], answers from the AI
 * providers ([router]) with the profile's own keys ([settings]). With a signed-in account in serverless
 * mode, endpoints that only the server can answer (account, security, library, MCP, …) go to
 * [fallback]; in the no-account profile they fail with `local_unavailable`.
 */
class LocalChatBackend(
    private val store: LocalChatStore,
    private val settings: LocalSettingsStore,
    private val defaults: ServerlessDefaults,
    private val router: DirectRouter,
    private val accountName: String,
    private val fallback: ChatBackend?,
    /** Model id → catalog mode (`chat`, `image`, …) of the models this profile can pick. */
    private val modeOf: (String) -> String,
    /** Called after every change so a sync can be scheduled. */
    private val onChanged: () -> Unit = {},
) : ChatBackend {

    // Every call runs on the IO dispatcher: the store reads and writes encrypted files.
    override suspend fun get(path: String, token: String?): JSONObject = withContext(Dispatchers.IO) { getLocal(path, token) }
    override suspend fun getArray(path: String, token: String?): JSONArray = withContext(Dispatchers.IO) { getArrayLocal(path, token) }
    override suspend fun post(path: String, payload: JSONObject, token: String?): JSONObject =
        withContext(Dispatchers.IO) { postLocal(path, payload, token) }
    override suspend fun put(path: String, payload: JSONObject, token: String): JSONObject =
        withContext(Dispatchers.IO) { putLocal(path, payload, token) }
    override suspend fun delete(path: String, token: String): JSONObject = withContext(Dispatchers.IO) { deleteLocal(path, token) }
    override suspend fun stream(path: String, payload: JSONObject, token: String, onAccepted: () -> Unit, onEvent: (JSONObject) -> Unit) =
        withContext(Dispatchers.IO) { streamLocal(path, payload, onAccepted, onEvent) }

    private suspend fun getLocal(path: String, token: String?): JSONObject {
        val (route, query) = split(path)
        return when {
            route == "/api/mobile/v1/me" -> fallback?.get(path, token)?.let { accountFrom(it) } ?: localAccount()
            route == "/api/threads" -> store.listThreads(query["page"]?.toIntOrNull() ?: 1, query["q"].orEmpty())
            route.startsWith("/api/threads/") && route.count { it == '/' } == 3 -> {
                val id = route.removePrefix("/api/threads/")
                store.getThread(id, query["limit"]?.toIntOrNull()?.coerceIn(1, 200), query["before_id"]?.toIntOrNull())
                    ?: throw ApiException(403, JSONObject().put("error", "403"))
            }
            route == "/api/mobile/v1/preferences" -> preferences(token)
            route.startsWith("/c/") && route.endsWith("/pdf") && store.threadExists(route.removePrefix("/c/").removeSuffix("/pdf")) ->
                threadPdf(route.removePrefix("/c/").removeSuffix("/pdf"), query["leaf_id"]?.toIntOrNull())
            route == "/api/batch/jobs" && fallback == null -> JSONObject().put("jobs", JSONArray())
            route == "/api/gemini/batch/status" && fallback == null -> JSONObject().put("completed", JSONArray())
            route == "/api/mcp/servers" && fallback == null -> JSONObject().put("servers", JSONArray())
            route == "/api/storage" && fallback == null -> JSONObject().put("used_bytes", store.usageBytes()).put("limit_bytes", JSONObject.NULL)
            route == "/api/files" && fallback == null -> JSONObject().put("files", JSONArray()).put("total", 0).put("has_more", false)
            else -> fallback?.get(path, token) ?: throw unavailable()
        }
    }

    private suspend fun getArrayLocal(path: String, token: String?): JSONArray = when (split(path).first) {
        "/api/gems" -> if (fallback != null) fallback.getArray(path, token) else settings.gems()
        else -> fallback?.getArray(path, token) ?: throw unavailable()
    }

    private suspend fun postLocal(path: String, payload: JSONObject, token: String?): JSONObject {
        val route = split(path).first
        return when {
            route == "/api/threads" -> store.createThread(payload.optBoolean("is_temporary")).also { onChanged() }
            route.startsWith("/api/threads/") && route.endsWith("/bookmark") -> {
                val row = store.updateThread(threadIdOf(route, "/bookmark")) { row ->
                    val next = !row.optBoolean("is_bookmarked")
                    row.put("is_bookmarked", next).put("bookmarked_at", if (next) System.currentTimeMillis() else 0L)
                } ?: throw ApiException(403, JSONObject().put("error", "403"))
                onChanged()
                JSONObject().put("status", "ok").put("is_bookmarked", row.optBoolean("is_bookmarked"))
            }
            route == "/api/temporary_chat/heartbeat" -> JSONObject().put("status", "ok")
            route == "/api/token_estimate" -> {
                // Server `/api/token_estimate` shape; files are not counted on the device.
                val prompt = estimateTokens(payload.optString("message") + payload.optString("quote_text"))
                val files = payload.optJSONArray("image_urls")?.length() ?: 0
                JSONObject().put("countable", true).put("tokens_total", prompt).put("tokens_prompt", prompt).put("tokens_files", 0)
                    .put("files_total", files).put("files_counted", 0).put("files_non_text", files).put("files_missing", 0).put("files_error", 0)
            }
            route == "/api/stop_chat" -> JSONObject().put("status", "ok")
            route == "/api/gems" && fallback == null -> settings.saveGem(null, payload)
            else -> fallback?.post(path, payload, token) ?: throw unavailable()
        }
    }

    private suspend fun putLocal(path: String, payload: JSONObject, token: String): JSONObject {
        val route = split(path).first
        return when {
            route.startsWith("/api/threads/") && route.endsWith("/title") -> {
                val title = normalizeTitle(payload.optString("title", "Untitled"))
                store.updateThread(threadIdOf(route, "/title")) { it.put("title", title) } ?: throw ApiException(403, JSONObject().put("error", "403"))
                onChanged()
                JSONObject().put("status", "ok").put("title", title)
            }
            route.startsWith("/api/threads/") && route.endsWith("/settings") -> {
                val row = store.updateThread(threadIdOf(route, "/settings"), touch = true) { row ->
                    if (payload.has("custom_instruction")) row.put("custom_instruction", payload.optString("custom_instruction").take(100_000))
                    if (payload.has("include_global_instruction")) row.put("include_global_instruction", payload.optBoolean("include_global_instruction"))
                    if (payload.has("is_temporary")) row.put("is_temporary", payload.optBoolean("is_temporary"))
                    if (payload.has("enable_prompt_caching")) row.put("enable_prompt_caching", payload.optBoolean("enable_prompt_caching"))
                } ?: throw ApiException(403, JSONObject().put("error", "403"))
                onChanged()
                JSONObject().put("status", "ok").put("is_temporary", row.optBoolean("is_temporary"))
                    .put("timeout_seconds", JSONObject.NULL).put("temp_chat_expires_at", JSONObject.NULL)
                    .put("temp_chat_remaining_seconds", JSONObject.NULL)
            }
            route == "/api/mobile/v1/preferences" -> savePreferences(payload, token)
            route.startsWith("/api/gems/") && fallback == null -> settings.saveGem(route.removePrefix("/api/gems/"), payload)
            else -> fallback?.put(path, payload, token) ?: throw unavailable()
        }
    }

    private suspend fun deleteLocal(path: String, token: String): JSONObject {
        val route = split(path).first
        return when {
            route.startsWith("/api/threads/") && route.count { it == '/' } == 3 -> {
                if (!store.deleteThread(route.removePrefix("/api/threads/"))) throw ApiException(403, JSONObject().put("error", "403"))
                onChanged()
                JSONObject().put("status", "ok")
            }
            route.startsWith("/api/messages/") -> {
                val id = route.removePrefix("/api/messages/").toIntOrNull() ?: throw ApiException(404, JSONObject().put("error", "not_found"))
                store.deleteMessage(id) ?: throw ApiException(404, JSONObject().put("error", "not_found"))
                onChanged()
                JSONObject().put("status", "ok")
            }
            route.startsWith("/api/gems/") && fallback == null -> { settings.deleteGem(route.removePrefix("/api/gems/")); JSONObject().put("status", "ok") }
            else -> fallback?.delete(path, token) ?: throw unavailable()
        }
    }

    private suspend fun streamLocal(path: String, payload: JSONObject, onAccepted: () -> Unit, onEvent: (JSONObject) -> Unit) {
        when (path) {
            "/chat_stream" -> generate(payload, onAccepted, onEvent)
            // Answers never keep running without the app, so there is nothing to rejoin.
            else -> throw ApiException(404, JSONObject().put("error", "not_found"))
        }
    }

    /** Server `_build_thread_pdf_payload`: the branch ending at [leafId] (or the newest message). */
    private fun threadPdf(id: String, leafId: Int?): JSONObject {
        val all = store.messages(id).sortedWith(compareBy<JSONObject> { it.optLong("created_ms") }.thenBy { it.optInt("id") })
        val byId = all.associateBy { it.optInt("id") }
        val path = ArrayList<JSONObject>()
        var cursor = leafId?.let { byId[it] } ?: all.lastOrNull()
        val seen = HashSet<Int>()
        while (cursor != null && seen.add(cursor.optInt("id"))) {
            path += cursor
            cursor = if (cursor.isNull("parent_id")) null else byId[cursor.optInt("parent_id")]
        }
        path.reverse()
        val rows = JSONArray()
        path.forEach { row ->
            val thought = row.opt("thought_data")?.takeIf { it != JSONObject.NULL }?.toString()
                ?.let { runCatching { JSONObject(it).optString("text") }.getOrDefault(it) }.orEmpty()
            rows.put(JSONObject().put("role", row.optString("role")).put("content", row.optString("content")).put("thought_text", thought))
        }
        return JSONObject().put("thread", JSONObject().put("id", id).put("title", store.threadRow(id)?.optString("title").orEmpty().ifBlank { "AI Chat" }))
            .put("messages", rows).put("leaf_id", path.lastOrNull()?.optInt("id") ?: JSONObject.NULL)
            .put("generated_at", java.time.Instant.now().toString())
    }

    // --- generation ---

    private suspend fun generate(body: JSONObject, onAccepted: () -> Unit, onEvent: (JSONObject) -> Unit) {
        val threadId = body.optString("thread_id")
        val thread = store.threadRow(threadId)?.takeIf { !it.optBoolean("deleted") }
            ?: throw ApiException(403, JSONObject().put("error", "403"))
        val model = body.optString("model")
        val message = body.optString("message")
        val apiKey = settings.apiKeyFor(model) ?: throw ApiException(400, JSONObject().put("code", "api_key_missing")
            .put("error", "このモデルのAPIキーが端末に設定されていません。").put("model", model))
        val route = router.route(model, modeOf(model)) ?: throw ApiException(400, JSONObject().put("code", "serverless_unsupported")
            .put("error", "このモデルはサーバー不使用モードでは使えません。"))
        if (body.optBoolean("batch_mode")) throw ApiException(400, JSONObject().put("code", "serverless_unsupported")
            .put("error", "Batchはサーバー不使用モードでは使えません。"))
        if (body.has("coding_target")) throw ApiException(400, JSONObject().put("code", "serverless_unsupported")
            .put("error", "Coding Modeはサーバー不使用モードではまだ使えません。"))
        val files = attachmentRefs(body)
        onAccepted()
        onEvent(JSONObject().put("type", "thread_id").put("content", threadId))
        val existing = store.messages(threadId)
        val parentId: Int? = when {
            body.optBoolean("parent_id_explicit") -> if (body.isNull("parent_id")) null else body.optInt("parent_id")
            body.has("parent_id") && !body.isNull("parent_id") -> body.optInt("parent_id")
            else -> existing.maxByOrNull { it.optLong("created_ms") }?.optInt("id")
        }
        val quote = body.optString("quote_text").takeIf { it.isNotBlank() && it != "null" }.orEmpty()
        val gemUuid = body.optString("gem_uuid").takeIf { it.isNotBlank() && it != "null" }.orEmpty()
        val gemName = if (gemUuid.isEmpty()) "" else (0 until settings.gems().length())
            .mapNotNull { settings.gems().optJSONObject(it) }.firstOrNull { it.optString("uuid") == gemUuid }?.optString("name").orEmpty()
        val userId = store.appendMessage(threadId, LocalChatStore.NewMessage("user", message, parentId, model = model,
            files = files, quote = quote, gemUuid = gemUuid, gemName = gemName))
        store.updateThread(threadId) { row ->
            row.put("last_model", model)
            if (gemUuid.isNotEmpty()) row.put("last_gem_uuid", gemUuid)
            if (row.optString("title", "New Chat") == "New Chat" && message.isNotBlank()) {
                val snippet = message.take(50).trim().replace('\n', ' ')
                row.put("title", normalizeTitle(snippet + if (message.length > 50) "..." else ""))
            }
        }
        onChanged()
        val preferences = promptPreferences()
        val system = SystemPromptBuilder.build(body, preferences,
            SystemPromptBuilder.ThreadSettings(thread.optString("custom_instruction"), thread.optBoolean("include_global_instruction", true)),
            defaults, route.provider)
        val all = store.messages(threadId)
        val turns = history(all, userId, preferences, quote)
        val assistantId = store.appendMessage(threadId, LocalChatStore.NewMessage("assistant", "", userId, model = model, gemUuid = gemUuid, gemName = gemName))
        var partialContent = ""
        var partialThought = ""
        var lastSave = 0L
        try {
            val result = route.engine.run(DirectRequest(model, apiKey, system, turns, body), onEvent) { content, thought ->
                partialContent = content; partialThought = thought
                val now = System.currentTimeMillis()
                if (now - lastSave > 3000) { lastSave = now; store.updateMessage(threadId, assistantId, content, thought) }
            }
            val outputs = result.files.map { file -> store.saveFile(file.name, file.mime, file.bytes.inputStream(), file.bytes.size.toLong()) }
            store.updateMessage(threadId, assistantId, result.content, result.thought, result.tokensIn, result.tokensOut,
                files = outputs.takeIf { it.isNotEmpty() })
            onEvent(JSONObject().put("type", "done"))
        } catch (e: CancellationException) {
            withContext(NonCancellable) { store.updateMessage(threadId, assistantId, partialContent, partialThought) }
            throw e
        } catch (e: Exception) {
            val text = when (e) {
                is DirectApiException -> "API Error${if (e.status > 0) " (${e.status})" else ""}: ${e.message}"
                else -> "Connection Error: ${e.message ?: e.javaClass.simpleName}"
            }
            store.updateMessage(threadId, assistantId, errorContent(text, partialContent), partialThought)
            onEvent(JSONObject().put("type", "error").put("content", text))
        } finally {
            onChanged()
        }
    }

    /** Ancestors of the new message (server `_iter_chat_history_ancestors`), oldest first, with attachments. */
    private fun history(all: List<JSONObject>, userId: Int, preferences: JSONObject, quote: String): List<DirectTurn> {
        val byId = all.associateBy { it.optInt("id") }
        val chain = ArrayList<JSONObject>()
        var cursor: JSONObject? = byId[userId]
        val seen = HashSet<Int>()
        while (cursor != null && seen.add(cursor.optInt("id"))) {
            chain += cursor
            cursor = if (cursor.isNull("parent_id")) null else byId[cursor.optInt("parent_id")]
        }
        chain.reverse()
        var budget = HISTORY_ATTACHMENT_BYTES
        return chain.mapIndexed { index, row ->
            val role = if (row.optString("role") == "assistant") "assistant" else "user"
            val current = index == chain.lastIndex
            val refs = messageRefs(row)
            val attachments = refs.mapNotNull { ref ->
                val info = store.fileInfo(ref) ?: return@mapNotNull null
                val size = info.optLong("size")
                if (!current && size > budget) return@mapNotNull null
                val bytes = store.loadFile(ref, MAX_ATTACHMENT_BYTES) ?: return@mapNotNull null
                if (!current) budget -= bytes.size
                AttachmentExtractor.prepare(info.optString("name", ref.substringAfterLast('/')), info.optString("mime"), bytes)
            }
            var text = row.optString("content")
            if (role == "user") {
                if (current && quote.isNotEmpty()) text = "Context (User Quote):\n\"\"\"\n$quote\n\"\"\"\n\nUser Message:\n$text"
                val names = attachments.filter { it.mime.startsWith("image/") }.map { it.name }
                val block = if (current) SystemPromptBuilder.attachmentNamesBlock(names, preferences, defaults) else ""
                if (block.isNotEmpty()) text = "$text\n\n$block"
            } else text = stripErrorFence(text)
            DirectTurn(role, text, if (role == "user") attachments else emptyList())
        }.filterNot { it.role == "assistant" && it.text.isBlank() }
    }

    private fun messageRefs(row: JSONObject): List<String> {
        val raw = row.opt("image_url")?.takeIf { it != JSONObject.NULL }?.toString().orEmpty()
        if (raw.isBlank()) return emptyList()
        return runCatching { JSONArray(raw).let { a -> (0 until a.length()).map { a.getString(it) } } }.getOrElse { listOf(raw) }
    }

    private fun attachmentRefs(body: JSONObject): List<String> {
        val refs = LinkedHashSet<String>()
        body.optJSONArray("image_items")?.let { items ->
            for (i in 0 until items.length()) items.optJSONObject(i)?.optString("path")?.takeIf { it.isNotBlank() }?.let(refs::add)
        }
        body.optJSONArray("image_urls")?.let { urls -> for (i in 0 until urls.length()) urls.optString(i).takeIf { it.isNotBlank() }?.let(refs::add) }
        val local = refs.filter { LocalChatStore.isLocalReference(it) }
        if (local.size != refs.size) throw ApiException(400, JSONObject().put("error", "サーバー上のファイルはサーバー不使用モードでは添付できません。"))
        return local
    }

    // --- settings ---

    private suspend fun preferences(token: String?): JSONObject {
        if (fallback == null) return settings.preferencesPayload(accountName).also { applyLocalDefaults(it) }
        val server = fallback.get("/api/mobile/v1/preferences", token)
        settings.cacheServerPreferences(server)
        return overlayKeys(server)
    }

    private suspend fun savePreferences(payload: JSONObject, token: String): JSONObject {
        if (payload.has("thread_id")) {
            val id = payload.optString("thread_id")
            if (store.threadExists(id)) {
                store.updateThread(id) { it.put("last_gem_uuid", payload.opt("last_gem_uuid") ?: JSONObject.NULL) }
                onChanged()
                return JSONObject().put("status", "ok")
            }
        }
        // API keys of serverless mode stay on the device; everything else follows the account (or the profile).
        val keys = JSONObject()
        val rest = JSONObject()
        payload.keys().forEach { key ->
            if (key in com.minashin1120.aiplayground.data.PROVIDER_KEY_FIELDS || key == LocalSettingsStore.MODEL_KEYS || key == LocalSettingsStore.VERTEX_JSON) keys.put(key, payload.opt(key))
            else rest.put(key, payload.opt(key))
        }
        if (keys.length() > 0) settings.savePreferences(keys)
        if (fallback == null) {
            settings.savePreferences(rest)
            return JSONObject().put("status", "ok")
        }
        return if (rest.length() > 0) fallback.put("/api/mobile/v1/preferences", rest, token) else JSONObject().put("status", "ok")
    }

    private fun overlayKeys(server: JSONObject): JSONObject {
        val local = settings.preferencesPayload(server.optString("username"))
        (com.minashin1120.aiplayground.data.PROVIDER_KEY_FIELDS + LocalSettingsStore.VERTEX_JSON + LocalSettingsStore.MODEL_KEYS).forEach { key ->
            server.put(key, local.opt(key))
        }
        return server
    }

    /** Settings the system prompt needs: the account's (cached) or this profile's. */
    private fun promptPreferences(): JSONObject {
        if (fallback != null) return settings.cachedServerPreferences() ?: JSONObject()
        return settings.preferencesPayload(accountName).also { applyLocalDefaults(it) }
    }

    private fun applyLocalDefaults(prefs: JSONObject) {
        if (!prefs.has("auto_system_prompt_notices_config")) prefs.put("auto_system_prompt_notices_config", defaults.noticesConfig)
        if (!prefs.has("global_system_prompt_uses_time_fallback")) prefs.put("global_system_prompt_uses_time_fallback", true)
        if (!prefs.has("default_model")) prefs.put("default_model", defaults.json.optString("default_model", "gemini-3.6-flash"))
    }

    private fun localAccount(): JSONObject = JSONObject().put("id", LOCAL_ACCOUNT_ID).put("username", accountName)
        .put("e2ee_enabled", false).put("encryption_mode", "device")
        .put("default_model", settings.preferences().optString("default_model").ifBlank { defaults.json.optString("default_model", "gemini-3.6-flash") })
        .put("models", runnable(defaults.json.optJSONArray("models") ?: JSONArray()))

    /** The server account, with only the models this device can run selectable. */
    private fun accountFrom(server: JSONObject): JSONObject = server.put("models", runnable(server.optJSONArray("models") ?: JSONArray()))

    /** Marks models the device cannot generate with (not yet supported in serverless mode) as not selectable. */
    private fun runnable(models: JSONArray): JSONArray {
        val out = JSONArray()
        for (i in 0 until models.length()) {
            val row = JSONObject(models.optJSONObject(i)?.toString() ?: continue)
            if (router.route(row.optString("id"), row.optString("mode", "chat")) == null) row.put("selectable", false)
            out.put(row)
        }
        return out
    }

    private fun split(path: String): Pair<String, Map<String, String>> {
        val route = path.substringBefore('?')
        val query = path.substringAfter('?', "").split('&').filter { it.contains('=') }.associate {
            URLDecoder.decode(it.substringBefore('='), "UTF-8") to URLDecoder.decode(it.substringAfter('='), "UTF-8")
        }
        return route to query
    }

    private fun threadIdOf(route: String, suffix: String) = route.removePrefix("/api/threads/").removeSuffix(suffix)

    private fun unavailable() = ApiException(403, JSONObject().put("code", "local_unavailable")
        .put("error", "この機能はサーバーにログインすると使えます。"))

    companion object {
        const val LOCAL_ACCOUNT_ID = 0
        private const val MAX_ATTACHMENT_BYTES = 48L * 1024 * 1024
        private const val HISTORY_ATTACHMENT_BYTES = 24L * 1024 * 1024

        /** Server `_normalize_thread_title`: one line, at most 200 characters. */
        fun normalizeTitle(value: String): String = value.replace(Regex("\\s+"), " ").trim().take(200).ifBlank { "Untitled" }

        /** Server `format_chat_error_content`. */
        fun errorContent(error: String, partial: String): String {
            val body = error.trim().ifBlank { "Unknown error" }.take(50_000).replace("```", "'''")
            val fence = "```chat_error\n$body\n```"
            return if (partial.isBlank()) fence else partial.trimEnd() + "\n\n" + fence
        }

        private fun stripErrorFence(text: String): String = text.replace(Regex("\\n*```chat_error\\n[\\s\\S]*?\\n```"), "").trimEnd()

        /** Rough token count (about 4 characters per token; CJK characters count one each). */
        fun estimateTokens(text: String): Int {
            val cjk = text.count { it.code in 0x3000..0x9FFF || it.code in 0xAC00..0xD7AF || it.code in 0xFF00..0xFFEF }
            return cjk + (text.length - cjk + 3) / 4
        }
    }
}
