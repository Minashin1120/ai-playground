package com.minashin1120.aiplayground.data.local

import com.minashin1120.aiplayground.data.ApiException
import com.minashin1120.aiplayground.data.Diagnostics
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
 * Answers the chat screen's endpoints on the device: answers come from the AI providers ([router])
 * with the profile's own keys ([settings]).
 *
 * - No-account profile ([fallback] null): every chat lives in [store]; endpoints only the server can
 *   answer fail with `local_unavailable`.
 * - Serverless mode of a signed-in account ([remote] set): chats are read from the server ([fallback])
 *   like the normal mode; an answer is generated here on top of the server history, kept in the outbox
 *   of [store] and sent to the server when it is finished. Device chats (created while the server was
 *   out of reach, temporary chats) stay in [store] until they are uploaded. Endpoints of the account
 *   (security, library, MCP, …) go to [fallback].
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
    /** Called after every change of the device store. */
    private val onChanged: () -> Unit = {},
    /** Server history and outbox upload of serverless mode; null in the no-account profile. */
    private val remote: ServerHistory? = null,
    /** Keeps the app process alive while an image or video is generated (a foreground service on the device). */
    private val keepAlive: GenerationKeepAlive = GenerationKeepAlive.None,
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
        withContext(Dispatchers.IO) { streamLocal(path, payload, token, onAccepted, onEvent) }

    /**
     * Screen ids of device messages are local id + this: in serverless mode every device message sits in
     * the pending range, so it can never be mistaken for a server message.
     */
    private val idOffset: Int get() = if (remote != null) LocalChatStore.PENDING_ID_BASE else 0

    /** A chat answered from [store]: every chat of the no-account profile, device chats in serverless mode. */
    private fun onDevice(id: String): Boolean = remote == null || store.isDeviceThread(id)

    /** `/api/threads/<id><suffix>` with a device chat already uploaded replaced by its server id. */
    private fun threadPath(route: String, suffix: String): Pair<String, String> {
        val id = store.resolveAlias(threadIdOf(route, suffix))
        return id to "/api/threads/$id$suffix"
    }

    private fun server(): ChatBackend = fallback ?: throw unavailable()

    private suspend fun getLocal(path: String, token: String?): JSONObject {
        val (route, query) = split(path)
        return when {
            route == "/api/mobile/v1/me" -> fallback?.get(path, token)?.let { accountFrom(it) } ?: localAccount()
            route == "/api/threads" -> if (remote == null) store.listThreads(query["page"]?.toIntOrNull() ?: 1, query["q"].orEmpty())
                else withDeviceThreads(server().get(path, token), query)
            route.startsWith("/api/threads/") && route.count { it == '/' } == 3 -> {
                val (id, resolved) = threadPath(route, "")
                if (onDevice(id)) {
                    store.getThread(id, query["limit"]?.toIntOrNull()?.coerceIn(1, 200), query["before_id"]?.toIntOrNull()?.minus(idOffset), idOffset)
                        ?: throw ApiException(403, JSONObject().put("error", "403"))
                } else {
                    val reply = server().get(resolved + path.substring(route.length), token)
                    if (query["before_id"] == null) store.overlayPending(id, reply) else reply
                }
            }
            route == "/api/mobile/v1/preferences" -> preferences(token)
            route.startsWith("/c/") && route.endsWith("/pdf") -> {
                val id = store.resolveAlias(route.removePrefix("/c/").removeSuffix("/pdf"))
                if (store.threadExists(id) && onDevice(id)) threadPdf(id, query["leaf_id"]?.toIntOrNull()?.minus(idOffset))
                else server().get("/c/$id/pdf" + path.substring(route.length), token)
            }
            route == "/api/batch/jobs" && fallback == null -> JSONObject().put("jobs", JSONArray())
            route == "/api/gemini/batch/status" && fallback == null -> JSONObject().put("completed", JSONArray())
            route == "/api/mcp/servers" && fallback == null -> JSONObject().put("servers", JSONArray())
            route == "/api/storage" && fallback == null -> JSONObject().put("used_bytes", store.usageBytes()).put("limit_bytes", JSONObject.NULL)
            route == "/api/files" && fallback == null -> JSONObject().put("files", JSONArray()).put("total", 0).put("has_more", false)
            else -> fallback?.get(path, token) ?: throw unavailable()
        }
    }

    /** The server's chat list with the device chats on top of the first page. */
    private fun withDeviceThreads(reply: JSONObject, query: Map<String, String>): JSONObject {
        if ((query["page"]?.toIntOrNull() ?: 1) != 1) return reply
        val device = store.deviceThreads(query["q"].orEmpty())
        if (device.length() == 0) return reply
        val server = reply.optJSONArray("threads") ?: JSONArray()
        for (i in 0 until server.length()) device.put(server.get(i))
        return reply.put("threads", device)
    }

    private suspend fun getArrayLocal(path: String, token: String?): JSONArray = when (split(path).first) {
        "/api/gems" -> if (fallback != null) fallback.getArray(path, token) else settings.gems()
        else -> fallback?.getArray(path, token) ?: throw unavailable()
    }

    private suspend fun postLocal(path: String, payload: JSONObject, token: String?): JSONObject {
        val route = split(path).first
        return when {
            route == "/api/threads" -> createThread(payload, token)
            route.startsWith("/api/threads/") && route.endsWith("/bookmark") -> {
                val (id, resolved) = threadPath(route, "/bookmark")
                if (!onDevice(id)) return server().post(resolved, payload, token)
                val row = store.updateThread(id) { row ->
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
            // A server answer being rejoined (it has a job id) is stopped on the server.
            route == "/api/stop_chat" -> if (fallback != null && payload.has("job_id")) fallback.post(path, payload, token)
                else JSONObject().put("status", "ok")
            route == "/api/gems" && fallback == null -> settings.saveGem(null, payload)
            else -> fallback?.post(path, payload, token) ?: throw unavailable()
        }
    }

    /**
     * A new chat: on the server in serverless mode, on the device in the no-account profile, for
     * temporary chats (the server does not take messages into them) and while the server is out of reach.
     */
    private suspend fun createThread(payload: JSONObject, token: String?): JSONObject {
        val temporary = payload.optBoolean("is_temporary")
        if (remote != null && fallback != null && !temporary && !remote.isOffline()) {
            try {
                return fallback.post("/api/threads", payload, token)
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                if (!ServerHistory.unreachable(e)) throw e
            }
        }
        return store.createThread(temporary).also { onChanged() }
    }

    private suspend fun putLocal(path: String, payload: JSONObject, token: String): JSONObject {
        val route = split(path).first
        return when {
            route.startsWith("/api/threads/") && route.endsWith("/title") -> {
                val (id, resolved) = threadPath(route, "/title")
                if (!onDevice(id)) return server().put(resolved, payload, token)
                val title = normalizeTitle(payload.optString("title", "Untitled"))
                store.updateThread(id) { it.put("title", title) } ?: throw ApiException(403, JSONObject().put("error", "403"))
                onChanged()
                JSONObject().put("status", "ok").put("title", title)
            }
            route.startsWith("/api/threads/") && route.endsWith("/settings") -> {
                val (id, resolved) = threadPath(route, "/settings")
                if (!onDevice(id)) return server().put(resolved, payload, token)
                val row = store.updateThread(id, touch = true) { row ->
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
                val (id, resolved) = threadPath(route, "")
                if (!onDevice(id)) return server().delete(resolved, token).also { store.dropOutbox(id); onChanged() }
                if (!store.deleteThread(id)) throw ApiException(403, JSONObject().put("error", "403"))
                onChanged()
                JSONObject().put("status", "ok")
            }
            route.startsWith("/api/messages/") -> {
                val id = route.removePrefix("/api/messages/").toIntOrNull() ?: throw ApiException(404, JSONObject().put("error", "not_found"))
                // In serverless mode, ids below the pending range are server messages; unsent ones under
                // them are dropped by the next sync (the server rejects their parent).
                if (remote != null && !LocalChatStore.isPendingId(id)) return server().delete(path, token).also { onChanged() }
                store.deleteMessage(id - idOffset) ?: throw ApiException(404, JSONObject().put("error", "not_found"))
                onChanged()
                JSONObject().put("status", "ok")
            }
            route.startsWith("/api/gems/") && fallback == null -> { settings.deleteGem(route.removePrefix("/api/gems/")); JSONObject().put("status", "ok") }
            else -> fallback?.delete(path, token) ?: throw unavailable()
        }
    }

    private suspend fun streamLocal(path: String, payload: JSONObject, token: String, onAccepted: () -> Unit, onEvent: (JSONObject) -> Unit) {
        when {
            path == "/chat_stream" -> {
                // Image and video answers take minutes; the hold lasts until the answer is saved and uploaded.
                val mode = modeOf(payload.optString("model"))
                val hold = if (needsKeepAlive(mode)) keepAlive.begin(mode) else null
                try { generate(payload, onAccepted, onEvent) } finally { hold?.close() }
            }
            // An answer running on the server (started on the Web) can be rejoined; device answers never outlive the app.
            fallback != null && !onDevice(store.resolveAlias(payload.optString("thread_id"))) ->
                fallback.stream(path, payload, token, onAccepted, onEvent)
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

    /**
     * The chat a new answer is added to: [rows] are its messages with screen ids (server messages and
     * unsent ones in the pending range, or a device chat's local ids), [outboxId] the [store] chat that
     * keeps the new messages until the server has them.
     */
    private class Conversation(
        val outboxId: String,
        val serverId: String?,
        val settings: JSONObject,
        val rows: List<JSONObject>,
        val parentId: Int?,
    )

    private suspend fun conversation(threadId: String, body: JSONObject): Conversation {
        val explicit = body.optBoolean("parent_id_explicit")
        val requested: Int? = if (body.has("parent_id") && !body.isNull("parent_id")) body.optInt("parent_id") else null
        val history = remote
        if (history == null || store.isDeviceThread(threadId)) {
            val thread = store.threadRow(threadId)?.takeIf { !it.optBoolean("deleted") }
                ?: throw ApiException(403, JSONObject().put("error", "403"))
            val rows = store.messages(threadId)
            val parent = when {
                requested != null -> requested - idOffset
                explicit -> null
                else -> rows.maxByOrNull { it.optLong("created_ms") }?.optInt("id")
            }
            return Conversation(threadId, null, thread, rows, parent)
        }
        // A server chat: the ancestors on the server, then the unsent messages on top.
        val pending = store.pendingRows(threadId)
        val pendingById = pending.associateBy { it.optInt("id") }
        var serverParent = requested
        val seen = HashSet<Int>()
        while (serverParent != null && LocalChatStore.isPendingId(serverParent) && seen.add(serverParent)) {
            val row = pendingById[serverParent] ?: break
            serverParent = if (row.isNull("parent_id")) null else row.optInt("parent_id")
        }
        val snapshot = history.thread(threadId, serverParent?.takeIf { !LocalChatStore.isPendingId(it) })
        val rows = snapshot.messages + pending
        val parent = when {
            requested != null -> requested
            explicit -> null
            else -> rows.maxOfOrNull { it.optInt("id") }
        }
        return Conversation(store.ensureOutboxThread(threadId), threadId, snapshot.thread, rows, parent)
    }

    private suspend fun generate(body: JSONObject, onAccepted: () -> Unit, onEvent: (JSONObject) -> Unit) {
        val started = System.currentTimeMillis()
        fun elapsed() = System.currentTimeMillis() - started
        Diagnostics.log("gen.start", "model" to body.optString("model"))
        val threadId = store.resolveAlias(body.optString("thread_id"))
        val model = body.optString("model")
        val message = body.optString("message")
        val found = router.route(model, modeOf(model)) ?: throw ApiException(400, JSONObject().put("code", "serverless_unsupported")
            .put("error", "このモデルはサーバー不使用モードでは使えません。"))
        val vertex = if (found.provider == "gemini" && found.engine is com.minashin1120.aiplayground.data.direct.GeminiDirect) vertexTarget() else null
        val route = if (vertex != null) DirectRouter.Route("gemini", router.vertexGemini(vertex)) else found
        val apiKey = if (vertex != null) "" else settings.apiKeyFor(model) ?: throw ApiException(400, JSONObject().put("code", "api_key_missing")
            .put("error", "このモデルのAPIキーが端末に設定されていません。").put("model", model))
        if (body.optBoolean("batch_mode")) throw ApiException(400, JSONObject().put("code", "serverless_unsupported")
            .put("error", "Batchはサーバー不使用モードでは使えません。"))
        if (body.has("coding_target")) throw ApiException(400, JSONObject().put("code", "serverless_unsupported")
            .put("error", "Coding Modeはサーバー不使用モードではまだ使えません。"))
        val files = attachmentRefs(body)
        Diagnostics.log("gen.route", "provider" to route.provider, "vertex" to (vertex != null), "files" to files.size, "ms" to elapsed())
        val chat = conversation(threadId, body)
        Diagnostics.log("gen.conversation", "server_chat" to (chat.serverId != null), "rows" to chat.rows.size, "ms" to elapsed())
        onAccepted()
        onEvent(JSONObject().put("type", "thread_id").put("content", body.optString("thread_id")))
        val quote = body.optString("quote_text").takeIf { it.isNotBlank() && it != "null" }.orEmpty()
        val gemUuid = body.optString("gem_uuid").takeIf { it.isNotBlank() && it != "null" }.orEmpty()
        val gemName = if (gemUuid.isEmpty()) "" else (0 until settings.gems().length())
            .mapNotNull { settings.gems().optJSONObject(it) }.firstOrNull { it.optString("uuid") == gemUuid }?.optString("name").orEmpty()
        // In a server chat the parent is a server message or an unsent one (pending range).
        val parent = chat.parentId
        val onServerParent = chat.serverId != null && parent != null && !LocalChatStore.isPendingId(parent)
        val userId = store.appendMessage(chat.outboxId, LocalChatStore.NewMessage("user", message,
            parentId = if (chat.serverId != null && parent != null && !onServerParent) parent - LocalChatStore.PENDING_ID_BASE
                else if (chat.serverId == null) parent else null,
            parentServerId = if (onServerParent) parent else null,
            model = model, files = files, quote = quote, gemUuid = gemUuid, gemName = gemName))
        // Server `chat_stream`: the first message names a new chat.
        val title = chat.settings.optString("title")
        val newTitle = if (title == "New Chat" && message.isNotBlank()) {
            val snippet = message.take(50).trim().replace('\n', ' ')
            normalizeTitle(snippet + if (message.length > 50) "..." else "")
        } else null
        if (chat.serverId == null) {
            store.updateThread(chat.outboxId) { row ->
                row.put("last_model", model)
                if (gemUuid.isNotEmpty()) row.put("last_gem_uuid", gemUuid)
                newTitle?.let { row.put("title", it) }
            }
        } else newTitle?.let { store.setPendingTitle(chat.serverId, it) }
        onChanged()
        val preferences = promptPreferences()
        val system = SystemPromptBuilder.build(body, preferences,
            SystemPromptBuilder.ThreadSettings(chat.settings.optString("custom_instruction").takeIf { it != "null" }.orEmpty(),
                chat.settings.optBoolean("include_global_instruction", true)),
            defaults, route.provider)
        val all: List<JSONObject>
        val userScreenId: Int
        if (chat.serverId == null) {
            all = store.messages(chat.outboxId); userScreenId = userId
        } else {
            all = chat.rows.filterNot { LocalChatStore.isPendingId(it.optInt("id")) } + store.pendingRows(chat.serverId)
            userScreenId = LocalChatStore.PENDING_ID_BASE + userId
        }
        val assistantId = store.appendMessage(chat.outboxId, LocalChatStore.NewMessage("assistant", "", userId, model = model,
            gemUuid = gemUuid, gemName = gemName, generating = true))
        Diagnostics.log("gen.saved_question", "ms" to elapsed())
        var partialContent = ""
        var partialThought = ""
        var lastSave = 0L
        val last: JSONObject = try {
            // Attachments that cannot be read (a server file out of reach) end the answer with an error.
            val turns = history(all, userScreenId, preferences, quote)
            Diagnostics.log("gen.turns", "turns" to turns.size, "attachments" to turns.sumOf { it.attachments.size },
                "attachment_bytes" to turns.sumOf { turn -> turn.attachments.sumOf { it.bytes?.size?.toLong() ?: 0L } },
                "ms" to elapsed(), "memory" to Diagnostics.memory())
            val result = route.engine.run(DirectRequest(model, apiKey, system, turns, body), onEvent) { content, thought ->
                partialContent = content; partialThought = thought
                val now = System.currentTimeMillis()
                if (now - lastSave > 3000) { lastSave = now; store.updateMessage(chat.outboxId, assistantId, content, thought) }
            }
            val outputs = result.files.map { file -> store.saveFile(file.name, file.mime, file.bytes.inputStream(), file.bytes.size.toLong()) }
            store.updateMessage(chat.outboxId, assistantId, result.content, result.thought, result.tokensIn, result.tokensOut,
                files = outputs.takeIf { it.isNotEmpty() }, done = true)
            Diagnostics.log("gen.done", "content_chars" to result.content.length, "files" to outputs.size, "ms" to elapsed())
            JSONObject().put("type", "done")
        } catch (e: CancellationException) {
            Diagnostics.log("gen.cancelled", "content_chars" to partialContent.length, "ms" to elapsed())
            // Stopped: what arrived so far is kept; the chat screen asks for the upload.
            withContext(NonCancellable) { store.updateMessage(chat.outboxId, assistantId, partialContent, partialThought, done = true) }
            onChanged()
            throw e
        } catch (e: Exception) {
            Diagnostics.failure("gen.error", e, "ms" to elapsed())
            val text = when (e) {
                is DirectApiException -> "API Error${if (e.status > 0) " (${e.status})" else ""}: ${e.message}"
                is ApiException -> e.payload.optString("error").ifBlank { e.message }
                else -> "Connection Error: ${e.message ?: e.javaClass.simpleName}"
            }
            store.updateMessage(chat.outboxId, assistantId, errorContent(text, partialContent), partialThought, done = true)
            JSONObject().put("type", "error").put("content", text)
        }
        onChanged()
        // The finished question and answer go to the server before the screen reloads the chat; a server out of
        // reach does not hold the answer open (the sync continues or is retried in the background).
        remote?.let { history ->
            try { history.afterAnswer() } catch (e: CancellationException) { throw e } catch (_: Exception) {}
            Diagnostics.log("gen.after_answer_sync", "ms" to elapsed())
        }
        onEvent(last)
    }

    private var vertexAuth: com.minashin1120.aiplayground.data.direct.VertexAuth? = null
    private var vertexCredentials: String? = null

    /** Gemini on Vertex AI when the profile (or the account) chose it and a service account JSON is on the device. */
    private fun vertexTarget(): com.minashin1120.aiplayground.data.direct.VertexTarget? {
        val prefs = promptPreferences()
        if (prefs.optString("gemini_backend") != "vertex_ai") return null
        val json = settings.vertexCredentials() ?: throw ApiException(400, JSONObject().put("code", "api_key_missing")
            .put("error", "Vertex AIのサービスアカウントJSONが端末に設定されていません（APIキータブ）。"))
        val auth = vertexAuth?.takeIf { vertexCredentials == json } ?: router.vertexAuth(json).also { vertexAuth = it; vertexCredentials = json }
        val project = prefs.optString("gemini_vertex_project").ifBlank { auth.projectId }
        if (project.isBlank()) throw ApiException(400, JSONObject().put("error", "Vertex AIのプロジェクトIDを設定してください。"))
        val location = prefs.optString("gemini_vertex_location").ifBlank { "global" }
        return com.minashin1120.aiplayground.data.direct.VertexTarget(project, location) { auth.token() }
    }

    /** Ancestors of the new message (server `_iter_chat_history_ancestors`), oldest first, with attachments. */
    private suspend fun history(all: List<JSONObject>, userId: Int, preferences: JSONObject, quote: String): List<DirectTurn> {
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
                if (!LocalChatStore.isLocalReference(ref)) {
                    // A server attachment (a server chat, or a file sent again): read from the server or the offline cache.
                    val loadStarted = System.currentTimeMillis()
                    val bytes = remote?.file(ref, if (current) MAX_ATTACHMENT_BYTES else minOf(budget, MAX_ATTACHMENT_BYTES))
                    Diagnostics.log("gen.attachment_loaded", "source" to "server", "bytes" to (bytes?.size ?: -1), "current" to current,
                        "ms" to System.currentTimeMillis() - loadStarted)
                    if (bytes == null) {
                        if (current) throw ApiException(400, JSONObject().put("error", "添付ファイルをサーバーから読み込めませんでした。"))
                        return@mapNotNull null
                    }
                    if (!current) budget -= bytes.size
                    val name = ref.substringAfterLast('/')
                    return@mapNotNull AttachmentExtractor.prepare(name,
                        java.net.URLConnection.guessContentTypeFromName(name) ?: "application/octet-stream", bytes)
                }
                val info = store.fileInfo(ref) ?: return@mapNotNull null
                val size = info.optLong("size")
                if (!current && size > budget) return@mapNotNull null
                val loadStarted = System.currentTimeMillis()
                val bytes = store.loadFile(ref, MAX_ATTACHMENT_BYTES)
                Diagnostics.log("gen.attachment_loaded", "source" to "device", "mime" to info.optString("mime"), "bytes" to size,
                    "loaded" to (bytes != null), "current" to current, "ms" to System.currentTimeMillis() - loadStarted,
                    "memory" to Diagnostics.memory())
                if (bytes == null) return@mapNotNull null
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
        // Serverless mode can send server files again (edit, regenerate); the no-account profile has none.
        if (remote == null && refs.any { !LocalChatStore.isLocalReference(it) }) {
            throw ApiException(400, JSONObject().put("error", "サーバー上のファイルはサーバー不使用モードでは添付できません。"))
        }
        return refs.toList()
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
            val id = store.resolveAlias(payload.optString("thread_id"))
            payload.put("thread_id", id)
            if (store.threadExists(id) && onDevice(id)) {
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
            return preferences(token)
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
