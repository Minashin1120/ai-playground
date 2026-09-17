package com.minashin1120.aiplayground

import android.app.Application
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.net.Uri
import android.os.Build
import android.os.SystemClock
import android.provider.OpenableColumns
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.minashin1120.aiplayground.data.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import okio.BufferedSink
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.net.URLEncoder
import java.util.UUID

data class ChatState(
    val starting: Boolean = true, val account: Account? = null,
    val pairing: Boolean = false, val userCode: String = "", val busy: Boolean = false,
    val threads: List<ThreadItem> = emptyList(), val nextPage: Int? = null, val search: String = "",
    val selected: ThreadItem? = null, val messages: List<ChatMessage> = emptyList(),
    val hasOlder: Boolean = false, val oldestId: String? = null,
    val allMessages: List<ChatMessage> = emptyList(), val leafId: Int? = null, val editingMessageId: String? = null,
    val customInstruction: String = "", val includeGlobalInstruction: Boolean = true,
    val newThreadTemporary: Boolean = false, val tempChatRemainingSeconds: Long? = null,
    val draft: String = "", val model: String = "", val attachments: List<Attachment> = emptyList(),
    val enableThinking: Boolean = false, val enableSearch: Boolean = false,
    val enableUrlContext: Boolean = false, val enableMaps: Boolean = false,
    val enableFileCreation: Boolean = true, val enableSystemPrompt: Boolean = false,
    val enablePromptCache: Boolean = false,
    val generationValues: Map<String, Map<String, String>> = emptyMap(),
    val batchMode: Boolean = false, val enablePython: Boolean = false, val enableMcp: Boolean = true,
    val canvasMode: Boolean = false, val codingMode: Boolean = false,
    val codingTarget: CodingTarget? = null,
    val imageMask: String? = null,
    val uploading: Boolean = false, val streaming: Boolean = false,
    val uploadSent: Long = 0L, val uploadTotal: Long = 0L, val uploadName: String = "",
    val library: List<LibraryFile> = emptyList(), val libraryBusy: Boolean = false,
    val libraryQuery: String = "", val libraryFavoritesOnly: Boolean = false,
    val libraryHasMore: Boolean = false, val libraryTotal: Int = 0,
    val gems: List<Gem> = emptyList(), val gemsBusy: Boolean = false, val selectedGem: Gem? = null,
    val preferences: Preferences? = null, val prefsBusy: Boolean = false,
    val compression: CompressionSettings = CompressionSettings(),
    val mcpServers: List<McpServerInfo> = emptyList(), val mcpBusy: Boolean = false,
    val feedbackItems: List<FeedbackItem> = emptyList(), val feedbackBusy: Boolean = false,
    val storage: StorageUsage? = null,
    val batchJobs: List<BatchJob> = emptyList(), val batchBusy: Boolean = false,
    val realtime: RealtimeState = RealtimeState(), val lyria: LyriaState = LyriaState(),
    val liveContent: String = "", val liveThought: String = "", val status: String = "",
    val cards: List<StatusCard> = emptyList(),
    val mcpDecision: McpDecision? = null,
    val offline: Boolean = false,
    val jobId: String? = null, val retryAvailable: Boolean = false, val notice: String? = null,
)

class ChatViewModel(application: Application) : AndroidViewModel(application) {
    private val api = PlaygroundApi()
    private val store = TokenStore(application)
    private val prefs = application.getSharedPreferences("navigation", 0)
    private val mutable = MutableStateFlow(ChatState())
    val state = mutable.asStateFlow()
    private var session: StoredSession? = null
    private var foreground = false
    private var pairingJob: Job? = null
    private var navigationJob: Job? = null
    private var streamJob: Job? = null
    private var uploadJob: Job? = null
    private var heartbeatJob: Job? = null
    private var libraryJob: Job? = null
    private var batchPollJob: Job? = null
    private var realtimeStreamJob: Job? = null
    private var realtimeCaptureJob: Job? = null
    private var realtimeTrack: AudioTrack? = null
    private var lyriaStreamJob: Job? = null
    private var lyriaTrack: AudioTrack? = null
    private data class Submission(val body: JSONObject, val files: List<Attachment>)
    private var failed: Submission? = null
    private var pendingParentId: Int? = null

    init { viewModelScope.launch {
        mutable.update { it.copy(compression = compressionSettingsFrom(prefs)) }
        session = withContext(Dispatchers.IO) { store.load() }
        if (session != null) runCatching { loadAccount() }.onFailure { report(it) }
        mutable.update { it.copy(starting = false) }
    } }
    private fun token(): String = session?.token ?: throw IOException("端末連携が必要です。")
    private suspend fun loadAccount() {
        val me = api.get("/api/mobile/v1/me", token())
        val serverModels = parseModels(me)
        val displayModels = runCatching {
            val json = withContext(Dispatchers.IO) {
                getApplication<Application>().assets.open("web-model-catalog.json").bufferedReader().use { it.readText() }
            }
            applyWebModelCatalog(serverModels, json)
        }.getOrDefault(serverModels)
        val account = Account(me.getInt("id"), me.getString("username"), displayModels,
            me.optString("default_model"), me.optBoolean("e2ee_enabled"))
        val chosen = prefs.getString("model_${account.id}", account.defaultModel).orEmpty()
            .takeIf { chosen -> account.models.any { it.id == chosen && it.selectable } }
            ?: account.models.firstOrNull { it.selectable }?.id.orEmpty()
        mutable.update { it.copy(account = account, model = chosen, pairing = false, userCode = "", offline = false) }
        fetchThreads(false)
        runCatching { fetchGems() }.onFailure { report(it) }
        runCatching { fetchPreferences(applyDefaults = true) }.onFailure { report(it) }
        runCatching { fetchBatchJobs(notify = false) }.onFailure { report(it) }
        if (foreground) startBatchPolling()
    }
    fun setForeground(value: Boolean) {
        val returning = value && !foreground
        foreground = value
        if (!value && state.value.streaming) {
            streamJob?.cancel()
            mutable.update { it.copy(streaming = false, status = "アプリに戻ると履歴を確認します。") }
        }
        if (!value && state.value.realtime.active) stopRealtime(save = false)
        if (!value && state.value.lyria.active) stopLyria(save = false)
        if (!value) heartbeatJob?.cancel()
        if (!value) batchPollJob?.cancel()
        if (value && state.value.account != null) startBatchPolling()
        if (returning && state.value.account != null && state.value.selected != null && !state.value.busy) refresh()
    }
    fun pair() {
        if (state.value.pairing) return
        pairingJob = viewModelScope.launch {
            mutable.update { it.copy(pairing = true, userCode = "", notice = null) }
            try {
                // Retry a saved login after a temporary network error without issuing another grant.
                if (session != null) { loadAccount(); return@launch }
                val config = api.get("/api/mobile/v1/config")
                require(config.getInt("api_version") == 1) { "このサーバーの接続方式には未対応です。" }
                val grant = api.post("/api/mobile/v1/device", JSONObject().put("client_id", "official-android")
                    .put("device_name", "${Build.MANUFACTURER} ${Build.MODEL}".take(80)))
                mutable.update { it.copy(userCode = grant.getString("user_code")) }
                val deadline = SystemClock.elapsedRealtime() + grant.getLong("expires_in") * 1000
                var interval = grant.optLong("interval", 5).coerceAtLeast(5)
                while (SystemClock.elapsedRealtime() < deadline) {
                    delay(interval * 1000)
                    if (!foreground) continue
                    try {
                        val reply = api.post("/api/mobile/v1/token", JSONObject().put("client_id", "official-android")
                            .put("device_code", grant.getString("device_code")))
                        val linked = StoredSession(reply.getString("access_token"), System.currentTimeMillis() + reply.getLong("expires_in") * 1000)
                        withContext(Dispatchers.IO) { store.save(linked) }
                        session = linked
                        // The grant is already consumed. Never poll it again if /me or history fails.
                        try { loadAccount() } catch (e: Exception) { report(e) }
                        return@launch
                    } catch (e: ApiException) {
                        when (e.code) {
                            "authorization_pending" -> Unit
                            "slow_down", "rate_limited" -> interval = maxOf(interval + 5, e.retryAfter)
                            "access_denied" -> throw IOException("端末連携が拒否されました。")
                            "expired_token" -> throw IOException("確認コードが期限切れです。新しく連携を開始してください。")
                            else -> if (e.status >= 500) interval = minOf(60, interval * 2) else throw e
                        }
                    } catch (e: IOException) {
                        if (e is ApiException || e.message?.startsWith("端末連携") == true || e.message?.startsWith("確認コード") == true) throw e
                        interval = minOf(60, interval * 2)
                    }
                }
                throw IOException("確認コードが期限切れです。新しく連携を開始してください。")
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) { report(e) }
            finally { mutable.update { it.copy(pairing = false) } }
        }
    }
    fun cancelPairing() { pairingJob?.cancel(); mutable.update { it.copy(pairing = false, userCode = "") } }
    fun dismissNotice() { mutable.update { it.copy(notice = null) } }
    fun notify(message: String) { mutable.update { it.copy(notice = message) } }
    fun draft(text: String) { mutable.update { it.copy(draft = text) } }
    fun quoteMessage(text: String) {
        val quoted = text.trim().lineSequence().filter { it.isNotBlank() }
            .joinToString("\n") { "> $it" }
        if (quoted.isBlank()) return
        mutable.update {
            it.copy(draft = if (it.draft.isBlank()) "$quoted\n\n" else it.draft.trimEnd() + "\n\n$quoted\n\n")
        }
    }
    fun chooseModel(model: String) {
        val info = state.value.account?.models?.firstOrNull { it.id == model && it.selectable } ?: return
        mutable.update { it.copy(model = model,
            enableThinking = it.enableThinking && info.supports("thinking"),
            enableSearch = it.enableSearch && info.supports("search"),
            enableUrlContext = it.enableUrlContext && (info.supports("url_context") || info.id.startsWith("gemini-")),
            enableMaps = it.enableMaps && (info.supports("maps") || info.id.startsWith("gemini-3")),
            enableFileCreation = if (info.mode == "chat" || info.mode == "agent") it.enableFileCreation else false,
            enableSystemPrompt = if (info.mode == "chat" || info.mode == "agent") it.enableSystemPrompt else false,
            canvasMode = if (info.mode == "chat" || info.mode == "agent") it.canvasMode else false,
            codingMode = if (info.mode == "chat" || info.mode == "agent") it.codingMode else false,
            codingTarget = if (info.mode == "chat" || info.mode == "agent") it.codingTarget else null,
            imageMask = if (info.id.startsWith("gpt-image")) it.imageMask else null,
            enablePromptCache = it.enablePromptCache && info.supports("prompt_cache"),
            batchMode = it.batchMode && info.supports("batch"),
            enablePython = it.enablePython && info.supports("python"),
            enableMcp = it.enableMcp && info.supports("mcp")) }
        state.value.account?.let { prefs.edit().putString("model_${it.id}", model).apply() }
    }
    fun toggleThinking() { mutable.update { it.copy(enableThinking = !it.enableThinking) } }
    fun toggleSearch() { mutable.update { it.copy(enableSearch = !it.enableSearch) } }
    fun toggleUrlContext() { mutable.update { it.copy(enableUrlContext = !it.enableUrlContext) } }
    fun toggleMaps() { mutable.update { it.copy(enableMaps = !it.enableMaps) } }
    fun toggleFileCreation() { mutable.update { it.copy(enableFileCreation = !it.enableFileCreation) } }
    fun toggleSystemPrompt() { mutable.update { it.copy(enableSystemPrompt = !it.enableSystemPrompt) } }
    fun togglePromptCache() { mutable.update { it.copy(enablePromptCache = !it.enablePromptCache) } }
    fun toggleBatchMode() { mutable.update { it.copy(batchMode = !it.batchMode) } }
    fun togglePython() { mutable.update { it.copy(enablePython = !it.enablePython) } }
    fun toggleMcp() { mutable.update { it.copy(enableMcp = !it.enableMcp) } }
    fun toggleCanvas() { mutable.update { it.copy(canvasMode = !it.canvasMode, codingMode = false) } }
    fun toggleCoding() { mutable.update { it.copy(codingMode = !it.codingMode, canvasMode = false, codingTarget = null) } }
    fun toggleTemporaryChat() {
        val selected = state.value.selected
        if (selected != null) {
            saveThreadSettings(selected.title, state.value.customInstruction, state.value.includeGlobalInstruction, !selected.isTemporary)
        } else {
            mutable.update { it.copy(newThreadTemporary = !it.newThreadTemporary) }
        }
    }
    fun selectCodingTarget(target: CodingTarget?) { mutable.update { it.copy(codingTarget = target, codingMode = target != null || it.codingMode) } }
    fun setImageMask(reference: String?) { mutable.update { it.copy(imageMask = reference) } }
    fun uploadImageMask(name: String, bytes: ByteArray) {
        if (bytes.isEmpty() || state.value.streaming || state.value.uploading) return
        viewModelScope.launch {
            try {
                val response = api.upload(name, bytes.toRequestBody("image/png".toMediaType()), token())
                setImageMask(response.getString("filename"))
                notify("画像マスクを設定しました。次の画像生成で適用されます。")
            } catch (e: Exception) { report(e) }
        }
    }
    fun generationOption(key: String, value: String) {
        if (state.value.streaming) return
        mutable.update { current -> current.copy(generationValues = current.generationValues +
            (current.model to (current.generationValues[current.model].orEmpty() + (key to value)))) }
    }
    fun removeAttachment(reference: String) { mutable.update { it.copy(attachments = it.attachments.filterNot { a -> a.reference == reference }) } }
    fun search(query: String) {
        mutable.update { it.copy(search = query) }
        navigationJob?.cancel()
        navigationJob = viewModelScope.launch { delay(300); try { fetchThreads(false) } catch (e: Exception) { report(e) } }
    }
    private suspend fun fetchThreads(more: Boolean) {
        val current = state.value
        val page = if (more) current.nextPage ?: return else 1
        val reply = api.get("/api/threads?page=$page&q=${URLEncoder.encode(current.search, "UTF-8")}", token())
        if (current.search != state.value.search) return
        val rows = reply.getJSONArray("threads")
        val items = (0 until rows.length()).map { parseThreadItem(rows.getJSONObject(it)) }
        mutable.update { it.copy(threads = if (more) (it.threads + items).distinctBy { t -> t.id } else items,
            nextPage = if (reply.optBoolean("has_next")) reply.optInt("next_page", page + 1) else null,
            offline = false) }
    }
    fun moreThreads() { navigationJob?.cancel(); navigationJob = viewModelScope.launch { try { fetchThreads(true) } catch (e: Exception) { report(e) } } }
    fun newChat(temporary: Boolean = false) {
        navigationJob?.cancel(); streamJob?.cancel(); failed = null
        heartbeatJob?.cancel()
        pendingParentId = null
        mutable.update { it.copy(selected = null, messages = emptyList(), allMessages = emptyList(), leafId = null,
            editingMessageId = null, jobId = null, streaming = false,
            liveContent = "", liveThought = "", status = "", busy = false, retryAvailable = false,
            cards = emptyList(), hasOlder = false, oldestId = null, customInstruction = "",
            includeGlobalInstruction = true, newThreadTemporary = temporary, tempChatRemainingSeconds = null,
            selectedGem = null, codingTarget = null, imageMask = null) }
    }
    fun openThread(thread: ThreadItem) {
        navigationJob?.cancel(); streamJob?.cancel(); heartbeatJob?.cancel(); failed = null
        pendingParentId = null
        val storedLeaf = prefs.getInt("leaf_${thread.id}", -1).takeIf { it > 0 }
        mutable.update { it.copy(selected = thread, messages = emptyList(), allMessages = emptyList(),
            leafId = storedLeaf, editingMessageId = null, streaming = false, busy = true,
            liveContent = "", liveThought = "", jobId = null, retryAvailable = false,
            cards = emptyList(), hasOlder = false, oldestId = null) }
        navigationJob = viewModelScope.launch {
            try { loadMessages(thread.id); if (foreground && state.value.jobId != null) resume() }
            catch (e: Exception) { report(e) }
            finally { mutable.update { it.copy(busy = false) } }
        }
    }
    private suspend fun loadMessages(id: String, older: Boolean = false) {
        val before = if (older) "&before_id=${state.value.oldestId ?: return}" else ""
        val reply = api.get("/api/threads/$id?limit=50$before", token())
        if (state.value.selected?.id != id) return
        val parsed = parseMessages(reply)
        val all = if (older) (parsed + state.value.allMessages).distinctBy { m -> m.id } else parsed
        val leaf = state.value.leafId?.takeIf { candidate -> all.any { numericId(it) == candidate } }
            ?: all.mapNotNull { numericId(it) }.maxOrNull()
        val path = activeBranchPath(all, leaf)
        mutable.update { it.copy(messages = path, allMessages = all, leafId = leaf,
            hasOlder = reply.optBoolean("has_older_messages"), oldestId = reply.nullableString("oldest_loaded_id"),
            jobId = if (older) it.jobId else reply.optJSONObject("pending_job")?.nullableString("job_id")?.ifBlank { null },
            selected = it.selected?.let { selected -> selected.copy(
                title = reply.optString("title", selected.title),
                model = reply.nullableString("last_model").ifBlank { selected.model },
                isTemporary = reply.optBoolean("is_temporary", selected.isTemporary),
            ) },
            customInstruction = if (older) it.customInstruction else reply.nullableString("custom_instruction"),
            includeGlobalInstruction = if (older) it.includeGlobalInstruction else reply.optBoolean("include_global_instruction", true),
            newThreadTemporary = if (older) it.newThreadTemporary else false,
            tempChatRemainingSeconds = if (older) it.tempChatRemainingSeconds else
                reply.optLong("temp_chat_remaining_seconds").takeIf { value -> !reply.isNull("temp_chat_remaining_seconds") && value >= 0 },
            liveContent = if (older) it.liveContent else "", liveThought = if (older) it.liveThought else "",
            cards = if (older) it.cards else emptyList(),
            selectedGem = if (older) it.selectedGem else {
                val uuid = reply.nullableString("last_gem_uuid")
                if (uuid.isBlank()) null else it.gems.firstOrNull { gem -> gem.uuid == uuid }
            }) }
        if (!older) syncHeartbeat()
    }

    private fun syncHeartbeat() {
        heartbeatJob?.cancel()
        val selected = state.value.selected ?: return
        if (!foreground || !selected.isTemporary) return
        heartbeatJob = viewModelScope.launch {
            while (isActive && foreground && state.value.selected?.id == selected.id && state.value.selected?.isTemporary == true) {
                try {
                    val reply = api.post("/api/temporary_chat/heartbeat",
                        JSONObject().put("thread_id", selected.id).put("active", true), token())
                    mutable.update { it.copy(tempChatRemainingSeconds =
                        reply.optLong("temp_chat_remaining_seconds").takeIf { value -> !reply.isNull("temp_chat_remaining_seconds") && value >= 0 }) }
                } catch (e: CancellationException) { throw e }
                catch (e: Exception) { report(e) }
                delay(15_000)
            }
        }
    }
    fun olderMessages() {
        val id = state.value.selected?.id ?: return
        navigationJob = viewModelScope.launch {
            mutable.update { it.copy(busy = true) }
            try { loadMessages(id, true) } catch (e: Exception) { report(e) }
            finally { mutable.update { it.copy(busy = false) } }
        }
    }
    fun refresh() {
        val thread = state.value.selected
        if (thread != null) openThread(thread)
        else navigationJob = viewModelScope.launch { try { fetchThreads(false) } catch (e: Exception) { report(e) } }
    }
    fun deleteThread(thread: ThreadItem) { viewModelScope.launch {
        try {
            api.delete("/api/threads/${thread.id}", token())
            if (state.value.selected?.id == thread.id) newChat()
            fetchThreads(false)
        } catch (e: Exception) { report(e) }
    } }
    fun toggleBookmark(thread: ThreadItem) { viewModelScope.launch {
        try {
            val reply = api.post("/api/threads/${thread.id}/bookmark", JSONObject(), token())
            val bookmarked = reply.optBoolean("is_bookmarked")
            mutable.update { current -> current.copy(
                threads = current.threads.map { if (it.id == thread.id) it.copy(isBookmarked = bookmarked) else it },
                selected = current.selected?.let { if (it.id == thread.id) it.copy(isBookmarked = bookmarked) else it },
            ) }
            fetchThreads(false)
        } catch (e: Exception) { report(e) }
    } }
    fun saveThreadSettings(title: String, instruction: String, includeGlobal: Boolean, temporary: Boolean) {
        val thread = state.value.selected ?: return
        viewModelScope.launch {
            mutable.update { it.copy(busy = true) }
            try {
                val normalizedTitle = title.trim().ifBlank { "新しいチャット" }
                if (normalizedTitle != thread.title) {
                    api.put("/api/threads/${thread.id}/title", JSONObject().put("title", normalizedTitle), token())
                }
                val reply = api.put("/api/threads/${thread.id}/settings", JSONObject()
                    .put("custom_instruction", instruction)
                    .put("include_global_instruction", includeGlobal)
                    .put("is_temporary", temporary), token())
                mutable.update { current -> current.copy(
                    selected = current.selected?.copy(title = normalizedTitle, isTemporary = temporary),
                    customInstruction = instruction,
                    includeGlobalInstruction = includeGlobal,
                    tempChatRemainingSeconds = reply.optLong("temp_chat_remaining_seconds")
                        .takeIf { value -> !reply.isNull("temp_chat_remaining_seconds") && value >= 0 },
                ) }
                fetchThreads(false)
                syncHeartbeat()
            } catch (e: Exception) { report(e) }
            finally { mutable.update { it.copy(busy = false) } }
        }
    }
    /** Prepares the composer to edit a user message, branching from its parent. */
    fun beginEdit(message: ChatMessage) {
        if (message.role != "user") return
        pendingParentId = message.parentId
        mutable.update { it.copy(
            editingMessageId = message.id,
            draft = message.content,
            attachments = message.files.map { reference -> Attachment(reference.substringAfterLast('/'), reference) },
        ) }
    }

    /** Re-sends the user message that produced an assistant reply, creating a sibling branch. */
    fun regenerate(message: ChatMessage) {
        if (message.role != "assistant" || state.value.streaming || state.value.busy) return
        val parent = message.parentId?.let { pid -> state.value.allMessages.firstOrNull { numericId(it) == pid } } ?: return
        beginEdit(parent)
        send()
    }

    fun cancelEdit() {
        pendingParentId = null
        mutable.update { it.copy(editingMessageId = null, draft = "", attachments = emptyList()) }
    }

    /** Switches the active path to the branch that contains [targetMessageId]. */
    fun switchBranch(targetMessageId: Int) {
        val all = state.value.allMessages
        val leaf = latestLeafId(all, targetMessageId)
        mutable.update { it.copy(leafId = leaf, messages = activeBranchPath(all, leaf)) }
        state.value.selected?.let { prefs.edit().putInt("leaf_${it.id}", leaf).apply() }
    }

    fun switchBranchByIndex(siblings: List<ChatMessage>, index: Int) {
        siblings.getOrNull(index)?.let { numericId(it)?.let(::switchBranch) }
    }

    fun send() {
        val current = state.value
        if (current.streaming || current.busy || current.uploading || (current.draft.isBlank() && current.attachments.isEmpty())) return
        if (current.model.isBlank()) { mutable.update { it.copy(notice = "モデルを選択してください。") }; return }
        val info = current.account?.models?.firstOrNull { it.id == current.model && it.selectable } ?: return
        val generation = try { generationOptionsPayload(info, current.generationValues[current.model].orEmpty()) }
            catch (e: IllegalArgumentException) { notify(e.message ?: "生成設定を確認してください。"); return }
        val body = JSONObject().put("model", current.model).put("message", current.draft)
            .put("client_request_id", UUID.randomUUID().toString()).put("image_urls", JSONArray(current.attachments.map { it.reference }))
            .put("enable_thinking", current.enableThinking).put("enable_search", current.enableSearch)
            .put("enable_url_context", current.enableUrlContext).put("enable_maps", current.enableMaps)
            .put("enable_file_creation", current.enableFileCreation).put("enable_system_prompt", current.enableSystemPrompt)
            .put("enable_prompt_caching", current.enablePromptCache)
            .put("batch_mode", current.batchMode).put("enable_python", current.enablePython)
            .put("enable_mcp", current.enableMcp)
            .put("coding_mode", current.codingMode)
            .put("canvas_mode", current.canvasMode)
            .put("temporary_chat", current.selected?.isTemporary ?: current.newThreadTemporary)
        current.imageMask?.let { body.put("image_mask", it) }
        if (current.codingMode) {
            val target = current.codingTarget
            if (target != null) {
                body.put("coding_target", JSONObject().put("id", target.id).put("source", "history")
                    .put("code", target.code).put("language", target.language).put("message_id", target.messageId).put("explicit", true))
                body.put("coding_candidates", JSONArray().put(JSONObject().put("id", target.id).put("source", "history")
                    .put("code", target.code).put("language", target.language).put("explicit", true)))
            } else {
                val blocks = Regex("```([^\\n`]*)\\n([\\s\\S]*?)```").findAll(current.draft).mapIndexed { index, match ->
                    val language = match.groupValues[1].trim().ifBlank { "text" }.take(40)
                    val code = match.groupValues[2]
                    JSONObject().put("id", "prompt-$index").put("source", "prompt")
                        .put("prompt_index", index).put("code", JSONObject.NULL)
                        .put("language", language).put("explicit", true) to (language to code)
                }.toList()
                if (blocks.isEmpty()) {
                    notify("Coding Modeでは入力にコードブロックを指定してください。")
                    return
                }
                val candidates = JSONArray().apply { blocks.forEach { put(it.first) } }
                val (language, _) = blocks.first().second
                body.put("coding_target", JSONObject().put("id", "prompt-0").put("source", "prompt")
                    .put("prompt_index", 0).put("language", language).put("explicit", true))
                body.put("coding_candidates", candidates)
            }
        }
        generation.keys().forEach { key -> body.put(key, generation.get(key)) }
        current.selectedGem?.let { body.put("gem_uuid", it.uuid) }
        if (current.editingMessageId != null) {
            // Branch from the edited message's parent; send null explicitly for the first message.
            body.put("parent_id", pendingParentId ?: JSONObject.NULL)
            body.put("parent_id_explicit", true)
        } else {
            // Keep normal sends on the currently selected branch.
            current.leafId?.let { body.put("parent_id", it) }
        }
        current.selected?.let { body.put("thread_id", it.id) }
        val submission = Submission(body, current.attachments)
        failed = submission
        pendingParentId = null
        mutable.update { it.copy(draft = "", attachments = emptyList(), editingMessageId = null) }
        submit(submission)
    }
    fun retry() { failed?.let { submit(it) } }
    private fun submit(submission: Submission) {
        streamJob?.cancel()
        streamJob = viewModelScope.launch {
            val owner = currentCoroutineContext().job
            mutable.update { it.copy(streaming = true, retryAvailable = false, status = "送信中…", liveContent = "", liveThought = "", cards = emptyList(), mcpDecision = null) }
            var id = submission.body.nullableString("thread_id")
            try {
                if (id.isBlank()) {
                    val created = api.post("/api/threads", JSONObject().put("is_temporary", state.value.newThreadTemporary), token())
                    id = created.get("id").toString()
                    submission.body.put("thread_id", id)
                    mutable.update { it.copy(selected = ThreadItem(id, created.optString("title", "新しいチャット"), it.model,
                        isTemporary = created.optBoolean("is_temporary")), newThreadTemporary = false) }
                    syncHeartbeat()
                }
                val userId = "local-${submission.body.getString("client_request_id")}"
                mutable.update { it.copy(messages = it.messages.filterNot { m -> m.id == userId } + ChatMessage(userId, "user",
                    submission.body.getString("message"), files = submission.files.map { a -> a.reference })) }
                try { api.stream("/chat_stream", submission.body, token()) { event -> if (streamJob === owner) acceptEvent(id, event) } }
                catch (e: ApiException) {
                    if (e.code != "request_already_accepted") throw e
                    try { api.stream("/chat_stream_resume", JSONObject().put("thread_id", id).put("job_id", e.payload.getString("job_id")), token()) { event -> if (streamJob === owner) acceptEvent(id, event) } }
                    catch (resumeError: ApiException) { if (resumeError.status != 404) throw resumeError }
                }
                failed = null
                loadMessages(id)
                fetchThreads(false)
                if (submission.body.optBoolean("batch_mode")) {
                    fetchBatchJobs(notify = false)
                    startBatchPolling()
                }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                report(e)
                val retryable = e !is ApiException || e.status >= 500 || e.status in listOf(409, 425, 429)
                if (session != null) mutable.update { it.copy(retryAvailable = retryable,
                    draft = if (!retryable) submission.body.optString("message") else it.draft,
                    attachments = if (!retryable) submission.files else it.attachments) }
                if (id.isNotBlank() && session != null) runCatching { loadMessages(id) }
            } finally { if (streamJob === owner) mutable.update { it.copy(streaming = false, status = "") } }
        }
    }
    private fun acceptEvent(threadId: String, event: JSONObject) {
        if (state.value.selected?.id != threadId) return
        when (event.optString("type")) {
            "python" -> {
                val payload = event.optJSONObject("content") ?: return
                mutable.update { it.copy(cards = upsertPythonCard(it.cards, payload), status = "ツールを実行しています…") }
                return
            }
            "search_status" -> {
                val value = event.opt("content")?.toString().orEmpty()
                mutable.update { it.copy(cards = upsertSearchCard(it.cards, value), status = "Webを検索しています…") }
                return
            }
            "mcp", "mcp_decision_request", "mcp_decision_resolved", "coding_diff" -> {
                val type = event.optString("type")
                val payload = event.optJSONObject("content")
                mutable.update { current -> current.copy(
                    cards = upsertToolCard(current.cards, type, event.opt("content")),
                    mcpDecision = when (type) {
                        "mcp_decision_request" -> payload?.let { McpDecision(
                            id = it.optString("id"), jobId = current.jobId.orEmpty(),
                            serverName = it.optString("server_name", "MCP"),
                            toolName = it.optString("tool_name"), argsPreview = it.optString("args_preview"),
                        ) }
                        "mcp_decision_resolved" -> null
                        else -> current.mcpDecision
                    },
                    status = "ツールを実行しています…",
                ) }
                return
            }
        }
        val content = event.opt("content")?.toString().orEmpty()
        mutable.update { when (event.optString("type")) {
            "job_id" -> it.copy(jobId = content, status = "応答を待っています…")
            "status" -> it.copy(status = content)
            "content" -> it.copy(liveContent = it.liveContent + content, status = "受信中…")
            "thought" -> it.copy(liveThought = it.liveThought + content)
            "error" -> it.copy(notice = content.take(500))
            else -> it
        } }
    }
    fun resume() {
        if (state.value.streaming) return
        val id = state.value.selected?.id ?: return
        val jobId = state.value.jobId ?: return
        streamJob = viewModelScope.launch {
            val owner = currentCoroutineContext().job
            mutable.update { it.copy(streaming = true, liveContent = "", liveThought = "", status = "再接続中…") }
            try {
                api.stream("/chat_stream_resume", JSONObject().put("thread_id", id).put("job_id", jobId), token()) { event -> if (streamJob === owner) acceptEvent(id, event) }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) { if (e !is ApiException || e.status != 404) report(e) }
            finally { if (streamJob === owner) mutable.update { it.copy(streaming = false, status = "") } }
            runCatching { loadMessages(id) }.onFailure { report(it) }
        }
    }
    fun stop() { viewModelScope.launch {
        val id = state.value.selected?.id ?: return@launch
        try {
            api.post("/api/stop_chat", JSONObject().put("thread_id", id).apply { state.value.jobId?.let { put("job_id", it) } }, token())
            streamJob?.cancelAndJoin()
            mutable.update { it.copy(streaming = false, status = "停止を要求しました。") }
            delay(1000); loadMessages(id)
        } catch (e: Exception) { report(e) }
    } }

    fun resolveMcpDecision(allow: Boolean) { viewModelScope.launch {
        val decision = state.value.mcpDecision ?: return@launch
        if (decision.jobId.isBlank()) return@launch
        mutable.update { it.copy(mcpDecision = null) }
        try {
            api.post("/api/mcp/chat/${URLEncoder.encode(decision.jobId, "UTF-8")}/decision",
                JSONObject().put("decision", if (allow) "allow" else "deny").put("id", decision.id), token())
        } catch (e: Exception) { report(e) }
    } }

    private suspend fun fetchBatchJobs(notify: Boolean) {
        val previous = state.value.batchJobs.associateBy { it.id }
        if (notify) runCatching { api.get("/api/gemini/batch/status", token()) }
        val jobs = parseBatchJobs(api.get("/api/batch/jobs", token()))
        if (notify) jobs.filter { !it.active && previous[it.id]?.active == true }.forEach { job ->
            notifyBatchCompletion(getApplication<Application>(), job)
            mutable.update { it.copy(notice = "Batch処理「${job.threadTitle}」: ${batchStateLabel(job)}") }
        }
        mutable.update { it.copy(batchJobs = jobs, batchBusy = false) }
    }

    fun refreshBatchJobs() {
        viewModelScope.launch {
            mutable.update { it.copy(batchBusy = true) }
            try { fetchBatchJobs(notify = true) } catch (e: Exception) { report(e) }
            finally { mutable.update { it.copy(batchBusy = false) } }
        }
    }

    private fun startBatchPolling() {
        if (batchPollJob?.isActive == true) return
        batchPollJob = viewModelScope.launch {
            while (foreground && session != null) {
                if (state.value.batchJobs.any { it.active }) runCatching { fetchBatchJobs(notify = true) }
                delay(30_000)
            }
        }
    }

    fun cancelBatchJob(job: BatchJob) { viewModelScope.launch {
        try { api.post("/api/batch/jobs/${job.id}/cancel", JSONObject(), token()); fetchBatchJobs(notify = false) }
        catch (e: Exception) { report(e) }
    } }

    fun deleteBatchJob(job: BatchJob) { viewModelScope.launch {
        try { api.delete("/api/batch/jobs/${job.id}", token()); fetchBatchJobs(notify = false) }
        catch (e: Exception) { report(e) }
    } }

    fun openThreadId(id: String) {
        val item = state.value.threads.firstOrNull { it.id == id } ?: ThreadItem(id, "Batchチャット", "")
        openThread(item)
    }

    // --- Native realtime audio sessions ---

    fun startRealtime(modelId: String, voice: String = "alloy") {
        if (state.value.realtime.active || modelId.isBlank()) return
        viewModelScope.launch {
            try {
                val started = api.post("/api/realtime/start", JSONObject().put("model", modelId).put("voice", voice), token())
                val sessionId = started.getString("session_id")
                val rateOut = started.optInt("rate_out", 24000).coerceIn(8000, 48000)
                realtimeTrack = createAudioTrack(rateOut, stereo = false)
                mutable.update { it.copy(realtime = RealtimeState(true, modelId, sessionId, "接続中…")) }
                realtimeCaptureJob = viewModelScope.launch(Dispatchers.IO) {
                    try { captureRealtimeAudio(sessionId, started.optInt("rate_in", rateOut).coerceIn(8000, 48000)) }
                    catch (e: CancellationException) { throw e }
                    catch (e: Exception) { mutable.update { it.copy(realtime = it.realtime.copy(error = e.message ?: "マイク入力を開始できません。")) } }
                }
                realtimeStreamJob = viewModelScope.launch(Dispatchers.IO) {
                    try {
                        api.streamSse("/api/realtime/stream?session_id=${URLEncoder.encode(sessionId, "UTF-8")}", token()) { event ->
                            handleRealtimeEvent(event, rateOut)
                        }
                    } catch (e: CancellationException) { throw e }
                    catch (e: Exception) { mutable.update { it.copy(realtime = it.realtime.copy(error = e.message ?: "Realtime接続が終了しました。")) } }
                    finally {
                        if (state.value.realtime.sessionId == sessionId) mutable.update { it.copy(realtime = it.realtime.copy(active = false, status = "終了")) }
                    }
                }
            } catch (e: Exception) { report(e) }
        }
    }

    private suspend fun captureRealtimeAudio(sessionId: String, rate: Int) {
        val min = AudioRecord.getMinBufferSize(rate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        if (min <= 0) throw IOException("マイクを初期化できません。")
        val recorder = try {
            AudioRecord(MediaRecorder.AudioSource.MIC, rate, AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT, (min * 2).coerceAtLeast(4096))
        } catch (e: SecurityException) { throw IOException("マイクの権限が必要です。", e) }
        try {
            recorder.startRecording()
            val buffer = ByteArray((rate / 5).coerceAtLeast(4096))
            while (currentCoroutineContext().isActive && state.value.realtime.active && state.value.realtime.sessionId == sessionId) {
                val read = recorder.read(buffer, 0, buffer.size)
                if (read > 0) api.postBytes("/api/realtime/audio?session_id=${URLEncoder.encode(sessionId, "UTF-8")}",
                    buffer.copyOf(read), "audio/pcm", token())
            }
        } finally { runCatching { recorder.stop() }; recorder.release() }
    }

    private fun createAudioTrack(rate: Int, stereo: Boolean): AudioTrack {
        val channels = if (stereo) AudioFormat.CHANNEL_OUT_STEREO else AudioFormat.CHANNEL_OUT_MONO
        val min = AudioTrack.getMinBufferSize(rate, channels, AudioFormat.ENCODING_PCM_16BIT).coerceAtLeast(4096)
        return AudioTrack.Builder().setAudioAttributes(AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_MEDIA).setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
            .setAudioFormat(AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                .setSampleRate(rate).setChannelMask(channels).build())
            .setBufferSizeInBytes(min * 2).setTransferMode(AudioTrack.MODE_STREAM).build().also { it.play() }
    }

    private fun handleRealtimeEvent(event: JSONObject, rateOut: Int) {
        when (event.optString("type")) {
            "status" -> mutable.update { it.copy(realtime = it.realtime.copy(status = event.optString("status"))) }
            "audio" -> {
                val encoded = event.nullableString("data")
                val bytes = runCatching { android.util.Base64.decode(encoded, android.util.Base64.DEFAULT) }.getOrNull() ?: ByteArray(0)
                if (bytes.isNotEmpty()) {
                    realtimeTrack?.write(bytes, 0, bytes.size, AudioTrack.WRITE_NON_BLOCKING)
                    mutable.update { it.copy(realtime = it.realtime.copy(audioBytes = it.realtime.audioBytes + bytes.size)) }
                }
            }
            "transcript" -> {
                val role = event.optString("role")
                val delta = event.nullableString("delta")
                mutable.update { current -> current.copy(realtime = current.realtime.copy(
                    userText = if (role == "user" && event.optBoolean("cumulative")) delta else if (role == "user") current.realtime.userText + delta else current.realtime.userText,
                    assistantText = if (role == "assistant") current.realtime.assistantText + delta else current.realtime.assistantText,
                    thoughtText = if (role == "thought") current.realtime.thoughtText + delta else current.realtime.thoughtText,
                )) }
            }
            "error" -> mutable.update { it.copy(realtime = it.realtime.copy(error = event.nullableString("message"))) }
            "final" -> mutable.update { it.copy(realtime = it.realtime.copy(active = false, status = "終了")) }
        }
    }

    fun commitRealtime() {
        val sid = state.value.realtime.sessionId.takeIf { it.isNotBlank() } ?: return
        viewModelScope.launch { runCatching { api.post("/api/realtime/commit", JSONObject().put("session_id", sid), token()) }
            .onFailure { report(it) } }
    }

    fun stopRealtime(save: Boolean) {
        val current = state.value.realtime
        if (!current.active && current.sessionId.isBlank()) return
        viewModelScope.launch {
            realtimeCaptureJob?.cancelAndJoin(); realtimeStreamJob?.cancelAndJoin()
            val sid = current.sessionId
            try {
                if (sid.isNotBlank()) {
                    val path = if (save) "/api/realtime/save" else "/api/realtime/cancel"
                    api.post(path, JSONObject().put("session_id", sid).apply { state.value.selected?.id?.let { put("thread_id", it) } }, token())
                }
            } catch (e: Exception) { report(e) }
            realtimeTrack?.let { track -> runCatching { track.stop(); track.release() } }; realtimeTrack = null
            mutable.update { it.copy(realtime = RealtimeState()) }
        }
    }

    // --- Native Lyria RealTime studio ---

    fun startLyria(prompt: String) {
        if (state.value.lyria.active || prompt.isBlank()) return
        viewModelScope.launch {
            try {
                val payload = JSONObject().put("weighted_prompts", JSONArray().put(JSONObject().put("text", prompt.trim().take(4000)).put("weight", 1.0)))
                val started = api.post("/api/gemini/music/start", payload, token())
                val sid = started.getString("session_id")
                lyriaTrack = createAudioTrack(48000, stereo = true)
                mutable.update { it.copy(lyria = LyriaState(true, sid, "接続中…", prompt.trim().take(4000))) }
                lyriaStreamJob = viewModelScope.launch(Dispatchers.IO) {
                    try {
                        api.streamSse("/api/gemini/music/stream?session_id=${URLEncoder.encode(sid, "UTF-8")}", token()) { event ->
                            handleLyriaEvent(event)
                        }
                    } catch (e: CancellationException) { throw e }
                    catch (e: Exception) { mutable.update { it.copy(lyria = it.lyria.copy(error = e.message ?: "Lyria接続が終了しました。")) } }
                    finally { if (state.value.lyria.sessionId == sid) mutable.update { it.copy(lyria = it.lyria.copy(active = false, status = "終了")) } }
                }
            } catch (e: Exception) { report(e) }
        }
    }

    private fun handleLyriaEvent(event: JSONObject) {
        val encoded = event.nullableString("audio").ifBlank { event.nullableString("snapshot") }
        if (encoded.isNotBlank()) {
            val bytes = runCatching { android.util.Base64.decode(encoded, android.util.Base64.DEFAULT) }.getOrNull() ?: ByteArray(0)
            if (bytes.isNotEmpty()) {
                lyriaTrack?.write(bytes, 0, bytes.size, AudioTrack.WRITE_NON_BLOCKING)
                mutable.update { it.copy(lyria = it.lyria.copy(audioBytes = it.lyria.audioBytes + bytes.size, status = event.optString("status").ifBlank { "生成中…" })) }
            }
        }
        event.nullableString("error").takeIf { it.isNotBlank() }?.let { error -> mutable.update { it.copy(lyria = it.lyria.copy(error = error)) } }
    }

    fun lyriaControl(action: String) {
        val sid = state.value.lyria.sessionId.takeIf { it.isNotBlank() } ?: return
        viewModelScope.launch { runCatching { api.post("/api/gemini/music/command", JSONObject().put("session_id", sid).put("type", "control").put("action", action), token()) }
            .onFailure { report(it) } }
    }

    fun stopLyria(save: Boolean) {
        val current = state.value.lyria
        if (!current.active && current.sessionId.isBlank()) return
        viewModelScope.launch {
            lyriaStreamJob?.cancelAndJoin()
            try {
                if (current.sessionId.isNotBlank()) api.post(if (save) "/api/gemini/music/save" else "/api/gemini/music/cancel",
                    JSONObject().put("session_id", current.sessionId).apply { state.value.selected?.id?.let { put("thread_id", it) } }, token())
            } catch (e: Exception) { report(e) }
            lyriaTrack?.let { track -> runCatching { track.stop(); track.release() } }; lyriaTrack = null
            mutable.update { it.copy(lyria = LyriaState()) }
        }
    }

    fun logout() { viewModelScope.launch {
        try {
            api.post("/api/mobile/v1/revoke", JSONObject(), token())
            clearSession()
        } catch (e: Exception) { report(e) }
    } }
    private suspend fun clearSession() {
        val caller = currentCoroutineContext().job
        listOf(pairingJob, navigationJob, streamJob, uploadJob, heartbeatJob, libraryJob, batchPollJob,
            realtimeStreamJob, realtimeCaptureJob, lyriaStreamJob).forEach { if (it !== caller) it?.cancel() }
        realtimeTrack?.let { track -> runCatching { track.stop(); track.release() } }; realtimeTrack = null
        lyriaTrack?.let { track -> runCatching { track.stop(); track.release() } }; lyriaTrack = null
        withContext(NonCancellable + Dispatchers.IO) { store.clear() }
        session = null; failed = null
        mutable.value = ChatState(starting = false)
    }
    private suspend fun report(error: Throwable) {
        if (error is CancellationException) throw error
        if (error is ApiException && error.status == 401) clearSession()
        val offline = error is java.net.ConnectException || error is java.net.UnknownHostException ||
            error is java.net.SocketTimeoutException || error is java.net.SocketException
        mutable.update { it.copy(
            notice = error.message?.take(500) ?: "通信に失敗しました。再試行してください。",
            offline = it.offline || offline) }
    }
    /** Clears the offline banner and retries the last account or history load. */
    fun reconnect() {
        mutable.update { it.copy(offline = false) }
        viewModelScope.launch {
            try {
                if (session == null) pair() else {
                    loadAccount()
                    state.value.selected?.let { loadMessages(it.id) }
                }
            } catch (e: Exception) { report(e) }
        }
    }
    /** Loads same-origin attachment bytes for preview; returns null when unavailable. */
    suspend fun loadAttachmentBytes(reference: String, thumbnail: Boolean, limit: Long = 8L * 1024 * 1024): ByteArray? {
        val active = session?.token ?: return null
        return withContext(Dispatchers.IO) {
            runCatching { api.loadFileBytes(reference, active, thumbnail, limit) }.getOrNull()
        }
    }
    fun upload(uris: List<Uri>) {
        if (uris.isEmpty() || state.value.uploading) return
        if (uris.size + state.value.attachments.size > 30) { mutable.update { it.copy(notice = "添付は30件までです。") }; return }
        uploadJob = viewModelScope.launch {
            mutable.update { it.copy(uploading = true, uploadSent = 0, uploadTotal = 0, uploadName = "") }
            try {
                for (uri in uris) {
                    val resolver = getApplication<Application>().contentResolver
                    val local = withContext(Dispatchers.IO) { queryLocalAttachment(resolver, uri) }
                    mutable.update { it.copy(uploadName = local.name, uploadSent = 0,
                        uploadTotal = if (local.size > 0) local.size else 0) }
                    val source = withContext(Dispatchers.IO) { prepareUpload(resolver, uri, local) }
                    mutable.update { it.copy(uploadName = source.name, uploadSent = 0,
                        uploadTotal = if (source.size > 0) source.size else 0) }
                    val uploaded = if (source.size > CHUNK_UPLOAD_THRESHOLD_BYTES) uploadInChunks(source)
                        else uploadWhole(source)
                    mutable.update { it.copy(attachments = it.attachments + Attachment(source.name, uploaded, source.mime)) }
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) { report(e) }
            finally { mutable.update { it.copy(uploading = false, uploadSent = 0, uploadTotal = 0, uploadName = "") } }
        }
    }

    fun cancelUpload() {
        if (!state.value.uploading) return
        uploadJob?.cancel()
        mutable.update { it.copy(notice = "アップロードをキャンセルしました。") }
    }

    /** Persists the local image-compression settings used before image uploads. */
    fun saveCompressionSettings(settings: CompressionSettings) {
        prefs.edit()
            .putBoolean("compression_enabled", settings.enabled)
            .putFloat("compression_max_size_mb", settings.maxSizeMB)
            .putInt("compression_max_dim", settings.maxDimension)
            .putString("compression_output_type", settings.outputType)
            .putBoolean("compression_format_only", settings.formatOnly)
            .apply()
        mutable.update { it.copy(compression = settings) }
    }

    private data class LocalAttachment(val name: String, val size: Long, val mime: String)

    private class LocalUpload(val name: String, val size: Long, val mime: String, val opener: () -> java.io.InputStream)

    private fun queryLocalAttachment(resolver: android.content.ContentResolver, uri: Uri): LocalAttachment {
        var name = "attachment"
        var size = -1L
        resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE), null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (nameIndex >= 0) cursor.getString(nameIndex)?.takeIf { it.isNotBlank() }?.let { name = it }
                val sizeIndex = cursor.getColumnIndex(OpenableColumns.SIZE)
                if (sizeIndex >= 0 && !cursor.isNull(sizeIndex)) size = cursor.getLong(sizeIndex)
            }
        }
        val mime = resolver.getType(uri).orEmpty()
        if (!name.contains('.') && mime.isNotBlank()) {
            val extension = extensionForMime(mime)
            if (extension.isNotBlank()) name = "$name.$extension"
        }
        return LocalAttachment(name, size, mime)
    }

    private fun prepareUpload(resolver: android.content.ContentResolver, uri: Uri, local: LocalAttachment): LocalUpload {
        val compressed = runCatching {
            compressImage(getApplication<Application>(), uri, local.name, local.mime, state.value.compression)
        }.getOrNull()
        if (compressed != null) {
            return LocalUpload(compressed.name, compressed.file.length(), compressed.mime) { compressed.file.inputStream() }
        }
        return LocalUpload(local.name, local.size, local.mime) {
            resolver.openInputStream(uri) ?: throw IOException("添付を開けません。")
        }
    }

    private suspend fun uploadWhole(local: LocalUpload): String {
        val known = local.size > 0
        val body = object : RequestBody() {
            override fun contentType() = local.mime.takeIf { it.isNotBlank() }?.toMediaTypeOrNull()
            override fun contentLength() = if (known) local.size else -1L
            override fun writeTo(sink: BufferedSink) {
                val input = local.opener()
                input.use {
                    val bytes = ByteArray(64 * 1024)
                    var sent = 0L
                    while (true) {
                        val count = it.read(bytes)
                        if (count < 0) break
                        sent += count
                        if (sent > MAX_SINGLE_UPLOAD_BYTES) throw IOException("添付は1ファイル64MiBまでです。")
                        sink.write(bytes, 0, count)
                        mutable.update { state -> state.copy(uploadSent = sent, uploadTotal = if (known) local.size else sent) }
                    }
                }
            }
        }
        return api.upload(local.name, body, token()).getString("filename")
    }

    private suspend fun uploadInChunks(local: LocalUpload): String {
        val init = api.uploadInit(local.name, local.size, token())
        val uploadId = init.getString("upload_id")
        val chunkSize = init.optLong("chunk_size", CHUNK_UPLOAD_THRESHOLD_BYTES).toInt().coerceAtLeast(1)
        val totalChunks = ((local.size + chunkSize - 1) / chunkSize).toInt()
        val input = local.opener()
        input.use {
            val buffer = ByteArray(chunkSize)
            var index = 0
            var sent = 0L
            while (index < totalChunks) {
                if (!currentCoroutineContext().isActive) throw CancellationException()
                val expected = minOf(chunkSize.toLong(), local.size - sent).toInt()
                var read = 0
                while (read < expected) {
                    val count = it.read(buffer, read, expected - read)
                    if (count < 0) break
                    read += count
                }
                if (read != expected) throw IOException("添付の読み込みに失敗しました。")
                api.uploadChunk(uploadId, index, totalChunks, buffer.copyOf(read), token())
                sent += read
                index++
                mutable.update { state -> state.copy(uploadSent = sent, uploadTotal = local.size) }
            }
        }
        return api.uploadComplete(uploadId, token()).getString("filename")
    }
    suspend fun downloadAttachment(reference: String): Pair<File, String> {
        val directory = File(getApplication<Application>().cacheDir, "shared").apply { mkdirs() }
        val suffix = reference.substringBefore('?').substringAfterLast('.', "bin").take(8).filter { it.isLetterOrDigit() }.ifBlank { "bin" }
        val target = File(directory, "${UUID.randomUUID()}.$suffix")
        val mime = withContext(Dispatchers.IO) { api.download(reference, target, token()) }
        return target to mime
    }

    fun openFile(reference: String, onReady: (File, String) -> Unit) { viewModelScope.launch {
        try {
            val (file, mime) = downloadAttachment(reference)
            onReady(file, mime)
        } catch (e: Exception) { report(e) }
    } }

    // --- File library ---

    private fun libraryPath(offset: Int): String {
        val current = state.value
        val query = URLEncoder.encode(current.libraryQuery, "UTF-8")
        val favorites = if (current.libraryFavoritesOnly) "&favorites_only=1" else ""
        return "/api/files?limit=40&offset=$offset&sort=newest&q=$query$favorites"
    }

    private suspend fun fetchLibrary(append: Boolean) {
        val offset = if (append) state.value.library.size else 0
        val reply = api.get(libraryPath(offset), token())
        val files = parseLibraryFiles(reply)
        mutable.update {
            it.copy(
                library = if (append) (it.library + files).distinctBy { file -> file.filepath } else files,
                libraryHasMore = reply.optBoolean("has_more"),
                libraryTotal = reply.optInt("total"),
                libraryBusy = false,
            )
        }
    }

    fun refreshLibrary() {
        libraryJob?.cancel()
        libraryJob = viewModelScope.launch {
            mutable.update { it.copy(libraryBusy = true) }
            try { fetchLibrary(false) } catch (e: Exception) { report(e) } finally { mutable.update { it.copy(libraryBusy = false) } }
        }
    }

    fun librarySearch(query: String) {
        mutable.update { it.copy(libraryQuery = query) }
        libraryJob?.cancel()
        libraryJob = viewModelScope.launch {
            delay(300)
            mutable.update { it.copy(libraryBusy = true) }
            try { fetchLibrary(false) } catch (e: Exception) { report(e) } finally { mutable.update { it.copy(libraryBusy = false) } }
        }
    }

    fun setLibraryFavoritesOnly(value: Boolean) {
        if (state.value.libraryFavoritesOnly == value) return
        mutable.update { it.copy(libraryFavoritesOnly = value) }
        refreshLibrary()
    }

    fun moreLibrary() {
        if (!state.value.libraryHasMore || state.value.libraryBusy) return
        libraryJob = viewModelScope.launch {
            mutable.update { it.copy(libraryBusy = true) }
            try { fetchLibrary(true) } catch (e: Exception) { report(e) } finally { mutable.update { it.copy(libraryBusy = false) } }
        }
    }

    fun toggleLibraryFavorite(file: LibraryFile) { viewModelScope.launch {
        try {
            val reply = api.post("/api/files/favorite", JSONObject().put("filepath", file.filepath), token())
            val favorite = reply.optBoolean("is_favorite")
            mutable.update { current -> current.copy(library = current.library.map {
                if (it.filepath == file.filepath) it.copy(isFavorite = favorite) else it
            }) }
        } catch (e: Exception) { report(e) }
    } }

    fun renameLibraryFile(file: LibraryFile, name: String) { viewModelScope.launch {
        try {
            val reply = api.post("/api/files/rename", JSONObject().put("filepath", file.filepath).put("filename", name), token())
            val display = reply.optString("filename", name)
            mutable.update { current -> current.copy(library = current.library.map {
                if (it.filepath == file.filepath) it.copy(displayName = display) else it
            }) }
        } catch (e: Exception) { report(e) }
    } }

    fun deleteLibraryFile(file: LibraryFile) { viewModelScope.launch {
        try {
            api.post("/api/files/delete", JSONObject().put("filenames", JSONArray().put(file.filepath)), token())
            mutable.update { current -> current.copy(library = current.library.filterNot { it.filepath == file.filepath }) }
        } catch (e: Exception) { report(e) }
    } }

    /** Reuses a library file as a composer attachment without re-uploading it. */
    fun reuseLibraryFile(file: LibraryFile) {
        if (state.value.attachments.any { it.reference == file.filepath }) {
            mutable.update { it.copy(notice = "この添付はすでに追加されています。") }
            return
        }
        if (state.value.attachments.size >= 30) {
            mutable.update { it.copy(notice = "添付は30件までです。") }
            return
        }
        mutable.update { it.copy(attachments = it.attachments + Attachment(file.displayName, file.filepath, "")) }
    }

    // --- Gems ---

    private suspend fun fetchGems() {
        val gems = parseGems(api.getArray("/api/gems", token()))
        mutable.update { current ->
            current.copy(gems = gems, selectedGem = current.selectedGem?.let { selected -> gems.firstOrNull { it.uuid == selected.uuid } })
        }
    }

    fun loadGems() {
        viewModelScope.launch {
            mutable.update { it.copy(gemsBusy = true) }
            try { fetchGems() } catch (e: Exception) { report(e) } finally { mutable.update { it.copy(gemsBusy = false) } }
        }
    }

    fun saveGem(uuid: String?, name: String, description: String, instruction: String, defaultModel: String, fixedPrompts: List<FixedPrompt>, onDone: (Boolean) -> Unit) {
        viewModelScope.launch {
            mutable.update { it.copy(gemsBusy = true) }
            try {
                val payload = JSONObject().put("name", name).put("description", description)
                    .put("instruction", instruction).put("default_model", defaultModel)
                    .put("fixed_prompts", JSONArray().apply {
                        fixedPrompts.forEach { put(JSONObject().put("name", it.name.trim()).put("content", it.content.trim())) }
                    })
                if (uuid.isNullOrBlank()) api.post("/api/gems", payload, token())
                else api.put("/api/gems/$uuid", payload, token())
                fetchGems()
                mutable.update { current -> current.copy(selectedGem = current.selectedGem?.let { selected -> current.gems.firstOrNull { it.uuid == selected.uuid } }) }
                onDone(true)
            } catch (e: Exception) { report(e); onDone(false) }
            finally { mutable.update { it.copy(gemsBusy = false) } }
        }
    }

    fun deleteGem(gem: Gem) { viewModelScope.launch {
        try {
            api.delete("/api/gems/${gem.uuid}", token())
            mutable.update { current -> current.copy(
                gems = current.gems.filterNot { it.uuid == gem.uuid },
                selectedGem = current.selectedGem?.takeIf { it.uuid != gem.uuid },
            ) }
        } catch (e: Exception) { report(e) }
    } }

    fun chooseGem(gem: Gem?) {
        if (state.value.streaming) return
        mutable.update { it.copy(selectedGem = gem) }
        gem?.defaultModel?.takeIf { it.isNotBlank() }?.let { chooseModel(it) }
    }

    /** Applies a Gem chosen from the `@` candidate list and removes its mention. */
    fun applyGemMention(gem: Gem, query: String) {
        if (state.value.streaming) return
        chooseGem(gem)
        mutable.update { it.copy(draft = replaceGemMention(it.draft, query)) }
    }

    // --- General preferences and this device's session ---

    private suspend fun fetchPreferences(applyDefaults: Boolean = false) {
        val preferences = parsePreferences(api.get("/api/mobile/v1/preferences", token()))
        if (!applyDefaults) {
            mutable.update { it.copy(preferences = preferences) }
            return
        }
        val useLast = preferences.useLastChatSettings && preferences.lastModel.isNotBlank()
        val modelId = if (useLast) preferences.lastModel else preferences.defaultModel
        val selectable = state.value.account?.models?.firstOrNull { it.id == modelId && it.selectable }?.id
            ?: state.value.model
        val thinkingOn = if (useLast) preferences.lastEnableThinking else preferences.defaultEnableThinking
        val thinkingLevel = if (useLast) preferences.lastThinkingLevel else preferences.defaultThinkingLevel
        val thinkingBudget = if (useLast) preferences.lastThinkingBudget else preferences.defaultThinkingBudget
        val effort = if (useLast) preferences.lastReasoningEffort else preferences.defaultReasoningEffort
        val safety = if (useLast) preferences.lastSafetySetting else preferences.defaultSafetySetting
        mutable.update { it.copy(preferences = preferences,
            model = selectable.ifBlank { it.model },
            enableThinking = thinkingOn,
            enableSearch = if (useLast) preferences.lastEnableSearch else preferences.defaultEnableSearch,
            enableUrlContext = if (useLast) preferences.lastEnableUrlContext else preferences.defaultEnableUrlContext,
            enableMaps = if (useLast) preferences.lastEnableMaps else preferences.defaultEnableMaps,
            enablePython = if (useLast) preferences.lastEnablePython else preferences.defaultEnablePython,
            enableFileCreation = if (useLast) preferences.lastEnableFileCreation else preferences.defaultEnableFileCreation,
            enableSystemPrompt = if (useLast) preferences.lastEnableSystemPrompt else preferences.defaultEnableSystemPrompt,
            enableMcp = if (useLast) preferences.lastEnableMcp else preferences.defaultEnableMcp,
            generationValues = it.generationValues + ((selectable.ifBlank { it.model }) to (it.generationValues[selectable.ifBlank { it.model }].orEmpty() + mapOf(
                "thinking_level" to thinkingLevel,
                "thinking_budget" to thinkingBudget.toString(),
                "reasoning_effort" to effort,
                "safety_setting" to safety,
            )))
        ) }
    }

    fun loadPreferences() {
        viewModelScope.launch {
            mutable.update { it.copy(prefsBusy = true) }
            try { fetchPreferences() } catch (e: Exception) { report(e) } finally { mutable.update { it.copy(prefsBusy = false) } }
        }
    }

    fun savePreferences(payload: JSONObject) {
        viewModelScope.launch {
            mutable.update { it.copy(prefsBusy = true) }
            try {
                val reply = api.put("/api/mobile/v1/preferences", payload, token())
                mutable.update { it.copy(preferences = parsePreferences(reply), notice = "設定を保存しました。") }
            } catch (e: Exception) { report(e) }
            finally { mutable.update { it.copy(prefsBusy = false) } }
        }
    }

    fun loadStorageUsage() {
        viewModelScope.launch {
            runCatching { mutable.update { it.copy(storage = parseStorageUsage(api.get("/api/storage", token()))) } }
                .onFailure { report(it) }
        }
    }

    fun loadFeedback() {
        viewModelScope.launch {
            mutable.update { it.copy(feedbackBusy = true) }
            try { mutable.update { it.copy(feedbackItems = parseFeedbackItems(api.get("/api/feedback", token()))) } }
            catch (e: Exception) { report(e) }
            finally { mutable.update { it.copy(feedbackBusy = false) } }
        }
    }

    fun submitFeedback(title: String, message: String) {
        if (message.isBlank()) return
        viewModelScope.launch {
            mutable.update { it.copy(feedbackBusy = true) }
            try {
                api.post("/api/feedback", JSONObject().put("title", title.trim()).put("message", message.trim()), token())
                loadFeedback()
                notify("フィードバックを送信しました。")
            } catch (e: Exception) { report(e) }
            finally { mutable.update { it.copy(feedbackBusy = false) } }
        }
    }

    fun loadMcpServers() {
        viewModelScope.launch {
            mutable.update { it.copy(mcpBusy = true) }
            try { mutable.update { it.copy(mcpServers = parseMcpServers(api.get("/api/mcp/servers", token()))) } }
            catch (e: Exception) { report(e) }
            finally { mutable.update { it.copy(mcpBusy = false) } }
        }
    }

    fun setMcpServerEnabled(server: McpServerInfo, enabled: Boolean) {
        viewModelScope.launch {
            mutable.update { it.copy(mcpBusy = true) }
            try {
                api.put("/api/mcp/servers/${server.id}", JSONObject().put("enabled", enabled), token())
                loadMcpServers()
            } catch (e: Exception) { report(e) }
            finally { mutable.update { it.copy(mcpBusy = false) } }
        }
    }

    fun applySlash(action: SlashAction) {
        when (action.id) {
            "canvas" -> toggleCanvas()
            "coding" -> toggleCoding()
            "search" -> toggleSearch()
            "urls" -> toggleUrlContext()
            "maps" -> toggleMaps()
            "python" -> togglePython()
            "file" -> toggleFileCreation()
            "mcp" -> toggleMcp()
            "sysprompt" -> toggleSystemPrompt()
            "promptcache" -> togglePromptCache()
            "tempchat" -> toggleTemporaryChat()
            "compress" -> saveCompressionSettings(state.value.compression.copy(enabled = !state.value.compression.enabled))
            "thinking" -> {
                val arg = action.argument.lowercase()
                if (arg == "off") mutable.update { it.copy(enableThinking = false) }
                else {
                    val level = when (arg) {
                        "min", "minimal" -> "minimal"
                        "low" -> "low"
                        "mid", "medium" -> "medium"
                        "high" -> "high"
                        else -> null
                    }
                    if (level != null) {
                        mutable.update { it.copy(enableThinking = true) }
                        generationOption("thinking_level", level)
                    } else notify("Thinkingは off / min / low / mid / high を指定してください。")
                }
            }
            "effort" -> {
                val arg = action.argument.lowercase()
                if (arg in listOf("none", "low", "medium", "med", "high", "xhigh", "max")) {
                    generationOption("reasoning_effort", if (arg == "med") "medium" else arg)
                } else notify("Effortは none / low / medium / high / xhigh / max を指定してください。")
            }
            "safety" -> {
                val arg = action.argument.lowercase()
                if (arg in listOf("default", "none")) generationOption("safety_setting", arg)
                else notify("Safetyは default / none を指定してください。")
            }
        }
        if (action.consumeDraft) mutable.update { it.copy(draft = "") }
    }

    /** Fetches the server thread payload and writes a native A4 PDF into the share cache. */
    fun exportPdf(onReady: (File) -> Unit) {
        val id = state.value.selected?.id ?: return
        viewModelScope.launch {
            mutable.update { it.copy(busy = true) }
            try {
                val payload = api.get("/c/$id/pdf", token())
                val messages = parsePdfMessages(payload)
                val title = payload.optJSONObject("thread")?.optString("title").orEmpty().ifBlank { "AI Chat" }
                val safeId = id.filter { it.isLetterOrDigit() || it == '-' || it == '_' }.take(24).ifBlank { "thread" }
                val directory = File(getApplication<Application>().cacheDir, "shared").apply { mkdirs() }
                val target = File(directory, "thread-$safeId.pdf")
                withContext(Dispatchers.IO) {
                    writeThreadPdf(title, payload.optString("generated_at"), messages, target)
                }
                onReady(target)
            } catch (e: Exception) { report(e) }
            finally { mutable.update { it.copy(busy = false) } }
        }
    }

    private companion object {
        const val CHUNK_UPLOAD_THRESHOLD_BYTES = 8L * 1024 * 1024
        const val MAX_SINGLE_UPLOAD_BYTES = 64L * 1024 * 1024
    }
}
