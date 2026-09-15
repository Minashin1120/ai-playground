package com.minashin1120.aiplayground

import android.app.Application
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
    val draft: String = "", val model: String = "", val attachments: List<Attachment> = emptyList(),
    val enableThinking: Boolean = false, val enableSearch: Boolean = false,
    val enablePromptCache: Boolean = false,
    val uploading: Boolean = false, val streaming: Boolean = false,
    val liveContent: String = "", val liveThought: String = "", val status: String = "",
    val cards: List<StatusCard> = emptyList(),
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
    private data class Submission(val body: JSONObject, val files: List<Attachment>)
    private var failed: Submission? = null

    init { viewModelScope.launch {
        session = withContext(Dispatchers.IO) { store.load() }
        if (session != null) runCatching { loadAccount() }.onFailure { report(it) }
        mutable.update { it.copy(starting = false) }
    } }
    private fun token(): String = session?.token ?: throw IOException("端末連携が必要です。")
    private suspend fun loadAccount() {
        val me = api.get("/api/mobile/v1/me", token())
        val account = Account(me.getInt("id"), me.getString("username"), parseModels(me),
            me.optString("default_model"), me.optBoolean("e2ee_enabled"))
        val chosen = prefs.getString("model_${account.id}", account.defaultModel).orEmpty()
            .takeIf { chosen -> account.models.any { it.id == chosen && it.selectable } }
            ?: account.models.firstOrNull { it.selectable }?.id.orEmpty()
        mutable.update { it.copy(account = account, model = chosen, pairing = false, userCode = "", offline = false) }
        fetchThreads(false)
    }
    fun setForeground(value: Boolean) {
        val returning = value && !foreground
        foreground = value
        if (!value && state.value.streaming) {
            streamJob?.cancel()
            mutable.update { it.copy(streaming = false, status = "アプリに戻ると履歴を確認します。") }
        }
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
            enablePromptCache = it.enablePromptCache && info.supports("prompt_cache")) }
        state.value.account?.let { prefs.edit().putString("model_${it.id}", model).apply() }
    }
    fun toggleThinking() { mutable.update { it.copy(enableThinking = !it.enableThinking) } }
    fun toggleSearch() { mutable.update { it.copy(enableSearch = !it.enableSearch) } }
    fun togglePromptCache() { mutable.update { it.copy(enablePromptCache = !it.enablePromptCache) } }
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
        val items = (0 until rows.length()).map {
            val row = rows.getJSONObject(it)
            ThreadItem(row.get("id").toString(), row.optString("title", "新しいチャット"), row.nullableString("last_model"))
        }
        mutable.update { it.copy(threads = if (more) (it.threads + items).distinctBy { t -> t.id } else items,
            nextPage = if (reply.optBoolean("has_next")) reply.optInt("next_page", page + 1) else null,
            offline = false) }
    }
    fun moreThreads() { navigationJob?.cancel(); navigationJob = viewModelScope.launch { try { fetchThreads(true) } catch (e: Exception) { report(e) } } }
    fun newChat() {
        navigationJob?.cancel(); streamJob?.cancel(); failed = null
        mutable.update { it.copy(selected = null, messages = emptyList(), jobId = null, streaming = false,
            liveContent = "", liveThought = "", status = "", busy = false, retryAvailable = false,
            cards = emptyList(), hasOlder = false, oldestId = null) }
    }
    fun openThread(thread: ThreadItem) {
        navigationJob?.cancel(); streamJob?.cancel(); failed = null
        mutable.update { it.copy(selected = thread, messages = emptyList(), streaming = false, busy = true,
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
        val messages = parseMessages(reply)
        mutable.update { it.copy(messages = if (older) (messages + it.messages).distinctBy { m -> m.id } else messages,
            hasOlder = reply.optBoolean("has_older_messages"), oldestId = reply.nullableString("oldest_loaded_id"),
            jobId = if (older) it.jobId else reply.optJSONObject("pending_job")?.nullableString("job_id")?.ifBlank { null },
            selected = it.selected?.let { selected -> selected.copy(title = reply.optString("title", selected.title)) },
            liveContent = if (older) it.liveContent else "", liveThought = if (older) it.liveThought else "",
            cards = if (older) it.cards else emptyList()) }
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
    fun send() {
        val current = state.value
        if (current.streaming || current.busy || current.uploading || (current.draft.isBlank() && current.attachments.isEmpty())) return
        if (current.model.isBlank()) { mutable.update { it.copy(notice = "モデルを選択してください。") }; return }
        val body = JSONObject().put("model", current.model).put("message", current.draft)
            .put("client_request_id", UUID.randomUUID().toString()).put("image_urls", JSONArray(current.attachments.map { it.reference }))
            .put("enable_thinking", current.enableThinking).put("enable_search", current.enableSearch)
            .put("enable_prompt_caching", current.enablePromptCache)
        current.selected?.let { body.put("thread_id", it.id) }
        val submission = Submission(body, current.attachments)
        failed = submission
        mutable.update { it.copy(draft = "", attachments = emptyList()) }
        submit(submission)
    }
    fun retry() { failed?.let { submit(it) } }
    private fun submit(submission: Submission) {
        streamJob?.cancel()
        streamJob = viewModelScope.launch {
            val owner = currentCoroutineContext().job
            mutable.update { it.copy(streaming = true, retryAvailable = false, status = "送信中…", liveContent = "", liveThought = "", cards = emptyList()) }
            var id = submission.body.nullableString("thread_id")
            try {
                if (id.isBlank()) {
                    val created = api.post("/api/threads", JSONObject(), token())
                    id = created.get("id").toString()
                    submission.body.put("thread_id", id)
                    mutable.update { it.copy(selected = ThreadItem(id, created.optString("title", "新しいチャット"), it.model)) }
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
    fun logout() { viewModelScope.launch {
        try {
            api.post("/api/mobile/v1/revoke", JSONObject(), token())
            clearSession()
        } catch (e: Exception) { report(e) }
    } }
    private suspend fun clearSession() {
        val caller = currentCoroutineContext().job
        listOf(pairingJob, navigationJob, streamJob, uploadJob).forEach { if (it !== caller) it?.cancel() }
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
        if (state.value.uploading) return
        if (uris.size + state.value.attachments.size > 30) { mutable.update { it.copy(notice = "添付は30件までです。") }; return }
        uploadJob = viewModelScope.launch {
            mutable.update { it.copy(uploading = true) }
            try {
                for (uri in uris) {
                    val resolver = getApplication<Application>().contentResolver
                    val name = withContext(Dispatchers.IO) { resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use {
                        if (it.moveToFirst()) it.getString(0) else null
                    } ?: "attachment" }
                    val body = object : RequestBody() {
                        override fun contentType() = resolver.getType(uri)?.toMediaTypeOrNull()
                        override fun writeTo(sink: BufferedSink) {
                            val input = resolver.openInputStream(uri) ?: throw IOException("添付を開けません。")
                            input.use {
                                val bytes = ByteArray(8192); var total = 0L
                                while (true) {
                                    val count = it.read(bytes); if (count < 0) break
                                    total += count
                                    if (total > 64L * 1024 * 1024) throw IOException("添付は1ファイル64MiBまでです。")
                                    sink.write(bytes, 0, count)
                                }
                            }
                        }
                    }
                    val uploaded = api.upload(name, body, token()).getString("filename")
                    mutable.update { it.copy(attachments = it.attachments + Attachment(name, uploaded)) }
                }
            } catch (e: Exception) { report(e) }
            finally { mutable.update { it.copy(uploading = false) } }
        }
    }
    fun openFile(reference: String, onReady: (File, String) -> Unit) { viewModelScope.launch {
        try {
            val directory = File(getApplication<Application>().cacheDir, "shared").apply { mkdirs() }
            val suffix = reference.substringAfterLast('.', "bin").take(8).filter { it.isLetterOrDigit() }.ifBlank { "bin" }
            val target = File(directory, "${UUID.randomUUID()}.$suffix")
            val mime = api.download(reference, target, token())
            onReady(target, mime)
        } catch (e: Exception) { report(e) }
    } }
}
