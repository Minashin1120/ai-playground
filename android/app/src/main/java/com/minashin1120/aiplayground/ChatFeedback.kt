package com.minashin1120.aiplayground

import android.app.Application
import android.content.Context
import android.net.Uri
import com.minashin1120.aiplayground.data.*
import com.minashin1120.aiplayground.data.backend.ChatBackend
import com.minashin1120.aiplayground.data.local.LocalChatStore
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import okhttp3.HttpUrl
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.IOException

/**
 * Feedback (Web `#feedback-modal` and `#fb-admin-list`): list, status change, reply, delete, the
 * submit box with its offline outbox, and the retry of persisted records. State lives in the view
 * model; this class only reads it and reports through `mutable`.
 */
internal class ChatFeedback(
    private val app: Application,
    private val state: StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val scope: CoroutineScope,
    private val api: PlaygroundApi,
    private val backend: () -> ChatBackend,
    private val token: () -> String,
    private val sessionToken: () -> String?,
    private val localChats: () -> LocalChatStore?,
    private val offlineCache: OfflineCacheStore,
    private val serverIdOf: suspend (String) -> String,
    private val notify: (String) -> Unit,
    private val notifyOffline: (String) -> Boolean,
    private val report: suspend (Throwable, String?) -> Unit,
    private val observeResponses: (PlaygroundApi) -> Unit,
) {
    private var apiOrigin: HttpUrl? = null
    private var feedbackApi: PlaygroundApi? = null
    private val pendingFeedback by lazy { PendingFeedbackStore(File(app.noBackupFilesDir, "pending_feedback")) }
    private val sendMutex = Mutex()
    private var retryJob: Job? = null

    private fun client(origin: HttpUrl): PlaygroundApi {
        if (apiOrigin != origin || feedbackApi == null) {
            apiOrigin = origin
            feedbackApi = PlaygroundApi(origin).also(observeResponses)
        }
        return feedbackApi!!
    }

    /** Drops idle connections of the feedback client (Web `discardIdleServerConnections`). */
    suspend fun discardIdleConnections() {
        feedbackApi?.discardIdleConnections()
    }

    fun loadFeedback() {
        scope.launch {
            if (notifyOffline("オフライン中はフィードバックを読み込めません。")) return@launch
            mutable.update { it.copy(feedbackBusy = true) }
            // Web `loadFeedback`: `all=1` returns every feedback to an administrator, otherwise the user's own.
            try {
                val list = parseFeedbackList(backend().get("/api/feedback?all=1", token()))
                mutable.update { it.copy(feedbackItems = list.items, feedbackAdmin = list.isAdmin) }
            }
            catch (e: Exception) { report(e, null) }
            finally { mutable.update { it.copy(feedbackBusy = false) } }
        }
    }

    /** Web `.fb-save` in `#fb-admin-list`: an administrator sets the status, then the list is read again. */
    fun updateFeedback(item: FeedbackItem, status: String) {
        scope.launch {
            if (notifyOffline("オフライン中はフィードバックを更新できません。")) return@launch
            mutable.update { it.copy(feedbackBusy = true) }
            try {
                backend().post("/api/feedback/${item.publicId.ifBlank { item.id.toString() }}/update",
                    JSONObject().put("status", status), token())
            } catch (e: Exception) {
                report(e, null)
            } finally { mutable.update { it.copy(feedbackBusy = false) } }
            loadFeedback()
        }
    }

    /**
     * Web `.fb-reply-send`: the sender (on their own feedback) or an administrator adds a reply. [withLogs] is the
     * ticked box: the last hour of the activity log goes with the reply (only while ログの収集を強化 is on);
     * [images] follow the reply once it was accepted.
     */
    fun replyFeedback(item: FeedbackItem, message: String, withLogs: Boolean, images: List<Uri> = emptyList(), withChat: Boolean = false) {
        val chatId = if (withChat) state.value.selected?.id else null
        if (withChat && chatId == null) { notify("コピーを送信するチャットが開かれていません"); return }
        if (message.isBlank()) return
        scope.launch {
            if (notifyOffline("オフライン中はフィードバックに返信できません。")) return@launch
            mutable.update { it.copy(feedbackBusy = true) }
            try {
                val payload = JSONObject().put("message", message.trim()).put("client", "android").put("version", BuildConfig.VERSION_NAME)
                val logs = if (!withLogs) null else if (ActivityLog.enabled) withContext(Dispatchers.IO) { ActivityLog.recent() } else if (state.value.preferences?.enableServerDiagnosticLog == true) JSONArray() else null
                if (logs != null) payload.put("client_logs", JSONObject().put("client", "android")
                    .put("version", BuildConfig.VERSION_NAME).put("window_seconds", ActivityLog.WINDOW_MS / 1000).put("entries", logs))
                val chatCopy = chatId?.let { feedbackChatCopy(it) }
                if (chatCopy != null) payload.put("chat_copy", chatCopy.payload)
                val feedbackId = item.publicId.ifBlank { item.id.toString() }
                val reply = backend().post("/api/feedback/$feedbackId/replies", payload, token())
                val replyId = reply.optInt("reply_id", 0)
                val imagesSaved = images.isEmpty() || (replyId > 0 && uploadFeedbackImages(feedbackId, images, replyId))
                var chatSaved = reply.optBoolean("chat_copy_saved", false)
                if (chatSaved && chatCopy != null) {
                    for (reference in chatCopy.deviceFiles) {
                        val temporary = File(app.cacheDir, "feedback-" + java.util.UUID.randomUUID())
                        try {
                            val store = localChats() ?: throw IOException("添付ファイルを読み込めません")
                            val info = withContext(Dispatchers.IO) { store.fileInfo(reference) }
                            val mime = withContext(Dispatchers.IO) { store.materializeFile(reference, temporary) }
                                ?: throw IOException("添付ファイルを読み込めません")
                            val saved = api.uploadFeedbackChatFile(feedbackId, reference,
                                info?.optString("name").orEmpty().ifBlank { reference.substringAfterLast('/') },
                                temporary, mime, token(), replyId)
                            if (!saved.optBoolean("saved")) chatSaved = false
                        } catch (e: CancellationException) { throw e }
                        catch (e: Exception) { Diagnostics.failure("error", e, "tab" to "feedback"); chatSaved = false }
                        finally { temporary.delete() }
                    }
                }
                loadFeedback()
                notify(feedbackReplySentText(logs?.let { reply.optInt("logs_count", it.length()) }, reply.optBoolean("logs_saved", true), images.size, imagesSaved, chatCopy != null, chatSaved))
            } catch (e: Exception) { report(e, "返信を送信できませんでした") }
            finally { mutable.update { it.copy(feedbackBusy = false) } }
        }
    }

    /** Deletes a feedback (the user's own, or any for an administrator); the server also removes its activity log and chat copy. */
    fun deleteFeedback(item: FeedbackItem) {
        scope.launch {
            if (notifyOffline("オフライン中はフィードバックを削除できません。")) return@launch
            mutable.update { it.copy(feedbackBusy = true) }
            try {
                backend().delete("/api/feedback/${item.publicId.ifBlank { item.id.toString() }}", token())
                mutable.update { it.copy(feedbackItems = it.feedbackItems.filterNot { row -> row.id == item.id }) }
                notify("フィードバックを削除しました")
            } catch (e: Exception) {
                report(e, "フィードバックを削除できませんでした")
            } finally { mutable.update { it.copy(feedbackBusy = false) } }
        }
    }

    /** [chatId] is the open chat whose copy goes with the feedback (null when the box is not ticked); [images] are the attached images. */
    fun submitFeedback(title: String, message: String, chatId: String? = null, images: List<Uri> = emptyList(), attachLogs: Boolean = false, onStored: () -> Unit = {}) {
        if (message.isBlank() || state.value.feedbackBusy) return
        scope.launch {
            if (state.value.localProfile || state.value.account == null) { notify("フィードバックの送信にはログインが必要です。"); return@launch }
            mutable.update { it.copy(feedbackBusy = true) }
            val origin = ServerOrigin.current
            val account = state.value.account!!.id
            val auth = sessionToken()
            try {
                val payload = JSONObject().put("title", title.trim()).put("message", message.trim())
                val logs = if (!attachLogs) null else if (ActivityLog.enabled) withContext(Dispatchers.IO) { ActivityLog.recent() } else if (state.value.preferences?.enableServerDiagnosticLog == true) JSONArray() else null
                if (logs != null) payload.put("client_logs", JSONObject().put("client", "android")
                    .put("version", BuildConfig.VERSION_NAME).put("window_seconds", ActivityLog.WINDOW_MS / 1000).put("entries", logs))
                val chatCopy = chatId?.let { feedbackChatCopy(it) }
                if (chatCopy != null) payload.put("chat_copy", chatCopy.payload)
                val record = withContext(Dispatchers.IO) {
                    val resolver = app.contentResolver
                    val attachments = images.mapIndexed { index, uri ->
                        PendingFeedbackStore.Attachment(resolver.getType(uri) ?: "application/octet-stream", "image-${index + 1}") {
                            resolver.openInputStream(uri) ?: throw IOException("画像を読み込めません")
                        }
                    } + chatCopy?.deviceFiles.orEmpty().map { reference ->
                        val store = localChats() ?: throw IOException("添付ファイルを読み込めません")
                        val info = store.fileInfo(reference)
                        PendingFeedbackStore.Attachment(info?.optString("mime") ?: "application/octet-stream",
                            info?.optString("name").orEmpty().ifBlank { reference.substringAfterLast('/') }, reference) {
                            // Snapshot encrypted local files now, so deletion of the chat cannot break retry.
                            val temporary = File(app.cacheDir, "feedback-" + java.util.UUID.randomUUID())
                            try {
                                if (store.materializeFile(reference, temporary) == null) throw IOException("添付ファイルを読み込めません")
                                temporary.inputStream().also { temporary.delete() }
                            } catch (e: Exception) { temporary.delete(); throw e }
                        }
                    }
                    pendingFeedback.enqueue(origin.toString(), account, payload, attachments)
                }
                onStored()
                ActivityLog.log("change", "tab" to "feedback", "state" to "pending", "persisted" to true)
                if (auth == null) {
                    notify("フィードバックを端末に保存しました。サーバーに接続できると自動送信します。")
                } else {
                    try {
                        val sent = sendMutex.withLock { PendingFeedbackSender(pendingFeedback, app.cacheDir, client(origin), auth).send(record) }
                        loadFeedback()
                        notify(sent)
                    } catch (e: CancellationException) { throw e }
                    catch (e: Exception) {
                        Diagnostics.failure("error", e, "tab" to "feedback", "state" to "pending", "retry" to true)
                        if (record.data.optBoolean("retry_paused"))
                            notify("フィードバックの送信が拒否されたため、自動再送を停止しました。送信待ちの内容は端末に保存されています。")
                        else notify("フィードバックを端末に保存しました。時間を空けて再送します。")
                    }
                }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) { report(e, "フィードバックを端末に保存できませんでした") }
            finally { mutable.update { it.copy(feedbackBusy = false) } }
        }
    }

    /** Every successful foreground probe also retries persisted records after process restart. */
    fun retryPending() {
        val account = state.value.account?.id ?: return
        if (state.value.localProfile || state.value.feedbackBusy || retryJob?.isActive == true) return
        val origin = ServerOrigin.current
        val auth = sessionToken() ?: return
        retryJob = scope.launch {
            try {
                val records = withContext(Dispatchers.IO) { pendingFeedback.records(origin.toString(), account) }
                var sentAny = false
                for (record in records) {
                    if (state.value.account?.id != account || ServerOrigin.current != origin || sessionToken() != auth) break
                    try {
                        val sent = sendMutex.withLock {
                            if (!PendingFeedbackSender.canRetry(record)) null
                            else PendingFeedbackSender(pendingFeedback, app.cacheDir, client(origin), auth).send(record)
                        }
                        if (sent != null) { sentAny = true; notify(sent) }
                    } catch (e: CancellationException) { throw e }
                    catch (e: Exception) {
                        Diagnostics.failure("error", e, "tab" to "feedback", "state" to "pending", "retry" to !record.data.optBoolean("retry_paused"))
                        if (record.data.optBoolean("retry_paused"))
                            notify("フィードバックの送信が拒否されたため、自動再送を停止しました。送信待ちの内容は端末に保存されています。")
                    }
                }
                if (sentAny) loadFeedback()
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) { Diagnostics.failure("error", e, "tab" to "feedback", "state" to "pending", "retry" to true) }
        }
    }

    /** Web `uploadFbImages`: sends the attached images to feedback [feedbackId] (or its reply [replyId]); false when they were not saved. */
    private suspend fun uploadFeedbackImages(feedbackId: String, uris: List<Uri>, replyId: Int? = null): Boolean = try {
        val resolver = app.contentResolver
        val files = withContext(Dispatchers.IO) {
            uris.mapIndexed { index, uri ->
                val bytes = resolver.openInputStream(uri)?.use { it.readBytes() } ?: throw IOException("画像を読み込めません")
                FeedbackImage("image-${index + 1}", resolver.getType(uri) ?: "application/octet-stream", bytes)
            }
        }
        api.uploadFeedbackImages(feedbackId, files, token(), replyId)
        true
    } catch (e: Exception) {
        Diagnostics.failure("feedback.images", e)
        false
    }

    private class FeedbackChatCopy(val payload: JSONObject, val deviceFiles: List<String>)

    /**
     * Web `chat_copy`: the server copies everything it has on the chat, decrypted. The device adds what only
     * it holds: the chat as it shows it while chats are on the device (a device chat, or unsent messages over
     * a server chat), its offline cache of the chat, preferences keyed by the chat, diagnostics entries not yet
     * sent, and the attachments kept only here (the encrypted outbox uploads these after the feedback is accepted).
     */
    private suspend fun feedbackChatCopy(id: String): FeedbackChatCopy {
        val resolved = serverIdOf(id)
        val ids = setOf(id, resolved)
        val copy = JSONObject().put("client", "android").put("version", BuildConfig.VERSION_NAME).put("thread_id", resolved)
        val view = if (localChats() != null) runCatching { backend().get("/api/threads/$resolved", token()) }.getOrNull() else null
        if (view != null) copy.put("thread", view)
        val accountId = state.value.account?.id
        withContext(Dispatchers.IO) {
            accountId?.let { account -> ids.firstNotNullOfOrNull { offlineCache.loadThread(account, it) } }
                ?.let { copy.put("offline_cache", it) }
            val stored = JSONObject()
            listOf("navigation", "branches", "settings_local").forEach { name ->
                val found = JSONObject()
                app.getSharedPreferences(name, Context.MODE_PRIVATE).all.forEach { (key, value) ->
                    if (ids.any { key.contains(it) }) found.put(key, value?.toString() ?: JSONObject.NULL)
                }
                if (found.length() > 0) stored.put(name, found)
            }
            copy.put("client_state", JSONObject().put("preferences", stored))
            copy.put("diagnostics", JSONArray(Diagnostics.pendingMentioning(ids)))
        }
        val messages = view?.optJSONArray("messages")
        val files = (0 until (messages?.length() ?: 0)).flatMap { index ->
            val raw = messages?.optJSONObject(index)?.opt("image_url")?.takeIf { it != JSONObject.NULL }?.toString().orEmpty()
            if (raw.isBlank()) emptyList()
            else runCatching { JSONArray(raw).let { refs -> (0 until refs.length()).map { refs.optString(it) } } }.getOrElse { listOf(raw) }
        }.filter { LocalChatStore.isLocalReference(it) }.distinct()
        copy.put("device_files", files.size)
        return FeedbackChatCopy(copy, files)
    }
}
