package com.minashin1120.aiplayground

import com.minashin1120.aiplayground.data.AccountApi
import com.minashin1120.aiplayground.data.ApiException
import com.minashin1120.aiplayground.data.ChatMessage
import com.minashin1120.aiplayground.data.ChatNavigationHistory
import com.minashin1120.aiplayground.data.Diagnostics
import com.minashin1120.aiplayground.data.OfflineCacheStore
import com.minashin1120.aiplayground.data.Preferences
import com.minashin1120.aiplayground.data.ThreadItem
import com.minashin1120.aiplayground.data.backend.ChatBackend
import com.minashin1120.aiplayground.data.needsReauth
import com.minashin1120.aiplayground.data.nullableString
import com.minashin1120.aiplayground.data.numericId
import com.minashin1120.aiplayground.data.parsePreferences
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.util.UUID

/**
 * スレッドとメッセージの操作（Web `deleteThread` / `renameThread` / `branchChat` /
 * `deleteMessage` / `toggleThreadEncryptionFromModal` / `save-thread-settings-btn`）だけを持つ：
 * 削除・名前変更・ブックマーク・分岐・分岐元を開く・メッセージ削除・暗号化の変更・設定の保存。
 * 状態の更新は `mutable` へ行い、画面遷移・履歴の再読込・読み込みは生成時に受け取った関数へ委ねる。
 */
internal class ChatThreadActions(
    private val state: StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val scope: CoroutineScope,
    private val backend: () -> ChatBackend,
    private val token: () -> String,
    private val generations: ChatGenerations,
    private val messageLoader: ChatMessageLoader,
    private val pendingSend: ChatPendingState,
    private val offlineCache: OfflineCacheStore,
    private val chatHistory: ChatNavigationHistory,
    private val syncRunner: ChatSyncRunner,
    private val uploadsLocal: () -> Boolean,
    private val notifyChatOffline: (String?, String) -> Boolean,
    private val notifyDeleteMessageOffline: (String?, Long, String) -> Boolean,
    private val notify: (String) -> Unit,
    private val accountApi: () -> AccountApi,
    private val fetchThreads: suspend (Boolean) -> Unit,
    private val syncHeartbeat: () -> Unit,
    private val loadMessages: suspend (String) -> Unit,
    private val openThread: (ThreadItem, Boolean, Int?) -> Unit,
    private val newChat: (Boolean, Boolean) -> Unit,
    private val applySavedPreferences: (ChatState, Preferences) -> ChatState,
    private val report: suspend (Throwable) -> Unit,
    private val reportWithMessage: suspend (Throwable, String) -> Unit,
    private val notifySendOffline: () -> Boolean,
) {
    private val branchRequestIds = HashMap<String, String>()

    /** [onNewChat] runs when the open chat was deleted (Web `startNewChat` also closes the phone sidebar). */
    fun deleteThread(thread: ThreadItem, onNewChat: () -> Unit = {}) { scope.launch {
        Diagnostics.log("thread.delete", "thread_id" to thread.id, "offline" to state.value.offline)
        if (notifyChatOffline(thread.id, "オフライン中は履歴を削除できません。")) return@launch
        try {
            generations.forThread(thread.id)?.let { answer ->
                val jobId = answer.snapshot.value.jobId
                if (!uploadsLocal() || jobId != null) {
                    jobId?.let { messageLoader.suppress(it) }
                    try {
                        backend().post("/api/stop_chat", JSONObject().put("thread_id", thread.id).apply {
                            jobId?.let { put("job_id", it) }
                        }, token())
                    } catch (e: ApiException) { if (e.status != 404) throw e }
                }
                answer.job?.cancelAndJoin()
            }
            backend().delete("/api/threads/${thread.id}", token())
            pendingSend.removeFailed(thread.id)
            Diagnostics.log("thread.deleted", "thread_id" to thread.id)
            state.value.account?.let { account -> withContext(Dispatchers.IO) { offlineCache.deleteThread(account.id, thread.id) } }
            chatHistory.removeThread(thread.id)
            mutable.update { it.copy(canGoBackInChats = chatHistory.canGoBack) }
            if (state.value.selected?.id == thread.id) { newChat(false, false); onNewChat() }
            fetchThreads(false)
        } catch (e: Exception) { report(e) }
    } }
    /** Web `renameThread`: `prompt("Title:")` then PUT the new title; empty input is ignored. */
    fun renameThread(thread: ThreadItem, title: String) { scope.launch {
        if (title.isEmpty()) return@launch
        if (notifyChatOffline(thread.id, "オフライン中はタイトルを変更できません。")) return@launch
        try {
            val reply = backend().put("/api/threads/${thread.id}/title", JSONObject().put("title", title), token())
            val saved = reply.optString("title", title).ifBlank { title }
            mutable.update { current -> current.copy(
                selected = current.selected?.let { if (it.id == thread.id) it.copy(title = saved) else it },
            ) }
            fetchThreads(false)
        } catch (e: Exception) { report(e) }
    } }

    fun branchChat(message: ChatMessage, confirmed: Boolean = false) {
        val current = state.value
        val source = current.selected ?: return
        val id = numericId(message) ?: return
        if (current.branchBusy || current.streaming || current.busy) return
        if (source.isTemporary && !confirmed) return
        val key = source.id + ":" + id
        val requestId = branchRequestIds.getOrPut(key) { UUID.randomUUID().toString().replace("-", "") }
        mutable.update { it.copy(branchBusy = true) }
        scope.launch {
            Diagnostics.log("branch.start", "op" to requestId)
            try {
                val reply = backend().post("/api/threads/${source.id}/branch", JSONObject()
                    .put("message_id", id).put("request_id", requestId).put("confirm_persist_temporary", confirmed), token())
                branchRequestIds.remove(key)
                openThread(ThreadItem(id = reply.getString("id"), title = reply.getString("title"), model = source.model), true, reply.getInt("leaf_id"))
                fetchThreads(false)
                if (state.value.serverless && !state.value.localProfile) syncRunner.afterAnswer()
                Diagnostics.log("branch.done", "op" to requestId)
                notify("新しいチャットに分岐しました")
            } catch (e: CancellationException) { Diagnostics.log("branch.cancelled", "op" to requestId); throw e }
            catch (e: Exception) { Diagnostics.log("branch.error", "op" to requestId); report(e) }
            finally { mutable.update { it.copy(branchBusy = false) } }
        }
    }

    fun openBranchOrigin() {
        val origin = state.value.branchOrigin ?: return
        if (!origin.optBoolean("available")) return
        openThread(ThreadItem(id = origin.getString("thread_id"), title = "分岐元のチャット", model = ""),
            true, origin.optInt("message_id").takeIf { it > 0 })
    }

    /** Web `deleteMessage`: removes the message and everything after it, then reloads the thread. */
    fun deleteMessage(message: ChatMessage) {
        val id = numericId(message) ?: return
        val thread = state.value.selected ?: return
        scope.launch {
            if (notifyDeleteMessageOffline(thread.id, id, "オフライン中はメッセージを削除できません。")) return@launch
            try {
                backend().delete("/api/messages/$id", token())
                if (state.value.selected?.id == thread.id) {
                    // Web reloads the chat without `preserveDraft`, which runs `cancelEdit`.
                    pendingSend.clearEdit()
                    mutable.update { it.copy(leafId = null, editingMessageId = null, draft = "", attachments = emptyList(), quote = "", quoteSourceId = null, quoteMessageId = null) }
                    loadMessages(thread.id)
                }
            } catch (e: Exception) { report(e) }
        }
    }

    /**
     * Web `toggleThreadEncryptionFromModal` (admin): decrypts or re-encrypts the open thread and reloads it.
     * A `reauth_required` answer hands [onReauth] a retry to run once the user confirms their identity.
     */
    fun setThreadEncryption(enable: Boolean, onReauth: (retry: () -> Unit) -> Unit, onDone: (Boolean) -> Unit) {
        val action = if (enable) "再暗号化" else "復号化"
        val thread = state.value.selected ?: run { notify("チャットがありません"); onDone(false); return }
        if (notifyChatOffline(thread.id, "オフライン中は暗号化を変更できません。")) { onDone(false); return }
        scope.launch {
            try {
                val reply = accountApi().setThreadEncryption(thread.id, enable)
                notify("${action}しました（${reply.optInt("changed")}件を変換）")
                if (state.value.selected?.id == thread.id) loadMessages(thread.id)
                onDone(true)
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                if (e.needsReauth()) onReauth { setThreadEncryption(enable, onReauth, onDone) }
                else notify((e as? ApiException)?.payload?.optString("error")?.ifBlank { null } ?: "${action}に失敗しました")
                onDone(false)
            }
        }
    }

    /** Pull-to-refresh of the sidebar thread list. */
    fun reloadThreads(onDone: () -> Unit) { scope.launch {
        try { fetchThreads(false) } catch (e: Exception) { report(e) } finally { onDone() }
    } }

    fun toggleBookmark(thread: ThreadItem) { scope.launch {
        if (notifyChatOffline(thread.id, "オフライン中はブックマークを変更できません。")) return@launch
        try {
            val reply = backend().post("/api/threads/${thread.id}/bookmark", JSONObject(), token())
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
        scope.launch {
            if (notifyChatOffline(thread.id, "オフライン中はチャット設定を変更できません。")) return@launch
            mutable.update { it.copy(busy = true) }
            try {
                val normalizedTitle = title.trim().ifBlank { "新しいチャット" }
                if (normalizedTitle != thread.title) {
                    backend().put("/api/threads/${thread.id}/title", JSONObject().put("title", normalizedTitle), token())
                }
                val reply = backend().put("/api/threads/${thread.id}/settings", JSONObject()
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
    /**
     * Web `save-thread-settings-btn`: the thread's own instruction, then the user system prompt and
     * auto-injected prompts. Toasts 保存されました only when both requests succeed.
     */
    fun saveChatInstructions(instruction: String, includeGlobal: Boolean, userPrompt: JSONObject, onDone: (Boolean) -> Unit) {
        val thread = state.value.selected ?: return onDone(false)
        scope.launch {
            if (notifyChatOffline(thread.id, "オフライン中はチャット設定を変更できません。")) { onDone(false); return@launch }
            try {
                backend().put("/api/threads/${thread.id}/settings", JSONObject()
                    .put("custom_instruction", instruction)
                    .put("include_global_instruction", includeGlobal), token())
                mutable.update { current ->
                    if (current.selected?.id != thread.id) current
                    else current.copy(customInstruction = instruction, includeGlobalInstruction = includeGlobal)
                }
                val reply = backend().put("/api/mobile/v1/preferences", userPrompt, token())
                mutable.update { applySavedPreferences(it, parsePreferences(reply)).copy(notice = "保存されました") }
                onDone(true)
            } catch (e: Exception) {
                if (e is ApiException) notify("保存に失敗しました") else notify("エラー: ${e.message}")
                onDone(false)
            }
        }
    }

    /** Web `applyTemporaryChatSetting`: an open chat sends only `is_temporary` (its title and instruction stay). */
    fun toggleTemporaryChat() {
        val selected = state.value.selected
        if (selected == null) {
            mutable.update { it.copy(newThreadTemporary = !it.newThreadTemporary) }
            return
        }
        val next = !selected.isTemporary
        scope.launch {
            if (notifyChatOffline(selected.id, "オフライン中はチャット設定を変更できません。")) return@launch
            try {
                val reply = backend().put("/api/threads/${selected.id}/settings", JSONObject().put("is_temporary", next), token())
                mutable.update { current -> current.copy(
                    selected = current.selected?.takeIf { it.id == selected.id }?.copy(isTemporary = reply.optBoolean("is_temporary", next))
                        ?: current.selected,
                    tempChatTimeoutSeconds = reply.optInt("timeout_seconds").takeIf { value -> reply.has("timeout_seconds") && value > 0 }
                        ?: current.tempChatTimeoutSeconds,
                    tempChatRemainingSeconds = reply.optLong("temp_chat_remaining_seconds")
                        .takeIf { value -> !reply.isNull("temp_chat_remaining_seconds") && value >= 0 },
                ) }
                fetchThreads(false)
                syncHeartbeat()
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) { reportWithMessage(e, "一時チャット設定の更新に失敗しました") }
        }
    }

    /** Web `openThreadModal`: a new chat is created first so its settings can be edited. */
    fun ensureThread(onReady: () -> Unit) {
        if (state.value.selected != null) { onReady(); return }
        if (notifySendOffline()) return
        scope.launch {
            try {
                val created = backend().post("/api/threads", JSONObject().put("is_temporary", state.value.newThreadTemporary), token())
                val id = created.get("id").toString()
                mutable.update { it.copy(selected = ThreadItem(id, created.nullableString("title"), it.model,
                    isTemporary = created.optBoolean("is_temporary")), newThreadTemporary = false) }
                syncHeartbeat()
                reloadThreads {}
                onReady()
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) { notify("チャットの作成に失敗しました") }
        }
    }
}
