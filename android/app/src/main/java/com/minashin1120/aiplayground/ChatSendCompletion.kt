package com.minashin1120.aiplayground

import com.minashin1120.aiplayground.data.Attachment
import com.minashin1120.aiplayground.data.ApiException
import com.minashin1120.aiplayground.data.Diagnostics
import com.minashin1120.aiplayground.data.backend.ChatBackend
import com.minashin1120.aiplayground.data.nullableString
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import org.json.JSONObject

/** 送信内容と、サーバーが受け付けなかったときに画面へ戻す入力の状態。 */
internal data class ChatSubmission(
    val body: JSONObject,
    val files: List<Attachment>,
    val mask: String? = null,
    val editingId: String? = null,
    val parentId: Int? = null,
    val settingsBeforeEdit: PromptEditSettings? = null,
)

/** 送信・編集の途中状態（受け付けられなかった送信の保持、編集中の親メッセージ、編集前の入力設定）。 */
internal class ChatPendingState {
    /** 送信を受け付けられなかった内容。再試行や、チャットを開いたときの入力の復元に使う。 */
    var failed: ChatSubmission? = null
        private set
    /** チャットごとの、受け付けられなかった送信。 */
    private val failedByThread = mutableMapOf<String, ChatSubmission>()
    /** 編集・再生成で次の送信がぶら下がる親メッセージ（分岐の基点）。 */
    var parentId: Int? = null
    /** 編集を始めたときの入力設定。キャンセル時に戻す。 */
    var settingsBeforeEdit: PromptEditSettings? = null

    fun clearFailed() { failed = null }

    fun saveFailed(threadId: String, submission: ChatSubmission) { failedByThread[threadId] = submission }

    fun restoreFailedForThread(threadId: String?): ChatSubmission? = threadId?.let { failedByThread[it] }

    fun takeFailedForThread(threadId: String?) {
        failed = restoreFailedForThread(threadId)
    }

    fun removeFailed(threadId: String) { failedByThread.remove(threadId) }

    fun setFailed(submission: ChatSubmission?) { failed = submission }

    fun clearAllFailed() { failed = null; failedByThread.clear() }

    fun clearEdit() { parentId = null; settingsBeforeEdit = null }

    /** 送信をサーバーが受け付けなかったとき、編集状態を入力の復元に合わせる。 */
    fun restoreEditing(submission: ChatSubmission) {
        parentId = submission.parentId
        if (settingsBeforeEdit == null) settingsBeforeEdit = submission.settingsBeforeEdit
    }
}

/**
 * Web `sendMessage` の送信後処理だけを持つ：回答終了時の履歴の再読込・タイトル生成・Batch 通知と、
 * サーバーが受け付けなかった送信の入力の復元・案内（APIキー不足・Turnstile・禁止・通信エラー）。
 * 状態は持たず、画面の更新や読み込みは生成時に受け取った関数へ委ねる
 * （`ChatViewModel` の送信本体とは別に読めるようにする）。
 */
internal class ChatSendCompletion(
    private val state: StateFlow<ChatState>,
    private val scope: CoroutineScope,
    private val pending: ChatPendingState,
    private val matchesScreen: (ChatGenerations.Answer) -> Boolean,
    private val updateAnswer: (ChatGenerations.Answer, (ChatState) -> ChatState) -> Unit,
    private val vibrate: () -> Unit,
    private val notify: (String) -> Unit,
    private val report: suspend (Throwable) -> Unit,
    private val showApiKeyPrompt: (ChatGenerations.Answer, String?, String) -> Unit,
    private val startTurnstile: () -> Unit,
    private val refreshThreads: suspend () -> Unit,
    private val reloadMessages: suspend (String) -> Unit,
    private val refreshBatch: suspend () -> Unit,
    private val notifyFinishedIfAway: (String, Boolean) -> Unit,
    private val isDeviceChat: (String) -> Boolean,
    private val applyTitle: (String, String, String) -> Unit,
    private val canReloadHistory: () -> Boolean,
) {
    /** 回答が終わったとき（正常終了・回答内のエラーとも）の後処理。 */
    suspend fun finish(
        answer: ChatGenerations.Answer,
        body: JSONObject,
        answerBackend: ChatBackend,
        answerToken: String,
        threadId: String,
        modelId: String,
        sendStarted: Long,
    ) {
        val answerFailed = answer.snapshot.value.live.error != null
        Diagnostics.log("send.done", "op" to answer.operationId, "ms" to System.currentTimeMillis() - sendStarted)
        vibrate()
        if (matchesScreen(answer)) pending.clearFailed()
        // The response may add a new leaf even for a normal continuation. Drop the old
        // leafId so loadMessages() selects the newest path instead of restoring the
        // previously active branch and hiding the just-sent exchange.
        updateAnswer(answer) { it.copy(leafId = null) }
        if (matchesScreen(answer)) state.value.selected?.id?.let { reloadMessages(it) }
        refreshThreads()
        if (!answerFailed && !isDeviceChat(threadId) && state.value.selected?.id == threadId &&
            state.value.selected?.title in NEW_THREAD_TITLES) {
            // Match Web: title generation must not delay completion of the answer.
            scope.launch {
                try {
                    val reply = answerBackend.post("/api/generate_title", JSONObject()
                        .put("thread_id", threadId).put("model_id", modelId), answerToken)
                    reply.nullableString("title").takeIf { it.isNotBlank() }?.let { title ->
                        applyTitle(answerToken, threadId, title)
                        refreshThreads()
                    }
                } catch (e: CancellationException) { throw e }
                catch (e: Exception) { Diagnostics.failure("thread.open_error", e) }
            }
        }
        if (body.optBoolean("batch_mode")) refreshBatch() else notifyFinishedIfAway(threadId, answerFailed)
    }

    /** 送信が失敗したときの後処理。`accepted` が false なら、送信前の入力へ戻す。 */
    suspend fun fail(
        answer: ChatGenerations.Answer,
        body: JSONObject,
        submission: ChatSubmission,
        threadId: String,
        modelId: String,
        accepted: Boolean,
        error: Throwable,
    ) {
        // Web: before the server accepted the send, the optimistic rows go away and the input,
        // attachments and quote stay; the error is shown as "Connection Error: …".
        if (!accepted) {
            if (threadId.isNotBlank()) pending.saveFailed(threadId, submission)
            val userId = "local-${body.optString("client_request_id")}"
            updateAnswer(answer) { it.copy(
                messages = it.messages.filterNot { m -> m.id == userId },
                draft = if (it.draft.isBlank()) body.optString("message") else it.draft,
                attachments = if (it.attachments.isEmpty()) submission.files else it.attachments,
                quote = if (it.quote.isBlank()) body.optString("quote_text") else it.quote,
                imageMask = it.imageMask ?: submission.mask,
                editingMessageId = it.editingMessageId ?: submission.editingId,
            ) }
            if (matchesScreen(answer) && state.value.editingMessageId == submission.editingId && submission.editingId != null) {
                pending.restoreEditing(submission)
            }
        }
        when {
            error is ApiException && error.code == "api_key_missing" -> {
                // Web `showApiKeyRequiredModalAsync`: set the key, switch model, or show the error.
                showApiKeyPrompt(answer, error.payload.optString("error").ifBlank { null },
                    error.payload.optString("model").ifBlank { modelId })
            }
            // Web re-runs Turnstile, then asks to send again.
            error is ApiException && error.code == "turnstile_required" -> startTurnstile()
            error is ApiException && (error.code == "banned" || error.status == 401) -> report(error)
            error is ApiException -> notify("Connection Error: " + error.payload.optString("error").ifBlank { "HTTP ${error.status}" })
            else -> notify("Connection Error: " + (error.message ?: "通信に失敗しました"))
        }
        if (threadId.isNotBlank() && matchesScreen(answer) && canReloadHistory()) {
            updateAnswer(answer) { it.copy(leafId = null) }
            runCatching { reloadMessages(threadId) }
        }
        notifyFinishedIfAway(threadId, true)
    }

    private companion object {
        /** Web: a freshly created chat is renamed once the first answer is in. */
        val NEW_THREAD_TITLES = listOf("New Chat", "新しいチャット", "No Title")
    }
}
