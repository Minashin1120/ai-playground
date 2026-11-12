package com.minashin1120.aiplayground

import com.minashin1120.aiplayground.data.ChatLocation
import com.minashin1120.aiplayground.data.ChatNavigationHistory
import com.minashin1120.aiplayground.data.Diagnostics
import com.minashin1120.aiplayground.data.ThreadItem
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * 画面のチャット移動（Web `startNewChat` / `loadMessages` の入口・戻る・再読込）だけを持つ：
 * 新規チャット、スレッドを開く（読み込み中の骨組み・失敗時の再試行を含む）、戻る、
 * 過去分の追加、前面復帰時の再読込。状態の更新は `mutable` へ行い、履歴の読み込みや
 * 回答の再結合は生成時に受け取った関数へ委ねる（`ChatViewModel` は薄い入口だけを残す）。
 */
internal class ChatNavigator(
    private val state: StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val scope: CoroutineScope,
    private val chatHistory: ChatNavigationHistory,
    private val pendingSend: ChatPendingState,
    private val composer: ChatComposer,
    private val generations: ChatGenerations,
    private val nextTransition: (ChatTransitionKind) -> Pair<Long, ChatTransitionKind>,
    private val pinnedLeaf: (String) -> Int?,
    private val cancelHeartbeat: () -> Unit,
    private val withNewChatDefaults: (ChatState) -> ChatState,
    private val withModelRules: (ChatState) -> ChatState,
    private val withLatestPromptOptions: (ChatState) -> ChatState,
    private val loadMessages: suspend (String, Boolean, Boolean) -> Unit,
    private val fetchThreads: suspend (Boolean) -> Unit,
    private val resumeAnswer: () -> Unit,
    private val isForeground: () -> Boolean,
    private val report: suspend (Throwable, String?) -> Unit,
) {
    private var job: Job? = null

    /** The running navigation job (BAN・ログアウトで止める対象). */
    val activeJob: Job? get() = job

    private fun currentLocation() = ChatLocation(state.value.selected, state.value.newThreadTemporary)

    private fun recordNavigation(next: ChatLocation) {
        chatHistory.record(currentLocation(), next)
        mutable.update { it.copy(canGoBackInChats = chatHistory.canGoBack) }
    }

    fun goBack() {
        val previous = chatHistory.pop() ?: return
        mutable.update { it.copy(canGoBackInChats = chatHistory.canGoBack) }
        if (previous.thread == null) newChat(previous.temporary, recordHistory = false)
        else openThread(previous.thread, recordHistory = false)
    }

    fun newChat(temporary: Boolean = false, recordHistory: Boolean = true) {
        if (recordHistory) recordNavigation(ChatLocation(null, temporary))
        job?.cancel(); pendingSend.clearFailed()
        cancelHeartbeat()
        pendingSend.clearEdit()
        composer.modelChosenManually = false
        val transition = nextTransition(ChatTransitionKind.NEW_CHAT)
        mutable.update { current -> withModelRules(withNewChatDefaults(withLatestPromptOptions(current).copy(selected = null, messages = emptyList(), allMessages = emptyList(), leafId = null, settingsBubbles = emptyList(),
            editingMessageId = null, jobId = null, streaming = false,
            liveContent = "", liveThought = "", live = LiveAnswer(), mcpDecision = null, status = "", busy = false, retryAvailable = false,
            cards = emptyList(), branchOrigin = null, hasOlder = false, oldestId = null, customInstruction = "",
            includeGlobalInstruction = true, newThreadTemporary = temporary, tempChatRemainingSeconds = null, tempChatTimeoutSeconds = null,
            selectedSkill = null, codingTarget = null, imageMask = null,
            // Web `startNewChat`: cancelEdit + resetUploadState, and PromptCache starts off.
            draft = "", attachments = emptyList(), quote = "", quoteSourceId = null, quoteNavigationRequest = 0L, enablePromptCache = false,
            chatTransitionId = transition.first, chatTransitionKind = transition.second,
            chatNavigationId = transition.first, chatNavigationKind = transition.second))) }
    }
    /**
     * Web `loadMessages`: the composer is cleared (`cancelEdit`), the branch pinned in ブランチ管理 or else the
     * latest one is shown. The chat transition starts at once and the loading skeleton stays until the history arrives.
     */
    fun openThread(thread: ThreadItem, recordHistory: Boolean = true, focusMessageId: Int? = null) {
        if (recordHistory) recordNavigation(ChatLocation(thread))
        job?.cancel(); cancelHeartbeat(); pendingSend.clearFailed()
        pendingSend.clearEdit()
        if (state.value.selected?.id != thread.id) composer.modelChosenManually = false
        val transition = nextTransition(ChatTransitionKind.OPEN_THREAD)
        val pinned = pinnedLeaf(thread.id)
        mutable.update { it.copy(selected = thread, messages = emptyList(), allMessages = emptyList(), settingsBubbles = emptyList(),
            branchOrigin = null, leafId = focusMessageId ?: pinned, editingMessageId = null, streaming = false, busy = true,
            draft = "", attachments = emptyList(), quote = "", quoteSourceId = null, quoteNavigationRequest = 0L,
            liveContent = "", liveThought = "", live = LiveAnswer(), mcpDecision = null, jobId = null, retryAvailable = false,
            cards = emptyList(), hasOlder = false, oldestId = null,
            // Web `playChatTransition('history')` + `buildChatLoadingSkeletonHtml`: swap to the new chat now
            // and show the skeleton there; the messages replace it in place when they arrive.
            chatTransitionId = transition.first, chatTransitionKind = transition.second,
            threadLoadingId = transition.first, threadLoadFailedId = null,
            chatNavigationId = transition.first, chatNavigationKind = transition.second) }
        generations.restore(mutable, thread.id)
        job = scope.launch {
            val started = System.currentTimeMillis()
            Diagnostics.log("thread.open", "thread_id" to thread.id)
            try {
                loadMessages(thread.id, false, false)
                pendingSend.restoreFailedForThread(state.value.selected?.id)?.let { submission ->
                    pendingSend.setFailed(submission)
                    pendingSend.settingsBeforeEdit = submission.settingsBeforeEdit
                    mutable.update { it.copy(draft = submission.body.optString("message"), attachments = submission.files,
                        quote = submission.body.optString("quote_text"), imageMask = submission.mask,
                        editingMessageId = submission.editingId) }
                    pendingSend.parentId = submission.parentId
                }
                Diagnostics.log("thread.opened", "thread_id" to thread.id, "ms" to System.currentTimeMillis() - started,
                    "messages" to state.value.messages.size)
                mutable.update { it.copy(busy = if (it.selected?.id == thread.id) false else it.busy,
                    threadLoadingId = if (it.threadLoadingId == transition.first) 0L else it.threadLoadingId) }
                if (isForeground() && state.value.jobId != null) resumeAnswer()
            }
            catch (e: CancellationException) {
                Diagnostics.log("thread.open_cancelled", "thread_id" to thread.id, "ms" to System.currentTimeMillis() - started)
                throw e
            }
            catch (e: Exception) {
                Diagnostics.failure("thread.open_error", e, "thread_id" to thread.id, "ms" to System.currentTimeMillis() - started)
                if (state.value.threadLoadingId == transition.first) mutable.update { it.copy(threadLoadFailedId = thread.id) }
                report(e, "チャットの読み込みに失敗しました")
            }
            finally {
                mutable.update { it.copy(busy = false,
                    threadLoadingId = if (it.threadLoadingId == transition.first) 0L else it.threadLoadingId) }
            }
        }
    }

    /** Opens the chat in the sidebar/batch/bubble list; an unknown id is shown as a Batch chat. */
    fun openThreadId(id: String) {
        val item = state.value.threads.firstOrNull { it.id == id } ?: ThreadItem(id, "Batchチャット", "")
        openThread(item)
    }

    /** A notification tap (`openThreadId` from the bubble/answer notification) waits for startup first. */
    fun openBubbleTarget(id: String?) {
        scope.launch {
            state.first { !it.starting }
            if (state.value.account == null) return@launch
            val threadId = bubbleThreadId(id)
            if (threadId == null) newChat() else openThreadId(threadId)
        }
    }

    /** Web `showChatLoadError` 再試行: loads the chat that failed to open again. */
    fun retryOpenThread() {
        state.value.selected?.let { openThread(it, recordHistory = false) }
    }

    fun olderMessages() {
        val id = state.value.selected?.id ?: return
        job = scope.launch {
            mutable.update { it.copy(busy = true) }
            try { loadMessages(id, true, false) } catch (e: Exception) { report(e, null) }
            finally { mutable.update { it.copy(busy = false) } }
        }
    }

    /** The lock countdown reached zero: the Web reloads the page; the app clears the overlay and reloads. */
    fun accountLockExpired() {
        mutable.update { it.copy(accountLock = null) }
        refresh()
    }

    fun refresh() {
        val thread = state.value.selected
        if (thread != null) openThread(thread)
        else job = scope.launch { try { fetchThreads(false) } catch (e: Exception) { report(e, null) } }
    }

    /**
     * Coming back to the app: the open chat picks up new answers, a server job to rejoin and the temporary
     * chat heartbeat without reopening it, so the scroll position, draft and loaded history stay. Web does
     * not reload the chat when the tab becomes visible again.
     */
    fun refreshInPlace() {
        val thread = state.value.selected ?: return
        job?.cancel()
        job = scope.launch {
            Diagnostics.log("thread.refresh", "thread_id" to thread.id)
            try { loadMessages(thread.id, false, true) }
            catch (e: CancellationException) { throw e }
            catch (e: Exception) { report(e, null) }
        }
    }
}
