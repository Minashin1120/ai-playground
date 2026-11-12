package com.minashin1120.aiplayground

import android.content.SharedPreferences
import com.minashin1120.aiplayground.data.ApiException
import com.minashin1120.aiplayground.data.Diagnostics
import com.minashin1120.aiplayground.data.PlaygroundApi
import com.minashin1120.aiplayground.data.StoredSession
import com.minashin1120.aiplayground.data.local.LocalChatStore
import com.minashin1120.aiplayground.data.local.LocalProfiles
import com.minashin1120.aiplayground.data.sync.SyncEngine
import com.minashin1120.aiplayground.data.sync.isRetryableSyncFailure
import com.minashin1120.aiplayground.data.sync.syncRetryDelayMillis
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Serverless mode's outbox: keeps sends made on the device and pushes them to the server when it is
 * reachable (`SyncEngine`, `/api/mobile/v1/sync/push`). Runs after an answer, when the app returns,
 * when the connection comes back, and from settings "今すぐ同期" ([ChatViewModel.syncNow]).
 * State lives in the view model; this class only reads it and reports progress through `mutable`.
 */
internal class ChatSyncRunner(
    private val state: StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val scope: CoroutineScope,
    private val store: () -> LocalChatStore?,
    private val session: () -> StoredSession?,
    private val token: () -> String,
    private val syncApi: () -> PlaygroundApi,
    private val hasUsableNetwork: () -> Boolean,
    private val generations: ChatGenerations,
    private val prefs: SharedPreferences,
    private val refreshThreads: suspend () -> Unit,
    private val reloadMessages: suspend (String) -> Unit,
    private val startSessionTurnstile: () -> Unit,
    private val report: suspend (Throwable) -> Unit,
) {
    private var scheduleJob: Job? = null
    private var job: Job? = null
    private var retryAttempt = 0

    /** Stops the scheduled and the running sync (when device chats are closed). */
    fun cancel() {
        scheduleJob?.cancel(); job?.cancel()
    }

    fun refreshPendingCount() {
        val chats = store() ?: return
        scope.launch {
            val count = withContext(Dispatchers.IO) { chats.pendingCount() }
            mutable.update { it.copy(pendingCount = count) }
        }
    }

    /** Sends what serverless mode still keeps on the device, when the server can be reached. */
    fun schedule(delayMillis: Long = 0) {
        if (!state.value.serverless || session() == null) return
        scheduleJob?.cancel()
        scheduleJob = scope.launch {
            delay(delayMillis)
            if (!hasUsableNetwork()) return@launch
            run(manual = false)
        }
    }

    /**
     * After an answer on the device: the outbox goes to the server, but the answer is finished without
     * waiting for uploads or a server that is out of reach. Sync continues in the background;
     * a failed one is tried again later ([scheduleRetry]).
     */
    suspend fun afterAnswer() {
        withContext(Dispatchers.Main.immediate) {
            if (state.value.offline || !hasUsableNetwork()) {
                scheduleRetry()
            } else {
                scope.launch { job?.join(); run(manual = false) }
            }
        }
    }

    /** Tries the outbox again later (30 seconds, doubling up to 10 minutes) after a sync that could not reach the server. */
    private fun scheduleRetry() {
        if (!state.value.serverless || session() == null) return
        retryAttempt++
        schedule(syncRetryDelayMillis(retryAttempt))
    }

    /** Settings "今すぐ同期". */
    fun syncNow() { scope.launch { run(manual = true) } }

    private suspend fun run(manual: Boolean) {
        val chats = store() ?: return
        // A cancelled sync still counts until it has really ended (a long file read finishes first), so
        // syncs never pile up reading the same attachments.
        if (!state.value.serverless || session() == null || job?.isCompleted == false) {
            Diagnostics.log("sync.skipped", "serverless" to state.value.serverless, "running" to (job?.isCompleted == false))
            return
        }
        val started = scope.launch {
            mutable.update { it.copy(syncing = true, syncMessage = null) }
            val begun = System.currentTimeMillis()
            Diagnostics.log("sync.start", "manual" to manual)
            try {
                val report = SyncEngine(chats, syncApi()) { token() }.run()
                for (answer in generations.all()) {
                    answer.threadId?.let { id ->
                        val mapped = withContext(Dispatchers.IO) { chats.resolveAlias(id) }
                        if (mapped != id) answer.bind(mapped)
                    }
                }
                Diagnostics.log("sync.done", "ms" to System.currentTimeMillis() - begun, "uploaded" to report.uploadedMessages,
                    "skipped_attachments" to report.skippedAttachments, "rejected" to report.rejected, "pending" to report.pending)
                retryAttempt = 0
                val now = System.currentTimeMillis()
                state.value.account?.let { account -> prefs.edit().putLong("last_sync_" + LocalProfiles.accountKey(account.id), now).apply() }
                val parts = listOfNotNull(
                    "送信 ${report.uploadedMessages}件",
                    report.skippedAttachments.takeIf { it > 0 }?.let { "サーバーへ送れなかった添付 ${it}件" },
                    report.rejected.takeIf { it > 0 }?.let { "次回に再送するメッセージ ${it}件" },
                )
                mutable.update { it.copy(syncing = false, lastSyncAt = now, pendingCount = report.pending, syncMessage = parts.joinToString("、")) }
                if (report.uploadedMessages > 0 && !state.value.streaming) {
                    runCatching { refreshThreads() }
                    state.value.selected?.id?.let { id -> if (!state.value.busy) runCatching { reloadMessages(id) } }
                }
            } catch (e: CancellationException) {
                Diagnostics.log("sync.cancelled", "ms" to System.currentTimeMillis() - begun)
                mutable.update { it.copy(syncing = false) }; throw e
            }
            catch (e: Exception) {
                Diagnostics.failure("sync.error", e, "ms" to System.currentTimeMillis() - begun)
                val message = when {
                    e is ApiException && e.code == "turnstile_required" -> "安全性の確認が必要です。送信欄から一度送信するか、しばらく待ってから同期してください。"
                    e is ApiException && e.code == "e2ee_migration_in_progress" -> "暗号化の切り替え中のため、完了後に同期します。"
                    e is ApiException && e.status == 401 -> "ログインの有効期限が切れました。"
                    else -> "同期できませんでした: " + (e.message ?: e.javaClass.simpleName) +
                        (if (isRetryableSyncFailure(e)) "（後で自動的に再試行します）" else "")
                }
                val pending = withContext(Dispatchers.IO) { runCatching { chats.pendingCount() }.getOrDefault(state.value.pendingCount) }
                mutable.update { it.copy(syncing = false, pendingCount = pending, syncMessage = message) }
                if (isRetryableSyncFailure(e)) scheduleRetry()
                if (manual && e is ApiException && e.code == "turnstile_required") startSessionTurnstile()
                if (e is ApiException && e.status == 401) report(e)
            }
        }
        job = started
        started.join()
    }
}
