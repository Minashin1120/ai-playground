package com.minashin1120.aiplayground

import android.app.Application
import com.minashin1120.aiplayground.data.ApiException
import com.minashin1120.aiplayground.data.BatchJob
import com.minashin1120.aiplayground.data.backend.ChatBackend
import com.minashin1120.aiplayground.data.parseBatchJobs
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import org.json.JSONObject

/**
 * Batch jobs (Web `loadBatchJobs` / `refreshGeminiBatchStatus` / `#batch-notification-*`): the
 * history list, the completion banner, the 2-second status polling, and stopping or deleting a
 * job. State lives in the view model; this class only reads it and reports through `mutable`.
 */
internal class ChatBatch(
    private val app: Application,
    private val state: StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val scope: CoroutineScope,
    private val backend: () -> ChatBackend,
    private val token: () -> String,
    private val notify: (String) -> Unit,
    private val report: suspend (Throwable, String?) -> Unit,
    private val isForeground: () -> Boolean,
    private val hasSession: () -> Boolean,
    private val reloadInPlace: suspend (String) -> Unit,
    private val reloadThread: suspend (String) -> Unit,
    private val openThread: (String) -> Unit,
) {
    private var pollJob: Job? = null
    /** The running status poll (BAN・ログアウト・バックグラウンドで止める対象). */
    val activeJob: Job? get() = pollJob

    suspend fun fetch(notify: Boolean, status: Boolean = notify) {
        val previous = state.value.batchJobs.associateBy { it.id }
        if (status) runCatching { pollStatus() }
        val jobs = parseBatchJobs(backend().get("/api/batch/jobs", token()))
        // The system notification is added on Android (ANDROID_ONLY.md); the in-app banner follows Web.
        if (notify) jobs.filter { !it.active && previous[it.id]?.active == true }.forEach { job ->
            notifyBatchCompletion(app, job)
        }
        mutable.update { it.copy(batchJobs = jobs, batchBusy = false) }
    }

    /**
     * Web `refreshGeminiBatchStatus`: lets the server collect finished jobs, shows the banner for them and
     * reloads the open chat when one of them belongs to it. Returns whether any job finished.
     */
    private suspend fun pollStatus(): Boolean {
        val status = backend().get("/api/gemini/batch/status", token())
        val completed = status.optJSONArray("completed") ?: return false
        if (completed.length() == 0) return false
        val first = completed.getJSONObject(0)
        val text = if (completed.length() == 1) "${first.optString("model")} のBatch処理が完了しました。"
            else "${completed.length()}件のBatch処理が完了しました。"
        mutable.update { it.copy(batchBanner = text to first.optString("thread_id")) }
        val current = state.value.selected?.id
        if (current != null && (0 until completed.length()).any { completed.getJSONObject(it).optString("thread_id") == current }) {
            runCatching { reloadInPlace(current) }
        }
        return true
    }

    /** Web `#batch-notification-open` / `#batch-notification-close`. */
    fun dismissBanner(open: Boolean) {
        val banner = state.value.batchBanner ?: return
        mutable.update { it.copy(batchBanner = null) }
        if (open && banner.second.isNotBlank()) openThread(banner.second)
    }

    fun refresh() {
        scope.launch {
            mutable.update { it.copy(batchBusy = true) }
            try { fetch(notify = true) } catch (e: Exception) { report(e, null) }
            finally { mutable.update { it.copy(batchBusy = false) } }
        }
    }

    fun startPolling() {
        if (pollJob?.isActive == true) return
        pollJob = scope.launch {
            // Web checks every 2 seconds; the job list itself is refreshed on completion and every 30 seconds.
            var tick = 0
            while (isForeground() && hasSession()) {
                if (state.value.batchJobs.any { it.active }) {
                    val finished = runCatching { pollStatus() }.getOrDefault(false)
                    if (finished || tick % 15 == 0) runCatching { fetch(notify = true, status = false) }
                }
                tick++
                delay(2_000)
            }
        }
    }

    /** Web `loadBatchJobs`; [silent] skips the failure toast (the 5-second refresh in the modal). */
    fun load(silent: Boolean) { scope.launch {
        try { fetch(notify = true) }
        catch (e: CancellationException) { throw e }
        catch (e: Exception) { if (!silent) notify("Batch処理の履歴を取得できませんでした") }
    } }

    fun cancel(job: BatchJob) { scope.launch {
        try { backend().post("/api/batch/jobs/${job.id}/cancel", JSONObject(), token()) }
        catch (e: ApiException) { notify(e.payload.optString("error").ifBlank { "Batch処理を停止できませんでした" }); return@launch }
        catch (e: Exception) { notify("Batch処理を停止できませんでした"); return@launch }
        notify("Batch処理を停止しました")
        runCatching { fetch(notify = false) }
        if (state.value.selected?.id == job.threadId) runCatching { reloadThread(job.threadId) }
    } }

    fun delete(job: BatchJob) { scope.launch {
        try { backend().delete("/api/batch/jobs/${job.id}", token()) }
        catch (e: ApiException) { notify(e.payload.optString("error").ifBlank { "Batch履歴を削除できませんでした" }); return@launch }
        catch (e: Exception) { notify("Batch履歴を削除できませんでした"); return@launch }
        notify("Batch履歴を削除しました")
        runCatching { fetch(notify = false) }
    } }
}
