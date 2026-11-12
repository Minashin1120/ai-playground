package com.minashin1120.aiplayground

import com.minashin1120.aiplayground.data.ActivityLog
import com.minashin1120.aiplayground.data.ApiException
import com.minashin1120.aiplayground.data.Diagnostics
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.filterNotNull
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject

/**
 * Administrator diagnostics and the device activity log. State lives in the view model; this class
 * only reads it and reports through [Diagnostics] / [ActivityLog] and the server.
 */
internal class ChatDiagnostics(
    private val scope: CoroutineScope,
    private val state: StateFlow<ChatState>,
    private val token: () -> String?,
    private val foreground: () -> Boolean,
    observe: (PlaygroundApi) -> Unit,
) {
    /** Separate client for diagnostics, so sending them never shows the global spinner. */
    private val api by lazy { PlaygroundApi().also(observe) }

    fun start() {
        startDiagnostics()
        startActivityLog()
    }

    /** Stops the diagnostics client's idle sockets (BAN・ログアウトの境界). */
    fun discardIdleConnections() {
        api.discardIdleConnections()
    }

    /**
     * Administrator accounts only ([Diagnostics]): follows the account's admin flag once its settings are
     * known, sends the recorded entries every 10 seconds, and records the app's thread stacks when an
     * answer, a sync or a chat load has not moved for 20 seconds.
     */
    private fun startDiagnostics() {
        scope.launch {
            state.map { current -> if (current.account == null || current.preferences == null) null
                else !current.localProfile && current.preferences.isAdmin }
                .filterNotNull().distinctUntilChanged().collect { admin -> Diagnostics.setEnabled(admin) }
        }
        scope.launch(Dispatchers.IO) {
            var lastDump = 0L
            while (isActive) {
                delay(DIAGNOSTICS_INTERVAL_MS)
                if (!Diagnostics.enabled) continue
                val current = state.value
                val now = System.currentTimeMillis()
                val working = current.streaming || current.syncing || current.busy || current.uploading
                if (working && now - Diagnostics.lastEventAt > DIAGNOSTICS_STALL_MS && now - lastDump > DIAGNOSTICS_STACKS_GAP_MS) {
                    lastDump = now
                    Diagnostics.stacks("no_progress", "streaming" to current.streaming, "syncing" to current.syncing,
                        "busy" to current.busy, "uploading" to current.uploading, "serverless" to current.serverless,
                        "offline" to current.offline, "foreground" to foreground())
                }
                val token = token() ?: continue
                if (current.offline) continue
                val batch = Diagnostics.pending(200, 600_000)
                if (batch.lines == 0) continue
                try {
                    api.post("/api/mobile/v1/diagnostics", JSONObject().put("entries", JSONArray(batch.entries)), token)
                    Diagnostics.drop(batch.lines)
                } catch (e: CancellationException) {
                    throw e
                } catch (e: ApiException) {
                    // Not an administrator any more: stop and delete; a malformed batch is not sent again.
                    if (e.status == 403) Diagnostics.setEnabled(false)
                    else if (e.status == 400 || e.status == 413) Diagnostics.drop(batch.lines)
                } catch (_: Exception) {
                }
            }
        }
    }

    /**
     * ログの収集を強化 ([ActivityLog]): keeps the log to the signed-in account and records which values of
     * [ChatState] changed (states and counts only, never the user's text or answers).
     */
    private fun startActivityLog() {
        scope.launch {
            state.map { current -> current.account?.let { if (current.localProfile) "local" else "${ServerOrigin.current}#${it.id}" } }
                .filterNotNull().distinctUntilChanged().collect { owner -> ActivityLog.setOwner(owner) }
        }
        scope.launch(Dispatchers.Default) {
            var previous: Map<String, Any?> = emptyMap()
            state.collect { current ->
                if (!ActivityLog.enabled) { previous = emptyMap(); return@collect }
                val snapshot = activitySnapshot(current)
                val changed = snapshot.filter { (key, value) -> key !in previous || previous[key] != value }
                previous = snapshot
                if (changed.isNotEmpty()) ActivityLog.log("state", *changed.map { (key, value) -> key to (value ?: "null") }.toTypedArray())
            }
        }
    }

    private fun activitySnapshot(s: ChatState): Map<String, Any?> = linkedMapOf(
        "starting" to s.starting, "signed_in" to (s.account != null), "local_profile" to s.localProfile,
        "serverless" to s.serverless, "pairing" to s.pairing, "busy" to s.busy, "auth_busy" to s.authBusy,
        "auth_error" to (s.authError != null), "setup_required" to s.setupRequired, "server_checking" to s.serverChecking,
        "server_error" to (s.serverError != null), "syncing" to s.syncing, "pending_count" to s.pendingCount, "sync_message" to (s.syncMessage != null),
        "thread_id" to s.selected?.id, "threads" to s.threads.size, "messages" to s.messages.size, "has_older" to s.hasOlder,
        "leaf" to s.leafId, "editing" to (s.editingMessageId != null), "temporary" to s.newThreadTemporary,
        "draft_empty" to s.draft.isEmpty(), "quote" to s.quote.isNotEmpty(), "model" to s.model, "attachments" to s.attachments.size,
        "thinking" to s.enableThinking, "search" to s.enableSearch, "url_context" to s.enableUrlContext, "maps" to s.enableMaps,
        "file_creation" to s.enableFileCreation, "system_prompt" to s.enableSystemPrompt, "prompt_cache" to s.enablePromptCache,
        "batch" to s.batchMode, "python" to s.enablePython, "mcp" to s.enableMcp, "canvas" to s.canvasMode, "coding" to s.codingMode,
        "selects" to s.chipValues.toString(), "vision_model" to s.visionModel, "uploading" to s.uploading,
        "upload_completed" to s.uploadCompleted, "upload_count" to s.uploadCount, "streaming" to s.streaming, "job" to s.jobId,
        "retry" to s.retryAvailable, "status" to (s.status != null), "live_pending" to s.live.pendingStatus, "live_accepted" to s.live.accepted,
        "live_search" to s.live.search, "live_error" to s.live.error, "cards" to s.cards.size, "notice" to (s.notice != null),
        "offline" to s.offline, "connection" to s.connectionStatus.name, "connection_message" to (s.connectionMessage != null),
        "banned" to s.banned, "locked" to (s.accountLock != null), "mcp_decision" to (s.mcpDecision != null),
        "api_key_prompt" to s.apiKeyPrompt, "x_link_prompt" to s.xLinkPrompt, "mic" to s.micMode,
        "realtime" to s.realtime.active, "lyria" to s.lyria.kind, "library_busy" to s.libraryBusy,
        "library_failed" to s.libraryFailed, "skills_busy" to s.skillsBusy, "skill" to (s.selectedSkill != null),
        "prefs_busy" to s.prefsBusy, "mcp_busy" to s.mcpBusy, "feedback_busy" to s.feedbackBusy, "batch_busy" to s.batchBusy,
        "cache_syncing" to s.cacheSyncing, "thread_loading" to (s.threadLoadingId != null), "thread_load_failed" to (s.threadLoadFailedId != null),
        "navigation" to s.chatNavigationKind.name, "low_bandwidth" to s.lowBandwidthMode, "settings_request" to s.settingsRequest,
        "settings_tab" to s.settingsRequestTab, "slash_command" to s.pendingSlashCommand,
        "turnstile" to (s.sessionTurnstileUrl != null || s.authTurnstileUrl != null), "batch_banner" to (s.batchBanner != null),
    )
}

private const val DIAGNOSTICS_INTERVAL_MS = 10_000L
/** No diagnostics entry for this long while work runs: the thread stacks are recorded. */
private const val DIAGNOSTICS_STALL_MS = 20_000L
private const val DIAGNOSTICS_STACKS_GAP_MS = 60_000L
