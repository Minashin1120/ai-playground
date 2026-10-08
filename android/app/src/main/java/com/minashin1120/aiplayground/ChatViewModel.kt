package com.minashin1120.aiplayground

import android.app.Application
import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.net.Uri
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.os.Build
import android.os.SystemClock
import android.provider.OpenableColumns
import android.webkit.MimeTypeMap
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.minashin1120.aiplayground.data.*
import com.minashin1120.aiplayground.data.backend.ChatBackend
import com.minashin1120.aiplayground.data.backend.ServerChatBackend
import com.minashin1120.aiplayground.data.direct.DirectHttp
import com.minashin1120.aiplayground.data.direct.DirectRouter
import com.minashin1120.aiplayground.data.direct.TranscriptionDirect
import com.minashin1120.aiplayground.data.local.LocalSettingsStore
import com.minashin1120.aiplayground.data.feedbackSentText
import com.minashin1120.aiplayground.data.local.LocalChatBackend
import com.minashin1120.aiplayground.data.local.LocalChatStore
import com.minashin1120.aiplayground.data.local.LocalProfiles
import com.minashin1120.aiplayground.data.local.ServerHistory
import com.minashin1120.aiplayground.data.sync.SyncEngine
import com.minashin1120.aiplayground.data.sync.isRetryableSyncFailure
import com.minashin1120.aiplayground.data.sync.syncRetryDelayMillis
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.filterNotNull
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
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

enum class ChatTransitionKind { NONE, OPEN_THREAD, NEW_CHAT }

/** Account export categories imported by the first-run wizard. */
private const val ACCOUNT_IMPORT_CATEGORIES =
    "settings,api_credentials,chats,gems,files,feedback,diagnostics"

data class ImportSettingChange(val field: String, val current: String, val incoming: String)

data class ChatState(
    val starting: Boolean = true, val account: Account? = null,
    val pairing: Boolean = false, val userCode: String = "", val busy: Boolean = false,
    val authBusy: Boolean = false, val authError: String? = null,
    val authTwoFactorTransaction: String? = null, val setupRequired: Boolean = false,
    val auth2faMethod: String = "totp", val credentialRequest: CredentialRequest? = null,
    val googleLoginRequest: Long = 0L, val googleServerClientId: String = "",
    val integrityProjectNumber: String = "", val authTurnstileUrl: String? = null,
    /** The server the app signs in to (login screen "接続先"): label, its config, check in progress, error, recent list. */
    val serverLabel: String = originLabel(ServerOrigin.DEFAULT), val serverInfo: ServerInfo? = null,
    val serverChecking: Boolean = false, val serverError: String? = null, val savedServers: List<String> = emptyList(),
    /** No-account profile ("サーバーを使わずに始める"): everything runs and is stored on the device. */
    val localProfile: Boolean = false,
    /** Serverless mode of a signed-in account: answers come straight from the providers and are then saved to the account. */
    val serverless: Boolean = false,
    /** Serverless mode upload of finished answers: running, messages still on the device, last success (epoch ms), last result or error. */
    val syncing: Boolean = false, val pendingCount: Int = 0, val lastSyncAt: Long? = null, val syncMessage: String? = null,
    /** The no-account profile has chats that can be copied into this account. */
    val localImportAvailable: Boolean = false,
    /** The chat Turnstile check page (Web `#bot-detection-overlay`) while it is open in the browser. */
    val sessionTurnstileUrl: String? = null,
    /** Web `#batch-notification-banner`: (text, thread to open) after a Batch job finished. */
    val batchBanner: Pair<String, String>? = null,
    val googleAuthDiagnostics: String? = null,
    val security: SecurityInfo? = null, val securityBusy: Boolean = false, val securityError: String? = null,
    val securityTotpSecret: String? = null, val securityTotpUri: String? = null,
    /** `data:image/png;base64,…` QR of the pending TOTP secret (same image as Web). */
    val securityTotpQr: String? = null,
    val setupImportBusy: Boolean = false, val setupImportName: String = "", val setupImportProgress: Int = 0,
    val setupImportTotalChunks: Int = 0, val setupImportError: String? = null, val setupImportDone: Boolean = false,
    val setupImportPendingUploadId: String? = null,
    val setupImportSettingsChanges: List<ImportSettingChange> = emptyList(),
    val setupModels: List<ModelInfo> = emptyList(),
    val setupDefaultModel: String = "gemini-3.6-flash",
    val threads: List<ThreadItem> = emptyList(), val nextPage: Int? = null, val search: String = "",
    val selected: ThreadItem? = null, val canGoBackInChats: Boolean = false,
    val messages: List<ChatMessage> = emptyList(),
    val hasOlder: Boolean = false, val oldestId: String? = null,
    val allMessages: List<ChatMessage> = emptyList(), val leafId: Int? = null, val editingMessageId: String? = null,
    val customInstruction: String = "", val includeGlobalInstruction: Boolean = true,
    val newThreadTemporary: Boolean = false, val tempChatRemainingSeconds: Long? = null,
    /** The open chat's temporary-chat timeout (`timeout_seconds`); null falls back to the setting. */
    val tempChatTimeoutSeconds: Int? = null,
    val draft: String = "", val model: String = "", val attachments: List<Attachment> = emptyList(),
    val enableThinking: Boolean = false, val enableSearch: Boolean = false,
    val enableUrlContext: Boolean = false, val enableMaps: Boolean = false,
    val enableFileCreation: Boolean = true, val enableSystemPrompt: Boolean = false,
    /** Web `enable-sys-prompt.dataset.restoreChecked`: the model turned SysPrompt off, [sysPromptRestore] is the choice to bring back. */
    val sysPromptSuppressed: Boolean = false, val sysPromptRestore: Boolean = false,
    val enablePromptCache: Boolean = false,
    /** Web generation panel inputs, shared across models like the Web DOM (see `generationPanels`). */
    val generationValues: Map<String, String> = emptyMap(),
    /** Web composer selects that do not depend on the model (Thinking level/Budget, Effort, Safety). */
    val chipValues: Map<String, String> = COMPOSER_SELECT_DEFAULTS,
    val batchMode: Boolean = false, val enablePython: Boolean = false, val enableMcp: Boolean = true,
    val canvasMode: Boolean = false, val codingMode: Boolean = false,
    val codingTarget: CodingTarget? = null,
    val imageMask: String? = null,
    /** Web `currentVisionModel` changed from the upload sheet; null follows the Vision Model setting. */
    val visionModel: String? = null,
    /** The attachment whose edited image is uploading (row status "編集反映中..."). */
    val editingAttachment: String? = null,
    val uploading: Boolean = false, val streaming: Boolean = false,
    val uploadSent: Long = 0L, val uploadTotal: Long = 0L, val uploadName: String = "",
    /** Files finished / queued in the running upload batch (Web `Preparing... (completed/total)`). */
    val uploadCompleted: Int = 0, val uploadCount: Int = 0,
    val library: List<LibraryFile> = emptyList(), val libraryBusy: Boolean = false,
    val libraryQuery: String = "", val libraryFavoritesOnly: Boolean = false,
    /** Web `lib-sort` (newest / oldest / name_asc / name_desc), kept on the device like Web localStorage. */
    val librarySort: String = "newest",
    /** Web `#bot-lock-overlay`: the lock message and when it ends (epoch millis). */
    val accountLock: AccountLock? = null,
    /** Web `pendingSlashCommand`: the command whose argument the input now holds (コマンドモード). */
    val pendingSlashCommand: String? = null,
    /** Temporary `/settings` bubbles shown after the conversation (not saved to the thread). */
    val settingsBubbles: List<SettingsBubble> = emptyList(),
    /** Web `#api-key-required-modal`: the model whose key is missing. */
    val apiKeyPrompt: String? = null,
    /** Web `#auto-search-banner`: an X link was found and the user is asked whether to search. */
    val xLinkPrompt: Boolean = false,
    /** Web recording mic: "", "preparing" (録音準備中…), "recording" (録音中…) or "transcribing". */
    val micMode: String = "",
    /** The last 24 input levels (0–1) for the `#mic-waveform` bars. */
    val micLevels: List<Float> = emptyList(),
    /** The first page failed to load (Web grid error state). */
    val libraryFailed: Boolean = false,
    val libraryHasMore: Boolean = false, val libraryTotal: Int = 0,
    val gems: List<Gem> = emptyList(), val gemsBusy: Boolean = false, val selectedGem: Gem? = null,
    val preferences: Preferences? = null, val prefsBusy: Boolean = false,
    val compression: CompressionSettings = CompressionSettings(),
    val mcpServers: List<McpServerInfo> = emptyList(), val mcpBusy: Boolean = false,
    val feedbackItems: List<FeedbackItem> = emptyList(), val feedbackBusy: Boolean = false,
    val storage: StorageUsage? = null,
    val historyCacheMode: HistoryCacheMode = HistoryCacheMode.VIEWED,
    val cacheMobileDataAllowed: Boolean = false,
    val offlineCacheStats: OfflineCacheStats = OfflineCacheStats(),
    val cacheSyncing: Boolean = false,
    val cacheSyncProgress: Int = 0,
    val cacheSyncTotal: Int = 0,
    val batchJobs: List<BatchJob> = emptyList(), val batchBusy: Boolean = false,
    val realtime: RealtimeState = RealtimeState(), val lyria: LyriaState = LyriaState(),
    val liveContent: String = "", val liveThought: String = "", val status: String = "",
    val cards: List<StatusCard> = emptyList(),
    /** The streamed answer as the Web draws it (`LiveAnswer`): skeleton, search box, analysis, thought placeholder, error. */
    val live: LiveAnswer = LiveAnswer(),
    val mcpDecision: McpDecision? = null,
    val offline: Boolean = false,
    val connectionStatus: ConnectionStatus = ConnectionStatus.UNKNOWN,
    val connectionMessage: String = "",
    val connectionBannerVisible: Boolean = false,
    val banned: Boolean = false,
    val banReason: String = "",
    val banAt: String = "",
    val jobId: String? = null, val retryAvailable: Boolean = false, val notice: String? = null,
    val chatTransitionId: Long = 0L,
    val chatTransitionKind: ChatTransitionKind = ChatTransitionKind.NONE,
    /** Advances the moment a history/new-chat navigation starts, before its content loads. */
    val chatNavigationId: Long = 0L,
    val chatNavigationKind: ChatTransitionKind = ChatTransitionKind.NONE,
    /** Web `buildChatLoadingSkeletonHtml`: the transition id of the chat whose history is still loading (0 when none). */
    val threadLoadingId: Long = 0L,
    /** Web `showChatLoadError`: the opened chat whose history failed to load (its id), until it is opened again. */
    val threadLoadFailedId: String? = null,
    /** Web low-bandwidth mode: preference (auto/on/off), effective state and the detection reason. */
    val lowBandwidthPreference: String = "auto",
    val lowBandwidthMode: Boolean = false,
    val lowBandwidthReason: String = "",
    /** Web `currentQuote`: text quoted from a message, sent as `quote_text` with the next message. */
    val quote: String = "",
    /** Incremented to ask the screen to open the settings modal (e.g. from the encryption status dialog). */
    val settingsRequest: Long = 0L,
    /** Tab (id or label) and card key the latest [settingsRequest] opens at; null keeps the last tab. */
    val settingsRequestTab: String? = null,
    val settingsRequestCard: String? = null,
    /** Incremented to move the focus into the prompt input (Web `input.focus()` after edit / quote). */
    val composerFocusRequest: Long = 0L,
)

class ChatViewModel(application: Application) : AndroidViewModel(application) {
    private val api = PlaygroundApi()
    private val serverBackend = ServerChatBackend(api)
    /** Set while a no-account profile or serverless mode answers chat endpoints on the device. */
    private var localBackend: ChatBackend? = null
    private val backend: ChatBackend get() = localBackend ?: serverBackend
    private val localProfiles = LocalProfiles(application)
    /** Device store of the active profile while chats are local (no-account profile or serverless mode). */
    private var localChats: LocalChatStore? = null
    /** Settings and API keys of the device profile while chats are local (voice input uses its OpenAI key). */
    private var localSettings: LocalSettingsStore? = null
    private val directHttp by lazy { DirectHttp() }
    /** Separate client for background sync, so its requests never show the global spinner. */
    private val syncApi by lazy { PlaygroundApi() }
    private var syncJob: Job? = null
    private var syncScheduleJob: Job? = null
    /** Automatic retries in a row after syncs that could not reach the server (reset by a successful sync). */
    private var syncRetryAttempt = 0
    private val store = TokenStore(application)
    private val playIntegrity = PlayIntegrityClient(application)
    private var integrityTurnstileTicket: String? = null
    private val offlineCache = OfflineCacheStore(application)
    private val prefs = application.getSharedPreferences("navigation", 0)
    private val connectivity = application.getSystemService(ConnectivityManager::class.java)
    private val connectionProbeMutex = Mutex()
    private val mutable = MutableStateFlow(ChatState())
    val state = mutable.asStateFlow()
    private val chatHistory = ChatNavigationHistory()
    /** Web global spinner label (`progress_spinner.js`); null while no tracked request runs. */
    val progressLabel = api.progress.label

    init {
        // Web `apiFetch`: any request answered with `account_locked` shows the lock overlay (never for admins).
        api.onAccountLocked = { message, seconds ->
            mutable.update { current ->
                if (current.preferences?.isAdmin == true || current.accountLock != null) current
                else current.copy(accountLock = AccountLock(message, System.currentTimeMillis() + seconds.coerceAtLeast(0) * 1000))
            }
        }
        // Web keeps the library order and favorites filter in localStorage.
        mutable.update { it.copy(
            librarySort = prefs.getString(LIB_SORT_KEY, null)?.takeIf { value -> value in LIBRARY_SORTS } ?: "newest",
            libraryFavoritesOnly = prefs.getBoolean(LIB_FAVORITES_ONLY_KEY, false),
        ) }
        startDiagnostics()
        startActivityLog()
    }

    /** Separate client for diagnostics, so sending them never shows the global spinner. */
    private val diagnosticsApi by lazy { PlaygroundApi() }

    /**
     * Administrator accounts only ([Diagnostics]): follows the account's admin flag once its settings are
     * known, sends the recorded entries every 10 seconds, and records the app's thread stacks when an
     * answer, a sync or a chat load has not moved for 20 seconds.
     */
    private fun startDiagnostics() {
        viewModelScope.launch {
            state.map { current -> if (current.account == null || current.preferences == null) null
                else !current.localProfile && current.preferences.isAdmin }
                .filterNotNull().distinctUntilChanged().collect { admin -> Diagnostics.setEnabled(admin) }
        }
        viewModelScope.launch(Dispatchers.IO) {
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
                        "offline" to current.offline, "foreground" to foreground)
                }
                val token = session?.token ?: continue
                if (current.offline) continue
                val batch = Diagnostics.pending(200, 600_000)
                if (batch.lines == 0) continue
                try {
                    diagnosticsApi.post("/api/mobile/v1/diagnostics", JSONObject().put("entries", JSONArray(batch.entries)), token)
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
        viewModelScope.launch {
            state.map { current -> current.account?.let { if (current.localProfile) "local" else "${ServerOrigin.current}#${it.id}" } }
                .filterNotNull().distinctUntilChanged().collect { owner -> ActivityLog.setOwner(owner) }
        }
        viewModelScope.launch(Dispatchers.Default) {
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
        "auth_error" to s.authError, "setup_required" to s.setupRequired, "server_checking" to s.serverChecking,
        "server_error" to s.serverError, "syncing" to s.syncing, "pending_count" to s.pendingCount, "sync_message" to s.syncMessage,
        "thread" to s.selected?.id, "threads" to s.threads.size, "messages" to s.messages.size, "has_older" to s.hasOlder,
        "leaf" to s.leafId, "editing" to (s.editingMessageId != null), "temporary" to s.newThreadTemporary,
        "draft_empty" to s.draft.isEmpty(), "quote" to s.quote.isNotEmpty(), "model" to s.model, "attachments" to s.attachments.size,
        "thinking" to s.enableThinking, "search" to s.enableSearch, "url_context" to s.enableUrlContext, "maps" to s.enableMaps,
        "file_creation" to s.enableFileCreation, "system_prompt" to s.enableSystemPrompt, "prompt_cache" to s.enablePromptCache,
        "batch" to s.batchMode, "python" to s.enablePython, "mcp" to s.enableMcp, "canvas" to s.canvasMode, "coding" to s.codingMode,
        "selects" to s.chipValues.toString(), "vision_model" to s.visionModel, "uploading" to s.uploading,
        "upload_completed" to s.uploadCompleted, "upload_count" to s.uploadCount, "streaming" to s.streaming, "job" to s.jobId,
        "retry" to s.retryAvailable, "status" to s.status, "live_pending" to s.live.pendingStatus, "live_accepted" to s.live.accepted,
        "live_search" to s.live.search, "live_error" to s.live.error, "cards" to s.cards.size, "notice" to s.notice,
        "offline" to s.offline, "connection" to s.connectionStatus.name, "connection_message" to s.connectionMessage,
        "banned" to s.banned, "locked" to (s.accountLock != null), "mcp_decision" to (s.mcpDecision != null),
        "api_key_prompt" to s.apiKeyPrompt, "x_link_prompt" to s.xLinkPrompt, "mic" to s.micMode,
        "realtime" to s.realtime.active, "lyria" to s.lyria.kind, "library_busy" to s.libraryBusy,
        "library_failed" to s.libraryFailed, "gems_busy" to s.gemsBusy, "gem" to (s.selectedGem != null),
        "prefs_busy" to s.prefsBusy, "mcp_busy" to s.mcpBusy, "feedback_busy" to s.feedbackBusy, "batch_busy" to s.batchBusy,
        "cache_syncing" to s.cacheSyncing, "thread_loading" to s.threadLoadingId, "thread_load_failed" to s.threadLoadFailedId,
        "navigation" to s.chatNavigationKind.name, "low_bandwidth" to s.lowBandwidthMode, "settings_request" to s.settingsRequest,
        "settings_tab" to s.settingsRequestTab, "slash_command" to s.pendingSlashCommand,
        "turnstile" to (s.sessionTurnstileUrl != null || s.authTurnstileUrl != null), "batch_banner" to (s.batchBanner != null),
    )

    private var session: StoredSession? = null
    private var foreground = false
    private var backgroundedAt = 0L
    private var pairingJob: Job? = null
    private var navigationJob: Job? = null
    private var streamJob: Job? = null
    private var uploadJob: Job? = null
    private var heartbeatJob: Job? = null
    private var connectionMonitorJob: Job? = null
    private var connectionRecoveredHideJob: Job? = null
    private var slowConnectionCount = 0
    private var networkCallbackRegistered = false
    private var libraryJob: Job? = null
    private var cacheSyncJob: Job? = null
    private var batchPollJob: Job? = null
    private var realtimeStreamJob: Job? = null
    private var realtimeCaptureJob: Job? = null
    private var realtimeTrack: AudioTrack? = null
    private var lyriaStreamJob: Job? = null
    private var lyriaTrack: AudioTrack? = null
    private var importJob: Job? = null
    private var pendingPasskeyName = "Androidのパスキー"
    /** A send in flight; the composer state is restored from it when the server never accepted it. */
    private data class Submission(
        val body: JSONObject,
        val files: List<Attachment>,
        val mask: String? = null,
        val editingId: String? = null,
        val parentId: Int? = null,
    )
    private var failed: Submission? = null
    private var pendingParentId: Int? = null
    private var chatTransitionSequence = 0L

    /** Stops account work at the same BAN boundary enforced by the Web server. */
    private fun enterBannedState(error: ApiException) {
        listOf(pairingJob, navigationJob, streamJob, uploadJob, heartbeatJob, libraryJob, cacheSyncJob, batchPollJob,
            realtimeStreamJob, realtimeCaptureJob, lyriaStreamJob, importJob).forEach { it?.cancel() }
        realtimeTrack?.let { track -> runCatching { track.stop(); track.release() } }; realtimeTrack = null
        lyriaTrack?.let { track -> runCatching { track.stop(); track.release() } }; lyriaTrack = null
        failed = null
        mutable.update { current -> current.copy(
            banned = true,
            banReason = error.payload.optString("reason").ifBlank { error.payload.optString("message") },
            banAt = error.payload.optString("banned_at"),
            pairing = false,
            busy = false,
            uploading = false,
            streaming = false,
            realtime = RealtimeState(),
            lyria = LyriaState(),
            jobId = null,
            retryAvailable = false,
            mcpDecision = null,
            status = "",
        ) }
    }

    private fun nextChatTransition(kind: ChatTransitionKind): Pair<Long, ChatTransitionKind> {
        chatTransitionSequence += 1L
        return chatTransitionSequence to kind
    }

    private val networkCallback = object : ConnectivityManager.NetworkCallback() {
        override fun onAvailable(network: Network) {
            if (foreground) viewModelScope.launch { probeServerConnection() }
        }

        override fun onLost(network: Network) {
            if (!hasUsableNetwork()) setConnectionUnavailable(ConnectionStatus.OFFLINE)
        }

        override fun onCapabilitiesChanged(network: Network, networkCapabilities: NetworkCapabilities) {
            // Web recomputes on `navigator.connection` change and toasts when the mode flips in auto.
            viewModelScope.launch { recomputeLowBandwidth(notify = state.value.lowBandwidthPreference == "auto") }
        }
    }

    private fun lowBandwidthSignal(): LowBandwidthSignal? {
        val manager = connectivity ?: return null
        val capabilities = manager.getNetworkCapabilities(manager.activeNetwork ?: return null) ?: return null
        val kbps = capabilities.linkDownstreamBandwidthKbps
        val saveData = manager.restrictBackgroundStatus == ConnectivityManager.RESTRICT_BACKGROUND_STATUS_ENABLED
        return LowBandwidthSignal(saveData, effectiveConnectionType(kbps), roundedDownlinkMbps(kbps))
    }

    /** Web `recomputeLowBandwidthMode`; [notify] shows the toast only when the effective mode changes. */
    private fun recomputeLowBandwidth(notify: Boolean) {
        val detection = detectLowBandwidth(runCatching { lowBandwidthSignal() }.getOrNull())
        val current = state.value
        val active = effectiveLowBandwidth(current.lowBandwidthPreference, detection.enabled)
        val changed = active != current.lowBandwidthMode
        mutable.update { it.copy(lowBandwidthMode = active, lowBandwidthReason = detection.reason) }
        if (notify && changed) this.notify(lowBandwidthToast(active, current.lowBandwidthPreference, detection.reason))
    }

    /** Sidebar button: auto → on → off, persisted like Web's localStorage preference. */
    fun cycleLowBandwidth() {
        val next = nextLowBandwidthPreference(state.value.lowBandwidthPreference)
        prefs.edit().apply { if (next == "auto") remove(LOW_BANDWIDTH_PREF_KEY) else putString(LOW_BANDWIDTH_PREF_KEY, next) }.apply()
        mutable.update { it.copy(lowBandwidthPreference = next) }
        val detection = detectLowBandwidth(runCatching { lowBandwidthSignal() }.getOrNull())
        val active = effectiveLowBandwidth(next, detection.enabled)
        mutable.update { it.copy(lowBandwidthMode = active, lowBandwidthReason = detection.reason) }
        notify(lowBandwidthToast(active, next, detection.reason))
    }

    init { viewModelScope.launch {
        mutable.update { it.copy(compression = compressionSettingsFrom(prefs),
            lowBandwidthPreference = normalizeLowBandwidthPreference(prefs.getString(LOW_BANDWIDTH_PREF_KEY, "auto"))) }
        recomputeLowBandwidth(notify = false)
        mutable.update { it.copy(
            historyCacheMode = HistoryCacheMode.from(prefs.getString("offline_history_cache_mode", HistoryCacheMode.VIEWED.value)),
            cacheMobileDataAllowed = prefs.getBoolean("offline_cache_mobile_data", false),
        ) }
        restoreServerOrigin(withContext(Dispatchers.IO) { store.loadForOffline() }?.origin ?: prefs.getString(PREF_SERVER_ORIGIN, null))
        if (prefs.getString(PREF_PROFILE_MODE, PROFILE_SERVER) == PROFILE_LOCAL) {
            // The no-account profile never contacts the Playground server.
            enterLocalProfile()
            mutable.update { it.copy(starting = false) }
            return@launch
        }
        runCatching { api.get("/api/mobile/v1/config") }.getOrNull()?.let { config ->
            val info = parseServerInfo(ServerOrigin.current, config)
            if (info != null) applyServerInfo(info) else mutable.update { it.copy(
                googleServerClientId = config.optString("google_server_client_id"),
                integrityProjectNumber = config.optString("play_integrity_cloud_project_number"),
            ) }
        }
        session = withContext(Dispatchers.IO) { store.load() }
        if (session != null) {
            runCatching {
                loadAccount()
            }.onFailure { error ->
                if (error is ApiException && error.code == "banned") enterBannedState(error)
                else if (error is ApiException && error.code == "setup_required") loadSetup()
                else if (error is ApiException && error.status == 401) clearSession()
                if (error !is ApiException || error.code != "banned") {
                    if (!restoreOfflineAccount()) report(error)
                }
            }
        } else if (withContext(Dispatchers.IO) { store.loadForOffline() } != null) {
            restoreOfflineAccount()
        }
        refreshOfflineCacheStats()
        mutable.update { it.copy(starting = false) }
    } }
    private fun token(): String = session?.token ?: if (state.value.localProfile) "" else throw IOException("端末連携が必要です。")

    /** No-account profile: every chat is on the device, so offline checks for chat operations do not apply. */
    private val chatLocal: Boolean get() = localChats != null && state.value.localProfile

    /** Answers and attachments stay on the device (no-account profile or serverless mode), so they work offline. */
    private val uploadsLocal: Boolean get() = localChats != null

    /** A chat kept on the device: any chat of the no-account profile, a device chat of serverless mode. */
    private fun deviceChat(id: String?): Boolean =
        chatLocal || (state.value.serverless && id != null && LocalChatStore.isDeviceThreadId(id))

    /** Opens the device store of [profile]; [fallback] is the server for a signed-in account in serverless mode. */
    private fun useLocalChats(profile: LocalProfiles.Profile, accountName: String, fallback: ChatBackend?) {
        val defaults = localProfiles.defaults()
        localChats = profile.chats
        localSettings = profile.settings
        val remote = if (fallback == null) null else ServerHistory(fallback, { token() },
            serverFile = { reference, limit -> api.loadFileBytes(reference, token(), thumbnail = false, limit = limit) },
            cachedThread = { id -> state.value.account?.let { account -> offlineCache.loadThread(account.id, id) } },
            cachedFile = { reference, limit -> cachedFileBytes(reference, limit) },
            isOffline = { state.value.offline },
            afterAnswer = { syncAfterAnswer() })
        localBackend = LocalChatBackend(profile.chats, profile.settings, defaults, DirectRouter(directHttp), accountName, fallback,
            modeOf = { id -> state.value.account?.models?.firstOrNull { it.id == id }?.mode
                ?: defaults.json.optJSONArray("models")?.let { rows -> (0 until rows.length()).mapNotNull { rows.optJSONObject(it) }
                    .firstOrNull { it.optString("id") == id }?.optString("mode") } ?: "chat" },
            onChanged = { if (fallback != null) refreshPendingCount() },
            remote = remote,
            keepAlive = GenerationService.keepAlive(getApplication<Application>()))
    }

    /** Attachment bytes from the offline cache (serverless mode answering while the server is out of reach). */
    private fun cachedFileBytes(reference: String, limit: Long): ByteArray? {
        val accountId = state.value.account?.id ?: return null
        val target = File(getApplication<Application>().cacheDir, "serverless-" + java.util.UUID.randomUUID())
        return try {
            offlineCache.materializeFile(accountId, reference, target) ?: return null
            if (target.length() > limit) null else target.readBytes()
        } finally { target.delete() }
    }

    private fun closeLocalChats() {
        syncScheduleJob?.cancel(); syncJob?.cancel()
        localChats = null; localBackend = null; localSettings = null
    }

    /**
     * Serverless mode: a device chat that went up to the server is followed under its server id (the open
     * chat is switched to it). Any other id is returned as it is.
     */
    private suspend fun serverIdOf(id: String): String {
        val store = localChats?.takeIf { state.value.serverless && LocalChatStore.isDeviceThreadId(id) } ?: return id
        val serverId = withContext(Dispatchers.IO) { store.resolveAlias(id) }
        if (serverId != id) mutable.update { current ->
            if (current.selected?.id == id) current.copy(selected = current.selected?.copy(id = serverId)) else current
        }
        return serverId
    }

    private fun refreshPendingCount() {
        val store = localChats ?: return
        viewModelScope.launch {
            val count = withContext(Dispatchers.IO) { store.pendingCount() }
            mutable.update { it.copy(pendingCount = count) }
        }
    }

    /** Sends what serverless mode still keeps on the device, when the server can be reached. */
    private fun scheduleSync(delayMillis: Long = 0) {
        if (!state.value.serverless || session == null) return
        syncScheduleJob?.cancel()
        syncScheduleJob = viewModelScope.launch {
            delay(delayMillis)
            if (!hasUsableNetwork()) return@launch
            runSync(manual = false)
        }
    }

    /**
     * After an answer on the device: the outbox goes to the server, but the answer is finished without
     * waiting for a server that is out of reach. A sync still running after [AFTER_ANSWER_SYNC_WAIT_MS]
     * continues in the background; a failed one is tried again later ([scheduleSyncRetry]).
     */
    private suspend fun syncAfterAnswer() {
        withContext(Dispatchers.Main.immediate) {
            if (state.value.offline || !hasUsableNetwork()) {
                scheduleSyncRetry()
            } else {
                val job = viewModelScope.launch { syncJob?.join(); runSync(manual = false) }
                withTimeoutOrNull(AFTER_ANSWER_SYNC_WAIT_MS) { job.join() }
            }
        }
    }

    /** Tries the outbox again later (30 seconds, doubling up to 10 minutes) after a sync that could not reach the server. */
    private fun scheduleSyncRetry() {
        if (!state.value.serverless || session == null) return
        syncRetryAttempt++
        scheduleSync(syncRetryDelayMillis(syncRetryAttempt))
    }

    /** Settings "今すぐ同期". */
    fun syncNow() { viewModelScope.launch { runSync(manual = true) } }

    private suspend fun runSync(manual: Boolean) {
        val store = localChats ?: return
        // A cancelled sync still counts until it has really ended (a long file read finishes first), so
        // syncs never pile up reading the same attachments.
        if (!state.value.serverless || session == null || syncJob?.isCompleted == false) {
            Diagnostics.log("sync.skipped", "serverless" to state.value.serverless, "running" to (syncJob?.isCompleted == false))
            return
        }
        val job = viewModelScope.launch {
            mutable.update { it.copy(syncing = true, syncMessage = null) }
            val started = System.currentTimeMillis()
            Diagnostics.log("sync.start", "manual" to manual)
            try {
                val report = SyncEngine(store, syncApi) { token() }.run()
                Diagnostics.log("sync.done", "ms" to System.currentTimeMillis() - started, "uploaded" to report.uploadedMessages,
                    "skipped_attachments" to report.skippedAttachments, "rejected" to report.rejected, "pending" to report.pending)
                syncRetryAttempt = 0
                val now = System.currentTimeMillis()
                state.value.account?.let { account -> prefs.edit().putLong("last_sync_" + LocalProfiles.accountKey(account.id), now).apply() }
                val parts = listOfNotNull(
                    "送信 ${report.uploadedMessages}件",
                    report.skippedAttachments.takeIf { it > 0 }?.let { "サーバーへ送れなかった添付 ${it}件" },
                    report.rejected.takeIf { it > 0 }?.let { "次回に再送するメッセージ ${it}件" },
                )
                mutable.update { it.copy(syncing = false, lastSyncAt = now, pendingCount = report.pending, syncMessage = parts.joinToString("、")) }
                if (report.uploadedMessages > 0 && !state.value.streaming) {
                    runCatching { fetchThreads(false) }
                    state.value.selected?.id?.let { id -> if (!state.value.busy) runCatching { loadMessages(id, autoResume = false) } }
                }
            } catch (e: CancellationException) {
                Diagnostics.log("sync.cancelled", "ms" to System.currentTimeMillis() - started)
                mutable.update { it.copy(syncing = false) }; throw e
            }
            catch (e: Exception) {
                Diagnostics.failure("sync.error", e, "ms" to System.currentTimeMillis() - started)
                val message = when {
                    e is ApiException && e.code == "turnstile_required" -> "安全性の確認が必要です。送信欄から一度送信するか、しばらく待ってから同期してください。"
                    e is ApiException && e.code == "e2ee_migration_in_progress" -> "暗号化の切り替え中のため、完了後に同期します。"
                    e is ApiException && e.status == 401 -> "ログインの有効期限が切れました。"
                    else -> "同期できませんでした: " + (e.message ?: e.javaClass.simpleName) +
                        (if (isRetryableSyncFailure(e)) "（後で自動的に再試行します）" else "")
                }
                val pending = withContext(Dispatchers.IO) { runCatching { store.pendingCount() }.getOrDefault(state.value.pendingCount) }
                mutable.update { it.copy(syncing = false, pendingCount = pending, syncMessage = message) }
                if (isRetryableSyncFailure(e)) scheduleSyncRetry()
                if (manual && e is ApiException && e.code == "turnstile_required") startSessionTurnstile()
                if (e is ApiException && e.status == 401) report(e)
            }
        }
        syncJob = job
        job.join()
    }

    /** Settings "サーバーのAPIキーを端末へ取り込む": the account's own keys (after re-authentication) go to the device store. */
    fun importServerSecrets(secrets: JSONObject) {
        val account = state.value.account ?: return
        val settings = localProfiles.account(account.id).settings
        viewModelScope.launch {
            val count = withContext(Dispatchers.IO) {
                settings.importSecrets(secrets, overwrite = true)
                PROVIDER_KEY_FIELDS.count { secrets.optString(it).isNotBlank() } + (secrets.optJSONObject("model_api_keys")?.length() ?: 0)
            }
            notify(if (count > 0) "APIキーを${count}件、端末に取り込みました" else "サーバーに保存されたAPIキーはありません")
            runCatching { fetchPreferences() }
        }
    }

    /** Settings: copies the no-account profile's chats into this account (uploaded by the next sync). */
    fun importLocalProfileChats() {
        val account = state.value.account ?: return
        viewModelScope.launch {
            val count = withContext(Dispatchers.IO) { localProfiles.account(account.id).chats.importFrom(localProfiles.local().chats) }
            notify(if (count > 0) "端末のチャットを${count}件取り込みました" else "取り込むチャットはありません")
            runCatching { fetchThreads(false) }
            refreshPendingCount()
            scheduleSync()
        }
    }

    /** Login screen "サーバーを使わずに始める": a device-only profile, no network to the Playground server. */
    fun startLocalProfile() {
        if (state.value.authBusy) return
        viewModelScope.launch {
            prefs.edit().putString(PREF_PROFILE_MODE, PROFILE_LOCAL).apply()
            enterLocalProfile()
        }
    }

    private suspend fun enterLocalProfile() {
        chatHistory.clear()
        useLocalChats(localProfiles.local(), LOCAL_PROFILE_NAME, fallback = null)
        mutable.update { it.copy(localProfile = true, serverless = false, canGoBackInChats = false,
            offline = false, connectionBannerVisible = false,
            connectionStatus = ConnectionStatus.ONLINE, authError = null) }
        runCatching { loadAccount() }.onFailure { report(it) }
    }

    /** Leaves the no-account profile for the login screen; the device chats stay for a later upload. */
    fun leaveLocalProfile() { viewModelScope.launch {
        streamJob?.cancel(); navigationJob?.cancel()
        chatHistory.clear()
        prefs.edit().putString(PREF_PROFILE_MODE, PROFILE_SERVER).apply()
        closeLocalChats()
        val server = state.value
        mutable.value = ChatState(starting = false, serverLabel = server.serverLabel, serverInfo = server.serverInfo,
            savedServers = server.savedServers, googleServerClientId = server.googleServerClientId,
            integrityProjectNumber = server.integrityProjectNumber, historyCacheMode = server.historyCacheMode,
            cacheMobileDataAllowed = server.cacheMobileDataAllowed)
        runCatching { api.get("/api/mobile/v1/config") }.getOrNull()?.let { config ->
            parseServerInfo(ServerOrigin.current, config)?.let(::applyServerInfo)
        }
    } }

    /** Settings "接続" card: serverless mode of the signed-in account (answers from the providers, chats on the device). */
    fun setServerless(enabled: Boolean) {
        val account = state.value.account ?: return
        if (state.value.localProfile || state.value.streaming) return
        viewModelScope.launch {
            prefs.edit().putBoolean(serverlessPrefKey(account.id), enabled).apply()
            applyServerlessMode(account)
            newChat()
            runCatching { fetchThreads(false) }.onFailure { report(it) }
            notify(if (enabled) "サーバー不使用モードをオンにしました" else "サーバー不使用モードをオフにしました")
        }
    }

    private fun serverlessPrefKey(accountId: Int) = "serverless_" + LocalProfiles.accountKey(accountId)

    /** Applies the saved serverless choice of [account] (called after the account loads). */
    private fun applyServerlessMode(account: Account) {
        if (state.value.localProfile) return
        val enabled = prefs.getBoolean(serverlessPrefKey(account.id), false)
        if (enabled) useLocalChats(localProfiles.account(account.id), account.name, fallback = serverBackend) else closeLocalChats()
        val key = LocalProfiles.accountKey(account.id)
        mutable.update { it.copy(serverless = enabled, pendingCount = 0,
            lastSyncAt = prefs.getLong("last_sync_$key", 0L).takeIf { value -> value > 0 }, syncMessage = null,
            localImportAvailable = enabled && localProfiles.hasLocalData()) }
        if (enabled) {
            // Stores of 1.38.0–1.40.x held a copy of every chat; only what the server lacks is kept.
            localChats?.let { store -> viewModelScope.launch {
                withContext(Dispatchers.IO) { runCatching { store.repairSyncIdentities(); store.migrateToOutbox() } }
                refreshPendingCount()
                scheduleSync()
            } }
        }
    }

    /** Stores an attachment on the device while chats are local; null means upload it to the server. */
    private fun storeLocalUpload(name: String, mime: String, size: Long, opener: () -> java.io.InputStream): String? {
        val store = localChats ?: return null
        val started = System.currentTimeMillis()
        return opener().use { input -> store.saveFile(name, mime, input, size) }.also {
            Diagnostics.log("attach.stored", "mime" to mime, "bytes" to size, "ms" to System.currentTimeMillis() - started)
        }
    }

    /** Points the app at the server saved with the session (or chosen last on the login screen). */
    private fun restoreServerOrigin(saved: String?) {
        saved?.toHttpUrlOrNull()?.let { parseServerOrigin(it.toString()) }?.let(ServerOrigin::set)
        mutable.update { it.copy(serverLabel = originLabel(ServerOrigin.current), savedServers = savedServers()) }
    }

    private fun savedServers(): List<String> =
        prefs.getString(PREF_SAVED_SERVERS, null).orEmpty().split('\n').filter { it.isNotBlank() }

    private fun applyServerInfo(info: ServerInfo) {
        mutable.update { it.copy(
            serverInfo = info, serverLabel = originLabel(ServerOrigin.current),
            googleServerClientId = info.googleServerClientId, integrityProjectNumber = info.integrityProjectNumber,
        ) }
    }

    /** Login screen "接続先": checks that [input] is an AI Playground server before switching to it. */
    fun selectServer(input: String) {
        if (state.value.serverChecking || state.value.authBusy) return
        val origin = parseServerOrigin(input) ?: run {
            mutable.update { it.copy(serverError = "https で接続できるサーバーのアドレス（例: ai.example.com）を入力してください。") }
            return
        }
        viewModelScope.launch {
            mutable.update { it.copy(serverChecking = true, serverError = null) }
            try {
                val config = PlaygroundApi(origin).get("/api/mobile/v1/config")
                val info = parseServerInfo(origin, config) ?: throw IOException("AI Playground のサーバーとして確認できませんでした。")
                ServerOrigin.set(origin)
                val recent = (listOf(originLabel(origin)) + savedServers()).distinct().take(5)
                prefs.edit().putString(PREF_SERVER_ORIGIN, origin.toString()).putString(PREF_SAVED_SERVERS, recent.joinToString("\n")).apply()
                integrityTurnstileTicket = null
                applyServerInfo(info)
                mutable.update { it.copy(serverChecking = false, savedServers = recent, authError = null,
                    authTwoFactorTransaction = null, authTurnstileUrl = null) }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                mutable.update { it.copy(serverChecking = false,
                    serverError = "接続先を確認できませんでした。" + (e.message?.let { message -> "（$message）" } ?: "")) }
            }
        }
    }
    private fun deviceName(): String = "${Build.MANUFACTURER} ${Build.MODEL}".trim().take(80)

    private suspend fun authPost(path: String, body: JSONObject): JSONObject {
        val request = JSONObject(body.toString())
        request.put("integrity_enabled", true)
        val ticket = integrityTurnstileTicket
        if (ticket != null) {
            request.put("integrity_turnstile_ticket", ticket)
            integrityTurnstileTicket = null
        } else if (state.value.integrityProjectNumber.isNotBlank()) {
            // Play Integrity is only verified by the official server (fixed package and service account).
            val fields = playIntegrity.requestFields(path, state.value.integrityProjectNumber, request)
            val keys = fields.keys()
            while (keys.hasNext()) {
                val key = keys.next()
                request.put(key, fields.get(key))
            }
        }
        return try {
            api.post(path, request)
        } catch (error: ApiException) {
            if (error.code == "turnstile_required") {
                val url = error.payload.optString("turnstile_url")
                if (url.startsWith("/android/integrity/turnstile?")) {
                    mutable.update { it.copy(authTurnstileUrl = withAppReturn(url)) }
                }
            }
            throw error
        }
    }

    /** Opens the browser Turnstile check for this signed-in account (server `mobile_security_turnstile`). */
    fun startSessionTurnstile() { viewModelScope.launch {
        try {
            val reply = backend.post("/api/mobile/v1/security/turnstile", JSONObject(), token())
            val url = reply.optString("turnstile_url")
            if (url.startsWith("/android/integrity/turnstile?")) mutable.update { it.copy(sessionTurnstileUrl = withAppReturn(url)) }
            else notify("安全性の確認を完了しました。もう一度送信してください。")
        } catch (e: CancellationException) { throw e }
        catch (e: Exception) { notify("安全性の確認を完了できませんでした。しばらく待ってから再送信してください。") }
    } }

    /** The browser check came back with a ticket that only this account can redeem. */
    fun completeSessionTurnstile(ticket: String) {
        if (ticket.length !in 20..128) return
        viewModelScope.launch {
            val ok = runCatching {
                backend.post("/api/mobile/v1/security/turnstile/complete", JSONObject().put("ticket", ticket), token())
            }.isSuccess
            mutable.update { it.copy(sessionTurnstileUrl = null) }
            notify(if (ok) "安全性の確認を完了しました。もう一度送信してください。"
                else "安全性の確認を完了できませんでした。しばらく待ってから再送信してください。")
        }
    }

    fun dismissSessionTurnstile() { mutable.update { it.copy(sessionTurnstileUrl = null) } }

    fun integrityTurnstileComplete(ticket: String) {
        if (ticket.length !in 20..128) return
        integrityTurnstileTicket = ticket
        mutable.update { it.copy(authTurnstileUrl = null, authError = "安全性を確認しました。認証をもう一度実行してください。") }
    }

    private suspend fun acceptAuthResponse(reply: JSONObject) {
        val accessToken = reply.optString("access_token")
        require(accessToken.isNotBlank()) { "認証トークンを取得できませんでした。" }
        session = StoredSession(accessToken, System.currentTimeMillis() + reply.optLong("expires_in", 2_592_000L) * 1000L,
            ServerOrigin.current.toString())
        withContext(Dispatchers.IO) { store.save(requireNotNull(session)) }
        mutable.update { it.copy(
            authBusy = false, authError = null, authTwoFactorTransaction = null,
            googleAuthDiagnostics = null,
            auth2faMethod = "totp", credentialRequest = null,
        ) }
        if (reply.optBoolean("setup_required")) loadSetup() else loadAccount()
    }

    private suspend fun processAuthResponse(reply: JSONObject) {
        if (reply.optString("status") == "2fa_required") {
            mutable.update { it.copy(
                authBusy = false,
                authTwoFactorTransaction = reply.getString("transaction_id"),
                auth2faMethod = reply.optString("default_method", "totp").ifBlank { "totp" },
                authError = null,
            ) }
        } else {
            acceptAuthResponse(reply)
        }
    }

    fun login(username: String, password: String) {
        authenticate("/api/mobile/v1/auth/login", username, password)
    }

    fun signup(username: String, password: String) {
        authenticate("/api/mobile/v1/auth/signup", username, password)
    }

    private fun authenticate(path: String, username: String, password: String) {
        if (state.value.authBusy) return
        viewModelScope.launch {
            mutable.update { it.copy(
                authBusy = true,
                authError = null,
                googleAuthDiagnostics = null,
                authTwoFactorTransaction = null,
            ) }
            try {
                val reply = authPost(path, JSONObject()
                    .put("username", username.trim())
                    .put("password", password)
                    .put("device_name", deviceName()))
                processAuthResponse(reply)
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                mutable.update { it.copy(authBusy = false, authError = e.message ?: "認証に失敗しました。") }
            }
        }
    }

    /** Starts Android Credential Manager's native Google account picker. */
    fun beginGoogleLogin() {
        if (state.value.authBusy) return
        if (state.value.googleServerClientId.isBlank()) {
            mutable.update { it.copy(authError = "Googleログインを設定できません。時間をおいて再試行してください。") }
            return
        }
        mutable.update { it.copy(
            authBusy = true,
            authError = null,
            googleAuthDiagnostics = null,
            authTwoFactorTransaction = null,
            googleLoginRequest = it.googleLoginRequest + 1L,
        ) }
    }

    fun completeGoogleLogin(credential: GoogleAuthClient.Result) {
        if (!state.value.authBusy) return
        viewModelScope.launch {
            try {
                processAuthResponse(authPost("/api/mobile/v1/auth/google", JSONObject()
                    .put("id_token", credential.idToken)
                    .put("nonce", credential.nonce)
                    .put("device_name", deviceName())))
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                mutable.update { it.copy(authBusy = false, authError = e.message ?: "Googleログインに失敗しました。") }
            }
        }
    }

    fun cancelGoogleLogin(
        message: String = "Googleログインをキャンセルしました。",
        diagnostics: String? = null,
    ) {
        if (state.value.authBusy) mutable.update {
            it.copy(authBusy = false, authError = message, googleAuthDiagnostics = diagnostics)
        }
    }

    fun verifyTotp(code: String) {
        val transaction = state.value.authTwoFactorTransaction ?: return
        if (state.value.authBusy) return
        viewModelScope.launch {
            mutable.update { it.copy(authBusy = true, authError = null) }
            try {
                acceptAuthResponse(authPost("/api/mobile/v1/auth/totp", JSONObject()
                    .put("transaction_id", transaction)
                    .put("code", code)
                    .put("device_name", deviceName())))
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                mutable.update { it.copy(authBusy = false, authError = e.message ?: "2段階認証に失敗しました。") }
            }
        }
    }

    /** Starts a browser login whose returned code only this app can redeem (PKCE). */
    fun browserLoginPath(provider: String): String {
        require(provider == "google" || provider == "minashin") { "Unsupported browser login" }
        val verifier = BrowserLoginPkce.newVerifier()
        prefs.edit().putString(PREF_BROWSER_LOGIN_VERIFIER, verifier)
            .putLong(PREF_BROWSER_LOGIN_STARTED_AT, System.currentTimeMillis()).apply()
        return withAppReturn("/android/auth/$provider/start?code_challenge_method=S256&code_challenge=" +
            BrowserLoginPkce.challenge(verifier))
    }

    /** Exchanges the one-time code returned to the verified HTTPS App Link. */
    fun exchangeNativeCode(code: String) {
        if (state.value.authBusy) return
        val verifier = prefs.getString(PREF_BROWSER_LOGIN_VERIFIER, null)
        val elapsed = System.currentTimeMillis() - prefs.getLong(PREF_BROWSER_LOGIN_STARTED_AT, 0L)
        prefs.edit().remove(PREF_BROWSER_LOGIN_VERIFIER).remove(PREF_BROWSER_LOGIN_STARTED_AT).apply()
        if (verifier == null || elapsed !in 0..BROWSER_LOGIN_TTL_MS) {
            // A link that this app did not start may carry someone else's account.
            notify("このアプリで開始していないログインのため、処理しませんでした。")
            return
        }
        viewModelScope.launch {
            mutable.update { it.copy(authBusy = true, authError = null) }
            try {
                val reply = authPost("/api/mobile/v1/auth/exchange", JSONObject()
                    .put("code", code)
                    .put("code_verifier", verifier))
                processAuthResponse(reply)
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                mutable.update { it.copy(authBusy = false, authError = e.message ?: "外部ログインに失敗しました。") }
            }
        }
    }

    /** Starts a passkey sign-in and hands the WebAuthn options to the UI. */
    fun beginPasskeyLogin(username: String) {
        if (state.value.authBusy) return
        viewModelScope.launch {
            mutable.update { it.copy(authBusy = true, authError = null, credentialRequest = null) }
            try {
                val reply = authPost("/api/mobile/v1/auth/passkey/options", JSONObject()
                    .put("username", username.trim())
                    .put("device_name", deviceName()))
                mutable.update { it.copy(
                    authBusy = false,
                    credentialRequest = CredentialRequest(
                        "login", reply.getString("transaction_id"), reply.getJSONObject("public_key").toString()),
                ) }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                mutable.update { it.copy(authBusy = false, authError = e.message ?: "パスキーを開始できませんでした。") }
            }
        }
    }

    /** Starts a WebAuthn second-factor ceremony for the pending 2FA transaction. */
    fun beginWebauthnTwoFactor() {
        val transaction = state.value.authTwoFactorTransaction ?: return
        if (state.value.authBusy) return
        viewModelScope.launch {
            mutable.update { it.copy(authBusy = true, authError = null, credentialRequest = null) }
            try {
                val reply = authPost("/api/mobile/v1/auth/2fa/webauthn/options",
                    JSONObject().put("transaction_id", transaction))
                mutable.update { it.copy(
                    authBusy = false,
                    credentialRequest = CredentialRequest(
                        "2fa", transaction, reply.getJSONObject("public_key").toString()),
                ) }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                mutable.update { it.copy(authBusy = false, authError = e.message ?: "パスキーを開始できませんでした。") }
            }
        }
    }

    /** Starts passkey registration from the security settings. */
    fun beginPasskeyRegistration(name: String = "") {
        if (state.value.securityBusy) return
        // Web sends the typed name; the server falls back to `Passkey N` when it is blank.
        pendingPasskeyName = name.trim().take(80)
        viewModelScope.launch {
            mutable.update { it.copy(securityBusy = true, securityError = null, credentialRequest = null) }
            try {
                val reply = backend.post("/api/mobile/v1/security/passkeys/options", JSONObject(), token())
                mutable.update { it.copy(
                    securityBusy = false,
                    credentialRequest = CredentialRequest(
                        "register", "", reply.getJSONObject("public_key").toString()),
                ) }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                mutable.update { it.copy(securityBusy = false, securityError = e.message ?: "パスキー登録を開始できませんでした。") }
            }
        }
    }

    /** Completes whichever WebAuthn ceremony the UI just ran through Credential Manager. */
    fun submitCredential(responseJson: String) {
        val request = state.value.credentialRequest ?: return
        viewModelScope.launch {
            try {
                val credential = JSONObject(responseJson)
                when (request.kind) {
                    "login" -> acceptAuthResponse(authPost("/api/mobile/v1/auth/passkey/verify", JSONObject()
                        .put("transaction_id", request.transactionId)
                        .put("credential", credential)))
                    "2fa" -> acceptAuthResponse(authPost("/api/mobile/v1/auth/2fa/webauthn/verify", JSONObject()
                        .put("transaction_id", request.transactionId)
                        .put("credential", credential)))
                    "register" -> {
                        val reply = backend.post("/api/mobile/v1/security/passkeys/verify", JSONObject()
                            .put("credential", credential)
                            .put("name", pendingPasskeyName), token())
                        mutable.update { it.copy(
                            securityBusy = false, securityError = null, credentialRequest = null,
                            security = parseSecurityInfo(reply),
                        ) }
                    }
                }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                mutable.update { it.copy(
                    authBusy = false, securityBusy = false, credentialRequest = null,
                    authError = e.message ?: "パスキー認証に失敗しました。",
                    securityError = e.message ?: "パスキー認証に失敗しました。",
                ) }
            }
        }
    }

    /** Clears a cancelled or failed Credential Manager ceremony. */
    fun cancelCredentialRequest(message: String? = null) {
        mutable.update { it.copy(
            credentialRequest = null, authBusy = false, securityBusy = false,
            authError = message ?: it.authError, securityError = message ?: it.securityError,
        ) }
    }

    fun loadSecurity() { viewModelScope.launch {
        try {
            val reply = backend.get("/api/mobile/v1/security", token())
            mutable.update { it.copy(security = parseSecurityInfo(reply), securityError = null) }
        } catch (e: CancellationException) { throw e }
        catch (e: Exception) { mutable.update { it.copy(securityError = e.message ?: "セキュリティ設定を取得できませんでした。") } }
    } }

    fun startTotpSetup() { viewModelScope.launch {
        mutable.update { it.copy(securityBusy = true, securityError = null) }
        try {
            val reply = backend.post("/api/mobile/v1/security/totp/setup", JSONObject(), token())
            mutable.update { it.copy(
                securityBusy = false,
                securityTotpSecret = reply.getString("secret"),
                securityTotpUri = reply.optString("otpauth_uri"),
                securityTotpQr = reply.optString("qr_image").ifBlank { null },
            ) }
        } catch (e: CancellationException) { throw e }
        catch (e: Exception) { mutable.update { it.copy(securityBusy = false, securityError = e.message ?: "TOTPを開始できませんでした。") } }
    } }

    fun cancelTotpSetup() { mutable.update { it.copy(securityTotpSecret = null, securityTotpUri = null) } }

    fun enableTotp(code: String) = securityAction {
        backend.post("/api/mobile/v1/security/totp/enable", JSONObject().put("code", code), token())
    }

    fun disableTotp(code: String) = securityAction {
        backend.post("/api/mobile/v1/security/totp/disable", JSONObject().put("code", code), token())
    }

    fun removePasskey(id: String) = securityAction {
        backend.post("/api/mobile/v1/security/passkeys/remove", JSONObject().put("id", id), token())
    }

    fun saveSecurityPreferences(default2fa: String, passkeyOnly: Boolean, skipGoogle: Boolean) = securityAction {
        backend.post("/api/mobile/v1/security/preferences", JSONObject()
            .put("default_2fa_method", default2fa)
            .put("passkey_only_login", passkeyOnly)
            .put("skip_2fa_on_google_login", skipGoogle), token())
    }

    private fun securityAction(block: suspend () -> JSONObject) {
        if (state.value.securityBusy) return
        viewModelScope.launch {
            mutable.update { it.copy(securityBusy = true, securityError = null) }
            try {
                val reply = block()
                mutable.update { it.copy(
                    securityBusy = false, security = parseSecurityInfo(reply),
                    securityTotpSecret = null, securityTotpUri = null,
                ) }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) { mutable.update { it.copy(securityBusy = false, securityError = e.message ?: "セキュリティ設定を保存できませんでした。") } }
        }
    }

    /** Uploads and imports an account ZIP in a resumable, cancellable chunk session. */
    fun importAccountZip(uri: Uri) {
        if (state.value.setupImportBusy) return
        importJob = viewModelScope.launch {
            mutable.update { it.copy(
                setupImportBusy = true, setupImportError = null, setupImportDone = false, setupImportProgress = 0,
                setupImportPendingUploadId = null, setupImportSettingsChanges = emptyList()) }
            var uploadId: String? = null
            try {
                val resolver = getApplication<Application>().contentResolver
                val resolved = resolver.query(uri, null, null, null, null)?.use { cursor ->
                    val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    val sizeIndex = cursor.getColumnIndex(OpenableColumns.SIZE)
                    if (cursor.moveToFirst()) {
                        cursor.getString(nameIndex).orEmpty().ifBlank { "account.zip" } to cursor.getLong(sizeIndex)
                    } else null
                } ?: throw IOException("ファイルを開けません。")
                val (name, size) = resolved
                require(size > 0) { "ファイルサイズを取得できません。" }
                mutable.update { it.copy(setupImportName = name) }
                val start = api.accountImportStart(size, token())
                val id = start.getString("upload_id")
                uploadId = id
                val chunkSize = start.getInt("chunk_size")
                val total = start.getInt("total_chunks")
                mutable.update { it.copy(setupImportTotalChunks = total) }
                resolver.openInputStream(uri)?.use { input ->
                    var received = 0L
                    for (index in 0 until total) {
                        val readSize = minOf(chunkSize.toLong(), size - received).toInt()
                        val buffer = ByteArray(readSize)
                        var offset = 0
                        while (offset < readSize) {
                            val count = input.read(buffer, offset, readSize - offset)
                            if (count < 0) throw IOException("ファイルの読み込みが中断されました。")
                            offset += count
                        }
                        received += readSize
                        api.accountImportChunk(id, index, buffer, token())
                        mutable.update { it.copy(setupImportProgress = index + 1) }
                    }
                } ?: throw IOException("ファイルを開けません。")
                api.accountImportComplete(id, token())
                val result = api.accountImport(id, ACCOUNT_IMPORT_CATEGORIES, token())
                if (result.optString("status") == "settings_confirmation") {
                    val changes = parseImportSettingChanges(result)
                    if (changes.isNotEmpty()) {
                        uploadId = null
                        mutable.update { it.copy(
                            setupImportBusy = false,
                            setupImportPendingUploadId = id,
                            setupImportSettingsChanges = changes,
                        ) }
                        return@launch
                    }
                }
                uploadId = null
                mutable.update { it.copy(setupImportBusy = false, setupImportDone = true) }
            } catch (e: CancellationException) {
                uploadId?.let { id -> withContext(NonCancellable) { runCatching { api.accountImportCancel(id, token()) } } }
                throw e
            } catch (e: Exception) {
                uploadId?.let { id -> runCatching { api.accountImportCancel(id, token()) } }
                mutable.update { it.copy(setupImportBusy = false, setupImportError = e.message ?: "インポートに失敗しました。") }
            }
        }
    }

    /** Applies the uploaded archive after the user confirms the settings changes. */
    fun confirmSetupImportSettings() {
        val uploadId = state.value.setupImportPendingUploadId ?: return
        if (state.value.setupImportBusy) return
        importJob = viewModelScope.launch {
            mutable.update { it.copy(setupImportBusy = true, setupImportError = null) }
            try {
                api.accountImport(uploadId, ACCOUNT_IMPORT_CATEGORIES, token(), confirmSettings = true)
                mutable.update { it.copy(
                    setupImportBusy = false,
                    setupImportDone = true,
                    setupImportPendingUploadId = null,
                    setupImportSettingsChanges = emptyList(),
                ) }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                mutable.update { it.copy(
                    setupImportBusy = false,
                    setupImportError = e.message ?: "インポートに失敗しました。",
                ) }
            }
        }
    }

    /** Rejects the settings overwrite and removes the completed upload session. */
    fun cancelSetupImportConfirmation() {
        val uploadId = state.value.setupImportPendingUploadId ?: return
        mutable.update { it.copy(
            setupImportPendingUploadId = null,
            setupImportSettingsChanges = emptyList(),
            setupImportError = null,
        ) }
        viewModelScope.launch {
            runCatching { api.accountImportCancel(uploadId, token()) }
        }
    }

    fun cancelSetupImport() {
        val pendingUploadId = state.value.setupImportPendingUploadId
        importJob?.cancel()
        mutable.update { it.copy(
            setupImportBusy = false, setupImportProgress = 0, setupImportError = null,
            setupImportPendingUploadId = null, setupImportSettingsChanges = emptyList(),
        ) }
        if (pendingUploadId != null) {
            viewModelScope.launch { runCatching { api.accountImportCancel(pendingUploadId, token()) } }
        }
    }

    private fun parseImportSettingChanges(reply: JSONObject): List<ImportSettingChange> {
        val rows = reply.optJSONArray("settings_changes") ?: return emptyList()
        fun display(value: Any?): String = when {
            value == null || value == JSONObject.NULL -> "未設定"
            value is JSONObject || value is JSONArray -> value.toString()
            else -> value.toString()
        }.let { value -> if (value.length > 2_000) value.take(2_000) + "…" else value }
        return buildList {
            for (index in 0 until rows.length()) {
                val row = rows.optJSONObject(index) ?: continue
                val field = row.optString("field").trim()
                if (field.isBlank()) continue
                add(ImportSettingChange(
                    field = field,
                    current = display(row.opt("current")),
                    incoming = display(row.opt("incoming")),
                ))
            }
        }
    }

    private suspend fun loadSetup() {
        chatHistory.clear()
        val reply = backend.get("/api/mobile/v1/setup", token())
        val models = parseModels(reply)
        val defaultModel = reply.optString("default_model", "gemini-3.6-flash")
        mutable.update { it.copy(
            account = null,
            canGoBackInChats = false,
            setupRequired = true,
            setupModels = models,
            setupDefaultModel = defaultModel,
            pairing = false,
            userCode = "",
            authBusy = false,
            authError = null,
        ) }
    }

    fun finishSetup(
        defaultModel: String,
        openaiKey: String,
        geminiKey: String,
        anthropicKey: String,
        deepseekKey: String,
        kimiKey: String,
        mistralKey: String,
        ideogramKey: String,
        zaiKey: String,
        xaiKey: String,
        googleKey: String,
        googleProject: String,
        vertexProject: String,
        vertexLocation: String,
        vertexCredentialsJson: String,
        enableE2ee: Boolean,
    ) {
        if (state.value.authBusy) return
        viewModelScope.launch {
            mutable.update { it.copy(authBusy = true, authError = null) }
            try {
                val reply = backend.put("/api/mobile/v1/setup", JSONObject()
                    .put("default_model", defaultModel)
                    .put("openai_api_key", openaiKey)
                    .put("gemini_api_key", geminiKey)
                    .put("anthropic_api_key", anthropicKey)
                    .put("deepseek_api_key", deepseekKey)
                    .put("kimi_api_key", kimiKey)
                    .put("mistral_api_key", mistralKey)
                    .put("ideogram_api_key", ideogramKey)
                    .put("zai_api_key", zaiKey)
                    .put("xai_api_key", xaiKey)
                    .put("google_api_key", googleKey)
                    .put("google_cloud_project", googleProject)
                    .put("gemini_vertex_project", vertexProject)
                    .put("gemini_vertex_location", vertexLocation)
                    .put("gemini_vertex_credentials_json", vertexCredentialsJson)
                    .put("enable_e2ee", enableE2ee), token())
                require(!reply.optBoolean("setup_required", true)) { "初回設定を完了できませんでした。" }
                mutable.update { it.copy(authBusy = false, setupRequired = false) }
                loadAccount()
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                mutable.update { it.copy(authBusy = false, authError = e.message ?: "初回設定に失敗しました。") }
            }
        }
    }
    private suspend fun loadAccount() {
        val me = backend.get("/api/mobile/v1/me", token())
        val serverModels = parseModels(me)
        val displayModels = displayModels(serverModels)
        val account = Account(me.getInt("id"), me.getString("username"), displayModels,
            me.optString("default_model"), me.optBoolean("e2ee_enabled"))
        if (state.value.account?.id != account.id) chatHistory.clear()
        val chosen = prefs.getString("model_${account.id}", account.defaultModel).orEmpty()
            .takeIf { chosen -> account.models.any { it.id == chosen && it.selectable } }
            ?: account.models.firstOrNull { it.selectable }?.id.orEmpty()
        if (!state.value.localProfile) {
            withContext(Dispatchers.IO) { offlineCache.saveAccount(account.id, me) }
            prefs.edit().putString("offline_cache_account_id", account.id.toString()).apply()
            markConnectionReachable()
            applyServerlessMode(account)
        }
        mutable.update { it.copy(
            account = account, model = chosen, pairing = false, userCode = "", offline = false,
            setupRequired = false, authBusy = false, authError = null,
            canGoBackInChats = chatHistory.canGoBack,
        ) }
        fetchThreads(false)
        runCatching { fetchGems() }.onFailure { report(it) }
        runCatching { fetchPreferences(applyDefaults = true) }.onFailure { report(it) }
        runCatching { fetchBatchJobs(notify = false) }.onFailure { report(it) }
        // The composer shows the MCP chip only when a server is enabled (Web `applyMcpPromptChipUi`).
        loadMcpServers()
        if (state.value.localProfile) return
        if (foreground) startBatchPolling()
        if (state.value.historyCacheMode == HistoryCacheMode.FULL) startCacheSyncIfAllowed()
    }

    private suspend fun displayModels(serverModels: List<ModelInfo>): List<ModelInfo> = runCatching {
        val json = withContext(Dispatchers.IO) {
            getApplication<Application>().assets.open("web-model-catalog.json").bufferedReader().use { it.readText() }
        }
        applyWebModelCatalog(serverModels, json)
    }.getOrDefault(serverModels)

    /** Restores the last account and locally stored data when the server cannot be reached. */
    private suspend fun restoreOfflineAccount(): Boolean {
        val accountId = prefs.getString("offline_cache_account_id", null)?.toIntOrNull() ?: return false
        val me = withContext(Dispatchers.IO) { offlineCache.loadAccount(accountId) } ?: return false
        val account = Account(
            me.optInt("id", accountId), me.optString("username", "この端末のアカウント"),
            displayModels(parseModels(me)), me.optString("default_model"), me.optBoolean("e2ee_enabled"),
        )
        if (state.value.account?.id != account.id) chatHistory.clear()
        val selected = prefs.getString("model_${account.id}", account.defaultModel).orEmpty()
            .takeIf { value -> account.models.any { it.id == value && it.selectable } }
            ?: account.models.firstOrNull { it.selectable }?.id.orEmpty()
        val cachedPrefs = withContext(Dispatchers.IO) { offlineCache.loadPreferences(account.id) }
        val cachedThreads = withContext(Dispatchers.IO) { offlineCache.loadThreads(account.id) }
        mutable.update { current -> current.copy(
            account = account,
            canGoBackInChats = chatHistory.canGoBack,
            model = selected,
            preferences = cachedPrefs?.let { parsePreferences(it) } ?: current.preferences,
            threads = cachedThreads,
            nextPage = null,
            offline = true,
            connectionStatus = ConnectionStatus.OFFLINE,
            connectionMessage = ConnectionStatus.OFFLINE.defaultMessage(),
            connectionBannerVisible = true,
            pairing = false,
        ) }
        refreshOfflineCacheStats(account.id)
        return true
    }
    /**
     * An answer is being generated on the device (it has no server job to rejoin). Cancelling it drops the
     * provider request and loses the result, so leaving the app or coming back must not cancel it.
     */
    private fun answeringOnDevice(): Boolean = isDeviceAnswer(state.value.streaming, uploadsLocal, state.value.jobId)

    fun setForeground(value: Boolean) {
        if (value != foreground) Diagnostics.log("app.foreground", "value" to value, "streaming" to state.value.streaming,
            "syncing" to state.value.syncing)
        val returning = value && !foreground
        foreground = value
        if (!value) backgroundedAt = SystemClock.elapsedRealtime()
        // Web keeps reading the answer while the tab is hidden. Android can cut an idle connection without
        // closing it after a while in the background, so after a long absence an answer the server took
        // (it has a job id) is rejoined instead. A send the server has not taken yet keeps retrying.
        if (returning && state.value.streaming && !answeringOnDevice() && state.value.jobId != null &&
            SystemClock.elapsedRealtime() - backgroundedAt >= STREAM_REJOIN_AFTER_MS) {
            Diagnostics.log("send.rejoin", "away_ms" to SystemClock.elapsedRealtime() - backgroundedAt)
            streamJob?.cancel()
            mutable.update { it.copy(streaming = false, status = "") }
        }
        if (!value && state.value.realtime.active) stopRealtime(save = false)
        if (!value && state.value.lyria.active) stopLyria(save = false)
        if (!value) heartbeatJob?.cancel()
        if (!value) batchPollJob?.cancel()
        if (value && !state.value.localProfile) startConnectionMonitor() else stopConnectionMonitor()
        if (returning) recomputeLowBandwidth(notify = false)
        if (value && state.value.account != null && !state.value.localProfile) startBatchPolling()
        if (returning && state.value.account != null && state.value.selected != null && !state.value.busy && !state.value.streaming) refreshInPlace()
        if (returning && state.value.account != null) startCacheSyncIfAllowed()
        if (returning && state.value.serverless) scheduleSync()
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
                val grant = backend.post("/api/mobile/v1/device", JSONObject().put("client_id", "official-android")
                    .put("device_name", "${Build.MANUFACTURER} ${Build.MODEL}".take(80)))
                mutable.update { it.copy(userCode = grant.getString("user_code")) }
                val deadline = SystemClock.elapsedRealtime() + grant.getLong("expires_in") * 1000
                var interval = grant.optLong("interval", 5).coerceAtLeast(5)
                while (SystemClock.elapsedRealtime() < deadline) {
                    delay(interval * 1000)
                    if (!foreground) continue
                    try {
                        val reply = backend.post("/api/mobile/v1/token", JSONObject().put("client_id", "official-android")
                            .put("device_code", grant.getString("device_code")))
                        val linked = StoredSession(reply.getString("access_token"), System.currentTimeMillis() + reply.getLong("expires_in") * 1000,
                            ServerOrigin.current.toString())
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

    private fun hasUsableNetwork(): Boolean {
        val manager = connectivity ?: return false
        val active = manager.activeNetwork ?: return false
        val capabilities = manager.getNetworkCapabilities(active) ?: return false
        return capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) &&
            capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED)
    }

    private fun isConnectionOperationActive(): Boolean {
        val current = state.value
        return current.pairing || current.busy || current.streaming || current.uploading ||
            current.realtime.active || current.lyria.active
    }

    private fun startConnectionMonitor() {
        if (!networkCallbackRegistered) {
            runCatching {
                connectivity?.registerDefaultNetworkCallback(networkCallback)
                networkCallbackRegistered = connectivity != null
            }
        }
        if (connectionMonitorJob?.isActive == true) return
        connectionMonitorJob = viewModelScope.launch {
            probeServerConnection()
            while (isActive && foreground) {
                delay(state.value.connectionStatus.probeIntervalMillis())
                probeServerConnection()
            }
        }
    }

    private fun stopConnectionMonitor() {
        connectionMonitorJob?.cancel()
        connectionMonitorJob = null
        if (networkCallbackRegistered) {
            runCatching { connectivity?.unregisterNetworkCallback(networkCallback) }
            networkCallbackRegistered = false
        }
    }

    private suspend fun probeServerConnection() = connectionProbeMutex.withLock {
        if (!foreground || isConnectionOperationActive()) return@withLock
        if (!hasUsableNetwork()) {
            setConnectionUnavailable(ConnectionStatus.OFFLINE)
            return@withLock
        }
        val startedAt = SystemClock.elapsedRealtime()
        try {
            val reply = withTimeout(3_000L) {
                backend.get("/api/version?heartbeat=${System.currentTimeMillis()}")
            }
            val latencyMs = SystemClock.elapsedRealtime() - startedAt
            val wasDisconnected = state.value.connectionStatus.isDisconnected()
            slowConnectionCount = if (latencyMs >= 2_000L) slowConnectionCount + 1 else 0
            if (slowConnectionCount >= 3) {
                setConnectionUnavailable(ConnectionStatus.UNSTABLE, "サーバーとの通信が不安定です（遅延 ${latencyMs}ms）")
            } else {
                markConnectionReachable(wasDisconnected)
            }
            // Parsing also ensures a proxy's non-JSON response is not accepted as a heartbeat.
            reply.optString("version")
        } catch (e: TimeoutCancellationException) {
            setConnectionUnavailable(ConnectionStatus.OFFLINE)
        } catch (e: CancellationException) {
            throw e
        } catch (e: ApiException) {
            val mode = connectionStatusForHttp(e.status)
            if (mode != null) setConnectionUnavailable(mode)
            else setConnectionUnavailable(ConnectionStatus.UNSTABLE, "サーバーでエラーが発生しています（HTTP ${e.status}）")
        } catch (ignored: Exception) {
            setConnectionUnavailable(ConnectionStatus.OFFLINE)
        }
    }

    private fun setConnectionUnavailable(requested: ConnectionStatus, requestedMessage: String = requested.defaultMessage()) {
        if (state.value.localProfile) return
        // Web `setUnavailable`: a failed request only shows this server is unreachable; the Internet-offline
        // wording is kept for a device without a network.
        val siteOnly = requested == ConnectionStatus.OFFLINE && hasUsableNetwork()
        val status = if (siteOnly) ConnectionStatus.UNSTABLE else requested
        val message = if (siteOnly) "このサイトへの通信が完了しませんでした。接続を再確認しています" else requestedMessage
        slowConnectionCount = 0
        connectionRecoveredHideJob?.cancel()
        connectionRecoveredHideJob = null
        mutable.update { it.copy(
            connectionStatus = status,
            connectionMessage = message,
            connectionBannerVisible = true,
            offline = status.isDisconnected(),
        ) }
    }

    private fun markConnectionReachable(wasDisconnected: Boolean = state.value.connectionStatus.isDisconnected()) {
        slowConnectionCount = 0
        connectionRecoveredHideJob?.cancel()
        mutable.update { it.copy(
            connectionStatus = ConnectionStatus.ONLINE,
            connectionMessage = if (wasDisconnected) ConnectionStatus.ONLINE.defaultMessage() else "",
            connectionBannerVisible = wasDisconnected,
            offline = false,
        ) }
        // Serverless mode: answers finished while the server was out of reach go up now.
        if (wasDisconnected && state.value.serverless) scheduleSync()
        if (wasDisconnected) {
            connectionRecoveredHideJob = viewModelScope.launch {
                delay(5_000L)
                mutable.update { current ->
                    if (current.connectionStatus == ConnectionStatus.ONLINE) current.copy(connectionBannerVisible = false)
                    else current
                }
            }
        }
    }

    override fun onCleared() {
        stopConnectionMonitor()
        connectionRecoveredHideJob?.cancel()
        super.onCleared()
    }

    fun draft(text: String) { mutable.update { it.copy(draft = text) } }
    fun quoteMessage(text: String) {
        val quoted = text.trim()
        if (quoted.isBlank()) return
        mutable.update { it.copy(quote = quoted, composerFocusRequest = it.composerFocusRequest + 1) }
    }
    fun clearQuote() { mutable.update { it.copy(quote = "") } }
    /**
     * Web `applySavedUserSystemPromptSettings`: after the user system prompt is saved from Settings or
     * Chat Instructions, the composer's SysPrompt switch follows it when its text or on/off changed.
     */
    private fun ChatState.withSavedPreferences(saved: Preferences): ChatState {
        fun active(p: Preferences?) = p != null && p.systemPrompt.isNotBlank() && p.systemPromptEnabled
        val before = preferences
        val next = copy(preferences = saved)
        val changed = before == null || before.systemPrompt != saved.systemPrompt || before.systemPromptEnabled != saved.systemPromptEnabled
        if (!changed) return next
        val on = active(saved)
        return if (sysPromptSuppressed) next.copy(sysPromptRestore = on) else next.copy(enableSystemPrompt = on)
    }
    /**
     * Web `toggleOptions()` writes the forced checkbox values and moves the Thinking level / Effort
     * selects to an allowed option; the new values stay after switching to another model.
     */
    private fun ChatState.withModelRules(): ChatState {
        val rules = composerRules(model, mcpServers.any { it.enabled })
        val level = chipValues["thinking_level"].orEmpty()
        val effort = chipValues["reasoning_effort"].orEmpty()
        val nextLevel = if (level !in rules.thinkingLevels && rules.thinkingFallback != null) rules.thinkingFallback else level
        val nextEffort = if (rules.effort.visible && effort !in rules.effortOptions) rules.effortFallback else effort
        // Web `toggleOptions()` remembers the SysPrompt choice while a model cannot use it and brings it back afterwards.
        val sysForced = rules.sysPrompt.forced
        val sysSuppressed = sysForced != null
        val sysChecked = when {
            sysForced != null -> sysForced
            sysPromptSuppressed && sysPromptRestore -> true
            else -> enableSystemPrompt
        }
        val sysRestore = sysSuppressed && (if (sysPromptSuppressed) sysPromptRestore else enableSystemPrompt)
        return copy(
            enableSearch = rules.search.forced ?: enableSearch,
            enableUrlContext = rules.urls.forced ?: enableUrlContext,
            enableMaps = rules.maps.forced ?: enableMaps,
            enablePython = rules.python.forced ?: enablePython,
            enableSystemPrompt = sysChecked,
            sysPromptSuppressed = sysSuppressed,
            sysPromptRestore = sysRestore,
            enableThinking = rules.thinking.forced ?: enableThinking,
            enablePromptCache = rules.promptCache.forced ?: enablePromptCache,
            batchMode = batchMode && rules.batch.visible,
            chipValues = chipValues + mapOf("thinking_level" to nextLevel, "reasoning_effort" to nextEffort),
        )
    }

    /**
     * Web keeps each option's checkbox when the model changes; only the per-model rules force values
     * (applied when sending). OCR turns Canvas/Coding off and non GPT-Image models drop the mask, as on Web.
     */
    fun chooseModel(model: String) {
        val info = state.value.account?.models?.firstOrNull { it.id == model && it.selectable } ?: return
        if (state.value.enablePromptCache) {
            val currentProvider = modelApiProvider(state.value.model)
            val nextProvider = modelApiProvider(info.id)
            if (currentProvider != null && nextProvider != null && currentProvider != nextProvider) {
                notify("PromptCache 有効中は他API（${PROVIDER_LABELS[nextProvider] ?: nextProvider}）のモデルに変更できません。現在: ${PROVIDER_LABELS[currentProvider] ?: currentProvider}")
                return
            }
        }
        val ocr = isMistralOcrModel(info.id)
        mutable.update { it.copy(model = model,
            canvasMode = it.canvasMode && !ocr,
            codingMode = it.codingMode && !ocr,
            codingTarget = if (ocr) null else it.codingTarget,
            imageMask = if (info.id.contains("gpt-image")) it.imageMask else null).withModelRules() }
        state.value.account?.let { prefs.edit().putString("model_${it.id}", model).apply() }
    }
    fun toggleThinking() { mutable.update { it.copy(enableThinking = !it.enableThinking) } }
    fun toggleSearch() { mutable.update { it.copy(enableSearch = !it.enableSearch) } }
    fun toggleUrlContext() { mutable.update { it.copy(enableUrlContext = !it.enableUrlContext) } }
    fun toggleMaps() { mutable.update { it.copy(enableMaps = !it.enableMaps) } }
    fun toggleFileCreation() { mutable.update { it.copy(enableFileCreation = !it.enableFileCreation) } }
    fun toggleSystemPrompt() { mutable.update { it.copy(enableSystemPrompt = !it.enableSystemPrompt) } }
    fun togglePromptCache() {
        mutable.update { it.copy(enablePromptCache = !it.enablePromptCache) }
        if (state.value.enablePromptCache) {
            val provider = modelApiProvider(state.value.model)
            val label = PROVIDER_LABELS[provider] ?: provider ?: "現在のAPI"
            notify("PromptCache を有効化しました。以降は $label 以外のモデルに変更できません。")
        }
    }
    /** Web `enable-batch-mode` change: turning Batch on releases Coding, which Batch cannot run. */
    fun toggleBatchMode() {
        val releaseCoding = !state.value.batchMode && state.value.codingMode
        mutable.update { it.copy(batchMode = !it.batchMode, codingMode = if (releaseCoding) false else it.codingMode) }
        if (releaseCoding) notify("Batch APIではCoding Modeを利用できないため解除しました")
    }
    fun togglePython() { mutable.update { it.copy(enablePython = !it.enablePython) } }
    fun toggleMcp() { mutable.update { it.copy(enableMcp = !it.enableMcp) } }
    fun toggleCanvas() { mutable.update { it.copy(canvasMode = !it.canvasMode) } }
    fun toggleCoding() { mutable.update { it.copy(codingMode = !it.codingMode) } }
    /** Web `applyTemporaryChatSetting`: an open chat sends only `is_temporary` (its title and instruction stay). */
    fun toggleTemporaryChat() {
        val selected = state.value.selected
        if (selected == null) {
            mutable.update { it.copy(newThreadTemporary = !it.newThreadTemporary) }
            return
        }
        val next = !selected.isTemporary
        viewModelScope.launch {
            if (state.value.offline && !deviceChat(selected.id)) { notify("オフライン中はチャット設定を変更できません。"); return@launch }
            try {
                val reply = backend.put("/api/threads/${selected.id}/settings", JSONObject().put("is_temporary", next), token())
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
            } catch (e: Exception) { report(e, "一時チャット設定の更新に失敗しました") }
        }
    }
    /** Web `selectCodingTargetFromButton` / `clear-coding-target-btn`; selecting does not turn Coding on. */
    fun selectCodingTarget(target: CodingTarget?) {
        mutable.update { it.copy(codingTarget = target) }
        notify(when {
            target == null -> "最新のコードブロックを自動選択します"
            state.value.codingMode -> "Coding Modeの編集対象に設定しました"
            else -> "編集対象を選択しました。プロンプトバーのCodingをオンにすると使用します"
        })
    }
    fun setImageMask(reference: String?) { mutable.update { it.copy(imageMask = reference) } }
    /** Web `uploadMaskFile`: the chosen mask image is uploaded as it is. */
    fun uploadImageMask(uri: Uri) {
        if (state.value.banned || (state.value.offline && !uploadsLocal)) return
        viewModelScope.launch {
            try {
                val resolver = getApplication<Application>().contentResolver
                val local = withContext(Dispatchers.IO) { queryLocalAttachment(resolver, uri) }
                val bytes = withContext(Dispatchers.IO) { resolver.openInputStream(uri)?.use { it.readBytes() } }
                    ?: throw IOException("Mask upload failed")
                val mime = local.mime.ifBlank { "image/png" }
                val uploaded = withContext(Dispatchers.IO) { storeLocalUpload(local.name, mime, bytes.size.toLong()) { bytes.inputStream() } }
                    ?: api.upload(local.name, bytes.toRequestBody(mime.toMediaType()), token()).getString("filename")
                setImageMask(uploaded)
            } catch (e: CancellationException) { throw e }
            catch (e: ApiException) { notify(e.payload.optString("error").ifBlank { "Mask upload failed" }) }
            catch (e: Exception) { notify("Mask upload failed") }
        }
    }
    /** Web `selectModel` while the vision picker is active. */
    fun setVisionModel(id: String) { mutable.update { it.copy(visionModel = id) } }
    /** Web `promptRowAttachmentName`: a blank name restores the default. */
    fun renameAttachment(reference: String, input: String) {
        val next = input.trim()
        mutable.update { current -> current.copy(attachments = current.attachments.map {
            if (it.reference != reference) it else it.copy(name = next.ifEmpty { it.defaultName })
        }) }
        notify(if (next.isEmpty()) "送信名をデフォルトに戻しました" else "送信名を更新しました")
    }
    /**
     * Web `saveMarkerToRow`: uploads the edited PNG as `<name>_marked.png` and swaps it into the row,
     * keeping the first pre-edit file as the row's original.
     */
    fun applyImageEdit(reference: String, png: ByteArray, attachOriginal: Boolean) {
        val row = state.value.attachments.firstOrNull { it.reference == reference } ?: return
        if (state.value.offline && !uploadsLocal) { notify("オフライン中はファイルをアップロードできません。"); return }
        val fileName = markedFileName(row.name.ifBlank { "marked.png" })
        mutable.update { current -> current.copy(editingAttachment = reference, attachments = current.attachments.map {
            if (it.reference == reference) it.copy(attachOriginal = attachOriginal) else it
        }) }
        viewModelScope.launch {
            try {
                val uploaded = withContext(Dispatchers.IO) { storeLocalUpload(fileName, "image/png", png.size.toLong()) { png.inputStream() } }
                    ?: api.upload(fileName, png.toRequestBody("image/png".toMediaType()), token()).getString("filename")
                val original = row.original ?: row.copy(original = null, attachOriginal = false, edited = false)
                mutable.update { current -> current.copy(attachments = current.attachments.map {
                    if (it.reference != reference) it
                    else Attachment(fileName, uploaded, "image/png", "upload", fileName, original, attachOriginal, edited = true)
                }) }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) { notify("編集画像のアップロードに失敗しました") }
            finally { mutable.update { it.copy(editingAttachment = null) } }
        }
    }
    fun generationOption(key: String, value: String) {
        if (key in COMPOSER_SELECT_DEFAULTS) {
            mutable.update { it.copy(chipValues = it.chipValues + (key to value)) }
            return
        }
        if (state.value.streaming) return
        mutable.update { current -> current.copy(generationValues = current.generationValues + (key to value)) }
    }
    fun removeAttachment(reference: String) { mutable.update { it.copy(attachments = it.attachments.filterNot { a -> a.reference == reference }) } }
    fun search(query: String) {
        mutable.update { it.copy(search = query) }
        navigationJob?.cancel()
        navigationJob = viewModelScope.launch {
            delay(300)
            try {
                if (state.value.offline && !chatLocal) applyCachedThreads(query) else fetchThreads(false)
            } catch (e: Exception) { report(e) }
        }
    }
    private suspend fun fetchThreads(more: Boolean) {
        if (state.value.offline && !chatLocal) {
            applyCachedThreads(state.value.search)
            return
        }
        val current = state.value
        val page = if (more) current.nextPage ?: return else 1
        val reply = try {
            backend.get("/api/threads?page=$page&q=${URLEncoder.encode(current.search, "UTF-8")}", token())
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            // Serverless mode keeps working without the server: the cached list and the device chats stand in.
            if (!current.serverless || more || !ServerHistory.unreachable(e)) throw e
            applyCachedThreads(current.search)
            return
        }
        if (current.search != state.value.search) return
        val rows = reply.getJSONArray("threads")
        val items = (0 until rows.length()).map { parseThreadItem(rows.getJSONObject(it)) }
        if (!chatLocal) state.value.account?.let { account ->
            withContext(Dispatchers.IO) { offlineCache.saveThreads(account.id, items.filterNot { LocalChatStore.isDeviceThreadId(it.id) }) }
            refreshOfflineCacheStats(account.id)
        }
        mutable.update { it.copy(threads = if (more) (it.threads + items).distinctBy { t -> t.id } else items,
            nextPage = if (reply.optBoolean("has_next")) reply.optInt("next_page", page + 1) else null,
            offline = false) }
    }
    private suspend fun applyCachedThreads(query: String) {
        val accountId = state.value.account?.id ?: return
        val cached = withContext(Dispatchers.IO) { offlineCache.loadThreads(accountId) }
        val normalized = query.trim()
        val visible = if (normalized.isBlank()) cached else cached.filter {
            it.title.contains(normalized, ignoreCase = true) || it.model.contains(normalized, ignoreCase = true)
        }
        // Serverless mode: chats created on the device while the server was out of reach come first.
        val device = localChats?.takeIf { state.value.serverless }?.let { store ->
            val rows = withContext(Dispatchers.IO) { store.deviceThreads(normalized) }
            (0 until rows.length()).map { parseThreadItem(rows.getJSONObject(it)) }
        }.orEmpty()
        mutable.update { it.copy(threads = device + visible, nextPage = null, offline = true) }
    }
    fun moreThreads() {
        if (state.value.offline && !chatLocal) return
        navigationJob?.cancel(); navigationJob = viewModelScope.launch { try { fetchThreads(true) } catch (e: Exception) { report(e) } }
    }
    private fun currentChatLocation() = ChatLocation(state.value.selected, state.value.newThreadTemporary)

    private fun recordChatNavigation(next: ChatLocation) {
        chatHistory.record(currentChatLocation(), next)
        mutable.update { it.copy(canGoBackInChats = chatHistory.canGoBack) }
    }

    fun goBackInChats() {
        val previous = chatHistory.pop() ?: return
        mutable.update { it.copy(canGoBackInChats = chatHistory.canGoBack) }
        if (previous.thread == null) newChat(previous.temporary, recordHistory = false)
        else openThread(previous.thread, recordHistory = false)
    }

    fun newChat(temporary: Boolean = false, recordHistory: Boolean = true) {
        if (recordHistory) recordChatNavigation(ChatLocation(null, temporary))
        navigationJob?.cancel(); streamJob?.cancel(); failed = null
        heartbeatJob?.cancel()
        pendingParentId = null
        val transition = nextChatTransition(ChatTransitionKind.NEW_CHAT)
        mutable.update { it.copy(selected = null, messages = emptyList(), allMessages = emptyList(), leafId = null, settingsBubbles = emptyList(),
            editingMessageId = null, jobId = null, streaming = false,
            liveContent = "", liveThought = "", status = "", busy = false, retryAvailable = false,
            cards = emptyList(), hasOlder = false, oldestId = null, customInstruction = "",
            includeGlobalInstruction = true, newThreadTemporary = temporary, tempChatRemainingSeconds = null, tempChatTimeoutSeconds = null,
            selectedGem = null, codingTarget = null, imageMask = null,
            // Web `startNewChat`: cancelEdit + resetUploadState, and PromptCache starts off.
            draft = "", attachments = emptyList(), quote = "", enablePromptCache = false,
            chatTransitionId = transition.first, chatTransitionKind = transition.second,
            chatNavigationId = transition.first, chatNavigationKind = transition.second) }
    }
    /**
     * Web `loadMessages`: the composer is cleared (`cancelEdit`), the branch pinned in ブランチ管理 or else the
     * latest one is shown. The chat transition starts at once and the loading skeleton stays until the history arrives.
     */
    fun openThread(thread: ThreadItem, recordHistory: Boolean = true) {
        if (recordHistory) recordChatNavigation(ChatLocation(thread))
        navigationJob?.cancel(); streamJob?.cancel(); heartbeatJob?.cancel(); failed = null
        pendingParentId = null
        val transition = nextChatTransition(ChatTransitionKind.OPEN_THREAD)
        val pinnedLeaf = getApplication<Application>().getSharedPreferences("branches", Context.MODE_PRIVATE)
            .getString("fixed_branch_${thread.id}", null)?.toIntOrNull()
        mutable.update { it.copy(selected = thread, messages = emptyList(), allMessages = emptyList(), settingsBubbles = emptyList(),
            leafId = pinnedLeaf, editingMessageId = null, streaming = false, busy = true,
            draft = "", attachments = emptyList(), quote = "",
            liveContent = "", liveThought = "", jobId = null, retryAvailable = false,
            cards = emptyList(), hasOlder = false, oldestId = null,
            // Web `playChatTransition('history')` + `buildChatLoadingSkeletonHtml`: swap to the new chat now
            // and show the skeleton there; the messages replace it in place when they arrive.
            chatTransitionId = transition.first, chatTransitionKind = transition.second,
            threadLoadingId = transition.first, threadLoadFailedId = null,
            chatNavigationId = transition.first, chatNavigationKind = transition.second) }
        navigationJob = viewModelScope.launch {
            val started = System.currentTimeMillis()
            Diagnostics.log("thread.open", "thread" to thread.id)
            try {
                loadMessages(thread.id)
                Diagnostics.log("thread.opened", "thread" to thread.id, "ms" to System.currentTimeMillis() - started,
                    "messages" to state.value.messages.size)
                mutable.update { it.copy(busy = if (it.selected?.id == thread.id) false else it.busy,
                    threadLoadingId = if (it.threadLoadingId == transition.first) 0L else it.threadLoadingId) }
                if (foreground && state.value.jobId != null) resume()
            }
            catch (e: CancellationException) {
                Diagnostics.log("thread.open_cancelled", "thread" to thread.id, "ms" to System.currentTimeMillis() - started)
                throw e
            }
            catch (e: Exception) {
                Diagnostics.failure("thread.open_error", e, "thread" to thread.id, "ms" to System.currentTimeMillis() - started)
                if (state.value.threadLoadingId == transition.first) mutable.update { it.copy(threadLoadFailedId = thread.id) }
                report(e, "チャットの読み込みに失敗しました")
            }
            finally {
                mutable.update { it.copy(busy = false,
                    threadLoadingId = if (it.threadLoadingId == transition.first) 0L else it.threadLoadingId) }
            }
        }
    }
    /** Web `showChatLoadError` 再試行: loads the chat that failed to open again. */
    fun retryOpenThread() {
        state.value.selected?.let { openThread(it, recordHistory = false) }
    }
    /**
     * [inPlace]: the open chat is updated without replacing what is on screen. Messages loaded with
     * 過去メッセージを読み込む stay, the chosen model and Gem are kept, and a failed request or a send
     * that started meanwhile leaves the chat as it is.
     */
    private suspend fun loadMessages(requestedId: String, older: Boolean = false, autoResume: Boolean = true, inPlace: Boolean = false) {
        val id = serverIdOf(requestedId)
        if (inPlace && state.value.offline && !deviceChat(id)) return
        if (state.value.offline && !deviceChat(id)) {
            loadCachedMessages(id, older)
            return
        }
        val before = if (older) "&before_id=${state.value.oldestId ?: return}" else ""
        val low = state.value.lowBandwidthMode
        val limit = when {
            older && low -> LOW_BANDWIDTH_OLDER_PAGE_SIZE
            older -> THREAD_OLDER_PAGE_SIZE
            low -> LOW_BANDWIDTH_INITIAL_MESSAGE_LIMIT
            else -> THREAD_INITIAL_MESSAGE_LIMIT
        }
        val reply = try {
            Diagnostics.log("thread.fetch", "thread" to id, "older" to older)
            backend.get("/api/threads/$id?limit=$limit$before", token())
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            val canUseCache = if (e is ApiException) e.status >= 500 else e is IOException
            if (!canUseCache) throw e
            if (inPlace) return
            val accountId = state.value.account?.id ?: throw e
            val cached = withContext(Dispatchers.IO) { offlineCache.loadThread(accountId, id) } ?: throw e
            setConnectionUnavailable(
                if (e is ApiException) ConnectionStatus.SERVER_DOWN else ConnectionStatus.OFFLINE,
            )
            loadCachedMessages(id, older, cached)
            return
        }
        Diagnostics.log("thread.fetched", "thread" to id, "rows" to (reply.optJSONArray("messages")?.length() ?: 0),
            "pending_job" to !reply.isNull("pending_job"))
        if (state.value.selected?.id != id) return
        if (inPlace && (state.value.streaming || state.value.busy)) return
        val parsed = parseMessages(reply)
        val parsedOldest = parsed.mapNotNull { numericId(it) }.filterNot { LocalChatStore.isPendingId(it) }.minOrNull()
        val earlier = if (inPlace && parsedOldest != null)
            state.value.allMessages.filter { m -> numericId(m)?.let { n -> n < parsedOldest } == true } else emptyList()
        val all = if (older) (parsed + state.value.allMessages).distinctBy { m -> m.id } else earlier + parsed
        val leaf = state.value.leafId?.takeIf { candidate -> all.any { numericId(it) == candidate } }
            ?: all.mapNotNull { numericId(it) }.maxOrNull()
        val path = activeBranchPath(all, leaf)
        val keepsEarlier = earlier.isNotEmpty()
        mutable.update { it.copy(messages = path, allMessages = all, leafId = leaf,
            hasOlder = if (keepsEarlier) it.hasOlder else reply.optBoolean("has_older_messages"),
            oldestId = if (keepsEarlier) it.oldestId else reply.nullableString("oldest_loaded_id"),
            jobId = if (older) it.jobId else reply.optJSONObject("pending_job")?.nullableString("job_id")?.ifBlank { null },
            selected = it.selected?.let { selected -> selected.copy(
                title = reply.optString("title", selected.title),
                model = reply.nullableString("last_model").ifBlank { selected.model },
                isTemporary = reply.optBoolean("is_temporary", selected.isTemporary),
            ) },
            customInstruction = if (older) it.customInstruction else reply.nullableString("custom_instruction"),
            includeGlobalInstruction = if (older) it.includeGlobalInstruction else reply.optBoolean("include_global_instruction", true),
            newThreadTemporary = if (older) it.newThreadTemporary else false,
            tempChatTimeoutSeconds = if (older) it.tempChatTimeoutSeconds else
                reply.optInt("timeout_seconds").takeIf { value -> reply.has("timeout_seconds") && value > 0 },
            tempChatRemainingSeconds = if (older) it.tempChatRemainingSeconds else
                reply.optLong("temp_chat_remaining_seconds").takeIf { value -> !reply.isNull("temp_chat_remaining_seconds") && value >= 0 },
            liveContent = if (older) it.liveContent else "", liveThought = if (older) it.liveThought else "",
            cards = if (older) it.cards else emptyList(),
            selectedGem = if (older || inPlace) it.selectedGem else {
                val uuid = reply.nullableString("last_gem_uuid")
                if (uuid.isBlank()) null else it.gems.firstOrNull { gem -> gem.uuid == uuid }
            }) }
        if (!older) {
            // Web `loadMessages`: the chat's PromptCache flag and last model come back with it (the flag
            // first, so a PromptCache lock from the previous chat never blocks the switch).
            mutable.update { it.copy(enablePromptCache = reply.optBoolean("enable_prompt_caching")) }
            if (!inPlace) reply.nullableString("last_model").takeIf { it.isNotBlank() && it != state.value.model }?.let(::chooseModel)
        }
        if (!deviceChat(id)) state.value.account?.let { account ->
            withContext(Dispatchers.IO) {
                offlineCache.saveThread(
                    // Unsent answers of serverless mode are not cached: they are shown from the outbox.
                    account.id, state.value.selected ?: return@withContext,
                    all.filterNot { m -> numericId(m)?.let { n -> state.value.serverless && LocalChatStore.isPendingId(n) } == true },
                    if (keepsEarlier) state.value.hasOlder else reply.optBoolean("has_older_messages"),
                    (if (keepsEarlier) state.value.oldestId else reply.nullableString("oldest_loaded_id"))?.ifBlank { null },
                    reply.nullableString("custom_instruction"), reply.optBoolean("include_global_instruction", true),
                    reply.optLong("temp_chat_remaining_seconds").takeIf { value -> !reply.isNull("temp_chat_remaining_seconds") && value >= 0 },
                    leaf,
                )
            }
            refreshOfflineCacheStats(account.id)
        }
        if (!older && !chatLocal) syncHeartbeat()
        // Web `loadMessages`: an answer still running on the server is rejoined automatically.
        if (!older && autoResume && state.value.jobId != null && !state.value.streaming) resume()
    }

    private suspend fun loadCachedMessages(id: String, older: Boolean, cachedPayload: JSONObject? = null) {
        val accountId = state.value.account?.id ?: return
        val outbox = localChats?.takeIf { state.value.serverless }
        val stored = cachedPayload ?: withContext(Dispatchers.IO) { offlineCache.loadThread(accountId, id) }
        // Serverless mode: answers not yet on the server are shown on top of the cached chat.
        val cached = if (outbox == null) stored ?: return else withContext(Dispatchers.IO) {
            val base = stored?.let { JSONObject(it.toString()) }
                ?: JSONObject().put("messages", JSONArray()).takeIf { outbox.outboxIdFor(id) != null }
            base?.let { outbox.overlayPending(id, it) }
        } ?: return
        val parsed = parseMessages(cached)
        val all = if (older) (parsed + state.value.allMessages).distinctBy { it.id } else parsed
        val leaf = state.value.leafId?.takeIf { candidate -> all.any { numericId(it) == candidate } }
            ?: all.mapNotNull { numericId(it) }.filter { outbox != null && LocalChatStore.isPendingId(it) }.maxOrNull()
            ?: cached.optInt("leaf_id").takeIf { it > 0 }
            ?: all.mapNotNull { numericId(it) }.maxOrNull()
        val path = activeBranchPath(all, leaf)
        mutable.update { it.copy(
            messages = path,
            allMessages = all,
            leafId = leaf,
            hasOlder = false,
            oldestId = cached.nullableString("oldest_loaded_id").ifBlank { null },
            jobId = null,
            customInstruction = cached.nullableString("custom_instruction"),
            includeGlobalInstruction = cached.optBoolean("include_global_instruction", true),
            tempChatRemainingSeconds = cached.optLong("temp_chat_remaining_seconds")
                .takeIf { value -> !cached.isNull("temp_chat_remaining_seconds") && value >= 0 },
            offline = true,
        ) }
    }

    private fun syncHeartbeat() {
        heartbeatJob?.cancel()
        val selected = state.value.selected ?: return
        if (!foreground || !selected.isTemporary) return
        heartbeatJob = viewModelScope.launch {
            while (isActive && foreground && state.value.selected?.id == selected.id && state.value.selected?.isTemporary == true) {
                try {
                    val reply = backend.post("/api/temporary_chat/heartbeat",
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
    /** The lock countdown reached zero: the Web reloads the page; the app clears the overlay and reloads. */
    fun accountLockExpired() {
        mutable.update { it.copy(accountLock = null) }
        refresh()
    }

    fun refresh() {
        val thread = state.value.selected
        if (thread != null) openThread(thread)
        else navigationJob = viewModelScope.launch { try { fetchThreads(false) } catch (e: Exception) { report(e) } }
    }
    /**
     * Coming back to the app: the open chat picks up new answers, a server job to rejoin and the temporary
     * chat heartbeat without reopening it, so the scroll position, draft and loaded history stay. Web does
     * not reload the chat when the tab becomes visible again.
     */
    private fun refreshInPlace() {
        val thread = state.value.selected ?: return
        navigationJob?.cancel()
        navigationJob = viewModelScope.launch {
            Diagnostics.log("thread.refresh", "thread" to thread.id)
            try { loadMessages(thread.id, inPlace = true) }
            catch (e: CancellationException) { throw e }
            catch (e: Exception) { report(e) }
        }
    }
    /** [onNewChat] runs when the open chat was deleted (Web `startNewChat` also closes the phone sidebar). */
    fun deleteThread(thread: ThreadItem, onNewChat: () -> Unit = {}) { viewModelScope.launch {
        Diagnostics.log("thread.delete", "thread" to thread.id, "offline" to state.value.offline)
        if (state.value.offline && !deviceChat(thread.id)) { notify("オフライン中は履歴を削除できません。"); return@launch }
        try {
            backend.delete("/api/threads/${thread.id}", token())
            Diagnostics.log("thread.deleted", "thread" to thread.id)
            state.value.account?.let { account -> withContext(Dispatchers.IO) { offlineCache.deleteThread(account.id, thread.id) } }
            chatHistory.removeThread(thread.id)
            mutable.update { it.copy(canGoBackInChats = chatHistory.canGoBack) }
            if (state.value.selected?.id == thread.id) { newChat(recordHistory = false); onNewChat() }
            fetchThreads(false)
        } catch (e: Exception) { report(e) }
    } }
    /** Web `renameThread`: `prompt("Title:")` then PUT the new title; empty input is ignored. */
    fun renameThread(thread: ThreadItem, title: String) { viewModelScope.launch {
        if (title.isEmpty()) return@launch
        if (state.value.offline && !deviceChat(thread.id)) { notify("オフライン中はタイトルを変更できません。"); return@launch }
        try {
            val reply = backend.put("/api/threads/${thread.id}/title", JSONObject().put("title", title), token())
            val saved = reply.optString("title", title).ifBlank { title }
            mutable.update { current -> current.copy(
                selected = current.selected?.let { if (it.id == thread.id) it.copy(title = saved) else it },
            ) }
            fetchThreads(false)
        } catch (e: Exception) { report(e) }
    } }

    fun requestSettings(tab: String? = null, card: String? = null) {
        mutable.update { it.copy(settingsRequest = it.settingsRequest + 1, settingsRequestTab = tab, settingsRequestCard = card) }
    }

    /** Web `deleteMessage`: removes the message and everything after it, then reloads the thread. */
    fun deleteMessage(message: ChatMessage) {
        val id = numericId(message) ?: return
        val thread = state.value.selected ?: return
        viewModelScope.launch {
            if (state.value.offline && !deviceChat(thread.id) && !(state.value.serverless && LocalChatStore.isPendingId(id))) {
                notify("オフライン中はメッセージを削除できません。"); return@launch
            }
            try {
                backend.delete("/api/messages/$id", token())
                if (state.value.selected?.id == thread.id) {
                    // Web reloads the chat without `preserveDraft`, which runs `cancelEdit`.
                    pendingParentId = null
                    mutable.update { it.copy(leafId = null, editingMessageId = null, draft = "", attachments = emptyList(), quote = "") }
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
        viewModelScope.launch {
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
    fun reloadThreads(onDone: () -> Unit) { viewModelScope.launch {
        try { fetchThreads(false) } catch (e: Exception) { report(e) } finally { onDone() }
    } }

    /** Pull-to-refresh of the sidebar Gem list. */
    fun reloadGems(onDone: () -> Unit) { viewModelScope.launch {
        try { fetchGems() } catch (e: Exception) { report(e) } finally { onDone() }
    } }

    /** `/static/legal/<kind>.md` shown by the terms / privacy modal (public, no token). */
    /** Account, session and data-transfer operations used by the settings modal. */
    fun accountApi(): AccountApi = AccountApi(api) { token() }

    /** Settings Data tab export / import / dedupe; outlives the settings modal like the Web tab. */
    val accountTransfer: AccountTransferController by lazy {
        AccountTransferController(viewModelScope, ::accountApi, getApplication<Application>().contentResolver, ::notify) { categories ->
            if ("chats" in categories) reloadThreads {}
            if ("gems" in categories) reloadGems {}
            if ("files" in categories) loadStorageUsage()
            if ("settings" in categories || "api_credentials" in categories) loadPreferences()
        }
    }

    /** App Link back from `/android/link/<provider>` (Web flash text after linking). */
    fun linkCompleted(provider: String) {
        notify(if (provider == "minashin") "Minashin アカウントと連携しました。" else "Google アカウントと連携しました。")
        loadPreferences()
    }

    /** Ends this device's sign-in after the account or every session was removed on the server. */
    fun signedOutRemotely() { viewModelScope.launch { clearSession() } }

    /** Web `openThreadModal`: a new chat is created first so its settings can be edited. */
    fun ensureThread(onReady: () -> Unit) {
        if (state.value.selected != null) { onReady(); return }
        if (state.value.offline && !uploadsLocal) { notify("オフライン中はメッセージを送信できません。"); return }
        viewModelScope.launch {
            try {
                val created = backend.post("/api/threads", JSONObject().put("is_temporary", state.value.newThreadTemporary), token())
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

    /** Web `schedulePromptTokenEstimate` request (`POST /api/token_estimate`). */
    suspend fun estimatePromptTokens(model: String, message: String, quote: String, imageUrls: List<String>): JSONObject =
        backend.post("/api/token_estimate", JSONObject().put("model", model).put("message", message)
            .put("quote_text", quote).put("image_urls", JSONArray(imageUrls)), token())

    suspend fun legalMarkdown(kind: String): String {
        val safe = if (kind == "privacy") "privacy" else "terms"
        return api.getText("/static/legal/$safe.md?t=${System.currentTimeMillis()}")
    }

    fun toggleBookmark(thread: ThreadItem) { viewModelScope.launch {
        if (state.value.offline && !deviceChat(thread.id)) { notify("オフライン中はブックマークを変更できません。"); return@launch }
        try {
            val reply = backend.post("/api/threads/${thread.id}/bookmark", JSONObject(), token())
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
            if (state.value.offline && !deviceChat(thread.id)) { notify("オフライン中はチャット設定を変更できません。"); return@launch }
            mutable.update { it.copy(busy = true) }
            try {
                val normalizedTitle = title.trim().ifBlank { "新しいチャット" }
                if (normalizedTitle != thread.title) {
                    backend.put("/api/threads/${thread.id}/title", JSONObject().put("title", normalizedTitle), token())
                }
                val reply = backend.put("/api/threads/${thread.id}/settings", JSONObject()
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
        viewModelScope.launch {
            if (state.value.offline && !deviceChat(thread.id)) { notify("オフライン中はチャット設定を変更できません。"); onDone(false); return@launch }
            try {
                backend.put("/api/threads/${thread.id}/settings", JSONObject()
                    .put("custom_instruction", instruction)
                    .put("include_global_instruction", includeGlobal), token())
                mutable.update { current ->
                    if (current.selected?.id != thread.id) current
                    else current.copy(customInstruction = instruction, includeGlobalInstruction = includeGlobal)
                }
                val reply = backend.put("/api/mobile/v1/preferences", userPrompt, token())
                mutable.update { it.withSavedPreferences(parsePreferences(reply)).copy(notice = "保存されました") }
                onDone(true)
            } catch (e: Exception) {
                if (e is ApiException) notify("保存に失敗しました") else notify("エラー: ${e.message}")
                onDone(false)
            }
        }
    }
    /** Prepares the composer to edit a user message, branching from its parent. */
    fun beginEdit(message: ChatMessage) {
        if (message.role != "user") return
        pendingParentId = message.parentId
        // Web `beginEditMessage`: the message's own quote comes back (or the quote bar clears) and the input gets focus.
        mutable.update { it.copy(
            editingMessageId = message.id,
            draft = message.content,
            attachments = message.files.map { reference -> Attachment(reference.substringAfterLast('/'), reference) },
            quote = message.quote,
            composerFocusRequest = it.composerFocusRequest + 1,
        ) }
    }

    /** Re-sends the user message that produced an assistant reply, creating a sibling branch. */
    fun regenerate(message: ChatMessage) {
        if (message.role != "assistant" || state.value.busy) return
        // Web `regenerateMessage`; while an answer streams, send() shows the same waiting notice as Web.
        val parent = message.parentId?.let { pid -> state.value.allMessages.firstOrNull { numericId(it) == pid } }
            ?: run { notify("再生成できるメッセージが見つかりません"); return }
        beginEdit(parent)
        send()
    }

    fun cancelEdit() {
        pendingParentId = null
        mutable.update { it.copy(editingMessageId = null, draft = "", attachments = emptyList(), quote = "") }
    }

    /** Switches the active path to the branch that contains [targetMessageId]. */
    fun switchBranch(targetMessageId: Int) {
        val all = state.value.allMessages
        val leaf = latestLeafId(all, targetMessageId)
        // Web keeps ‹ › choices for this visit only; reopening shows the pinned or the latest branch.
        mutable.update { it.copy(leafId = leaf, messages = activeBranchPath(all, leaf)) }
    }

    fun switchBranchByIndex(siblings: List<ChatMessage>, index: Int) {
        siblings.getOrNull(index)?.let { numericId(it)?.let(::switchBranch) }
    }

    /**
     * [xLinkChecked] is set once the X-link question was answered; [disableAutoSearch] tells the server the
     * user chose to answer without search (Web `disable_auto_search`).
     */
    fun send(xLinkChecked: Boolean = false, disableAutoSearch: Boolean = false, preChecked: Boolean = false) {
        if (!preChecked && !beginSend()) return
        var current = state.value
        if (current.banned) return
        // Serverless mode answers on the device, so it can send while the server is out of reach.
        if (current.offline && !uploadsLocal) { mutable.update { it.copy(notice = "オフライン中はメッセージを送信できません。") }; return }
        if (current.busy || (current.draft.isBlank() && current.attachments.isEmpty())) return
        if (current.model.isBlank()) { mutable.update { it.copy(notice = "モデルを選択してください。") }; return }
        current.account?.models?.firstOrNull { it.id == current.model && it.selectable } ?: return
        // Web sendMessage checks: audio / video the model cannot take, and what Mistral OCR accepts.
        val (audioOk, videoOk) = modelMediaSupport(current.model)
        val hasAudio = current.attachments.any { isAudioPath(it.reference) }
        val hasVideo = current.attachments.any { isVideoPath(it.reference) }
        if ((hasAudio && !audioOk) || (hasVideo && !videoOk)) {
            notify("このモデルは音声/動画入力に対応していません")
            mutable.update { it.copy(attachments = it.attachments.filterNot { a ->
                (isAudioPath(a.reference) && !audioOk) || (isVideoPath(a.reference) && !videoOk) }) }
            return
        }
        if (isMistralOcrModel(current.model)) {
            if (hasAudio || hasVideo) { notify("Mistral OCR は音声・動画に対応していません。PDF / 画像 / DOCX / PPTX を添付してください。"); return }
            if (current.attachments.isEmpty() && !Regex("https?://\\S+", RegexOption.IGNORE_CASE).containsMatchIn(current.draft)) {
                notify("Mistral OCR は文書専用です。PDF・画像・DOCX・PPTX を添付するか、公開URLを入力してください。")
                return
            }
        }
        // Web: an X link switches to Search + Grok 4 Fast Reasoning, or asks first when that is turned off.
        val hasXLink = X_LINK_PATTERN.containsMatchIn(current.draft) || X_LINK_PATTERN.containsMatchIn(current.quote)
        if (!xLinkChecked && hasXLink && !isMistralOcrModel(current.model) && !current.enableSearch) {
            if (current.preferences?.autoSearchOnLinks != false) {
                applyXLinkSearch()
                current = state.value
            } else {
                mutable.update { it.copy(xLinkPrompt = true) }
                return
            }
        }
        val generation = try { generationOptionsPayload(current.model, current.generationValues) }
            catch (e: IllegalArgumentException) { notify(e.message ?: "生成設定を確認してください。"); return }
        // Web toggleOptions: hidden options are off and forced checkboxes send their forced value.
        val rules = composerRules(current.model, current.mcpServers.any { it.enabled })
        if (current.batchMode && rules.batch.visible && current.codingMode) {
            notify("Batch APIではCoding Modeを利用できません。Batchを解除するかCodingを解除してください。")
            return
        }
        fun effective(value: Boolean, rule: OptionRule) = rule.forced ?: value
        val effort = current.chipValues["reasoning_effort"].orEmpty()
        val deepSeekNonThinking = current.model.lowercase().contains("deepseek") && effort.lowercase() == "none"
        val items = attachmentItemsForSend(current.attachments)
        if (current.imageMask != null && isGptImageModel(current.model) && items.isEmpty()) {
            notify("Mask は画像入力が必要です")
            return
        }
        val body = JSONObject().put("model", current.model).put("message", current.draft)
            .put("client_request_id", UUID.randomUUID().toString()).put("image_urls", JSONArray(items.map { it.reference }))
            .put("image_items", JSONArray().apply {
                items.forEach { put(JSONObject().put("path", it.reference).put("source", it.source).put("name", it.name)) }
            })
            .put("uploaded_image_urls", JSONArray(items.filter { it.source == "upload" }.map { it.reference }))
            .put("marker_system_prompt", if (current.attachments.any { it.edited }) MARKER_HINT_TEXT else JSONObject.NULL)
            .put("image_vision_model", current.visionModel ?: current.preferences?.defaultVisionModel ?: JSONObject.NULL)
            .put("enable_thinking", !deepSeekNonThinking && effective(current.enableThinking, rules.thinking))
            .put("enable_search", effective(current.enableSearch, rules.search))
            .put("enable_url_context", effective(current.enableUrlContext, rules.urls))
            .put("enable_maps", effective(current.enableMaps, rules.maps))
            .put("enable_file_creation", effective(current.enableFileCreation, rules.file))
            .put("enable_system_prompt", effective(current.enableSystemPrompt, rules.sysPrompt))
            .put("enable_prompt_caching", effective(current.enablePromptCache, rules.promptCache))
            .put("batch_mode", current.batchMode && rules.batch.visible).put("enable_python", effective(current.enablePython, rules.python))
            .put("enable_mcp", current.enableMcp && rules.mcp.visible)
            .put("coding_mode", current.codingMode)
            .put("canvas_mode", current.canvasMode)
            .put("temporary_chat", current.selected?.isTemporary ?: current.newThreadTemporary)
            .put("disable_auto_search", disableAutoSearch)
        current.imageMask?.let { body.put("image_mask", it) }
        if (current.quote.isNotBlank()) body.put("quote_text", current.quote)
        if (current.codingMode) {
            val plan = planCodingSend(current.draft, historyCodingTargets(current.messages), current.codingTarget, current.model)
            plan.error?.let { notify(it); return }
            val target = plan.target
            if (plan.active && target != null) {
                fun JSONObject.candidateFields(candidate: CodingCandidate) = put("id", candidate.candidateId)
                    .put("code", if (candidate.promptSource) JSONObject.NULL else candidate.code)
                    .put("language", candidate.language.ifBlank { "text" })
                    .put("source", if (candidate.promptSource) "prompt" else "history")
                    .put("explicit", candidate.explicit)
                body.put("coding_target", JSONObject().candidateFields(target).put("key", target.key)
                    .put("message_id", target.messageId ?: JSONObject.NULL))
                body.put("coding_candidates", JSONArray().apply {
                    plan.candidates.forEach { put(JSONObject().candidateFields(it).put("prompt_index", it.promptIndex ?: JSONObject.NULL)) }
                })
            } else body.put("coding_mode", false)
        }
        generation.keys().forEach { key -> body.put(key, generation.get(key)) }
        COMPOSER_SELECT_DEFAULTS.forEach { (key, fallback) -> body.put(key, current.chipValues[key] ?: fallback) }
        // Web part14: the server never reads Gem.instruction, the client sends it as the system prompt.
        current.selectedGem?.let { gem ->
            body.put("gem_uuid", gem.uuid)
            body.put("system_prompt", gem.instruction).put("enable_system_prompt", true)
        }
        if (current.editingMessageId != null) {
            // Branch from the edited message's parent; send null explicitly for the first message.
            body.put("parent_id", pendingParentId ?: JSONObject.NULL)
            body.put("parent_id_explicit", true)
        } else {
            // Keep normal sends on the currently selected branch.
            current.leafId?.let { body.put("parent_id", it) }
        }
        current.selected?.let { body.put("thread_id", it.id) }
        val submission = Submission(body, current.attachments, current.imageMask, current.editingMessageId, pendingParentId)
        failed = submission
        pendingParentId = null
        // Web `resetUploadState` + `clearQuote`: the mask goes with the attachments.
        mutable.update { it.copy(draft = "", attachments = emptyList(), editingMessageId = null, quote = "", imageMask = null) }
        submit(submission)
    }
    fun retry() { failed?.let { submit(it) } }

    /**
     * Web `sendMessage` entry: a short vibration, then the waiting notices while an answer streams or files
     * upload. Returns false when the send must stop there.
     */
    fun beginSend(): Boolean {
        vibrate(50)
        when {
            state.value.streaming -> notify("回答生成中です。完了までお待ちいただくか、停止してください。")
            state.value.uploading -> notify("ファイルの送信・処理中です。しばらくお待ちください。")
            else -> return true
        }
        return false
    }

    /** Web `vibrateHelper` (`navigator.vibrate`): one pulse of [timings] ms, or on/off/on… pulses. */
    private fun vibrate(vararg timings: Long) {
        runCatching {
            val context = getApplication<Application>()
            val vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                context.getSystemService(android.os.VibratorManager::class.java)?.defaultVibrator
            } else {
                @Suppress("DEPRECATION")
                context.getSystemService(android.os.Vibrator::class.java)
            }
            if (vibrator?.hasVibrator() != true) return
            vibrator.vibrate(
                if (timings.size == 1) android.os.VibrationEffect.createOneShot(timings[0], android.os.VibrationEffect.DEFAULT_AMPLITUDE)
                else android.os.VibrationEffect.createWaveform(longArrayOf(0L) + timings, -1),
            )
        }
    }

    private var micRecorder: android.media.MediaRecorder? = null
    private var micFile: File? = null
    private var micLevelJob: Job? = null

    /**
     * Web `mic-btn` for chat models: the first tap records, the second stops and sends the clip to
     * `/transcribe` (STT API or the current LLM, as set in 設定), then appends the text to the input.
     * The caller has already obtained the microphone permission.
     */
    fun toggleMicRecording() {
        when (state.value.micMode) {
            "recording" -> { stopMicRecording(); return }
            "preparing", "transcribing" -> return
        }
        mutable.update { it.copy(micMode = "preparing", micLevels = emptyList()) }
        val app = getApplication<Application>()
        val file = File(app.cacheDir, "recording.m4a")
        try {
            val recorder = if (Build.VERSION.SDK_INT >= 31) android.media.MediaRecorder(app) else android.media.MediaRecorder()
            recorder.setAudioSource(android.media.MediaRecorder.AudioSource.MIC)
            recorder.setOutputFormat(android.media.MediaRecorder.OutputFormat.MPEG_4)
            recorder.setAudioEncoder(android.media.MediaRecorder.AudioEncoder.AAC)
            recorder.setAudioSamplingRate(44_100)
            recorder.setAudioEncodingBitRate(96_000)
            recorder.setOutputFile(file.absolutePath)
            recorder.prepare()
            recorder.start()
            micRecorder = recorder
            micFile = file
        } catch (e: Exception) {
            micRecorder?.release()
            micRecorder = null
            mutable.update { it.copy(micMode = "", micLevels = emptyList()) }
            notify("Microphone access denied or not available.")
            return
        }
        mutable.update { it.copy(micMode = "recording") }
        micLevelJob?.cancel()
        micLevelJob = viewModelScope.launch {
            while (true) {
                val amplitude = runCatching { micRecorder?.maxAmplitude ?: 0 }.getOrDefault(0)
                val level = (amplitude / 32767f).coerceIn(0f, 1f)
                mutable.update { it.copy(micLevels = (it.micLevels + level).takeLast(24)) }
                delay(75)
            }
        }
    }

    private fun stopMicRecording() {
        micLevelJob?.cancel()
        val recorder = micRecorder ?: return
        micRecorder = null
        val file = micFile
        val stopped = runCatching { recorder.stop() }.isSuccess
        recorder.release()
        if (!stopped || file == null || !file.exists()) {
            mutable.update { it.copy(micMode = "", micLevels = emptyList()) }
            notify("Audio processing error: 録音できませんでした")
            return
        }
        val modelId = state.value.model
        if (state.value.preferences?.micTranscribeMode == "llm" && !modelMediaSupport(modelId).first) {
            mutable.update { it.copy(micMode = "", micLevels = emptyList()) }
            notify("現在のモデルはLLM音声文字起こし（音声入力）に対応していません")
            file.delete()
            return
        }
        mutable.update { it.copy(micMode = "transcribing", micLevels = emptyList()) }
        viewModelScope.launch {
            try {
                val data = localSettings?.let { settings -> transcribeOnDevice(settings, file) } ?: api.transcribe(file, modelId, token())
                val transcript = data.optString("transcript")
                if (transcript.isNotEmpty()) mutable.update { it.copy(draft = if (it.draft.isEmpty()) transcript else it.draft + " " + transcript) }
                else notify(data.optString("error").ifBlank { "Transcription failed" })
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) { notify("Audio processing error: " + (e.message ?: "")) }
            finally {
                file.delete()
                mutable.update { it.copy(micMode = "", micLevels = emptyList()) }
            }
        }
    }

    /** Voice input while chats are local: the device's OpenAI key and speech-to-text model (`/transcribe` shape). */
    private suspend fun transcribeOnDevice(settings: LocalSettingsStore, file: File): JSONObject {
        val key = withContext(Dispatchers.IO) { settings.providerKey("openai_key") }
            ?: return JSONObject().put("error", "音声入力にはOpenAI APIキーが必要です（APIキータブで設定）")
        val model = state.value.preferences?.sttModel?.takeIf { it.startsWith("gpt") || it.startsWith("whisper") } ?: "gpt-4o-mini-transcribe"
        val bytes = withContext(Dispatchers.IO) { file.readBytes() }
        val text = TranscriptionDirect(directHttp).transcribe(model, key, file.name, "audio/mp4", bytes)
        return JSONObject().put("transcript", text)
    }

    /** Web `applyXLinkAuto`. */
    private fun applyXLinkSearch() {
        mutable.update { it.copy(enableSearch = true) }
        if (state.value.model != "grok-4-fast-reasoning") chooseModel("grok-4-fast-reasoning")
    }

    /** Web `#auto-search-banner`: 検索ONで続行 / 検索OFFで回答, optionally remembered (次回から尋ねない). */
    fun resolveXLinkPrompt(enable: Boolean, remember: Boolean) {
        mutable.update { it.copy(xLinkPrompt = false) }
        if (enable) {
            applyXLinkSearch()
            if (remember) viewModelScope.launch {
                runCatching {
                    val reply = backend.put("/api/mobile/v1/preferences", JSONObject().put("auto_search_on_links", true), token())
                    mutable.update { it.copy(preferences = parsePreferences(reply)) }
                }
            }
        }
        send(xLinkChecked = true, disableAutoSearch = !enable)
    }

    /** Web `api-key-modal-save-btn`: saves the key for the model's provider, then sends again. */
    fun saveApiKeyAndResend(key: String) {
        val modelId = state.value.apiKeyPrompt ?: return
        val trimmed = key.trim()
        if (trimmed.isEmpty()) { notify("APIキーを入力してください"); return }
        val info = apiKeyInfoFor(modelId) ?: return
        viewModelScope.launch {
            try {
                val reply = backend.put("/api/mobile/v1/preferences", JSONObject().put(info.keyField, trimmed), token())
                mutable.update { it.copy(preferences = parsePreferences(reply), apiKeyPrompt = null) }
                send()
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) { notify("APIキーの保存に失敗しました") }
        }
    }

    /** `api-key-modal-cancel-btn` shows the original error; `api-key-modal-fallback-btn` leaves the model picker to the UI. */
    fun dismissApiKeyPrompt(showError: Boolean) {
        val modelId = state.value.apiKeyPrompt ?: return
        mutable.update { it.copy(apiKeyPrompt = null) }
        if (showError) notify(apiKeyMissingMessage ?: "${modelDisplayName(modelId)} のAPIキーが設定されていません")
    }

    private var apiKeyMissingMessage: String? = null

    fun modelDisplayName(modelId: String): String =
        state.value.account?.models?.firstOrNull { it.id == modelId }?.name ?: modelId

    /** Web `showPendingSlashCommandIndicator` / `hidePendingSlashCommandIndicator` (leaving `/settings` clears its conversation). */
    fun setPendingSlashCommand(id: String?) {
        if (id == null && state.value.pendingSlashCommand == "settings") aiSettingsConversation.clear()
        mutable.update { it.copy(pendingSlashCommand = id) }
    }

    /** Web `aiSettingsConversation` (kept for this app session like the Web sessionStorage). */
    private val aiSettingsConversation = mutableListOf<Pair<String, String>>()

    private fun appendAiSettingsConversation(role: String, content: String) {
        val text = content.trim()
        if (text.isEmpty()) return
        aiSettingsConversation += role to text.take(1600)
        while (aiSettingsConversation.size > 10) aiSettingsConversation.removeAt(0)
    }

    /** Web `runAiSettingsCommand`: `/settings <instruction>` changes or inspects settings through the selected model. */
    fun runAiSettings(instruction: String) {
        val modelId = state.value.model
        if (modelId.isBlank()) { notify("モデルを選択してください"); return }
        if (isMistralOcrModel(modelId)) { notify("Mistral OCR は設定変更コマンドに使えません。チャットモデルを選んでください。"); return }
        if (state.value.pendingSlashCommand != "settings") mutable.update { it.copy(pendingSlashCommand = "settings") }
        val history = JSONArray(aiSettingsConversation.map { (role, content) -> JSONObject().put("role", role).put("content", content) })
        appendAiSettingsConversation("user", instruction)
        val stamp = System.currentTimeMillis()
        val pendingId = "settings-pending-$stamp"
        mutable.update { it.copy(draft = "", settingsBubbles = it.settingsBubbles +
            SettingsBubble("settings-user-$stamp", "user", "/settings $instruction") +
            SettingsBubble(pendingId, "assistant", "", modelId, pending = true)) }
        viewModelScope.launch {
            fun finish(bubble: SettingsBubble?) = mutable.update { current ->
                current.copy(settingsBubbles = current.settingsBubbles.filterNot { b -> b.id == pendingId } + listOfNotNull(bubble))
            }
            try {
                val data = backend.post("/api/settings/apply-ai-prompt", JSONObject().put("prompt", instruction).put("model", modelId)
                    .put("conversation", history), token())
                val inspect = data.optString("mode") == "inspect"
                val values = if (inspect) data.optJSONObject("current") else data.optJSONObject("applied")
                if (data.optString("status") == "ok" && values != null) {
                    appendAiSettingsConversation("assistant", summarizeAiSettings(values, inspect))
                    val count = values.length()
                    notify(if (inspect) "現在の設定を確認しました（${count}項目）" else "設定を更新しました（${count}項目）")
                    if (!inspect) runCatching { fetchPreferences() }
                    val entries = values.keys().asSequence().map { key -> key to formatAiSettingValue(values.opt(key)) }.toList()
                    val text = when {
                        entries.isNotEmpty() && inspect -> "現在の設定を確認しました。\n\n確認した項目をタップすると、設定画面の該当箇所へ移動できます。"
                        entries.isNotEmpty() -> "設定を更新しました。\n\n変更した項目をタップすると、設定画面の該当箇所へ移動できます。"
                        inspect -> "確認できる設定項目がありませんでした。"
                        else -> "変更された設定項目はありませんでした。"
                    }
                    finish(SettingsBubble("settings-result-${System.currentTimeMillis()}", "assistant", text, modelId, entries))
                    return@launch
                }
                val message = data.optString("message").ifBlank { data.optString("error") }.ifBlank { "設定変更に失敗しました" }
                appendAiSettingsConversation("assistant", "設定操作に失敗しました: $message")
                finish(SettingsBubble("settings-error-${System.currentTimeMillis()}", "assistant", "設定変更に失敗しました。\n\n$message", modelId))
                notify(message)
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                val message = (e as? ApiException)?.payload?.let { p -> p.optString("message").ifBlank { p.optString("error") } }
                if (!message.isNullOrBlank()) {
                    appendAiSettingsConversation("assistant", "設定操作に失敗しました: $message")
                    finish(SettingsBubble("settings-error-${System.currentTimeMillis()}", "assistant", "設定変更に失敗しました。\n\n$message", modelId))
                    notify(message)
                } else {
                    appendAiSettingsConversation("assistant", "設定操作の通信に失敗しました。")
                    finish(SettingsBubble("settings-error-${System.currentTimeMillis()}", "assistant",
                        "設定変更の通信に失敗しました。時間をおいて再度お試しください。", modelId))
                    notify("設定変更の通信に失敗しました")
                }
            }
        }
    }
    private var searchClearJob: Job? = null

    /** Web `fetchChatStreamWithUnavailableRetry`: which responses keep the send waiting instead of failing. */
    private fun waitingStatusFor(error: Throwable): String? = when {
        error is ApiException && error.status == 503 -> "メンテナンス終了を待っています..."
        error is ApiException && connectionStatusForHttp(error.status) == ConnectionStatus.SERVER_DOWN -> "サーバーの復帰を待っています..."
        error is ApiException && error.status == 425 && error.code == "submission_in_progress" -> "サーバーの復帰を待っています..."
        error is ApiException -> null
        error is IOException -> "インターネット接続の復帰を待っています..."
        else -> null
    }

    private fun updateLive(transform: (LiveAnswer) -> LiveAnswer) = mutable.update { it.copy(live = transform(it.live)) }

    /** Web `markApiAccepted`: once, the skeleton says the connection is up and the model is being awaited. */
    private fun markApiAccepted() = updateLive { live ->
        if (live.accepted || live.pendingStatus == null) live.copy(accepted = true)
        else live.copy(accepted = true, pendingStatus = "接続完了。モデル応答を待機中...", pendingSub = "キュー待機や初期化中の可能性があります")
    }

    private fun submit(submission: Submission) {
        streamJob?.cancel()
        streamJob = viewModelScope.launch {
            val owner = currentCoroutineContext().job
            val body = submission.body
            val modelId = body.optString("model")
            val reasoning = showsReasoningProgress(modelId, body.optBoolean("enable_thinking"), body.optString("reasoning_effort"))
            mutable.update { it.copy(streaming = true, retryAvailable = false, status = "", liveContent = "", liveThought = "",
                cards = emptyList(), mcpDecision = null,
                live = LiveAnswer(model = modelId, pendingStatus = "APIに送信中...",
                    thoughtPlaceholder = if (reasoning) "推論プロセスを準備中..." else null)) }
            val flow = api.progress.startFlow("chat")
            var id = body.nullableString("thread_id")
            var accepted = false
            val sendStarted = System.currentTimeMillis()
            Diagnostics.log("send.start", "model" to modelId, "serverless" to state.value.serverless, "new_thread" to id.isBlank(),
                "attachments" to submission.files.map { it.mime.ifBlank { "unknown" } }, "text_chars" to body.optString("message").length,
                "prompt_cache" to body.optBoolean("enable_prompt_caching"), "memory" to Diagnostics.memory())
            val seenEvents = java.util.Collections.synchronizedSet(HashSet<String>())
            try {
                if (id.isBlank()) {
                    val created = backend.post("/api/threads", JSONObject().put("is_temporary", state.value.newThreadTemporary), token())
                    id = created.get("id").toString()
                    body.put("thread_id", id)
                    mutable.update { it.copy(selected = ThreadItem(id, created.nullableString("title"), it.model,
                        isTemporary = created.optBoolean("is_temporary")), newThreadTemporary = false) }
                    syncHeartbeat()
                }
                if (body.optBoolean("parent_id_explicit")) {
                    // Branching send (edit-and-resend, regenerate): drop the old branch's tail
                    // from the visible path immediately, instead of leaving the previous
                    // prompt/reply bubbles on screen until the new answer finishes streaming.
                    val parentId = body.opt("parent_id") as? Int
                    mutable.update { current ->
                        val truncated = if (parentId == null) emptyList()
                        else current.messages.indexOfFirst { numericId(it) == parentId }
                            .let { idx -> if (idx >= 0) current.messages.subList(0, idx + 1) else current.messages }
                        current.copy(messages = truncated)
                    }
                }
                val userId = "local-${body.getString("client_request_id")}"
                mutable.update { it.copy(messages = it.messages.filterNot { m -> m.id == userId } + ChatMessage(userId, "user",
                    body.getString("message"), files = submission.files.map { a -> a.reference },
                    quote = body.optString("quote_text"), gemName = it.selectedGem?.name.orEmpty())) }
                val threadId = id
                val onEvent: (JSONObject) -> Unit = { event ->
                    val type = event.optString("type")
                    if (type == "status" || type == "error" || seenEvents.add(type)) {
                        // Status texts and errors come from the app, the server or the AI provider, never from the user.
                        Diagnostics.log("send.event", "type" to type, "ms" to System.currentTimeMillis() - sendStarted,
                            "status" to if (type == "status" || type == "error") event.optString("content").take(1000) else null)
                    }
                    if (streamJob === owner) {
                        flow.setPhase("receiving")
                        acceptEvent(threadId, event)
                    }
                }
                val onAccepted: () -> Unit = {
                    accepted = true
                    flow.setPhase("waiting")
                    markConnectionReachable()
                    markApiAccepted()
                }
                var retryCount = 0
                while (true) {
                    try {
                        backend.stream("/chat_stream", body, token(), onAccepted, onEvent)
                        break
                    } catch (e: CancellationException) { throw e }
                    catch (e: ApiException) {
                        if (e.code == "request_already_accepted") {
                            accepted = true
                            val jobId = e.payload.optString("job_id")
                            mutable.update { it.copy(jobId = jobId.ifBlank { it.jobId }) }
                            reconnectUntilAvailable(threadId, jobId, owner)
                            break
                        }
                        val waiting = waitingStatusFor(e) ?: throw e
                        retryCount += 1
                        connectionStatusForHttp(e.status)?.let { setConnectionUnavailable(it) }
                        updateLive { it.copy(pendingStatus = waiting, pendingSub = "送信内容を保持して自動再試行中（${retryCount}回目）") }
                    } catch (e: IOException) {
                        if (accepted) {
                            // Web: the answer keeps running on the server; reconnect to it in the background.
                            setConnectionUnavailable(ConnectionStatus.OFFLINE)
                            notify("回答への接続が切れました。バックグラウンド処理へ自動再接続します。")
                            reconnectUntilAvailable(threadId, state.value.jobId.orEmpty(), owner)
                            break
                        }
                        retryCount += 1
                        setConnectionUnavailable(ConnectionStatus.OFFLINE)
                        updateLive { it.copy(pendingStatus = "インターネット接続の復帰を待っています...",
                            pendingSub = "送信内容を保持して自動再試行中（${retryCount}回目）") }
                    }
                    delay(CONNECTION_RETRY_DELAY_MS)
                }
                Diagnostics.log("send.done", "ms" to System.currentTimeMillis() - sendStarted)
                vibrate(100, 50, 100)
                failed = null
                // The response may add a new leaf even for a normal continuation. Drop the old
                // leafId so loadMessages() selects the newest path instead of restoring the
                // previously active branch and hiding the just-sent exchange.
                mutable.update { current ->
                    if (current.selected?.id == id) current.copy(leafId = null) else current
                }
                loadMessages(id)
                fetchThreads(false)
                if (body.optBoolean("batch_mode")) {
                    fetchBatchJobs(notify = false)
                    startBatchPolling()
                }
            } catch (e: CancellationException) {
                Diagnostics.log("send.cancelled", "ms" to System.currentTimeMillis() - sendStarted, "accepted" to accepted)
                throw e
            }
            catch (e: Exception) {
                Diagnostics.failure("send.error", e, "ms" to System.currentTimeMillis() - sendStarted, "accepted" to accepted)
                // Web: before the server accepted the send, the optimistic rows go away and the input,
                // attachments and quote stay; the error is shown as "Connection Error: …".
                val userId = "local-${body.optString("client_request_id")}"
                mutable.update { it.copy(
                    messages = it.messages.filterNot { m -> m.id == userId },
                    draft = if (it.draft.isBlank()) body.optString("message") else it.draft,
                    attachments = if (it.attachments.isEmpty()) submission.files else it.attachments,
                    quote = if (it.quote.isBlank()) body.optString("quote_text") else it.quote,
                    imageMask = it.imageMask ?: submission.mask,
                    editingMessageId = it.editingMessageId ?: submission.editingId,
                ) }
                if (state.value.editingMessageId == submission.editingId && submission.editingId != null) pendingParentId = submission.parentId
                when {
                    e is ApiException && e.code == "api_key_missing" -> {
                        // Web `showApiKeyRequiredModalAsync`: set the key, switch model, or show the error.
                        apiKeyMissingMessage = e.payload.optString("error").ifBlank { null }
                        mutable.update { it.copy(apiKeyPrompt = e.payload.optString("model").ifBlank { modelId }) }
                    }
                    // Web re-runs Turnstile, then asks to send again.
                    e is ApiException && e.code == "turnstile_required" -> startSessionTurnstile()
                    e is ApiException && (e.code == "banned" || e.status == 401) -> report(e)
                    e is ApiException -> notify("Connection Error: " + e.payload.optString("error").ifBlank { "HTTP ${e.status}" })
                    else -> notify("Connection Error: " + (e.message ?: "通信に失敗しました"))
                }
                if (id.isNotBlank() && session != null) runCatching { loadMessages(id) }
            } finally {
                flow.finish()
                if (streamJob === owner) mutable.update { it.copy(streaming = false, status = "", live = LiveAnswer()) }
            }
        }
    }

    /**
     * Web `reconnectPendingStreamUntilAvailable`: while the answer runs on the server, wait and rejoin it
     * through `/chat_stream_resume` until it finishes (404 means it already ended).
     */
    private suspend fun reconnectUntilAvailable(threadId: String, jobId: String, owner: Job) {
        while (true) {
            updateLive { it.copy(pendingStatus = if (it.pendingStatus != null) "サーバーへの再接続を待っています..." else null,
                pendingSub = "回答処理はバックグラウンドで継続しています") }
            delay(CONNECTION_RETRY_DELAY_MS)
            if (state.value.selected?.id != threadId || jobId.isBlank()) return
            try {
                backend.stream("/chat_stream_resume", JSONObject().put("thread_id", threadId).put("job_id", jobId), token(),
                    onAccepted = { markConnectionReachable() }) { event -> if (streamJob === owner) acceptEvent(threadId, event) }
                return
            } catch (e: CancellationException) { throw e }
            catch (e: ApiException) {
                if (e.status == 404) return
                if (waitingStatusFor(e) == null) throw e
            } catch (e: IOException) { /* keep waiting */ }
        }
    }

    private fun acceptEvent(threadId: String, event: JSONObject) {
        if (state.value.selected?.id != threadId) return
        val type = event.optString("type")
        val content = event.opt("content")?.takeIf { it != JSONObject.NULL }?.toString().orEmpty()
        when (type) {
            "thread_id" -> { markApiAccepted(); return }
            "job_id" -> { markApiAccepted(); mutable.update { it.copy(jobId = content) }; return }
            "search_status" -> {
                if (content == "searching") updateLive { if (it.search.isEmpty()) it.copy(search = "searching") else it }
                else if (content == "done" && state.value.live.search == "searching") {
                    updateLive { it.copy(search = "done") }
                    searchClearJob?.cancel()
                    searchClearJob = viewModelScope.launch { delay(2000); updateLive { if (it.search == "done") it.copy(search = "") else it } }
                }
                return
            }
            "mcp", "mcp_decision_request", "mcp_decision_resolved" -> {
                val payload = event.optJSONObject("content")
                mutable.update { current -> current.copy(
                    cards = upsertToolCard(current.cards, type, event.opt("content")),
                    mcpDecision = when (type) {
                        "mcp_decision_request" -> payload?.let { McpDecision(
                            id = it.optString("id"), jobId = current.jobId.orEmpty(),
                            serverName = it.optString("server_name").ifBlank { "不明なサーバー" },
                            toolName = it.optString("tool_name"), argsPreview = it.optString("args_preview"),
                        ) }
                        "mcp_decision_resolved" -> current.mcpDecision?.takeIf { d -> d.id != payload?.optString("id") }
                        else -> if (payload?.optString("type") == "decision_resolved" &&
                            current.mcpDecision?.id == payload?.optString("id")) null else current.mcpDecision
                    },
                ) }
                return
            }
            "status" -> {
                markApiAccepted()
                updateLive { live ->
                    live.copy(
                        pendingStatus = live.pendingStatus?.let { content.ifBlank { "モデル処理中..." } },
                        pendingSub = if (live.pendingStatus != null) "応答開始までの進捗を表示しています" else live.pendingSub,
                        thoughtPlaceholder = live.thoughtPlaceholder?.let { content.ifBlank { "推論プロセスを準備中..." } },
                    )
                }
                return
            }
            "done" -> return
        }
        // Web `beginPendingToStreamTransition`: the first answer event replaces the skeleton.
        updateLive { it.copy(pendingStatus = null, pendingSub = "") }
        when (type) {
            "python" -> event.optJSONObject("content")?.let { payload -> mutable.update { it.copy(cards = upsertPythonCard(it.cards, payload)) } }
            "coding_diff" -> mutable.update { it.copy(cards = upsertToolCard(it.cards, type, event.opt("content"))) }
            "image_analysis" -> updateLive { it.copy(imageAnalysis = content) }
            "thought" -> mutable.update { it.copy(liveThought = it.liveThought + content, live = it.live.copy(thoughtPlaceholder = null)) }
            "content" -> mutable.update { it.copy(liveContent = it.liveContent + content) }
            "error" -> mutable.update { it.copy(live = it.live.copy(error = content.ifBlank { "Unknown error" }), notice = content.ifBlank { "Unknown error" }.take(500)) }
        }
    }

    /** Web `resumePendingStream`: rejoin an answer still running on the server when the thread is opened. */
    fun resume() {
        if (state.value.streaming) return
        val id = state.value.selected?.id ?: return
        val jobId = state.value.jobId ?: return
        streamJob = viewModelScope.launch {
            val owner = currentCoroutineContext().job
            val flow = api.progress.startFlow("chatResume")
            mutable.update { it.copy(streaming = true, liveContent = "", liveThought = "", status = "",
                live = LiveAnswer(model = it.model, pendingStatus = "回答を生成中...")) }
            try {
                backend.stream("/chat_stream_resume", JSONObject().put("thread_id", id).put("job_id", jobId), token(),
                    onAccepted = { flow.setPhase("waiting") }) { event ->
                    if (streamJob === owner) { flow.setPhase("receiving"); acceptEvent(id, event) }
                }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) { if (e !is ApiException || e.status != 404) report(e) }
            finally {
                flow.finish()
                if (streamJob === owner) mutable.update { it.copy(streaming = false, status = "", live = LiveAnswer()) }
            }
            runCatching { loadMessages(id, autoResume = false) }.onFailure { report(it) }
        }
    }
    fun stop() { viewModelScope.launch {
        val id = state.value.selected?.id ?: return@launch
        Diagnostics.log("send.stop", "thread" to id, "streaming" to state.value.streaming, "job" to (state.value.jobId != null))
        try {
            // The answer is generated on the device (a server job is only rejoined, with its job id):
            // cancelling the request saves what arrived so far, and serverless mode then uploads it.
            if (uploadsLocal && state.value.jobId == null) {
                val started = System.currentTimeMillis()
                streamJob?.cancelAndJoin()
                Diagnostics.log("send.stopped", "thread" to id, "wait_ms" to System.currentTimeMillis() - started)
                mutable.update { it.copy(streaming = false, status = "停止を要求しました。", live = LiveAnswer()) }
                loadMessages(id)
                scheduleSync()
                return@launch
            }
            backend.post("/api/stop_chat", JSONObject().put("thread_id", id).apply { state.value.jobId?.let { put("job_id", it) } }, token())
            streamJob?.cancelAndJoin()
            mutable.update { it.copy(streaming = false, status = "停止を要求しました。") }
            delay(1000); loadMessages(id)
        } catch (e: Exception) { report(e) }
    } }

    fun resolveMcpDecision(allow: Boolean) { viewModelScope.launch {
        if (state.value.offline) { notify("オフライン中はMCPの確認に応答できません。"); return@launch }
        val decision = state.value.mcpDecision ?: return@launch
        if (decision.jobId.isBlank()) return@launch
        mutable.update { it.copy(mcpDecision = null) }
        try {
            backend.post("/api/mcp/chat/${URLEncoder.encode(decision.jobId, "UTF-8")}/decision",
                JSONObject().put("decision", if (allow) "allow" else "deny").put("id", decision.id), token())
        } catch (e: Exception) { report(e) }
    } }

    private suspend fun fetchBatchJobs(notify: Boolean, status: Boolean = notify) {
        val previous = state.value.batchJobs.associateBy { it.id }
        if (status) runCatching { pollBatchStatus() }
        val jobs = parseBatchJobs(backend.get("/api/batch/jobs", token()))
        // The system notification is added on Android (ANDROID_ONLY.md); the in-app banner follows Web.
        if (notify) jobs.filter { !it.active && previous[it.id]?.active == true }.forEach { job ->
            notifyBatchCompletion(getApplication<Application>(), job)
        }
        mutable.update { it.copy(batchJobs = jobs, batchBusy = false) }
    }

    /**
     * Web `refreshGeminiBatchStatus`: lets the server collect finished jobs, shows the banner for them and
     * reloads the open chat when one of them belongs to it. Returns whether any job finished.
     */
    private suspend fun pollBatchStatus(): Boolean {
        val status = backend.get("/api/gemini/batch/status", token())
        val completed = status.optJSONArray("completed") ?: return false
        if (completed.length() == 0) return false
        val first = completed.getJSONObject(0)
        val text = if (completed.length() == 1) "${first.optString("model")} のBatch処理が完了しました。"
            else "${completed.length()}件のBatch処理が完了しました。"
        mutable.update { it.copy(batchBanner = text to first.optString("thread_id")) }
        val current = state.value.selected?.id
        if (current != null && (0 until completed.length()).any { completed.getJSONObject(it).optString("thread_id") == current }) {
            runCatching { loadMessages(current, autoResume = false) }
        }
        return true
    }

    /** Web `#batch-notification-open` / `#batch-notification-close`. */
    fun dismissBatchBanner(open: Boolean) {
        val banner = state.value.batchBanner ?: return
        mutable.update { it.copy(batchBanner = null) }
        if (open && banner.second.isNotBlank()) openThreadId(banner.second)
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
            // Web checks every 2 seconds; the job list itself is refreshed on completion and every 30 seconds.
            var tick = 0
            while (foreground && session != null) {
                if (state.value.batchJobs.any { it.active }) {
                    val finished = runCatching { pollBatchStatus() }.getOrDefault(false)
                    if (finished || tick % 15 == 0) runCatching { fetchBatchJobs(notify = true, status = false) }
                }
                tick++
                delay(2_000)
            }
        }
    }

    /** Web `loadBatchJobs`; [silent] skips the failure toast (the 5-second refresh in the modal). */
    fun loadBatchJobs(silent: Boolean) { viewModelScope.launch {
        try { fetchBatchJobs(notify = true) }
        catch (e: CancellationException) { throw e }
        catch (e: Exception) { if (!silent) notify("Batch処理の履歴を取得できませんでした") }
    } }

    fun cancelBatchJob(job: BatchJob) { viewModelScope.launch {
        try { backend.post("/api/batch/jobs/${job.id}/cancel", JSONObject(), token()) }
        catch (e: ApiException) { notify(e.payload.optString("error").ifBlank { "Batch処理を停止できませんでした" }); return@launch }
        catch (e: Exception) { notify("Batch処理を停止できませんでした"); return@launch }
        notify("Batch処理を停止しました")
        runCatching { fetchBatchJobs(notify = false) }
        if (state.value.selected?.id == job.threadId) runCatching { loadMessages(job.threadId) }
    } }

    fun deleteBatchJob(job: BatchJob) { viewModelScope.launch {
        try { backend.delete("/api/batch/jobs/${job.id}", token()) }
        catch (e: ApiException) { notify(e.payload.optString("error").ifBlank { "Batch履歴を削除できませんでした" }); return@launch }
        catch (e: Exception) { notify("Batch履歴を削除できませんでした"); return@launch }
        notify("Batch履歴を削除しました")
        runCatching { fetchBatchJobs(notify = false) }
    } }

    fun openThreadId(id: String) {
        val item = state.value.threads.firstOrNull { it.id == id } ?: ThreadItem(id, "Batchチャット", "")
        openThread(item)
    }

    fun openBubbleTarget(id: String?) {
        viewModelScope.launch {
            state.first { !it.starting }
            if (state.value.account == null) return@launch
            val threadId = bubbleThreadId(id)
            if (threadId == null) newChat() else openThreadId(threadId)
        }
    }

    // --- Native realtime audio sessions ---

    fun startRealtime(
        modelId: String,
        voice: String = "alloy",
        targetLanguage: String = "ja",
        thinkingLevel: String = "minimal",
        transcriptionMode: String = "VERBATIM",
        customVocabulary: String = "",
        includeThoughts: Boolean = false,
        speed: Float? = null,
        rateIn: Int? = null,
        rateOut: Int? = null,
        autoPlay: Boolean = true,
        reasoningEffort: String? = null,
    ) {
        if (state.value.offline) { notify("オフライン中はRealtimeを開始できません。"); return }
        rtAutoPlay = autoPlay
        rtResponseDone = 0
        rtLastAudioAt = 0L
        rtSpeechActive = false
        rtStreamError = null
        if (state.value.realtime.active || modelId.isBlank()) return
        rtStopping = false
        mutable.update { it.copy(realtime = RealtimeState(model = modelId, status = "接続中...")) }
        viewModelScope.launch {
            try {
                val started = backend.post("/api/realtime/start", JSONObject()
                    .put("model", modelId)
                    .put("voice", voice)
                    .put("target_lang", targetLanguage.trim().lowercase().take(16).ifBlank { "ja" })
                    .put("thinking_level", thinkingLevel.trim().lowercase().ifBlank { "minimal" })
                    .put("include_thoughts", includeThoughts)
                    .put("transcription_mode", transcriptionMode.trim().uppercase().ifBlank { "VERBATIM" })
                    .put("custom_vocabulary", JSONArray(customVocabulary.split(',', '、', '\n')
                        .map { it.trim() }.filter { it.isNotBlank() }.take(1000)))
                    .apply {
                        speed?.let { put("speed", it.toDouble()) }
                        rateIn?.let { put("rate_in", it) }
                        rateOut?.let { put("rate_out", it) }
                        reasoningEffort?.let { put("reasoning_effort", it) }
                    }, token())
                val sessionId = started.getString("session_id")
                val rateOut = started.optInt("rate_out", 24000).coerceIn(8000, 48000)
                realtimeTrack = createAudioTrack(rateOut, stereo = false)
                mutable.update { it.copy(realtime = RealtimeState(true, modelId, sessionId, "話してください...")) }
                realtimeCaptureJob = viewModelScope.launch(Dispatchers.IO) {
                    try { captureRealtimeAudio(sessionId, started.optInt("rate_in", rateOut).coerceIn(8000, 48000)) }
                    catch (e: CancellationException) { throw e }
                    catch (e: Exception) {
                        mutable.update { it.copy(realtime = it.realtime.copy(status = "マイクエラー", error = e.message)) }
                        notify("マイクを利用できません: " + (e.message ?: ""))
                    }
                }
                realtimeStreamJob = viewModelScope.launch(Dispatchers.IO) {
                    try {
                        api.streamSse("/api/realtime/stream?session_id=${URLEncoder.encode(sessionId, "UTF-8")}", token()) { event ->
                            handleRealtimeEvent(event, rateOut)
                        }
                    } catch (e: CancellationException) { throw e }
                    catch (e: Exception) {
                        if (state.value.realtime.sessionId == sessionId && !rtStopping) {
                            mutable.update { it.copy(realtime = it.realtime.copy(status = "ストリームエラー")) }
                            notify("リアルタイム接続が切断されました")
                        }
                    }
                }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                val message = (e as? ApiException)?.payload?.optString("error")?.ifBlank { null } ?: e.message ?: "セッション開始に失敗しました"
                mutable.update { it.copy(realtime = RealtimeState(status = "接続エラー")) }
                notify("リアルタイムセッションを開始できませんでした: $message")
            }
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

    private var rtAutoPlay = true
    private var rtResponseDone = 0
    private var rtLastAudioAt = 0L
    private var rtSpeechActive = false
    private var rtStreamError: String? = null
    private var rtStopping = false

    private fun setRealtimeStatus(text: String) = mutable.update { it.copy(realtime = it.realtime.copy(status = text)) }

    /** Web `RealtimeVoiceSession._handleEvent`. */
    private fun handleRealtimeEvent(event: JSONObject, rateOut: Int) {
        when (event.optString("type")) {
            "status" -> if (event.optString("status") == "ready" && state.value.realtime.active && !rtStopping) setRealtimeStatus("話してください...")
            "audio" -> {
                rtLastAudioAt = System.currentTimeMillis()
                if (!rtAutoPlay) return
                val encoded = event.nullableString("data")
                val bytes = runCatching { android.util.Base64.decode(encoded, android.util.Base64.DEFAULT) }.getOrNull() ?: ByteArray(0)
                if (bytes.isNotEmpty()) {
                    realtimeTrack?.write(bytes, 0, bytes.size, AudioTrack.WRITE_NON_BLOCKING)
                    mutable.update { it.copy(realtime = it.realtime.copy(audioBytes = it.realtime.audioBytes + bytes.size,
                        status = if (rtStopping) it.realtime.status else "再生中...")) }
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
            "speech_started" -> {
                rtSpeechActive = true
                realtimeTrack?.let { track -> runCatching { track.pause(); track.flush(); track.play() } }
                if (!rtStopping) setRealtimeStatus("聞き取り中...")
            }
            "speech_stopped" -> { rtSpeechActive = false; if (!rtStopping) setRealtimeStatus("応答待ち...") }
            "interrupted" -> realtimeTrack?.let { track -> runCatching { track.pause(); track.flush(); track.play() } }
            "response_done", "turn_complete" -> rtResponseDone += 1
            // Recoverable provider error: the session stays open (Web shows a warning toast).
            "notice" -> event.nullableString("message").takeIf { it.isNotBlank() }?.let { notify("リアルタイム音声: $it") }
            "error" -> {
                rtStreamError = event.nullableString("message").ifBlank { "リアルタイムエラー" }
                mutable.update { it.copy(realtime = it.realtime.copy(status = "エラー", error = rtStreamError)) }
            }
            "final" -> if (state.value.realtime.active && !rtStopping) stopRealtime(save = true)
        }
    }

    /**
     * Web `RealtimeVoiceSession.stop` / `_cancel`: saving commits the last audio, waits for the reply (or a
     * quiet moment, up to 20s) and stores the conversation in the chat; cancelling discards it.
     */
    fun stopRealtime(save: Boolean) {
        if (stsRecorder != null) { stopOneShotSts(save); return }
        val current = state.value.realtime
        if (!current.active && current.sessionId.isBlank()) return
        if (rtStopping) return
        rtStopping = true
        viewModelScope.launch {
            val sid = current.sessionId
            realtimeCaptureJob?.cancelAndJoin()
            if (!save) {
                realtimeStreamJob?.cancelAndJoin()
                if (sid.isNotBlank()) runCatching { backend.post("/api/realtime/cancel", JSONObject().put("session_id", sid), token()) }
                finishRealtime("Canceled", 800)
                return@launch
            }
            setRealtimeStatus("応答を待っています...")
            runCatching { backend.post("/api/realtime/commit", JSONObject().put("session_id", sid), token()) }
            val startedAt = System.currentTimeMillis()
            val before = rtResponseDone
            var lastActivity = rtLastAudioAt
            while (System.currentTimeMillis() - startedAt < 20_000) {
                if (rtResponseDone > before) break
                if (rtLastAudioAt > lastActivity) lastActivity = rtLastAudioAt
                val now = System.currentTimeMillis()
                if (!rtSpeechActive && now - startedAt > 2_000 && now - lastActivity > 2_500) break
                delay(250)
            }
            realtimeStreamJob?.cancelAndJoin()
            try {
                val reply = backend.post("/api/realtime/save", JSONObject().put("session_id", sid)
                    .apply { state.value.selected?.id?.let { put("thread_id", it) } }, token())
                val error = rtStreamError
                if (error != null) {
                    finishRealtime("エラー", 0)
                    notify("リアルタイム会話でエラーが発生しました: $error")
                } else {
                    finishRealtime("保存しました", 1200)
                    val threadId = reply.optString("thread_id").ifBlank { state.value.selected?.id.orEmpty() }
                    if (threadId.isNotBlank()) {
                        if (state.value.selected?.id == threadId) runCatching { loadMessages(threadId) } else openThreadId(threadId)
                    }
                }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                finishRealtime("保存エラー", 0)
                notify("音声会話の保存に失敗しました: " + ((e as? ApiException)?.payload?.optString("error")?.ifBlank { null } ?: e.message ?: ""))
            }
        }
    }

    /** Releases the session and shows [status]; after [resetMillis] the dock returns to "Tap to speak". */
    // --- Web one-shot speech-to-speech (`/sts`) for transcription models ---

    private var stsRecorder: android.media.MediaRecorder? = null
    private var stsFile: File? = null
    private var stsRequest: Map<String, String> = emptyMap()
    private var stsAutoPlay = true
    private var stsAutoRestart = false

    /**
     * Web `mic-btn` for models that are not realtime sessions (gpt-transcribe, gpt-live-transcribe,
     * gpt-realtime-whisper): records until the next tap, then sends the clip to `/sts`.
     */
    fun startOneShotSts(modelId: String, fields: Map<String, String>, autoPlay: Boolean, autoRestart: Boolean) {
        if (state.value.offline) { notify("オフライン中はRealtimeを開始できません。"); return }
        if (stsRecorder != null || state.value.realtime.active) return
        stsRequest = fields + ("model" to modelId)
        stsAutoPlay = autoPlay
        stsAutoRestart = autoRestart
        val app = getApplication<Application>()
        val file = File(app.cacheDir, "sts_recording.m4a")
        try {
            val recorder = if (Build.VERSION.SDK_INT >= 31) android.media.MediaRecorder(app) else android.media.MediaRecorder()
            recorder.setAudioSource(android.media.MediaRecorder.AudioSource.MIC)
            recorder.setOutputFormat(android.media.MediaRecorder.OutputFormat.MPEG_4)
            recorder.setAudioEncoder(android.media.MediaRecorder.AudioEncoder.AAC)
            recorder.setAudioSamplingRate(44_100)
            recorder.setAudioEncodingBitRate(96_000)
            recorder.setOutputFile(file.absolutePath)
            recorder.prepare()
            recorder.start()
            stsRecorder = recorder
            stsFile = file
        } catch (e: Exception) {
            stsRecorder?.release()
            stsRecorder = null
            notify("Microphone access denied or not available.")
            return
        }
        mutable.update { it.copy(realtime = RealtimeState(active = true, model = modelId, status = "Recording... Tap to stop")) }
    }

    private fun stopOneShotSts(save: Boolean) {
        val recorder = stsRecorder ?: return
        stsRecorder = null
        runCatching { recorder.stop() }
        recorder.release()
        val file = stsFile
        val modelId = stsRequest["model"].orEmpty()
        if (!save || file == null || !file.exists() || file.length() == 0L) {
            file?.delete()
            viewModelScope.launch { finishRealtime("Canceled", 800) }
            return
        }
        mutable.update { it.copy(realtime = it.realtime.copy(active = false, status = "Sending audio...")) }
        ensureThread {
            viewModelScope.launch {
                val threadId = state.value.selected?.id.orEmpty()
                var track: AudioTrack? = null
                var firstAudio = true
                var saved = false
                try {
                    val transcription = modelId == "gpt-transcribe" || modelId == "gpt-live-transcribe" || modelId == "gpt-realtime-whisper"
                    mutable.update { it.copy(realtime = it.realtime.copy(status = if (transcription) "Transcribing..." else "Processing audio...")) }
                    val rate = stsRequest["sts_rate_out"]?.toIntOrNull() ?: 24000
                    withContext(Dispatchers.IO) {
                        api.sts(file, stsRequest + ("thread_id" to threadId), token()) { chunk ->
                            chunk.optString("error").takeIf { it.isNotBlank() && !chunk.isNull("error") }?.let { throw IOException(it) }
                            val audio = chunk.optString("audio_delta")
                            if (audio.isNotEmpty() && stsAutoPlay) {
                                if (firstAudio) {
                                    firstAudio = false
                                    mutable.update { it.copy(realtime = it.realtime.copy(status = "Playing response...")) }
                                }
                                val pcm = android.util.Base64.decode(audio, android.util.Base64.DEFAULT)
                                val out = track ?: createAudioTrack(rate, stereo = false).also { track = it }
                                out.write(pcm, 0, pcm.size)
                            }
                            chunk.optString("input_delta").takeIf { it.isNotEmpty() }?.let { delta ->
                                mutable.update { it.copy(realtime = it.realtime.copy(userText = it.realtime.userText + delta)) }
                            }
                            chunk.optString("transcript_delta").takeIf { it.isNotEmpty() }?.let { delta ->
                                mutable.update { it.copy(realtime = it.realtime.copy(assistantText = it.realtime.assistantText + delta)) }
                            }
                            if (chunk.optBoolean("final") || chunk.has("audio_url")) saved = true
                        }
                    }
                    if (saved && threadId.isNotBlank()) runCatching { loadMessages(threadId) }
                } catch (e: CancellationException) { throw e }
                catch (e: Exception) {
                    notify("Audio processing error: " + ((e as? ApiException)?.payload?.optString("error")?.ifBlank { null } ?: e.message ?: ""))
                    saved = false
                } finally {
                    file.delete()
                    // stop() lets the written audio finish; release it once that has played.
                    track?.let { played -> runCatching { played.stop() }; viewModelScope.launch { delay(1500); runCatching { played.release() } } }
                }
                if (saved && stsAutoRestart && state.value.model == modelId) {
                    mutable.update { it.copy(realtime = RealtimeState(model = modelId, status = "Listening...")) }
                    delay(500)
                    startOneShotSts(modelId, stsRequest - "model" - "thread_id", stsAutoPlay, stsAutoRestart)
                } else mutable.update { it.copy(realtime = RealtimeState(model = modelId, status = "Tap to speak")) }
            }
        }
    }

    private suspend fun finishRealtime(status: String, resetMillis: Long) {
        realtimeTrack?.let { track -> runCatching { track.stop(); track.release() } }; realtimeTrack = null
        rtStopping = false
        mutable.update { it.copy(realtime = RealtimeState(status = status)) }
        if (resetMillis > 0) {
            delay(resetMillis)
            mutable.update { if (!it.realtime.active && it.realtime.status == status) it.copy(realtime = RealtimeState()) else it }
        }
    }


    // --- Native Lyria RealTime studio ---

    /** Web `openLyriaStudio(promptText)`: the sent text becomes the studio's first prompt (while no session runs). */
    fun prepareLyriaPrompt(text: String) {
        mutable.update { if (it.lyria.active) it else it.copy(lyria = it.lyria.copy(prompt = text)) }
    }

    private fun setLyriaStatus(text: String, kind: String) =
        mutable.update { it.copy(lyria = it.lyria.copy(status = text, kind = kind)) }

    /** Web `collectPrompts`: non-empty rows with their weights. */
    private fun lyriaPromptsJson(prompts: List<LyriaPrompt>): JSONArray = JSONArray().apply {
        prompts.filter { it.text.isNotBlank() }.forEach { put(JSONObject().put("text", it.text.trim().take(4000)).put("weight", it.weight.toDouble())) }
    }

    /** Web `startSession`: starts a Lyria RealTime session with the weighted prompts and music settings. */
    fun lyriaStart(prompts: List<LyriaPrompt>, config: JSONObject) {
        if (state.value.lyria.busy) return
        val weighted = lyriaPromptsJson(prompts)
        if (weighted.length() == 0) { notify("プロンプトを入力してください"); return }
        mutable.update { it.copy(lyria = it.lyria.copy(busy = true, status = "接続中...", kind = "connecting")) }
        viewModelScope.launch {
            try {
                val started = backend.post("/api/gemini/music/start", JSONObject().put("weighted_prompts", weighted).put("config", config), token())
                val sid = started.getString("session_id")
                lyriaTrack?.let { track -> runCatching { track.stop(); track.release() } }
                lyriaTrack = createAudioTrack(48000, stereo = true)
                mutable.update { it.copy(lyria = it.lyria.copy(sessionId = sid, startedAt = 0L, status = "接続中...", kind = "connecting")) }
                openLyriaStream(sid)
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                val message = (e as? ApiException)?.payload?.optString("error")?.ifBlank { null } ?: e.message ?: "セッション開始に失敗しました"
                setLyriaStatus("エラー: $message", "error")
                notify("Lyria RealTime: $message")
            } finally { mutable.update { it.copy(lyria = it.lyria.copy(busy = false)) } }
        }
    }

    /** Web `openStream`: server events carry audio, status snapshots, errors and the end; a dropped stream reconnects. */
    private fun openLyriaStream(sid: String) {
        lyriaStreamJob?.cancel()
        lyriaStreamJob = viewModelScope.launch(Dispatchers.IO) {
            while (state.value.lyria.sessionId == sid) {
                try {
                    api.streamSse("/api/gemini/music/stream?session_id=${URLEncoder.encode(sid, "UTF-8")}", token()) { event -> handleLyriaEvent(event) }
                    break
                } catch (e: CancellationException) { throw e }
                catch (e: Exception) {
                    if (state.value.lyria.sessionId != sid) break
                    setLyriaStatus("ストリーム切断。再接続します…", "connecting")
                    delay(1200)
                }
            }
        }
    }

    private fun handleLyriaEvent(event: JSONObject) {
        if (event.optBoolean("snapshot")) {
            when (val status = event.optString("status")) {
                "error" -> setLyriaStatus("エラー", "error")
                "closed", "stopped" -> setLyriaStatus("終了", "closed")
                else -> if (status == "paused") setLyriaStatus("一時停止中", "paused") else setLyriaStatus("接続中...", "connecting")
            }
            return
        }
        val encoded = event.nullableString("audio")
        if (encoded.isNotBlank()) {
            val bytes = runCatching { android.util.Base64.decode(encoded, android.util.Base64.DEFAULT) }.getOrNull() ?: ByteArray(0)
            if (bytes.isNotEmpty()) lyriaTrack?.write(bytes, 0, bytes.size, AudioTrack.WRITE_NON_BLOCKING)
            mutable.update { it.copy(lyria = it.lyria.copy(status = "再生中...", kind = "streaming",
                startedAt = if (it.lyria.startedAt == 0L) System.currentTimeMillis() else it.lyria.startedAt)) }
            return
        }
        event.nullableString("error").takeIf { it.isNotBlank() }?.let { error -> setLyriaStatus("エラー: $error", "error"); return }
        if (event.optBoolean("final")) setLyriaStatus("終了", "closed")
    }

    private suspend fun lyriaCommand(type: String, fill: JSONObject.() -> Unit) {
        val sid = state.value.lyria.sessionId
        backend.post("/api/gemini/music/command", JSONObject().put("session_id", sid).put("type", type).apply(fill), token())
    }

    private fun lyriaErrorMessage(e: Exception, fallback: String) =
        (e as? ApiException)?.payload?.optString("error")?.ifBlank { null } ?: e.message ?: fallback

    /** Web `control`: PLAY / PAUSE / STOP / RESET_CONTEXT. */
    fun lyriaControl(action: String) {
        if (!state.value.lyria.active) return
        mutable.update { it.copy(lyria = it.lyria.copy(busy = true)) }
        viewModelScope.launch {
            try {
                lyriaCommand("control") { put("action", action) }
                when (action) {
                    "PLAY" -> setLyriaStatus("再生中...", "streaming")
                    "PAUSE" -> setLyriaStatus("一時停止中", "paused")
                    "STOP" -> setLyriaStatus("停止中", "stopped")
                    "RESET_CONTEXT" -> setLyriaStatus("コンテキストをリセット...", "connecting")
                }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                val message = lyriaErrorMessage(e, "コマンド送信に失敗しました")
                notify("Lyria RealTime: $message")
                setLyriaStatus("エラー: $message", "error")
            } finally { mutable.update { it.copy(lyria = it.lyria.copy(busy = false)) } }
        }
    }

    /** Web `applyPrompts`. */
    fun lyriaApplyPrompts(prompts: List<LyriaPrompt>) {
        if (!state.value.lyria.active) return
        val weighted = lyriaPromptsJson(prompts)
        if (weighted.length() == 0) { notify("プロンプトを入力してください"); return }
        mutable.update { it.copy(lyria = it.lyria.copy(busy = true)) }
        viewModelScope.launch {
            try {
                lyriaCommand("prompts") { put("weighted_prompts", weighted) }
                setLyriaStatus("プロンプトを適用しました", if (state.value.lyria.kind == "paused") "paused" else "streaming")
                notify("プロンプトを適用しました")
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) { notify("Lyria RealTime: " + lyriaErrorMessage(e, "コマンド送信に失敗しました")) }
            finally { mutable.update { it.copy(lyria = it.lyria.copy(busy = false)) } }
        }
    }

    /** Web `applyConfig`: BPM or scale changes reset the context. */
    fun lyriaApplyConfig(config: JSONObject, resetContext: Boolean) {
        if (!state.value.lyria.active) return
        mutable.update { it.copy(lyria = it.lyria.copy(busy = true)) }
        viewModelScope.launch {
            try {
                lyriaCommand("config") { put("config", config).put("reset_context", resetContext) }
                val text = if (resetContext) "設定を適用しました（コンテキストをリセット）" else "設定を適用しました"
                setLyriaStatus(text, if (state.value.lyria.kind == "paused") "paused" else "streaming")
                notify(text)
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) { notify("Lyria RealTime: " + lyriaErrorMessage(e, "コマンド送信に失敗しました")) }
            finally { mutable.update { it.copy(lyria = it.lyria.copy(busy = false)) } }
        }
    }

    /** Web `saveSession`: stores the played audio in the chat as WAV, opens that chat and closes the studio. */
    fun lyriaSave(onSaved: () -> Unit) {
        val sid = state.value.lyria.sessionId.takeIf { it.isNotBlank() } ?: return
        mutable.update { it.copy(lyria = it.lyria.copy(busy = true, status = "保存中...", kind = "connecting")) }
        viewModelScope.launch {
            try {
                val data = backend.post("/api/gemini/music/save", JSONObject().put("session_id", sid)
                    .put("thread_id", state.value.selected?.id ?: JSONObject.NULL), token())
                setLyriaStatus("保存しました", "closed")
                notify("チャットに保存しました")
                closeLyriaSession(cancel = false)
                val threadId = data.optString("thread_id").ifBlank { state.value.selected?.id.orEmpty() }
                if (threadId.isNotBlank()) openThreadId(threadId)
                onSaved()
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                val message = lyriaErrorMessage(e, "保存に失敗しました")
                setLyriaStatus("エラー: $message", "error")
                notify("Lyria RealTime: $message")
            } finally { mutable.update { it.copy(lyria = it.lyria.copy(busy = false)) } }
        }
    }

    /** Web `closeAndCleanup`: closing the studio cancels the session. */
    fun stopLyria(save: Boolean = false) {
        if (save) return lyriaSave {}
        closeLyriaSession(cancel = true)
    }

    private fun closeLyriaSession(cancel: Boolean) {
        val sid = state.value.lyria.sessionId
        lyriaStreamJob?.cancel()
        lyriaStreamJob = null
        if (cancel && sid.isNotBlank()) viewModelScope.launch {
            runCatching { backend.post("/api/gemini/music/cancel", JSONObject().put("session_id", sid), token()) }
        }
        lyriaTrack?.let { track -> runCatching { track.stop(); track.release() } }; lyriaTrack = null
        mutable.update { it.copy(lyria = LyriaState()) }
    }

    fun logout() { viewModelScope.launch {
        if (state.value.localProfile) { leaveLocalProfile(); return@launch }
        try {
            backend.post("/api/mobile/v1/revoke", JSONObject(), token())
            clearSession()
        } catch (e: Exception) {
            // Keep the Web BAN screen's logout escape hatch even if revoke fails.
            if (state.value.banned) clearSession() else report(e)
        }
    } }
    private suspend fun clearSession() {
        chatHistory.clear()
        val caller = currentCoroutineContext().job
        listOf(pairingJob, navigationJob, streamJob, uploadJob, heartbeatJob, libraryJob, cacheSyncJob, batchPollJob,
            realtimeStreamJob, realtimeCaptureJob, lyriaStreamJob, importJob).forEach { if (it !== caller) it?.cancel() }
        realtimeTrack?.let { track -> runCatching { track.stop(); track.release() } }; realtimeTrack = null
        lyriaTrack?.let { track -> runCatching { track.stop(); track.release() } }; lyriaTrack = null
        withContext(NonCancellable + Dispatchers.IO) { store.clear() }
        closeLocalChats()
        GoogleAuthClient.clearCredentialState(getApplication())
        cancelChatBubble(getApplication())
        session = null; failed = null
        val server = state.value
        mutable.value = ChatState(
            starting = false,
            serverLabel = server.serverLabel, serverInfo = server.serverInfo, savedServers = server.savedServers,
            googleServerClientId = server.googleServerClientId, integrityProjectNumber = server.integrityProjectNumber,
            historyCacheMode = HistoryCacheMode.from(prefs.getString("offline_history_cache_mode", HistoryCacheMode.VIEWED.value)),
            cacheMobileDataAllowed = prefs.getBoolean("offline_cache_mobile_data", false),
        )
    }
    /** [message] replaces the error text for the Web toasts that use a fixed wording. */
    private suspend fun report(error: Throwable, message: String? = null) {
        if (error is CancellationException) throw error
        if (error is ApiException && error.code == "banned") {
            enterBannedState(error)
            return
        }
        if (error is ApiException && error.status == 401 && !state.value.localProfile) clearSession()
        val networkFailure = error !is ApiException && (error is java.net.ConnectException || error is java.net.UnknownHostException ||
            error is java.net.SocketTimeoutException || error is java.net.SocketException || error is IOException
        )
        when {
            error is ApiException && connectionStatusForHttp(error.status) != null ->
                setConnectionUnavailable(connectionStatusForHttp(error.status)!!)
            error is ApiException && error.status >= 500 ->
                setConnectionUnavailable(ConnectionStatus.UNSTABLE)
            networkFailure ->
                setConnectionUnavailable(if (hasUsableNetwork()) ConnectionStatus.UNSTABLE else ConnectionStatus.OFFLINE)
        }
        mutable.update { it.copy(
            notice = message ?: error.message?.take(500) ?: "通信に失敗しました。再試行してください。",
        ) }
    }

    private suspend fun refreshOfflineCacheStats(accountId: Int? = state.value.account?.id) {
        val stats = accountId?.let { id -> withContext(Dispatchers.IO) { offlineCache.stats(id) } } ?: OfflineCacheStats()
        mutable.update { it.copy(offlineCacheStats = stats) }
    }

    fun saveCacheSettings(mode: HistoryCacheMode, mobileDataAllowed: Boolean) {
        prefs.edit()
            .putString("offline_history_cache_mode", mode.value)
            .putBoolean("offline_cache_mobile_data", mobileDataAllowed)
            .apply()
        mutable.update { it.copy(historyCacheMode = mode, cacheMobileDataAllowed = mobileDataAllowed) }
        if (mode == HistoryCacheMode.FULL) startCacheSyncIfAllowed()
    }

    fun clearOfflineCache(category: CacheCategory) {
        val accountId = state.value.account?.id ?: return
        viewModelScope.launch {
            withContext(Dispatchers.IO) { offlineCache.clear(accountId, category) }
            if (category == CacheCategory.CHAT_HISTORY) {
                mutable.update { it.copy(threads = emptyList(), selected = null, messages = emptyList(), allMessages = emptyList(), hasOlder = false) }
            } else {
                mutable.update { it.copy(library = emptyList(), libraryHasMore = false, libraryTotal = 0) }
            }
            refreshOfflineCacheStats(accountId)
            notify(if (category == CacheCategory.CHAT_HISTORY) "チャット履歴キャッシュを削除しました。" else "ファイルキャッシュを削除しました。")
        }
    }

    fun syncOfflineCache() {
        if (cacheSyncJob?.isActive == true) return
        val accountId = state.value.account?.id ?: return
        if (!cacheNetworkAllowed()) {
            notify(if (state.value.cacheMobileDataAllowed) "ネットワークに接続してから同期してください。" else "Wi‑Fi接続時に同期できます。")
            return
        }
        cacheSyncJob = viewModelScope.launch {
            mutable.update { it.copy(cacheSyncing = true, cacheSyncProgress = 0, cacheSyncTotal = 0) }
            try {
                performFullCacheSync(accountId)
                notify("キャッシュの全件同期が完了しました。")
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                report(e)
            } finally {
                mutable.update { it.copy(cacheSyncing = false) }
                refreshOfflineCacheStats(accountId)
            }
        }
    }

    fun cancelCacheSync() { cacheSyncJob?.cancel(); mutable.update { it.copy(cacheSyncing = false) } }

    private fun startCacheSyncIfAllowed() {
        if (state.value.historyCacheMode == HistoryCacheMode.FULL && cacheNetworkAllowed()) syncOfflineCache()
    }

    private fun cacheNetworkAllowed(): Boolean {
        val manager = getApplication<Application>().getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager
            ?: return false
        val network = manager.activeNetwork ?: return false
        val capabilities = manager.getNetworkCapabilities(network) ?: return false
        if (!capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)) return false
        if (capabilities.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) ||
            capabilities.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET)) return true
        return state.value.cacheMobileDataAllowed && capabilities.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR)
    }

    private suspend fun performFullCacheSync(accountId: Int) {
        val threads = mutableListOf<ThreadItem>()
        var page = 1
        var hasNext: Boolean
        do {
            val reply = backend.get("/api/threads?page=$page&q=", token())
            val rows = reply.getJSONArray("threads")
            val items = (0 until rows.length()).map { parseThreadItem(rows.getJSONObject(it)) }
            threads += items
            withContext(Dispatchers.IO) { offlineCache.saveThreads(accountId, items, merge = page != 1) }
            hasNext = reply.optBoolean("has_next")
            page = reply.optInt("next_page", page + 1)
        } while (hasNext && currentCoroutineContext().isActive)

        val uniqueThreads = threads.distinctBy { it.id }
        mutable.update { it.copy(cacheSyncTotal = uniqueThreads.size.coerceAtLeast(1), cacheSyncProgress = 0) }
        uniqueThreads.forEachIndexed { index, thread ->
            var before: String? = null
            var older: Boolean
            val messages = LinkedHashMap<String, ChatMessage>()
            do {
                val suffix = before?.let { "&before_id=${URLEncoder.encode(it, "UTF-8")}" }.orEmpty()
                val reply = backend.get("/api/threads/${thread.id}?limit=200$suffix", token())
                parseMessages(reply).forEach { messages[it.id] = it }
                older = reply.optBoolean("has_older_messages")
                before = reply.nullableString("oldest_loaded_id").ifBlank { null }
                withContext(Dispatchers.IO) {
                    offlineCache.saveThread(
                        accountId, thread, messages.values.toList(), older, before,
                        reply.nullableString("custom_instruction"), reply.optBoolean("include_global_instruction", true),
                        reply.optLong("temp_chat_remaining_seconds").takeIf { value -> !reply.isNull("temp_chat_remaining_seconds") && value >= 0 },
                        messages.values.mapNotNull { numericId(it) }.maxOrNull(),
                    )
                }
            } while (older && !before.isNullOrBlank() && currentCoroutineContext().isActive)
            mutable.update { it.copy(cacheSyncProgress = index + 1) }
        }

        var offset = 0
        var moreFiles: Boolean
        do {
            val reply = backend.get("/api/files?limit=40&offset=$offset&sort=newest&q=")
            val files = parseLibraryFiles(reply)
            withContext(Dispatchers.IO) { offlineCache.saveLibrary(accountId, files) }
            moreFiles = reply.optBoolean("has_more")
            offset += files.size
        } while (moreFiles && offset > 0 && currentCoroutineContext().isActive)

        val cachedThreads = withContext(Dispatchers.IO) { offlineCache.loadThreads(accountId) }
        mutable.update { current ->
            val query = current.search.trim()
            val visible = if (query.isBlank()) cachedThreads else cachedThreads.filter {
                it.title.contains(query, true) || it.model.contains(query, true)
            }
            current.copy(threads = visible, nextPage = null, offline = false)
        }
    }

    /** Clears the offline banner and retries the last account or history load. */
    fun reconnect() {
        connectionRecoveredHideJob?.cancel()
        mutable.update { it.copy(offline = false, connectionStatus = ConnectionStatus.UNKNOWN, connectionMessage = "", connectionBannerVisible = false) }
        viewModelScope.launch {
            try {
                if (session == null) pair() else {
                    loadAccount()
                    state.value.selected?.let { loadMessages(it.id) }
                }
                probeServerConnection()
            } catch (e: Exception) { report(e) }
        }
    }
    /** Loads same-origin attachment bytes for preview; returns null when unavailable. */
    suspend fun loadAttachmentBytes(reference: String, thumbnail: Boolean, limit: Long = 8L * 1024 * 1024): ByteArray? {
        if (state.value.banned) return null
        // Images on other sites in answers load like Web `<img>`: without the token, https only.
        if (reference.startsWith("https://") && fileReferencePath(reference) == null) {
            if (state.value.offline) return null
            return withContext(Dispatchers.IO) { runCatching { api.loadExternalImage(reference, limit) }.getOrNull() }
        }
        if (LocalChatStore.isLocalReference(reference)) {
            val store = localChats ?: return null
            return withContext(Dispatchers.IO) { store.loadFile(reference, maxOf(limit, 64L * 1024 * 1024)) }
        }
        val accountId = state.value.account?.id ?: return null
        val cached = withContext(Dispatchers.IO) { offlineCache.loadFile(accountId, reference, thumbnail, limit) }
        if (cached != null) return cached.bytes
        val active = session?.token ?: return null
        return withContext(Dispatchers.IO) {
            runCatching {
                val bytes = api.loadFileBytes(reference, active, thumbnail, limit)
                offlineCache.saveFile(accountId, reference, thumbnail, bytes, mimeForReference(reference))
                refreshOfflineCacheStats(accountId)
                bytes
            }.getOrNull()
        }
    }
    fun upload(uris: List<Uri>) {
        if (state.value.banned) return
        if (state.value.offline && !uploadsLocal) { mutable.update { it.copy(notice = "オフライン中はファイルをアップロードできません。") }; return }
        if (uris.isEmpty() || state.value.uploading) return
        uploadJob = viewModelScope.launch {
            mutable.update { it.copy(uploading = true, uploadSent = 0, uploadTotal = 0, uploadName = "",
                uploadCompleted = 0, uploadCount = uris.size) }
            try {
                for (uri in uris) {
                    val resolver = getApplication<Application>().contentResolver
                    val local = withContext(Dispatchers.IO) { queryLocalAttachment(resolver, uri) }
                    mutable.update { it.copy(uploadName = local.name, uploadSent = 0,
                        uploadTotal = if (local.size > 0) local.size else 0) }
                    val source = withContext(Dispatchers.IO) { prepareUpload(resolver, uri, local) }
                    mutable.update { it.copy(uploadName = source.name, uploadSent = 0,
                        uploadTotal = if (source.size > 0) source.size else 0) }
                    val uploaded = withContext(Dispatchers.IO) { storeLocalUpload(source.name, source.mime, source.size, source.opener) }
                        ?: if (source.size > CHUNK_UPLOAD_THRESHOLD_BYTES) uploadInChunks(source) else uploadWhole(source)
                    mutable.update { it.copy(attachments = it.attachments + Attachment(source.name, uploaded, source.mime),
                        uploadCompleted = it.uploadCompleted + 1) }
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) { report(e) }
            finally { mutable.update { it.copy(uploading = false, uploadSent = 0, uploadTotal = 0, uploadName = "",
                uploadCompleted = 0, uploadCount = 0) } }
        }
    }

    /** 画像分割 (ANDROID_ONLY.md): images waiting in the dialog, from the upload sheet or the share target. */
    private val imageSplitMutable = MutableStateFlow<ImageSplitRequest?>(null)
    internal val imageSplit = imageSplitMutable.asStateFlow()

    fun openImageSplit(uris: List<Uri>) {
        if (uris.isEmpty()) return
        if (uris.size > IMAGE_SPLIT_MAX_IMAGES) notify("画像の分割は一度に${IMAGE_SPLIT_MAX_IMAGES}枚までです。先頭${IMAGE_SPLIT_MAX_IMAGES}枚のみ使います。")
        imageSplitMutable.value = ImageSplitRequest(uris.take(IMAGE_SPLIT_MAX_IMAGES))
    }

    fun closeImageSplit() {
        if (imageSplitMutable.value?.busy != true) imageSplitMutable.value = null
    }

    /**
     * Splits every image of the dialog, then attaches the files ([attach]) or only saves them to the
     * device (Pictures/AI Playground, or [tree] before Android 10). Saving works without an account.
     */
    internal fun runImageSplit(options: ImageSplitOptions, attach: Boolean, tree: Uri? = null) {
        val request = imageSplitMutable.value ?: return
        if (request.busy) return
        if (attach && state.value.uploading) { notify("アップロードが終わってから、もう一度お試しください。"); return }
        imageSplitMutable.value = request.copy(busy = true, error = null)
        viewModelScope.launch {
            val app = getApplication<Application>()
            val root = File(app.cacheDir, "shared/image_split")
            val dir = File(root, UUID.randomUUID().toString())
            // Earlier batches are no longer read once their upload has finished.
            val clearEarlier = !state.value.uploading
            try {
                val outputs = withContext(Dispatchers.IO) {
                    if (clearEarlier) root.deleteRecursively()
                    request.uris.flatMap { renderImageSplit(app, it, options, dir) }
                }
                if (attach) {
                    imageSplitMutable.value = null
                    upload(outputs.map { it.uploadUri })
                } else {
                    val saved = withContext(Dispatchers.IO) {
                        try { saveSplitOutputs(app, outputs, tree) } finally { dir.deleteRecursively() }
                    }
                    imageSplitMutable.value = null
                    notify(if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) "${saved}枚の画像を「Pictures/$IMAGE_SPLIT_SAVE_FOLDER」に保存しました。"
                        else "${saved}枚の画像を保存しました。")
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: OutOfMemoryError) {
                imageSplitMutable.value = request.copy(busy = false,
                    error = "画像が大きすぎるため分割できませんでした。分割数を減らすか、小さい画像でお試しください。")
            } catch (e: Exception) {
                imageSplitMutable.value = request.copy(busy = false,
                    error = "画像を分割できませんでした: ${e.message ?: e.javaClass.simpleName}")
            }
        }
    }

    /** Removing the uploading row (Web `uploadCancelTokens`): the rest of this batch is not uploaded. */
    fun cancelUpload() {
        if (!state.value.uploading) return
        uploadJob?.cancel()
    }

    /** Web `resetUploadState`: the composer ✕ and "リストをクリア" drop every attachment and the mask. */
    fun resetUploads() {
        uploadJob?.cancel()
        mutable.update { it.copy(attachments = emptyList(), imageMask = null) }
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
        val accountId = state.value.account?.id ?: throw IOException("アカウント情報がありません。")
        val directory = File(getApplication<Application>().cacheDir, "shared").apply { mkdirs() }
        val suffix = reference.substringBefore('?').substringAfterLast('.', "bin").take(8).filter { it.isLetterOrDigit() }.ifBlank { "bin" }
        val target = File(directory, "${UUID.randomUUID()}.$suffix")
        if (LocalChatStore.isLocalReference(reference)) {
            val store = localChats ?: throw IOException("この添付は別のプロファイルの端末内ファイルです。")
            val mime = withContext(Dispatchers.IO) { store.materializeFile(reference, target) } ?: throw IOException("添付を開けません。")
            return target to mime
        }
        val cachedMime = withContext(Dispatchers.IO) { offlineCache.materializeFile(accountId, reference, target) }
        if (cachedMime != null) return target to cachedMime
        val mime = withContext(Dispatchers.IO) {
            val result = api.download(reference, target, token())
            offlineCache.saveFileFromFile(accountId, reference, target, result)
            refreshOfflineCacheStats(accountId)
            result
        }
        return target to mime
    }

    private fun mimeForReference(reference: String): String {
        val extension = reference.substringBefore('?').substringAfterLast('.', "").lowercase()
        return MimeTypeMap.getSingleton().getMimeTypeFromExtension(extension) ?: "application/octet-stream"
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
        return "/api/files?limit=40&offset=$offset&sort=${current.librarySort}&q=$query$favorites"
    }

    private suspend fun fetchLibrary(append: Boolean) {
        if (state.value.offline) {
            val accountId = state.value.account?.id ?: return
            val files = withContext(Dispatchers.IO) { offlineCache.loadLibrary(accountId) }
                .filter { file ->
                    (state.value.libraryQuery.isBlank() || file.displayName.contains(state.value.libraryQuery, true)) &&
                        (!state.value.libraryFavoritesOnly || file.isFavorite)
                }
                .let { sortLibraryFiles(it, state.value.librarySort) }
            mutable.update { it.copy(library = files, libraryHasMore = false, libraryTotal = files.size, libraryBusy = false) }
            return
        }
        val offset = if (append) state.value.library.size else 0
        val reply = backend.get(libraryPath(offset), token())
        val files = parseLibraryFiles(reply)
        state.value.account?.let { account ->
            withContext(Dispatchers.IO) { offlineCache.saveLibrary(account.id, files) }
            refreshOfflineCacheStats(account.id)
        }
        mutable.update {
            it.copy(
                library = if (append) (it.library + files).distinctBy { file -> file.filepath } else files,
                libraryHasMore = reply.optBoolean("has_more"),
                libraryTotal = reply.optInt("total"),
                libraryBusy = false,
                libraryFailed = false,
            )
        }
    }

    fun refreshLibrary() {
        libraryJob?.cancel()
        libraryJob = viewModelScope.launch {
            mutable.update { it.copy(libraryBusy = true) }
            try { fetchLibrary(false) }
            catch (e: CancellationException) { throw e }
            catch (e: Exception) { mutable.update { it.copy(libraryFailed = true) } }
            finally { mutable.update { it.copy(libraryBusy = false) } }
        }
    }

    fun librarySearch(query: String) {
        mutable.update { it.copy(libraryQuery = query) }
        libraryJob?.cancel()
        libraryJob = viewModelScope.launch {
            delay(300)
            mutable.update { it.copy(libraryBusy = true) }
            try { fetchLibrary(false) }
            catch (e: CancellationException) { throw e }
            catch (e: Exception) { mutable.update { it.copy(libraryFailed = true) } }
            finally { mutable.update { it.copy(libraryBusy = false) } }
        }
    }

    fun setLibraryFavoritesOnly(value: Boolean) {
        if (state.value.libraryFavoritesOnly == value) return
        prefs.edit().putBoolean(LIB_FAVORITES_ONLY_KEY, value).apply()
        mutable.update { it.copy(libraryFavoritesOnly = value) }
        refreshLibrary()
    }

    fun moreLibrary() {
        if (state.value.offline || !state.value.libraryHasMore || state.value.libraryBusy) return
        libraryJob = viewModelScope.launch {
            mutable.update { it.copy(libraryBusy = true) }
            try { fetchLibrary(true) }
            catch (e: CancellationException) { throw e }
            catch (e: Exception) { notify("追加読み込みに失敗しました。もう一度お試しください。") }
            finally { mutable.update { it.copy(libraryBusy = false) } }
        }
    }

    fun toggleLibraryFavorite(file: LibraryFile) { viewModelScope.launch {
        if (state.value.offline) { notify("オフライン中はファイルのお気に入りを変更できません。"); return@launch }
        try {
            val reply = backend.post("/api/files/favorite", JSONObject().put("filepath", file.filepath), token())
            val favorite = reply.optBoolean("is_favorite")
            mutable.update { current -> current.copy(
                library = current.library.map { if (it.filepath == file.filepath) it.copy(isFavorite = favorite) else it },
                notice = if (favorite) "お気に入りに追加しました" else "お気に入りから外しました",
            ) }
            state.value.account?.let { account -> withContext(Dispatchers.IO) { offlineCache.saveLibrary(account.id, state.value.library) } }
        } catch (e: Exception) { notify("お気に入りの更新に失敗しました") }
    } }

    fun renameLibraryFile(file: LibraryFile, name: String) { viewModelScope.launch {
        if (state.value.offline) { notify("オフライン中はファイル名を変更できません。"); return@launch }
        try {
            if (name.isBlank()) { notify("ファイル名を入力してください"); return@launch }
            val reply = backend.post("/api/files/rename", JSONObject().put("filepath", file.filepath).put("filename", name.trim()), token())
            val display = reply.optString("filename", name.trim())
            mutable.update { current -> current.copy(
                library = current.library.map { if (it.filepath == file.filepath) it.copy(displayName = display) else it },
                attachments = current.attachments.map { if (it.reference == file.filepath) it.copy(name = display) else it },
                notice = "ファイル名を変更しました",
            ) }
            state.value.account?.let { account -> withContext(Dispatchers.IO) { offlineCache.saveLibrary(account.id, state.value.library) } }
        } catch (e: ApiException) { notify(e.payload.optString("error").ifBlank { "名前変更に失敗しました" }) }
        catch (e: Exception) { notify("名前変更に失敗しました") }
    } }

    fun deleteLibraryFile(file: LibraryFile) { viewModelScope.launch {
        if (state.value.offline) { notify("オフライン中はファイルを削除できません。"); return@launch }
        try {
            backend.post("/api/files/delete", JSONObject().put("filenames", JSONArray().put(file.filepath)), token())
            state.value.account?.let { account -> withContext(Dispatchers.IO) { offlineCache.deleteLibraryFile(account.id, file.filepath) } }
            mutable.update { current -> current.copy(library = current.library.filterNot { it.filepath == file.filepath }) }
        } catch (e: Exception) { notify("削除に失敗しました") }
    } }

    /** Web `deleteSelectedFiles`: one batch request, then the list is reloaded. */
    fun deleteLibraryFiles(filepaths: List<String>) { viewModelScope.launch {
        if (state.value.offline) { notify("オフライン中はファイルを削除できません。"); return@launch }
        try {
            backend.post("/api/files/delete", JSONObject().put("filenames", JSONArray(filepaths)), token())
            state.value.account?.let { account ->
                withContext(Dispatchers.IO) { filepaths.forEach { offlineCache.deleteLibraryFile(account.id, it) } }
            }
            mutable.update { it.copy(libraryBusy = true) }
            try { fetchLibrary(false) } finally { mutable.update { it.copy(libraryBusy = false) } }
        } catch (e: Exception) { notify("削除エラー") }
    } }

    fun setLibrarySort(order: String) {
        if (order !in LIBRARY_SORTS || state.value.librarySort == order) return
        prefs.edit().putString(LIB_SORT_KEY, order).apply()
        mutable.update { it.copy(librarySort = order) }
        refreshLibrary()
    }

    /**
     * Web `attachSelectedLibraryFiles`: adds the files without uploading again, skipping audio or video
     * the current model cannot take (`getModelMediaSupport`).
     */
    fun attachLibraryFiles(files: List<LibraryFile>) {
        val support = modelMediaSupport(state.value.model)
        var skippedAudio = 0
        var skippedVideo = 0
        val added = mutableListOf<Attachment>()
        files.forEach { file ->
            val audio = isAudioPath(file.filepath)
            val video = isVideoPath(file.filepath)
            if ((audio && !support.first) || (video && !support.second)) {
                if (audio) skippedAudio += 1
                if (video) skippedVideo += 1
                return@forEach
            }
            if (state.value.attachments.none { it.reference == file.filepath } && added.none { it.reference == file.filepath }) {
                added += Attachment(file.displayName, file.filepath, "", source = "library")
            }
        }
        val parts = listOfNotNull(skippedAudio.takeIf { it > 0 }?.let { "${it}件の音声" }, skippedVideo.takeIf { it > 0 }?.let { "${it}件の動画" })
        mutable.update { it.copy(
            attachments = it.attachments + added,
            notice = if (parts.isNotEmpty()) "このモデルは${parts.joinToString("・")}入力に非対応のため除外しました" else "ライブラリから添付しました",
        ) }
    }

    /** Web `showSelectedFileUsage`: chats that reference the file (up to 100) and whether more exist. */
    suspend fun libraryFileUsage(file: LibraryFile): Pair<List<FileUsageChat>, Boolean> {
        val reply = backend.get("/api/files/usage?filepath=" + URLEncoder.encode(file.filepath, "UTF-8"), token())
        val rows = reply.optJSONArray("chats") ?: JSONArray()
        val chats = (0 until rows.length()).map { index ->
            val row = rows.getJSONObject(index)
            FileUsageChat(row.optString("id"), row.nullableString("title").ifBlank { "新しいチャット" }, row.nullableString("updated_at"))
        }
        return chats to reply.optBoolean("has_more")
    }

    /** Web `downloadSelectedLibraryFiles`: fetches each file for the chosen save location. */
    fun saveLibraryFiles(files: List<LibraryFile>, target: suspend (LibraryFile, File, String) -> Unit) { viewModelScope.launch {
        var saved = 0
        files.forEach { file ->
            runCatching {
                val (local, mime) = downloadAttachment(file.url.ifBlank { file.filepath })
                target(file, local, mime)
                local.delete()
            }.onSuccess { saved += 1 }
        }
        if (saved == files.size) notify("${files.size}件のファイルをダウンロードしました")
        else notify("ダウンロードに失敗しました（${files.size - saved}件）")
    } }

    /** Web `reuseCurrentImage`: the viewer's image becomes an attachment of the next message. */
    fun reuseImage(reference: String, name: String = ""): Boolean {
        if (state.value.attachments.any { it.reference == reference }) { notify("この画像は既に添付されています"); return false }
        mutable.update { it.copy(attachments = it.attachments + Attachment(name.ifBlank { reference.substringAfterLast('/') }, reference, "", source = "library")) }
        notify("画像を添付ファイルに追加しました")
        return true
    }

    /** Reuses a library file as a composer attachment without re-uploading it. */
    fun reuseLibraryFile(file: LibraryFile) {
        if (state.value.attachments.any { it.reference == file.filepath }) {
            mutable.update { it.copy(notice = "この添付はすでに追加されています。") }
            return
        }
        mutable.update { it.copy(attachments = it.attachments + Attachment(file.displayName, file.filepath, "", source = "library")) }
    }

    // --- Gems ---

    private suspend fun fetchGems() {
        val gems = parseGems(backend.getArray("/api/gems", token()))
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

    /** Web `save-gem-btn`: an empty default model ("Use current model") is sent as null. */
    fun saveGem(uuid: String?, name: String, description: String, instruction: String, defaultModel: String, fixedPrompts: List<FixedPrompt>, onDone: (Boolean) -> Unit) {
        viewModelScope.launch {
            mutable.update { it.copy(gemsBusy = true) }
            try {
                val payload = JSONObject().put("name", name).put("description", description)
                    .put("instruction", instruction).put("default_model", defaultModel.ifBlank { null } ?: JSONObject.NULL)
                    .put("fixed_prompts", if (fixedPrompts.isEmpty()) JSONObject.NULL else JSONArray().apply {
                        fixedPrompts.forEach { put(JSONObject().put("name", it.name).put("content", it.content)) }
                    })
                if (uuid.isNullOrBlank()) backend.post("/api/gems", payload, token())
                else backend.put("/api/gems/$uuid", payload, token())
                fetchGems()
                mutable.update { current -> current.copy(selectedGem = current.selectedGem?.let { selected -> current.gems.firstOrNull { it.uuid == selected.uuid } }) }
                onDone(true)
            } catch (e: Exception) { report(e); onDone(false) }
            finally { mutable.update { it.copy(gemsBusy = false) } }
        }
    }

    fun deleteGem(gem: Gem) { viewModelScope.launch {
        try {
            backend.delete("/api/gems/${gem.uuid}", token())
            mutable.update { current -> current.copy(
                gems = current.gems.filterNot { it.uuid == gem.uuid },
                selectedGem = current.selectedGem?.takeIf { it.uuid != gem.uuid },
            ) }
        } catch (e: Exception) { report(e) }
    } }

    /**
     * Web `activateGem` / `clearActiveGem`: in an open chat the Gem is announced and saved as the chat's
     * `last_gem_uuid` right away; in a new chat it waits for the first message.
     */
    fun chooseGem(gem: Gem?) {
        mutable.update { it.copy(selectedGem = gem) }
        gem?.defaultModel?.takeIf { it.isNotBlank() }?.let { chooseModel(it) }
        val thread = state.value.selected ?: return
        if (gem != null) notify("Gem \"${gem.name}\" をこのチャットに適用しました")
        if (state.value.offline) return
        viewModelScope.launch {
            runCatching {
                backend.put("/api/mobile/v1/preferences",
                    JSONObject().put("last_gem_uuid", gem?.uuid ?: JSONObject.NULL).put("thread_id", thread.id), token())
            }
        }
    }

    /** Web `selectGemSuggestion`: applies the Gem picked from the `@` candidates and trims the mention. */
    fun applyGemMention(gem: Gem) {
        mutable.update { it.copy(draft = replaceGemMention(it.draft), composerFocusRequest = it.composerFocusRequest + 1) }
        chooseGem(gem)
    }

    // --- General preferences and this device's session ---

    private suspend fun fetchPreferences(applyDefaults: Boolean = false) {
        val payload = backend.get("/api/mobile/v1/preferences", token())
        val preferences = parsePreferences(payload)
        state.value.account?.let { account ->
            withContext(Dispatchers.IO) { offlineCache.savePreferences(account.id, payload) }
        }
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
            chipValues = it.chipValues + mapOf(
                "thinking_level" to thinkingLevel,
                "thinking_budget" to thinkingBudget.toString(),
                "reasoning_effort" to effort,
                "safety_setting" to safety,
            ),
        ).withModelRules() }
    }

    fun loadPreferences() {
        viewModelScope.launch {
            mutable.update { it.copy(prefsBusy = true) }
            try {
                val accountId = state.value.account?.id
                if (state.value.offline && accountId != null) {
                    withContext(Dispatchers.IO) { offlineCache.loadPreferences(accountId) }
                        ?.let { payload -> mutable.update { it.copy(preferences = parsePreferences(payload)) } }
                } else fetchPreferences()
            } catch (e: Exception) { report(e) } finally { mutable.update { it.copy(prefsBusy = false) } }
        }
    }

    /** Web `saveRichPastePromptPreferences`: saved quietly while the rich paste prompt is edited. */
    fun saveRichPastePrompt(prompt: String, useCustomDefault: Boolean) {
        if (state.value.offline) return
        viewModelScope.launch {
            runCatching {
                val reply = backend.put("/api/mobile/v1/preferences", JSONObject().put("rich_paste_prompt_default", prompt)
                    .put("rich_paste_prompt_use_custom_default", useCustomDefault), token())
                mutable.update { it.copy(preferences = parsePreferences(reply)) }
            }
        }
    }

    fun savePreferences(payload: JSONObject, message: String = "設定を保存しました") {
        viewModelScope.launch {
            if (state.value.offline) { notify("オフライン中はアカウント設定を保存できません。接続後に再試行してください。"); return@launch }
            mutable.update { it.copy(prefsBusy = true) }
            try {
                val reply = backend.put("/api/mobile/v1/preferences", payload, token())
                mutable.update { it.withSavedPreferences(parsePreferences(reply)).copy(notice = message) }
            } catch (e: Exception) { report(e) }
            finally { mutable.update { it.copy(prefsBusy = false) } }
        }
    }

    fun loadStorageUsage() {
        viewModelScope.launch {
            if (state.value.offline) return@launch
            runCatching { mutable.update { it.copy(storage = parseStorageUsage(backend.get("/api/storage", token()))) } }
                .onFailure { report(it) }
        }
    }

    fun loadFeedback() {
        viewModelScope.launch {
            if (state.value.offline) return@launch
            mutable.update { it.copy(feedbackBusy = true) }
            try { mutable.update { it.copy(feedbackItems = parseFeedbackItems(backend.get("/api/feedback", token()))) } }
            catch (e: Exception) { report(e) }
            finally { mutable.update { it.copy(feedbackBusy = false) } }
        }
    }

    /** [chatId] is the open chat whose copy goes with the feedback (null when the box is not ticked). */
    fun submitFeedback(title: String, message: String, chatId: String? = null) {
        if (message.isBlank()) return
        viewModelScope.launch {
            if (state.value.offline) { notify("オフライン中はフィードバックを送信できません。"); return@launch }
            mutable.update { it.copy(feedbackBusy = true) }
            try {
                val payload = JSONObject().put("title", title.trim()).put("message", message.trim())
                // ログの収集を強化: the last hour of the activity log goes with the feedback.
                val logs = if (ActivityLog.enabled) withContext(Dispatchers.IO) { ActivityLog.recent() } else null
                if (logs != null) payload.put("client_logs", JSONObject().put("client", "android")
                    .put("version", BuildConfig.VERSION_NAME).put("window_seconds", ActivityLog.WINDOW_MS / 1000).put("entries", logs))
                val chatCopy = chatId?.let { feedbackChatCopy(it) }
                if (chatCopy != null) payload.put("chat_copy", chatCopy.payload)
                // The server gathers the chat's records, logs and attachments before it answers.
                val reply = if (chatCopy != null) api.postLong("/api/feedback", payload, token()) else backend.post("/api/feedback", payload, token())
                var chatSaved = reply.optBoolean("chat_copy_saved", false)
                val feedbackId = reply.optInt("feedback_id", 0)
                if (chatCopy != null && chatSaved && chatCopy.deviceFiles.isNotEmpty()) {
                    chatSaved = feedbackId > 0 && uploadFeedbackChatFiles(feedbackId, chatCopy.deviceFiles)
                }
                loadFeedback()
                notify(feedbackSentText(logs?.length(), reply.optBoolean("logs_saved", true), chatCopy != null, chatSaved))
            } catch (e: Exception) { report(e) }
            finally { mutable.update { it.copy(feedbackBusy = false) } }
        }
    }

    private class FeedbackChatCopy(val payload: JSONObject, val deviceFiles: List<String>)

    /**
     * Web `chat_copy`: the server copies everything it has on the chat, decrypted. The device adds what only
     * it holds: the chat as it shows it while chats are on the device (a device chat, or unsent messages over
     * a server chat), its offline cache of the chat, preferences keyed by the chat, diagnostics entries not yet
     * sent, and the attachments kept only here ([uploadFeedbackChatFiles] after the feedback is accepted).
     */
    private suspend fun feedbackChatCopy(id: String): FeedbackChatCopy {
        val resolved = serverIdOf(id)
        val ids = setOf(id, resolved)
        val copy = JSONObject().put("client", "android").put("version", BuildConfig.VERSION_NAME).put("thread_id", resolved)
        val view = if (localChats != null) runCatching { backend.get("/api/threads/$resolved", token()) }.getOrNull() else null
        if (view != null) copy.put("thread", view)
        val accountId = state.value.account?.id
        withContext(Dispatchers.IO) {
            accountId?.let { account -> ids.firstNotNullOfOrNull { offlineCache.loadThread(account, it) } }
                ?.let { copy.put("offline_cache", it) }
            val app = getApplication<Application>()
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

    /** Sends the attachments kept only on this device to the chat copy of [feedbackId]; false when one was not saved. */
    private suspend fun uploadFeedbackChatFiles(feedbackId: Int, references: List<String>): Boolean {
        val store = localChats ?: return false
        var saved = true
        for (reference in references) {
            val target = File(getApplication<Application>().cacheDir, "feedback-" + java.util.UUID.randomUUID())
            try {
                val (mime, name) = withContext(Dispatchers.IO) {
                    store.materializeFile(reference, target) to store.fileInfo(reference)?.optString("name").orEmpty()
                }
                val ok = mime != null && api.uploadFeedbackChatFile(feedbackId, reference,
                    name.ifBlank { reference.substringAfterLast('/') }, target, mime, token()).optBoolean("saved")
                if (!ok) saved = false
            } catch (e: Exception) {
                if (e is kotlinx.coroutines.CancellationException) throw e
                Diagnostics.failure("feedback.chat_file", e)
                saved = false
            } finally { target.delete() }
        }
        return saved
    }

    fun loadMcpServers() {
        viewModelScope.launch {
            if (state.value.offline) return@launch
            mutable.update { it.copy(mcpBusy = true) }
            try { mutable.update { it.copy(mcpServers = parseMcpServers(backend.get("/api/mcp/servers", token()))) } }
            catch (e: Exception) { report(e) }
            finally { mutable.update { it.copy(mcpBusy = false) } }
        }
    }

    fun setMcpServerEnabled(server: McpServerInfo, enabled: Boolean) {
        viewModelScope.launch {
            mutable.update { it.copy(mcpBusy = true) }
            try {
                backend.put("/api/mcp/servers/${server.id}", JSONObject().put("enabled", enabled), token())
                loadMcpServers()
            } catch (e: Exception) { report(e) }
            finally { mutable.update { it.copy(mcpBusy = false) } }
        }
    }

    /** The minimal-mode popup item a slash command drives (Web `MINIMAL_POPUP_ITEMS`). */
    private class SlashItem(val label: String, val rule: OptionRule?, val checked: Boolean, val toggle: () -> Unit)

    private fun slashItem(key: String, current: ChatState, rules: ComposerRules): SlashItem? = when (key) {
        "canvas" -> SlashItem("Canvas", rules.canvas, current.canvasMode, ::toggleCanvas)
        "coding" -> SlashItem("Coding", rules.coding, current.codingMode, ::toggleCoding)
        "search" -> SlashItem("Search", rules.search, current.enableSearch, ::toggleSearch)
        "urls" -> SlashItem("URLs", rules.urls, current.enableUrlContext, ::toggleUrlContext)
        "maps" -> SlashItem("Maps", rules.maps, current.enableMaps, ::toggleMaps)
        "python" -> SlashItem("Python", rules.python, current.enablePython, ::togglePython)
        "file" -> SlashItem("File", rules.file, current.enableFileCreation, ::toggleFileCreation)
        "mcp" -> SlashItem("MCP", rules.mcp, current.enableMcp, ::toggleMcp)
        "sysprompt" -> SlashItem("SysPrompt", rules.sysPrompt, current.enableSystemPrompt, ::toggleSystemPrompt)
        "thinking" -> SlashItem("Thinking", rules.thinking, current.enableThinking, ::toggleThinking)
        "promptcache" -> SlashItem("PromptCache", rules.promptCache, current.enablePromptCache, ::togglePromptCache)
        "compress" -> SlashItem("Compress", null, current.compression.enabled) {
            saveCompressionSettings(state.value.compression.copy(enabled = !state.value.compression.enabled))
        }
        "tempchat" -> SlashItem("一時チャット", null, current.selected?.isTemporary ?: current.newThreadTemporary, ::toggleTemporaryChat)
        else -> null
    }

    /**
     * Web `executeMinimalSlashCommand`: runs a minimal-mode command with the Web notices. Popup-only actions
     * (＋ menu, attach, voice, rich paste) go to [onLocal]. Returns false when the argument was not usable.
     */
    fun runSlashCommand(command: SlashCommand, argument: String, onLocal: (String) -> Unit): Boolean {
        when (command.id) {
            "options", "attach", "voice", "paste" -> { onLocal(command.id); return true }
        }
        val current = state.value
        val rules = composerRules(current.model, current.mcpServers.any { it.enabled })
        val raw = command.presetArgument.ifBlank { argument }.trim()
        if (command.id == "effort" || command.id == "safety") {
            if (command.id == "effort" && !rules.effort.visible) { notify("/${command.id} は現在のモデルでは利用できません"); return true }
            if (command.id == "effort" && (rules.effort.disabled || rules.effort.dimmed)) { notify("/${command.id} は現在変更できません"); return true }
            if (raw.isEmpty()) {
                notify("使い方: ${command.label} ${if (command.id == "effort") "none / low / medium / high / xhigh / max" else "default / none"}")
                return false
            }
            val options = if (command.id == "effort") rules.effortOptions.map { it to EFFORT_OPTION_LABELS.getValue(it) }
                else listOf("default" to "Default", "none" to "None")
            val normalized = raw.lowercase()
            val option = options.firstOrNull { (value, label) -> value == normalized || label.lowercase() == normalized }
            if (option == null) { notify("${command.label}: 指定値「$raw」は利用できません"); return false }
            generationOption(if (command.id == "effort") "reasoning_effort" else "safety_setting", option.first)
            notify("${if (command.id == "effort") "Effort" else "Safety"}: ${option.second}")
            return true
        }
        val item = slashItem(command.itemKey, current, rules) ?: return false
        if (item.rule?.visible == false) { notify("/${command.id} は現在のモデルでは利用できません"); return true }
        val disabled = item.rule != null && (item.rule.disabled || item.rule.dimmed)
        if (disabled && command.itemKey != "thinking") { notify("/${command.id} は現在変更できません"); return true }
        if (command.itemKey == "thinking" && raw.isNotEmpty()) {
            val normalized = raw.lowercase()
            SLASH_THINKING_LEVELS[normalized]?.let { level ->
                if (!current.enableThinking && item.rule?.disabled != true) toggleThinking()
                generationOption("thinking_level", level)
                notify("Thinking: $normalized")
                return true
            }
            if (parseSlashToggle(raw) == SlashToggle.INVALID) {
                notify("使い方: /thinking on / off / min / low / mid / high")
                return false
            }
        }
        if (raw.isNotEmpty()) {
            when (val desired = parseSlashToggle(raw)) {
                SlashToggle.INVALID -> { notify("使い方: ${command.label} on / off"); return false }
                SlashToggle.ON, SlashToggle.OFF -> if (item.checked == (desired == SlashToggle.ON)) {
                    notify("${item.label}: ${if (item.checked) "ON" else "OFF"}")
                    return true
                }
                SlashToggle.TOGGLE -> Unit
            }
        }
        if (disabled || item.rule?.disabled == true) return true
        item.toggle()
        return true
    }

    /** Fetches the server thread payload and writes a native A4 PDF into the share cache. */
    private var pdfExporting = false

    /** Web `openThreadPdfPrintDialog` guards and failure toast, then a native PDF for the share sheet. */
    fun exportPdf(onReady: (File) -> Unit) {
        val id = state.value.selected?.id
        if (id == null) { notify("PDF化するスレッドを開いてください"); return }
        if (pdfExporting) { notify("PDF出力の準備中です。しばらくお待ちください。"); return }
        if (state.value.offline && !deviceChat(id)) { notify("PDF出力に失敗しました"); return }
        pdfExporting = true
        // Web `openThreadPdfPrintDialog`: the branch on screen (`leaf_id`) with the 準備中 progress toast.
        notify("PDF出力の準備中です")
        val leaf = state.value.leafId
        viewModelScope.launch {
            mutable.update { it.copy(busy = true) }
            try {
                val payload = backend.get("/c/$id/pdf" + (leaf?.let { "?leaf_id=$it" } ?: ""), token())
                val messages = parsePdfMessages(payload)
                val title = payload.optJSONObject("thread")?.optString("title").orEmpty().ifBlank { "AI Chat" }
                val safeId = id.filter { it.isLetterOrDigit() || it == '-' || it == '_' }.take(24).ifBlank { "thread" }
                val directory = File(getApplication<Application>().cacheDir, "shared").apply { mkdirs() }
                val target = File(directory, "thread-$safeId.pdf")
                withContext(Dispatchers.IO) {
                    writeThreadPdf(title, payload.optString("generated_at"), messages, target)
                }
                onReady(target)
            } catch (e: CancellationException) { throw e }
            catch (ignored: Exception) { notify("PDF出力に失敗しました") }
            finally { pdfExporting = false; mutable.update { it.copy(busy = false) } }
        }
    }

    private companion object {
        const val CHUNK_UPLOAD_THRESHOLD_BYTES = 8L * 1024 * 1024
        /** Web `CONNECTION_RETRY_DELAY_MS`. */
        const val CONNECTION_RETRY_DELAY_MS = 2000L
        /** Away from the app at least this long, a server answer is rejoined rather than read on. */
        const val STREAM_REJOIN_AFTER_MS = 30_000L
        /** Longest wait for the outbox to go up before a device answer is finished (the sync then continues in the background). */
        const val AFTER_ANSWER_SYNC_WAIT_MS = 3000L
        const val DIAGNOSTICS_INTERVAL_MS = 10_000L
        /** No diagnostics entry for this long while work runs: the thread stacks are recorded. */
        const val DIAGNOSTICS_STALL_MS = 20_000L
        const val DIAGNOSTICS_STACKS_GAP_MS = 60_000L
        const val LIB_SORT_KEY = "lib_sort_order"
        const val LIB_FAVORITES_ONLY_KEY = "lib_favorites_only"
        const val MAX_SINGLE_UPLOAD_BYTES = 64L * 1024 * 1024
        const val PREF_BROWSER_LOGIN_VERIFIER = "browser_login_pkce_verifier"
        const val PREF_SERVER_ORIGIN = "server_origin"
        const val PREF_PROFILE_MODE = "profile_mode"
        const val PROFILE_LOCAL = "local"
        const val PROFILE_SERVER = "server"
        const val LOCAL_PROFILE_NAME = "この端末"
        const val PREF_SAVED_SERVERS = "saved_servers"
        const val PREF_BROWSER_LOGIN_STARTED_AT = "browser_login_started_at"
        const val BROWSER_LOGIN_TTL_MS = 10L * 60 * 1000
    }
}

/** A running answer that the device itself generates: no server job id, and chats are answered on the device. */
internal fun isDeviceAnswer(streaming: Boolean, uploadsLocal: Boolean, jobId: String?): Boolean =
    streaming && uploadsLocal && jobId == null
