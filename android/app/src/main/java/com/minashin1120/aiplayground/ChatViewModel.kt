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
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
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
    /** Web composer selects that do not depend on the model (Thinking level/Budget, Effort, Safety). */
    val chipValues: Map<String, String> = COMPOSER_SELECT_DEFAULTS,
    val batchMode: Boolean = false, val enablePython: Boolean = false, val enableMcp: Boolean = true,
    val canvasMode: Boolean = false, val codingMode: Boolean = false,
    val codingTarget: CodingTarget? = null,
    val imageMask: String? = null,
    val uploading: Boolean = false, val streaming: Boolean = false,
    val uploadSent: Long = 0L, val uploadTotal: Long = 0L, val uploadName: String = "",
    /** Files finished / queued in the running upload batch (Web `Preparing... (completed/total)`). */
    val uploadCompleted: Int = 0, val uploadCount: Int = 0,
    val library: List<LibraryFile> = emptyList(), val libraryBusy: Boolean = false,
    val libraryQuery: String = "", val libraryFavoritesOnly: Boolean = false,
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
    /** Web low-bandwidth mode: preference (auto/on/off), effective state and the detection reason. */
    val lowBandwidthPreference: String = "auto",
    val lowBandwidthMode: Boolean = false,
    val lowBandwidthReason: String = "",
    /** Web `currentQuote`: text quoted from a message, sent as `quote_text` with the next message. */
    val quote: String = "",
    /** Incremented to ask the screen to open the settings modal (e.g. from the encryption status dialog). */
    val settingsRequest: Long = 0L,
)

class ChatViewModel(application: Application) : AndroidViewModel(application) {
    private val api = PlaygroundApi()
    private val store = TokenStore(application)
    private val playIntegrity = PlayIntegrityClient(application)
    private var integrityTurnstileTicket: String? = null
    private val offlineCache = OfflineCacheStore(application)
    private val prefs = application.getSharedPreferences("navigation", 0)
    private val connectivity = application.getSystemService(ConnectivityManager::class.java)
    private val connectionProbeMutex = Mutex()
    private val mutable = MutableStateFlow(ChatState())
    val state = mutable.asStateFlow()
    private var session: StoredSession? = null
    private var foreground = false
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
    private data class Submission(val body: JSONObject, val files: List<Attachment>)
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
        runCatching { api.get("/api/mobile/v1/config") }.getOrNull()?.let { config ->
            mutable.update { it.copy(
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
    private fun token(): String = session?.token ?: throw IOException("端末連携が必要です。")
    private fun deviceName(): String = "${Build.MANUFACTURER} ${Build.MODEL}".trim().take(80)

    private suspend fun authPost(path: String, body: JSONObject): JSONObject {
        val request = JSONObject(body.toString())
        request.put("integrity_enabled", true)
        val ticket = integrityTurnstileTicket
        if (ticket != null) {
            request.put("integrity_turnstile_ticket", ticket)
            integrityTurnstileTicket = null
        } else {
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
                    mutable.update { it.copy(authTurnstileUrl = url) }
                }
            }
            throw error
        }
    }

    fun integrityTurnstileComplete(ticket: String) {
        if (ticket.length !in 20..128) return
        integrityTurnstileTicket = ticket
        mutable.update { it.copy(authTurnstileUrl = null, authError = "安全性を確認しました。認証をもう一度実行してください。") }
    }

    private suspend fun acceptAuthResponse(reply: JSONObject) {
        val accessToken = reply.optString("access_token")
        require(accessToken.isNotBlank()) { "認証トークンを取得できませんでした。" }
        session = StoredSession(accessToken, System.currentTimeMillis() + reply.optLong("expires_in", 2_592_000L) * 1000L)
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
        return "/android/auth/$provider/start?code_challenge_method=S256&code_challenge=" +
            BrowserLoginPkce.challenge(verifier)
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
                val reply = api.post("/api/mobile/v1/security/passkeys/options", JSONObject(), token())
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
                        val reply = api.post("/api/mobile/v1/security/passkeys/verify", JSONObject()
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
            val reply = api.get("/api/mobile/v1/security", token())
            mutable.update { it.copy(security = parseSecurityInfo(reply), securityError = null) }
        } catch (e: CancellationException) { throw e }
        catch (e: Exception) { mutable.update { it.copy(securityError = e.message ?: "セキュリティ設定を取得できませんでした。") } }
    } }

    fun startTotpSetup() { viewModelScope.launch {
        mutable.update { it.copy(securityBusy = true, securityError = null) }
        try {
            val reply = api.post("/api/mobile/v1/security/totp/setup", JSONObject(), token())
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
        api.post("/api/mobile/v1/security/totp/enable", JSONObject().put("code", code), token())
    }

    fun disableTotp(code: String) = securityAction {
        api.post("/api/mobile/v1/security/totp/disable", JSONObject().put("code", code), token())
    }

    fun removePasskey(id: String) = securityAction {
        api.post("/api/mobile/v1/security/passkeys/remove", JSONObject().put("id", id), token())
    }

    fun saveSecurityPreferences(default2fa: String, passkeyOnly: Boolean, skipGoogle: Boolean) = securityAction {
        api.post("/api/mobile/v1/security/preferences", JSONObject()
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
        val reply = api.get("/api/mobile/v1/setup", token())
        val models = parseModels(reply)
        val defaultModel = reply.optString("default_model", "gemini-3.6-flash")
        mutable.update { it.copy(
            account = null,
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
                val reply = api.put("/api/mobile/v1/setup", JSONObject()
                    .put("default_model", defaultModel)
                    .put("openai_api_key", openaiKey)
                    .put("gemini_api_key", geminiKey)
                    .put("anthropic_api_key", anthropicKey)
                    .put("deepseek_api_key", deepseekKey)
                    .put("kimi_api_key", kimiKey)
                    .put("mistral_api_key", mistralKey)
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
        val me = api.get("/api/mobile/v1/me", token())
        val serverModels = parseModels(me)
        val displayModels = displayModels(serverModels)
        val account = Account(me.getInt("id"), me.getString("username"), displayModels,
            me.optString("default_model"), me.optBoolean("e2ee_enabled"))
        val chosen = prefs.getString("model_${account.id}", account.defaultModel).orEmpty()
            .takeIf { chosen -> account.models.any { it.id == chosen && it.selectable } }
            ?: account.models.firstOrNull { it.selectable }?.id.orEmpty()
        withContext(Dispatchers.IO) { offlineCache.saveAccount(account.id, me) }
        prefs.edit().putString("offline_cache_account_id", account.id.toString()).apply()
        markConnectionReachable()
        mutable.update { it.copy(
            account = account, model = chosen, pairing = false, userCode = "", offline = false,
            setupRequired = false, authBusy = false, authError = null,
        ) }
        fetchThreads(false)
        runCatching { fetchGems() }.onFailure { report(it) }
        runCatching { fetchPreferences(applyDefaults = true) }.onFailure { report(it) }
        runCatching { fetchBatchJobs(notify = false) }.onFailure { report(it) }
        // The composer shows the MCP chip only when a server is enabled (Web `applyMcpPromptChipUi`).
        loadMcpServers()
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
        val selected = prefs.getString("model_${account.id}", account.defaultModel).orEmpty()
            .takeIf { value -> account.models.any { it.id == value && it.selectable } }
            ?: account.models.firstOrNull { it.selectable }?.id.orEmpty()
        val cachedPrefs = withContext(Dispatchers.IO) { offlineCache.loadPreferences(account.id) }
        val cachedThreads = withContext(Dispatchers.IO) { offlineCache.loadThreads(account.id) }
        mutable.update { current -> current.copy(
            account = account,
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
        if (value) startConnectionMonitor() else stopConnectionMonitor()
        if (returning) recomputeLowBandwidth(notify = false)
        if (value && state.value.account != null) startBatchPolling()
        if (returning && state.value.account != null && state.value.selected != null && !state.value.busy) refresh()
        if (returning && state.value.account != null) startCacheSyncIfAllowed()
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
                api.get("/api/version?heartbeat=${System.currentTimeMillis()}")
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
            setConnectionUnavailable(if (hasUsableNetwork()) ConnectionStatus.UNSTABLE else ConnectionStatus.OFFLINE)
        } catch (e: CancellationException) {
            throw e
        } catch (e: ApiException) {
            val mode = connectionStatusForHttp(e.status)
                ?: if (e.status >= 500) ConnectionStatus.UNSTABLE else null
            if (mode != null) setConnectionUnavailable(mode)
        } catch (_: Exception) {
            setConnectionUnavailable(if (hasUsableNetwork()) ConnectionStatus.UNSTABLE else ConnectionStatus.OFFLINE)
        }
    }

    private fun setConnectionUnavailable(status: ConnectionStatus, message: String = status.defaultMessage()) {
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
        mutable.update { it.copy(quote = quoted) }
    }
    fun clearQuote() { mutable.update { it.copy(quote = "") } }
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
        return copy(
            enableSearch = rules.search.forced ?: enableSearch,
            enableUrlContext = rules.urls.forced ?: enableUrlContext,
            enableMaps = rules.maps.forced ?: enableMaps,
            enablePython = rules.python.forced ?: enablePython,
            enableSystemPrompt = rules.sysPrompt.forced ?: enableSystemPrompt,
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
    fun toggleBatchMode() { mutable.update { it.copy(batchMode = !it.batchMode) } }
    fun togglePython() { mutable.update { it.copy(enablePython = !it.enablePython) } }
    fun toggleMcp() { mutable.update { it.copy(enableMcp = !it.enableMcp) } }
    fun toggleCanvas() { mutable.update { it.copy(canvasMode = !it.canvasMode) } }
    fun toggleCoding() { mutable.update { it.copy(codingMode = !it.codingMode) } }
    fun toggleTemporaryChat() {
        val selected = state.value.selected
        if (selected != null) {
            saveThreadSettings(selected.title, state.value.customInstruction, state.value.includeGlobalInstruction, !selected.isTemporary)
        } else {
            mutable.update { it.copy(newThreadTemporary = !it.newThreadTemporary) }
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
    fun uploadImageMask(name: String, bytes: ByteArray) {
        if (bytes.isEmpty() || state.value.banned || state.value.streaming || state.value.uploading) return
        viewModelScope.launch {
            try {
                val response = api.upload(name, bytes.toRequestBody("image/png".toMediaType()), token())
                setImageMask(response.getString("filename"))
                notify("画像マスクを設定しました。次の画像生成で適用されます。")
            } catch (e: Exception) { report(e) }
        }
    }
    fun generationOption(key: String, value: String) {
        if (key in COMPOSER_SELECT_DEFAULTS) {
            mutable.update { it.copy(chipValues = it.chipValues + (key to value)) }
            return
        }
        if (state.value.streaming) return
        mutable.update { current -> current.copy(generationValues = current.generationValues +
            (current.model to (current.generationValues[current.model].orEmpty() + (key to value)))) }
    }
    fun clearAttachments() { mutable.update { it.copy(attachments = emptyList()) } }
    fun removeAttachment(reference: String) { mutable.update { it.copy(attachments = it.attachments.filterNot { a -> a.reference == reference }) } }
    fun search(query: String) {
        mutable.update { it.copy(search = query) }
        navigationJob?.cancel()
        navigationJob = viewModelScope.launch {
            delay(300)
            try {
                if (state.value.offline) applyCachedThreads(query) else fetchThreads(false)
            } catch (e: Exception) { report(e) }
        }
    }
    private suspend fun fetchThreads(more: Boolean) {
        if (state.value.offline) {
            applyCachedThreads(state.value.search)
            return
        }
        val current = state.value
        val page = if (more) current.nextPage ?: return else 1
        val reply = api.get("/api/threads?page=$page&q=${URLEncoder.encode(current.search, "UTF-8")}", token())
        if (current.search != state.value.search) return
        val rows = reply.getJSONArray("threads")
        val items = (0 until rows.length()).map { parseThreadItem(rows.getJSONObject(it)) }
        state.value.account?.let { account ->
            withContext(Dispatchers.IO) { offlineCache.saveThreads(account.id, items) }
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
        mutable.update { it.copy(threads = visible, nextPage = null, offline = true) }
    }
    fun moreThreads() {
        if (state.value.offline) return
        navigationJob?.cancel(); navigationJob = viewModelScope.launch { try { fetchThreads(true) } catch (e: Exception) { report(e) } }
    }
    fun newChat(temporary: Boolean = false) {
        navigationJob?.cancel(); streamJob?.cancel(); failed = null
        heartbeatJob?.cancel()
        pendingParentId = null
        val transition = nextChatTransition(ChatTransitionKind.NEW_CHAT)
        mutable.update { it.copy(selected = null, messages = emptyList(), allMessages = emptyList(), leafId = null,
            editingMessageId = null, jobId = null, streaming = false,
            liveContent = "", liveThought = "", status = "", busy = false, retryAvailable = false,
            cards = emptyList(), hasOlder = false, oldestId = null, customInstruction = "",
            includeGlobalInstruction = true, newThreadTemporary = temporary, tempChatRemainingSeconds = null,
            selectedGem = null, codingTarget = null, imageMask = null,
            chatTransitionId = transition.first, chatTransitionKind = transition.second,
            chatNavigationId = transition.first, chatNavigationKind = transition.second) }
    }
    fun openThread(thread: ThreadItem) {
        navigationJob?.cancel(); streamJob?.cancel(); heartbeatJob?.cancel(); failed = null
        pendingParentId = null
        val transition = nextChatTransition(ChatTransitionKind.OPEN_THREAD)
        val storedLeaf = prefs.getInt("leaf_${thread.id}", -1).takeIf { it > 0 }
        mutable.update { it.copy(selected = thread, messages = emptyList(), allMessages = emptyList(),
            leafId = storedLeaf, editingMessageId = null, streaming = false, busy = true,
            liveContent = "", liveThought = "", jobId = null, retryAvailable = false,
            cards = emptyList(), hasOlder = false, oldestId = null,
            // Keep the current transition id while the history loads so the AnimatedContent swap
            // happens exactly once, when the messages arrive, instead of animating to a placeholder first.
            chatTransitionKind = ChatTransitionKind.NONE,
            chatNavigationId = transition.first, chatNavigationKind = transition.second) }
        navigationJob = viewModelScope.launch {
            try {
                loadMessages(thread.id)
                if (state.value.selected?.id == thread.id) {
                    mutable.update { it.copy(busy = false, chatTransitionId = transition.first, chatTransitionKind = transition.second) }
                }
                if (foreground && state.value.jobId != null) resume()
            }
            catch (e: Exception) { report(e) }
            finally { mutable.update { it.copy(busy = false) } }
        }
    }
    private suspend fun loadMessages(id: String, older: Boolean = false) {
        if (state.value.offline) {
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
        val reply = api.get("/api/threads/$id?limit=$limit$before", token())
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
        // Keep the per-thread remembered branch in sync with what was actually resolved here,
        // so a stale branch (e.g. from before an edit/regenerate created a new one) does not
        // reassert itself the next time this thread is opened.
        state.value.selected?.let { thread -> leaf?.let { prefs.edit().putInt("leaf_${thread.id}", it).apply() } }
        state.value.account?.let { account ->
            withContext(Dispatchers.IO) {
                offlineCache.saveThread(
                    account.id, state.value.selected ?: return@withContext, all,
                    reply.optBoolean("has_older_messages"), reply.nullableString("oldest_loaded_id").ifBlank { null },
                    reply.nullableString("custom_instruction"), reply.optBoolean("include_global_instruction", true),
                    reply.optLong("temp_chat_remaining_seconds").takeIf { value -> !reply.isNull("temp_chat_remaining_seconds") && value >= 0 },
                    leaf,
                )
            }
            refreshOfflineCacheStats(account.id)
        }
        if (!older) syncHeartbeat()
    }

    private suspend fun loadCachedMessages(id: String, older: Boolean) {
        val accountId = state.value.account?.id ?: return
        val cached = withContext(Dispatchers.IO) { offlineCache.loadThread(accountId, id) } ?: return
        val parsed = parseMessages(cached)
        val all = if (older) (parsed + state.value.allMessages).distinctBy { it.id } else parsed
        val leaf = state.value.leafId?.takeIf { candidate -> all.any { numericId(it) == candidate } }
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
        if (state.value.offline) { notify("オフライン中は履歴を削除できません。"); return@launch }
        try {
            api.delete("/api/threads/${thread.id}", token())
            state.value.account?.let { account -> withContext(Dispatchers.IO) { offlineCache.deleteThread(account.id, thread.id) } }
            if (state.value.selected?.id == thread.id) newChat()
            fetchThreads(false)
        } catch (e: Exception) { report(e) }
    } }
    /** Web `renameThread`: `prompt("Title:")` then PUT the new title; empty input is ignored. */
    fun renameThread(thread: ThreadItem, title: String) { viewModelScope.launch {
        if (title.isEmpty()) return@launch
        if (state.value.offline) { notify("オフライン中はタイトルを変更できません。"); return@launch }
        try {
            val reply = api.put("/api/threads/${thread.id}/title", JSONObject().put("title", title), token())
            val saved = reply.optString("title", title).ifBlank { title }
            mutable.update { current -> current.copy(
                selected = current.selected?.let { if (it.id == thread.id) it.copy(title = saved) else it },
            ) }
            fetchThreads(false)
        } catch (e: Exception) { report(e) }
    } }

    fun requestSettings() { mutable.update { it.copy(settingsRequest = it.settingsRequest + 1) } }

    /** Web `deleteMessage`: removes the message and everything after it, then reloads the thread. */
    fun deleteMessage(message: ChatMessage) {
        val id = numericId(message) ?: return
        val thread = state.value.selected ?: return
        viewModelScope.launch {
            if (state.value.offline) { notify("オフライン中はメッセージを削除できません。"); return@launch }
            try {
                api.delete("/api/messages/$id", token())
                if (state.value.selected?.id == thread.id) {
                    mutable.update { it.copy(leafId = null) }
                    prefs.edit().remove("leaf_${thread.id}").apply()
                    loadMessages(thread.id)
                }
            } catch (e: Exception) { report(e) }
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

    /** Ends this device's sign-in after the account or every session was removed on the server. */
    fun signedOutRemotely() { viewModelScope.launch { clearSession() } }

    /** Web `openThreadModal`: a new chat is created first so its settings can be edited. */
    fun ensureThread(onReady: () -> Unit) {
        if (state.value.selected != null) { onReady(); return }
        if (state.value.offline) { notify("オフライン中はメッセージを送信できません。"); return }
        viewModelScope.launch {
            try {
                val created = api.post("/api/threads", JSONObject().put("is_temporary", state.value.newThreadTemporary), token())
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
        api.post("/api/token_estimate", JSONObject().put("model", model).put("message", message)
            .put("quote_text", quote).put("image_urls", JSONArray(imageUrls)), token())

    suspend fun legalMarkdown(kind: String): String {
        val safe = if (kind == "privacy") "privacy" else "terms"
        return api.getText("/static/legal/$safe.md?t=${System.currentTimeMillis()}")
    }

    fun toggleBookmark(thread: ThreadItem) { viewModelScope.launch {
        if (state.value.offline) { notify("オフライン中はブックマークを変更できません。"); return@launch }
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
            if (state.value.offline) { notify("オフライン中はチャット設定を変更できません。"); return@launch }
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
        if (current.banned) return
        if (current.offline) { mutable.update { it.copy(notice = "オフライン中はメッセージを送信できません。") }; return }
        if (current.streaming || current.busy || current.uploading || (current.draft.isBlank() && current.attachments.isEmpty())) return
        if (current.model.isBlank()) { mutable.update { it.copy(notice = "モデルを選択してください。") }; return }
        val info = current.account?.models?.firstOrNull { it.id == current.model && it.selectable } ?: return
        val generation = try { generationOptionsPayload(info, current.generationValues[current.model].orEmpty()) }
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
        val body = JSONObject().put("model", current.model).put("message", current.draft)
            .put("client_request_id", UUID.randomUUID().toString()).put("image_urls", JSONArray(current.attachments.map { it.reference }))
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
        mutable.update { it.copy(draft = "", attachments = emptyList(), editingMessageId = null, quote = "") }
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
                    mutable.update { it.copy(selected = ThreadItem(id, created.nullableString("title"), it.model,
                        isTemporary = created.optBoolean("is_temporary")), newThreadTemporary = false) }
                    syncHeartbeat()
                }
                if (submission.body.optBoolean("parent_id_explicit")) {
                    // Branching send (edit-and-resend, regenerate): drop the old branch's tail
                    // from the visible path immediately, instead of leaving the previous
                    // prompt/reply bubbles on screen until the new answer finishes streaming.
                    val parentId = submission.body.opt("parent_id") as? Int
                    mutable.update { current ->
                        val truncated = if (parentId == null) emptyList()
                        else current.messages.indexOfFirst { numericId(it) == parentId }
                            .let { idx -> if (idx >= 0) current.messages.subList(0, idx + 1) else current.messages }
                        current.copy(messages = truncated)
                    }
                }
                val userId = "local-${submission.body.getString("client_request_id")}"
                mutable.update { it.copy(messages = it.messages.filterNot { m -> m.id == userId } + ChatMessage(userId, "user",
                    submission.body.getString("message"), files = submission.files.map { a -> a.reference },
                    quote = submission.body.optString("quote_text"), gemName = it.selectedGem?.name.orEmpty())) }
                try { api.stream("/chat_stream", submission.body, token()) { event -> if (streamJob === owner) acceptEvent(id, event) } }
                catch (e: ApiException) {
                    if (e.code != "request_already_accepted") throw e
                    try { api.stream("/chat_stream_resume", JSONObject().put("thread_id", id).put("job_id", e.payload.getString("job_id")), token()) { event -> if (streamJob === owner) acceptEvent(id, event) } }
                    catch (resumeError: ApiException) { if (resumeError.status != 404) throw resumeError }
                }
                failed = null
                // Branching sends (edit-and-resend, regenerate) create a new leaf whose id is unknown
                // yet; drop the stale leafId so loadMessages() falls back to the newest message
                // instead of keeping the previous branch selected.
                if (submission.body.optBoolean("parent_id_explicit")) mutable.update { it.copy(leafId = null) }
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
        if (state.value.offline) { notify("オフライン中はMCPの確認に応答できません。"); return@launch }
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
    ) {
        if (state.value.offline) { notify("オフライン中はRealtimeを開始できません。"); return }
        if (state.value.realtime.active || modelId.isBlank()) return
        viewModelScope.launch {
            try {
                val started = api.post("/api/realtime/start", JSONObject()
                    .put("model", modelId)
                    .put("voice", voice)
                    .put("target_lang", targetLanguage.trim().lowercase().take(16).ifBlank { "ja" })
                    .put("thinking_level", thinkingLevel.trim().lowercase().ifBlank { "minimal" })
                    .put("transcription_mode", transcriptionMode.trim().uppercase().ifBlank { "VERBATIM" })
                    .put("custom_vocabulary", JSONArray(customVocabulary.split(',', '、', '\n')
                        .map { it.trim() }.filter { it.isNotBlank() }.take(1000))), token())
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
            "interaction_status" -> mutable.update { it.copy(realtime = it.realtime.copy(status = event.optString("status"))) }
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
        if (state.value.offline) { notify("オフライン中はLyriaを開始できません。"); return }
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
        } catch (e: Exception) {
            // Keep the Web BAN screen's logout escape hatch even if revoke fails.
            if (state.value.banned) clearSession() else report(e)
        }
    } }
    private suspend fun clearSession() {
        val caller = currentCoroutineContext().job
        listOf(pairingJob, navigationJob, streamJob, uploadJob, heartbeatJob, libraryJob, cacheSyncJob, batchPollJob,
            realtimeStreamJob, realtimeCaptureJob, lyriaStreamJob, importJob).forEach { if (it !== caller) it?.cancel() }
        realtimeTrack?.let { track -> runCatching { track.stop(); track.release() } }; realtimeTrack = null
        lyriaTrack?.let { track -> runCatching { track.stop(); track.release() } }; lyriaTrack = null
        withContext(NonCancellable + Dispatchers.IO) { store.clear() }
        GoogleAuthClient.clearCredentialState(getApplication())
        cancelChatBubble(getApplication())
        session = null; failed = null
        mutable.value = ChatState(
            starting = false,
            historyCacheMode = HistoryCacheMode.from(prefs.getString("offline_history_cache_mode", HistoryCacheMode.VIEWED.value)),
            cacheMobileDataAllowed = prefs.getBoolean("offline_cache_mobile_data", false),
        )
    }
    private suspend fun report(error: Throwable) {
        if (error is CancellationException) throw error
        if (error is ApiException && error.code == "banned") {
            enterBannedState(error)
            return
        }
        if (error is ApiException && error.status == 401) clearSession()
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
            notice = error.message?.take(500) ?: "通信に失敗しました。再試行してください。",
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
            val reply = api.get("/api/threads?page=$page&q=", token())
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
                val reply = api.get("/api/threads/${thread.id}?limit=200$suffix", token())
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
            val reply = api.get("/api/files?limit=40&offset=$offset&sort=newest&q=")
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
        if (state.value.offline) { mutable.update { it.copy(notice = "オフライン中はファイルをアップロードできません。") }; return }
        if (uris.isEmpty() || state.value.uploading) return
        if (uris.size + state.value.attachments.size > 30) { mutable.update { it.copy(notice = "添付は30件までです。") }; return }
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
                    val uploaded = if (source.size > CHUNK_UPLOAD_THRESHOLD_BYTES) uploadInChunks(source)
                        else uploadWhole(source)
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
        val accountId = state.value.account?.id ?: throw IOException("アカウント情報がありません。")
        val directory = File(getApplication<Application>().cacheDir, "shared").apply { mkdirs() }
        val suffix = reference.substringBefore('?').substringAfterLast('.', "bin").take(8).filter { it.isLetterOrDigit() }.ifBlank { "bin" }
        val target = File(directory, "${UUID.randomUUID()}.$suffix")
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
        return "/api/files?limit=40&offset=$offset&sort=newest&q=$query$favorites"
    }

    private suspend fun fetchLibrary(append: Boolean) {
        if (state.value.offline) {
            val accountId = state.value.account?.id ?: return
            val files = withContext(Dispatchers.IO) { offlineCache.loadLibrary(accountId) }
                .filter { file ->
                    (state.value.libraryQuery.isBlank() || file.displayName.contains(state.value.libraryQuery, true)) &&
                        (!state.value.libraryFavoritesOnly || file.isFavorite)
                }
            mutable.update { it.copy(library = files, libraryHasMore = false, libraryTotal = files.size, libraryBusy = false) }
            return
        }
        val offset = if (append) state.value.library.size else 0
        val reply = api.get(libraryPath(offset), token())
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
        if (state.value.offline || !state.value.libraryHasMore || state.value.libraryBusy) return
        libraryJob = viewModelScope.launch {
            mutable.update { it.copy(libraryBusy = true) }
            try { fetchLibrary(true) } catch (e: Exception) { report(e) } finally { mutable.update { it.copy(libraryBusy = false) } }
        }
    }

    fun toggleLibraryFavorite(file: LibraryFile) { viewModelScope.launch {
        if (state.value.offline) { notify("オフライン中はファイルのお気に入りを変更できません。"); return@launch }
        try {
            val reply = api.post("/api/files/favorite", JSONObject().put("filepath", file.filepath), token())
            val favorite = reply.optBoolean("is_favorite")
            mutable.update { current -> current.copy(library = current.library.map {
                if (it.filepath == file.filepath) it.copy(isFavorite = favorite) else it
            }) }
            state.value.account?.let { account -> withContext(Dispatchers.IO) { offlineCache.saveLibrary(account.id, state.value.library) } }
        } catch (e: Exception) { report(e) }
    } }

    fun renameLibraryFile(file: LibraryFile, name: String) { viewModelScope.launch {
        if (state.value.offline) { notify("オフライン中はファイル名を変更できません。"); return@launch }
        try {
            val reply = api.post("/api/files/rename", JSONObject().put("filepath", file.filepath).put("filename", name), token())
            val display = reply.optString("filename", name)
            mutable.update { current -> current.copy(library = current.library.map {
                if (it.filepath == file.filepath) it.copy(displayName = display) else it
            }) }
            state.value.account?.let { account -> withContext(Dispatchers.IO) { offlineCache.saveLibrary(account.id, state.value.library) } }
        } catch (e: Exception) { report(e) }
    } }

    fun deleteLibraryFile(file: LibraryFile) { viewModelScope.launch {
        if (state.value.offline) { notify("オフライン中はファイルを削除できません。"); return@launch }
        try {
            api.post("/api/files/delete", JSONObject().put("filenames", JSONArray().put(file.filepath)), token())
            state.value.account?.let { account -> withContext(Dispatchers.IO) { offlineCache.deleteLibraryFile(account.id, file.filepath) } }
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
        val payload = api.get("/api/mobile/v1/preferences", token())
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

    fun savePreferences(payload: JSONObject, message: String = "設定を保存しました") {
        viewModelScope.launch {
            if (state.value.offline) { notify("オフライン中はアカウント設定を保存できません。接続後に再試行してください。"); return@launch }
            mutable.update { it.copy(prefsBusy = true) }
            try {
                val reply = api.put("/api/mobile/v1/preferences", payload, token())
                mutable.update { it.copy(preferences = parsePreferences(reply), notice = message) }
            } catch (e: Exception) { report(e) }
            finally { mutable.update { it.copy(prefsBusy = false) } }
        }
    }

    fun loadStorageUsage() {
        viewModelScope.launch {
            if (state.value.offline) return@launch
            runCatching { mutable.update { it.copy(storage = parseStorageUsage(api.get("/api/storage", token()))) } }
                .onFailure { report(it) }
        }
    }

    fun loadFeedback() {
        viewModelScope.launch {
            if (state.value.offline) return@launch
            mutable.update { it.copy(feedbackBusy = true) }
            try { mutable.update { it.copy(feedbackItems = parseFeedbackItems(api.get("/api/feedback", token()))) } }
            catch (e: Exception) { report(e) }
            finally { mutable.update { it.copy(feedbackBusy = false) } }
        }
    }

    fun submitFeedback(title: String, message: String) {
        if (message.isBlank()) return
        viewModelScope.launch {
            if (state.value.offline) { notify("オフライン中はフィードバックを送信できません。"); return@launch }
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
            if (state.value.offline) return@launch
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
    private var pdfExporting = false

    /** Web `openThreadPdfPrintDialog` guards and failure toast, then a native PDF for the share sheet. */
    fun exportPdf(onReady: (File) -> Unit) {
        val id = state.value.selected?.id
        if (id == null) { notify("PDF化するスレッドを開いてください"); return }
        if (pdfExporting) { notify("PDF出力の準備中です。しばらくお待ちください。"); return }
        if (state.value.offline) { notify("PDF出力に失敗しました"); return }
        pdfExporting = true
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
            } catch (e: CancellationException) { throw e }
            catch (_: Exception) { notify("PDF出力に失敗しました") }
            finally { pdfExporting = false; mutable.update { it.copy(busy = false) } }
        }
    }

    private companion object {
        const val CHUNK_UPLOAD_THRESHOLD_BYTES = 8L * 1024 * 1024
        const val MAX_SINGLE_UPLOAD_BYTES = 64L * 1024 * 1024
        const val PREF_BROWSER_LOGIN_VERIFIER = "browser_login_pkce_verifier"
        const val PREF_BROWSER_LOGIN_STARTED_AT = "browser_login_started_at"
        const val BROWSER_LOGIN_TTL_MS = 10L * 60 * 1000
    }
}
