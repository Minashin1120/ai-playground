package com.minashin1120.aiplayground

import android.app.Application
import android.content.Context
import android.net.ConnectivityManager
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.minashin1120.aiplayground.data.*
import com.minashin1120.aiplayground.data.backend.ChatBackend
import com.minashin1120.aiplayground.data.backend.ServerChatBackend
import com.minashin1120.aiplayground.data.direct.DirectHttp
import com.minashin1120.aiplayground.data.local.LocalProfiles
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import org.json.JSONObject
import java.io.File
import java.io.IOException

class ChatViewModel(application: Application) : AndroidViewModel(application) {
    internal val api = PlaygroundApi()
    internal val serverBackend = ServerChatBackend(api)
    internal val backend: ChatBackend get() = profiles.localBackend ?: serverBackend
    internal val localProfiles = LocalProfiles(application)
    internal val directHttp by lazy { DirectHttp() }
    /** Separate client for background sync, so its requests never show the global spinner. */
    internal val syncApi by lazy { PlaygroundApi().also(::observeServerResponses) }

    internal val store = TokenStore(application)
    internal val playIntegrity = PlayIntegrityClient(application)
    internal val offlineCache = OfflineCacheStore(application)
    internal val prefs = application.getSharedPreferences("navigation", 0)
    internal val connectivity: ConnectivityManager? = application.getSystemService(ConnectivityManager::class.java)
    internal val mutable = MutableStateFlow(ChatState())
    val state = mutable.asStateFlow()
    internal val chatHistory = ChatNavigationHistory()
    /** Web global spinner label (`progress_spinner.js`); null while no tracked request runs. */
    val progressLabel = api.progress.label
    internal var session: StoredSession? = null
    /** Keeps the app running (foreground service and its notification) while an answer streams. */
    internal val answerKeepAlive by lazy { GenerationService.keepAlive(getApplication<Application>()) }
    internal var pairingJob: Job? = null
    internal val generations = ChatGenerations()
    private var chatTransitionSequence = 0L

    /**
     * 協作者（受信・送信・添付・同期・音声・設定など）の組み立てと配線（[ChatGraph]）。
     * 画面が呼ぶ入口はこのクラスの薄い公開関数だけにし、状態と協作者はここから参照する。
     */
    private val graph = ChatGraph(this, application, viewModelScope)

    private val lowBandwidth get() = graph.lowBandwidth
    private val connection get() = graph.connection
    private val diagnostics get() = graph.diagnostics
    private val accountImport get() = graph.accountImport
    private val accountTransferOps get() = graph.accountTransferOps
    private val auth get() = graph.auth
    private val feedback get() = graph.feedback
    private val attachmentUploader get() = graph.attachmentUploader
    private val imageSplitManager get() = graph.imageSplitManager
    private val micInput get() = graph.micInput
    private val deviceAnswer get() = graph.deviceAnswer
    private val pendingSend get() = graph.pendingSend
    private val composer get() = graph.composer
    private val slashCommands get() = graph.slashCommands
    private val pdfExport get() = graph.pdfExport
    private val preferences get() = graph.preferences
    private val aiSettings get() = graph.aiSettings
    private val accountSetup get() = graph.accountSetup
    private val threadList get() = graph.threadList
    private val batch get() = graph.batch
    private val skills get() = graph.skills
    private val messageLoader get() = graph.messageLoader
    private val profiles get() = graph.profiles
    private val foreground get() = graph.foreground
    private val attachments get() = graph.attachments
    private val sendEntry get() = graph.sendEntry
    private val editing get() = graph.editing
    private val navigator get() = graph.navigator
    private val threadActions get() = graph.threadActions
    private val realtimeSession get() = graph.realtimeSession
    private val lyriaStudio get() = graph.lyriaStudio
    private val banState get() = graph.banState
    private val botGuard get() = graph.botGuard
    private val sendRunner get() = graph.sendRunner
    private val lifecycle get() = graph.lifecycle
    private val startup get() = graph.startup
    private val library get() = graph.library

    init {
        observeServerResponses(api)
        // Web `apiFetch`: any request answered with `account_locked` shows the lock overlay (never for admins).
        api.onAccountLocked = { message, seconds ->
            mutable.update { current ->
                if (current.preferences?.isAdmin == true || current.accountLock != null) current
                else current.copy(accountLock = AccountLock(message, System.currentTimeMillis() + seconds.coerceAtLeast(0) * 1000))
            }
        }
        // Web keeps the library order and favorites filter in localStorage.
        mutable.update { it.copy(
            librarySort = prefs.getString(ChatLibrary.LIB_SORT_KEY, null)?.takeIf { value -> value in LIBRARY_SORTS } ?: "newest",
            libraryFavoritesOnly = prefs.getBoolean(ChatLibrary.LIB_FAVORITES_ONLY_KEY, false),
        ) }
        diagnostics.start()
        startup.start()
    }


    // region: Ban state, transitions, network, and low bandwidth
    internal fun enterBannedState(error: ApiException) = banState.enter(error)

    /** Web `submit_ban_appeal`; returns true when the appeal was stored (the form is cleared). */
    suspend fun submitBanAppeal(message: String): Boolean = banState.submitAppeal(message)
    internal fun nextChatTransition(kind: ChatTransitionKind): Pair<Long, ChatTransitionKind> {
        chatTransitionSequence += 1L
        return chatTransitionSequence to kind
    }

    /** Sidebar button: auto → on → off ([ChatLowBandwidth]). */
    fun cycleLowBandwidth() = lowBandwidth.cycle()


    // region: Profiles, offline guards, and server origin ([ChatProfiles])
    internal val chatLocal: Boolean get() = profiles.chatLocal
    internal val uploadsLocal: Boolean get() = profiles.uploadsLocal
    internal fun deviceChat(id: String?): Boolean = profiles.deviceChat(id)
    internal fun notifySendOffline(offline: Boolean = state.value.offline): Boolean = profiles.notifySendOffline(offline)
    internal fun isChatOfflineBlocked(id: String?): Boolean = profiles.isChatOfflineBlocked(id)
    internal fun notifyDeleteMessageOffline(threadId: String?, messageId: Int, notice: String): Boolean =
        profiles.notifyDeleteMessageOffline(threadId, messageId, notice)
    internal fun isThreadListOfflineBlocked(): Boolean = profiles.isThreadListOfflineBlocked()
    internal fun notifyChatOffline(id: String?, notice: String): Boolean = profiles.notifyChatOffline(id, notice)
    internal fun notifyOffline(notice: String): Boolean = profiles.notifyOffline(notice)
    /** Session token; the no-account profile talks to the device only, so it has none. */
    internal fun token(): String = session?.token ?: if (state.value.localProfile) "" else throw IOException("端末連携が必要です。")
    internal suspend fun serverIdOf(id: String): String = profiles.serverIdOf(id)
    /** Settings "今すぐ同期" / "サーバーのAPIキーを端末へ取り込む" / 端末チャットの取り込み。 */
    fun syncNow() = profiles.syncNow()
    fun importServerSecrets(secrets: JSONObject) = profiles.importServerSecrets(secrets)
    fun importLocalProfileChats() = profiles.importLocalProfileChats()
    fun startLocalProfile() = profiles.startLocalProfile()
    fun leaveLocalProfile() = profiles.leaveLocalProfile()
    fun setServerless(enabled: Boolean) = profiles.setServerless(enabled)
    fun selectServer(input: String) = profiles.selectServer(input)

    // region: Authentication, Turnstile, passkeys, and security ([ChatAuth])
    fun startSessionTurnstile() = auth.startSessionTurnstile()
    fun completeSessionTurnstile(ticket: String) = auth.completeSessionTurnstile(ticket)
    fun dismissSessionTurnstile() = auth.dismissSessionTurnstile()
    fun integrityTurnstileComplete(ticket: String) = auth.integrityTurnstileComplete(ticket)
    fun login(username: String, password: String) = auth.login(username, password)
    fun signup(username: String, password: String) = auth.signup(username, password)
    fun beginGoogleLogin() = auth.beginGoogleLogin()
    fun completeGoogleLogin(credential: GoogleAuthClient.Result) = auth.completeGoogleLogin(credential)
    fun cancelGoogleLogin(
        message: String = "Googleログインをキャンセルしました。",
        diagnostics: String? = null,
    ) = auth.cancelGoogleLogin(message, diagnostics)

    fun verifyTotp(code: String) = auth.verifyTotp(code)

    /** Starts a browser login whose returned code only this app can redeem (PKCE). */
    fun browserLoginPath(provider: String): String = auth.browserLoginPath(provider)

    /** Exchanges the one-time code returned to the verified HTTPS App Link. */
    fun exchangeNativeCode(code: String) = auth.exchangeNativeCode(code)
    fun beginPasskeyLogin(username: String) = auth.beginPasskeyLogin(username)
    fun beginWebauthnTwoFactor() = auth.beginWebauthnTwoFactor()
    fun beginPasskeyRegistration(name: String = "") = auth.beginPasskeyRegistration(name)
    fun submitCredential(responseJson: String) = auth.submitCredential(responseJson)
    fun cancelCredentialRequest(message: String? = null) = auth.cancelCredentialRequest(message)
    fun loadSecurity() = auth.loadSecurity()
    fun startTotpSetup() = auth.startTotpSetup()
    fun cancelTotpSetup() = auth.cancelTotpSetup()
    fun enableTotp(code: String) = auth.enableTotp(code)
    fun disableTotp(code: String) = auth.disableTotp(code)
    fun removePasskey(id: String) = auth.removePasskey(id)
    fun saveSecurityPreferences(default2fa: String, passkeyOnly: Boolean, skipGoogle: Boolean) =
        auth.saveSecurityPreferences(default2fa, passkeyOnly, skipGoogle)

    // region: Account import and setup ([ChatAccountImport])
    /** Uploads and imports an account ZIP in a resumable, cancellable chunk session. */
    fun importAccountZip(uri: Uri) = accountImport.importAccountZip(uri)

    /** Applies the uploaded archive after the user confirms the settings changes. */
    fun confirmSetupImportSettings() = accountImport.confirmSetupImportSettings()

    /** Rejects the settings overwrite and removes the completed upload session. */
    fun cancelSetupImportConfirmation() = accountImport.cancelSetupImportConfirmation()
    fun cancelSetupImport() = accountImport.cancelSetupImport()


    internal suspend fun loadSetup() = accountSetup.loadSetup()
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
    ) = accountSetup.finishSetup(
        defaultModel, openaiKey, geminiKey, anthropicKey, deepseekKey, kimiKey, mistralKey, ideogramKey,
        zaiKey, xaiKey, googleKey, googleProject, vertexProject, vertexLocation, vertexCredentialsJson, enableE2ee,
    )

    internal suspend fun loadAccount() = accountSetup.loadAccount()
    // region: Foreground, pairing, notices, and connection
    /**
     * An answer is being generated on the device (it has no server job to rejoin). Cancelling it drops the
     * provider request and loses the result, so leaving the app or coming back must not cancel it.
     */
    internal fun answeringOnDevice(): Boolean = deviceAnswer.answeringOnDevice()
    fun setForeground(value: Boolean) = foreground.setForeground(value)
    fun cancelPairing() { pairingJob?.cancel(); foreground.cancelPairing() }
    fun dismissNotice() = foreground.dismissNotice()
    fun notify(message: String) { mutable.update { it.copy(notice = message) } }

    // region: Server connectivity ([ChatConnectionMonitor])
    internal fun hasUsableNetwork(): Boolean = connection.hasUsableNetwork()
    internal fun startConnectionMonitor() = connection.start()
    internal fun stopConnectionMonitor() = connection.stop()
    internal fun observeServerResponses(client: PlaygroundApi) = connection.observe(client)
    internal fun setConnectionUnavailable(status: ConnectionStatus, message: String = status.defaultMessage()) =
        connection.setUnavailable(status, message)

    internal fun markConnectionReachable(
        wasDisconnected: Boolean = state.value.connectionStatus.isDisconnected(),
        resetSlowCount: Boolean = true,
    ) = connection.markReachable(wasDisconnected, resetSlowCount)

    /** Web `discardIdleServerConnections`: 各クライアントの待機ソケットを落とす（[ChatConnectionMonitor]）。 */
    internal fun discardIdleServerConnections() = connection.discardIdleServerConnections()


    override fun onCleared() {
        graph.release()
        super.onCleared()
    }


    // region: Draft, model, and generation options (ChatComposer)
    fun draft(text: String) = composer.draft(text)
    fun insertSharedText(text: String) = composer.insertSharedText(text)
    fun quoteMessage(text: String, sourceMessageId: Int? = null, sourceKey: String? = sourceMessageId?.toString()) =
        composer.quoteMessage(text, sourceMessageId, sourceKey)
    fun navigateToQuote() = composer.navigateToQuote()
    fun clearQuote() = composer.clearQuote()
    fun chooseModel(model: String) = composer.chooseModel(model)
    fun chooseModelByUser(model: String) = composer.chooseModelByUser(model)
    fun toggleThinking() = composer.toggleThinking()
    fun toggleSearch() = composer.toggleSearch()
    fun toggleUrlContext() = composer.toggleUrlContext()
    fun toggleMaps() = composer.toggleMaps()
    fun toggleFileCreation() = composer.toggleFileCreation()
    fun toggleSystemPrompt() = composer.toggleSystemPrompt()
    fun togglePromptCache() = composer.togglePromptCache()
    fun clearPromptCache() = composer.clearPromptCache()
    fun toggleBatchMode() = composer.toggleBatchMode()
    fun togglePython() = composer.togglePython()
    fun toggleMcp() = composer.toggleMcp()
    fun toggleSkills() = composer.toggleSkills()
    fun toggleCanvas() = composer.toggleCanvas()
    fun toggleCoding() = composer.toggleCoding()
    fun toggleTemporaryChat() = threadActions.toggleTemporaryChat()
    fun selectCodingTarget(target: CodingTarget?) = composer.selectCodingTarget(target)
    fun setImageMask(reference: String?) = composer.setImageMask(reference)
    /** Web `uploadMaskFile`: the chosen mask image is uploaded as it is. */
    fun uploadImageMask(uri: Uri) = attachmentUploader.uploadMask(uri)
    fun uploadImageMask(png: ByteArray) = attachmentUploader.uploadMask(png)
    fun setVisionModel(id: String) = composer.setVisionModel(id)
    fun renameAttachment(reference: String, input: String) = composer.renameAttachment(reference, input)
    /**
     * Web `saveMarkerToRow`: uploads the edited PNG as `<name>_marked.png` and swaps it into the row,
     * keeping the first pre-edit file as the row's original.
     */
    fun applyImageEdit(reference: String, png: ByteArray, attachOriginal: Boolean) =
        attachmentUploader.uploadEditedAttachment(reference, png, attachOriginal)
    fun generationOption(key: String, value: String) = composer.generationOption(key, value)
    fun removeAttachment(reference: String) = composer.removeAttachment(reference)
    // region: Thread list and search ([ChatThreadList])
    fun search(query: String) = threadList.search(query)
    internal suspend fun fetchThreads(more: Boolean) = threadList.fetch(more)
    fun moreThreads() = threadList.more()
    // region: Chat navigation and thread opening ([ChatNavigator])
    /** 新規・戻る・スレッドを開く・再読込は [ChatNavigator]。ここは画面が呼ぶ入口だけを残す。 */
    fun goBackInChats() = navigator.goBack()
    fun newChat(temporary: Boolean = false, recordHistory: Boolean = true) = navigator.newChat(temporary, recordHistory)
    fun openThread(thread: ThreadItem, recordHistory: Boolean = true, focusMessageId: Int? = null) =
        navigator.openThread(thread, recordHistory, focusMessageId)

    fun retryOpenThread() = navigator.retryOpenThread()

    // region: Message loading, cache, and refresh
    /** チャット履歴の読み込みは [ChatMessageLoader]。ここは画面と他部品が呼ぶ入口だけを残す。 */
    internal suspend fun loadMessages(requestedId: String, older: Boolean = false, autoResume: Boolean = true, inPlace: Boolean = false, streamFinished: Boolean = false) =
        messageLoader.load(requestedId, older, autoResume, inPlace, streamFinished)

    internal fun syncHeartbeat() = foreground.syncHeartbeat()
    fun olderMessages() = navigator.olderMessages()
    fun accountLockExpired() = navigator.accountLockExpired()
    fun refresh() = navigator.refresh()
    internal fun refreshInPlace() = navigator.refreshInPlace()

    // region: Thread and message actions ([ChatThreadActions])
    /** 削除・名前変更・分岐・メッセージ削除・暗号化・設定の保存は [ChatThreadActions]。ここは画面が呼ぶ入口だけを残す。 */
    fun deleteThread(thread: ThreadItem, onNewChat: () -> Unit = {}) = threadActions.deleteThread(thread, onNewChat)
    fun renameThread(thread: ThreadItem, title: String) = threadActions.renameThread(thread, title)
    fun branchChat(message: ChatMessage, confirmed: Boolean = false) = threadActions.branchChat(message, confirmed)
    fun openBranchOrigin() = threadActions.openBranchOrigin()
    fun deleteMessage(message: ChatMessage) = threadActions.deleteMessage(message)
    fun setThreadEncryption(enable: Boolean, onReauth: (retry: () -> Unit) -> Unit, onDone: (Boolean) -> Unit) =
        threadActions.setThreadEncryption(enable, onReauth, onDone)

    fun reloadThreads(onDone: () -> Unit) = threadActions.reloadThreads(onDone)
    fun toggleBookmark(thread: ThreadItem) = threadActions.toggleBookmark(thread)
    fun saveThreadSettings(title: String, instruction: String, includeGlobal: Boolean, temporary: Boolean) =
        threadActions.saveThreadSettings(title, instruction, includeGlobal, temporary)

    fun saveChatInstructions(instruction: String, includeGlobal: Boolean, userPrompt: JSONObject, onDone: (Boolean) -> Unit) =
        threadActions.saveChatInstructions(instruction, includeGlobal, userPrompt, onDone)

    fun requestSettings(tab: String? = null, card: String? = null) {
        mutable.update { it.copy(settingsRequest = it.settingsRequest + 1, settingsRequestTab = tab, settingsRequestCard = card) }
    }
    /** Pull-to-refresh of the sidebar Skill list. */
    fun reloadSkills(onDone: () -> Unit) { viewModelScope.launch {
        if (notifyOffline("オフライン中はスキルを更新できません。")) { onDone(); return@launch }
        try { fetchSkills() } catch (e: Exception) { report(e) } finally { onDone() }
    } }

    /** `/static/legal/<kind>.md` shown by the terms / privacy modal (public, no token). */
    /** Account, session and data-transfer operations used by the settings modal ([ChatAccountTransfer]). */
    fun accountApi(): AccountApi = accountTransferOps.api()

    /** Settings Data tab export / import / dedupe; outlives the settings modal like the Web tab. */
    val accountTransfer: AccountTransferController get() = accountTransferOps.controller

    /** App Link back from `/android/link/<provider>` (Web flash text after linking). */
    fun linkCompleted(provider: String) {
        notify(if (provider == "minashin") "Minashin アカウントと連携しました。" else "Google アカウントと連携しました。")
        loadPreferences()
    }

    /** Ends this device's sign-in after the account or every session was removed on the server. */
    fun signedOutRemotely() { viewModelScope.launch { clearSession() } }

    /** Web `openThreadModal`: a new chat is created first so its settings can be edited. */
    fun ensureThread(onReady: () -> Unit) = threadActions.ensureThread(onReady)

    /** Web `schedulePromptTokenEstimate` request (`POST /api/token_estimate`). 本体は [ChatComposer]。 */
    suspend fun estimatePromptTokens(model: String, message: String, quote: String, imageUrls: List<String>): JSONObject =
        composer.estimatePromptTokens(model, message, quote, imageUrls)

    suspend fun estimatePromptTokenDetails(model: String, message: String, quote: String, imageUrls: List<String>): JSONObject =
        composer.estimatePromptTokenDetails(model, message, quote, imageUrls)

    suspend fun legalMarkdown(kind: String): String {
        val safe = if (kind == "privacy") "privacy" else "terms"
        return api.getText("/static/legal/$safe.md?t=${System.currentTimeMillis()}")
    }

    // region: Prompt options, editing, and branching ([ChatEditing])

    /** Prepares the composer to edit a user message, branching from its parent. */
    fun beginEdit(message: ChatMessage) = editing.beginEdit(message)

    /** Re-sends the user message that produced an assistant reply, creating a sibling branch. */
    fun regenerate(message: ChatMessage) = editing.regenerate(message)
    fun cancelEdit() = editing.cancelEdit()

    /** Switches the active path to the branch that contains [targetMessageId]. */
    fun switchBranch(targetMessageId: Int) = editing.switchBranch(targetMessageId)
    fun switchBranchByIndex(siblings: List<ChatMessage>, index: Int) = editing.switchBranchByIndex(siblings, index)

    // region: Send entry and helpers
    fun send(xLinkChecked: Boolean = false, disableAutoSearch: Boolean = false, preChecked: Boolean = false) =
        sendRunner.send(xLinkChecked, disableAutoSearch, preChecked)

    fun retry() { pendingSend.failed?.let { sendRunner.submit(it) } }

    /** Web `sendMessage` の入口（連投ロック・待機中の案内）。本体は [ChatBotGuard]。 */
    fun beginSend(): Boolean = botGuard.beginSend()

    /** Web `botTelemetry` の入口（`MainActivity.dispatchTouchEvent` から呼ぶ）。 */
    fun recordBotTouch(action: Int, eventTime: Long) = botGuard.recordTouch(action, eventTime)
    fun resume() = sendRunner.resume()
    fun stop() = sendRunner.stop()


    /** Web `mic-btn`: 録音の開始・停止と文字起こし。 */
    fun toggleMicRecording() = micInput.toggle()

    /** Web `applyXLinkAuto`. */
    internal fun applyXLinkSearch() = sendEntry.applyXLinkSearch()

    /** Web `#auto-search-banner`: 検索ONで続行 / 検索OFFで回答, optionally remembered (次回から尋ねない). */
    fun resolveXLinkPrompt(enable: Boolean, remember: Boolean) = sendEntry.resolveXLinkPrompt(enable, remember)

    /** Web `api-key-modal-save-btn`: saves the key for the model's provider, then sends again. */
    fun saveApiKeyAndResend(key: String) = sendEntry.saveApiKeyAndResend(key)

    /** `api-key-modal-cancel-btn` shows the original error; `api-key-modal-fallback-btn` leaves the model picker to the UI. */
    fun dismissApiKeyPrompt(showError: Boolean) = sendEntry.dismissApiKeyPrompt(showError)
    fun modelDisplayName(modelId: String): String = sendEntry.modelDisplayName(modelId)

    /** Web `showPendingSlashCommandIndicator` / `hidePendingSlashCommandIndicator`。会話は [ChatAiSettings]。 */
    fun setPendingSlashCommand(id: String?) = aiSettings.setPendingSlashCommand(id)

    /** Web `runAiSettingsCommand`: `/settings <instruction>` changes or inspects settings through the selected model. */
    fun runAiSettings(instruction: String) = aiSettings.run(instruction)

    // region: MCP

    fun resolveMcpDecision(allow: Boolean) = preferences.resolveMcpDecision(allow)

    // region: Batch jobs ([ChatBatch])
    internal suspend fun fetchBatchJobs(notify: Boolean, status: Boolean = notify) = batch.fetch(notify, status)
    fun dismissBatchBanner(open: Boolean) = batch.dismissBanner(open)
    fun refreshBatchJobs() = batch.refresh()
    internal fun startBatchPolling() = batch.startPolling()

    /** Web `loadBatchJobs`; [silent] skips the failure toast (the 5-second refresh in the modal). */
    fun loadBatchJobs(silent: Boolean) = batch.load(silent)
    fun cancelBatchJob(job: BatchJob) = batch.cancel(job)
    fun deleteBatchJob(job: BatchJob) = batch.delete(job)

    /** 一覧・通知からスレッドIDで開く。本体は [ChatNavigator]（未知のIDは Batch チャット扱い）。 */
    fun openThreadId(id: String) = navigator.openThreadId(id)
    fun openBubbleTarget(id: String?) = navigator.openBubbleTarget(id)

    // region: Native realtime audio sessions
    /** Web `RealtimeVoiceSession`: start/stop and the one-shot `/sts` recorder live in `ChatRealtimeSession`. */
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
    ) = realtimeSession.start(modelId, voice, targetLanguage, thinkingLevel, transcriptionMode, customVocabulary,
        includeThoughts, speed, rateIn, rateOut, autoPlay, reasoningEffort)

    fun stopRealtime(save: Boolean) = realtimeSession.stop(save)
    fun startOneShotSts(modelId: String, fields: Map<String, String>, autoPlay: Boolean, autoRestart: Boolean) =
        realtimeSession.startOneShot(modelId, fields, autoPlay, autoRestart)

    // region: Native Lyria RealTime studio
    /** Web `openLyriaStudio(promptText)` and the studio controls live in `ChatLyriaStudio`. */
    fun prepareLyriaPrompt(text: String) = lyriaStudio.preparePrompt(text)
    fun lyriaStart(prompts: List<LyriaPrompt>, config: JSONObject) = lyriaStudio.start(prompts, config)
    fun lyriaControl(action: String) = lyriaStudio.control(action)
    fun lyriaApplyPrompts(prompts: List<LyriaPrompt>) = lyriaStudio.applyPrompts(prompts)
    fun lyriaApplyConfig(config: JSONObject, resetContext: Boolean) = lyriaStudio.applyConfig(config, resetContext)
    fun lyriaSave(onSaved: () -> Unit) = lyriaStudio.saveToChat(onSaved)
    fun stopLyria(save: Boolean = false) = lyriaStudio.stop(save)
    fun logout() = lifecycle.logout()
    private suspend fun clearSession() = lifecycle.clearSession()
    /** [message] replaces the error text for the Web toasts that use a fixed wording. */
    internal suspend fun report(error: Throwable, message: String? = null) = lifecycle.report(error, message)
    internal suspend fun refreshOfflineCacheStats(accountId: Int? = state.value.account?.id) =
        preferences.refreshOfflineCacheStats(accountId)

    fun saveCacheSettings(mode: HistoryCacheMode, mobileDataAllowed: Boolean) = preferences.saveCacheSettings(mode, mobileDataAllowed)
    fun clearOfflineCache(category: CacheCategory) = preferences.clearOfflineCache(category)
    fun syncOfflineCache() = preferences.syncOfflineCache()
    fun cancelCacheSync() = preferences.cancelCacheSync()
    internal fun startCacheSyncIfAllowed() = preferences.startCacheSyncIfAllowed()

    /** Clears the offline banner and retries the last account or history load ([ChatStartup]). */
    fun reconnect() = startup.reconnect()
    // region: Attachments ([ChatAttachments])
    /** Account/profile-scoped decoded-image identity; safe to call from composition without IO. */
    fun attachmentImageCacheKey(reference: String): String = attachments.imageCacheKey(reference)
    suspend fun loadAttachmentBytes(reference: String, thumbnail: Boolean, limit: Long = 8L * 1024 * 1024): ByteArray? =
        attachments.loadAttachmentBytes(reference, thumbnail, limit)
    fun upload(uris: List<Uri>) = attachments.upload(uris)
    private fun upload(uris: List<Uri>, skipCompression: Boolean) = attachments.upload(uris, skipCompression)
    /** メッセージとしてプレビュー (ANDROID_ONLY.md). */
    internal val sharedTextPreview = state.map { it.sharedTextPreview }.stateIn(viewModelScope, SharingStarted.Eagerly, null)
    fun openSharedTextPreview(text: String) = attachments.openSharedTextPreview(text)
    fun closeSharedTextPreview() = attachments.closeSharedTextPreview()
    /** 画像分割 (ANDROID_ONLY.md). */
    internal val imageSplit = imageSplitManager.state
    fun openImageSplit(uris: List<Uri>) = attachments.openImageSplit(uris)
    fun closeImageSplit() = attachments.closeImageSplit()
    internal fun runImageSplit(options: ImageSplitOptions, attach: Boolean, tree: Uri? = null) =
        attachments.runImageSplit(options, attach, tree)
    fun cancelUpload() = attachments.cancelUpload()
    fun resetUploads() = attachments.resetUploads()
    fun saveCompressionSettings(settings: CompressionSettings) = attachments.saveCompression(settings)
    suspend fun downloadAttachment(reference: String): Pair<File, String> = attachments.downloadAttachment(reference)
    fun openFile(reference: String, onReady: (File, String) -> Unit) = attachments.openFile(reference, onReady)

    // region: File library
    /** Web `#files-modal`: list, search, rename, delete, attach and download live in `ChatLibrary`. */
    fun refreshLibrary() = library.refresh()
    fun librarySearch(query: String) = library.search(query)
    fun setLibraryFavoritesOnly(value: Boolean) = library.setFavoritesOnly(value)
    fun moreLibrary() = library.more()
    fun toggleLibraryFavorite(file: LibraryFile) = library.toggleFavorite(file)
    fun renameLibraryFile(file: LibraryFile, name: String) = library.rename(file, name)
    fun deleteLibraryFile(file: LibraryFile) = library.delete(file)
    fun deleteLibraryFiles(filepaths: List<String>) = library.deleteAll(filepaths)
    fun setLibrarySort(order: String) = library.setSort(order)
    fun attachLibraryFiles(files: List<LibraryFile>) = library.attach(files)
    suspend fun libraryFileUsage(file: LibraryFile): Pair<List<FileUsageChat>, Boolean> = library.usage(file)
    fun saveLibraryFiles(files: List<LibraryFile>, target: suspend (LibraryFile, File, String) -> Unit) =
        library.saveFiles(files, target)
    fun reuseImage(reference: String, name: String = ""): Boolean = library.reuseImage(reference, name)
    fun reuseLibraryFile(file: LibraryFile) = library.reuse(file)

    // region: Skills ([ChatSkills])

    internal suspend fun fetchSkills() = skills.fetch()
    fun loadSkills() = skills.load()
    fun saveSkill(uuid: String?, name: String, description: String, instruction: String, onDone: (Boolean) -> Unit) =
        skills.save(uuid, name, description, instruction, onDone)

    fun deleteSkill(skill: Skill) = skills.delete(skill)
    fun chooseSkill(skill: Skill?) = skills.choose(skill)
    fun applySkillMention(skill: Skill) = skills.applyMention(skill)

    // region: General preferences and this device's session

    fun loadPreferences() = preferences.loadPreferences()

    /** Web `saveRichPastePromptPreferences`: saved quietly while the rich paste prompt is edited. */
    fun saveRichPastePrompt(prompt: String, useCustomDefault: Boolean) = preferences.saveRichPastePrompt(prompt, useCustomDefault)
    fun savePreferences(payload: JSONObject, message: String = "設定を保存しました") = preferences.savePreferences(payload, message)
    fun loadStorageUsage() = preferences.loadStorageUsage()
    fun loadFeedback() = feedback.loadFeedback()

    /** Web `.fb-save` in `#fb-admin-list`: an administrator sets the status, then the list is read again. */
    fun updateFeedback(item: FeedbackItem, status: String) = feedback.updateFeedback(item, status)

    /**
     * Web `.fb-reply-send`: the sender (on their own feedback) or an administrator adds a reply. [withLogs] is the
     * ticked box: the last hour of the activity log goes with the reply (only while ログの収集を強化 is on);
     * [images] follow the reply once it was accepted.
     */
    fun replyFeedback(item: FeedbackItem, message: String, withLogs: Boolean, images: List<Uri> = emptyList(), withChat: Boolean = false) =
        feedback.replyFeedback(item, message, withLogs, images, withChat)

    /** Deletes a feedback (the user's own, or any for an administrator); the server also removes its activity log and chat copy. */
    fun deleteFeedback(item: FeedbackItem) = feedback.deleteFeedback(item)

    /** [chatId] is the open chat whose copy goes with the feedback (null when the box is not ticked); [images] are the attached images. */
    fun submitFeedback(title: String, message: String, chatId: String? = null, images: List<Uri> = emptyList(), attachLogs: Boolean = false, onStored: () -> Unit = {}) =
        feedback.submitFeedback(title, message, chatId, images, attachLogs, onStored)

    fun loadMcpServers() = preferences.loadMcpServers()
    fun setMcpServerEnabled(server: McpServerInfo, enabled: Boolean) = preferences.setMcpServerEnabled(server, enabled)

    /** Web `executeMinimalSlashCommand`。本体は [ChatSlashCommands]。 */
    fun runSlashCommand(command: SlashCommand, argument: String, onLocal: (String) -> Unit): Boolean =
        slashCommands.run(command, argument, onLocal)

    /** Fetches the server thread payload and writes a native A4 PDF into the share cache. */
    /** Web `openThreadPdfPrintDialog` guards and failure toast, then a native PDF for the share sheet. */
    fun exportPdf(onReady: (File) -> Unit) = pdfExport.export(onReady)

    internal companion object {
        const val CHUNK_UPLOAD_THRESHOLD_BYTES = CHAT_UPLOAD_CHUNK_THRESHOLD_BYTES
        /** Web `CONNECTION_RETRY_DELAY_MS`. */
        const val CONNECTION_RETRY_DELAY_MS = 2000L
        const val STOP_RELOAD_ATTEMPTS = 8
        /** Away from the app at least this long, a server answer is rejoined rather than read on. */
        const val STREAM_REJOIN_AFTER_MS = 30_000L
        const val MAX_SINGLE_UPLOAD_BYTES = CHAT_UPLOAD_MAX_SINGLE_BYTES
        const val PREF_SERVER_ORIGIN = "server_origin"
        const val PREF_PROFILE_MODE = "profile_mode"
        const val PROFILE_LOCAL = "local"
        const val PROFILE_SERVER = "server"
        const val LOCAL_PROFILE_NAME = "この端末"
        const val PREF_SAVED_SERVERS = "saved_servers"
    }
}

/**
 * Coming back after [awayMs], a server answer is rejoined unless the app kept running the whole time
 * and its events were still arriving (quiet for less than [afterMs]).
 */
internal fun shouldRejoinStream(awayMs: Long, keptRunning: Boolean, quietMs: Long, afterMs: Long): Boolean =
    awayMs >= afterMs && (!keptRunning || quietMs >= afterMs)

/** Publish the ready attachment and retire its temporary upload row in the same state update. */
internal fun ChatState.withUploadedAttachment(attachment: Attachment): ChatState = copy(
    attachments = attachments + attachment, uploadName = "", uploadSent = 0, uploadTotal = 0,
)
