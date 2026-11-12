package com.minashin1120.aiplayground

import com.minashin1120.aiplayground.data.Account
import com.minashin1120.aiplayground.data.AccountLock
import com.minashin1120.aiplayground.data.Attachment
import com.minashin1120.aiplayground.data.BanAppealState
import com.minashin1120.aiplayground.data.BatchJob
import com.minashin1120.aiplayground.data.COMPOSER_SELECT_DEFAULTS
import com.minashin1120.aiplayground.data.ChatMessage
import com.minashin1120.aiplayground.data.CodingTarget
import com.minashin1120.aiplayground.data.CompressionSettings
import com.minashin1120.aiplayground.data.ConnectionStatus
import com.minashin1120.aiplayground.data.CredentialRequest
import com.minashin1120.aiplayground.data.FeedbackItem
import com.minashin1120.aiplayground.data.HistoryCacheMode
import com.minashin1120.aiplayground.data.LibraryFile
import com.minashin1120.aiplayground.data.LiveAnswer
import com.minashin1120.aiplayground.data.LyriaState
import com.minashin1120.aiplayground.data.McpDecision
import com.minashin1120.aiplayground.data.McpServerInfo
import com.minashin1120.aiplayground.data.ModelInfo
import com.minashin1120.aiplayground.data.OfflineCacheStats
import com.minashin1120.aiplayground.data.Preferences
import com.minashin1120.aiplayground.data.RealtimeState
import com.minashin1120.aiplayground.data.SecurityInfo
import com.minashin1120.aiplayground.data.ServerInfo
import com.minashin1120.aiplayground.data.ServerOrigin
import com.minashin1120.aiplayground.data.SettingsBubble
import com.minashin1120.aiplayground.data.Skill
import com.minashin1120.aiplayground.data.StatusCard
import com.minashin1120.aiplayground.data.StorageUsage
import com.minashin1120.aiplayground.data.ThreadItem
import com.minashin1120.aiplayground.data.originLabel
import org.json.JSONObject

enum class ChatTransitionKind { NONE, OPEN_THREAD, NEW_CHAT }

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
    val branchOrigin: JSONObject? = null, val branchBusy: Boolean = false,
    val messages: List<ChatMessage> = emptyList(),
    val hasOlder: Boolean = false, val oldestId: String? = null,
    val allMessages: List<ChatMessage> = emptyList(), val leafId: Int? = null, val editingMessageId: String? = null,
    val customInstruction: String = "", val includeGlobalInstruction: Boolean = true,
    val newThreadTemporary: Boolean = false, val tempChatRemainingSeconds: Long? = null,
    /** The open chat's temporary-chat timeout (`timeout_seconds`); null falls back to the setting. */
    val tempChatTimeoutSeconds: Int? = null,
    /** One-shot UI identity; never serialized or synchronized. */
    val promptSendMotionId: String? = null,
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
    val enableSkills: Boolean = true,
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
    val skills: List<Skill> = emptyList(), val skillsBusy: Boolean = false, val selectedSkill: Skill? = null,
    val preferences: Preferences? = null, val prefsBusy: Boolean = false,
    val compression: CompressionSettings = CompressionSettings(),
    val mcpServers: List<McpServerInfo> = emptyList(), val mcpBusy: Boolean = false,
    val feedbackItems: List<FeedbackItem> = emptyList(), val feedbackBusy: Boolean = false,
    /** The server answered that this account is an administrator: [feedbackItems] then holds every feedback. */
    val feedbackAdmin: Boolean = false,
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
    val banAppeal: BanAppealState = BanAppealState(),
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
    /** Id of the message [quote] was selected from (Web `currentQuoteMessageId`), sent as `quote_message_id`. */
    val quoteMessageId: Int? = null,
    /** Local or server message key used only to navigate back to a quote. */
    val quoteSourceId: String? = null,
    val quoteNavigationRequest: Long = 0L,
    /** Incremented to ask the screen to open the settings modal (e.g. from the encryption status dialog). */
    val settingsRequest: Long = 0L,
    /** Tab (id or label) and card key the latest [settingsRequest] opens at; null keeps the last tab. */
    val settingsRequestTab: String? = null,
    val settingsRequestCard: String? = null,
    /** Incremented to move the focus into the prompt input (Web `input.focus()` after edit / quote). */
    val composerFocusRequest: Long = 0L,
    /** Incremented after server keys are imported to the device, so an open settings form reloads its key fields. */
    val importedSecretsRevision: Long = 0L,
    /** Text shared from another app shown as a preview bubble. */
    val sharedTextPreview: String? = null,
    /** PDF export in progress. */
    val pdfExporting: Boolean = false,
)
