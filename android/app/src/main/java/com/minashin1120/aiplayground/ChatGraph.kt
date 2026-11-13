package com.minashin1120.aiplayground

import android.app.Application
import android.content.Context
import android.content.SharedPreferences
import android.net.ConnectivityManager
import com.minashin1120.aiplayground.data.*
import com.minashin1120.aiplayground.data.backend.ChatBackend
import com.minashin1120.aiplayground.data.backend.ServerChatBackend
import com.minashin1120.aiplayground.data.direct.DirectHttp
import com.minashin1120.aiplayground.data.local.LocalProfiles
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update

/**
 * チャットの協作者（受信・送信・添付・同期・音声・設定など）の組み立てと配線。
 *
 * 状態・保存先・ジョブは [ChatViewModel] 側に残し、ここは「どの協作者に何を渡すか」だけを持つ。
 * 画面が呼ぶ公開関数は [ChatViewModel] の薄い入口だけにし、このクラスは内部専用。
 */
internal class ChatGraph(
    private val host: ChatViewModel,
    private val application: Application,
    private val viewModelScope: CoroutineScope,
) {
    internal val lowBandwidth: ChatLowBandwidth = ChatLowBandwidth(
        connectivity = connectivity,
        state = state,
        mutable = mutable,
        prefs = prefs,
        notify = { message -> notify(message) },
    )
    /** サーバー到達性（ネットワーク監視・ハートビート・接続バナー）。 */
    internal val connection: ChatConnectionMonitor = ChatConnectionMonitor(
        application = application,
        state = state,
        mutable = mutable,
        scope = viewModelScope,
        backend = { backend },
        foreground = { foreground.isForeground },
        retryFeedback = { feedback.retryPending() },
        scheduleSync = { syncRunner.schedule() },
        client = { api },
        syncClient = { syncApi },
        discardIdle = {
            diagnostics.discardIdleConnections()
            feedback.discardIdleConnections()
        },
        lowBandwidthChanged = { lowBandwidth.recompute(state.value.lowBandwidthPreference == "auto") },
    )
    /** 管理者向け診断の送信と端末の活動ログ ([ChatDiagnostics])。 */
    internal val diagnostics: ChatDiagnostics = ChatDiagnostics(
        scope = viewModelScope,
        state = state,
        token = { session?.token },
        foreground = { foreground.isForeground },
        observe = { client -> observeServerResponses(client) },
    )
    /** 初回設定のアカウントZIP取り込み ([ChatAccountImport])。 */
    internal val accountImport: ChatAccountImport = ChatAccountImport(
        context = application,
        api = api,
        state = state,
        mutable = mutable,
        scope = viewModelScope,
        token = { token() },
    )
    /** 設定のデータタブ（エクスポート・取り込み・重複修復）([ChatAccountTransfer])。 */
    internal val accountTransferOps: ChatAccountTransfer = ChatAccountTransfer(
        scope = viewModelScope,
        api = api,
        token = { token() },
        application = application,
        notify = ::notify,
        reloadThreads = { done -> reloadThreads(done) },
        reloadSkills = { done -> reloadSkills(done) },
        loadStorageUsage = { loadStorageUsage() },
        loadPreferences = { loadPreferences() },
    )
    /** ログイン・2段階認証・パスキー・セキュリティ設定 ([ChatAuth])。 */
    internal val auth: ChatAuth = ChatAuth(
        api = api,
        backend = { backend },
        state = state,
        mutable = mutable,
        scope = viewModelScope,
        store = store,
        playIntegrity = playIntegrity,
        prefs = prefs,
        token = { token() },
        saveSession = { session = it },
        notify = { message -> notify(message) },
        loadSetup = { loadSetup() },
        loadAccount = { loadAccount() },
    )


    /** 回答の状態更新（`updateAnswer` / `updateLive` / `markApiAccepted`）。各協作者はこれを受け取る。 */
    internal val answers: ChatAnswerState = ChatAnswerState(generations, mutable)
    internal val streamJob: Job? get() = generations.current(state.value)?.job
    /** 端末モードの未送信分をサーバーへ送る同期（予約・再試行・今すぐ同期）。 */
    internal val syncRunner: ChatSyncRunner = ChatSyncRunner(
        state = state,
        mutable = mutable,
        scope = viewModelScope,
        store = { profiles.localChats },
        session = { session },
        token = { token() },
        syncApi = { syncApi },
        hasUsableNetwork = { hasUsableNetwork() },
        generations = generations,
        prefs = prefs,
        refreshThreads = { fetchThreads(false) },
        reloadMessages = { id -> loadMessages(id, autoResume = false, inPlace = true) },
        startSessionTurnstile = { startSessionTurnstile() },
        report = { e -> report(e) },
    )
    /** 添付の読み込み・プレビュー・保存（端末内ファイル／サーバー／オフラインキャッシュ）。 */
    internal val attachmentFiles: ChatAttachmentFiles = ChatAttachmentFiles(
        state = state,
        store = { profiles.localChats },
        cache = offlineCache,
        api = api,
        sessionToken = { session?.token },
        token = { token() },
        cacheDir = application.cacheDir,
        refreshCacheStats = { id -> refreshOfflineCacheStats(id) },
    )
    /** ファイルライブラリ（一覧・検索・並び順・お気に入り・名前変更・削除・添付・保存）。 */
    internal val library: ChatLibrary = ChatLibrary(
        state = state,
        mutable = mutable,
        scope = viewModelScope,
        backend = { backend },
        token = { token() },
        prefs = prefs,
        offlineCache = offlineCache,
        attachmentFiles = attachmentFiles,
        notify = { notify(it) },
        notifyOffline = { message -> notifyOffline(message) },
        refreshCacheStats = { id -> refreshOfflineCacheStats(id) },
    )
    /** フィードバック（一覧・状態変更・返信・削除・送信と端末側の再送）。 */
    internal val feedback: ChatFeedback = ChatFeedback(
        app = application,
        state = state,
        mutable = mutable,
        scope = viewModelScope,
        api = api,
        backend = { backend },
        token = { token() },
        sessionToken = { session?.token },
        localChats = { profiles.localChats },
        offlineCache = offlineCache,
        serverIdOf = { id -> serverIdOf(id) },
        notify = { notify(it) },
        notifyOffline = { message -> notifyOffline(message) },
        report = { e, message -> report(e, message) },
        observeResponses = { client -> observeServerResponses(client) },
    )
    /** 添付アップロード本体（キュー・転送・進捗）。 */
    internal val attachmentUploader: ChatAttachmentUploader = ChatAttachmentUploader(
        scope = viewModelScope,
        state = state,
        mutable = mutable,
        context = application,
        api = api,
        token = { token() },
        uploadsLocal = { uploadsLocal },
        localBackend = { profiles.localBackend },
        localChats = { profiles.localChats },
        syncApi = { syncApi },
        report = { e, msg -> report(e, msg) },
    )
    /** 画像分割 (ANDROID_ONLY.md): ダイアログ状態と分割・添付・保存の流れ。 */
    internal val imageSplitManager: ChatImageSplitManager = ChatImageSplitManager(
        scope = viewModelScope,
        app = application,
        isUploading = { state.value.uploading },
        upload = { uris, skip -> attachmentUploader.upload(uris, skip) },
        notify = ::notify,
    )
    /** チャット入力の音声録音と文字起こし。 */
    internal val micInput: ChatMicRecorder = ChatMicRecorder(
        scope = viewModelScope,
        state = state,
        mutable = mutable,
        app = application,
        api = { api },
        token = { token() },
        localSettings = { profiles.localSettings },
        directHttp = { directHttp },
        notify = ::notify,
    )
    /** 端末モードの回答（送信の端末モード判定・停止）。 */
    internal val deviceAnswer: ChatDeviceAnswer = ChatDeviceAnswer(
        scope = viewModelScope,
        state = state,
        mutable = mutable,
        generations = generations,
        syncRunner = syncRunner,
        updateAnswer = answers::update,
        loadMessages = { id, inPlace, streamFinished -> loadMessages(id, inPlace = inPlace, streamFinished = streamFinished) },
        report = { e -> viewModelScope.launch { report(e) } },
        uploadsLocal = { uploadsLocal },
    )
    /** 回答ストリームの受信（イベント適用・再接続）。送信や添付とは別ファイルで読めるよう切り出している。 */
    internal val streamReceiver: ChatStreamReceiver = ChatStreamReceiver(
        isAlive = { generations.contains(it) },
        updateAnswer = answers::update,
        updateLive = answers::updateLive,
        markAccepted = answers::markAccepted,
        markReachable = { markConnectionReachable() },
        waitingStatusFor = ::streamWaitingStatusFor,
    )
    /** 回答の送信と再接続（受け付け待ちの再試行ループ）。送信全体とは別ファイルで読めるよう切り出している。 */
    internal val sendStream: ChatSendStream = ChatSendStream(
        receiver = streamReceiver,
        isAlive = { generations.contains(it) },
        waitingStatusFor = ::streamWaitingStatusFor,
        updateAnswer = answers::update,
        updateLive = answers::updateLive,
        markReachable = { markConnectionReachable() },
        markAccepted = answers::markAccepted,
        notify = { notify(it) },
    )
    /** 送信・編集の途中状態（受け付けられなかった送信、編集中の親、編集前の入力設定）。 */
    internal val pendingSend: ChatPendingState = ChatPendingState()
    /** 送信後の後処理（履歴の再読込・タイトル生成・Batch 通知・受け付けられなかった送信の復元）。 */
    internal val sendCompletion: ChatSendCompletion = ChatSendCompletion(
        state = state,
        scope = viewModelScope,
        pending = pendingSend,
        matchesScreen = { it.matches(state.value) },
        updateAnswer = answers::update,
        vibrate = { botGuard.vibrate(100, 50, 100) },
        notify = { notify(it) },
        report = { report(it) },
        showApiKeyPrompt = { answer, message, model ->
            sendEntry.apiKeyMissingMessage = message
            answers.update(answer) { it.copy(apiKeyPrompt = model) }
        },
        startTurnstile = { startSessionTurnstile() },
        refreshThreads = { fetchThreads(false) },
        reloadMessages = { id -> loadMessages(id, inPlace = true, streamFinished = true) },
        refreshBatch = { fetchBatchJobs(notify = false); startBatchPolling() },
        notifyFinishedIfAway = { id, didFail -> foreground.notifyAnswerFinishedIfAway(id, didFail) },
        isDeviceChat = { deviceChat(it) },
        applyTitle = { answerToken, id, title ->
            if (session?.token == answerToken) mutable.update { current -> current.copy(selected = current.selected?.let {
                if (it.id == id) it.copy(title = title) else it
            }) }
        },
        canReloadHistory = { session != null || chatLocal },
    )
    /** 送信準備（入力・引用・モデル／オプション・添付名）。画面の状態は ViewModel に置いたまま切り出す。 */
    internal val composer: ChatComposer = ChatComposer(
        state = state,
        mutable = mutable,
        scope = viewModelScope,
        prefs = prefs,
        backend = { backend },
        token = { token() },
        notify = ::notify,
        report = ::report,
        notifyChatOffline = ::notifyChatOffline,
    )
    /** 最小モードのスラッシュコマンド（Web `executeMinimalSlashCommand`）。 */
    internal val slashCommands: ChatSlashCommands = ChatSlashCommands(
        state = state,
        composer = composer,
        notify = ::notify,
        saveCompressionSettings = { settings -> saveCompressionSettings(settings) },
        toggleTemporaryChat = ::toggleTemporaryChat,
    )
    /** スレッドのPDF出力（Web `openThreadPdfPrintDialog`）。 */
    internal val pdfExport: ChatPdfExport = ChatPdfExport(
        application = application,
        state = state,
        mutable = mutable,
        scope = viewModelScope,
        backend = { backend },
        token = { token() },
        notify = ::notify,
        notifyChatOffline = ::notifyChatOffline,
    )
    /** 全般設定（設定の保存読込・ストレージ使用量・MCP）とオフライン履歴キャッシュの設定。 */
    internal val preferences: ChatPreferences = ChatPreferences(
        state = state,
        mutable = mutable,
        scope = viewModelScope,
        backend = { backend },
        token = { token() },
        prefs = prefs,
        offlineCache = offlineCache,
        composer = composer,
        connectivity = connectivity,
        generations = generations,
        notify = ::notify,
        notifyOffline = ::notifyOffline,
        report = { e -> report(e) },
    )
    /** AI設定コマンド（`/settings <指示>`）の会話・モデル経由の設定変更・結果バブル。 */
    internal val aiSettings: ChatAiSettings = ChatAiSettings(
        state = state,
        mutable = mutable,
        scope = viewModelScope,
        backend = { backend },
        token = { token() },
        notify = ::notify,
        refreshPreferences = { preferences.fetchPreferences() },
    )
    /** アカウント読込・初回設定・オフライン復元。 */
    internal val accountSetup: ChatAccountSetup = ChatAccountSetup(
        application = application,
        state = state,
        mutable = mutable,
        scope = viewModelScope,
        backend = { backend },
        token = { token() },
        prefs = prefs,
        offlineCache = offlineCache,
        composer = composer,
        chatHistory = chatHistory,
        markConnectionReachable = { markConnectionReachable() },
        applyServerlessMode = { account -> profiles.applyServerlessMode(account) },
        fetchThreads = { more -> fetchThreads(more) },
        fetchSkills = { fetchSkills() },
        fetchBatchJobs = { notify -> fetchBatchJobs(notify) },
        loadMcpServers = ::loadMcpServers,
        fetchPreferences = { applyDefaults -> preferences.fetchPreferences(applyDefaults) },
        refreshCacheStats = { id -> refreshOfflineCacheStats(id) },
        foreground = { foreground.isForeground },
        startBatchPolling = { startBatchPolling() },
        startCacheSyncIfAllowed = { startCacheSyncIfAllowed() },
        report = { e -> report(e) },
    )
    /** スレッド一覧と検索（デバウンス・ページ送り・オフラインキャッシュ・端末モードの端末チャット）。 */
    internal val threadList: ChatThreadList = ChatThreadList(
        state = state,
        mutable = mutable,
        scope = viewModelScope,
        backend = { backend },
        token = { token() },
        store = { profiles.localChats },
        offlineCache = offlineCache,
        isOfflineBlocked = { isThreadListOfflineBlocked() },
        isChatLocal = { chatLocal },
        refreshCacheStats = { id -> refreshOfflineCacheStats(id) },
        report = { e -> report(e) },
    )
    /** Batch処理の履歴・完了バナー・状態ポーリング・停止と削除。 */
    internal val batch: ChatBatch = ChatBatch(
        app = application,
        state = state,
        mutable = mutable,
        scope = viewModelScope,
        backend = { backend },
        token = { token() },
        notify = { notify(it) },
        report = { e, message -> report(e, message) },
        isForeground = { foreground.isForeground },
        hasSession = { session != null },
        reloadInPlace = { id -> loadMessages(id, autoResume = false, inPlace = true) },
        reloadThread = { id -> loadMessages(id) },
        openThread = { id -> openThreadId(id) },
    )
    /** 数学ステップの進捗保存と「もっとやさしく」([ChatMathSteps])。 */
    internal val mathSteps: ChatMathSteps = ChatMathSteps(
        mutable = mutable,
        scope = viewModelScope,
        backend = { backend },
        token = { token() },
    )
    /** Skill の一覧・保存・削除と、このチャットに適用する Skill。 */
    internal val skills: ChatSkills = ChatSkills(
        state = state,
        mutable = mutable,
        scope = viewModelScope,
        backend = { backend },
        token = { token() },
        notify = { notify(it) },
        notifyOffline = { message -> notifyOffline(message) },
        notifyChatOffline = { id, message -> notifyChatOffline(id, message) },
        report = { e, message -> report(e, message) },
    )
    /** チャット履歴の読み込み（サーバー／オフラインキャッシュ・過去分・送信中を保つ再読込）。 */
    internal val messageLoader: ChatMessageLoader = ChatMessageLoader(
        state = state,
        mutable = mutable,
        backend = { backend },
        token = { token() },
        store = { profiles.localChats },
        offlineCache = offlineCache,
        generations = generations,
        serverIdOf = { id -> serverIdOf(id) },
        isChatOfflineBlocked = { id -> isChatOfflineBlocked(id) },
        deviceChat = { id -> deviceChat(id) },
        isChatLocal = { chatLocal },
        setConnectionUnavailable = { status -> setConnectionUnavailable(status) },
        refreshCacheStats = { id -> refreshOfflineCacheStats(id) },
        chooseModel = { model -> chooseModel(model) },
        manualModelChosen = { composer.modelChosenManually },
        latestPromptOptions = { current -> current.withLatestPromptOptions() },
        syncHeartbeat = { syncHeartbeat() },
        resume = { resume() },
    )

    /** 端末プロフィール・オフライン可否・接続先 ([ChatProfiles])。 */
    internal val profiles: ChatProfiles = ChatProfiles(
        state = state,
        mutable = mutable,
        scope = viewModelScope,
        prefs = prefs,
        localProfiles = localProfiles,
        api = api,
        serverBackend = serverBackend,
        offlineCache = offlineCache,
        attachmentFiles = attachmentFiles,
        syncRunner = syncRunner,
        syncApi = { syncApi },
        directHttp = { directHttp },
        generations = generations,
        deviceAnswer = deviceAnswer,
        pendingSend = pendingSend,
        chatHistory = chatHistory,
        token = { token() },
        keepAlive = { answerKeepAlive },
        stopChatWork = { generations.cancelAll().joinAll() },
        cancelNavigation = { navigationJob?.cancel() },
        cancelThreadList = { threadListJob?.cancel() },
        loadAccount = { loadAccount() },
        newChat = { navigator.newChat() },
        fetchThreads = { more -> fetchThreads(more) },
        fetchPreferences = { preferences.fetchPreferences() },
        clearIntegrityTicket = { auth.clearIntegrityTicket() },
        notify = { message -> notify(message) },
        report = { e -> report(e) },
    )
    /** 前面／背面の切替と一時チャットの期限更新 ([ChatForeground])。 */
    internal val foreground: ChatForeground = ChatForeground(
        application = application,
        state = state,
        mutable = mutable,
        scope = viewModelScope,
        generations = generations,
        lowBandwidth = lowBandwidth,
        syncRunner = syncRunner,
        backend = { backend },
        token = { token() },
        answeringOnDevice = { answeringOnDevice() },
        streamJob = { streamJob },
        refreshInPlace = { refreshInPlace() },
        discardIdleConnections = { discardIdleServerConnections() },
        stopRealtime = { stopRealtime(save = false) },
        stopLyria = { stopLyria(save = false) },
        cancelBatchPoll = { batchPollJob?.cancel() },
        startConnectionMonitor = { startConnectionMonitor() },
        stopConnectionMonitor = { stopConnectionMonitor() },
        startBatchPolling = { startBatchPolling() },
        startCacheSyncIfAllowed = { startCacheSyncIfAllowed() },
        report = { e -> report(e) },
    )
    /** 添付の入口（アップロード・中止・圧縮・ダウンロード・画像分割）([ChatAttachments])。 */
    internal val attachments: ChatAttachments = ChatAttachments(
        mutable = mutable,
        scope = viewModelScope,
        prefs = prefs,
        attachmentFiles = attachmentFiles,
        attachmentUploader = attachmentUploader,
        imageSplitManager = imageSplitManager,
        notify = { message -> notify(message) },
        report = { e -> report(e) },
    )
    /** Xリンクの自動検索とAPIキーの保存 ([ChatSendEntry])。 */
    internal val sendEntry: ChatSendEntry = ChatSendEntry(
        state = state,
        mutable = mutable,
        scope = viewModelScope,
        backend = { backend },
        token = { token() },
        composer = composer,
        notify = { message -> notify(message) },
        send = { send() },
    )
    /** プロンプトオプションの復元・編集・分岐 ([ChatEditing])。 */
    internal val editing: ChatEditing = ChatEditing(
        state = state,
        mutable = mutable,
        composer = composer,
        pendingSend = pendingSend,
        notify = { message -> notify(message) },
        send = { send() },
    )
    /** 画面のチャット移動（新規・戻る・スレッドを開く・再読込）。 */
    internal val navigator: ChatNavigator = ChatNavigator(
        state = state,
        mutable = mutable,
        scope = viewModelScope,
        chatHistory = chatHistory,
        pendingSend = pendingSend,
        composer = composer,
        generations = generations,
        nextTransition = { kind -> nextChatTransition(kind) },
        pinnedLeaf = { id ->
            application.getSharedPreferences("branches", Context.MODE_PRIVATE)
                .getString("fixed_branch_$id", null)?.toIntOrNull()
        },
        cancelHeartbeat = { foreground.heartbeat?.cancel() },
        withNewChatDefaults = { current -> current.withNewChatDefaults() },
        withModelRules = { current -> composer.applyModelRules(current) },
        withLatestPromptOptions = { current -> current.withLatestPromptOptions() },
        loadMessages = { id, older, inPlace -> loadMessages(id, older = older, inPlace = inPlace) },
        fetchThreads = { more -> fetchThreads(more) },
        resumeAnswer = { resume() },
        isForeground = { foreground.isForeground },
        report = { e, message -> report(e, message) },
    )
    /** スレッドとメッセージの削除・名前変更・分岐・設定の保存。 */
    internal val threadActions: ChatThreadActions = ChatThreadActions(
        state = state,
        mutable = mutable,
        scope = viewModelScope,
        backend = { backend },
        token = { token() },
        generations = generations,
        messageLoader = messageLoader,
        pendingSend = pendingSend,
        offlineCache = offlineCache,
        chatHistory = chatHistory,
        syncRunner = syncRunner,
        uploadsLocal = { uploadsLocal },
        notifyChatOffline = { id, message -> notifyChatOffline(id, message) },
        notifyDeleteMessageOffline = { id, messageId, notice -> notifyDeleteMessageOffline(id, messageId, notice) },
        notify = { notify(it) },
        accountApi = { accountApi() },
        fetchThreads = { more -> fetchThreads(more) },
        syncHeartbeat = { syncHeartbeat() },
        loadMessages = { id -> loadMessages(id) },
        openThread = { thread, record, focus -> navigator.openThread(thread, record, focus) },
        newChat = { temporary, record -> navigator.newChat(temporary, record) },
        applySavedPreferences = { current, saved -> current.withSavedPreferences(saved) },
        report = { e -> report(e) },
        reportWithMessage = { e, message -> report(e, message) },
        notifySendOffline = { notifySendOffline() },
    )
    /** Realtime voice session and the one-shot `/sts` recorder (Web `RealtimeVoiceSession`). */
    internal val realtimeSession: ChatRealtimeSession = ChatRealtimeSession(
        state = state,
        mutable = mutable,
        scope = viewModelScope,
        application = application,
        api = { api },
        backend = { backend },
        token = { token() },
        notify = { notify(it) },
        notifyOffline = { message -> notifyOffline(message) },
        ensureThread = { onReady -> ensureThread(onReady) },
        openThreadId = { id -> openThreadId(id) },
        loadMessages = { id -> loadMessages(id) },
    )
    /** Lyria RealTime studio (Web `LyriaRealtimeStudio`). */
    internal val lyriaStudio: ChatLyriaStudio = ChatLyriaStudio(
        state = state,
        mutable = mutable,
        scope = viewModelScope,
        api = { api },
        backend = { backend },
        token = { token() },
        notify = { notify(it) },
        openThreadId = { id -> openThreadId(id) },
    )
    /** BAN状態と異議申し立て ([ChatBanState])。 */
    internal val banState: ChatBanState = ChatBanState(
        state = state,
        mutable = mutable,
        scope = viewModelScope,
        backend = { backend },
        token = { token() },
        stopAccountWork = {
            generations.cancelAll()
            pendingSend.clearAllFailed()
            listOf(pairingJob, navigationJob, threadListJob, streamJob, attachmentUploader.activeJob, foreground.heartbeat, libraryJob, preferences.activeJob, batchPollJob,
                importJob).forEach { it?.cancel() }
            realtimeSession.release()
            lyriaStudio.release()
        },
        reloadAccount = { loadAccount() },
        report = { e -> report(e) },
    )
    /** 連投ロック・タップのテレメトリ・振動 ([ChatBotGuard])。 */
    internal val botGuard: ChatBotGuard = ChatBotGuard(
        app = application,
        state = state,
        mutable = mutable,
        scope = viewModelScope,
        api = api,
        token = { token() },
        hasSession = { session != null },
        enterBannedState = { error -> enterBannedState(error) },
        notify = { message -> notify(message) },
    )
    /** 送信の確認・送信本文の組み立て・送信ジョブの開始・再開・停止 ([ChatSubmissionRunner])。 */
    internal val sendRunner: ChatSubmissionRunner = ChatSubmissionRunner(
        api = api,
        state = state,
        mutable = mutable,
        scope = viewModelScope,
        backend = { backend },
        token = { token() },
        generations = generations,
        keepAlive = { answerKeepAlive },
        pending = pendingSend,
        messageLoader = messageLoader,
        streamReceiver = streamReceiver,
        sendStream = sendStream,
        sendCompletion = sendCompletion,
        deviceAnswer = deviceAnswer,
        waitingStatusFor = ::streamWaitingStatusFor,
        updateAnswer = answers::update,
        notify = { message -> notify(message) },
        report = { error -> report(error) },
        beginSend = { beginSend() },
        notifySendOffline = { offline -> notifySendOffline(offline) },
        applyXLinkSearch = { applyXLinkSearch() },
        syncHeartbeat = { syncHeartbeat() },
        setConnectionUnavailable = { status -> setConnectionUnavailable(status) },
        reloadAfterResume = { id -> loadMessages(id, autoResume = false, inPlace = true) },
        reloadAfterStop = { id -> loadMessages(id, inPlace = true, streamFinished = true) },
        notifyFinishedIfAway = { id, failed -> foreground.notifyAnswerFinishedIfAway(id, failed) },
    )
    /** ログアウト・401 の後始末・失敗の通知 ([ChatSessionLifecycle])。 */
    internal val lifecycle: ChatSessionLifecycle = ChatSessionLifecycle(
        application = application,
        state = state,
        mutable = mutable,
        scope = viewModelScope,
        backend = { backend },
        token = { token() },
        prefs = prefs,
        store = store,
        chatHistory = chatHistory,
        generations = generations,
        pendingSend = pendingSend,
        profiles = profiles,
        realtimeSession = realtimeSession,
        lyriaStudio = lyriaStudio,
        banState = banState,
        connection = connection,
        jobs = { listOf(pairingJob, navigationJob, threadListJob, streamJob, attachmentUploader.activeJob,
            foreground.heartbeat, libraryJob, preferences.activeJob, batchPollJob, importJob, mathSteps.activeJob) },
        saveSession = { session = it },
    )
    /** The library's running list request (BAN・ログアウトで止める対象)。 */
    internal val libraryJob: Job? get() = library.activeJob
    /** The thread list request (BAN・ログアウトで止める対象)。 */
    internal val threadListJob: Job? get() = threadList.activeJob
    /** The Batch status poll (BAN・ログアウト・バックグラウンドで止める対象)。 */
    internal val batchPollJob: Job? get() = batch.activeJob
    internal val importJob: Job? get() = accountImport.activeJob
    /** The running chat navigation ([ChatNavigator], BAN・ログアウトで止める対象)。 */
    internal val navigationJob: Job? get() = navigator.activeJob

    /** 起動時の復元（[ChatStartup]）。 */
    internal val startup: ChatStartup = ChatStartup(
        state = state,
        mutable = mutable,
        scope = viewModelScope,
        prefs = prefs,
        store = store,
        api = api,
        profiles = profiles,
        lowBandwidth = lowBandwidth,
        accountSetup = accountSetup,
        banState = banState,
        lifecycle = lifecycle,
        preferences = preferences,
        connection = connection,
        hasSession = { session != null },
        notify = { message -> notify(message) },
        reloadThread = { id -> loadMessages(id, inPlace = true) },
        saveSession = { session = it },
    )

    // region: Host helpers (ChatViewModel) — 組み立てから使う薄い入口
    private val api: PlaygroundApi get() = host.api
    private val serverBackend: ServerChatBackend get() = host.serverBackend
    private val backend: ChatBackend get() = host.backend
    private val localProfiles: LocalProfiles get() = host.localProfiles
    private val directHttp: DirectHttp get() = host.directHttp
    private val syncApi: PlaygroundApi get() = host.syncApi
    private val store: TokenStore get() = host.store
    private val playIntegrity: PlayIntegrityClient get() = host.playIntegrity
    private val offlineCache: OfflineCacheStore get() = host.offlineCache
    private val prefs: SharedPreferences get() = host.prefs
    private val connectivity: ConnectivityManager? get() = host.connectivity
    private val mutable: MutableStateFlow<ChatState> get() = host.mutable
    private val state: StateFlow<ChatState> get() = host.state
    private val chatHistory: ChatNavigationHistory get() = host.chatHistory
    private var session: StoredSession?
        get() = host.session
        set(value) { host.session = value }
    private val answerKeepAlive get() = host.answerKeepAlive
    private val pairingJob get() = host.pairingJob
    private val generations get() = host.generations

    private fun ChatState.withSavedPreferences(saved: Preferences): ChatState = composer.applySavedPreferences(this, saved)
    private fun ChatState.withLatestPromptOptions(): ChatState = editing.withLatestPromptOptions(this)

    private fun notify(message: String) = host.notify(message)
    private suspend fun report(error: Throwable, message: String? = null) = host.report(error, message)
    private fun token(): String = host.token()
    private fun deviceChat(id: String?): Boolean = host.deviceChat(id)
    private val chatLocal: Boolean get() = host.chatLocal
    private val uploadsLocal: Boolean get() = host.uploadsLocal
    private fun notifySendOffline(offline: Boolean = state.value.offline): Boolean = host.notifySendOffline(offline)
    private fun isChatOfflineBlocked(id: String?): Boolean = host.isChatOfflineBlocked(id)
    private fun notifyDeleteMessageOffline(threadId: String?, messageId: Int, notice: String): Boolean =
        host.notifyDeleteMessageOffline(threadId, messageId, notice)
    private fun isThreadListOfflineBlocked(): Boolean = host.isThreadListOfflineBlocked()
    private fun notifyChatOffline(id: String?, notice: String): Boolean = host.notifyChatOffline(id, notice)
    private fun notifyOffline(notice: String): Boolean = host.notifyOffline(notice)
    private suspend fun serverIdOf(id: String): String = host.serverIdOf(id)
    private suspend fun fetchThreads(more: Boolean) = host.fetchThreads(more)
    private suspend fun fetchSkills() = host.fetchSkills()
    private suspend fun fetchBatchJobs(notify: Boolean, status: Boolean = notify) = host.fetchBatchJobs(notify, status)
    private fun loadMcpServers() = host.loadMcpServers()
    private fun startBatchPolling() = host.startBatchPolling()
    private fun startCacheSyncIfAllowed() = host.startCacheSyncIfAllowed()
    private suspend fun refreshOfflineCacheStats(accountId: Int? = state.value.account?.id) =
        host.refreshOfflineCacheStats(accountId)
    private fun markConnectionReachable(
        wasDisconnected: Boolean = state.value.connectionStatus.isDisconnected(),
        resetSlowCount: Boolean = true,
    ) = host.markConnectionReachable(wasDisconnected, resetSlowCount)
    private fun observeServerResponses(client: PlaygroundApi) = host.observeServerResponses(client)
    private fun hasUsableNetwork(): Boolean = host.hasUsableNetwork()
    private fun setConnectionUnavailable(status: ConnectionStatus, message: String = status.defaultMessage()) =
        host.setConnectionUnavailable(status, message)
    private fun discardIdleServerConnections() = host.discardIdleServerConnections()
    private fun startConnectionMonitor() = host.startConnectionMonitor()
    private fun stopConnectionMonitor() = host.stopConnectionMonitor()
    private fun answeringOnDevice(): Boolean = host.answeringOnDevice()
    private fun refreshInPlace() = host.refreshInPlace()
    private fun syncHeartbeat() = host.syncHeartbeat()
    private suspend fun loadMessages(
        requestedId: String,
        older: Boolean = false,
        autoResume: Boolean = true,
        inPlace: Boolean = false,
        streamFinished: Boolean = false,
    ) = host.loadMessages(requestedId, older, autoResume, inPlace, streamFinished)
    private suspend fun loadSetup() = host.loadSetup()
    private suspend fun loadAccount() = host.loadAccount()
    private fun chooseModel(model: String) = host.chooseModel(model)
    private fun openThreadId(id: String) = host.openThreadId(id)
    private fun ensureThread(onReady: () -> Unit) = host.ensureThread(onReady)
    private fun send() = host.send()
    private fun resume() = host.resume()
    private fun beginSend() = host.beginSend()
    private fun applyXLinkSearch() = host.applyXLinkSearch()
    private fun accountApi(): AccountApi = host.accountApi()
    private fun stopRealtime(save: Boolean) = host.stopRealtime(save)
    private fun stopLyria(save: Boolean = false) = host.stopLyria(save)
    private fun startSessionTurnstile() = host.startSessionTurnstile()
    private fun reloadThreads(done: () -> Unit) = host.reloadThreads(done)
    private fun reloadSkills(done: () -> Unit) = host.reloadSkills(done)
    private fun saveCompressionSettings(settings: CompressionSettings) = host.saveCompressionSettings(settings)
    private fun toggleTemporaryChat() = host.toggleTemporaryChat()
    private fun loadStorageUsage() = host.loadStorageUsage()
    private fun loadPreferences() = host.loadPreferences()
    private fun nextChatTransition(kind: ChatTransitionKind): Pair<Long, ChatTransitionKind> = host.nextChatTransition(kind)
    private fun enterBannedState(error: ApiException) = host.enterBannedState(error)

    /** [ChatViewModel.onCleared]: 協作者とジョブを止める。 */
    internal fun release() {
        generations.cancelAll()
        micInput.cancel()
        connection.release()
    }
}
