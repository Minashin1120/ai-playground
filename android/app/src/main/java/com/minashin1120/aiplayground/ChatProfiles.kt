package com.minashin1120.aiplayground

import android.content.SharedPreferences
import com.minashin1120.aiplayground.data.Account
import com.minashin1120.aiplayground.data.ChatNavigationHistory
import com.minashin1120.aiplayground.data.ConnectionStatus
import com.minashin1120.aiplayground.data.OfflineCacheStore
import com.minashin1120.aiplayground.data.PROVIDER_KEY_FIELDS
import com.minashin1120.aiplayground.data.PlaygroundApi
import com.minashin1120.aiplayground.data.ServerInfo
import com.minashin1120.aiplayground.data.ServerOrigin
import com.minashin1120.aiplayground.data.backend.ChatBackend
import com.minashin1120.aiplayground.data.direct.DirectHttp
import com.minashin1120.aiplayground.data.direct.DirectRouter
import com.minashin1120.aiplayground.data.local.GenerationKeepAlive
import com.minashin1120.aiplayground.data.local.LocalChatBackend
import com.minashin1120.aiplayground.data.local.LocalChatStore
import com.minashin1120.aiplayground.data.local.LocalProfiles
import com.minashin1120.aiplayground.data.local.LocalSettingsStore
import com.minashin1120.aiplayground.data.local.ServerHistory
import com.minashin1120.aiplayground.data.originLabel
import com.minashin1120.aiplayground.data.parseServerInfo
import com.minashin1120.aiplayground.data.parseServerOrigin
import com.minashin1120.aiplayground.data.sync.SyncEngine
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import org.json.JSONObject
import java.io.IOException

/**
 * 端末プロフィールの切替（アカウントなし／サーバー不使用モード）、オフライン時の可否判定、
 * 接続先（サーバー）の保存と選択だけを持つ協作者。
 *
 * - チャットを端末に持つ間の [localChats] / [localSettings] / [localBackend] を所有する。
 * - 状態の更新は `mutable` へ行い、画面遷移・履歴の再読込・アカウント読込は生成時に受け取った関数へ委ねる。
 */
internal class ChatProfiles(
    private val state: StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val scope: CoroutineScope,
    private val prefs: SharedPreferences,
    private val localProfiles: LocalProfiles,
    private val api: PlaygroundApi,
    private val serverBackend: ChatBackend,
    private val offlineCache: OfflineCacheStore,
    private val attachmentFiles: ChatAttachmentFiles,
    private val syncRunner: ChatSyncRunner,
    private val syncApi: () -> PlaygroundApi,
    private val directHttp: () -> DirectHttp,
    private val generations: ChatGenerations,
    private val deviceAnswer: ChatDeviceAnswer,
    private val pendingSend: ChatPendingState,
    private val chatHistory: ChatNavigationHistory,
    private val token: () -> String,
    private val keepAlive: () -> GenerationKeepAlive,
    /** Stops the answers of the active profile (`generations.cancelAll().joinAll()`). */
    private val stopChatWork: suspend () -> Unit,
    private val cancelNavigation: () -> Unit,
    private val cancelThreadList: () -> Unit,
    private val loadAccount: suspend () -> Unit,
    private val newChat: () -> Unit,
    private val fetchThreads: suspend (Boolean) -> Unit,
    private val fetchPreferences: suspend () -> Unit,
    private val clearIntegrityTicket: () -> Unit,
    private val notify: (String) -> Unit,
    private val report: suspend (Throwable) -> Unit,
) {
    /** Device store of the active profile while chats are local (no-account profile or serverless mode). */
    var localChats: LocalChatStore? = null
        private set

    /** Settings and API keys of the device profile while chats are local (voice input uses its OpenAI key). */
    var localSettings: LocalSettingsStore? = null
        private set

    /** Set while a no-account profile or serverless mode answers chat endpoints on the device. */
    var localBackend: ChatBackend? = null
        private set

    /** No-account profile: every chat is on the device, so offline checks for chat operations do not apply. */
    val chatLocal: Boolean get() = localChats != null && state.value.localProfile

    /** Answers and attachments stay on the device (no-account profile or serverless mode), so they work offline. */
    val uploadsLocal: Boolean get() = localChats != null

    /** A chat kept on the device (delegates the device-mode rule to [ChatDeviceAnswer]). */
    fun deviceChat(id: String?): Boolean = deviceAnswer.isDeviceChat(id, chatLocal, state.value.serverless)

    /** Notifies and returns true when sending or creating a thread cannot run offline. */
    fun notifySendOffline(offline: Boolean = state.value.offline): Boolean {
        val notice = deviceAnswer.offlineSendNotice(offline) ?: return false
        notify(notice)
        return true
    }

    /** Offline and this chat is not kept on the device, so a server operation cannot run. */
    fun isChatOfflineBlocked(id: String?): Boolean = state.value.offline && !deviceChat(id)

    /** Delete a message: a pending local message in serverless mode can be removed offline (not synced to server). */
    fun isDeleteMessageOfflineBlocked(threadId: String?, messageId: Long): Boolean =
        isChatOfflineBlocked(threadId) && !(state.value.serverless && LocalChatStore.isPendingId(messageId))

    /** Notifies and returns true when deleting [messageId] cannot run offline (pending local messages in serverless mode are allowed). */
    fun notifyDeleteMessageOffline(threadId: String?, messageId: Long, notice: String): Boolean {
        if (!isDeleteMessageOfflineBlocked(threadId, messageId)) return false
        notify(notice); return true
    }

    /** The thread list cannot be fetched from the server while offline, except in the local (no-account) profile. */
    fun isThreadListOfflineBlocked(): Boolean = state.value.offline && !chatLocal

    /** Notifies [notice] and returns true when this thread's server operation cannot run offline. */
    fun notifyChatOffline(id: String?, notice: String): Boolean {
        if (!isChatOfflineBlocked(id)) return false
        notify(notice); return true
    }

    /** Notifies [notice] and returns true when offline (an operation that always needs the server cannot run). */
    fun notifyOffline(notice: String): Boolean {
        if (!state.value.offline) return false
        notify(notice); return true
    }

    /** Opens the device store of [profile]; [fallback] is the server for a signed-in account in serverless mode. */
    private fun useLocalChats(profile: LocalProfiles.Profile, accountName: String, fallback: ChatBackend?) {
        val defaults = localProfiles.defaults()
        localChats = profile.chats
        localSettings = profile.settings
        val remote = if (fallback == null) null else ServerHistory(fallback, { token() },
            serverFile = { reference, limit -> api.loadFileBytes(reference, token(), thumbnail = false, limit = limit) },
            cachedThread = { id -> state.value.account?.let { account -> offlineCache.loadThread(account.id, id) } },
            cachedFile = { reference, limit -> attachmentFiles.cachedFileBytes(reference, limit) },
            isOffline = { state.value.offline },
            afterAnswer = { syncRunner.afterAnswer() })
        localBackend = LocalChatBackend(profile.chats, profile.settings, defaults, DirectRouter(directHttp()), accountName, fallback,
            modeOf = { id -> state.value.account?.models?.firstOrNull { it.id == id }?.mode
                ?: defaults.json.optJSONArray("models")?.let { rows -> (0 until rows.length()).mapNotNull { rows.optJSONObject(it) }
                    .firstOrNull { it.optString("id") == id }?.optString("mode") } ?: "chat" },
            onChanged = { if (fallback != null) syncRunner.refreshPendingCount() },
            remote = remote,
            keepAlive = keepAlive(),
            uploadAttachment = { reference -> SyncEngine(profile.chats, syncApi()) { token() }.uploadAttachment(reference) })
    }

    fun closeLocalChats() {
        syncRunner.cancel()
        localChats = null; localBackend = null; localSettings = null
    }

    /**
     * Serverless mode: a device chat that went up to the server is followed under its server id (the open
     * chat is switched to it). Any other id is returned as it is.
     */
    suspend fun serverIdOf(id: String): String {
        val store = localChats?.takeIf { state.value.serverless && LocalChatStore.isDeviceThreadId(id) } ?: return id
        val serverId = withContext(Dispatchers.IO) { store.resolveAlias(id) }
        if (serverId != id) generations.rebind(id, serverId)
        if (serverId != id) mutable.update { current ->
            if (current.selected?.id == id) current.copy(selected = current.selected?.copy(id = serverId)) else current
        }
        return serverId
    }

    /** Settings "今すぐ同期". */
    fun syncNow() { syncRunner.syncNow() }

    /** Settings "サーバーのAPIキーを端末へ取り込む": the account's own keys (after re-authentication) go to the device store. */
    fun importServerSecrets(secrets: JSONObject) {
        val account = state.value.account ?: return
        val settings = localProfiles.account(account.id).settings
        scope.launch {
            val count = withContext(Dispatchers.IO) {
                settings.importSecrets(secrets, overwrite = true)
                PROVIDER_KEY_FIELDS.count { secrets.optString(it).isNotBlank() } + (secrets.optJSONObject("model_api_keys")?.length() ?: 0)
            }
            notify(if (count > 0) "APIキーを${count}件、端末に取り込みました" else "サーバーに保存されたAPIキーはありません")
            runCatching { fetchPreferences() }
            mutable.update { it.copy(importedSecretsRevision = it.importedSecretsRevision + 1) }
        }
    }

    /** Settings: copies the no-account profile's chats into this account (uploaded by the next sync). */
    fun importLocalProfileChats() {
        val account = state.value.account ?: return
        scope.launch {
            val count = withContext(Dispatchers.IO) { localProfiles.account(account.id).chats.importFrom(localProfiles.local().chats) }
            notify(if (count > 0) "端末のチャットを${count}件取り込みました" else "取り込むチャットはありません")
            runCatching { fetchThreads(false) }
            syncRunner.refreshPendingCount()
            syncRunner.schedule()
        }
    }

    /** Login screen "サーバーを使わずに始める": a device-only profile, no network to the Playground server. */
    fun startLocalProfile() {
        if (state.value.authBusy) return
        scope.launch {
            prefs.edit().putString(ChatViewModel.PREF_PROFILE_MODE, ChatViewModel.PROFILE_LOCAL).apply()
            enterLocalProfile()
        }
    }

    suspend fun enterLocalProfile() {
        chatHistory.clear()
        useLocalChats(localProfiles.local(), ChatViewModel.LOCAL_PROFILE_NAME, fallback = null)
        mutable.update { it.copy(localProfile = true, serverless = false, canGoBackInChats = false,
            offline = false, connectionBannerVisible = false,
            connectionStatus = ConnectionStatus.ONLINE, authError = null) }
        runCatching { loadAccount() }.onFailure { report(it) }
    }

    /** Leaves the no-account profile for the login screen; the device chats stay for a later upload. */
    fun leaveLocalProfile() { scope.launch {
        stopChatWork(); cancelNavigation(); cancelThreadList()
        pendingSend.clearAllFailed()
        chatHistory.clear()
        prefs.edit().putString(ChatViewModel.PREF_PROFILE_MODE, ChatViewModel.PROFILE_SERVER).apply()
        closeLocalChats()
        val server = state.value
        pendingSend.clearEdit()
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
        if (state.value.localProfile) return
        if (generations.active) { notify("回答生成中です。完了までお待ちいただくか、停止してください。"); return }
        scope.launch {
            prefs.edit().putBoolean(serverlessPrefKey(account.id), enabled).apply()
            applyServerlessMode(account)
            newChat()
            runCatching { fetchThreads(false) }.onFailure { report(it) }
            notify(if (enabled) "サーバー不使用モードをオンにしました" else "サーバー不使用モードをオフにしました")
        }
    }

    private fun serverlessPrefKey(accountId: Int) = LocalProfiles.serverlessPrefKey(accountId)

    /** Applies the saved serverless choice of [account] (called after the account loads). */
    fun applyServerlessMode(account: Account) {
        if (state.value.localProfile) return
        val enabled = prefs.getBoolean(serverlessPrefKey(account.id), false)
        if (enabled) useLocalChats(localProfiles.account(account.id), account.name, fallback = serverBackend) else closeLocalChats()
        val key = LocalProfiles.accountKey(account.id)
        mutable.update { it.copy(serverless = enabled, pendingCount = 0,
            lastSyncAt = prefs.getLong("last_sync_$key", 0L).takeIf { value -> value > 0 }, syncMessage = null,
            localImportAvailable = enabled && localProfiles.hasLocalData()) }
        if (enabled) {
            // Stores of 1.38.0–1.40.x held a copy of every chat; only what the server lacks is kept.
            localChats?.let { store -> scope.launch {
                withContext(Dispatchers.IO) { runCatching { store.repairSyncIdentities(); store.migrateToOutbox() } }
                syncRunner.refreshPendingCount()
                syncRunner.schedule()
            } }
        }
    }

    /** Points the app at the server saved with the session (or chosen last on the login screen). */
    fun restoreServerOrigin(saved: String?) {
        saved?.toHttpUrlOrNull()?.let { parseServerOrigin(it.toString()) }?.let(ServerOrigin::set)
        mutable.update { it.copy(serverLabel = originLabel(ServerOrigin.current), savedServers = savedServers()) }
    }

    private fun savedServers(): List<String> =
        prefs.getString(ChatViewModel.PREF_SAVED_SERVERS, null).orEmpty().split('\n').filter { it.isNotBlank() }

    fun applyServerInfo(info: ServerInfo) {
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
        scope.launch {
            mutable.update { it.copy(serverChecking = true, serverError = null) }
            try {
                val config = PlaygroundApi(origin).get("/api/mobile/v1/config")
                val info = parseServerInfo(origin, config) ?: throw IOException("AI Playground のサーバーとして確認できませんでした。")
                ServerOrigin.set(origin)
                val recent = (listOf(originLabel(origin)) + savedServers()).distinct().take(5)
                prefs.edit().putString(ChatViewModel.PREF_SERVER_ORIGIN, origin.toString())
                    .putString(ChatViewModel.PREF_SAVED_SERVERS, recent.joinToString("\n")).apply()
                clearIntegrityTicket()
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
}
