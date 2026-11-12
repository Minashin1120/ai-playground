package com.minashin1120.aiplayground

import android.content.SharedPreferences
import com.minashin1120.aiplayground.data.ApiException
import com.minashin1120.aiplayground.data.ConnectionStatus
import com.minashin1120.aiplayground.data.HistoryCacheMode
import com.minashin1120.aiplayground.data.LOW_BANDWIDTH_PREF_KEY
import com.minashin1120.aiplayground.data.PlaygroundApi
import com.minashin1120.aiplayground.data.ServerOrigin
import com.minashin1120.aiplayground.data.StoredSession
import com.minashin1120.aiplayground.data.TokenStore
import com.minashin1120.aiplayground.data.compressionSettingsFrom
import com.minashin1120.aiplayground.data.normalizeLowBandwidthPreference
import com.minashin1120.aiplayground.data.parseServerInfo
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * 起動時の復元と再接続だけを持つ（Web の初期化と接続バナーの「再接続」）。
 * 端末内のプロフィール・保存済みアカウント・サーバー情報を読み、失敗は
 * `ChatSessionLifecycle` の通知・後始末へ委ねる。画面の状態は `mutable` へ書く。
 */
internal class ChatStartup(
    private val state: StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val scope: CoroutineScope,
    private val prefs: SharedPreferences,
    private val store: TokenStore,
    private val api: PlaygroundApi,
    private val profiles: ChatProfiles,
    private val lowBandwidth: ChatLowBandwidth,
    private val accountSetup: ChatAccountSetup,
    private val banState: ChatBanState,
    private val lifecycle: ChatSessionLifecycle,
    private val preferences: ChatPreferences,
    private val connection: ChatConnectionMonitor,
    private val hasSession: () -> Boolean,
    private val notify: (String) -> Unit,
    private val reloadThread: suspend (String) -> Unit,
    private val saveSession: (StoredSession?) -> Unit,
) {
    /** The `init` block of the view model: restore the profile, the saved session and the server config. */
    fun start() = scope.launch {
        mutable.update { it.copy(compression = compressionSettingsFrom(prefs),
            lowBandwidthPreference = normalizeLowBandwidthPreference(prefs.getString(LOW_BANDWIDTH_PREF_KEY, "auto"))) }
        lowBandwidth.recompute(false)
        mutable.update { it.copy(
            historyCacheMode = HistoryCacheMode.from(prefs.getString("offline_history_cache_mode", HistoryCacheMode.VIEWED.value)),
            cacheMobileDataAllowed = prefs.getBoolean("offline_cache_mobile_data", false),
        ) }
        profiles.restoreServerOrigin(withContext(Dispatchers.IO) { store.loadForOffline() }?.origin
            ?: prefs.getString(ChatViewModel.PREF_SERVER_ORIGIN, null))
        if (prefs.getString(ChatViewModel.PREF_PROFILE_MODE, ChatViewModel.PROFILE_SERVER) == ChatViewModel.PROFILE_LOCAL) {
            // The no-account profile never contacts the Playground server.
            profiles.enterLocalProfile()
            mutable.update { it.copy(starting = false) }
            return@launch
        }
        val session: StoredSession? = withContext(Dispatchers.IO) { store.load() }
        saveSession(session)
        // A signed-in account is shown from the device cache at once; the server refreshes it in the background,
        // so the startup spinner never waits for the sequential requests below.
        if (session != null && accountSetup.restoreOfflineAccount(asOffline = false)) mutable.update { it.copy(starting = false) }
        runCatching { api.get("/api/mobile/v1/config") }.getOrNull()?.let { config ->
            val info = parseServerInfo(ServerOrigin.current, config)
            if (info != null) profiles.applyServerInfo(info) else mutable.update { it.copy(
                googleServerClientId = config.optString("google_server_client_id"),
                integrityProjectNumber = config.optString("play_integrity_cloud_project_number"),
            ) }
        }
        if (session != null) {
            runCatching {
                accountSetup.loadAccount()
            }.onFailure { error ->
                if (error is ApiException && error.code == "banned") banState.enter(error)
                else if (error is ApiException && error.code == "setup_required") accountSetup.loadSetup()
                else if (error is ApiException && error.status == 401) lifecycle.clearSession()
                if (error !is ApiException || error.code != "banned") {
                    if (!accountSetup.restoreOfflineAccount()) lifecycle.report(error)
                }
            }
        } else if (withContext(Dispatchers.IO) { store.loadForOffline() } != null) {
            accountSetup.restoreOfflineAccount()
        }
        preferences.refreshOfflineCacheStats()
        mutable.update { it.copy(starting = false) }
    }

    /** Clears the offline banner and retries the last account or history load. */
    fun reconnect() {
        connection.cancelRecoveredHide()
        mutable.update { it.copy(offline = false, connectionStatus = ConnectionStatus.UNKNOWN, connectionMessage = "", connectionBannerVisible = false) }
        scope.launch {
            try {
                if (!hasSession()) notify("再接続するにはログインしてください。") else {
                    accountSetup.loadAccount()
                    state.value.selected?.let { reloadThread(it.id) }
                }
                connection.probe()
            } catch (e: Exception) { lifecycle.report(e) }
        }
    }
}
