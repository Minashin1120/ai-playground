package com.minashin1120.aiplayground

import android.app.Application
import android.content.SharedPreferences
import com.minashin1120.aiplayground.data.ApiException
import com.minashin1120.aiplayground.data.ChatNavigationHistory
import com.minashin1120.aiplayground.data.ConnectionStatus
import com.minashin1120.aiplayground.data.GoogleAuthClient
import com.minashin1120.aiplayground.data.HistoryCacheMode
import com.minashin1120.aiplayground.data.StoredSession
import com.minashin1120.aiplayground.data.TokenStore
import com.minashin1120.aiplayground.data.backend.ChatBackend
import com.minashin1120.aiplayground.data.connectionStatusForHttp
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.joinAll
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject

/**
 * ログアウト・401 の後始末・失敗の通知だけを持つ（Web の `logout` / `apiFetch` の失敗処理）。
 * セッションの保存先・実行中のジョブ・音声セッション・端末バブルは生成時に受け取った
 * 関数と協作者へ委ね、画面の状態は `mutable` へ書く。
 */
internal class ChatSessionLifecycle(
    private val application: Application,
    private val state: StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val scope: CoroutineScope,
    private val backend: () -> ChatBackend,
    private val token: () -> String,
    private val prefs: SharedPreferences,
    private val store: TokenStore,
    private val chatHistory: ChatNavigationHistory,
    private val generations: ChatGenerations,
    private val pendingSend: ChatPendingState,
    private val profiles: ChatProfiles,
    private val realtimeSession: ChatRealtimeSession,
    private val lyriaStudio: ChatLyriaStudio,
    private val banState: ChatBanState,
    private val connection: ChatConnectionMonitor,
    /** この ViewModel が持つジョブ（`pairingJob` / `streamJob` と各協作者の `activeJob`）。 */
    private val jobs: () -> List<Job?>,
    private val saveSession: (StoredSession?) -> Unit,
) {
    fun logout() { scope.launch {
        if (state.value.localProfile) { profiles.leaveLocalProfile(); return@launch }
        try {
            backend().post("/api/mobile/v1/revoke", JSONObject(), token())
            clearSession()
        } catch (e: Exception) {
            // Keep the Web BAN screen's logout escape hatch even if revoke fails.
            if (state.value.banned) clearSession() else report(e)
        }
    } }

    suspend fun clearSession() {
        chatHistory.clear()
        val caller = currentCoroutineContext().job
        generations.cancelAll(except = caller).filter { it !== caller }.joinAll()
        pendingSend.clearAllFailed()
        pendingSend.clearEdit()
        jobs().forEach { if (it !== caller) it?.cancel() }
        realtimeSession.release(except = caller)
        lyriaStudio.release(except = caller)
        withContext(NonCancellable + Dispatchers.IO) { store.clear() }
        profiles.closeLocalChats()
        GoogleAuthClient.clearCredentialState(application)
        cancelChatBubble(application)
        saveSession(null)
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
    suspend fun report(error: Throwable, message: String? = null) {
        if (error is CancellationException) throw error
        if (error is ApiException && error.code == "banned") {
            banState.enter(error)
            return
        }
        if (error is ApiException && error.status == 401 && !state.value.localProfile) clearSession()
        val networkFailure = error !is ApiException && (error is java.net.ConnectException || error is java.net.UnknownHostException ||
            error is java.net.SocketTimeoutException || error is java.net.SocketException || error is java.io.IOException
        )
        when {
            error is ApiException && connectionStatusForHttp(error.status) != null ->
                connection.setUnavailable(connectionStatusForHttp(error.status)!!)
            error is ApiException && error.status >= 500 ->
                connection.setUnavailable(ConnectionStatus.UNSTABLE)
            networkFailure ->
                connection.setUnavailable(if (connection.hasUsableNetwork()) ConnectionStatus.UNSTABLE else ConnectionStatus.OFFLINE)
        }
        mutable.update { it.copy(
            notice = message ?: if (networkFailure) "通信が途切れました。接続を確認して再試行してください。"
                else error.message?.take(500) ?: "通信に失敗しました。再試行してください。",
        ) }
    }
}
