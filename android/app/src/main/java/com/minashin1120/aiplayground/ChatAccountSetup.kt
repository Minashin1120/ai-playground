package com.minashin1120.aiplayground

import android.app.Application
import android.content.SharedPreferences
import com.minashin1120.aiplayground.data.Account
import com.minashin1120.aiplayground.data.ConnectionStatus
import com.minashin1120.aiplayground.data.HistoryCacheMode
import com.minashin1120.aiplayground.data.ModelInfo
import com.minashin1120.aiplayground.data.OfflineCacheStore
import com.minashin1120.aiplayground.data.applyWebModelCatalog
import com.minashin1120.aiplayground.data.backend.ChatBackend
import com.minashin1120.aiplayground.data.parseModels
import com.minashin1120.aiplayground.data.parsePreferences
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject

/**
 * アカウント読込と初回設定（Web `loadAccount` / `loadSetup` / `finishSetup` / オフライン復元）だけを持つ：
 * `/api/mobile/v1/me` と `/setup` の取得・保存、表示用モデル一覧の合成、最後のアカウントの復元。
 * 画面状態は `mutable` へ書き、スレッド・Skill・Batch・設定の読込は生成時に受け取った関数へ委ねる。
 */
internal class ChatAccountSetup(
    private val application: Application,
    private val state: StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val scope: CoroutineScope,
    private val backend: () -> ChatBackend,
    private val token: () -> String,
    private val prefs: SharedPreferences,
    private val offlineCache: OfflineCacheStore,
    private val composer: ChatComposer,
    private val chatHistory: ChatNavigationHistory,
    private val markConnectionReachable: () -> Unit,
    private val applyServerlessMode: (Account) -> Unit,
    private val fetchThreads: suspend (Boolean) -> Unit,
    private val fetchSkills: suspend () -> Unit,
    private val fetchBatchJobs: suspend (Boolean) -> Unit,
    private val loadMcpServers: () -> Unit,
    private val fetchPreferences: suspend (Boolean) -> Unit,
    private val refreshCacheStats: suspend (Int?) -> Unit,
    private val foreground: () -> Boolean,
    private val startBatchPolling: () -> Unit,
    private val startCacheSyncIfAllowed: () -> Unit,
    private val report: suspend (Throwable) -> Unit,
) {
    suspend fun loadSetup() {
        chatHistory.clear()
        val reply = backend().get("/api/mobile/v1/setup", token())
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
        scope.launch {
            mutable.update { it.copy(authBusy = true, authError = null) }
            try {
                val reply = backend().put("/api/mobile/v1/setup", JSONObject()
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

    suspend fun loadAccount() {
        val me = backend().get("/api/mobile/v1/me", token())
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
            account = account, model = if (it.account?.id == account.id && account.models.any { m -> m.id == it.model && m.selectable }) it.model else chosen, pairing = false, userCode = "", offline = false,
            setupRequired = false, authBusy = false, authError = null,
            canGoBackInChats = chatHistory.canGoBack,
        ) }
        fetchThreads(false)
        runCatching { fetchSkills() }.onFailure { report(it) }
        runCatching {
            fetchPreferences(state.value.selected == null && !state.value.streaming && !composer.modelChosenManually)
        }.onFailure { report(it) }
        runCatching { fetchBatchJobs(false) }.onFailure { if (it is CancellationException) throw it }
        // The composer shows the MCP chip only when a server is enabled (Web `applyMcpPromptChipUi`).
        loadMcpServers()
        if (state.value.localProfile) return
        if (foreground()) startBatchPolling()
        if (state.value.historyCacheMode == HistoryCacheMode.FULL) startCacheSyncIfAllowed()
    }

    private suspend fun displayModels(serverModels: List<ModelInfo>): List<ModelInfo> = runCatching {
        val json = withContext(Dispatchers.IO) {
            application.assets.open("web-model-catalog.json").bufferedReader().use { it.readText() }
        }
        applyWebModelCatalog(serverModels, json)
    }.getOrDefault(serverModels)

    /**
     * Restores the last account and locally stored data when the server cannot be reached.
     * [asOffline] false shows the same data as an online account (startup, before the server answers).
     */
    suspend fun restoreOfflineAccount(asOffline: Boolean = true): Boolean {
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
        if (!asOffline) applyServerlessMode(account)
        mutable.update { current -> current.copy(
            account = account,
            canGoBackInChats = chatHistory.canGoBack,
            model = selected,
            preferences = cachedPrefs?.let { parsePreferences(it) } ?: current.preferences,
            threads = cachedThreads,
            nextPage = null,
            pairing = false,
        ).let { restored ->
            // Apply cached defaults before exposing the startup composer, including while offline.
            if (current.account == null && !composer.modelChosenManually && restored.selected == null && !restored.streaming)
                composer.applyModelRules(restored.withNewChatDefaults()) else restored
        }.let { restored ->
            if (!asOffline) restored else restored.copy(
                offline = true,
                connectionStatus = ConnectionStatus.OFFLINE,
                connectionMessage = ConnectionStatus.OFFLINE.defaultMessage(),
                connectionBannerVisible = true,
            )
        } }
        refreshCacheStats(account.id)
        return true
    }
}
