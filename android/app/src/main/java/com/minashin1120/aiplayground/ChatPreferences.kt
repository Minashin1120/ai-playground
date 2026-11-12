package com.minashin1120.aiplayground

import android.content.SharedPreferences
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import com.minashin1120.aiplayground.data.*
import com.minashin1120.aiplayground.data.backend.ChatBackend
import com.minashin1120.aiplayground.data.local.OfflineCacheStore
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.net.URLEncoder

/**
 * General account settings and this device's session (Web settings modal): the preferences payload
 * (load/save, rich paste prompt), storage usage, MCP servers, and the offline history cache (mode,
 * mobile data, full sync, clear). State stays in the view model; this class only reads it and
 * reports through `mutable`.
 */
internal class ChatPreferences(
    private val state: StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val scope: CoroutineScope,
    private val backend: () -> ChatBackend,
    private val token: () -> String,
    private val prefs: SharedPreferences,
    private val offlineCache: OfflineCacheStore,
    private val composer: ChatComposer,
    private val connectivity: ConnectivityManager?,
    private val generations: ChatGenerations,
    private val notify: (String) -> Unit,
    private val notifyOffline: (String) -> Boolean,
    private val report: suspend (Throwable) -> Unit,
) {
    private var cacheSyncJob: Job? = null

    /** The running full-cache sync (cancelled at the BAN and logout boundaries). */
    val activeJob: Job? get() = cacheSyncJob

    suspend fun fetchPreferences(applyDefaults: Boolean = false) {
        val transitionId = state.value.chatTransitionId
        val accountId = state.value.account?.id
        val payload = backend().get("/api/mobile/v1/preferences", token())
        if (state.value.account?.id != accountId) return
        val preferences = parsePreferences(payload)
        state.value.account?.let { account ->
            withContext(Dispatchers.IO) { offlineCache.savePreferences(account.id, payload) }
        }
        if (!applyDefaults || state.value.chatTransitionId != transitionId || state.value.selected != null || state.value.streaming) {
            mutable.update { it.copy(preferences = preferences) }
            return
        }
        mutable.update { it.copy(preferences = preferences)
            .withNewChatDefaults(preserveSelectedModel = composer.modelChosenManually).let(composer::applyModelRules) }
    }

    fun loadPreferences() {
        scope.launch {
            mutable.update { it.copy(prefsBusy = true) }
            try {
                val accountId = state.value.account?.id
                if (state.value.offline && accountId != null) {
                    withContext(Dispatchers.IO) { offlineCache.loadPreferences(accountId) }
                        ?.let { payload -> mutable.update { it.copy(preferences = parsePreferences(payload)) } }
                } else if (notifyOffline("オフライン中は設定を読み込めません。")) {
                    // nothing
                } else fetchPreferences()
            } catch (e: Exception) { report(e) } finally { mutable.update { it.copy(prefsBusy = false) } }
        }
    }

    /** Web `saveRichPastePromptPreferences`: saved quietly while the rich paste prompt is edited. */
    fun saveRichPastePrompt(prompt: String, useCustomDefault: Boolean) {
        scope.launch {
            if (notifyOffline("オフライン中は設定を保存できません。")) return@launch
            runCatching {
                val reply = backend().put("/api/mobile/v1/preferences", JSONObject().put("rich_paste_prompt_default", prompt)
                    .put("rich_paste_prompt_use_custom_default", useCustomDefault), token())
                mutable.update { it.copy(preferences = parsePreferences(reply)) }
            }
        }
    }

    fun savePreferences(payload: JSONObject, message: String = "設定を保存しました") {
        scope.launch {
            if (notifyOffline("オフライン中はアカウント設定を保存できません。接続後に再試行してください。")) return@launch
            mutable.update { it.copy(prefsBusy = true) }
            try {
                val reply = backend().put("/api/mobile/v1/preferences", payload, token())
                mutable.update { it.withSaved(parsePreferences(reply)).copy(notice = message) }
            } catch (e: Exception) { report(e) }
            finally { mutable.update { it.copy(prefsBusy = false) } }
        }
    }

    private fun ChatState.withSaved(saved: Preferences): ChatState = composer.applySavedPreferences(this, saved)

    fun loadStorageUsage() {
        scope.launch {
            if (notifyOffline("オフライン中はストレージ使用量を取得できません。")) return@launch
            runCatching { mutable.update { it.copy(storage = parseStorageUsage(backend().get("/api/storage", token()))) } }
                .onFailure { report(it) }
        }
    }

    fun loadMcpServers() {
        scope.launch {
            if (state.value.offline) return@launch
            mutable.update { it.copy(mcpBusy = true) }
            try { mutable.update { it.copy(mcpServers = parseMcpServers(backend().get("/api/mcp/servers", token()))) } }
            catch (e: Exception) { report(e) }
            finally { mutable.update { it.copy(mcpBusy = false) } }
        }
    }

    fun setMcpServerEnabled(server: McpServerInfo, enabled: Boolean) {
        scope.launch {
            mutable.update { it.copy(mcpBusy = true) }
            try {
                backend().put("/api/mcp/servers/${server.id}", JSONObject().put("enabled", enabled), token())
                loadMcpServers()
            } catch (e: Exception) { report(e) }
            finally { mutable.update { it.copy(mcpBusy = false) } }
        }
    }

    /** Web `resolveMcpDecision`: answers the MCP approval bubble of the running answer. */
    fun resolveMcpDecision(allow: Boolean) { scope.launch {
        if (notifyOffline("オフライン中はMCPの確認に応答できません。")) return@launch
        val decision = state.value.mcpDecision ?: return@launch
        if (decision.jobId.isBlank()) return@launch
        generations.current(state.value)?.let { answer -> generations.update(answer, mutable) { it.copy(mcpDecision = null) } }
            ?: mutable.update { it.copy(mcpDecision = null) }
        try {
            backend().post("/api/mcp/chat/${URLEncoder.encode(decision.jobId, "UTF-8")}/decision",
                JSONObject().put("decision", if (allow) "allow" else "deny").put("id", decision.id), token())
        } catch (e: Exception) { report(e) }
    } }

    suspend fun refreshOfflineCacheStats(accountId: Int? = state.value.account?.id) {
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
        scope.launch {
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
        cacheSyncJob = scope.launch {
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

    fun startCacheSyncIfAllowed() {
        if (state.value.historyCacheMode == HistoryCacheMode.FULL && cacheNetworkAllowed()) syncOfflineCache()
    }

    private fun cacheNetworkAllowed(): Boolean {
        val manager = connectivity ?: return false
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
            val reply = backend().get("/api/threads?page=$page&q=", token())
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
                val reply = backend().get("/api/threads/${thread.id}?limit=200$suffix", token())
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
            val reply = backend().get("/api/files?limit=40&offset=$offset&sort=newest&q=")
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
}
