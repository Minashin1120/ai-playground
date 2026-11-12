package com.minashin1120.aiplayground

import com.minashin1120.aiplayground.data.OfflineCacheStore
import com.minashin1120.aiplayground.data.backend.ChatBackend
import com.minashin1120.aiplayground.data.local.LocalChatStore
import com.minashin1120.aiplayground.data.local.ServerHistory
import com.minashin1120.aiplayground.data.parseThreadItem
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.net.URLEncoder

/**
 * The thread list and its search (Web `loadThreads` and the search box): the 300ms debounce, paging,
 * the offline cache and the device chats of serverless mode. State lives in the view model; this
 * class only reads it and reports through `mutable`.
 */
internal class ChatThreadList(
    private val state: StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val scope: CoroutineScope,
    private val backend: () -> ChatBackend,
    private val token: () -> String,
    private val store: () -> LocalChatStore?,
    private val offlineCache: OfflineCacheStore,
    private val isOfflineBlocked: () -> Boolean,
    private val isChatLocal: () -> Boolean,
    private val refreshCacheStats: suspend (Int) -> Unit,
    private val report: suspend (Throwable) -> Unit,
) {
    private var listJob: Job? = null
    private var fetchGeneration = 0L

    /** The running list request (BAN・ログアウトで止める対象). */
    val activeJob: Job? get() = listJob

    fun search(query: String) {
        fetchGeneration++
        mutable.update { it.copy(search = query, nextPage = null) }
        listJob?.cancel()
        listJob = scope.launch {
            delay(300)
            try {
                if (isOfflineBlocked()) applyCached(query) else fetch(more = false)
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) { report(e) }
        }
    }

    /** Web `loadThreads`; [more] appends the next page ("さらに読み込む"). */
    suspend fun fetch(more: Boolean) {
        val generation = ++fetchGeneration
        if (isOfflineBlocked()) {
            applyCached(state.value.search)
            return
        }
        val current = state.value
        val page = if (more) current.nextPage ?: return else 1
        val reply = try {
            backend().get("/api/threads?page=$page&q=${URLEncoder.encode(current.search, "UTF-8")}", token())
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            // Serverless mode keeps working without the server: the cached list and the device chats stand in.
            if (!current.serverless || more || !ServerHistory.unreachable(e)) throw e
            applyCached(current.search)
            return
        }
        if (generation != fetchGeneration || current.search != state.value.search) return
        val rows = reply.getJSONArray("threads")
        val items = (0 until rows.length()).map { parseThreadItem(rows.getJSONObject(it)) }
        if (!isChatLocal()) state.value.account?.let { account ->
            withContext(Dispatchers.IO) { offlineCache.saveThreads(account.id, items.filterNot { LocalChatStore.isDeviceThreadId(it.id) }) }
            refreshCacheStats(account.id)
        }
        if (generation != fetchGeneration || current.search != state.value.search) return
        mutable.update { it.copy(threads = if (more) (it.threads + items).distinctBy { t -> t.id } else items,
            nextPage = if (reply.optBoolean("has_next")) reply.optInt("next_page", page + 1) else null,
            offline = false) }
    }

    private suspend fun applyCached(query: String) {
        val accountId = state.value.account?.id ?: return
        val cached = withContext(Dispatchers.IO) { offlineCache.loadThreads(accountId) }
        val normalized = query.trim()
        val visible = if (normalized.isBlank()) cached else cached.filter {
            it.title.contains(normalized, ignoreCase = true) || it.model.contains(normalized, ignoreCase = true)
        }
        // Serverless mode: chats created on the device while the server was out of reach come first.
        val device = store()?.takeIf { state.value.serverless }?.let { chatStore ->
            val rows = withContext(Dispatchers.IO) { chatStore.deviceThreads(normalized) }
            (0 until rows.length()).map { parseThreadItem(rows.getJSONObject(it)) }
        }.orEmpty()
        if (query != state.value.search || accountId != state.value.account?.id) return
        mutable.update { it.copy(threads = device + visible, nextPage = null, offline = true) }
    }

    fun more() {
        if (isOfflineBlocked() || state.value.nextPage == null || listJob?.isActive == true) return
        listJob = scope.launch {
            try { fetch(more = true) }
            catch (e: CancellationException) { throw e }
            catch (e: Exception) { report(e) }
        }
    }
}
