package com.minashin1120.aiplayground

import com.minashin1120.aiplayground.data.*
import com.minashin1120.aiplayground.data.backend.ChatBackend
import com.minashin1120.aiplayground.data.local.LocalChatStore
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException

/**
 * Loading a chat's history (Web `loadMessages`): the server reply or the offline cache, older
 * pages, the in-place refresh, and the stopped-job suppression that keeps a stopped answer from
 * being rejoined. State lives in the view model; this class only reads it and reports through
 * `mutable`.
 */
internal class ChatMessageLoader(
    private val state: StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val backend: () -> ChatBackend,
    private val token: () -> String,
    private val store: () -> LocalChatStore?,
    private val offlineCache: OfflineCacheStore,
    private val generations: ChatGenerations,
    private val serverIdOf: suspend (String) -> String,
    private val isChatOfflineBlocked: (String?) -> Boolean,
    private val deviceChat: (String?) -> Boolean,
    private val isChatLocal: () -> Boolean,
    private val setConnectionUnavailable: (ConnectionStatus) -> Unit,
    private val refreshCacheStats: suspend (Int) -> Unit,
    private val chooseModel: (String) -> Unit,
    private val manualModelChosen: () -> Boolean,
    private val latestPromptOptions: (ChatState) -> ChatState,
    private val syncHeartbeat: () -> Unit,
    private val resume: () -> Unit,
) {
    private var generation = 0L
    /** Web `suppressedPendingJobIds`: jobs the user stopped are not rejoined while the server winds them down. */
    private val suppressedJobIds = mutableSetOf<String>()

    /** Whether the last loaded chat still had a stopped job running on the server. */
    var stoppedJobStillRunning = false
        private set

    /** A send invalidates an in-flight history request (Web bumps the fetch generation). */
    fun invalidate() {
        ++generation
    }

    /** Web `stopGeneration`: the job is never rejoined, even while the server still lists it as running. */
    fun suppress(jobId: String) {
        suppressedJobIds.add(jobId)
    }

    /** An in-place refresh of a server chat while offline in non-serverless mode must not disrupt the view. */
    private fun shouldSkipInPlaceMessageRefresh(id: String): Boolean =
        isChatOfflineBlocked(id) && !state.value.serverless

    /**
     * [inPlace]: the open chat is updated without replacing what is on screen. Messages loaded with
     * 過去メッセージを読み込む stay, the chosen model and Skill are kept, and a failed request or a send
     * that started meanwhile leaves the chat as it is.
     */
    suspend fun load(requestedId: String, older: Boolean = false, autoResume: Boolean = true, inPlace: Boolean = false, streamFinished: Boolean = false) {
        if (state.value.selected?.id != requestedId) return
        if (inPlace && !streamFinished && (state.value.streaming || state.value.busy)) return
        val fetchGeneration = if (older) generation else ++generation
        val id = serverIdOf(requestedId)
        if (inPlace && shouldSkipInPlaceMessageRefresh(id)) return
        if (isChatOfflineBlocked(id)) {
            loadCachedMessages(id, older, fetchGeneration = fetchGeneration, streamFinished = streamFinished)
            return
        }
        val before = if (older) "&before_id=${state.value.oldestId ?: return}" else
            state.value.leafId?.takeIf { state.value.allMessages.isEmpty() }?.let { "&before_id=${it.toLong() + 1}" }.orEmpty()
        val low = state.value.lowBandwidthMode
        val limit = when {
            older && low -> LOW_BANDWIDTH_OLDER_PAGE_SIZE
            older -> THREAD_OLDER_PAGE_SIZE
            low -> LOW_BANDWIDTH_INITIAL_MESSAGE_LIMIT
            else -> THREAD_INITIAL_MESSAGE_LIMIT
        }
        val reply = try {
            Diagnostics.log("thread.fetch", "thread_id" to id, "older" to older)
            backend().get("/api/threads/$id?limit=$limit$before", token())
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            val canUseCache = if (e is ApiException) e.status >= 500 else e is IOException
            if (!canUseCache) throw e
            if (inPlace) return
            val accountId = state.value.account?.id ?: throw e
            val cached = withContext(Dispatchers.IO) { offlineCache.loadThread(accountId, id) } ?: throw e
            setConnectionUnavailable(
                if (e is ApiException) ConnectionStatus.SERVER_DOWN else ConnectionStatus.OFFLINE,
            )
            loadCachedMessages(id, older, cached, fetchGeneration, streamFinished = streamFinished)
            return
        }
        Diagnostics.log("thread.fetched", "thread_id" to id, "rows" to (reply.optJSONArray("messages")?.length() ?: 0),
            "pending_job" to !reply.isNull("pending_job"))
        if (state.value.selected?.id != id || fetchGeneration != generation) return
        if (inPlace && !streamFinished && (state.value.streaming || state.value.busy)) return
        val pendingJobId = reply.optJSONObject("pending_job")?.nullableString("job_id")?.ifBlank { null }
        val pendingJobStopped = pendingJobId != null && pendingJobId in suppressedJobIds
        if (!older) stoppedJobStillRunning = pendingJobStopped
        // Sync changes pending ids to server ids without changing the displayed conversation.
        // Use the exact store mapping so even repeated questions and replies retain their row identity.
        val previousRows = state.value.allMessages
        val conversationKeys = if (inPlace) withContext(Dispatchers.IO) {
            previousRows.associate { message ->
                val resolved = numericId(message)?.let { store()?.resolveMessageId(id, it) ?: it }?.toString() ?: message.id
                resolved to (message.conversationKey ?: message.id)
            }
        } else emptyMap()
        val parsed = generations.history(reply, id, finished = streamFinished).map { message ->
            conversationKeys[message.id]?.let { message.copy(conversationKey = it) } ?: message
        }
        val parsedOldest = parsed.mapNotNull { numericId(it) }.filterNot { LocalChatStore.isPendingId(it) }.minOrNull()
        val earlier = if (inPlace && parsedOldest != null)
            state.value.allMessages.filter { m -> numericId(m)?.let { n -> n < parsedOldest } == true } else emptyList()
        val all = if (older) (parsed + state.value.allMessages).distinctBy { m -> m.id } else earlier + parsed
        val selectedLeaf = state.value.leafId?.let { candidate ->
            withContext(Dispatchers.IO) { store()?.resolveMessageId(id, candidate) ?: candidate }
        }
        val leaf = selectedLeaf?.takeIf { candidate -> all.any { numericId(it) == candidate } }
            ?: all.mapNotNull { numericId(it) }.maxOrNull()
        val path = activeBranchPath(all, leaf)
        val keepsEarlier = earlier.isNotEmpty()
        if (state.value.selected?.id != id || fetchGeneration != generation ||
            (inPlace && !streamFinished && (state.value.streaming || state.value.busy))) return
        mutable.update { it.copy(messages = path, allMessages = all, leafId = leaf,
            hasOlder = if (keepsEarlier) it.hasOlder else reply.optBoolean("has_older_messages"),
            oldestId = if (keepsEarlier) it.oldestId else reply.nullableString("oldest_loaded_id"),
            jobId = if (older) it.jobId else pendingJobId.takeUnless { pendingJobStopped },
            selected = it.selected?.let { selected -> selected.copy(
                title = reply.optString("title", selected.title),
                model = reply.nullableString("last_model").ifBlank { selected.model },
                isTemporary = reply.optBoolean("is_temporary", selected.isTemporary),
            ) },
            branchOrigin = if (older) it.branchOrigin else reply.optJSONObject("branch_origin"),
            customInstruction = if (older) it.customInstruction else reply.nullableString("custom_instruction"),
            includeGlobalInstruction = if (older) it.includeGlobalInstruction else reply.optBoolean("include_global_instruction", true),
            newThreadTemporary = if (older) it.newThreadTemporary else false,
            tempChatTimeoutSeconds = if (older) it.tempChatTimeoutSeconds else
                reply.optInt("timeout_seconds").takeIf { value -> reply.has("timeout_seconds") && value > 0 },
            tempChatRemainingSeconds = if (older) it.tempChatRemainingSeconds else
                reply.optLong("temp_chat_remaining_seconds").takeIf { value -> !reply.isNull("temp_chat_remaining_seconds") && value >= 0 },
            liveContent = if (older) it.liveContent else "", liveThought = if (older) it.liveThought else "",
            cards = if (older) it.cards else emptyList(),
            selectedSkill = if (older || inPlace) it.selectedSkill else {
                val uuid = reply.nullableString("last_skill_uuid").ifBlank { reply.nullableString("last_gem_uuid") }
                if (uuid.isBlank()) null else it.skills.firstOrNull { skill -> skill.uuid == uuid }
            }) }
        if (!older && !inPlace) {
            // Web `loadMessages`: the chat's PromptCache flag and last model come back with it (the flag
            // first, so a PromptCache lock from the previous chat never blocks the switch).
            mutable.update { it.copy(enablePromptCache = reply.optBoolean("enable_prompt_caching")) }
            if (!inPlace && !manualModelChosen()) reply.nullableString("last_model").takeIf { it.isNotBlank() && it != state.value.model }?.let(chooseModel)
            if (!inPlace && autoResume) mutable.update { latestPromptOptions(it) }
        }
        if (!deviceChat(id)) state.value.account?.let { account ->
            withContext(Dispatchers.IO) {
                offlineCache.saveThread(
                    // Unsent answers of serverless mode are not cached: they are shown from the outbox.
                    account.id, state.value.selected ?: return@withContext,
                    all.filterNot { m -> state.value.serverless &&
                        (numericId(m)?.let { LocalChatStore.isPendingId(it) } == true || m.files.any { LocalChatStore.isLocalReference(it) }) },
                    if (keepsEarlier) state.value.hasOlder else reply.optBoolean("has_older_messages"),
                    (if (keepsEarlier) state.value.oldestId else reply.nullableString("oldest_loaded_id"))?.ifBlank { null },
                    reply.nullableString("custom_instruction"), reply.optBoolean("include_global_instruction", true),
                    reply.optLong("temp_chat_remaining_seconds").takeIf { value -> !reply.isNull("temp_chat_remaining_seconds") && value >= 0 },
                    leaf,
                )
            }
            refreshCacheStats(account.id)
        }
        if (!older && !isChatLocal()) syncHeartbeat()
        generations.restore(mutable, id, finished = streamFinished)
        // Web `loadMessages`: an answer still running on the server is rejoined automatically.
        if (!older && autoResume && state.value.jobId != null && !state.value.streaming) resume()
    }

    private suspend fun loadCachedMessages(id: String, older: Boolean, cachedPayload: JSONObject? = null, fetchGeneration: Long, streamFinished: Boolean = false) {
        val accountId = state.value.account?.id ?: return
        val outbox = store()?.takeIf { state.value.serverless }
        val stored = cachedPayload ?: withContext(Dispatchers.IO) { offlineCache.loadThread(accountId, id) }
        // Serverless mode: answers not yet on the server are shown on top of the cached chat.
        val cached = if (outbox == null) stored ?: return else withContext(Dispatchers.IO) {
            val base = stored?.let { JSONObject(it.toString()) }
                ?: JSONObject().put("messages", JSONArray()).takeIf { outbox.outboxIdFor(id) != null }
            base?.let { outbox.overlayPending(id, it, observedOnServer = false, encrypted = state.value.account?.encrypted) }
        } ?: return
        val parsed = generations.history(cached, id, finished = streamFinished)
        val all = if (older) (parsed + state.value.allMessages).distinctBy { it.id } else parsed
        val selectedLeaf = state.value.leafId?.let { candidate ->
            withContext(Dispatchers.IO) { store()?.resolveMessageId(id, candidate) ?: candidate }
        }
        val leaf = selectedLeaf?.takeIf { candidate -> all.any { numericId(it) == candidate } }
            ?: all.mapNotNull { numericId(it) }.filter { outbox != null && LocalChatStore.isPendingId(it) }.maxOrNull()
            ?: cached.optInt("leaf_id").takeIf { it > 0 }
            ?: all.mapNotNull { numericId(it) }.maxOrNull()
        val path = activeBranchPath(all, leaf)
        if (state.value.selected?.id != id || fetchGeneration != generation) return
        mutable.update { it.copy(
            messages = path,
            allMessages = all,
            leafId = leaf,
            hasOlder = false,
            oldestId = cached.nullableString("oldest_loaded_id").ifBlank { null },
            jobId = null,
            branchOrigin = cached.optJSONObject("branch_origin"),
            customInstruction = cached.nullableString("custom_instruction"),
            includeGlobalInstruction = cached.optBoolean("include_global_instruction", true),
            tempChatRemainingSeconds = cached.optLong("temp_chat_remaining_seconds")
                .takeIf { value -> !cached.isNull("temp_chat_remaining_seconds") && value >= 0 },
            offline = true,
        ) }
        if (!older) mutable.update { latestPromptOptions(it) }
        generations.restore(mutable, id, finished = streamFinished)
    }
}
