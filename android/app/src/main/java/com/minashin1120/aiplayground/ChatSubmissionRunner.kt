package com.minashin1120.aiplayground

import android.os.SystemClock
import com.minashin1120.aiplayground.data.*
import com.minashin1120.aiplayground.data.backend.ChatBackend
import com.minashin1120.aiplayground.data.local.GenerationKeepAlive
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import org.json.JSONObject

/**
 * Web `sendMessage` の送信・再開・停止だけを持つ：送信前の確認と送信本文の組み立て、
 * 回答ジョブの開始（新規スレッド作成を含む）、サーバーで継続中の回答への再結合、
 * 停止要求（端末生成の回答は要求の取り消し）。状態は持たず、画面の更新や読み込みは
 * 生成時に受け取った関数へ委ねる（`ChatViewModel` の送信入口とは別に読めるようにする）。
 */
internal class ChatSubmissionRunner(
    private val api: PlaygroundApi,
    private val state: StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val scope: CoroutineScope,
    private val backend: () -> ChatBackend,
    private val token: () -> String,
    private val generations: ChatGenerations,
    private val keepAlive: () -> GenerationKeepAlive,
    private val pending: ChatPendingState,
    private val messageLoader: ChatMessageLoader,
    private val streamReceiver: ChatStreamReceiver,
    private val sendStream: ChatSendStream,
    private val sendCompletion: ChatSendCompletion,
    private val deviceAnswer: ChatDeviceAnswer,
    private val waitingStatusFor: (Throwable) -> String?,
    private val updateAnswer: (ChatGenerations.Answer, (ChatState) -> ChatState) -> Unit,
    private val notify: (String) -> Unit,
    private val report: suspend (Throwable) -> Unit,
    private val beginSend: () -> Boolean,
    private val notifySendOffline: (Boolean) -> Boolean,
    private val applyXLinkSearch: () -> Unit,
    private val syncHeartbeat: () -> Unit,
    private val setConnectionUnavailable: (ConnectionStatus) -> Unit,
    private val reloadAfterResume: suspend (String) -> Unit,
    private val reloadAfterStop: suspend (String) -> Unit,
    private val notifyFinishedIfAway: (String, Boolean) -> Unit,
) {
    /**
     * [xLinkChecked] is set once the X-link question was answered; [disableAutoSearch] tells the server the
     * user chose to answer without search (Web `disable_auto_search`).
     */
    fun send(xLinkChecked: Boolean = false, disableAutoSearch: Boolean = false, preChecked: Boolean = false) {
        if (!preChecked && !beginSend()) return
        var current = state.value
        val removeBackground = isIdeogramBackgroundRemoval(current.model, current.generationValues)
        val removeObject = isIdeogramObjectRemoval(current.model, current.generationValues)
        if (removeBackground || removeObject) {
            current = current.copy(draft = ideogramPromptForSend(current.model, current.generationValues, current.draft), quote = "")
        }
        if (current.banned) return
        // Serverless/local mode answers on the device, so it can send while the server is out of reach.
        if (notifySendOffline(current.offline)) return
        if (current.busy || (current.draft.isBlank() && current.attachments.isEmpty())) return
        if (current.model.isBlank()) { mutable.update { it.copy(notice = "モデルを選択してください。") }; return }
        current.account?.models?.firstOrNull { it.id == current.model && it.selectable } ?: return
        // Web sendMessage checks: audio / video the model cannot take, and what Mistral OCR accepts.
        val (audioOk, videoOk) = modelMediaSupport(current.model)
        val hasAudio = current.attachments.any { isAudioPath(it.reference) }
        val hasVideo = current.attachments.any { isVideoPath(it.reference) }
        if ((hasAudio && !audioOk) || (hasVideo && !videoOk)) {
            notify("このモデルは音声/動画入力に対応していません")
            mutable.update { it.copy(attachments = it.attachments.filterNot { a ->
                (isAudioPath(a.reference) && !audioOk) || (isVideoPath(a.reference) && !videoOk) }) }
            return
        }
        if (isMistralOcrModel(current.model)) {
            if (hasAudio || hasVideo) { notify("Mistral OCR は音声・動画に対応していません。PDF / 画像 / DOCX / PPTX を添付してください。"); return }
            if (current.attachments.isEmpty() && !Regex("https?://\\S+", RegexOption.IGNORE_CASE).containsMatchIn(current.draft)) {
                notify("Mistral OCR は文書専用です。PDF・画像・DOCX・PPTX を添付するか、公開URLを入力してください。")
                return
            }
        }
        // Web: an X link switches to Search + Grok 4 Fast Reasoning, or asks first when that is turned off.
        val hasXLink = X_LINK_PATTERN.containsMatchIn(current.draft) || X_LINK_PATTERN.containsMatchIn(current.quote)
        if (!xLinkChecked && hasXLink && !isMistralOcrModel(current.model) && !current.enableSearch) {
            if (current.preferences?.autoSearchOnLinks != false) {
                applyXLinkSearch()
                current = state.value
            } else {
                mutable.update { it.copy(xLinkPrompt = true) }
                return
            }
        }
        val generation = try { generationOptionsPayload(current.model, current.generationValues) }
            catch (e: IllegalArgumentException) { notify(e.message ?: "生成設定を確認してください。"); return }
        // Web toggleOptions: hidden options are off and forced checkboxes send their forced value.
        val rules = composerRules(current.model, current.mcpServers.any { it.enabled })
        if (current.batchMode && rules.batch.visible && current.codingMode) {
            notify("Batch APIではCoding Modeを利用できません。Batchを解除するかCodingを解除してください。")
            return
        }
        val items = attachmentItemsForSend(current.attachments)
        if (current.imageMask != null && isGptImageModel(current.model) && items.isEmpty()) {
            notify("Mask は画像入力が必要です")
            return
        }
        if (removeObject && current.imageMask == null) {
            notify("Object Remover にはマスク画像が必要です")
            return
        }
        val body = when (val request = ChatSendRequest.build(current, items, generation, rules, pending.parentId, disableAutoSearch)) {
            is ChatSendRequest.Result.Rejected -> { notify(request.message); return }
            is ChatSendRequest.Result.Ready -> request.body
        }
        val submission = ChatSubmission(body, current.attachments, current.imageMask, current.editingMessageId, pending.parentId, pending.settingsBeforeEdit)
        pending.clearEdit()
        pending.setFailed(submission)
        // Web `resetUploadState` + `clearQuote`: the mask goes with the attachments.
        mutable.update { it.copy(promptSendMotionId = "local-${body.getString("client_request_id")}", draft = if (removeBackground || removeObject) it.draft else "", attachments = emptyList(), editingMessageId = null, quote = "", quoteSourceId = null, quoteMessageId = null, imageMask = null) }
        submit(submission)
    }

    fun submit(submission: ChatSubmission) {
        if (generations.current(state.value) != null) return
        messageLoader.invalidate()
        val answerBackend = backend()
        val answerToken = token()
        val answer = generations.begin(state.value)
        answer.threadId?.let { pending.removeFailed(it) }
        generations.launch(answer, scope, keepAlive()) {
            val body = submission.body
            val modelId = body.optString("model")
            val reasoning = showsReasoningProgress(modelId, body.optBoolean("enable_thinking"), body.optString("reasoning_effort"))
            updateAnswer(answer) { it.copy(streaming = true, retryAvailable = false, status = "", liveContent = "", liveThought = "",
                cards = emptyList(), mcpDecision = null,
                live = LiveAnswer(model = modelId, pendingStatus = "APIに送信中...",
                    pendingStartedAt = if (body.optBoolean("batch_mode")) null else SystemClock.elapsedRealtime(),
                    thoughtPlaceholder = if (reasoning) "推論プロセスを準備中..." else null)) }
            val flow = api.progress.startFlow("chat")
            var id = body.nullableString("thread_id")
            var accepted = false
            val sendStarted = System.currentTimeMillis()
            Diagnostics.log("send.start", "op" to answer.operationId, "model" to modelId, "serverless" to state.value.serverless, "new_thread" to id.isBlank(),
                "attachments" to submission.files.map { it.mime.ifBlank { "unknown" } }, "text_chars" to body.optString("message").length,
                "prompt_cache" to body.optBoolean("enable_prompt_caching"), "memory" to Diagnostics.memory())
            try {
                if (body.optBoolean("parent_id_explicit")) {
                    // Branching send (edit-and-resend, regenerate): drop the old branch's tail
                    // from the visible path immediately, instead of leaving the previous
                    // prompt/reply bubbles on screen until the new answer finishes streaming.
                    val parentId = body.opt("parent_id") as? Int
                    updateAnswer(answer) { current ->
                        val truncated = if (parentId == null) emptyList()
                        else current.messages.indexOfFirst { numericId(it) == parentId }
                            .let { idx -> if (idx >= 0) current.messages.subList(0, idx + 1) else current.messages }
                        current.copy(messages = truncated)
                    }
                }
                val userId = "local-${body.getString("client_request_id")}"
                updateAnswer(answer) { it.copy(messages = it.messages.filterNot { m -> m.id == userId } + ChatMessage(userId, "user",
                    body.getString("message"), files = submission.files.map { a -> a.reference },
                    quote = body.optString("quote_text"), skillName = it.selectedSkill?.name.orEmpty(),
                    promptOptions = com.minashin1120.aiplayground.data.promptOptionsRecord(body))) }
                if (id.isBlank()) {
                    val created = answerBackend.post("/api/threads", JSONObject().put("is_temporary", answer.snapshot.value.newThreadTemporary), answerToken)
                    id = created.get("id").toString()
                    body.put("thread_id", id)
                    updateAnswer(answer) { it.copy(selected = ThreadItem(id, created.nullableString("title"), it.model,
                        isTemporary = created.optBoolean("is_temporary")), newThreadTemporary = false) }
                    answer.bind(id)
                    if (answer.matches(state.value)) syncHeartbeat()
                }
                val threadId = id
                sendStream.send(answer, body, answerBackend, answerToken, threadId, sendStarted,
                    acceptedOnly = { accepted = true },
                    setPhase = { flow.setPhase(it) },
                    connectionUnavailable = { setConnectionUnavailable(it) })
                sendCompletion.finish(answer, body, answerBackend, answerToken, id, modelId, sendStarted)
            } catch (e: CancellationException) {
                Diagnostics.log("send.cancelled", "op" to answer.operationId, "ms" to System.currentTimeMillis() - sendStarted, "accepted" to accepted)
                throw e
            }
            catch (e: Exception) {
                Diagnostics.failure("send.error", e, "op" to answer.operationId, "ms" to System.currentTimeMillis() - sendStarted, "accepted" to accepted)
                sendCompletion.fail(answer, body, submission, id, modelId, accepted, e)
            } finally {
                flow.finish()
                if (generations.contains(answer)) updateAnswer(answer) { it.copy(streaming = false, status = "", live = LiveAnswer(), liveContent = "", liveThought = "", cards = emptyList(), jobId = null, mcpDecision = null) }
            }
        }
    }

    /** Web `resumePendingStream`: rejoin an answer still running on the server when the thread is opened. */
    fun resume() {
        generations.current(state.value)?.let { answer ->
            mutable.update { answer.render(it) }
            return
        }
        if (state.value.streaming) return
        val id = state.value.selected?.id ?: return
        val jobId = state.value.jobId ?: return
        val answerBackend = backend()
        val answerToken = token()
        val answer = generations.begin(state.value)
        generations.launch(answer, scope, keepAlive()) {
            val flow = api.progress.startFlow("chatResume")
            updateAnswer(answer) { it.copy(streaming = true, liveContent = "", liveThought = "", status = "",
                live = LiveAnswer(model = it.model, pendingStatus = "回答を生成中...")) }
            var failed = false
            Diagnostics.log("send.stream.start", "op" to answer.operationId)
            try {
                answerBackend.stream("/chat_stream_resume", JSONObject().put("thread_id", id).put("job_id", jobId), answerToken,
                    onAccepted = { flow.setPhase("waiting") }) { event ->
                    if (generations.contains(answer)) { flow.setPhase("receiving"); streamReceiver.accept(answer, event) }
                }
                failed = answer.snapshot.value.live.error != null
            } catch (e: CancellationException) {
                Diagnostics.log("send.cancelled", "op" to answer.operationId)
                throw e
            }
            catch (e: Exception) {
                Diagnostics.failure("send.stream.error", e, "op" to answer.operationId)
                if (waitingStatusFor(e) != null) {
                    try {
                        streamReceiver.reconnect(answer, id, answer.snapshot.value.jobId ?: jobId, answerBackend, answerToken)
                        failed = answer.snapshot.value.live.error != null
                    } catch (cancelled: CancellationException) { throw cancelled }
                    catch (retryError: Exception) { failed = true; report(retryError) }
                } else if (e !is ApiException || e.status != 404) { failed = true; report(e) }
            }
            finally {
                flow.finish()
                if (generations.contains(answer)) updateAnswer(answer) { it.copy(streaming = false, status = "", live = LiveAnswer(), liveContent = "", liveThought = "", cards = emptyList(), jobId = null, mcpDecision = null) }
                generations.finish(answer)
            }
            runCatching { reloadAfterResume(id) }.onFailure { report(it) }
            Diagnostics.log("send.stream.done", "op" to answer.operationId, "failed" to failed)
            notifyAnswerFinishedIfAway(id, failed)
        }
    }
    fun stop() { scope.launch {
        val id = state.value.selected?.id ?: return@launch
        val answer = generations.forThread(id)
        val targetJob = answer?.job
        val targetJobId = answer?.snapshot?.value?.jobId ?: state.value.jobId
        val targetBackend = backend()
        val targetToken = token()
        Diagnostics.log("send.stop", "thread_id" to id, "streaming" to state.value.streaming, "job" to (state.value.jobId != null))
        try {
            // The answer is generated on the device (a server job is only rejoined, with its job id):
            // cancelling the request saves what arrived so far, and serverless mode then uploads it.
            if (deviceAnswer.shouldStopOnDevice(targetJobId)) {
                deviceAnswer.stop(id, targetJob)
                return@launch
            }
            // Web `stopGeneration`: the stopped job is never rejoined, even while the server still lists it as running.
            targetJobId?.let { messageLoader.suppress(it) }
            try {
                val stopped = targetBackend.post("/api/stop_chat", JSONObject().put("thread_id", id).apply { targetJobId?.let { put("job_id", it) } }, targetToken)
                stopped.optString("job_id").takeIf { it.isNotBlank() }?.let { messageLoader.suppress(it) }
            } catch (e: ApiException) {
                // Web ignores the answer to the stop request: 404 means the job already ended on the server.
                if (e.status != 404) throw e
            }
            targetJob?.cancelAndJoin()
            mutable.update { if (it.selected?.id == id && generations.forThread(id) == null) it.copy(streaming = false, status = "停止を要求しました。", jobId = null, leafId = null) else it }
            // The server saves the partial answer once the worker notices the stop; reload until the job is gone.
            repeat(ChatViewModel.STOP_RELOAD_ATTEMPTS) {
                delay(1000)
                if (state.value.selected?.id != id || state.value.streaming) return@launch
                reloadAfterStop(id)
                if (!messageLoader.stoppedJobStillRunning) return@launch
            }
        } catch (e: Exception) { report(e) }
    } }
}
