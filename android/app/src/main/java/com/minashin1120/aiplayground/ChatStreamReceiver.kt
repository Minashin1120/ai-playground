package com.minashin1120.aiplayground

import android.os.SystemClock
import com.minashin1120.aiplayground.data.ApiException
import com.minashin1120.aiplayground.data.LiveAnswer
import com.minashin1120.aiplayground.data.McpDecision
import com.minashin1120.aiplayground.data.backend.ChatBackend
import com.minashin1120.aiplayground.data.upsertPythonCard
import com.minashin1120.aiplayground.data.upsertToolCard
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay
import org.json.JSONObject
import java.io.IOException

/**
 * 回答ストリームの受信側。イベントの適用（[accept]）と、サーバーで継続中の回答への再接続（[reconnect]）だけを持つ。
 * 状態は持たず、回答・画面の更新は生成時に受け取った関数へ委ねる（`ChatViewModel` の送信・添付・同期とは別に読めるようにする）。
 */
internal class ChatStreamReceiver(
    private val isAlive: (ChatGenerations.Answer) -> Boolean,
    private val updateAnswer: (ChatGenerations.Answer, (ChatState) -> ChatState) -> Unit,
    private val updateLive: (ChatGenerations.Answer, (LiveAnswer) -> LiveAnswer) -> Unit,
    private val markAccepted: (ChatGenerations.Answer) -> Unit,
    private val markReachable: () -> Unit,
    private val waitingStatusFor: (Throwable) -> String?,
) {
    /**
     * Web `reconnectPendingStreamUntilAvailable`: while the answer runs on the server, wait and rejoin it
     * through `/chat_stream_resume` until it finishes (404 means it already ended).
     */
    suspend fun reconnect(answer: ChatGenerations.Answer, threadId: String, jobId: String, answerBackend: ChatBackend, answerToken: String) {
        var activeJobId = jobId
        while (true) {
            updateLive(answer) { it.copy(pendingStatus = if (it.pendingStatus != null) "サーバーへの再接続を待っています..." else null,
                pendingSub = "回答処理はバックグラウンドで継続しています") }
            delay(ChatViewModel.CONNECTION_RETRY_DELAY_MS)
            if (!isAlive(answer)) return
            try {
                if (activeJobId.isBlank()) {
                    activeJobId = answerBackend.get("/api/threads/$threadId?limit=1", answerToken)
                        .optJSONObject("pending_job")?.optString("job_id").orEmpty()
                    if (activeJobId.isBlank()) return
                    updateAnswer(answer) { it.copy(jobId = activeJobId) }
                }
                answerBackend.stream("/chat_stream_resume", JSONObject().put("thread_id", threadId).put("job_id", activeJobId), answerToken,
                    onAccepted = { markReachable() }) { event -> if (isAlive(answer)) accept(answer, event) }
                return
            } catch (e: CancellationException) { throw e }
            catch (e: ApiException) {
                if (e.status == 404) return
                if (waitingStatusFor(e) == null) throw e
            } catch (e: IOException) { /* keep waiting */ }
        }
    }

    /** Web `acceptEvent`: apply one answer event to the answer and the screen showing it. */
    fun accept(answer: ChatGenerations.Answer, event: JSONObject) {
        answer.lastEventAt = SystemClock.elapsedRealtime()
        if (!isAlive(answer)) return
        val type = event.optString("type")
        val content = event.opt("content")?.takeIf { it != JSONObject.NULL }?.toString().orEmpty()
        when (type) {
            "thread_id" -> { markAccepted(answer); return }
            "job_id" -> { markAccepted(answer); updateAnswer(answer) { it.copy(jobId = content) }; return }
            "search_status" -> {
                if (content == "searching") updateLive(answer) { it.copy(search = "searching") }
                else if (content == "done" && answer.snapshot.value.live.search == "searching") {
                    updateLive(answer) { it.copy(search = "done") }

                }
                return
            }
            "mcp", "mcp_decision_request", "mcp_decision_resolved" -> {
                val payload = event.optJSONObject("content")
                updateAnswer(answer) { current -> current.copy(
                    cards = upsertToolCard(current.cards, type, event.opt("content")),
                    mcpDecision = when (type) {
                        "mcp_decision_request" -> payload?.let { McpDecision(
                            id = it.optString("id"), jobId = current.jobId.orEmpty(),
                            serverName = it.optString("server_name").ifBlank { "不明なサーバー" },
                            toolName = it.optString("tool_name"), argsPreview = it.optString("args_preview"),
                        ) }
                        "mcp_decision_resolved" -> current.mcpDecision?.takeIf { d -> d.id != payload?.optString("id") }
                        else -> if (payload?.optString("type") == "decision_resolved" &&
                            current.mcpDecision?.id == payload?.optString("id")) null else current.mcpDecision
                    },
                ) }
                return
            }
            "status" -> {
                if (event.optString("phase") != "uploading") markAccepted(answer)
                updateLive(answer) { live ->
                    live.copy(
                        pendingStatus = live.pendingStatus?.let { content.ifBlank { "モデル処理中..." } },
                        pendingSub = if (live.pendingStatus != null) "応答開始までの進捗を表示しています" else live.pendingSub,
                        thoughtPlaceholder = live.thoughtPlaceholder?.let { content.ifBlank { "推論プロセスを準備中..." } },
                    )
                }
                return
            }
            "done" -> { updateAnswer(answer) { it.copy(jobId = null) }; return }
        }
        // Web `beginPendingToStreamTransition`: the first answer event replaces the skeleton.
        updateLive(answer) { it.copy(pendingStatus = null, pendingSub = "") }
        when (type) {
            "python" -> event.optJSONObject("content")?.let { payload -> updateAnswer(answer) { it.copy(cards = upsertPythonCard(it.cards, payload)) } }
            "coding_diff" -> updateAnswer(answer) { it.copy(cards = upsertToolCard(it.cards, type, event.opt("content"))) }
            "image_analysis" -> updateLive(answer) { it.copy(imageAnalysis = content) }
            "thought" -> updateAnswer(answer) { it.copy(liveThought = it.liveThought + content, live = it.live.copy(thoughtPlaceholder = null)) }
            "content" -> updateAnswer(answer) { it.copy(liveContent = it.liveContent + content) }
            "error" -> updateAnswer(answer) { it.copy(live = it.live.copy(error = content.ifBlank { "Unknown error" }), notice = content.ifBlank { "Unknown error" }.take(500)) }
        }
    }
}
