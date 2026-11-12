package com.minashin1120.aiplayground

import com.minashin1120.aiplayground.data.ApiException
import com.minashin1120.aiplayground.data.ConnectionStatus
import com.minashin1120.aiplayground.data.Diagnostics
import com.minashin1120.aiplayground.data.LiveAnswer
import com.minashin1120.aiplayground.data.backend.ChatBackend
import com.minashin1120.aiplayground.data.connectionStatusForHttp
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay
import org.json.JSONObject
import java.io.IOException

/**
 * Web `fetchChatStreamWithUnavailableRetry`：回答の送信と再接続だけを持つ。
 * 送信を受け付けたか、待つべき失敗か（メンテナンス・サーバー停止・オフライン）を判断し、
 * 受け付け後の切断はサーバーで継続中の回答へつなぎ直す。
 * 状態は持たず、画面の更新は生成時に受け取った関数へ委ねる（`ChatViewModel` の送信全体とは別に読めるようにする）。
 */
internal class ChatSendStream(
    private val receiver: ChatStreamReceiver,
    private val isAlive: (ChatGenerations.Answer) -> Boolean,
    private val waitingStatusFor: (Throwable) -> String?,
    private val updateAnswer: (ChatGenerations.Answer, (ChatState) -> ChatState) -> Unit,
    private val updateLive: (ChatGenerations.Answer, (LiveAnswer) -> LiveAnswer) -> Unit,
    private val markReachable: () -> Unit,
    private val markAccepted: (ChatGenerations.Answer) -> Unit,
    private val notify: (String) -> Unit,
) {
    /**
     * `POST /chat_stream` を送り、受け付けるまで待機する。受け付けたかどうかは [acceptedOnly] で呼び出し元へ返す
     * （失敗時に送信内容を復元するかどうかの判断に使う）。
     */
    suspend fun send(
        answer: ChatGenerations.Answer,
        body: JSONObject,
        answerBackend: ChatBackend,
        answerToken: String,
        threadId: String,
        sendStarted: Long,
        acceptedOnly: () -> Unit,
        setPhase: (String) -> Unit,
        connectionUnavailable: (ConnectionStatus) -> Unit,
    ) {
        var accepted = false
        val seenEvents = java.util.Collections.synchronizedSet(HashSet<String>())
        val onEvent: (JSONObject) -> Unit = { event ->
            val type = event.optString("type")
            if (type == "status" || type == "error" || seenEvents.add(type)) {
                // Only the fixed event type is diagnostic; status/error text may contain private content.
                Diagnostics.log("send.event", "op" to answer.operationId, "type" to type, "ms" to System.currentTimeMillis() - sendStarted,
                    "failed" to (type == "error"))
            }
            if (isAlive(answer)) {
                setPhase("receiving")
                receiver.accept(answer, event)
            }
        }
        val onAccepted: () -> Unit = {
            accepted = true
            acceptedOnly()
            setPhase("waiting")
            markReachable()
            markAccepted(answer)
        }
        var retryCount = 0
        while (true) {
            try {
                answerBackend.stream("/chat_stream", body, answerToken, onAccepted, onEvent)
                break
            } catch (e: CancellationException) { throw e }
            catch (e: ApiException) {
                if (e.code == "request_already_accepted") {
                    accepted = true
                    acceptedOnly()
                    val jobId = e.payload.optString("job_id")
                    updateAnswer(answer) { it.copy(jobId = jobId.ifBlank { it.jobId }) }
                    receiver.reconnect(answer, threadId, jobId, answerBackend, answerToken)
                    break
                }
                val waiting = waitingStatusFor(e) ?: throw e
                retryCount += 1
                connectionStatusForHttp(e.status)?.let { connectionUnavailable(it) }
                updateLive(answer) { it.copy(pendingStatus = waiting, pendingSub = "送信内容を保持して自動再試行中（${retryCount}回目）") }
            } catch (e: IOException) {
                if (accepted) {
                    // Web: the answer keeps running on the server; reconnect to it in the background.
                    connectionUnavailable(ConnectionStatus.OFFLINE)
                    notify("回答への接続が切れました。バックグラウンド処理へ自動再接続します。")
                    receiver.reconnect(answer, threadId, answer.snapshot.value.jobId.orEmpty(), answerBackend, answerToken)
                    break
                }
                retryCount += 1
                connectionUnavailable(ConnectionStatus.OFFLINE)
                updateLive(answer) { it.copy(pendingStatus = "インターネット接続の復帰を待っています...",
                    pendingSub = "送信内容を保持して自動再試行中（${retryCount}回目）") }
            }
            delay(ChatViewModel.CONNECTION_RETRY_DELAY_MS)
        }
    }
}

/** Web `fetchChatStreamWithUnavailableRetry`: which responses keep the send waiting instead of failing. */
internal fun streamWaitingStatusFor(error: Throwable): String? = when {
    error is ApiException && error.status == 503 -> "メンテナンス終了を待っています..."
    error is ApiException && connectionStatusForHttp(error.status) == ConnectionStatus.SERVER_DOWN -> "サーバーの復帰を待っています..."
    error is ApiException && error.status == 425 && error.code == "submission_in_progress" -> "サーバーの復帰を待っています..."
    error is ApiException -> null
    error is IOException -> "インターネット接続の復帰を待っています..."
    else -> null
}
