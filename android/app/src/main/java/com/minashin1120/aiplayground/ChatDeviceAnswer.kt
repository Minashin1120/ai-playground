package com.minashin1120.aiplayground

import com.minashin1120.aiplayground.data.Diagnostics
import com.minashin1120.aiplayground.data.LiveAnswer
import com.minashin1120.aiplayground.data.local.LocalChatStore
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update

internal class ChatDeviceAnswer(
    private val scope: CoroutineScope,
    private val state: kotlinx.coroutines.flow.StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val generations: ChatGenerations,
    private val syncRunner: ChatSyncRunner,
    private val updateAnswer: (ChatGenerations.Answer, (ChatState) -> ChatState) -> Unit,
    private val loadMessages: suspend (String, Boolean, Boolean) -> Unit,
    private val report: (Throwable) -> Unit,
    private val uploadsLocal: () -> Boolean = { state.value.localProfile },
) {
    fun answeringOnDevice(): Boolean = isDeviceAnswer(state.value.streaming, uploadsLocal(), state.value.jobId)

    /** True when the running answer is generated on this device (no server job id). */
    fun shouldStopOnDevice(jobId: String?): Boolean = isDeviceAnswer(state.value.streaming, uploadsLocal(), jobId)

    /** A chat kept on the device: any chat of the no-account profile, or a device chat in serverless mode. */
    fun isDeviceChat(id: String?, chatLocal: Boolean, serverless: Boolean): Boolean =
        isDeviceChatThread(id, chatLocal, serverless)

    suspend fun stop(id: String, targetJob: Job?) {
        val started = System.currentTimeMillis()
        targetJob?.cancelAndJoin()
        Diagnostics.log("send.stopped", "thread_id" to id, "wait_ms" to System.currentTimeMillis() - started)
        mutable.update { if (it.selected?.id == id && generations.forThread(id) == null) it.copy(streaming = false, status = "停止を要求しました。", live = com.minashin1120.aiplayground.data.LiveAnswer(), leafId = null) else it }
        loadMessages(id, false, true)
        syncRunner.schedule()
    }

    fun canSendWhileOffline(): Boolean = uploadsLocal()

    /** Notice shown when the user cannot send because the device is offline (serverless/local chats can send offline). */
    fun offlineSendNotice(offline: Boolean): String? =
        if (offline && !canSendWhileOffline()) "オフライン中はメッセージを送信できません。" else null
}

/** A running answer that the device itself generates: no server job id, and chats are answered on the device. */
internal fun isDeviceAnswer(streaming: Boolean, uploadsLocal: Boolean, jobId: String?): Boolean =
    streaming && uploadsLocal && jobId == null

/** A chat stored on this device (no-account profile) or a device-only thread in serverless mode. */
internal fun isDeviceChatThread(id: String?, chatLocal: Boolean, serverless: Boolean): Boolean =
    chatLocal || (serverless && id != null && LocalChatStore.isDeviceThreadId(id))
