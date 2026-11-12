package com.minashin1120.aiplayground

import com.minashin1120.aiplayground.data.LiveAnswer
import kotlinx.coroutines.flow.MutableStateFlow

/**
 * Answer updates shared by the send, stream and device helpers (Web `updateAnswer` / `updateLive` /
 * `markApiAccepted`). The answer itself and the screen belong to the view model; this class only
 * applies the same transform to both through [ChatGenerations.update].
 */
internal class ChatAnswerState(
    private val generations: ChatGenerations,
    private val mutable: MutableStateFlow<ChatState>,
) {
    fun update(answer: ChatGenerations.Answer, transform: (ChatState) -> ChatState) {
        generations.update(answer, mutable, transform)
    }

    fun updateLive(answer: ChatGenerations.Answer, transform: (LiveAnswer) -> LiveAnswer) =
        update(answer) { it.copy(live = transform(it.live)) }

    /** Web `markApiAccepted`: once, the skeleton says the connection is up and the model is being awaited. */
    fun markAccepted(answer: ChatGenerations.Answer) = updateLive(answer) { live ->
        if (live.accepted || live.pendingStatus == null) live.copy(accepted = true)
        else live.copy(accepted = true, pendingStatus = "接続しました。応答を待っています...", pendingSub = "応答開始までの状況を表示しています")
    }
}
