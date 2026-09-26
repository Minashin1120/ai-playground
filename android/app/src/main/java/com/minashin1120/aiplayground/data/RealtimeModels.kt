package com.minashin1120.aiplayground.data

data class RealtimeState(
    val active: Boolean = false,
    val model: String = "",
    val sessionId: String = "",
    val status: String = "",
    val userText: String = "",
    val assistantText: String = "",
    val thoughtText: String = "",
    val audioBytes: Long = 0L,
    val error: String? = null,
)

/**
 * Web `LyriaRealtimeStudio` session state. [kind] is the Web status key (idle / connecting / streaming /
 * paused / stopped / error / closed) that colours the status dot and enables the transport buttons.
 */
data class LyriaState(
    val sessionId: String = "",
    val status: String = "準備完了",
    val kind: String = "idle",
    /** When audio first arrived (the `00:00` elapsed timer), 0 before that. */
    val startedAt: Long = 0L,
    val busy: Boolean = false,
    /** The text sent with Lyria RealTime selected, used as the first prompt row. */
    val prompt: String = "",
) {
    val active: Boolean get() = sessionId.isNotBlank()
}

/** One weighted prompt row of the Lyria studio. */
data class LyriaPrompt(val text: String, val weight: Float = 1f)
