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

data class LyriaState(
    val active: Boolean = false,
    val sessionId: String = "",
    val status: String = "",
    val prompt: String = "",
    val audioBytes: Long = 0L,
    val error: String? = null,
)
