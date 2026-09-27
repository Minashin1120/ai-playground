package com.minashin1120.aiplayground.data.direct

import org.json.JSONObject

/** One attachment sent to a provider: raw bytes (images, PDF, audio) or text extracted on the device. */
data class DirectAttachment(val name: String, val mime: String, val bytes: ByteArray? = null, val text: String? = null)

/** One turn of the conversation sent to a provider (history and the new message). */
data class DirectTurn(
    val role: String,
    val text: String,
    val attachments: List<DirectAttachment> = emptyList(),
    /** Gemini thought signatures of this assistant turn, sent back unchanged. */
    val thoughtSignatures: List<String> = emptyList(),
)

/** Everything a provider call needs; [options] is the chat request body (`enable_search`, `thinking_level`, …). */
data class DirectRequest(
    val model: String,
    val apiKey: String,
    val system: String,
    val turns: List<DirectTurn>,
    val options: JSONObject,
)

/** A file the model produced (generated image or audio), saved by the caller. */
data class DirectOutputFile(val name: String, val mime: String, val bytes: ByteArray)

data class DirectResult(
    val content: String,
    val thought: String = "",
    val tokensIn: Int? = null,
    val tokensOut: Int? = null,
    val tokensThought: Int? = null,
    val files: List<DirectOutputFile> = emptyList(),
    val thoughtSignatures: List<String> = emptyList(),
)

/**
 * A provider call that streams the same events as the server's `/chat_stream` NDJSON (`status`, `thought`,
 * `content`, `search_status`, `python`), so the chat screen shows it unchanged. [emit] is called on a
 * background thread; the partial text is also reported through [onProgress] so it can be saved while
 * the answer is still arriving.
 */
interface DirectEngine {
    suspend fun run(request: DirectRequest, emit: (JSONObject) -> Unit, onProgress: (content: String, thought: String) -> Unit): DirectResult
}

internal fun event(type: String, content: Any): JSONObject = JSONObject().put("type", type).put("content", content)

internal fun JSONObject.flag(name: String): Boolean = optBoolean(name) || optString(name) == "true"
