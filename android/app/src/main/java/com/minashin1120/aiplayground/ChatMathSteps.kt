package com.minashin1120.aiplayground

import com.minashin1120.aiplayground.data.ChatMessage
import com.minashin1120.aiplayground.data.Diagnostics
import com.minashin1120.aiplayground.data.MathStepPart
import com.minashin1120.aiplayground.data.MathStepsState
import com.minashin1120.aiplayground.data.backend.ChatBackend
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import org.json.JSONObject

/**
 * Per-message progress of 「数学ステップ解説」:
 * `POST /api/math_steps/progress` and `POST /api/math_steps/simplify`.
 *
 * The step view changes immediately; failures only log (no notification) because the answer is
 * still readable locally and the next device load picks the server state up again.
 */
internal class ChatMathSteps(
    private val mutable: MutableStateFlow<ChatState>,
    private val scope: CoroutineScope,
    private val backend: () -> ChatBackend,
    private val token: () -> String,
) {
    private var progressJob: Job? = null
    private var simplifyJob: Job? = null

    /** The latest request (BAN・ログアウトで止める対象). */
    val activeJob: Job? get() = progressJob ?: simplifyJob

    fun saveProgress(message: ChatMessage, current: Int, maxReached: Int) {
        val id = message.id.toIntOrNull() ?: return
        val step = current.coerceIn(0, MathStepsState.MAX_STEP)
        val max = maxReached.coerceAtLeast(step).coerceIn(0, MathStepsState.MAX_STEP)
        // Optimistic: the open screen keeps the new position even while the request runs.
        updateState(id) { state ->
            (state ?: MathStepsState()).copy(current = step, maxReached = maxOf(state?.maxReached ?: 0, max))
        }
        progressJob = scope.launch {
            try {
                val reply = backend().post(
                    "/api/math_steps/progress",
                    JSONObject().put("message_id", id).put("current", step).put("max_reached", max),
                    token(),
                )
                MathStepsState.parse(reply.opt("math_steps_state"))?.let { saved ->
                    updateState(id) { saved }
                }
                Diagnostics.log("math_steps.progress_done", "index" to step, "count" to max)
            } catch (e: Exception) {
                Diagnostics.log("math_steps.progress_error", "index" to step)
            }
        }
    }

    /** Drop a saved simplified body or expanded step so another device shows the original. */
    fun clearVariants(message: ChatMessage, sourceIndex: Int, currentIndex: Int) {
        val id = message.id.toIntOrNull() ?: return
        val index = sourceIndex.coerceIn(0, MathStepsState.MAX_STEP)
        val current = currentIndex.coerceIn(0, MathStepsState.MAX_STEP)
        updateState(id) { state ->
            val base = state ?: MathStepsState()
            base.copy(
                current = current,
                maxReached = maxOf(base.maxReached, current),
                simplified = base.simplified - index,
                expanded = base.expanded - index,
            )
        }
        progressJob = scope.launch {
            try {
                backend().post(
                    "/api/math_steps/progress",
                    JSONObject()
                        .put("message_id", id)
                        .put("current", current)
                        .put("max_reached", maxOf(current, message.mathStepsState?.maxReached ?: current))
                        .put("remove_simplified", org.json.JSONArray().put(index))
                        .put("remove_expanded", org.json.JSONArray().put(index)),
                    token(),
                )
                Diagnostics.log("math_steps.progress_done", "index" to current, "count" to 1)
            } catch (e: Exception) {
                Diagnostics.log("math_steps.progress_error", "index" to current)
            }
        }
    }

    /** Splits the current step (or one of its expanded parts) into easier smaller steps. */
    fun simplify(
        message: ChatMessage,
        stepIndex: Int,
        partIndex: Int?,
        done: (List<MathStepPart>?) -> Unit,
    ) {
        val id = message.id.toIntOrNull()
        if (id == null) { done(null); return }
        val index = stepIndex.coerceIn(0, MathStepsState.MAX_STEP)
        simplifyJob = scope.launch {
            try {
                Diagnostics.log("math_steps.simplify_start", "index" to index, "part" to (partIndex ?: -1))
                val payload = JSONObject().put("message_id", id).put("step_index", index)
                if (partIndex != null) payload.put("part_index", partIndex)
                val reply = backend().post("/api/math_steps/simplify", payload, token())
                val rows = reply.optJSONArray("steps")
                val parts = ArrayList<MathStepPart>()
                if (rows != null) {
                    for (i in 0 until minOf(rows.length(), MathStepsState.MAX_EXPANDED_PER_STEP)) {
                        val row = rows.optJSONObject(i) ?: continue
                        val body = row.optString("body")
                        if (body.isBlank() || body == "null") continue
                        parts += MathStepPart(row.optString("title"), body)
                    }
                }
                if (parts.isEmpty()) { done(null); return@launch }
                val saved = MathStepsState.parse(reply.opt("math_steps_state"))
                updateState(id) { current ->
                    val base = current ?: MathStepsState()
                    val server = saved ?: base
                    // The simplify call does not move the reader; keep this device's step.
                    server.copy(
                        current = base.current,
                        maxReached = maxOf(base.maxReached, server.maxReached),
                        simplified = server.simplified - index,
                        expanded = server.expanded + (index to parts.toList()),
                    )
                }
                Diagnostics.log("math_steps.simplify_done", "index" to index, "parts" to parts.size)
                done(parts)
            } catch (e: Exception) {
                Diagnostics.log("math_steps.simplify_error", "index" to index)
                done(null)
            }
        }
    }

    private fun updateState(messageId: Int, transform: (MathStepsState?) -> MathStepsState) {
        mutable.update { current ->
            current.copy(
                messages = current.messages.map { message ->
                    if (message.id.toIntOrNull() == messageId) message.copy(mathStepsState = transform(message.mathStepsState)) else message
                },
                allMessages = current.allMessages.map { message ->
                    if (message.id.toIntOrNull() == messageId) message.copy(mathStepsState = transform(message.mathStepsState)) else message
                },
            )
        }
    }
}
