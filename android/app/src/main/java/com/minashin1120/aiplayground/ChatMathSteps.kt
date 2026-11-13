package com.minashin1120.aiplayground

import com.minashin1120.aiplayground.data.ChatMessage
import com.minashin1120.aiplayground.data.Diagnostics
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

    /** Drop the saved simplified variant so another device (and a reload) shows the original. */
    fun clearSimplified(message: ChatMessage, stepIndex: Int) {
        val id = message.id.toIntOrNull() ?: return
        val index = stepIndex.coerceIn(0, MathStepsState.MAX_STEP)
        updateState(id) { state ->
            (state ?: MathStepsState()).copy(simplified = (state?.simplified.orEmpty() - index))
        }
        progressJob = scope.launch {
            try {
                backend().post(
                    "/api/math_steps/progress",
                    JSONObject()
                        .put("message_id", id)
                        .put("current", index)
                        .put("max_reached", index)
                        .put("remove_simplified", org.json.JSONArray().put(index)),
                    token(),
                )
                Diagnostics.log("math_steps.progress_done", "index" to index, "count" to 1)
            } catch (e: Exception) {
                Diagnostics.log("math_steps.progress_error", "index" to index)
            }
        }
    }

    fun simplify(message: ChatMessage, stepIndex: Int, done: (String?) -> Unit) {
        val id = message.id.toIntOrNull()
        if (id == null) { done(null); return }
        val index = stepIndex.coerceIn(0, MathStepsState.MAX_STEP)
        simplifyJob = scope.launch {
            try {
                Diagnostics.log("math_steps.simplify_start", "index" to index)
                val reply = backend().post(
                    "/api/math_steps/simplify",
                    JSONObject().put("message_id", id).put("step_index", index),
                    token(),
                )
                val text = reply.optString("text").takeIf { it.isNotBlank() && it != "null" }
                val saved = MathStepsState.parse(reply.opt("math_steps_state"))
                updateState(id) { current ->
                    when {
                        saved != null -> (current ?: MathStepsState()).let { base ->
                            // The simplify call does not move the reader; keep this device's step.
                            base.copy(maxReached = maxOf(base.maxReached, saved.maxReached), simplified = saved.simplified)
                        }
                        text != null -> (current ?: MathStepsState(current = index, maxReached = index)).withSimplified(index, text)
                        else -> current ?: MathStepsState()
                    }
                }
                Diagnostics.log("math_steps.simplify_done", "index" to index, "chars" to (text?.length ?: 0))
                done(text)
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
