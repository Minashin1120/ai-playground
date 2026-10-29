package com.minashin1120.aiplayground

import java.util.concurrent.CopyOnWriteArrayList
import com.minashin1120.aiplayground.data.ChatMessage
import com.minashin1120.aiplayground.data.parseMessages
import org.json.JSONArray
import org.json.JSONObject
import com.minashin1120.aiplayground.data.local.GenerationKeepAlive
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.flow.MutableStateFlow

/** In-flight answers belong to chats, never to the currently displayed screen. */
internal class ChatGenerations {
    internal class Answer(initial: ChatState) {
        val operationId: String = java.util.UUID.randomUUID().toString()
        val snapshot = MutableStateFlow(initial)
        val navigationId = initial.chatNavigationId
        val beganNew = initial.selected == null
        @Volatile var threadId: String? = initial.selected?.id
        private val aliases = java.util.concurrent.ConcurrentHashMap.newKeySet<String>()
        var job: Job? = null
        @Volatile var lastEventAt = 0L

        fun owns(id: String): Boolean = threadId == id || id in aliases
        fun matches(screen: ChatState): Boolean = screen.selected?.id?.let(::owns)
            ?: (beganNew && screen.chatNavigationId == navigationId)

        fun bind(id: String) {
            threadId?.let { aliases.add(it) }
            threadId = id
        }

        /** Only the answer's presentation follows navigation; drafts and history belong to the screen. */
        fun render(screen: ChatState): ChatState {
            val answer = snapshot.value
            return screen.copy(streaming = true, live = answer.live, liveContent = answer.liveContent,
                liveThought = answer.liveThought, status = answer.status, cards = answer.cards,
                jobId = answer.jobId, mcpDecision = answer.mcpDecision, retryAvailable = false)
        }
    }

    private val answers = CopyOnWriteArrayList<Answer>()
    fun begin(initial: ChatState): Answer = Answer(initial).also { answers.add(it) }
    fun current(screen: ChatState): Answer? = answers.firstOrNull { it.matches(screen) }
    fun forThread(id: String): Answer? = answers.firstOrNull { it.owns(id) }
    fun all(): List<Answer> = answers.toList()
    val active: Boolean get() = answers.isNotEmpty()
    fun contains(answer: Answer): Boolean = answer in answers

    fun update(answer: Answer, screen: MutableStateFlow<ChatState>, transform: (ChatState) -> ChatState) {
        synchronized(answer) {
            if (!contains(answer)) return
            answer.snapshot.update(transform)
            screen.update { if (contains(answer) && answer.matches(it)) transform(it) else it }
        }
    }

    fun restore(screen: MutableStateFlow<ChatState>, threadId: String, finished: Boolean = false) {
        if (finished) return
        forThread(threadId)?.let { answer ->
            synchronized(answer) {
                screen.update { if (contains(answer) && answer.matches(it)) answer.render(it) else it }
            }
        }
    }

    /** Register before starting; release protection even on cancellation or an early exception. */
    fun launch(answer: Answer, scope: CoroutineScope, keepAlive: GenerationKeepAlive, block: suspend () -> Unit) {
        answer.job = scope.launch(start = CoroutineStart.LAZY) {
            val hold = keepAlive.begin("chat")
            try { block() } finally {
                finish(answer)
                hold.close()
            }
        }
        answer.job?.start()
    }
    /** A device answer is checkpointed while streaming; render it once, in the live bubble. */
    fun history(payload: JSONObject, threadId: String, finished: Boolean = false): List<ChatMessage> {
        if (finished || forThread(threadId) == null) return parseMessages(payload)
        val rows = payload.optJSONArray("messages") ?: return emptyList()
        val saved = (0 until rows.length()).map { rows.getJSONObject(it) }
            .filterNot { it.optString("role") == "assistant" && it.optBoolean("generating") }
        return parseMessages(JSONObject().put("messages", JSONArray(saved)))
    }

    fun finish(answer: Answer) {
        answers.remove(answer)
    }
    fun cancelAll(except: Job? = null): List<Job> {
        val jobs = answers.mapNotNull { it.job }
        answers.clear()
        jobs.filter { it !== except }.forEach { it.cancel() }
        return jobs
    }
    fun rebind(from: String, to: String) { forThread(from)?.bind(to) }
}
