package com.minashin1120.aiplayground

import com.minashin1120.aiplayground.data.ChatMessage
import com.minashin1120.aiplayground.data.LiveAnswer
import com.minashin1120.aiplayground.data.ThreadItem
import com.minashin1120.aiplayground.data.local.GenerationKeepAlive
import com.minashin1120.aiplayground.data.local.GenerationTracker
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.yield
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35])
class ChatGenerationsTest {
    private fun screen(id: String, navigation: Long = 1) = ChatState(
        selected = ThreadItem(id, "Chat", "model"), chatNavigationId = navigation)

    @Test fun switchingChatsPreservesTheOriginalStreamAndItsSavedResult() = runBlocking {
        val generations = ChatGenerations()
        val ui = MutableStateFlow(screen("a"))
        val answer = generations.begin(ui.value)
        val tracker = GenerationTracker()
        val finish = CompletableDeferred<Unit>()
        var saved = false
        generations.launch(answer, this, GenerationKeepAlive { tracker.begin(it) }) {
            generations.update(answer, ui) { it.copy(streaming = true, liveContent = "first") }
            finish.await()
            generations.update(answer, ui) { it.copy(liveContent = it.liveContent + " second") }
            saved = true
        }
        yield()
        ui.value = screen("b", 2).copy(draft = "draft in b")
        assertTrue(answer.job!!.isActive)
        assertEquals(listOf("chat"), tracker.active.value)
        finish.complete(Unit)
        answer.job!!.join()
        assertTrue(saved)
        assertEquals("first second", answer.snapshot.value.liveContent)
        assertEquals("", ui.value.liveContent)
        assertEquals("draft in b", ui.value.draft)
        assertTrue(tracker.active.value.isEmpty())
        assertFalse(generations.active)
    }

    @Test fun returningRestoresLiveProgressWithoutReplacingHistoryOrDrafts() {
        val generations = ChatGenerations()
        val initial = screen("a").copy(liveContent = "partial", liveThought = "thinking",
            jobId = "job-a", live = LiveAnswer(search = "searching"))
        val answer = generations.begin(initial)
        val ui = MutableStateFlow(screen("b", 2).copy(draft = "b draft"))
        generations.update(answer, ui) { it.copy(liveContent = it.liveContent + " answer") }
        assertEquals("", ui.value.liveContent)
        val messages = listOf(ChatMessage("1", "user", "question"))
        ui.value = answer.render(screen("a", 3).copy(messages = messages, draft = "a draft"))
        assertSame(answer, generations.current(ui.value))
        assertTrue(ui.value.streaming)
        assertEquals("partial answer", ui.value.liveContent)
        assertEquals("thinking", ui.value.liveThought)
        assertEquals("searching", ui.value.live.search)
        assertEquals("job-a", ui.value.jobId)
        assertEquals(messages, ui.value.messages)
        assertEquals("a draft", ui.value.draft)
    }

    @Test fun stoppingOneOfTwoAnswersKeepsTheOtherAndItsServiceHold() = runBlocking {
        val generations = ChatGenerations()
        val tracker = GenerationTracker()
        val ui = MutableStateFlow(screen("a"))
        val a = generations.begin(ui.value)
        generations.launch(a, this, GenerationKeepAlive { tracker.begin(it) }) { awaitCancellation() }
        ui.value = screen("b", 2)
        val b = generations.begin(ui.value)
        generations.launch(b, this, GenerationKeepAlive { tracker.begin(it) }) { awaitCancellation() }
        yield()
        assertEquals(2, tracker.active.value.size)
        assertSame(b, generations.current(ui.value))
        b.job!!.cancelAndJoin()
        assertTrue(a.job!!.isActive)
        assertNull(generations.forThread("b"))
        assertEquals(1, tracker.active.value.size)
        generations.update(a, ui) { it.copy(liveContent = "a is still receiving") }
        assertEquals("", ui.value.liveContent)
        a.job!!.cancelAndJoin()
        assertTrue(tracker.active.value.isEmpty())
    }

    @Test fun aNewChatGettingItsIdCannotReplaceTheChatOpenedMeanwhile() {
        val generations = ChatGenerations()
        val initial = ChatState(chatNavigationId = 10)
        val answer = generations.begin(initial)
        val ui = MutableStateFlow(screen("other", 11))
        generations.update(answer, ui) { it.copy(selected = ThreadItem("created", "New Chat", "model")) }
        answer.bind("created")
        assertEquals("other", ui.value.selected!!.id)
        assertSame(answer, generations.forThread("created"))
        assertNull(generations.current(ChatState(chatNavigationId = 12)))
    }

    @Test fun synchronizedThreadIdsKeepBothAliasesForTheSameAnswer() {
        val generations = ChatGenerations()
        val answer = generations.begin(screen("device-a"))
        generations.rebind("device-a", "server-a")
        assertSame(answer, generations.forThread("device-a"))
        assertSame(answer, generations.current(screen("server-a", 4)))
        assertEquals("server-a", answer.threadId)
    }

    @Test fun logoutCancelsEveryAnswerAndDropsLateEventsEvenForTheSameThreadId() = runBlocking {
        val generations = ChatGenerations()
        val tracker = GenerationTracker()
        val ui = MutableStateFlow(screen("a"))
        val a = generations.begin(ui.value)
        val b = generations.begin(screen("b", 2))
        for (answer in listOf(a, b)) {
            generations.launch(answer, this, GenerationKeepAlive { tracker.begin(it) }) { awaitCancellation() }
        }
        yield()
        generations.cancelAll().forEach { it.join() }
        assertTrue(a.job!!.isCancelled)
        assertTrue(b.job!!.isCancelled)
        assertTrue(tracker.active.value.isEmpty())
        generations.begin(ui.value)
        generations.update(a, ui) { it.copy(liveContent = "old account", draft = "old draft") }
        assertEquals("", ui.value.liveContent)
        assertEquals("", ui.value.draft)
    }

    @Test fun aCheckpointedPartialAnswerIsNotDuplicatedWhenReturningToItsChat() {
        val generations = ChatGenerations()
        val answer = generations.begin(screen("a"))
        val payload = org.json.JSONObject().put("messages", org.json.JSONArray()
            .put(org.json.JSONObject().put("id", 1).put("role", "user").put("content", "question"))
            .put(org.json.JSONObject().put("id", 2).put("role", "assistant").put("content", "partial").put("generating", true)))
        assertEquals(listOf("user"), generations.history(payload, "a").map { it.role })
        assertEquals(2, generations.history(payload, "a", finished = true).size)
        generations.finish(answer)
        assertEquals(2, generations.history(payload, "a").size)
    }

    @Test fun completedHistoryDoesNotBringBackTheLiveBubbleBeforeCleanupFinishes() {
        val generations = ChatGenerations()
        generations.begin(screen("a").copy(streaming = true, liveContent = "answer"))
        val finalRows = listOf(ChatMessage("2", "assistant", "answer"))
        val ui = MutableStateFlow(screen("a").copy(messages = finalRows))
        generations.restore(ui, "a", finished = true)
        assertFalse(ui.value.streaming)
        assertEquals("", ui.value.liveContent)
        assertEquals(finalRows, ui.value.messages)
    }

    @Test fun failureReleasesOnlyTheFailingAnswersProtection() = runBlocking {
        val generations = ChatGenerations()
        val tracker = GenerationTracker()
        val failed = generations.begin(screen("a"))
        val keepAlive = GenerationKeepAlive { tracker.begin(it) }
        var observedFailure = false
        // An API failure is handled by the caller; its finally must still release this answer's hold.
        generations.launch(failed, this, keepAlive) {
            try { throw java.io.IOException("synthetic failure") }
            catch (_: java.io.IOException) { observedFailure = true }
        }
        failed.job!!.join()
        assertTrue(observedFailure)
        assertFalse(generations.active)
        assertTrue(tracker.active.value.isEmpty())
    }
}
