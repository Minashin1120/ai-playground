package com.minashin1120.aiplayground.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ProgressTextTest {
    @Test fun requestLabelsFollowProgressSpinner() {
        assertEquals("読み込み中...", ProgressText.forRequest("GET", "/api/settings"))
        assertEquals("削除中...", ProgressText.forRequest("DELETE", "/api/threads/1"))
        assertEquals("削除中...", ProgressText.forRequest("POST", "/api/files/delete"))
        assertEquals("送信中...", ProgressText.forRequest("POST", "/api/feedback"))
        assertEquals("保存中...", ProgressText.forRequest("PUT", "/api/mobile/v1/preferences"))
        assertEquals("アップロード中...", ProgressText.forRequest("POST", "/upload"))
        assertEquals("生成中...", ProgressText.forRequest("POST", "/api/generate_title"))
        assertTrue(ProgressText.isPassive("/api/version"))
        assertTrue(ProgressText.isPassive("/api/temporary_chat/heartbeat"))
        assertFalse(ProgressText.isPassive("/api/versions-list"))
    }

    @Test fun trackerShowsTheNewestOperationUntilAllFinish() {
        val tracker = ProgressTracker()
        assertNull(tracker.label.value)
        val chat = tracker.startFlow("chat")
        assertEquals("送信中...", tracker.label.value)
        val load = tracker.start("読み込み中...")
        assertEquals("読み込み中...", tracker.label.value)
        load.finish()
        chat.setPhase("waiting")
        assertEquals("モデルの応答待機中...", tracker.label.value)
        chat.finish()
        chat.finish()
        assertNull(tracker.label.value)
        chat.setPhase("receiving")
        assertNull(tracker.label.value)
    }

    @Test fun pendingSkeletonAndReasoningMatchWeb() {
        assertEquals("image", pendingSkeletonKind("gpt-image-2"))
        assertEquals("video", pendingSkeletonKind("grok-imagine-video"))
        // Like the Web, Veo ids do not contain "video" and use the text skeleton.
        assertEquals("text", pendingSkeletonKind("veo-3.1-generate-preview"))
        assertEquals("audio", pendingSkeletonKind("gpt-4o-mini-tts"))
        assertEquals("code", pendingSkeletonKind("grok-code-fast-1"))
        assertEquals("text", pendingSkeletonKind("mistral-ocr-4-0"))
        assertEquals("text", pendingSkeletonKind("gemini-3.6-flash"))
        assertTrue(showsReasoningProgress("gemini-3.6-flash", true, ""))
        assertTrue(showsReasoningProgress("gpt-5.5", false, "high"))
        assertFalse(showsReasoningProgress("gpt-5.5", false, "none"))
        assertFalse(showsReasoningProgress("grok-4-fast-non-reasoning", true, ""))
        assertFalse(showsReasoningProgress("claude-opus-5", true, ""))
    }
}
