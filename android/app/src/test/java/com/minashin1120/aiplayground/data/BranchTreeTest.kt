package com.minashin1120.aiplayground.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class BranchTreeTest {
    private fun msg(id: Int, parent: Int?, model: String = "m", tokens: Int = 10) =
        ChatMessage(id.toString(), if (id % 2 == 1) "user" else "assistant", "", parentId = parent, model = model, tokens = tokens)

    @Test
    fun parentIsCenteredOverSideBySideChildren() {
        val messages = listOf(msg(1, null), msg(2, 1), msg(3, 1))
        val layout = layoutBranchTree(messages, 0f) { _, _ -> 120f }
        val byId = layout.nodes.associateBy { it.id }
        assertEquals(256f, layout.width) // 120 + 16 + 120
        assertEquals(byId.getValue(2).y, byId.getValue(3).y)
        assertEquals(136f, byId.getValue(3).x - byId.getValue(2).x)
        val parentCenter = byId.getValue(1).x + 60f
        assertEquals((byId.getValue(2).x + byId.getValue(3).x + 120f) / 2f, parentCenter)
        assertTrue(byId.getValue(1).hasChildren)
        assertFalse(byId.getValue(2).hasChildren)
    }

    @Test
    fun pathTokensAndBreakdownFollowParents() {
        val messages = listOf(msg(1, null, "a", 5), msg(2, 1, "b", 7), msg(3, 2, "a", 11))
        assertEquals(23, branchPathTokens(messages, 3))
        val layout = layoutBranchTree(messages, 400f) { _, _ -> 120f }
        assertEquals(23, layout.nodes.first { it.id == 3 }.pathTokens)
        val breakdown = branchModelBreakdown(messages, 3)
        assertEquals(listOf("a", "b"), breakdown.map { it.first })
        assertEquals(16, breakdown.first().second[0])
    }

    @Test
    fun longLinearChatDoesNotRecurse() {
        val messages = (1..3000).map { msg(it, if (it == 1) null else it - 1) }
        val layout = layoutBranchTree(messages, 300f) { _, _ -> 120f }
        assertEquals(3000, layout.nodes.size)
    }

    @Test
    fun orphansAreSkippedLikeWeb() {
        val layout = layoutBranchTree(listOf(msg(1, null), msg(5, 99)), 0f) { _, _ -> 120f }
        assertEquals(listOf(1), layout.nodes.map { it.id })
    }

    @Test
    fun libraryHelpersMatchWeb() {
        fun file(name: String, ts: Long) = LibraryFile(name, "u/$name", "", "", "file", "", false, ts)
        val files = listOf(file("b.txt", 1), file("a.txt", 2), file("c.txt", 3))
        assertEquals(listOf("c.txt", "a.txt", "b.txt"), sortLibraryFiles(files, "newest").map { it.displayName })
        assertEquals(listOf("b.txt", "a.txt", "c.txt"), sortLibraryFiles(files, "oldest").map { it.displayName })
        assertEquals(listOf("a.txt", "b.txt", "c.txt"), sortLibraryFiles(files, "name_asc").map { it.displayName })
        assertEquals(listOf("c.txt", "b.txt", "a.txt"), sortLibraryFiles(files, "name_desc").map { it.displayName })
        assertEquals(true to true, modelMediaSupport("gemini-3.6-flash"))
        assertEquals(false to false, modelMediaSupport("gemini-3.1-flash-image"))
        assertEquals(false to false, modelMediaSupport("gpt-5.5"))
        assertTrue(isAudioPath("u/voice.M4A"))
        assertTrue(isVideoPath("u/clip.mov"))
        assertFalse(isVideoPath("u/doc.pdf"))
    }

    @Test
    fun createdAtIsShownLikeTheWebBranchDetail() {
        val tokyo = java.time.ZoneId.of("Asia/Tokyo")
        assertEquals("2026/09/27 12:04", branchCreatedAt("2026-09-27T03:04:05Z", tokyo))
        assertEquals("-", branchCreatedAt(""))
        assertEquals("not a date", branchCreatedAt("not a date"))
        val parsed = parseMessages(org.json.JSONObject("""{"messages":[{"id":1,"role":"user","content":"hi","created_at":"2026-09-27T03:04:05Z"}]}"""))
        assertEquals("2026-09-27T03:04:05Z", parsed.single().createdAt)
    }
}
