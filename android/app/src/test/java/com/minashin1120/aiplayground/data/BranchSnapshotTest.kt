package com.minashin1120.aiplayground.data

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class BranchSnapshotTest {
    private fun row(id: Int, parent: Int?, stamp: Long = id.toLong()) = JSONObject()
        .put("id", id).put("parent_id", parent ?: JSONObject.NULL).put("created_ms", stamp)

    @Test fun includesOtherRoutesBeforeTheChosenMessage() {
        val rows = listOf(row(1, null), row(2, 1), row(3, 1), row(4, 2), row(5, 3), row(6, 4), row(7, 1, 4))
        assertEquals(listOf(1, 2, 3, 4), branchSnapshot(rows, 4).map { it.getInt("id") })
    }

    @Test(expected = IllegalStateException::class) fun rejectsOrphans() {
        branchSnapshot(listOf(row(2, 1)), 2)
    }

    @Test(expected = IllegalStateException::class) fun rejectsRunningAnswers() {
        branchSnapshot(listOf(row(1, null).put("generating", true)), 1)
    }

    @Test fun generatedLocalFileReferencesIncludeTheExtension() {
        val row = JSONObject().put("content", "<audio controls src=\"local/abcdef1234.wav\"></audio>")
        assertEquals(setOf("local/abcdef1234.wav"), messageFileReferences(row))
        assertEquals("ファイルがありません", missingFileMarkup(row.getString("content"), setOf("local/abcdef1234.wav")))
    }

    @Test fun missingFileMarkupKeepsCodeAndStoredReferences() {
        val text = "[document](/files/1/a.pdf)\n`[document](/files/1/a.pdf)`"
        assertEquals("ファイルがありません\n`[document](/files/1/a.pdf)`", missingFileMarkup(text, setOf("1/a.pdf")))
        assertEquals(setOf("1/a.pdf"), messageFileReferences(JSONObject().put("content", text)))
    }
}
