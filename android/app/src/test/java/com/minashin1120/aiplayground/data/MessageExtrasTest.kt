package com.minashin1120.aiplayground.data

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class MessageExtrasTest {
    @Test fun executionImagesStayInDetailsAndOnlyTheLastImageAppearsBelowTheAnswer() {
        val run = PythonExecution("display(crop)", "", listOf("/files/1/step.png", "/files/1/result.png"))
        val (text, runs) = extractPythonExecutions("完了。" + pythonExecutionMarkdown(run))
        assertEquals(listOf(run), runs)
        assertEquals("完了。\n\n**Pythonの最後の画像**\n\n![Pythonの最後の画像](/files/1/result.png)", text)
        val (cited, _) = extractPythonExecutions("![結果](/files/1/result.png)" + pythonExecutionMarkdown(run))
        assertEquals("![結果](/files/1/result.png)", cited)
    }

    @Test fun olderDirectPythonAnswersAreRecoveredWithoutHidingOrdinaryExamples() {
        val (text, runs) = extractPythonExecutions("```python\ndisplay(crop)\n```\n**Output:**\n```\nok\n```\n答え。\n![Agentic View](/files/1/crop.png)")
        assertEquals(listOf(PythonExecution("display(crop)", "ok", listOf("/files/1/crop.png"))), runs)
        assertEquals("答え。\n\n**Pythonの最後の画像**\n\n![Pythonの最後の画像](/files/1/crop.png)", text)
        val plain = "```python\nprint(1)\n```"
        assertEquals(plain to emptyList<PythonExecution>(), extractPythonExecutions(plain))
    }

    @Test fun pythonRunsLeaveTheAnswerWithTheirCopies() {
        val raw = "計算します。\n```python\nprint(1)\n```\n**Output:**\n```\n1\n```\n```pyexec\n{\"code\": \"print(1)\", \"output\": \"1\"}\n```\n結果は1です。"
        val (text, runs) = extractPythonExecutions(raw)
        assertEquals(listOf(PythonExecution("print(1)", "1")), runs)
        assertEquals("計算します。\n\n結果は1です。", text)
        val (_, broken) = extractPythonExecutions("```pyexec\nnot json\n```")
        assertEquals(listOf(PythonExecution("not json", "")), broken)
    }

    @Test fun mcpNoticesMoveAfterTheProse() {
        val text = "> 🔧 **MCPツール実行: search**\n本文です。\n> 🚫 **MCPツール実行は拒否されました**"
        assertEquals("本文です。\n\n> 🔧 **MCPツール実行: search**\n> 🚫 **MCPツール実行は拒否されました**", moveMcpNotesToEnd(text))
        assertEquals("本文だけ", moveMcpNotesToEnd("本文だけ"))
    }

    @Test fun batchCardWording() {
        assertNull(parseBatchInfo(JSONObject("{\"batch_job\": null}")))
        val running = parseBatchInfo(JSONObject("{\"batch_job\": {\"state\": \"job_state_running\", \"status_text\": null}}"))!!
        assertEquals("Batch APIで処理中です", batchStatusText(running))
        assertEquals("Batch処理が完了しました", batchStatusText(BatchInfo("JOB_STATE_SUCCEEDED")))
        assertEquals("Batch処理に失敗しました", batchStatusText(BatchInfo("JOB_STATE_FAILED")))
        assertEquals("待機中", batchStatusText(BatchInfo("JOB_STATE_PENDING", "待機中")))
    }
}
