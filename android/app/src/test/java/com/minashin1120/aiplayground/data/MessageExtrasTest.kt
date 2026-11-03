package com.minashin1120.aiplayground.data

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class MessageExtrasTest {
    @Test fun generatedImageLinksDisplayInlineWithoutChangingOtherLinksOrCode() {
        val image = "/files/1/result.png"
        val run = PythonExecution("display(result)", "", listOf(image))
        val others = "[PDF](/files/1/result.pdf) [external](https://example.com/result.png)"
        val (text, _) = extractPythonExecutions("[切り抜いた画像をダウンロード]($image)\n" + others + pythonExecutionMarkdown(run))
        assertEquals("![切り抜いた画像をダウンロード]($image)\n$others", text)
        val code = "```md\n[example]($image)\n```\n`[example]($image)`"
        val (withCode, _) = extractPythonExecutions(code + pythonExecutionMarkdown(run))
        assertEquals(code + "\n\n**Pythonの最後の実行画像**\n\n![Pythonの最後の実行画像]($image)", withCode)
    }

    @Test fun allThreeImagesFromTheLastImageRunAppearBelowTheAnswer() {
        val step = PythonExecution("display(step)", "", listOf("/files/1/step.png"))
        val run = PythonExecution("display(a); display(b); display(c)", "", (1..3).map { "/files/1/result$it.png" })
        val raw = "完了。" + pythonExecutionMarkdown(step) + pythonExecutionMarkdown(run) +
            pythonExecutionMarkdown(PythonExecution("print('done')", "done"))
        val (text, runs) = extractPythonExecutions(raw)
        assertEquals(listOf(step, run, PythonExecution("print('done')", "done")), runs)
        assertEquals("完了。\n\n**Pythonの最後の実行画像**\n\n" +
            run.images.joinToString("\n\n") { "![Pythonの最後の実行画像]($it)" }, text)
        val explicit = run.images.joinToString("\n") { "![結果]($it)" }
        assertEquals(explicit, extractPythonExecutions(explicit + pythonExecutionMarkdown(step) + pythonExecutionMarkdown(run)).first)
    }

    @Test fun imagesAreAssociatedWithTheirPrecedingExecutionInLegacyAndStructuredAnswers() {
        for (structured in listOf(false, true)) {
            fun execution(code: String) = if (structured) pythonExecutionMarkdown(PythonExecution(code, ""))
                else "\n```python\n$code\n```\n**Output:**\n```\n\n```\n"
            val raw = execution("display(step)") + "![Agentic View](/files/1/step.png)\n" +
                execution("display(a); display(b); display(c)") +
                (1..3).joinToString("\n") { "![Agentic View](/files/1/final$it.png)" } + "\n完了。"
            val (text, runs) = extractPythonExecutions(raw)
            assertEquals(listOf(listOf("/files/1/step.png"), (1..3).map { "/files/1/final$it.png" }), runs.map { it.images })
            assertEquals("完了。\n\n**Pythonの最後の実行画像**\n\n" +
                (1..3).joinToString("\n\n") { "![Pythonの最後の実行画像](/files/1/final$it.png)" }, text)
        }
    }

    @Test fun olderDirectPythonAnswersAreRecoveredWithoutHidingOrdinaryExamples() {
        val (text, runs) = extractPythonExecutions("```python\ndisplay(crop)\n```\n**Output:**\n```\nok\n```\n答え。\n![Agentic View](/files/1/crop.png)")
        assertEquals(listOf(PythonExecution("display(crop)", "ok", listOf("/files/1/crop.png"))), runs)
        assertEquals("答え。\n\n**Pythonの最後の実行画像**\n\n![Pythonの最後の実行画像](/files/1/crop.png)", text)
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
