package com.minashin1120.aiplayground.data

import org.json.JSONObject

/** One Python run saved in an answer (```` ```pyexec {"code","output"} ```` fence). */
data class PythonExecution(val code: String, val output: String, val images: List<String> = emptyList())

fun pythonExecutionMarkdown(run: PythonExecution): String = "\n```pyexec\n" + JSONObject()
    .put("code", run.code).put("output", run.output).put("images", org.json.JSONArray(run.images))
    .toString().replace("\\/", "/") + "\n```\n"

private val AGENTIC_VIEW = Regex("!\\[Agentic View]\\(([^)]+)\\)")

private fun normalizeNewlines(text: String) = text.replace("\r\n", "\n").replace('\r', '\n')

private val PYEXEC = Regex("(?:^|\\n)(`{3,}|~{3,})pyexec[ \\t]*\\n([\\s\\S]*?)\\n\\1[ \\t]*(?=\\n|$)")

/** Web `stripExactFencedBlock`: removes the fenced copy of [body] the model also printed. */
private fun stripExactFencedBlock(text: String, language: String, body: String): String {
    var result = normalizeNewlines(text)
    val bodyStr = normalizeNewlines(body)
    val langs = if (language.isNotEmpty()) listOf(language, "") else listOf("")
    for (fenceChar in listOf('`', '~')) for (n in 3..10) {
        val fence = fenceChar.toString().repeat(n)
        for (lang in langs) {
            val candidate = "$fence$lang\n$bodyStr\n$fence"
            if (candidate in result) result = result.split(candidate).joinToString("")
        }
    }
    return result
}

/** Web `stripVisiblePythonOutputBlock`: removes a printed "**Output:**" block with the same output. */
private fun stripVisiblePythonOutputBlock(text: String, output: String): String {
    var result = normalizeNewlines(text)
    val out = normalizeNewlines(output)
    for (prefix in listOf("**Output:**\n", "**Output:** \n", "**Output:**")) for (fenceChar in listOf('`', '~')) for (n in 3..10) {
        val fence = fenceChar.toString().repeat(n)
        listOf("$prefix$fence\n$out\n$fence", "$prefix\n$fence\n$out\n$fence", "\n$prefix$fence\n$out\n$fence", "\n$prefix\n$fence\n$out\n$fence")
            .forEach { candidate -> if (candidate in result) result = result.split(candidate).joinToString("\n") }
    }
    return result
}

private fun tidy(text: String) = text.replace(Regex("[ \\t]+\\n"), "\n").replace(Regex("\\n{3,}"), "\n\n")
    .replace(Regex("^\\n+"), "").replace(Regex("\\n+$"), "")

/**
 * Web `extractPythonExecutionsFromContent`: takes the runs out of the answer (they open from the footer's
 * "Python" button) along with the copies of their code and output the model repeated.
 */
fun extractPythonExecutions(raw: String): Pair<String, List<PythonExecution>> {
    val source = normalizeNewlines(raw)
    if (source.isEmpty()) return "" to emptyList()
    val runs = mutableListOf<PythonExecution>()
    val runOffsets = mutableListOf<Int>()
    var cleaned = PYEXEC.replace(source) { match ->
        runOffsets += match.range.first
        val body = match.groupValues[2].trim()
        runs += runCatching {
            val obj = JSONObject(body)
            PythonExecution(if (obj.isNull("code")) "" else obj.opt("code")?.toString().orEmpty(),
                if (obj.isNull("output")) "" else obj.opt("output")?.toString().orEmpty(),
                obj.optJSONArray("images")?.let { images -> (0 until images.length()).mapNotNull { images.optString(it).takeIf(String::isNotBlank) } }.orEmpty())
        }.getOrElse { PythonExecution(body, "") }
        "\n"
    }
    // Older direct answers saved only the visible code and Agentic View images.
    // Ordinary Python examples without execution images remain in the answer.
    if (runs.isEmpty() && AGENTIC_VIEW.containsMatchIn(source)) {
        val legacy = Regex("(?:^|\\n)```python\\n([\\s\\S]*?)\\n```(?:\\n+\\*\\*Output:\\*\\*\\n```\\n([\\s\\S]*?)\\n```)?")
        cleaned = legacy.replace(source) { match ->
            runOffsets += match.range.first
            runs += PythonExecution(match.groupValues[1], match.groupValues[2]); "\n"
        }
    }
    if (runs.isNotEmpty()) {
        val known = runs.flatMap { it.images }.toSet()
        // Use positions in the original answer; removing code/metadata changes the offsets.
        AGENTIC_VIEW.findAll(source).forEach { match ->
            val image = match.groupValues[1]
            if (image !in known) {
                val index = runOffsets.indexOfLast { it <= match.range.first }.coerceAtLeast(0)
                val run = runs[index]
                runs[index] = run.copy(images = (run.images + image).distinct())
            }
        }
        cleaned = AGENTIC_VIEW.replace(cleaned, "")
    }
    runs.forEach { run ->
        if (run.code.isNotEmpty()) {
            cleaned = stripExactFencedBlock(cleaned, "python", run.code)
            cleaned = stripExactFencedBlock(cleaned, "py", run.code)
        }
        cleaned = stripVisiblePythonOutputBlock(cleaned, run.output)
    }
    runs.lastOrNull { it.images.isNotEmpty() }?.let { lastRun ->
        val images = lastRun.images.filterNot { cleaned.contains(it) }
        if (images.isNotEmpty()) cleaned += "\n\n**Pythonの最後の実行画像**\n\n" +
            images.joinToString("\n\n") { "![Pythonの最後の実行画像]($it)" }
    }
    return tidy(cleaned) to runs
}

private val MCP_NOTE = Regex("^>\\s*(?:🔧|🚫)\\s*\\*\\*MCPツール実行(?:[:：]|は|（)")

/** Web `extractMcpExecutionNotesFromContent` + `appendMcpExecutionNotes`: MCP notices go after the prose. */
fun moveMcpNotesToEnd(text: String): String {
    val source = normalizeNewlines(text)
    if (source.isEmpty()) return ""
    val notes = mutableListOf<String>()
    val kept = source.split('\n').filter { line -> if (MCP_NOTE.containsMatchIn(line)) { notes += line.trim(); false } else true }
    val body = tidy(kept.joinToString("\n")).trim()
    if (notes.isEmpty()) return body
    return if (body.isNotEmpty()) "$body\n\n${notes.joinToString("\n")}" else notes.joinToString("\n")
}

/** Web message `batch_job`: the provider Batch state shown above the answer. */
data class BatchInfo(val state: String, val statusText: String = "")

fun parseBatchInfo(row: JSONObject): BatchInfo? {
    val job = row.optJSONObject("batch_job") ?: return null
    return BatchInfo(job.optString("state").uppercase(), if (job.isNull("status_text")) "" else job.optString("status_text"))
}

/** Web `batchStatusHtml` wording. */
fun batchStatusText(info: BatchInfo): String = info.statusText.ifBlank {
    when (info.state) {
        "JOB_STATE_SUCCEEDED" -> "Batch処理が完了しました"
        "JOB_STATE_FAILED" -> "Batch処理に失敗しました"
        else -> "Batch APIで処理中です"
    }
}
