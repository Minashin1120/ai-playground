package com.minashin1120.aiplayground.data

/*
 * Coding Mode target resolution, ported from chat_core part03 (`extractPromptCodingTargets`,
 * `collectCodingCandidates`, `resolveCodingTarget`, `syncCodingModeUi`) and the send step of part14.
 */

/** Largest code (single target, or all prompt candidates) the Web sends in Coding Mode. */
const val CODING_MAX_CHARS = 300_000

data class CodingCandidate(
    val candidateId: String,
    val code: String,
    val language: String,
    val key: String,
    val messageId: String? = null,
    val promptSource: Boolean = false,
    val promptIndex: Int? = null,
    val explicit: Boolean = false,
)

/** Closed ``` / ~~~ fences typed in the prompt; unfinished fences are ignored like on Web. */
fun extractPromptCodingTargets(prompt: String): List<CodingCandidate> {
    val completed = mutableListOf<CodingCandidate>()
    var markerChar = ' '
    var markerLength = 0
    var language = ""
    var buffer: MutableList<String>? = null
    for (line in prompt.replace("\r\n", "\n").replace('\r', '\n').split('\n')) {
        val active = buffer
        if (active == null) {
            val opening = Regex("^\\s*(`{3,}|~{3,})(.*)$").find(line) ?: continue
            markerChar = opening.groupValues[1][0]
            markerLength = opening.groupValues[1].length
            val info = opening.groupValues[2].trim()
            language = (info.split(Regex("\\s+")).firstOrNull().orEmpty().ifEmpty { "text" })
                .replace(Regex("^\\{?\\.?"), "").replace(Regex("\\}$"), "").ifEmpty { "text" }
            buffer = mutableListOf()
            continue
        }
        val trimmed = line.trim()
        if (trimmed.length >= markerLength && trimmed.all { it == markerChar }) {
            val code = active.joinToString("\n")
            if (code.isNotBlank()) completed += CodingCandidate(
                candidateId = "prompt-${completed.size + 1}", code = code, language = language,
                key = codingTargetKey("prompt", language, code), promptSource = true, promptIndex = completed.size,
            )
            buffer = null
            continue
        }
        active += line
    }
    return completed
}

/** Code blocks of assistant answers that show the Web "編集対象に指定" button (not `diff`). */
fun historyCodingTargets(messages: List<ChatMessage>): List<CodingCandidate> =
    messages.filter { it.role != "user" }.flatMap { message ->
        extractPromptCodingTargets(message.content)
            .filter { it.language.lowercase() !in setOf("diff", "pyexec", "chat_error") }
            .map { it.copy(key = codingTargetKey(it.language, it.code), messageId = message.id, promptSource = false, promptIndex = null) }
    }

fun codingTargetKey(language: String, code: String): String = "$language:${code.hashCode()}"

private fun codingTargetKey(prefix: String, language: String, code: String): String = "$prefix:$language:${code.hashCode()}"

/** Web `collectCodingCandidates`: an explicit selection wins; otherwise prompt fences, then up to 20 history blocks. */
fun collectCodingCandidates(prompt: String, history: List<CodingCandidate>, selection: CodingTarget?): List<CodingCandidate> {
    if (selection != null) return listOf(selection.asCandidate())
    val candidates = extractPromptCodingTargets(prompt).toMutableList()
    val seen = candidates.map { "${it.language}\n${it.code}" }.toMutableSet()
    val historyTargets = history.filter { seen.add("${it.language}\n${it.code}") }
    historyTargets.takeLast(20).forEachIndexed { index, target ->
        candidates += target.copy(candidateId = "history-${index + 1}", explicit = false)
    }
    return candidates
}

/** Web `resolveCodingTarget`: selection, else the last prompt fence, else the latest history block. */
fun resolveCodingTarget(prompt: String, history: List<CodingCandidate>, selection: CodingTarget?): CodingCandidate? =
    selection?.asCandidate() ?: extractPromptCodingTargets(prompt).lastOrNull() ?: history.lastOrNull()

private fun CodingTarget.asCandidate() = CodingCandidate(
    candidateId = "selected-1", code = code, language = language, key = id,
    messageId = messageId.ifBlank { null }, explicit = true,
)

/** Text of `#coding-target-text`. */
fun codingBarText(prompt: String, history: List<CodingCandidate>, selection: CodingTarget?): String {
    val target = resolveCodingTarget(prompt, history, selection)
    val candidates = if (selection != null) listOfNotNull(target) else collectCodingCandidates(prompt, history, null)
    return when {
        selection != null && target != null -> "編集対象: ${target.language.ifBlank { "text" }} コードブロック"
        candidates.size > 1 -> {
            val promptCount = candidates.count { it.promptSource }
            "モデルが編集対象を判断: 入力${promptCount}件 / 履歴${candidates.size - promptCount}件"
        }
        target != null && target.promptSource -> "入力中: ${target.language.ifBlank { "text" }} コードブロック"
        target != null -> "自動選択: 最新の ${target.language.ifBlank { "text" }} コードブロック"
        else -> "コードブロック生成後に自動有効化"
    }
}

/** What the send step decided: the payload candidates, or the toast that stops the send. */
data class CodingSendPlan(val target: CodingCandidate?, val candidates: List<CodingCandidate>, val error: String? = null) {
    val active: Boolean get() = error == null && target != null && target.code.isNotBlank()
}

/** Web part14: size limits, history budget, and the text-model check. */
fun planCodingSend(prompt: String, history: List<CodingCandidate>, selection: CodingTarget?, model: String): CodingSendPlan {
    val all = collectCodingCandidates(prompt, history, selection)
    val promptCandidates = all.filter { it.promptSource }
    val historyCandidates = all.filterNot { it.promptSource }
    val promptChars = promptCandidates.sumOf { it.code.length }
    if (promptChars > CODING_MAX_CHARS) return CodingSendPlan(null, emptyList(), "入力内の編集候補コード合計が大きすぎます（上限300,000文字）")
    var remaining = CODING_MAX_CHARS - promptChars
    val selectedHistory = ArrayDeque<CodingCandidate>()
    for (candidate in historyCandidates.asReversed()) {
        if (candidate.code.length > remaining) continue
        selectedHistory.addFirst(candidate)
        remaining -= candidate.code.length
    }
    val candidates = if (selection != null) selectedHistory.toList().takeLast(1) else promptCandidates + selectedHistory
    val target = if (selection != null) candidates.firstOrNull() else promptCandidates.lastOrNull() ?: candidates.lastOrNull()
    val effective = target != null && target.code.isNotBlank()
    if (effective && target!!.code.length > CODING_MAX_CHARS) return CodingSendPlan(null, emptyList(), "編集対象コードが大きすぎます（上限300,000文字）")
    if (effective && Regex("(image|video|tts|audio|native-audio)").containsMatchIn(model.lowercase())) {
        return CodingSendPlan(null, emptyList(), "Coding Modeではテキスト生成モデルを選択してください")
    }
    return CodingSendPlan(target, candidates)
}
