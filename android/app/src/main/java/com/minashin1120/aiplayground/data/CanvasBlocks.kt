package com.minashin1120.aiplayground.data

/** One fenced block shown in the Canvas panel (Web `normalizeCanvasBlock`); [open] while its fence is still streaming. */
data class CanvasBlock(val index: Int, val lang: String, val code: String, val open: Boolean) {
    /** Web `hashString(lang || 'TEXT' + '\n' + code)`: identifies the block across re-renders. */
    val key: String get() = "${lang.ifEmpty { "TEXT" }}\n$code".hashCode().toString()
}

/** Web `parseCanvasMarkdown` result: the answer with each fence replaced by [CANVAS_PLACEHOLDER_LINE], and the fences. */
data class CanvasData(val renderText: String, val blocks: List<CanvasBlock>)

/**
 * The line that stands for a fenced block in a Canvas-mode answer (Web
 * `<div class="canvas-code-placeholder">Canvasで表示中</div>`).
 */
const val CANVAS_PLACEHOLDER_LINE = "⁣canvas-code-placeholder⁣"
const val CANVAS_PLACEHOLDER_TEXT = "Canvasで表示中"

private val CANVAS_FENCE_START = Regex("^(\\s*)(`{3,}|~{3,})(.*)$")

/** Web `parseCanvasMarkdown`: every fence (any indent) becomes a block; an unclosed fence stays [CanvasBlock.open]. */
fun parseCanvasMarkdown(text: String): CanvasData {
    val output = mutableListOf<String>()
    val blocks = mutableListOf<Triple<String, MutableList<String>, Boolean>>()
    var fence: String? = null
    text.split(Regex("\r?\n")).forEach { line ->
        val active = fence
        if (active == null) {
            val match = CANVAS_FENCE_START.matchEntire(line)
            if (match != null) {
                fence = match.groupValues[2]
                blocks += Triple(match.groupValues[3].trim(), mutableListOf(), true)
                output += CANVAS_PLACEHOLDER_LINE
            } else output += line
            return@forEach
        }
        val trimmed = line.trim()
        if (trimmed.isNotEmpty() && trimmed.replace(Regex("\\s+"), "") == active) {
            val last = blocks.removeAt(blocks.lastIndex)
            blocks += last.copy(third = false)
            fence = null
        } else blocks.last().second += line
    }
    return CanvasData(
        output.joinToString("\n"),
        blocks.mapIndexed { index, (lang, lines, open) -> CanvasBlock(index, lang, lines.joinToString("\n"), open) },
    )
}

/** Web `isCanvasHtmlPreviewCandidate`. */
fun isCanvasHtml(lang: String, code: String): Boolean {
    val token = lang.trim().lowercase()
    if (token == "html" || token == "htm" || token == "xhtml") return true
    if (token.isNotEmpty()) return false
    return Regex("<!doctype\\s+html", RegexOption.IGNORE_CASE).containsMatchIn(code) ||
        Regex("<html[\\s>]", RegexOption.IGNORE_CASE).containsMatchIn(code)
}

/**
 * The Canvas panel state (Web `canvasPreviewState`): the blocks of the latest answer, which one is shown,
 * and whether the user picked it ([manual] keeps the choice while it exists; otherwise the last block).
 */
data class CanvasSelection(val index: Int = -1, val manual: Boolean = false)

/** Web `updateCanvasPreviewState`. */
fun nextCanvasSelection(blocks: List<CanvasBlock>, previous: CanvasSelection): CanvasSelection = when {
    blocks.isEmpty() -> CanvasSelection()
    previous.manual && previous.index in blocks.indices -> previous
    else -> CanvasSelection(blocks.lastIndex)
}

/** Web `refreshCanvasPreviewPanel` title. */
fun canvasTitle(blocks: List<CanvasBlock>, index: Int): String {
    val block = blocks.getOrNull(index) ?: return CANVAS_PLACEHOLDER_TEXT
    val position = if (blocks.size > 1) " #${index + 1}/${blocks.size}" else ""
    return if (isCanvasHtml(block.lang, block.code)) "HTML Canvas Preview$position"
        else "Canvas Preview: ${block.lang.ifEmpty { "text" }}$position"
}

/**
 * Web `refreshCanvasPreviewPanel` status. HTML is not run on Android (ANDROID_ONLY.md), so it gets the
 * code wording rather than "HTML をリアルタイムでプレビューしています".
 */
fun canvasStatus(block: CanvasBlock?): String = when {
    block == null -> "コードブロックを待機中"
    block.open -> "コードブロックを生成中"
    else -> "コードブロックをプレビューしています"
}

/** Web `renderCanvasBlockChips` preview: the first non-blank line, whitespace collapsed, at most 120 characters. */
fun canvasBlockPreview(block: CanvasBlock): String =
    (block.code.split(Regex("\r?\n")).firstOrNull { it.isNotBlank() } ?: "空のコードブロック").trim().replace(Regex("\\s+"), " ").take(120)
