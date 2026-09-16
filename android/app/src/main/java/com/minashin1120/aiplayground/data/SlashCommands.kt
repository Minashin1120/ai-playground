package com.minashin1120.aiplayground.data

data class SlashCommand(
    val id: String,
    val label: String,
    val description: String,
    val argument: String? = null,
)

data class SlashAction(
    val id: String,
    val argument: String = "",
    val consumeDraft: Boolean = true,
)

val SLASH_COMMANDS = listOf(
    SlashCommand("settings", "/settings", "設定を開く"),
    SlashCommand("options", "/options", "詳細設定を開く"),
    SlashCommand("attach", "/attach", "ファイル添付を開く"),
    SlashCommand("voice", "/voice", "音声入力を開始"),
    SlashCommand("paste", "/paste", "リッチ貼り付けを開く"),
    SlashCommand("canvas", "/canvas", "Canvasを切り替える"),
    SlashCommand("coding", "/coding", "Codingを切り替える"),
    SlashCommand("search", "/search", "Searchを切り替える"),
    SlashCommand("urls", "/urls", "URLsを切り替える"),
    SlashCommand("maps", "/maps", "Mapsを切り替える"),
    SlashCommand("python", "/python", "Pythonを切り替える"),
    SlashCommand("file", "/file", "Fileを切り替える"),
    SlashCommand("mcp", "/mcp", "MCPを切り替える"),
    SlashCommand("sysprompt", "/sysprompt", "SysPromptを切り替える"),
    SlashCommand("thinking", "/thinking", "Thinkingを指定（off / min / low / mid / high）"),
    SlashCommand("effort", "/effort", "Effortを指定（none / low / medium / high / xhigh / max）"),
    SlashCommand("safety", "/safety", "Safetyを指定（default / none）"),
    SlashCommand("promptcache", "/promptcache", "PromptCacheを切り替える"),
    SlashCommand("compress", "/compress", "画像圧縮を切り替える"),
    SlashCommand("tempchat", "/tempchat", "一時チャットを切り替える"),
    SlashCommand("realtime", "/realtime", "Realtime音声を開く"),
    SlashCommand("lyria", "/lyria", "Lyria音楽を開く"),
)

fun slashToken(draft: String): Pair<String, String>? {
    if (!draft.startsWith("/")) return null
    if (draft.contains('\n')) return null
    val body = draft.drop(1)
    val space = body.indexOf(' ')
    val command = if (space < 0) body else body.substring(0, space)
    if (command.isEmpty() || !command.all { it.isLetter() }) return null
    val argument = if (space < 0) "" else body.substring(space + 1).trim()
    return command.lowercase() to argument
}

fun matchingSlashCommands(draft: String): List<SlashCommand> {
    val token = slashToken(draft) ?: return emptyList()
    val (command, argument) = token
    if (argument.isNotEmpty()) {
        return SLASH_COMMANDS.filter { it.id == command }
    }
    return SLASH_COMMANDS.filter { it.id.startsWith(command) || it.label.drop(1).startsWith(command) }
}

fun parseSlashAction(draft: String): SlashAction? {
    val token = slashToken(draft) ?: return null
    val (command, argument) = token
    val match = SLASH_COMMANDS.firstOrNull { it.id == command } ?: return null
    if (match.id in listOf("thinking", "effort", "safety") && argument.isBlank()) return null
    return SlashAction(match.id, argument)
}
