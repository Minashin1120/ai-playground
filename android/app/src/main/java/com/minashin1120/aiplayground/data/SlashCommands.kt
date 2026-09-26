package com.minashin1120.aiplayground.data

/**
 * A Web slash command (`SLASH_COMMANDS` / `MINIMAL_SLASH_COMMANDS`). [itemKey] names the minimal-mode
 * popup item it drives; [iconName] is the Web `fa-*` class without the prefix.
 */
data class SlashCommand(
    val id: String,
    val label: String,
    val description: String,
    val iconName: String,
    val minimal: Boolean = true,
    val itemKey: String = "",
    val requiresArgument: Boolean = false,
    val autocompleteArgument: Boolean = false,
    val presetArgument: String = "",
    val argumentHint: String = "",
)

private fun minimal(id: String, label: String, description: String, icon: String, itemKey: String = id,
                    requiresArgument: Boolean = false, autocomplete: Boolean = false, preset: String = "", hint: String = "") =
    SlashCommand(id, label, description, icon, true, itemKey, requiresArgument, autocomplete, preset, hint)

/**
 * The Web commands in order. `/fast` is left out: the Android app has no browser fast mode
 * (ANDROID_ONLY.md).
 */
val SLASH_COMMANDS = listOf(
    SlashCommand("settings", "/settings", "AIで自然言語を使って設定を変更（現在選択中のモデルを使用）", "cog", minimal = false),
    minimal("options", "/options", "＋メニューを開く", "plus", itemKey = ""),
    minimal("attach", "/attach", "ファイル添付を開く", "paperclip"),
    minimal("voice", "/voice", "Voice Inputを開始・停止", "microphone", itemKey = "voice-input"),
    minimal("paste", "/paste", "リッチ貼り付けを開く", "paste", itemKey = "rich-paste"),
    minimal("canvas", "/canvas", "Canvasを切り替える（on / off）", "window-restore"),
    minimal("coding", "/coding", "Codingを切り替える（on / off）", "code-branch"),
    minimal("search", "/search", "Searchを切り替える（on / off）", "search"),
    minimal("urls", "/urls", "URLsを切り替える（on / off）", "link"),
    minimal("maps", "/maps", "Mapsを切り替える（on / off）", "map-location-dot"),
    minimal("python", "/python", "Pythonを切り替える（on / off）", "code"),
    minimal("file", "/file", "Fileを切り替える（on / off）", "file-lines"),
    minimal("mcp", "/mcp", "MCPを切り替える（on / off）", "plug"),
    minimal("sysprompt", "/sysprompt", "SysPromptを切り替える（on / off）", "terminal"),
    minimal("thinking", "/thinking", "Thinkingの値を選択（off / min / low / mid / high）", "brain", requiresArgument = true, autocomplete = true),
    minimal("thinking-off", "/thinking off", "ThinkingをOFFにする", "brain", itemKey = "thinking", preset = "off"),
    minimal("thinking-min", "/thinking min", "ThinkingをMinにする", "brain", itemKey = "thinking", preset = "min"),
    minimal("thinking-low", "/thinking low", "ThinkingをLowにする", "brain", itemKey = "thinking", preset = "low"),
    minimal("thinking-mid", "/thinking mid", "ThinkingをMidにする", "brain", itemKey = "thinking", preset = "mid"),
    minimal("thinking-high", "/thinking high", "ThinkingをHighにする", "brain", itemKey = "thinking", preset = "high"),
    minimal("effort", "/effort", "Effortを調整", "sliders-h", requiresArgument = true,
        hint = "Effortを入力（none / low / medium / high / xhigh / max）..."),
    minimal("safety", "/safety", "Safetyを調整", "shield-halved", requiresArgument = true, hint = "Safetyを入力（default / none）..."),
    minimal("promptcache", "/promptcache", "PromptCacheを切り替える（on / off）", "database"),
    minimal("compress", "/compress", "Compressを切り替える（on / off）", "compress-alt"),
    minimal("tempchat", "/tempchat", "一時チャットを切り替える（on / off）", "hourglass-half"),
)

/** Web `extractSlashCommandToken`: the command word after a leading `/`, or null when the text is not a command. */
fun slashCommandToken(value: String): String? {
    val trimmed = value.trimStart()
    if (!trimmed.startsWith("/")) return null
    val token = trimmed.substring(1).split(Regex("\\s+")).firstOrNull().orEmpty()
    return Regex("^[a-z][\\w-]*", RegexOption.IGNORE_CASE).find(token)?.value ?: token
}

/** Web `slashCommandSuggestionFilter`: `/thinking <arg>` filters the thinking presets by their argument. */
fun slashSuggestionFilter(value: String): String {
    val token = slashCommandToken(value) ?: return ""
    if (!token.equals("thinking", ignoreCase = true)) return token
    val match = Regex("^/thinking(\\s+.*)$", RegexOption.IGNORE_CASE).find(value.trimStart())
    return match?.let { "thinking${it.groupValues[1]}".lowercase() } ?: token
}

/** Web `visibleSlashCommands`: minimal-mode commands only in minimal mode; label or description contains the filter. */
fun visibleSlashCommands(filter: String, minimalMode: Boolean): List<SlashCommand> {
    val normalized = filter.lowercase()
    return SLASH_COMMANDS.filter { command ->
        (!command.minimal || minimalMode) &&
            (command.label.lowercase().contains(normalized) || command.description.lowercase().contains(normalized))
    }
}

/** The palette shows while the input starts with `/` (Web input handler). */
fun slashPaletteFilter(draft: String): String? = if (draft.trim().startsWith("/")) slashSuggestionFilter(draft) else null

/**
 * Web `selectSlashCommand`: the text left in the input after the command word is taken out
 * (text typed right after the command without a space stays as its argument).
 */
fun stripSlashCommand(draft: String): String {
    val token = slashCommandToken(draft)
    if (token != null) return draft.trimStart().substring(1 + token.length).trimStart()
    val lastSlash = draft.lastIndexOf('/')
    return if (lastSlash >= 0) draft.substring(0, lastSlash).trimEnd() else ""
}

/** Web `parseSlashToggleArgument` results. */
enum class SlashToggle { ON, OFF, TOGGLE, INVALID }

fun parseSlashToggle(argument: String): SlashToggle {
    val value = argument.trim().lowercase()
    return when {
        value.isEmpty() || value in setOf("toggle", "切替", "切り替え") -> SlashToggle.TOGGLE
        value in setOf("on", "true", "1", "オン", "有効") -> SlashToggle.ON
        value in setOf("off", "false", "0", "オフ", "無効") -> SlashToggle.OFF
        else -> SlashToggle.INVALID
    }
}

/** Web thinking-level aliases for `/thinking <level>`. */
val SLASH_THINKING_LEVELS = mapOf("min" to "minimal", "minimal" to "minimal", "low" to "low", "mid" to "medium", "medium" to "medium", "high" to "high")
