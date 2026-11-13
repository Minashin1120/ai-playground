package com.minashin1120.aiplayground

import com.minashin1120.aiplayground.data.ComposerRules
import com.minashin1120.aiplayground.data.CompressionSettings
import com.minashin1120.aiplayground.data.EFFORT_LEVELS
import com.minashin1120.aiplayground.data.OptionRule
import com.minashin1120.aiplayground.data.SLASH_EFFORT_THINKING_LEVELS
import com.minashin1120.aiplayground.data.SLASH_THINKING_LEVELS
import com.minashin1120.aiplayground.data.SlashCommand
import com.minashin1120.aiplayground.data.SlashToggle
import com.minashin1120.aiplayground.data.THINKING_LEVELS
import com.minashin1120.aiplayground.data.composerRules
import com.minashin1120.aiplayground.data.parseSlashToggle
import kotlinx.coroutines.flow.StateFlow

/**
 * 最小モードのスラッシュコマンド（Web `executeMinimalSlashCommand`）だけを持つ：PopUp項目の
 * ON/OFF・Thinking の段階・Safety・Compress・一時チャットの切替と、その案内文。
 * 状態の更新は [ChatComposer] と生成時に受け取った関数へ委ねる。
 */
internal class ChatSlashCommands(
    private val state: StateFlow<ChatState>,
    private val composer: ChatComposer,
    private val notify: (String) -> Unit,
    private val saveCompressionSettings: (CompressionSettings) -> Unit,
    private val toggleTemporaryChat: () -> Unit,
) {
    private class SlashItem(val label: String, val rule: OptionRule?, val checked: Boolean, val toggle: () -> Unit)

    private fun slashItem(key: String, current: ChatState, rules: ComposerRules): SlashItem? = when (key) {
        "canvas" -> SlashItem("Canvas", rules.canvas, current.canvasMode, composer::toggleCanvas)
        "coding" -> SlashItem("Coding", rules.coding, current.codingMode, composer::toggleCoding)
        "gemini-free" -> SlashItem("無料枠", rules.geminiFree, current.geminiFreeTier, composer::toggleGeminiFreeTier)
        "search" -> SlashItem("Search", rules.search, current.enableSearch, composer::toggleSearch)
        "urls" -> SlashItem("URLs", rules.urls, current.enableUrlContext, composer::toggleUrlContext)
        "maps" -> SlashItem("Maps", rules.maps, current.enableMaps, composer::toggleMaps)
        "python" -> SlashItem("Python", rules.python, current.enablePython, composer::togglePython)
        "file" -> SlashItem("File", rules.file, current.enableFileCreation, composer::toggleFileCreation)
        "mcp" -> SlashItem("MCP", rules.mcp, current.enableMcp, composer::toggleMcp)
        "sysprompt" -> SlashItem("SysPrompt", rules.sysPrompt, current.enableSystemPrompt, composer::toggleSystemPrompt)
        "thinking" -> SlashItem(
            "Thinking",
            rules.thinking,
            if (rules.effortAsThinking) current.chipValues["reasoning_effort"].orEmpty() != "none" else current.enableThinking,
        ) {
            if (rules.effortAsThinking) {
                val levels = EFFORT_LEVELS.filter { it in rules.thinkingLevels }
                val next = if (state.value.chipValues["reasoning_effort"].orEmpty() != "none") "none"
                else levels.firstOrNull { it != "none" } ?: rules.effortFallback
                composer.generationOption("reasoning_effort", next)
            } else composer.toggleThinking()
        }
        "promptcache" -> SlashItem("PromptCache", rules.promptCache, current.enablePromptCache, composer::togglePromptCache)
        "compress" -> SlashItem("Compress", null, current.compression.enabled) {
            saveCompressionSettings(state.value.compression.copy(enabled = !state.value.compression.enabled))
        }
        "tempchat" -> SlashItem("一時チャット", null, current.selected?.isTemporary ?: current.newThreadTemporary, toggleTemporaryChat)
        else -> null
    }

    /**
     * Web `executeMinimalSlashCommand`: runs a minimal-mode command with the Web notices. Popup-only actions
     * (＋ menu, attach, voice, rich paste) go to [onLocal]. Returns false when the argument was not usable.
     */
    fun run(command: SlashCommand, argument: String, onLocal: (String) -> Unit): Boolean {
        when (command.id) {
            "options", "attach", "voice", "paste" -> { onLocal(command.id); return true }
        }
        val current = state.value
        val rules = composerRules(current.model, current.mcpServers.any { it.enabled }, current.preferences?.geminiBackend ?: "gemini_api")
        val raw = command.presetArgument.ifBlank { argument }.trim()
        if (command.id == "safety") {
            if (raw.isEmpty()) {
                notify("使い方: ${command.label} default / none")
                return false
            }
            val options = listOf("default" to "Default", "none" to "None")
            val normalized = raw.lowercase()
            val option = options.firstOrNull { (value, label) -> value == normalized || label.lowercase() == normalized }
            if (option == null) { notify("${command.label}: 指定値「$raw」は利用できません"); return false }
            composer.generationOption("safety_setting", option.first)
            notify("Safety: ${option.second}")
            return true
        }
        val item = slashItem(command.itemKey, current, rules) ?: return false
        if (item.rule?.visible == false) { notify("/${command.id} は現在のモデルでは利用できません"); return true }
        val disabled = item.rule != null && (item.rule.disabled || item.rule.dimmed)
        if (disabled && command.itemKey != "thinking") { notify("/${command.id} は現在変更できません"); return true }
        if (command.itemKey == "thinking" && raw.isNotEmpty()) {
            val normalized = raw.lowercase()
            val thinkingKey = if (rules.effortAsThinking) "reasoning_effort" else "thinking_level"
            val availableLevels = (if (rules.effortAsThinking) EFFORT_LEVELS else THINKING_LEVELS)
                .filter { it in rules.thinkingLevels }
            val alias = if (rules.effortAsThinking) SLASH_EFFORT_THINKING_LEVELS[normalized] else SLASH_THINKING_LEVELS[normalized]
            if (alias != null) {
                if (alias !in availableLevels) {
                    notify("Thinking: 「$normalized」はこのモデルでは利用できません")
                    return false
                }
                if (alias == "none") {
                    if (item.rule?.disabled == true) { notify("Thinking: このモデルではOFFにできません"); return false }
                    composer.generationOption(thinkingKey, "none")
                } else {
                    if (!rules.effortAsThinking && !current.enableThinking && item.rule?.disabled != true) composer.toggleThinking()
                    composer.generationOption(thinkingKey, alias)
                }
                notify("Thinking: $normalized")
                return true
            }
            if (parseSlashToggle(raw) == SlashToggle.INVALID) {
                notify(if (rules.effortAsThinking)
                    "使い方: /thinking on / off / min / low / mid / high / xhigh / max"
                else "使い方: /thinking on / off / min / low / mid / high")
                return false
            }
        }
        if (raw.isNotEmpty()) {
            when (val desired = parseSlashToggle(raw)) {
                SlashToggle.INVALID -> { notify("使い方: ${command.label} on / off"); return false }
                SlashToggle.ON, SlashToggle.OFF -> if (item.checked == (desired == SlashToggle.ON)) {
                    notify("${item.label}: ${if (item.checked) "ON" else "OFF"}")
                    return true
                }
                SlashToggle.TOGGLE -> Unit
            }
        }
        if (disabled || item.rule?.disabled == true) return true
        item.toggle()
        return true
    }
}
