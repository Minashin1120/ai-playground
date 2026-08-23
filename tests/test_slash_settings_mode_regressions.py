from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


def _chat_core_source():
    assets = list((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
    assert len(assets) == 1, f"Expected one current chat core asset, got: {assets}"
    return assets[0].read_text(encoding="utf-8")


class SlashSettingsModeRegressionTests(unittest.TestCase):
    def test_settings_command_uses_normal_chat_bubble_flow(self):
        source = _chat_core_source()
        runner = source[source.index("async function runAiSettingsCommand") :]
        runner = runner[: runner.index("// === Gem suggestion helpers")]

        self.assertIn("renderMessage(`settings-user-${timestamp}`, 'user'", runner)
        self.assertIn("buildPendingSkeletonHtml(modelId", runner)
        self.assertIn("renderAiSettingsResultBubble(data.applied, modelId)", runner)
        self.assertIn("welcome.classList.add('hidden')", runner)
        self.assertIn("conversation: aiSettingsConversation", runner)
        self.assertIn("appendAiSettingsConversation('user', instruction)", runner)
        self.assertEqual(source.count("apiFetch('/api/settings/apply-ai-prompt'"), 1)
        self.assertEqual(source.count("await runAiSettingsCommand(instruction,"), 2)

    def test_result_bubble_lists_every_change_as_a_jump_button(self):
        source = _chat_core_source()
        renderer = source[source.index("function renderAiSettingsResultBubble") :]
        renderer = renderer[: renderer.index("async function runAiSettingsCommand")]

        self.assertIn("Object.entries(values || {})", renderer)
        self.assertNotIn(".slice(", renderer)
        self.assertIn("button.addEventListener('click', () => openAiSettingJumpTarget(key))", renderer)
        self.assertIn("formatAiSettingValue(value)", renderer)
        self.assertIn("ai-settings-result-item", renderer)

    def test_inspection_result_uses_same_actionable_answer_bubble_without_refreshing_settings(self):
        source = _chat_core_source()
        runner = source[source.index("async function runAiSettingsCommand") :]
        runner = runner[: runner.index("// === Gem suggestion helpers")]
        inspect_branch = runner[runner.index("data.mode === 'inspect'") : runner.index("data.applied")]

        self.assertIn("data.current", inspect_branch)
        self.assertIn("renderAiSettingsResultBubble(data.current, modelId, 'inspect')", inspect_branch)
        self.assertNotIn("populateAiSafeFormFields", inspect_branch)
        renderer = source[source.index("function renderAiSettingsResultBubble") :]
        renderer = renderer[: renderer.index("async function runAiSettingsCommand")]
        self.assertIn("mode === 'inspect'", renderer)
        self.assertIn("現在の設定を確認しました", renderer)

    def test_safe_ai_settings_have_named_jump_targets(self):
        source = _chat_core_source()
        mapping = source[source.index("const AI_SETTING_JUMP_TARGETS = {") :]
        mapping = mapping[: mapping.index("function formatAiSettingValue")]
        expected = {
            "default_model", "default_enable_search", "default_enable_url_context", "default_enable_maps",
            "default_enable_python", "default_enable_thinking", "default_thinking_level",
            "default_thinking_budget", "default_reasoning_effort", "default_enable_system_prompt",
            "default_safety_setting", "default_vision_model", "system_prompt", "system_prompt_enabled",
            "apply_global_system_prompt", "apply_auto_system_prompt_notices",
            "auto_system_prompt_notices_config", "mic_transcribe_mode", "stt_model",
            "llm_transcribe_prompt", "enter_to_send", "use_sw_cache", "compact_prompt_mode",
            "minimal_prompt_mode",
            "auto_search_on_links", "use_last_chat_settings", "temp_chat_timeout_seconds",
            "theme_color", "liquid_glass_enabled", "rich_paste_prompt_default",
            "rich_paste_prompt_use_custom_default", "enable_latency_metrics",
            "enable_client_debug_log", "bot_detection_enabled", "skip_2fa_on_google_login",
            "default_2fa_method",
        }
        for key in expected:
            self.assertIn(f"{key}:", mapping)

        jump = source[source.index("function openAiSettingJumpTarget") :]
        jump = jump[: jump.index("function renderAiSettingsResultBubble")]
        self.assertIn("jumpToSetting(target.tab, element)", jump)
        self.assertIn("openRichPasteModal()", jump)

    def test_tapping_palette_item_selects_before_blur_hides_it(self):
        source = _chat_core_source()
        palette = source[source.index("function showSlashCommandSuggestions") :]
        palette = palette[: palette.index("function selectSlashCommand")]

        pointerdown = palette.index("item.addEventListener('pointerdown'")
        click = palette.index("item.addEventListener('click'")
        append = palette.index("listEl.appendChild(item)")
        self.assertLess(pointerdown, click)
        self.assertLess(click, append)
        self.assertIn("event.preventDefault()", palette[pointerdown:click])
        self.assertIn("selectSlashCommand(cmd.id)", palette[pointerdown:click])
        self.assertIn("if (!selectedByPointer)", palette[click:append])

    def test_typed_or_pasted_command_is_not_committed_until_enter_or_click(self):
        source = _chat_core_source()
        self.assertNotIn("function activateTypedSlashCommand(input)", source)
        self.assertNotIn("input.value = match[2]", source)

        input_handler = source[source.index("get('prompt-input').addEventListener('input'") :]
        input_handler = input_handler[: input_handler.index("get('prompt-input').addEventListener('blur'")]
        self.assertNotIn("pendingSlashCommand = cmd.id", input_handler)
        self.assertIn("extractSlashCommandToken(val)", input_handler)
        self.assertIn("if (pendingSlashCommand)", input_handler)

        selector = source[source.index("function selectSlashCommand(cmdId)") :]
        selector = selector[: selector.index("const AI_SETTING_JUMP_TARGETS")]
        self.assertIn("pendingSlashCommand = cmdId", selector)
        self.assertIn("showPendingSlashCommandIndicator(cmdId)", selector)

    def test_empty_pending_command_keeps_mode_active(self):
        source = _chat_core_source()
        pending = source[source.index("// Handle pending slash command") :]
        pending = pending[: pending.index("if (browserFastModeEnabled)")]

        empty_check = pending.index("if (!instruction)")
        self.assertIn("get('prompt-input').focus();", pending[empty_check:])
        self.assertIn("pendingSlashCommand", pending)
        self.assertNotIn("hidePendingSlashCommandIndicator(); // valid command", pending)

    def test_settings_history_is_restored_and_persisted_for_followups(self):
        source = _chat_core_source()
        self.assertIn("sessionStorage.getItem(AI_SETTINGS_CONVERSATION_KEY)", source)
        self.assertIn("sessionStorage.setItem(AI_SETTINGS_CONVERSATION_KEY", source)
        self.assertIn("pendingSlashCommand = 'settings';", source)
        self.assertIn("aiSettingsConversation.length > 0", source)

    def test_direct_command_detection_requires_token_boundary(self):
        source = _chat_core_source()
        direct = source[source.index("// === /settings command") :]
        direct = direct[: direct.index("if (isGeminiLocalPythonMode")]
        self.assertIn("/^\\/settings(?:\\s|$)/i.test(trimmedRaw)", direct)
        self.assertNotIn("startsWith('/settings')", direct)
        self.assertIn("showSlashCommandSuggestions(hintFilter)", direct)
        self.assertIn("input.focus()", direct)

    def test_palette_rerenders_only_when_command_name_filter_changes(self):
        source = _chat_core_source()
        input_handler = source[source.index("get('prompt-input').addEventListener('input'") :]
        input_handler = input_handler[: input_handler.index("get('prompt-input').addEventListener('blur'")]
        slash_branch = input_handler[input_handler.index("} else if (val.startsWith('/')) {") :]
        slash_branch = slash_branch[: slash_branch.index("} else {")]
        self.assertIn("lastSlashFilter", slash_branch)
        self.assertIn("filter !== lastSlashFilter", slash_branch)
        self.assertIn("showSlashCommandSuggestions(filter)", slash_branch)

    def test_palette_filter_uses_leading_command_token(self):
        source = _chat_core_source()
        self.assertIn("function extractSlashCommandToken(val)", source)
        helper = source[source.index("function extractSlashCommandToken(val)") :]
        helper = helper[: helper.index("function hideSlashCommandSuggestions()")]
        self.assertIn("token.match(/^[a-z][\\w-]*/i)", helper)

        input_handler = source[source.index("get('prompt-input').addEventListener('input'") :]
        input_handler = input_handler[: input_handler.index("get('prompt-input').addEventListener('blur'")]
        slash_branch = input_handler[input_handler.index("} else if (val.startsWith('/')) {") :]
        slash_branch = slash_branch[: slash_branch.index("} else {")]
        self.assertIn("extractSlashCommandToken(val)", slash_branch)
        self.assertNotIn("val.substring(1)", slash_branch)

    def test_keyboard_navigation_also_uses_leading_command_token(self):
        source = _chat_core_source()
        keydown = source[source.index("// Slash command keyboard navigation") :]
        keydown = keydown[: keydown.index("// Cancel pending slash command mode with Escape")]
        self.assertEqual(keydown.count("extractSlashCommandToken(input.value)"), 3)
        self.assertNotIn("input.value.trim().substring(1)", keydown)

    def test_selecting_command_keeps_text_following_command_without_space(self):
        source = _chat_core_source()
        selector = source[source.index("function selectSlashCommand(cmdId)") :]
        selector = selector[: selector.index("const AI_SETTING_JUMP_TARGETS")]
        self.assertIn("extractSlashCommandToken(val)", selector)
        self.assertIn("trimmed.substring(1 + token.length).trimStart()", selector)


if __name__ == "__main__":
    unittest.main()
