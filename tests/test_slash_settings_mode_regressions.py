from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


def _chat_core_source():
    assets = list((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
    assert len(assets) == 1, f"Expected one current chat core asset, got: {assets}"
    return assets[0].read_text(encoding="utf-8")


class SlashSettingsModeRegressionTests(unittest.TestCase):
    def test_typed_or_pasted_settings_command_enters_visible_mode(self):
        source = _chat_core_source()
        helper = source[source.index("function activateTypedSlashCommand(input)") :]
        helper = helper[: helper.index("// === Gem suggestion helpers")]

        self.assertIn("input.value.match", helper)
        self.assertIn("pendingSlashCommand = cmd.id", helper)
        self.assertIn("showPendingSlashCommandIndicator(cmd.id)", helper)
        self.assertIn("input.value = match[2]", helper)

        input_handler = source[source.index("get('prompt-input').addEventListener('input'") :]
        input_handler = input_handler[: input_handler.index("get('prompt-input').addEventListener('blur'")]
        self.assertIn("activateTypedSlashCommand(this)", input_handler)
        self.assertIn("if (pendingSlashCommand)", input_handler)

    def test_empty_pending_command_keeps_mode_active(self):
        source = _chat_core_source()
        pending = source[source.index("// Handle pending slash command") :]
        pending = pending[: pending.index("if (browserFastModeEnabled)")]

        empty_check = pending.index("if (!instruction)")
        leave_mode = pending.index("hidePendingSlashCommandIndicator(); // valid command")
        self.assertLess(empty_check, leave_mode)
        self.assertIn("get('prompt-input').focus();", pending[empty_check:leave_mode])

    def test_direct_command_detection_requires_token_boundary(self):
        source = _chat_core_source()
        direct = source[source.index("// === /settings command") :]
        direct = direct[: direct.index("if (isGeminiLocalPythonMode")]
        self.assertIn("/^\\/settings(?:\\s|$)/i.test(trimmedRaw)", direct)
        self.assertNotIn("startsWith('/settings')", direct)
        self.assertIn("activateTypedSlashCommand(input)", direct)
        self.assertIn("input.focus()", direct)


if __name__ == "__main__":
    unittest.main()
