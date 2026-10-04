"""Regression tests: the user system prompt must save and apply from both editors.

The settings modal and the chat settings modal (the SysPrompt gear) edit the
same user system prompt.  part07-part10 form one DOMContentLoaded callback, so
helpers declared there are invisible to the other parts unless exposed on
window.  Bare references used to throw ReferenceError that was swallowed:
the chat settings modal never saved the user prompt, and applyChatDefaults
aborted page initialisation before the SysPrompt switch was turned on.
"""
from pathlib import Path
import re
import unittest

PARTS = Path(__file__).resolve().parents[1] / "static" / "js" / "chat_core_parts"


def _part(prefix):
    matches = sorted(PARTS.glob(f"chat_core.{prefix}_*.js"))
    assert len(matches) == 1, prefix
    return matches[0].read_text(encoding="utf-8")


class UserSystemPromptSyncRegressionTests(unittest.TestCase):
    def test_chat_settings_modal_uses_exposed_auto_prompt_collector(self):
        part08 = _part("part08")
        part15 = _part("part15")
        self.assertIn("window.collectAutoSystemPromptConfigFromForm = collectAutoSystemPromptConfigFromForm;", part08)
        self.assertIsNone(re.search(r"(?<![.\w])collectAutoSystemPromptConfigFromForm\(", part15))
        self.assertIn("window.collectAutoSystemPromptConfigFromForm('thread')", part15)
        self.assertIn("apply_global_system_prompt: get('thread-apply-global-sys-prompt')", part15)

    def test_apply_chat_defaults_calls_exposed_toggle_options(self):
        part04 = _part("part04")
        part07 = _part("part07")
        self.assertIn("window.toggleOptions = toggleOptions;", part07)
        self.assertIsNone(re.search(r"(?<![.\w])toggleOptions\(\)", part04))

    def test_both_editors_sync_cache_and_prompt_bar_switch(self):
        part01 = _part("part01")
        self.assertIn("window.applySavedUserSystemPromptSettings = (saved) =>", part01)
        self.assertIn("window.applySavedUserSystemPromptSettings({", _part("part08"))
        self.assertIn("window.applySavedUserSystemPromptSettings(userPromptPayload)", _part("part15"))


if __name__ == "__main__":
    unittest.main()
