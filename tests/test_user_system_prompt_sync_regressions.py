"""Regression tests: the user system prompt must save and apply from both editors.

The settings modal and the chat settings modal (the SysPrompt gear) edit the
same user system prompt.  The DOMContentLoaded callback is split across several
chat_core parts, so helpers declared there are invisible to the other parts
unless exposed on window.  Bare references used to throw ReferenceError that
was swallowed: the chat settings modal never saved the user prompt, and
applyChatDefaults aborted page initialisation before the SysPrompt switch was
turned on.

The checks read the combined chat_core source so a later part re-split does not
break them.
"""
import re
import unittest

from tests.app_source import read_chat_core_source

SOURCE = read_chat_core_source()


def _between(start_marker, end_marker):
    start = SOURCE.index(start_marker)
    return SOURCE[start:SOURCE.index(end_marker, start + len(start_marker))]


class UserSystemPromptSyncRegressionTests(unittest.TestCase):
    def test_chat_settings_modal_uses_exposed_auto_prompt_collector(self):
        self.assertIn(
            "window.collectAutoSystemPromptConfigFromForm = collectAutoSystemPromptConfigFromForm;",
            SOURCE,
        )
        self.assertIsNone(
            re.search(r"(?<![.\w])collectAutoSystemPromptConfigFromForm\('thread'\)", SOURCE)
        )
        self.assertIn("window.collectAutoSystemPromptConfigFromForm('thread')", SOURCE)
        self.assertIn("apply_global_system_prompt: get('thread-apply-global-sys-prompt')", SOURCE)

    def test_apply_chat_defaults_calls_exposed_toggle_options(self):
        self.assertIn("window.toggleOptions = toggleOptions;", SOURCE)
        defaults = _between("function applyChatDefaults(d) {", "\n        function ")
        self.assertIsNone(re.search(r"(?<![.\w])toggleOptions\(\)", defaults))

    def test_both_editors_sync_cache_and_prompt_bar_switch(self):
        self.assertIn("window.applySavedUserSystemPromptSettings = (saved) =>", SOURCE)
        self.assertIn("window.applySavedUserSystemPromptSettings({", SOURCE)
        self.assertIn("window.applySavedUserSystemPromptSettings(userPromptPayload)", SOURCE)


if __name__ == "__main__":
    unittest.main()
