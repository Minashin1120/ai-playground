"""Regression tests: a model the user picks survives editing, regenerating and cancelling an edit."""
import unittest
from pathlib import Path

from tests.app_source import read_chat_core_source


APP_ROOT = Path(__file__).resolve().parents[1]
PARTS = APP_ROOT / "static" / "js" / "chat_core_parts"
ANDROID = APP_ROOT / "android" / "app" / "src" / "main" / "java" / "com" / "minashin1120" / "aiplayground"


def _part():
    return read_chat_core_source()


def _between(source, start, end):
    begin = source.index(start)
    return source[begin: source.index(end, begin + len(start))]


class WebEditKeepsManualModelTests(unittest.TestCase):
    def test_model_list_and_welcome_mark_a_manual_pick(self):
        part07 = _part()
        self.assertIn("item.onclick = () => selectModelByUser(m.id, m.name);", part07)
        helper = _between(part07, "function selectModelByUser(id, name) {", "\n        }\n")
        self.assertIn("window._visionPickerActive", helper)
        self.assertIn("el.value === id) modelChosenManually = true", helper)
        quick = _between(part07, "window.quickStart = (m) => {", "};")
        self.assertIn("selectModelByUser(", quick)

    def test_restore_keeps_manual_model_but_restores_options(self):
        part04 = _part()
        restore = _between(part04, "function restorePromptOptionsForEdit(message) {", "function beginEditMessage(")
        self.assertIn("if (!modelChosenManually && message.model", restore)
        self.assertIn("key === 'enable_prompt_caching' && modelChosenManually", restore)
        self.assertIn("const ids = PROMPT_OPTION_INPUT_IDS;", restore)
        self.assertIn("enable_search: 'enable-search'", part04)

    def test_flag_resets_only_for_another_or_new_chat(self):
        part16 = _part()
        load = _between(part16, "async function loadMessages(tid, opts = {}) {", "async function loadSkills()")
        self.assertIn("if (nextThreadId !== currentThreadId) modelChosenManually = false;", load)
        self.assertIn("if (threadData.last_model && !modelChosenManually) {", load)
        new_chat = _between(part16, "function startNewChat(opts = {}) {", "restoreLatestPromptOptions();")
        self.assertIn("modelChosenManually = false;", new_chat)


class AndroidEditKeepsManualModelTests(unittest.TestCase):
    def setUp(self):
        self.vm = (ANDROID / "ChatViewModel.kt").read_text(encoding="utf-8")
        self.composer = (ANDROID / "ChatComposer.kt").read_text(encoding="utf-8")
        self.editing = (ANDROID / "ChatEditing.kt").read_text(encoding="utf-8")
        self.loader = (ANDROID / "ChatMessageLoader.kt").read_text(encoding="utf-8")
        self.navigator = (ANDROID / "ChatNavigator.kt").read_text(encoding="utf-8")

    def test_pickers_mark_a_manual_pick(self):
        # PlaygroundScreen.kt was split into ui/*.kt parts; search all of them.
        screen = "\n".join(p.read_text(encoding="utf-8") for p in sorted((ANDROID / "ui").glob("*.kt")))
        self.assertIn("onSelect = { model.chooseModelByUser(it); overlays.modelPicker = false }", screen)
        self.assertIn("onChoose = { id -> model.chooseModelByUser(id); welcomeDismissed = true }", screen)
        helper = _between(self.composer, "fun chooseModelByUser(model: String) {", "\n    }\n")
        self.assertIn("if (state.value.model == model) modelChosenManually = true", helper)

    def test_restore_keeps_manual_model_but_restores_options(self):
        restore = _between(self.editing, "private fun ChatState.withPromptOptionsFrom(", "fun withLatestPromptOptions(")
        self.assertIn("val modelId = if (composer.modelChosenManually) model else", restore)
        self.assertIn("enablePromptCache = if (composer.modelChosenManually) enablePromptCache else", restore)
        self.assertIn('enableSearch = savedBoolean("enable_search", enableSearch)', restore)

    def test_flag_resets_only_for_another_or_new_chat(self):
        new_chat = _between(self.navigator, "fun newChat(", "mutable.update")
        self.assertIn("modelChosenManually = false", new_chat)
        open_thread = _between(self.navigator, "fun openThread(thread: ThreadItem", "mutable.update")
        self.assertIn("if (state.value.selected?.id != thread.id) composer.modelChosenManually = false", open_thread)
        self.assertIn('if (!inPlace && !manualModelChosen()) reply.nullableString("last_model")', self.loader)


if __name__ == "__main__":
    unittest.main()
