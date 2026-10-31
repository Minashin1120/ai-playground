"""Regression tests: a model the user picks survives editing, regenerating and cancelling an edit."""
import unittest
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
PARTS = APP_ROOT / "static" / "js" / "chat_core_parts"
ANDROID = APP_ROOT / "android" / "app" / "src" / "main" / "java" / "com" / "minashin1120" / "aiplayground"


def _part(prefix):
    return next(PARTS.glob(prefix + "_*.js")).read_text(encoding="utf-8")


def _between(source, start, end):
    begin = source.index(start)
    return source[begin: source.index(end, begin + len(start))]


class WebEditKeepsManualModelTests(unittest.TestCase):
    def test_model_list_and_welcome_mark_a_manual_pick(self):
        part06 = _part("chat_core.part06")
        self.assertIn("item.onclick = () => selectModelByUser(m.id, m.name);", part06)
        helper = _between(part06, "function selectModelByUser(id, name) {", "\n        }\n")
        self.assertIn("window._visionPickerActive", helper)
        self.assertIn("el.value === id) modelChosenManually = true", helper)
        quick = _between(part06, "window.quickStart = (m) => {", "};")
        self.assertIn("selectModelByUser(", quick)

    def test_restore_keeps_manual_model_but_restores_options(self):
        part04 = _part("chat_core.part04")
        restore = _between(part04, "function restorePromptOptionsForEdit(message) {", "function beginEditMessage(")
        self.assertIn("if (!modelChosenManually && message.model", restore)
        self.assertIn("key === 'enable_prompt_caching' && modelChosenManually", restore)
        self.assertIn("const ids = PROMPT_OPTION_INPUT_IDS;", restore)
        self.assertIn("enable_search: 'enable-search'", part04)

    def test_flag_resets_only_for_another_or_new_chat(self):
        part15 = _part("chat_core.part15")
        load = _between(part15, "async function loadMessages(tid, opts = {}) {", "async function loadGems()")
        self.assertIn("if (nextThreadId !== currentThreadId) modelChosenManually = false;", load)
        self.assertIn("if (threadData.last_model && !modelChosenManually) {", load)
        new_chat = _between(part15, "function startNewChat(opts = {}) {", "restoreLatestPromptOptions();")
        self.assertIn("modelChosenManually = false;", new_chat)


class AndroidEditKeepsManualModelTests(unittest.TestCase):
    def setUp(self):
        self.vm = (ANDROID / "ChatViewModel.kt").read_text(encoding="utf-8")

    def test_pickers_mark_a_manual_pick(self):
        screen = (ANDROID / "ui" / "PlaygroundScreen.kt").read_text(encoding="utf-8")
        self.assertIn("onSelect = { model.chooseModelByUser(it); modelPicker = false }", screen)
        self.assertIn("onChoose = { id -> model.chooseModelByUser(id); welcomeDismissed = true }", screen)
        helper = _between(self.vm, "fun chooseModelByUser(model: String) {", "\n    }\n")
        self.assertIn("if (state.value.model == model) modelChosenManually = true", helper)

    def test_restore_keeps_manual_model_but_restores_options(self):
        restore = _between(self.vm, "private fun ChatState.withPromptOptionsFrom(", "private fun ChatState.withLatestPromptOptions()")
        self.assertIn("val modelId = if (modelChosenManually) model else", restore)
        self.assertIn("enablePromptCache = if (modelChosenManually) enablePromptCache else", restore)
        self.assertIn('enableSearch = savedBoolean("enable_search", enableSearch)', restore)

    def test_flag_resets_only_for_another_or_new_chat(self):
        new_chat = _between(self.vm, "fun newChat(", "mutable.update")
        self.assertIn("modelChosenManually = false", new_chat)
        open_thread = _between(self.vm, "fun openThread(thread: ThreadItem", "mutable.update")
        self.assertIn("if (state.value.selected?.id != thread.id) modelChosenManually = false", open_thread)
        self.assertIn('if (!inPlace && !modelChosenManually) reply.nullableString("last_model")', self.vm)


if __name__ == "__main__":
    unittest.main()
