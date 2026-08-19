from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]
CHAT_JS_ASSETS = list((APP_ROOT / "static/js").glob("chat_core.v4.8.*.js"))
assert len(CHAT_JS_ASSETS) == 1, "Only the latest versioned chat core asset should remain"
CHAT_JS = CHAT_JS_ASSETS[0].read_text(encoding="utf-8")


class DefaultModelStabilityRegressionTests(unittest.TestCase):
    def test_default_model_dropdown_restores_stored_value_not_first_option(self):
        start = CHAT_JS.index("const populateDefaultModelOptions = () => {")
        end = CHAT_JS.index("const populateDefaultVisionModelOptions = () => {")
        block = CHAT_JS[start:end]
        self.assertIn("userSettingsSnapshot", block)
        self.assertIn("userSettingsSnapshot.default_model", block)
        self.assertIn("Array.from(sel.options).some(o => o.value === stored)", block)
        self.assertNotIn("if (current) sel.value = current;", block)

    def test_vision_model_dropdown_restores_stored_value_not_first_option(self):
        start = CHAT_JS.index("const populateDefaultVisionModelOptions = () => {")
        end = CHAT_JS.index("window.openSettingsModal = ")
        block = CHAT_JS[start:end]
        self.assertIn("userSettingsSnapshot.default_vision_model", block)
        self.assertIn("Array.from(sel.options).some(o => o.value === stored)", block)
        self.assertNotIn("if (current) sel.value = current;", block)

    def test_open_settings_waits_for_cached_settings_before_populating(self):
        start = CHAT_JS.index("window.openSettingsModal = ")
        end = CHAT_JS.index("showModal('settings-modal')")
        opener = CHAT_JS[start:end]
        self.assertIn("await ensureUserSettingsSnapshot()", opener)
        self.assertLess(
            opener.index("await ensureUserSettingsSnapshot()"),
            opener.index("populateDefaultModelOptions()"),
        )


if __name__ == "__main__":
    unittest.main()
