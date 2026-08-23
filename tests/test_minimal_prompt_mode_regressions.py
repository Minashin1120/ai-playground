from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


def _current_asset(folder, pattern):
    assets = sorted((APP_ROOT / "static" / folder).glob(pattern))
    assert len(assets) == 1, f"Expected only the latest asset for {pattern}, got: {assets}"
    return assets[0].read_text(encoding="utf-8")


class MinimalPromptModeRegressionTests(unittest.TestCase):
    def test_settings_offer_minimal_prompt_mode(self):
        template = (APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        self.assertIn('id="set-minimal-prompt-mode"', template)
        self.assertIn('id="set-prompt-bar-mode-normal"', template)
        self.assertIn('id="set-compact-prompt-mode"', template)
        self.assertIn("ミニマル表示（送信・プラスのみ）", template)
        self.assertIn('id="top-model-bar"', template)
        self.assertIn("minimalPromptMode:", template)
        self.assertIn("minimal-prompt-mode", template)

    def test_js_moves_model_selector_and_keeps_send_plus_only(self):
        script = _current_asset("js", "chat_core.v4.8.*.js")
        self.assertIn("function setMinimalPromptMode", script)
        self.assertIn("function applyMinimalPromptMode", script)
        self.assertIn("function placeModelSelectorButton", script)
        self.assertIn("function readPromptBarModeFromForm", script)
        self.assertIn("function writePromptBarModeToForm", script)
        self.assertIn("document.body.classList.toggle('minimal-prompt-mode'", script)
        self.assertIn("fas fa-plus", script)
        self.assertIn("minimal_prompt_mode", script)
        self.assertIn("control: 'set-minimal-prompt-mode'", script)

    def test_css_hides_clutter_and_shows_top_model_bar(self):
        css = _current_asset("css", "chat.custom.v4.8.*.css")
        self.assertIn("body.minimal-prompt-mode #top-model-bar", css)
        self.assertIn("body.minimal-prompt-mode #prompt-controls-row", css)
        self.assertIn("body.minimal-prompt-mode #rich-paste-btn", css)
        self.assertIn("body.minimal-prompt-mode #mask-btn", css)
        self.assertIn("body.minimal-prompt-mode #mic-btn", css)
        self.assertIn("display: none !important;", css)

    def test_backend_persists_minimal_prompt_mode(self):
        app_source = (APP_ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("minimal_prompt_mode = db.Column(db.Boolean, default=False)", app_source)
        self.assertIn("'minimal_prompt_mode'", app_source)
        self.assertIn("ensure_user_minimal_prompt_mode_column", app_source)
        self.assertIn("ALTER TABLE user ADD COLUMN minimal_prompt_mode BOOLEAN DEFAULT 0", app_source)
        self.assertIn("プロンプトバーをミニマル表示", app_source)


if __name__ == "__main__":
    unittest.main()
