from pathlib import Path
import unittest

from tests.chat_template import read_chat_markup

from tests.app_source import read_app_source
APP_ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = read_app_source()
CHAT_JS_ASSETS = list((APP_ROOT / "static/js").glob("chat_core.v4.8.*.js"))
assert len(CHAT_JS_ASSETS) == 1, "Only the latest versioned chat core asset should remain"
CHAT_JS = CHAT_JS_ASSETS[0].read_text(encoding="utf-8")
CHAT_HTML = read_chat_markup()


class VoiceStudioRegressionTests(unittest.TestCase):

    def test_voice_studio_setting_backend_plumbing(self):
        # DB column + migration
        self.assertIn("voice_studio_ui = db.Column", APP_SOURCE)
        self.assertIn("ensure_user_voice_studio_ui_column", APP_SOURCE)
        self.assertIn("ADD COLUMN voice_studio_ui BOOLEAN DEFAULT 1", APP_SOURCE)
        # settings query response
        self.assertIn("'voice_studio_ui': current_user.voice_studio_ui", APP_SOURCE)
        # handle_settings POST
        self.assertIn("if 'voice_studio_ui' in d: current_user.voice_studio_ui = bool(d['voice_studio_ui'])", APP_SOURCE)
        # AI settings editable + apply
        self.assertIn("'voice_studio_ui'", APP_SOURCE[APP_SOURCE.index("AI_SAFE_EDITABLE_FIELDS = {"):APP_SOURCE.index("AI_NEVER_EDITABLE_FIELDS")])
        self.assertIn("elif key == 'voice_studio_ui'", APP_SOURCE)
        # account export/import fields
        self.assertIn('"voice_studio_ui"', APP_SOURCE[APP_SOURCE.index("ACCOUNT_SETTING_FIELDS = ("):APP_SOURCE.index("ACCOUNT_SECRET_FIELDS")])
        self.assertIn('"voice_studio_ui"', APP_SOURCE[APP_SOURCE.index("ACCOUNT_BOOL_SETTING_FIELDS"):APP_SOURCE.index("ACCOUNT_INT_SETTING_FIELDS")])
        # AI settings schema
        self.assertIn('"voice_studio_ui": {"type": "boolean"', APP_SOURCE)

    def test_voice_studio_ui_elements_exist(self):
        for elem in (
            'id="voice-studio-bar"',
            'id="voice-studio-open-btn"',
            'id="voice-studio-modal"',
            'id="voice-studio-title"',
            'id="voice-studio-transcript"',
            'id="voice-studio-panel-host"',
            'id="voice-studio-file-host"',
            'id="voice-studio-close"',
            'id="set-voice-studio-ui"',
        ):
            self.assertIn(elem, CHAT_HTML, elem)

    def test_voice_studio_js_controller_exist(self):
        for symbol in (
            "const VoiceStudio",
            "window.VoiceStudio",
            "window.VoiceStudioOpen",
            "voiceStudioUiEnabled",
            "voice-studio-open-btn",
            "voice-studio-transcript",
            "movePanelIntoModal",
            "voice-studio-panel-host",
        ):
            self.assertIn(symbol, CHAT_JS, symbol)

    def test_update_sts_ui_is_studio_aware(self):
        segment = CHAT_JS[CHAT_JS.index("function updateStsUi"):CHAT_JS.index("function updateStsOptions")]
        self.assertIn("voiceStudioUiEnabled", segment)
        self.assertIn("voice-studio-bar", segment)
        self.assertIn("window.VoiceStudio", segment)
        self.assertIn("closeIfOpen", segment)

    def test_mic_handler_logs_to_voice_studio(self):
        segment = CHAT_JS[CHAT_JS.index("get('mic-btn').onclick"):]
        self.assertIn("window.VoiceStudio.log('user'", segment)
        self.assertIn("window.VoiceStudio.log('assistant'", segment)
        self.assertIn("studioInput", segment)
        self.assertIn("studioAssistant", segment)

    def test_settings_form_binds_voice_studio_toggle(self):
        self.assertIn("set-voice-studio-ui", CHAT_JS)
        self.assertIn("voice_studio_ui: get('set-voice-studio-ui')", CHAT_JS)
        self.assertIn("voiceStudioUiEnabled = b.voice_studio_ui", CHAT_JS)

    def test_voice_studio_ui_defaults_to_enabled(self):
        self.assertIn("let voiceStudioUiEnabled = true;", CHAT_JS)
        self.assertIn("voice_studio_ui !== false", CHAT_JS)

    def test_lyria_studio_still_present(self):
        # The previous feature must remain intact.
        self.assertIn('id="lyria-studio-modal"', CHAT_HTML)
        self.assertIn("LyriaRealtimeStudio", CHAT_JS)


if __name__ == "__main__":
    unittest.main()
