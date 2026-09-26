from pathlib import Path
import unittest

from tests.app_source import read_app_source
from tests.chat_template import read_chat_markup

APP_ROOT = Path(__file__).resolve().parents[1]


class XaiSttModelRegressionTests(unittest.TestCase):
    def test_backend_accepts_and_routes_grok_voice_transcribe_models(self):
        source = read_app_source()

        self.assertIn('"grok-voice-transcribe-2.0",\n    "grok-voice-transcribe-1.0",\n}', source)
        self.assertIn('XAI_STT_MODELS = {"grok-voice-transcribe-2.0", "grok-voice-transcribe-1.0"}', source)
        self.assertIn('return _transcribe_with_xai_stt(audio_content, fname, model, current_user)', source)
        self.assertIn('"https://api.x.ai/v1/stt"', source)
        self.assertIn('decrypt_val(user.xai_api_key)', source)
        # Option fields must be sent before the file part.
        self.assertIn('data=[("model", model)],\n        files=[("file",', source)

    def test_web_and_android_expose_grok_voice_transcribe_models(self):
        template = read_chat_markup()
        self.assertIn('value="grok-voice-transcribe-2.0">grok-voice-transcribe-2.0（xAI）', template)
        self.assertIn('value="grok-voice-transcribe-1.0">grok-voice-transcribe-1.0（xAI）', template)

        settings = (
            APP_ROOT / "android" / "app" / "src" / "main" / "java" / "com" / "minashin1120"
            / "aiplayground" / "ui" / "SettingsTabs.kt"
        ).read_text(encoding="utf-8")
        # Android shows the same option labels as the Web select.
        self.assertIn('"grok-voice-transcribe-2.0" to "grok-voice-transcribe-2.0（xAI）"', settings)
        self.assertIn('"grok-voice-transcribe-1.0" to "grok-voice-transcribe-1.0（xAI）"', settings)


if __name__ == "__main__":
    unittest.main()
