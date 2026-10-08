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
        self.assertIn('f"https://{_XAI_API_HOST}/v1/stt"', source)
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

    def test_non_live_grok_transcribe_is_an_sts_file_model_using_the_batch_endpoint(self):
        source = read_app_source()

        self.assertIn('"grok-voice-transcribe-2.0-file": {"provider": "xai", "mode": "transcription"', source)
        self.assertIn('XAI_STT_FILE_STS_MODELS = {"grok-voice-transcribe-2.0-file": "grok-voice-transcribe-2.0"}', source)
        self.assertIn('model_key in XAI_STT_FILE_STS_MODELS', source)
        self.assertIn('XAI_STT_FILE_STS_MODELS[model_key]', source)
        # The live model keeps its own id and stays a streaming session.
        self.assertIn('XAI_LIVE_STT_MODELS = {"grok-voice-transcribe-2.0"}', source)

    def test_non_live_grok_transcribe_is_not_a_conversation_model(self):
        import app as target

        model = "grok-voice-transcribe-2.0-file"
        self.assertEqual(target.get_sts_provider(model), "xai")
        self.assertFalse(target._rt_is_conversation_model(model))
        self.assertEqual(target._mobile_model_mode(model), "transcription")
        self.assertIn(model, target.XAI_STT_FILE_STS_MODELS)
        self.assertNotIn(model, target.XAI_LIVE_STT_MODELS)

    def test_web_lists_both_grok_voice_transcribe_2_variants(self):
        parts = APP_ROOT / "static" / "js" / "chat_core_parts"
        settings = (parts / "chat_core.part05_settings_modal.js").read_text(encoding="utf-8")
        self.assertRegex(settings, r'id:\s*"grok-voice-transcribe-2.0-file"[^}]*name:\s*"Grok Voice Transcribe 2.0"')
        self.assertRegex(settings, r'id:\s*"grok-voice-transcribe-2.0"[^}]*name:\s*"Grok Voice Transcribe 2.0 \(Live\)"')
        self.assertIn("model === 'grok-voice-transcribe-2.0-file'", settings)

        android = (
            APP_ROOT / "android" / "app" / "src" / "main" / "java" / "com" / "minashin1120"
            / "aiplayground" / "ui" / "RealtimeStudio.kt"
        ).read_text(encoding="utf-8")
        self.assertIn('GROK_FILE_TRANSCRIBE_MODEL = "grok-voice-transcribe-2.0-file"', android)


if __name__ == "__main__":
    unittest.main()
