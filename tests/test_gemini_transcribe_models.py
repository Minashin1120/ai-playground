from pathlib import Path
import unittest

from tests.chat_template import read_chat_markup

from tests.app_source import read_app_source
APP_ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = read_app_source()
CHAT_JS_ASSETS = list((APP_ROOT / "static/js").glob("chat_core.v4.8.*.js"))
assert len(CHAT_JS_ASSETS) == 1, "Only the latest versioned chat core asset should remain"
CHAT_JS = CHAT_JS_ASSETS[0].read_text(encoding="utf-8")
SETUP_HTML = (APP_ROOT / "templates/setup.html").read_text(encoding="utf-8")
CHAT_HTML = read_chat_markup()


class GeminiTranscribeRegressionTests(unittest.TestCase):
    TRANSRIBE_MODELS = ("gemini-3.5-transcribe", "gemini-3.5-transcribe-live")

    def test_transcribe_models_registered_across_ui_and_backend(self):
        for model_id in self.TRANSRIBE_MODELS:
            self.assertIn(model_id, APP_SOURCE)
            self.assertIn(model_id, CHAT_JS)
            self.assertIn(model_id, SETUP_HTML)
            # implementedAt must be on the definition line
            self.assertIn(
                f'id: "{model_id}",',
                CHAT_JS,
            )
            self.assertRegex(
                CHAT_JS,
                rf'id:\s*"{re_escape(model_id)}",\s*implementedAt:\s*"\d{{4}}-\d{{2}}-\d{{2}}"',
                f"missing implementedAt for {model_id}",
            )

    def test_transcribe_models_not_deprecated(self):
        for model_id in self.TRANSRIBE_MODELS:
            definition_at = CHAT_JS.index(f'{{ id: "{model_id}"')
            self.assertNotIn(
                "deprecated: true",
                CHAT_JS[definition_at:definition_at + 500],
                model_id,
            )

    def test_unary_transcribe_is_not_live_model(self):
        # The unary model must route through the Interactions API branch,
        # not the STS / Live API path.
        self.assertIn("gemini-3.5-transcribe-live", APP_SOURCE)
        self.assertIn('mk == "gemini-3.5-transcribe"', APP_SOURCE)
        self.assertIn('"gemini-3.5-transcribe", "gemini-3.5-transcribe-live"', APP_SOURCE)
        # In the worker, the transcribe branch runs before TTS/image/text.
        transcribe_idx = APP_SOURCE.index("Gemini Transcribe (audio file -> text")
        self.assertLess(transcribe_idx, APP_SOURCE.index("# Gemini TTS (Preview)"))

    def test_rest_transcription_helper_present(self):
        self.assertIn("def _gemini_transcribe_rest(", APP_SOURCE)
        self.assertIn('url = "https://generativelanguage.googleapis.com/v1beta/interactions"', APP_SOURCE)
        self.assertIn('"store": False', APP_SOURCE)
        self.assertIn("transcription_config", APP_SOURCE)

    def test_live_transcribe_registered_as_google_sts(self):
        # Live variant must be in STS_MODELS with provider google (both sides).
        self.assertIn(
            '"gemini-3.5-transcribe-live": {"provider": "google", "mode": "transcription"',
            APP_SOURCE,
        )
        self.assertIn("'gemini-3.5-transcribe-live',", CHAT_JS)
        # /api/gemini/session must build TEXT + input_audio_transcription for it.
        self.assertIn("is_live_transcribe = (model_key == \"gemini-3.5-transcribe-live\")", APP_SOURCE)
        self.assertIn("'input_audio_transcription': {},", APP_SOURCE)
        self.assertIn("'response_modalities': ['TEXT']", APP_SOURCE)
        # The installed SDK rejects `translation_config` inside the ephemeral token
        # config; the WebSocket setup message carries it instead.
        self.assertIn("translationConfig", CHAT_JS)

    def test_frontend_live_transcribe_helpers(self):
        self.assertIn("isGeminiLiveTranscribeModel", CHAT_JS)
        self.assertIn("value === 'gemini-3.5-transcribe-live'", CHAT_JS)
        self.assertIn("transcriptionConfig", CHAT_JS)
        # STS panel offers a mode selector + custom vocabulary input.
        self.assertIn("sts-transcribe-mode", CHAT_HTML)
        self.assertIn("sts-custom-vocab", CHAT_HTML)

    def test_setup_options_include_both(self):
        for model_id in self.TRANSRIBE_MODELS:
            self.assertIn(f'<option value="{model_id}">', SETUP_HTML)

    def test_send_message_payload_carries_transcription_options(self):
        self.assertIn("transcription_language_codes:", CHAT_JS)
        self.assertIn("transcription_custom_vocabulary:", CHAT_JS)
        self.assertIn("transcription_mode: 'verbatim'", CHAT_JS)
        self.assertIn("transcription_diarization:", CHAT_JS)
        self.assertIn("transcription_word_timestamps:", CHAT_JS)
        # Backend options mirror the payload keys.
        self.assertIn("'transcription_language_codes': data.get('transcription_language_codes')", APP_SOURCE)
        self.assertIn("'transcription_word_timestamps': data.get('transcription_word_timestamps')", APP_SOURCE)


def re_escape(s):
    import re
    return re.escape(s)


if __name__ == "__main__":
    unittest.main()
