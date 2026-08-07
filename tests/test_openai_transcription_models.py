from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


class OpenAITranscriptionModelRegressionTests(unittest.TestCase):
    def test_backend_registers_file_and_realtime_transcription_models(self):
        source = (APP_ROOT / "app.py").read_text(encoding="utf-8")

        self.assertIn('"gpt-transcribe": {"provider": "openai", "mode": "transcription"', source)
        self.assertIn('"gpt-live-transcribe": {"provider": "openai", "mode": "transcription"', source)
        self.assertIn('async def _openai_realtime_transcribe(', source)
        self.assertIn('"wss://api.openai.com/v1/realtime?intent=transcription"', source)
        self.assertIn('"type": "transcription"', source)
        self.assertIn('"conversation.item.input_audio_transcription.delta"', source)
        self.assertIn('"conversation.item.input_audio_transcription.completed"', source)
        self.assertIn('"gpt-transcribe",\n            "gpt-4o-mini-transcribe"', source)

    def test_frontend_exposes_both_models_and_handles_text_only_completion(self):
        assets = list((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1)
        source = assets[0].read_text(encoding="utf-8")

        self.assertRegex(source, r'id:\s*"gpt-transcribe"[^}]*name:\s*"GPT Transcribe"')
        self.assertRegex(source, r'id:\s*"gpt-live-transcribe"[^}]*name:\s*"GPT Live Transcribe"')
        self.assertIn("stsData.audio_url || stsData.transcription_only", source)

        template = (APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        self.assertIn('value="gpt-transcribe">gpt-transcribe（推奨・高精度）', template)


if __name__ == "__main__":
    unittest.main()
