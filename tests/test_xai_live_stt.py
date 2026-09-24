import asyncio
import json
import unittest
from pathlib import Path

import app as target

from tests.app_source import read_app_source
from tests.chat_template import read_chat_markup

APP_ROOT = Path(__file__).resolve().parents[1]


class _FakeWs:
    def __init__(self, messages):
        self.messages = [json.dumps(m) for m in messages]
        self.sent = []

    async def recv(self):
        if not self.messages:
            raise AssertionError("no more messages")
        return self.messages.pop(0)

    async def send(self, data):
        self.sent.append(data)


def _session(model="grok-voice-transcribe-2.0", data=None):
    params = target._normalize_rt_params("xai", model, data or {})
    return target.RtSession("rt_test", 1, model, "key", params)


class XaiLiveSttTests(unittest.TestCase):
    def test_model_is_registered_as_server_session_transcription(self):
        self.assertEqual(target.get_sts_provider("grok-voice-transcribe-2.0"), "xai")
        self.assertTrue(target._rt_is_conversation_model("grok-voice-transcribe-2.0"))
        self.assertEqual(target._mobile_model_mode("grok-voice-transcribe-2.0"), "realtime_audio")
        # Batch STT slugs stay file transcription models.
        self.assertEqual(target._mobile_model_mode("grok-voice-transcribe-1.0"), "transcription")

    def test_params_force_16k_and_cap_keyterms(self):
        terms = [f"term{i}" for i in range(150)] + ["x" * 80]
        session = _session(data={"custom_vocabulary": terms, "rate_in": 48000})
        self.assertEqual(session.rate_in, 16000)
        self.assertEqual(len(session.params["keyterms"]), 100)
        self.assertTrue(all(len(t) <= 50 for t in session.params["keyterms"]))

    def test_join_transcript_spacing(self):
        self.assertEqual(target._rt_join_transcript("こんにちは。", "元気です"), "こんにちは。元気です")
        self.assertEqual(target._rt_join_transcript("Hello", "world"), "Hello world")
        self.assertEqual(target._rt_join_transcript("", " a "), "a")

    def test_receive_loop_accumulates_finals_and_stops_on_done(self):
        session = _session()
        ws = _FakeWs([
            {"type": "transcript.partial", "text": "こんに", "is_final": False, "speech_final": False},
            {"type": "transcript.partial", "text": "こんにちは。", "is_final": True, "speech_final": False},
            {"type": "transcript.partial", "text": "元気", "is_final": False, "speech_final": False},
            {"type": "transcript.partial", "text": "元気です。", "is_final": True, "speech_final": True},
            {"type": "transcript.done", "text": "こんにちは。元気です。", "duration": 2.1},
        ])
        asyncio.run(target._rt_xai_stt_receive_loop(session, ws))
        self.assertEqual(session.user_transcript, "こんにちは。元気です。")
        self.assertTrue(session.stop_event.is_set())
        types = [e["type"] for e in session.pending]
        self.assertIn("turn_complete", types)
        shown = [e["delta"] for e in session.pending if e["type"] == "transcript"]
        self.assertEqual(shown[0], "こんに")
        self.assertEqual(shown[2], "こんにちは。元気")

    def test_receive_loop_reports_provider_error(self):
        session = _session()
        ws = _FakeWs([{"type": "error", "message": "bad audio"}])
        asyncio.run(target._rt_xai_stt_receive_loop(session, ws))
        self.assertEqual(session.status, "error")
        self.assertIn("bad audio", session.error)

    def test_send_loop_streams_binary_then_audio_done(self):
        session = _session()
        session.audio_in.put(("audio", b"\x00\x01" * 10))
        session.audio_in.put(("commit",))
        ws = _FakeWs([])
        asyncio.run(target._rt_xai_stt_send_loop(session, ws))
        self.assertEqual(ws.sent[0], b"\x00\x01" * 10)
        self.assertEqual(json.loads(ws.sent[1]), {"type": "audio.done"})

    def test_source_wiring(self):
        source = read_app_source()
        self.assertIn('url = f"wss://{_XAI_API_HOST}/v1/stt?{urlencode(query)}"', source)
        self.assertIn('("interim_results", "true")', source)
        self.assertIn("elif _rt_is_live_transcription_session(session):", source)
        self.assertIn('user_text = "音声文字起こし" if live_transcript else ""', source)

    def test_web_and_android_expose_live_model(self):
        assets = list((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1)
        js = assets[0].read_text(encoding="utf-8")
        self.assertRegex(js, r'id:\s*"grok-voice-transcribe-2.0"[^}]*name:\s*"Grok Voice Transcribe 2.0 \(Live\)"')
        self.assertIn("'grok-voice-transcribe-2.0'\n        ]);", js)
        self.assertIn("isXaiLiveTranscribeModel()", js)
        self.assertIn("Grok は最大100語", read_chat_markup())
        studio = (APP_ROOT / "android" / "app" / "src" / "main" / "java" / "com" / "minashin1120"
                  / "aiplayground" / "ui" / "RealtimeStudio.kt").read_text(encoding="utf-8")
        self.assertIn('model == "grok-voice-transcribe-2.0"', studio)


if __name__ == "__main__":
    unittest.main()
