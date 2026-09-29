"""Reasoning effort for OpenAI Realtime 2.x, GPT-Live 1 sessions and Gemini 3.8 TTS
(style metadata and custom voice IDs through the Interactions API)."""
import asyncio
import base64
import json
import unittest
from pathlib import Path
from unittest import mock

import app as target
from tests.test_realtime_provider_protocols import _FakeWs, _session

APP_ROOT = Path(__file__).resolve().parents[1]


class RealtimeReasoningTests(unittest.TestCase):
    def test_reasoning_effort_is_sent_for_reasoning_models_only(self):
        session = _session("openai", "gpt-realtime-2.1", {"reasoning_effort": "XHigh"})
        cfg = target._rt_openai_xai_session_config(session, "gpt-realtime-2.1")
        self.assertEqual(cfg["reasoning"], {"effort": "xhigh"})
        self.assertNotIn("reasoning", target._rt_openai_xai_session_config(
            _session("openai", "gpt-realtime-2.1-mini"), "gpt-realtime-2.1-mini"))
        self.assertNotIn("reasoning_effort", _session("openai", "gpt-realtime-1.5", {"reasoning_effort": "high"}).params)
        self.assertNotIn("reasoning_effort", _session("openai", "gpt-realtime-2", {"reasoning_effort": "bogus"}).params)

    def test_one_shot_sts_passes_reasoning_and_rejects_gpt_live(self):
        source = (APP_ROOT / "server" / "routes_media.py").read_text(encoding="utf-8")
        self.assertIn("request.form.get('sts_reasoning_effort')", source)
        with self.assertRaises(ValueError):
            asyncio.run(target._openai_sts_realtime(b"\x00\x00", "key", "gpt-live-1"))


class GptLiveTests(unittest.TestCase):
    def test_voice_and_session_start_config(self):
        session = _session("openai", "gpt-live-1", {"voice": "quartz", "speed": "1.3"})
        self.assertEqual(session.params["voice"], "quartz")
        self.assertNotIn("speed", session.params)
        self.assertEqual(_session("openai", "gpt-live-1").params["voice"], "marin")
        self.assertEqual(_session("openai", "gpt-realtime-2.1", {"voice": "quartz"}).params["voice"], "alloy")
        cfg = target._rt_openai_live_session_config(session)
        self.assertEqual(cfg["model"], "gpt-live-1")
        self.assertEqual(cfg["audio"], {"format": {"type": "audio/pcm", "rate": 24000}, "output": {"voice": "quartz"}})
        self.assertEqual(cfg["delegation"]["type"], "responses")
        self.assertEqual(cfg["delegation"]["responses"]["model"], target.OPENAI_LIVE_DELEGATION_MODEL)
        self.assertTrue(target._rt_is_conversation_model("gpt-live-1"))
        self.assertTrue(target._rt_drains_on_stop(session))

    def test_send_loop_uses_live_events(self):
        session = _session("openai", "gpt-live-1")
        session.audio_in.put(("audio", b"\x01\x00" * 8))
        session.audio_in.put(("commit",))
        ws = _FakeWs()

        async def run():
            task = asyncio.ensure_future(target._rt_openai_live_send_loop(session, ws))
            await asyncio.sleep(0.15)
            session.stop_event.set()
            await task

        asyncio.run(run())
        self.assertEqual([json.loads(m)["type"] for m in ws.sent], ["session.input_audio.append", "session.close"])

    def test_receive_loop_groups_full_duplex_turns(self):
        session = _session("openai", "gpt-live-1")
        ws = _FakeWs([
            {"type": "session.input_transcript.delta", "delta": "What is", "start_ms": 0, "end_ms": 200},
            {"type": "session.input_transcript.delta", "delta": " the time?"},
            {"type": "session.output_transcript.delta", "delta": "It is noon."},
            {"type": "session.output_audio.delta", "delta": "AAAA"},
            {"type": "response.event", "event": {"type": "response.output_text.delta", "delta": "x"}},
            {"type": "session.usage.updated", "usage": {"seconds": 3}},
            {"type": "session.input_transcript.delta", "delta": "Thanks"},
            {"type": "session.output_transcript.delta", "delta": "You're welcome."},
            {"type": "error", "error": {"message": "moderation cut"}},
            {"type": "session.closed", "reason": "close_requested"},
        ])
        asyncio.run(target._rt_openai_live_receive_loop(session, ws))
        self.assertEqual(session.user_transcript, "What is the time?\nThanks")
        self.assertEqual(session.assistant_transcript, "It is noon.\nYou're welcome.")
        self.assertTrue(session.assistant_audio)
        self.assertTrue(session.stop_event.is_set())
        types = [e["type"] for e in session.pending]
        self.assertIn("notice", types)
        self.assertEqual(types[-1], "response_done")

    def test_worker_dispatches_live_sessions(self):
        source = (APP_ROOT / "server" / "realtime.py").read_text(encoding="utf-8")
        self.assertIn('"wss://api.openai.com/v1/live/sessions"', source)
        self.assertIn("loop.run_until_complete(_rt_openai_live_session_async(session))", source)


class Gemini38TtsTests(unittest.TestCase):
    def test_voice_resolution(self):
        self.assertEqual(target._gemini_tts_voice("Puck", ""), "Puck")
        self.assertEqual(target._gemini_tts_voice("Puck", "voice_abc-123"), "voice_abc-123")
        self.assertEqual(target._gemini_tts_voice("Puck", "voicekey_XYZ"), "voicekey_XYZ")
        self.assertEqual(target._gemini_tts_voice("bogus", "bad id!"), "Kore")

    def test_interactions_request_and_wav_response(self):
        wav = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 32 + b"\x01\x02"
        body = {"status": "completed", "steps": [
            {"type": "user_input"},
            {"type": "model_output", "content": [{"type": "audio", "mime_type": "audio/wav",
                                                   "data": base64.b64encode(wav).decode()}]},
        ]}
        resp = mock.Mock(status_code=200)
        resp.json.return_value = body
        with mock.patch.object(target.httpx, "post", return_value=resp) as post:
            audio, mime = target._gemini_tts_interactions_rest("k", "gemini-3.8-flash-tts", "Hello", "voice_1",
                                                                style="  whispered urgently ")
        self.assertEqual((audio, mime), (wav, "audio/wav"))
        url = post.call_args.args[0]
        payload = post.call_args.kwargs["json"]
        self.assertTrue(url.endswith("/v1beta/interactions"))
        self.assertEqual(payload["response_format"], {"type": "audio"})
        self.assertEqual(payload["generation_config"], {"speech_config": [{"voice": "voice_1"}]})
        content = payload["input"][0]["content"][0]
        self.assertEqual(content["text"], "Hello")
        self.assertEqual(content["annotations"], [{"type": "speech_metadata", "style": "whispered urgently"}])
        with mock.patch.object(target.httpx, "post", return_value=resp) as post:
            target._gemini_tts_interactions_rest("k", "gemini-3.8-flash-lite-tts", "Hi", "Kore")
        self.assertNotIn("annotations", post.call_args.kwargs["json"]["input"][0]["content"][0])

    def test_api_errors_are_reported(self):
        resp = mock.Mock(status_code=400, text="bad")
        resp.json.return_value = {"error": {"message": "unknown voice"}}
        with mock.patch.object(target.httpx, "post", return_value=resp):
            with self.assertRaisesRegex(RuntimeError, "unknown voice"):
                target._gemini_tts_interactions_rest("k", "gemini-3.8-flash-tts", "Hi", "voice_x")

    def test_chat_route_and_background_use_the_interactions_path(self):
        self.assertTrue(target.is_gemini_interactions_tts_model_key("gemini-3.8-flash-tts"))
        self.assertFalse(target.is_gemini_interactions_tts_model_key("gemini-3.1-flash-tts-preview"))
        background = (APP_ROOT / "server" / "background.py").read_text(encoding="utf-8")
        block = background[background.index("# Gemini TTS (Preview)"):background.index("# Image Generation")]
        self.assertIn('is_gemini_interactions_tts_model_key(model_key) and gemini_backend_mode != "vertex_ai"', block)
        self.assertIn("style=options.get('tts_style')", block)
        routes = (APP_ROOT / "server" / "routes_chat.py").read_text(encoding="utf-8")
        self.assertIn("'tts_style': data.get('tts_style')", routes)


class WebControlsTests(unittest.TestCase):
    def test_panels_and_payloads(self):
        parts = APP_ROOT / "static" / "js" / "chat_core_parts"
        settings = (parts / "chat_core.part05_settings_modal.js").read_text(encoding="utf-8")
        self.assertIn("OPENAI_LIVE_STS_MODELS.has(m)", settings)
        self.assertIn("'gpt-live-1',", settings)
        self.assertIn("GEMINI_INTERACTIONS_TTS_MODELS.has(model)", settings)
        self.assertIn("reasoning_effort: get('sts-reasoning-effort')",
                      (parts / "chat_core.part09_domcontent_popstate_modals.js").read_text(encoding="utf-8"))
        self.assertIn("fd.append('sts_reasoning_effort'",
                      (parts / "chat_core.part10_domcontent_final.js").read_text(encoding="utf-8"))
        self.assertIn("tts_style:", (parts / "chat_core.part14_send_message_browser_fast.js").read_text(encoding="utf-8"))
        panels = (APP_ROOT / "templates" / "chat" / "composer_panels.html").read_text(encoding="utf-8")
        self.assertIn('id="sts-reasoning-effort"', panels)
        media = (APP_ROOT / "templates" / "chat" / "composer_gen_media.html").read_text(encoding="utf-8")
        self.assertIn('id="tts-style"', media)


if __name__ == "__main__":
    unittest.main()
