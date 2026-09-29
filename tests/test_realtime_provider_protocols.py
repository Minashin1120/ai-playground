"""Speech-to-speech session protocols (OpenAI GA / translation, xAI, Gemini Live)
and the Redis bridge that lets any gunicorn worker serve a session's requests."""
import asyncio
import json
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import app as target

APP_ROOT = Path(__file__).resolve().parents[1]


class _FakeWs:
    def __init__(self, messages=None):
        self.messages = [m if isinstance(m, (str, bytes)) else json.dumps(m) for m in (messages or [])]
        self.sent = []

    async def recv(self):
        if not self.messages:
            raise AssertionError("no more messages")
        return self.messages.pop(0)

    async def send(self, data):
        self.sent.append(data)


class _FakePipeline:
    def __init__(self, redis):
        self.redis = redis
        self.calls = []

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return self
        return call

    def execute(self):
        return [getattr(self.redis, name)(*args, **kwargs) for name, args, kwargs in self.calls]


class _FakeRedis:
    """Enough of redis-py for the realtime bridge (lists + hashes)."""

    def __init__(self):
        self.data = {}
        self.cond = threading.Condition()

    def pipeline(self):
        return _FakePipeline(self)

    def rpush(self, key, value):
        with self.cond:
            if isinstance(value, str):
                value = value.encode()
            self.data.setdefault(key, []).append(value)
            self.cond.notify_all()

    def blpop(self, keys, timeout=0):
        deadline = time.time() + (timeout or 0)
        with self.cond:
            while True:
                for key in keys:
                    if self.data.get(key):
                        return key.encode(), self.data[key].pop(0)
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self.cond.wait(remaining)

    def hset(self, key, field=None, value=None, mapping=None):
        with self.cond:
            h = self.data.setdefault(key, {})
            if mapping:
                h.update({k.encode(): str(v).encode() for k, v in mapping.items()})
            if field is not None:
                h[field.encode()] = str(value).encode()

    def hgetall(self, key):
        return dict(self.data.get(key) or {})

    def expire(self, key, seconds):
        return True

    def exists(self, key):
        return 1 if key in self.data else 0

    def delete(self, *keys):
        with self.cond:
            for key in keys:
                self.data.pop(key, None)


def _session(provider, model, data=None):
    params = target._normalize_rt_params(provider, model, data or {})
    return target.RtSession("rt_test", 1, model, "key", params)


class OpenAiRealtimeTests(unittest.TestCase):
    def test_conversation_session_uses_ga_shape(self):
        session = _session("openai", "gpt-realtime-2", {"voice": "coral", "speed": "1.2"})
        cfg = target._rt_openai_xai_session_config(session, "gpt-realtime-2")
        self.assertEqual(cfg["type"], "realtime")
        self.assertNotIn("voice", cfg)
        self.assertNotIn("speed", cfg)
        self.assertEqual(cfg["audio"]["output"]["voice"], "coral")
        self.assertEqual(cfg["audio"]["output"]["speed"], 1.2)
        self.assertEqual(cfg["audio"]["input"]["turn_detection"], {"type": "server_vad"})
        self.assertEqual(cfg["audio"]["input"]["format"], {"type": "audio/pcm", "rate": 24000})
        self.assertIn("model", cfg["audio"]["input"]["transcription"])

    def test_no_beta_header_for_ga_protocol(self):
        for rel in ("server/realtime.py", "server/settings_ai.py"):
            source = (APP_ROOT / rel).read_text(encoding="utf-8")
            self.assertNotIn('"OpenAI-Beta"', source, rel)

    def test_translation_session_config_and_language(self):
        session = _session("openai", "gpt-realtime-translate", {"target_lang": "zh-CN", "voice": "coral"})
        self.assertEqual(session.params["target_lang"], "zh")
        cfg = target._rt_openai_xai_session_config(session, "gpt-realtime-translate")
        self.assertEqual(cfg["audio"]["output"], {"language": "zh"})
        self.assertNotIn("type", cfg)
        self.assertNotIn("voice", cfg["audio"]["output"])

    def test_translation_send_loop_uses_translation_events(self):
        session = _session("openai", "gpt-realtime-translate")
        session.audio_in.put(("audio", b"\x01\x00" * 8))
        session.audio_in.put(("commit",))
        ws = _FakeWs()

        async def run():
            task = asyncio.ensure_future(target._rt_openai_translate_send_loop(session, ws))
            await asyncio.sleep(0.15)
            session.stop_event.set()
            await task

        asyncio.run(run())
        types = [json.loads(m)["type"] for m in ws.sent]
        self.assertEqual(types, ["session.input_audio_buffer.append", "session.close"])

    def test_translation_receive_loop(self):
        session = _session("openai", "gpt-realtime-translate")
        ws = _FakeWs([
            {"type": "session.input_transcript.delta", "delta": "Hello"},
            {"type": "session.output_transcript.delta", "delta": "こんにちは"},
            {"type": "session.output_audio.delta", "delta": "AAAA"},
            {"type": "session.closed"},
        ])
        asyncio.run(target._rt_openai_translate_receive_loop(session, ws))
        self.assertEqual(session.user_transcript, "Hello")
        self.assertEqual(session.assistant_transcript, "こんにちは")
        self.assertTrue(session.assistant_audio)
        self.assertTrue(session.stop_event.is_set())
        self.assertIn("response_done", [e["type"] for e in session.pending])

    def test_multi_turn_transcripts_and_recoverable_errors(self):
        session = _session("openai", "gpt-realtime-2")
        ws = _FakeWs([
            {"type": "conversation.item.input_audio_transcription.delta", "item_id": "a", "delta": "Hi"},
            {"type": "conversation.item.input_audio_transcription.completed", "item_id": "a", "transcript": "Hi there"},
            {"type": "response.output_audio_transcript.delta", "delta": "Hello!"},
            {"type": "response.done", "response": {"status": "completed"}},
            {"type": "error", "error": {"message": "buffer too small"}},
            {"type": "conversation.item.input_audio_transcription.completed", "item_id": "b", "transcript": "Bye"},
            {"type": "response.output_audio_transcript.delta", "delta": "See you"},
        ])
        asyncio.run(target._rt_openai_xai_receive_loop(session, ws))
        # The fake socket raises when drained; everything before it was applied.
        self.assertEqual(session.user_transcript, "Hi there\nBye")
        self.assertEqual(session.assistant_transcript, "Hello!\nSee you")
        notices = [e for e in session.pending if e["type"] == "notice"]
        self.assertEqual(notices[0]["message"], "buffer too small")

    def test_commit_appends_silence_only_while_speaking(self):
        for speaking, expected_appends in ((True, True), (False, False)):
            session = _session("xai", "grok-voice-think-fast-2.0")
            session.speech_active = speaking
            session.audio_in.put(("commit",))
            session.audio_in.put(("audio", b"\x01\x00"))
            ws = _FakeWs()

            async def run():
                task = asyncio.ensure_future(target._rt_openai_xai_send_loop(session, ws))
                await asyncio.sleep(0.15)
                session.stop_event.set()
                await task

            asyncio.run(run())
            types = [json.loads(m)["type"] for m in ws.sent]
            self.assertNotIn("input_audio_buffer.commit", types)
            self.assertEqual(bool(types), expected_appends)
            self.assertTrue(session.input_closed)

    def test_realtime_whisper_is_one_shot_transcription(self):
        self.assertEqual(target.STS_MODELS["gpt-realtime-whisper"].get("mode"), "transcription")
        self.assertFalse(target._rt_is_conversation_model("gpt-realtime-whisper"))


class XaiRealtimeTests(unittest.TestCase):
    def test_session_config(self):
        session = _session("xai", "grok-voice-think-fast-2.0", {"voice": "Eve", "rate_in": 16000})
        cfg = target._rt_openai_xai_session_config(session, "grok-voice-think-fast-2.0")
        self.assertEqual(cfg["voice"], "eve")
        self.assertEqual(cfg["turn_detection"], {"type": "server_vad"})
        self.assertEqual(cfg["audio"]["input"]["format"]["rate"], 16000)
        self.assertEqual(cfg["audio"]["input"]["transcription"], {"model": "grok-transcribe"})

    def test_cumulative_user_transcript(self):
        session = _session("xai", "grok-voice-think-fast-2.0")
        ws = _FakeWs([
            {"type": "conversation.item.input_audio_transcription.updated", "item_id": "i1", "transcript": "こん"},
            {"type": "conversation.item.input_audio_transcription.updated", "item_id": "i1", "transcript": "こんにちは"},
        ])
        asyncio.run(target._rt_openai_xai_receive_loop(session, ws))
        self.assertEqual(session.user_transcript, "こんにちは")


class GeminiLiveTests(unittest.TestCase):
    def test_translate_config_is_inside_generation_config(self):
        session = _session("google", "gemini-3.5-live-translate-preview", {"target_lang": "en"})
        setup = target._rt_gemini_setup_message(session)["setup"]
        self.assertNotIn("translationConfig", setup)
        self.assertEqual(setup["generationConfig"]["translationConfig"], {"targetLanguageCode": "en", "echoTargetLanguage": True})
        self.assertNotIn("speechConfig", setup["generationConfig"])
        self.assertIn("inputAudioTranscription", setup)

    def test_thinking_level_only_where_supported(self):
        for model, expected in (
            ("gemini-2.5-flash-native-audio-preview-12-2025", False),
            ("gemini-3.8-live", False),
            ("gemini-3.1-flash-live-preview", True),
            ("gemini-3.8-live-extended-thinking", True),
        ):
            session = _session("google", model, {"thinking_level": "low"})
            gen = target._rt_gemini_setup_message(session)["setup"]["generationConfig"]
            self.assertEqual("thinkingConfig" in gen, expected, model)

    def test_transcribe_setup_and_interim(self):
        session = _session("google", "gemini-3.5-transcribe-live", {"transcription_mode": "smart", "custom_vocabulary": ["Gemini"]})
        setup = target._rt_gemini_setup_message(session)["setup"]
        self.assertEqual(setup["generationConfig"]["responseModalities"], ["TEXT"])
        self.assertEqual(setup["inputAudioTranscription"], {"mode": "SMART", "customVocabulary": ["Gemini"]})
        ws = _FakeWs([
            {"serverContent": {"interimInputTranscription": {"text": "こんに"}}},
            {"serverContent": {"inputTranscription": {"text": "こんにちは。"}}},
        ])
        asyncio.run(target._rt_gemini_receive_loop(session, ws))
        shown = [e["delta"] for e in session.pending if e.get("role") == "user"]
        self.assertEqual(shown, ["こんに", "こんにちは。"])
        self.assertTrue(target._rt_is_transcription_session(session))

    def test_commit_sends_audio_stream_end(self):
        session = _session("google", "gemini-3.8-live")
        session.audio_in.put(("commit",))
        ws = _FakeWs()

        async def run():
            task = asyncio.ensure_future(target._rt_gemini_send_loop(session, ws))
            await asyncio.sleep(0.1)
            session.stop_event.set()
            await task

        asyncio.run(run())
        self.assertEqual(json.loads(ws.sent[0]), {"realtimeInput": {"audioStreamEnd": True}})


class RealtimeBridgeTests(unittest.TestCase):
    def test_commands_from_any_worker_reach_the_owner(self):
        fake = _FakeRedis()
        session = _session("openai", "gpt-realtime-2")
        session.session_id = "rt_bridge_test"
        session.bridged = True
        saved = {}

        def fake_finish(sess, thread_id):
            saved["thread_id"] = thread_id
            saved["audio"] = bytes(sess.user_audio)
            return {"status": "ok", "thread_id": "42"}, 200

        with mock.patch.object(target, "redis_conn", fake), \
                mock.patch.object(target, "_rt_finish_and_save", side_effect=fake_finish):
            pump = threading.Thread(target=target._rt_input_pump, args=(session,), daemon=True)
            pump.start()
            target._rt_send_command(session.session_id, b"A" + b"\x01\x02")
            target._rt_send_command(session.session_id, b"C")
            target._rt_send_command(session.session_id, b"F" + json.dumps({"thread_id": "7"}).encode())
            result = fake.blpop([target._rt_key("res", session.session_id)], timeout=5)
            pump.join(timeout=5)

        self.assertIsNotNone(result)
        payload = json.loads(result[1])
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["_status"], 200)
        self.assertEqual(saved, {"thread_id": "7", "audio": b"\x01\x02"})
        self.assertEqual(session.audio_in.get_nowait(), ("audio", b"\x01\x02"))
        self.assertEqual(session.audio_in.get_nowait(), ("commit",))
        self.assertFalse(fake.exists(target._rt_key("meta", session.session_id)))

    def test_events_are_published_for_the_stream(self):
        fake = _FakeRedis()
        session = _session("openai", "gpt-realtime-2")
        session.session_id = "rt_event_test"
        session.bridged = True
        with mock.patch.object(target, "redis_conn", fake):
            target._rt_push_event(session, {"type": "audio", "data": "AAAA"})
        self.assertEqual(session.pending, [])
        raw = fake.blpop([target._rt_key("ev", session.session_id)], timeout=1)[1]
        self.assertEqual(json.loads(raw), {"type": "audio", "data": "AAAA"})

    def test_routes_do_not_depend_on_worker_memory(self):
        source = (APP_ROOT / "server" / "routes_realtime.py").read_text(encoding="utf-8")
        block = source[source.index("def realtime_audio"):source.index("@app.route('/api/gemini/session'")]
        self.assertNotIn("RT_SESSIONS.get", block)
        self.assertIn("_rt_send_command", block)


class WebClientTests(unittest.TestCase):
    def setUp(self):
        assets = list((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.js = assets[0].read_text(encoding="utf-8")

    def test_gemini_translation_config_in_generation_config(self):
        self.assertIn("setupMsg.setup.generationConfig.translationConfig = config.translationConfig", self.js)
        self.assertNotIn("setupMsg.setup.translationConfig =", self.js)

    def test_output_transcription_is_not_deduplicated(self):
        self.assertNotIn("this.assistantText.includes(sc.outputTranscription.text)", self.js)

    def test_audio_posts_are_serialized(self):
        segment = self.js[self.js.index("class RealtimeVoiceSession"):self.js.index("const rtVoiceSession")]
        self.assertIn("async _flushAudio()", segment)
        self.assertIn("case 'notice':", segment)
        self.assertIn("model === 'gpt-realtime-translate'", segment)

    def test_grok_rates_are_valid(self):
        self.assertIn("const GROK_PCM_RATES = [8000,16000,22050,24000,32000,44100,48000];", self.js)


if __name__ == "__main__":
    unittest.main()
