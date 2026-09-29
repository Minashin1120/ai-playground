"""Audio models other than speech-to-speech: TTS, file / microphone transcription,
Lyria RealTime and the shared ffmpeg audio conversion."""
import json
import subprocess
import threading
import types as pytypes
import unittest
from pathlib import Path
from unittest import mock

import app as target
from tests.test_realtime_provider_protocols import _FakeRedis

APP_ROOT = Path(__file__).resolve().parents[1]


def _lavfi_audio(muxer, seconds=1):
    return subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-f", muxer, "pipe:1"],
        check=True, capture_output=True,
    ).stdout


class AudioConversionTests(unittest.TestCase):
    def test_demuxer_is_detected_from_the_header(self):
        for muxer, expected in (("wav", "wav"), ("mp3", "mp3"), ("ogg", "ogg"), ("webm", "matroska"),
                                ("flac", "flac"), ("adts", "aac")):
            data = _lavfi_audio(muxer)
            self.assertEqual(target._audio_ffmpeg_demuxer(data), expected, muxer)
            pcm = target._convert_audio_to_pcm(data, rate=16000)
            self.assertGreater(len(pcm), 16000, muxer)

    def test_playlists_and_unknown_input_are_rejected(self):
        for data in (b"#EXTM3U\n#EXTINF:1,\nfile:///etc/passwd\n", b"ffconcat version 1.0\nfile /etc/passwd\n", b""):
            with self.assertRaises(ValueError):
                target._convert_audio_to_pcm(data)

    def test_ffmpeg_input_is_restricted_to_the_pipe(self):
        source = (APP_ROOT / "server" / "providers.py").read_text(encoding="utf-8")
        block = source[source.index("def _convert_audio_to_pcm"):source.index("def _pcm_to_wav_bytes")]
        self.assertIn('"-protocol_whitelist", "pipe"', block)
        self.assertIn('"-f", demuxer', block)


class GeminiTranscribeTests(unittest.TestCase):
    def test_mime_types_follow_the_supported_list(self):
        self.assertEqual(target._gemini_transcribe_mime("audio/webm;codecs=opus", "rec.webm"), "audio/webm")
        self.assertEqual(target._gemini_transcribe_mime("audio/x-m4a", "a.m4a"), "audio/m4a")
        self.assertEqual(target._gemini_transcribe_mime("application/octet-stream", "a.ogg"), "audio/ogg")
        self.assertEqual(target._gemini_transcribe_mime("audio/mpeg", "a.mp3"), "audio/mpeg")

    def test_webm_is_no_longer_truncated_by_conversion(self):
        source = (APP_ROOT / "server" / "background.py").read_text(encoding="utf-8")
        block = source[source.index("# Gemini Transcribe (audio file -> text"):source.index("# Gemini TTS (Preview)")]
        self.assertNotIn("_convert_audio_to_pcm", block)
        self.assertIn("_gemini_transcribe_mime(audio_mime, audio_name)", block)
        self.assertIn('== "FAILED"', block)
        # custom_vocabulary cannot be combined with diarization / word timestamps.
        self.assertIn("wants_word_info", block)

    def test_diarization_and_timestamps_are_rendered(self):
        words = [
            {"type": "word_info", "text": "Hello", "speaker": "spk_1", "start_offset": "0.100s", "end_offset": "0.4s"},
            {"type": "word_info", "text": "world", "speaker": "spk_1", "start_offset": "0.5s", "end_offset": "0.8s"},
            {"type": "word_info", "text": "こんにちは", "speaker": "spk_2", "start_offset": "61.2s", "end_offset": "62s"},
        ]
        self.assertEqual(
            target._format_gemini_word_info(words, True, True),
            "[00:00.1] spk_1: Hello world\n[01:01.2] spk_2: こんにちは",
        )
        self.assertEqual(target._format_gemini_word_info(words, True, False), "spk_1: Hello world\nspk_2: こんにちは")

    def test_rest_response_uses_word_annotations_when_requested(self):
        body = {"status": "completed", "steps": [{"type": "model_output", "content": [{
            "type": "text", "text": "Hi there",
            "annotations": [
                {"type": "word_info", "text": "Hi", "speaker": "spk_1"},
                {"type": "word_info", "text": "there", "speaker": "spk_2"},
            ],
        }]}]}
        resp = mock.Mock(status_code=200)
        resp.json.return_value = body
        with mock.patch.object(target.httpx, "post", return_value=resp):
            plain = target._gemini_transcribe_rest("k", "uri", "audio/wav", {"language_codes": []})
            diarized = target._gemini_transcribe_rest("k", "uri", "audio/wav", {"mode": {"type": "verbatim", "diarization_mode": "speaker"}})
        self.assertEqual(plain, "Hi there")
        self.assertEqual(diarized, "spk_1: Hi\nspk_2: there")


class TtsTests(unittest.TestCase):
    def setUp(self):
        self.source = (APP_ROOT / "server" / "background.py").read_text(encoding="utf-8")

    def test_xai_tts_voice_and_errors(self):
        block = self.source[self.source.index("elif 'grok-tts' in model_key"):self.source.index("# OpenAI TTS (default")]
        self.assertIn("options.get('tts_voice_custom')", block)
        self.assertIn('("eve", "ara", "rex", "sal", "leo")', block)
        self.assertIn('xAI TTS error {resp.status_code}', block)
        self.assertNotIn("resp.raise_for_status()", block)

    def test_gemini_tts_joins_parts_and_detects_wav(self):
        block = self.source[self.source.index("# Gemini TTS (Preview)"):self.source.index("# Image Generation")]
        self.assertIn("for p0 in parts0:", block)
        self.assertIn('audio_bytes[:4] == b"RIFF"', block)
        self.assertIn('re.search(r"rate=(\\d+)", audio_mime)', block)


class MicTranscriptionTests(unittest.TestCase):
    def test_openai_llm_transcription_uses_chat_completions_audio_model(self):
        wav = _lavfi_audio("wav")
        created = {}

        class _Completions:
            def create(self, **kwargs):
                created.update(kwargs)
                msg = pytypes.SimpleNamespace(content="こんにちは")
                return pytypes.SimpleNamespace(choices=[pytypes.SimpleNamespace(message=msg)])

        client = pytypes.SimpleNamespace(chat=pytypes.SimpleNamespace(completions=_Completions()))
        user = pytypes.SimpleNamespace(openai_api_key=None)
        with mock.patch.object(target, "_get_openai_client", return_value=client), \
                mock.patch.object(target, "_get_model_specific_api_key", return_value="sk-test"), \
                mock.patch.object(target, "get_user_llm_transcribe_prompt", return_value="文字起こし"):
            text = target._transcribe_audio_with_llm(wav, "rec.wav", "gpt-5.5", user)
        self.assertEqual(text, "こんにちは")
        self.assertEqual(created["model"], target.OPENAI_LLM_TRANSCRIBE_AUDIO_MODEL)
        self.assertEqual(created["messages"][0]["content"][1]["type"], "input_audio")
        self.assertTrue(target._is_openai_audio_input_model("gpt-audio-1.5"))
        self.assertFalse(target._is_openai_audio_input_model("gpt-5.5"))

    def test_xai_batch_stt_sends_wav_for_browser_webm(self):
        webm = _lavfi_audio("webm")
        sent = {}

        def fake_post(url, headers=None, data=None, files=None, timeout=None):
            sent["url"] = url
            sent["file"] = files[0][1]
            return mock.Mock(status_code=200, json=lambda: {"text": "ok"})

        user = pytypes.SimpleNamespace(xai_api_key=None)
        with target.app.test_request_context(), \
                mock.patch.object(target, "_get_model_specific_api_key", return_value="xai-test"), \
                mock.patch.object(target.requests, "post", side_effect=fake_post):
            resp = target._transcribe_with_xai_stt(webm, "rec.webm", "grok-voice-transcribe-2.0", user)
        self.assertEqual(resp.get_json()["transcript"], "ok")
        name, content, mime = sent["file"]
        self.assertEqual(name, "rec.wav")
        self.assertEqual(mime, "audio/wav")
        self.assertEqual(content[:4], b"RIFF")
        self.assertTrue(sent["url"].endswith("/v1/stt"))


class NewAudioModelTests(unittest.TestCase):
    def test_realtime_2_1_models_use_the_openai_realtime_session(self):
        for model in ("gpt-realtime-2.1", "gpt-realtime-2.1-mini"):
            self.assertIn(model, target.ALL_VALID_MODEL_IDS)
            self.assertEqual(target.get_sts_provider(model), "openai")
            self.assertTrue(target._rt_is_conversation_model(model))
            self.assertEqual(target._mobile_model_metadata(model)["mode"], "realtime_audio")
            params = target._normalize_rt_params("openai", model, {"voice": "marin", "rate_in": "16000"})
            self.assertEqual((params["voice"], params["rate_in"]), ("marin", 24000))

    def test_gemini_3_8_tts_models_use_the_gemini_tts_branch(self):
        for model in ("gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts"):
            self.assertIn(model, target.ALL_VALID_MODEL_IDS)
            self.assertTrue(target.is_gemini_model_key(model))
            self.assertFalse(target.is_gemini_image_model_key(model))
            self.assertFalse(target.is_gemini_transcribe_model_key(model))
            self.assertEqual(target._mobile_model_metadata(model)["mode"], "tts")

    def test_new_models_are_listed_in_the_web_catalog(self):
        source = (APP_ROOT / "static" / "js" / "chat_core_parts" / "chat_core.part05_settings_modal.js").read_text(encoding="utf-8")
        for model in ("gpt-realtime-2.1", "gpt-realtime-2.1-mini", "gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts"):
            self.assertIn(f'{{ id: "{model}", implementedAt: "2026-09-29"', source)
        sts_block = source[source.index("const STS_MODELS = new Set(["):]
        sts_block = sts_block[:sts_block.index("]);")]
        self.assertIn("'gpt-realtime-2.1'", sts_block)
        self.assertIn("'gpt-realtime-2.1-mini'", sts_block)


class LyriaRealtimeBridgeTests(unittest.TestCase):
    def test_commands_and_save_reach_the_owner(self):
        fake = _FakeRedis()
        session = target.LyriaSession("lyria_bridge_test", 1, "key", [{"text": "jazz", "weight": 1.0}], {})
        session.bridged = True

        def fake_finish(sess, thread_id):
            return {"status": "ok", "thread_id": thread_id}, 200

        with mock.patch.object(target, "redis_conn", fake), \
                mock.patch.object(target, "_lyria_finish_and_save", side_effect=fake_finish):
            pump = threading.Thread(target=target._lyria_input_pump, args=(session,), daemon=True)
            pump.start()
            target._lyria_send_command(session.session_id, {"type": "control", "action": "PAUSE"})
            target._lyria_send_command(session.session_id, {"type": "save", "thread_id": "9"})
            result = fake.blpop([target._lyria_key("res", session.session_id)], timeout=5)
            pump.join(timeout=5)
        self.assertEqual(session.cmd_queue.get_nowait(), {"type": "control", "action": "PAUSE"})
        payload = json.loads(result[1])
        self.assertEqual(payload["thread_id"], "9")
        self.assertEqual(payload["_status"], 200)

    def test_audio_events_are_published(self):
        fake = _FakeRedis()
        session = target.LyriaSession("lyria_event_test", 1, "key", [], {})
        session.bridged = True
        with mock.patch.object(target, "redis_conn", fake):
            target._lyria_push_event(session, {"audio": "AAAA"})
        raw = fake.blpop([target._lyria_key("ev", session.session_id)], timeout=1)[1]
        self.assertEqual(json.loads(raw), {"audio": "AAAA"})

    def test_stream_sends_status_snapshot_not_the_whole_recording(self):
        source = (APP_ROOT / "server" / "routes_realtime.py").read_text(encoding="utf-8")
        block = source[source.index("def gemini_music_stream"):source.index("def gemini_music_command")]
        self.assertIn("'snapshot': True", block)
        self.assertNotIn("audio_buffer", block)
        self.assertNotIn("LYRIA_SESSIONS.get", source)


if __name__ == "__main__":
    unittest.main()
