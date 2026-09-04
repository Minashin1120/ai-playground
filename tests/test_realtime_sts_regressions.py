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


class RealtimeStsRegressionTests(unittest.TestCase):

    def test_backend_session_endpoints_exist(self):
        for route in (
            "('/api/realtime/start'",
            "('/api/realtime/stream'",
            "('/api/realtime/audio'",
            "('/api/realtime/commit'",
            "('/api/realtime/cancel'",
            "('/api/realtime/save'",
        ):
            self.assertIn(route, APP_SOURCE, route)

    def test_backend_session_helpers_exist(self):
        for symbol in (
            "class RtSession",
            "RT_SESSIONS",
            "RT_SESSIONS_LOCK",
            "def _rt_worker",
            "def _rt_get_session",
            "def _rt_is_conversation_model",
            "def _normalize_rt_params",
            "def _rt_push_event",
            "async def _rt_openai_xai_session_async",
            "async def _rt_gemini_session_async",
            "async def _rt_openai_xai_receive_loop",
            "async def _rt_gemini_receive_loop",
            "BidiGenerateContent",
            "input_audio_buffer.speech_started",
            "input_audio_buffer.speech_stopped",
            "server_vad",
        ):
            self.assertIn(symbol, APP_SOURCE, symbol)

    def test_rt_conversation_model_classification(self):
        from app import _rt_is_conversation_model

        # Conversation STS models -> True
        self.assertTrue(_rt_is_conversation_model("gpt-realtime-2"))
        self.assertTrue(_rt_is_conversation_model("gpt-realtime-translate"))
        self.assertTrue(_rt_is_conversation_model("grok-voice-think-fast-2.0"))
        self.assertTrue(_rt_is_conversation_model("grok-voice-latest"))
        self.assertTrue(_rt_is_conversation_model("gemini-2.5-flash-native-audio-preview-12-2025"))
        # One-shot / browser-direct / transcription -> False
        self.assertFalse(_rt_is_conversation_model("gpt-transcribe"))
        self.assertFalse(_rt_is_conversation_model("gpt-live-transcribe"))
        self.assertFalse(_rt_is_conversation_model("gpt-realtime-whisper"))
        self.assertFalse(_rt_is_conversation_model("gemini-3.1-flash-live-preview"))
        self.assertFalse(_rt_is_conversation_model("gemini-3.5-live-translate-preview"))
        self.assertFalse(_rt_is_conversation_model("gpt-4o"))

    def test_rt_params_normalization(self):
        from app import _normalize_rt_params

        # OpenAI
        p = _normalize_rt_params("openai", "gpt-realtime-2", {
            "voice": "coral", "speed": "1.2", "rate_in": "24000", "rate_out": "24000"
        })
        self.assertEqual(p["voice"], "coral")
        self.assertEqual(p["speed"], 1.2)
        self.assertEqual(p["rate_in"], 24000)
        self.assertEqual(p["rate_out"], 24000)
        # Invalid voice falls back to alloy.
        p2 = _normalize_rt_params("openai", "gpt-realtime-2", {"voice": "bogus"})
        self.assertEqual(p2["voice"], "alloy")

        # xAI
        p3 = _normalize_rt_params("xai", "grok-voice-think-fast-2.0", {"voice": "eve"})
        self.assertEqual(p3["voice"], "eve")
        p4 = _normalize_rt_params("xai", "grok-voice-think-fast-2.0", {"voice": "bogus"})
        self.assertEqual(p4["voice"], "Ara")

        # Gemini native-audio
        p5 = _normalize_rt_params("google", "gemini-2.5-flash-native-audio-preview-12-2025", {
            "voice": "Kore", "thinking_level": "high", "include_thoughts": True
        })
        self.assertEqual(p5["voice"], "Kore")
        self.assertEqual(p5["thinking_level"], "high")
        self.assertTrue(p5["include_thoughts"])

    def test_rt_resolve_api_key_shape(self):
        self.assertIn("def _rt_resolve_api_key(", APP_SOURCE)

    def test_js_controller_exist(self):
        for symbol in (
            "class RealtimeVoiceSession",
            "const rtVoiceSession",
            "isRealtimeSessionModel",
            "pcm16FromFloat32",
            "/api/realtime/start",
            "/api/realtime/stream",
            "/api/realtime/audio",
            "/api/realtime/commit",
            "/api/realtime/save",
        ):
            self.assertIn(symbol, CHAT_JS, symbol)

    def test_mic_handler_routes_realtime_session(self):
        segment = CHAT_JS[CHAT_JS.index("get('mic-btn').onclick"):]
        self.assertIn("rtVoiceSession.isActive()", segment)
        self.assertIn("rtVoiceSession.stop()", segment)
        self.assertIn("await rtVoiceSession.start()", segment)
        self.assertIn("isRealtimeSessionModel()", segment)

    def test_cancel_recording_handles_realtime_session(self):
        segment = CHAT_JS[CHAT_JS.index("function cancelRecording()"):]
        self.assertIn("rtVoiceSession.isActive()", segment)
        self.assertIn("rtVoiceSession._cancel()", segment)

    def test_voice_studio_close_cancels_realtime_session(self):
        segment = CHAT_JS[CHAT_JS.index("const VoiceStudio"):CHAT_JS.index("VoiceStudio.init()")]
        self.assertIn("rtVoiceSession.isActive()", segment)

    def test_realtime_stream_uses_sse_events(self):
        segment = CHAT_JS[CHAT_JS.index("class RealtimeVoiceSession"):CHAT_JS.index("const rtVoiceSession")]
        for token in (
            "speech_started",
            "speech_stopped",
            "response_done",
            "turn_complete",
            "RealTimeAudioPlayer",
            "X-CSRF-Token",
            "application/octet-stream",
        ):
            self.assertIn(token, segment, token)

    def test_lyria_and_voice_studio_still_present(self):
        # Previous features must remain intact.
        self.assertIn("LyriaRealtimeStudio", CHAT_JS)
        self.assertIn("const VoiceStudio", CHAT_JS)
        self.assertIn("class GeminiLiveClient", CHAT_JS)


if __name__ == "__main__":
    unittest.main()
