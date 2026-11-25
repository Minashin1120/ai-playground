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
            "translationConfig",
            "customVocabulary",
            "gemini-3.5-transcribe-live",
            "GEMINI_LIVE_CONSTRAINED_URL",
            "auth_tokens.create",
            "new_session_expire_time",
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
        # One-shot transcription models remain excluded, while Gemini Live is
        # now proxied through the authenticated server session for native apps.
        self.assertFalse(_rt_is_conversation_model("gpt-transcribe"))
        self.assertFalse(_rt_is_conversation_model("gpt-live-transcribe"))
        self.assertFalse(_rt_is_conversation_model("gpt-realtime-whisper"))
        self.assertTrue(_rt_is_conversation_model("gemini-3.1-flash-live-preview"))
        self.assertTrue(_rt_is_conversation_model("gemini-3.5-live-translate-preview"))
        self.assertTrue(_rt_is_conversation_model("gemini-3.5-transcribe-live"))
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
        self.assertEqual(p4["voice"], "ara")
        # The Web / Android selects show "Rex"; xAI voice IDs are lowercase.
        self.assertEqual(_normalize_rt_params("xai", "grok-voice-think-fast-2.0", {"voice": "Rex"})["voice"], "rex")
        p4r = _normalize_rt_params("xai", "grok-voice-think-fast-2.0", {"rate_in": "22050", "rate_out": "21050"})
        self.assertEqual(p4r["rate_in"], 22050)
        self.assertEqual(p4r["rate_out"], 24000)
        # OpenAI Realtime is fixed at 24 kHz even if a stale xAI rate is sent.
        self.assertEqual(_normalize_rt_params("openai", "gpt-realtime-2", {"rate_in": "16000"})["rate_in"], 24000)

        # Gemini Live thinking levels (3.1 Flash Live).
        p5 = _normalize_rt_params("google", "gemini-3.1-flash-live-preview", {
            "voice": "Kore", "thinking_level": "high", "include_thoughts": True
        })
        self.assertEqual(p5["voice"], "Kore")
        self.assertEqual(p5["thinking_level"], "high")
        self.assertTrue(p5["include_thoughts"])
        # Gemini 2.5 native audio uses a thinking budget, not thinkingLevel.
        p5b = _normalize_rt_params("google", "gemini-2.5-flash-native-audio-preview-12-2025", {
            "thinking_level": "high", "include_thoughts": True
        })
        self.assertIsNone(p5b["thinking_level"])
        self.assertFalse(p5b["include_thoughts"])

        # Gemini Live native-client parameters are normalized without
        # exposing provider credentials to the client.
        p6 = _normalize_rt_params("google", "gemini-3.5-live-translate-preview", {
            "target_lang": "EN", "thinking_level": "high",
        })
        self.assertEqual(p6["target_lang"], "en")
        self.assertEqual(p6["transcription_mode"], "VERBATIM")
        self.assertEqual(
            _normalize_rt_params("google", "gemini-3.5-live-translate-preview", {"target_lang": "zh-CN"})["target_lang"],
            "zh-Hans",
        )
        self.assertEqual(
            _normalize_rt_params("google", "gemini-3.5-live-translate-preview", {"target_lang": "zh-TW"})["target_lang"],
            "zh-Hant",
        )
        self.assertEqual(
            _normalize_rt_params("google", "gemini-3.5-live-translate-preview", {"target_lang": "pt"})["target_lang"],
            "pt-BR",
        )
        p7 = _normalize_rt_params("google", "gemini-3.5-transcribe-live", {
            "transcription_mode": "smart", "custom_vocabulary": ["AI Playground", "  Gemini  "],
        })
        self.assertEqual(p7["transcription_mode"], "SMART")
        self.assertEqual(p7["custom_vocabulary"], ["AI Playground", "Gemini"])

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

    def test_voice_studio_close_keeps_realtime_session_in_dock(self):
        # The studio is an enlarged view of the inline voice dock: closing it moves
        # the controls back to the composer and must not cancel the conversation.
        segment = CHAT_JS[CHAT_JS.index("const VoiceStudio"):CHAT_JS.index("VoiceStudio.init()")]
        close_fn = segment[segment.index("function close()"):segment.index("function closeIfOpen")]
        self.assertIn("movePanelBack()", close_fn)
        self.assertNotIn("cancelRecording", close_fn)
        self.assertIn("function cancelRecording", CHAT_JS)
        self.assertIn("rtVoiceSession.isActive()", CHAT_JS)

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

    def test_gemini_live_uses_a_browser_direct_ephemeral_token(self):
        # Gemini Live connects directly from the browser with a one-use
        # ephemeral token (official ephemeral-token flow); the API key never
        # reaches the client.
        self.assertIn("/api/gemini/session", CHAT_JS)
        self.assertIn("access_token=", CHAT_JS)
        self.assertIn("sessionResumption", CHAT_JS)
        self.assertIn("BidiGenerateContentConstrained", APP_SOURCE)
        self.assertNotIn("/api/skillini", CHAT_JS)

    def test_gemini_live_falls_back_to_server_session(self):
        segment = CHAT_JS[CHAT_JS.index("const isRealtimeSessionModel"):]
        segment = segment[:segment.index("const getStsProvider")]
        self.assertNotIn("isGeminiLiveModel()) return false", segment)
        mic = CHAT_JS[CHAT_JS.index("get('mic-btn').onclick"):]
        self.assertIn("if (isRealtimeSessionModel())", mic)
        self.assertIn("_tryStartDirect", CHAT_JS)
        self.assertNotIn("beginGeminiLiveMic", mic)

    def test_gemini_session_issues_one_use_ephemeral_token(self):
        from unittest import mock
        import app as target

        token = mock.Mock()
        token.name = 'ephemeral-token'
        token_client = mock.Mock()
        token_client.auth_tokens.create.return_value = token
        with target.app.test_request_context(
            '/api/gemini/session', method='POST', json={'model': 'gemini-3.8-live'}
        ):
            with mock.patch.dict(target.app.config, {'LOGIN_DISABLED': True}), \
                 mock.patch.object(target, '_resolve_gemini_runtime', return_value={'backend': 'gemini_api', 'api_key': 'k'}), \
                 mock.patch.object(target, '_get_model_specific_api_key', return_value=None), \
                 mock.patch.object(target, '_get_gemini_client', return_value=token_client) as get_client:
                body = target.gemini_session().get_json()
        self.assertTrue(body['direct'])
        self.assertEqual(body['token'], 'ephemeral-token')
        self.assertIn('BidiGenerateContentConstrained', body['url'])
        self.assertEqual(body['rate_in'], 16000)
        self.assertEqual(body['rate_out'], 24000)
        get_client.assert_called_once_with(api_key='k', api_version='v1beta')
        config = token_client.auth_tokens.create.call_args.kwargs['config']
        self.assertEqual(config['uses'], 1)
        self.assertIn('expire_time', config)
        self.assertIn('new_session_expire_time', config)

    def test_gemini_session_keeps_vertex_on_the_server_session(self):
        from unittest import mock
        import app as target

        with target.app.test_request_context(
            '/api/gemini/session', method='POST', json={'model': 'gemini-3.8-live'}
        ):
            with mock.patch.dict(target.app.config, {'LOGIN_DISABLED': True}), \
                 mock.patch.object(target, '_resolve_gemini_runtime', return_value={'backend': 'vertex_ai'}), \
                 mock.patch.object(target, '_get_gemini_client') as get_client:
                body = target.gemini_session().get_json()
        self.assertEqual(body, {'direct': False, 'reason': 'vertex_ai'})
        get_client.assert_not_called()

    def test_gemini_transcribe_is_a_realtime_mobile_model(self):
        from app import _mobile_model_mode

        self.assertEqual(_mobile_model_mode("gemini-3.5-transcribe-live"), "realtime_audio")
        self.assertEqual(_mobile_model_mode("gpt-live-transcribe"), "transcription")
        self.assertEqual(_mobile_model_mode("grok-voice-transcribe-2.0"), "realtime_audio")

    def test_gemini_transcription_drains_before_save(self):
        from app import RtSession, _normalize_rt_params, _rt_drains_on_stop

        params = _normalize_rt_params("google", "gemini-3.5-transcribe-live", {})
        session = RtSession("rt_test", 1, "gemini-3.5-transcribe-live", "key", params)
        self.assertTrue(_rt_drains_on_stop(session))


if __name__ == "__main__":
    unittest.main()
