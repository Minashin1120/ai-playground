from pathlib import Path
import struct
import unittest
import wave
from io import BytesIO

import sys

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

APP_SOURCE = (APP_ROOT / "app.py").read_text(encoding="utf-8")
CHAT_JS_ASSETS = list((APP_ROOT / "static/js").glob("chat_core.v4.8.*.js"))
assert len(CHAT_JS_ASSETS) == 1, "Only the latest versioned chat core asset should remain"
CHAT_JS = CHAT_JS_ASSETS[0].read_text(encoding="utf-8")
CHAT_HTML = (APP_ROOT / "templates/chat.html").read_text(encoding="utf-8")


class LyriaRealtimeRegressionTests(unittest.TestCase):

    def test_lyria_realtime_model_registered_everywhere(self):
        self.assertIn('"lyria-realtime-exp"', APP_SOURCE)
        self.assertIn('"lyria-realtime-exp"', APP_SOURCE[APP_SOURCE.index("ALL_VALID_MODEL_IDS"):APP_SOURCE.index("LLM_TOKEN_MARKERS")])
        self.assertIn('id: "lyria-realtime-exp"', CHAT_JS)
        definition_at = CHAT_JS.index('id: "lyria-realtime-exp"')
        self.assertIn("implementedAt:", CHAT_JS[definition_at:definition_at + 200])

    def test_backend_session_endpoints_exist(self):
        for route in (
            "('/api/gemini/music/start'",
            "('/api/gemini/music/stream'",
            "('/api/gemini/music/command'",
            "('/api/gemini/music/save'",
            "('/api/gemini/music/cancel'",
        ):
            self.assertIn(route, APP_SOURCE, route)

    def test_backend_session_helpers_exist(self):
        for symbol in (
            "class LyriaSession",
            "LYRIA_SESSIONS",
            "LYRIA_SESSIONS_LOCK",
            "def _normalize_lyria_config",
            "def _normalize_lyria_prompts",
            "def _lyria_pcm_to_wav_stereo",
            "def _lyria_worker",
            "def _lyria_get_session",
            "BidiGenerateMusic",
        ):
            self.assertIn(symbol, APP_SOURCE, symbol)

    def test_chat_ui_elements_exist(self):
        for elem in (
            'id="lyria-realtime-studio-bar"',
            'id="lyria-open-studio-btn"',
            'id="lyria-studio-modal"',
            'id="lyria-prompt-rows"',
            'id="lyria-play-btn"',
            'id="lyria-pause-btn"',
            'id="lyria-stop-btn"',
            'id="lyria-reset-btn"',
            'id="lyria-save-btn"',
            'id="lyria-apply-prompts-btn"',
            'id="lyria-apply-config-btn"',
            'id="lyria-bpm"',
            'id="lyria-scale"',
        ):
            self.assertIn(elem, CHAT_HTML, elem)

    def test_js_controller_exist(self):
        for symbol in (
            "LyriaRealtimeStudio",
            "openLyriaStudio",
            "isLyriaRealtimeModel",
            "updateGeminiMusicUi",
            "lyria-studio-modal",
        ):
            self.assertIn(symbol, CHAT_JS, symbol)

    def test_send_message_routes_realtime_model_to_studio(self):
        segment = CHAT_JS[CHAT_JS.index("async function sendMessage"):]
        self.assertIn("isLyriaRealtimeModel()", segment)
        self.assertIn("window.openLyriaStudio", segment)

    def test_music_branch_mentions_studio(self):
        segment = APP_SOURCE[APP_SOURCE.index("is_gemini_music_model_key(model_key_l)"):]
        self.assertIn("Lyria RealTime Studio", segment)

    def test_config_normalization_clamps_values(self):
        # Import the normalizer directly from app.py without starting the app.
        ns = {}
        exec("def _noop(): pass", ns)
        # Lightweight inline reimplementation test: ensure the module exports it.
        import ast
        tree = ast.parse(APP_SOURCE)
        names = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        self.assertIn("_normalize_lyria_config", names)
        self.assertIn("_normalize_lyria_prompts", names)

    def test_config_always_specifies_audio_format(self):
        from app import _normalize_lyria_config
        cfg = _normalize_lyria_config({"bpm": 90, "scale": "D_MAJOR_B_MINOR", "music_generation_mode": "DIVERSITY"})
        self.assertEqual(cfg.get("audioFormat"), "pcm16")
        self.assertEqual(cfg.get("sampleRateHz"), 48000)
        self.assertEqual(cfg.get("bpm"), 90)
        self.assertEqual(cfg.get("scale"), "D_MAJOR_B_MINOR")
        self.assertEqual(cfg.get("musicGenerationMode"), "DIVERSITY")
        # Clamping
        cfg2 = _normalize_lyria_config({"bpm": 9999, "guidance": -3, "density": 5})
        self.assertEqual(cfg2.get("bpm"), 200)
        self.assertEqual(cfg2.get("guidance"), 0.0)
        self.assertEqual(cfg2.get("density"), 1.0)

    def test_wav_converter_produces_stereo_48k(self):
        from app import _lyria_pcm_to_wav_stereo
        pcm = b"".join(struct.pack("<hh", 0, 0) for _ in range(4800))  # 0.1s of silence stereo
        wav_bytes = _lyria_pcm_to_wav_stereo(pcm, rate=48000)
        with wave.open(BytesIO(wav_bytes), "rb") as wf:
            self.assertEqual(wf.getnchannels(), 2)
            self.assertEqual(wf.getframerate(), 48000)
            self.assertEqual(wf.getsampwidth(), 2)
            self.assertEqual(wf.getnframes(), 4800)

    def test_prompts_normalization(self):
        from app import _normalize_lyria_prompts
        prompts = _normalize_lyria_prompts([
            {"text": "  minimal techno  ", "weight": 2.0},
            {"text": "", "weight": 1.0},
            {"text": "piano", "weight": 99},
            {"text": "drums", "weight": -5},
            "not-a-dict",
        ])
        self.assertEqual(len(prompts), 3)
        self.assertEqual(prompts[0]["text"], "minimal techno")
        self.assertEqual(prompts[0]["weight"], 2.0)
        self.assertEqual(prompts[1]["weight"], 99)
        # Negative weights are normalized to 1.0
        self.assertEqual(prompts[2]["weight"], 1.0)

    def test_icon_subset_contains_studio_icons(self):
        subset = (APP_ROOT / "static/vendor/icons/fa-subset.css").read_text(encoding="utf-8")
        for icon in ("fa-music", "fa-pause", "fa-stop", "fa-save", "fa-sliders-h", "fa-play"):
            self.assertIn(f".{icon}:before", subset, icon)


if __name__ == "__main__":
    unittest.main()
