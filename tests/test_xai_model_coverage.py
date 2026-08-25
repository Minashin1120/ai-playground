from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "app.py").read_text(encoding="utf-8")
JS = next((ROOT / "static/js").glob("chat_core.v4.8.*.js")).read_text(encoding="utf-8")
HTML = (ROOT / "templates/chat.html").read_text(encoding="utf-8")
SETUP = (ROOT / "templates/setup.html").read_text(encoding="utf-8")


class XaiModelCoverageRegressionTests(unittest.TestCase):
    def test_current_xai_models_are_registered_everywhere(self):
        current = (
            "grok-4.20-0309-reasoning",
            "grok-4.20-0309-non-reasoning",
            "grok-4.20-multi-agent-0309",
            "grok-imagine-video-1.5",
            "grok-voice-think-fast-2.0",
        )
        for model_id in current:
            self.assertIn(model_id, APP)
            self.assertIn(model_id, JS)
            self.assertIn(model_id, SETUP)

    def test_xai_chat_parameters_are_forwarded(self):
        for key in (
            "xai_temperature", "xai_top_p", "xai_max_completion_tokens",
            "xai_seed", "xai_presence_penalty", "xai_frequency_penalty",
            "xai_stop", "xai_response_format", "xai_tool_choice",
            "xai_parallel_tool_calls", "xai_logprobs", "xai_top_logprobs",
        ):
            self.assertIn(key, APP)
            self.assertIn(key, JS)
        self.assertIn('"max_tokens": _optional_int("xai_max_completion_tokens", 1)', APP)
        self.assertIn('create_kwargs["response_format"] = response_format', APP)
        self.assertIn('create_kwargs["parallel_tool_calls"]', APP)

    def test_image_and_video_parameters_are_wired(self):
        for key in (
            "grok_image_count", "grok_image_format", "grok_video_duration",
            "grok_video_aspect", "grok_video_resolution",
        ):
            self.assertIn(key, APP)
            self.assertIn(key, JS)
        self.assertIn('"n": image_count', APP)
        self.assertIn('"response_format": img_response_format', APP)
        self.assertIn('"model": model_key', APP)
        self.assertIn('value="1080p"', HTML)

    def test_voice_alias_is_normalized_to_current_model(self):
        self.assertIn('"grok-voice-latest": "grok-voice-think-fast-2.0"', APP)
        self.assertIn('XAI_STS_MODEL_ALIASES.get(model_key, model_key)', APP)


if __name__ == "__main__":
    unittest.main()
