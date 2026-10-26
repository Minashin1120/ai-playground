from pathlib import Path
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from tests.chat_template import read_chat_markup

from tests.app_source import read_app_source
ROOT = Path(__file__).resolve().parents[1]
APP = read_app_source()
JS = next((ROOT / "static/js").glob("chat_core.v4.8.*.js")).read_text(encoding="utf-8")
HTML = read_chat_markup()
SETUP = (ROOT / "templates/setup.html").read_text(encoding="utf-8")


class XaiModelCoverageRegressionTests(unittest.TestCase):
    def test_native_chat_only_sends_tool_controls_with_tools(self):
        source = (ROOT / "server/background.py").read_text(encoding="utf-8")
        native_branch = source.index('# --- 2. xAI Grok (Native SDK) ---')
        start = source.index('                create_kwargs = {"model": model_key}', native_branch)
        end = source.index('                grok_sys = ', start)
        request_code = compile(textwrap.dedent(source[start:end]), "<grok-request>", "exec")
        for tools, legacy_search in (([], None), ([], object()), ([object()], None)):
            for choice in (None, "auto", "none", "required", "invalid"):
                with self.subTest(tools=bool(tools), legacy_search=bool(legacy_search), choice=choice):
                    create = Mock()
                    started = Mock()
                    namespace = {
                        "model_key": "grok-4.20-0309-non-reasoning",
                        "model_key_l": "grok-4.20-0309-non-reasoning",
                        "options": {"xai_tool_choice": choice, "xai_parallel_tool_calls": False},
                        "search_params": legacy_search,
                        "tools": tools,
                        "include": [],
                        "grok_reasoning_effort_supported": False,
                        "grok_reasoning_supported": False,
                        "_mark_provider_request_started": started,
                        "x_client": SimpleNamespace(chat=SimpleNamespace(create=create)),
                    }
                    exec(request_code, namespace)
                    started.assert_called_once_with()
                    create.assert_called_once()
                    kwargs = create.call_args.kwargs
                    expected_choice = choice or "auto"
                    if tools:
                        self.assertEqual(kwargs["tools"], tools)
                        self.assertFalse(kwargs["parallel_tool_calls"])
                        if expected_choice in ("auto", "none", "required"):
                            self.assertEqual(kwargs["tool_choice"], expected_choice)
                        else:
                            self.assertNotIn("tool_choice", kwargs)
                    else:
                        self.assertNotIn("tools", kwargs)
                        self.assertNotIn("tool_choice", kwargs)
                        self.assertNotIn("parallel_tool_calls", kwargs)
                    if legacy_search:
                        self.assertIs(kwargs["search_parameters"], legacy_search)

    def test_current_xai_models_are_registered_everywhere(self):
        current = (
            "grok-4.20-0309-reasoning",
            "grok-4.20-0309-non-reasoning",
            "grok-4.20-multi-agent-0309",
            "grok-imagine-video-1.5",
            "grok-imagine-video-1.5-lite",
            "grok-voice-3.0",
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
        self.assertIn('"grok-voice-latest": "grok-voice-3.0"', APP)
        self.assertIn('XAI_STS_MODEL_ALIASES.get(model_key, model_key)', APP)


if __name__ == "__main__":
    unittest.main()
