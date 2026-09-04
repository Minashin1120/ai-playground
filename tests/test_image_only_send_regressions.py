from pathlib import Path
import unittest


from tests.app_source import read_app_source
APP_ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = read_app_source()


def _current_chat_js():
    assets = sorted((APP_ROOT / "static/js").glob("chat_core.v4.8.*.js"))
    assert len(assets) == 1, "Expected only the latest versioned chat JS asset"
    return assets[0].read_text(encoding="utf-8")


class ImageOnlySendRegressionTests(unittest.TestCase):
    def test_chat_stream_accepts_empty_message_when_attachments_present(self):
        route = APP_SOURCE[APP_SOURCE.index("def chat_stream():") :]
        route = route[: route.index("@app.route('/chat_stream_resume'")]
        message_validation = route[: route.index("model_key = str(data.get('model') or '').strip()")]

        self.assertIn("raw_img_hint = data.get('image_urls') or []", message_validation)
        self.assertIn("has_attachment_hint = bool([u for u in raw_img_hint if u])", message_validation)
        self.assertIn("if not raw_message.strip() and not has_attachment_hint:", message_validation)
        # The unconditional strip() rejection that produced the 400 must be gone.
        self.assertNotIn("not raw_message.strip() or len(raw_message) > 500_000", message_validation)

    def test_browser_fast_save_accepts_empty_message_when_attachments_present(self):
        route = APP_SOURCE[APP_SOURCE.index("def save_browser_fast_mode_chat():") :]
        message_validation = route[: route.index("if not isinstance(assistant_text, str)")]

        self.assertIn("fast_refs = data.get('image_urls') or []", message_validation)
        self.assertIn("if not user_text.strip() and not bool([r for r in fast_refs if r]):", message_validation)
        self.assertNotIn("not user_text.strip() or len(user_text) > 500_000", message_validation)

    def test_gemini_branch_skips_empty_text_part(self):
        branch = APP_SOURCE[APP_SOURCE.index("log_force(\"Routing: Gemini Branch\")") :]
        build = branch[: branch.index("_collect_grounding(")]
        self.assertIn("curr_parts = []", build)
        self.assertIn("if final_message_text and str(final_message_text).strip():", build)
        self.assertIn("curr_parts.append(types.Part(text=final_message_text))", build)

    def test_claude_branch_skips_empty_text_part(self):
        branch = APP_SOURCE[APP_SOURCE.index("log_force(\"Routing: Claude Branch\")") :]
        build = branch[: branch.index("claude_kwargs = {")]
        self.assertIn("curr_parts = []", build)
        self.assertIn("if final_message_text and str(final_message_text).strip():", build)
        self.assertIn('curr_parts.append({"type": "text", "text": final_message_text})', build)

    def test_responses_api_branch_skips_empty_text_part(self):
        branch = APP_SOURCE[APP_SOURCE.index("log_force(\"Routing: Responses API Branch\")") :]
        build = branch[: branch.index("input_data.append({\"role\": \"user\", \"content\": curr_content})")]
        self.assertIn('curr_content.append({"type": text_type, "text": message_text})', build)
        self.assertIn("if message_text and str(message_text).strip():", build)

    def test_grok_native_branch_guards_empty_text(self):
        branch = APP_SOURCE[APP_SOURCE.index("log_force(\"Routing: Grok Branch (Native SDK)\")") :]
        build = branch[: branch.index("_mark_provider_request_started()\n                stream = chat_session.stream()")]
        self.assertIn("curr_user_content = []", build)
        self.assertIn("if curr_user_content:", build)
        self.assertIn("chat_session.append(x_user(*curr_user_content))", build)

    def test_deepseek_and_kimi_image_only_use_vision_analysis_as_user_turn(self):
        for branch_marker in (
            "Routing: DeepSeek V4 Branch (Chat Completions)",
            "Routing: Kimi K3 Branch (Chat Completions)",
        ):
            branch = APP_SOURCE[APP_SOURCE.index(f'log_force("{branch_marker}")') :]
            build = branch[: branch.index("messages.append({\"role\": \"user\", \"content\": user_text})")]
            self.assertIn("# Image-only send: use the vision analysis as the user turn.", build)
            self.assertIn("if user_text.strip():", build)
            self.assertIn("messages.append({\"role\": \"system\", \"content\": analysis_block})", build)

    def test_client_allows_image_only_send(self):
        source = _current_chat_js()
        self.assertIn("if(!rawText.trim() && imageUrlsToSend.length === 0) return;", source)
        self.assertIn("image_urls: imageUrlsToSend,", source)


if __name__ == "__main__":
    unittest.main()
