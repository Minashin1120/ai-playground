from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = (APP_ROOT / "app.py").read_text(encoding="utf-8")


def _current_chat_js():
    assets = sorted((APP_ROOT / "static/js").glob("chat_core.v4.8.*.js"))
    assert len(assets) == 1, "Expected only the latest versioned chat JS asset"
    return assets[0].read_text(encoding="utf-8")


class ApiKeyPromptRegressionTests(unittest.TestCase):
    def test_chat_route_returns_structured_auth_error_before_message_save(self):
        route = APP_SOURCE[APP_SOURCE.index("def chat_stream():") :]
        route = route[: route.index("@app.route('/api/token_estimate'")]

        auth_check_at = route.index("_resolve_chat_model_auth(current_user, model_key)")
        message_save_at = route.index("user_msg = Message(")
        self.assertLess(auth_check_at, message_save_at)
        self.assertIn('"code": resolved_auth["error_code"]', route)
        self.assertIn('"model": model_key', route)
        self.assertIn('"provider": resolved_auth.get("provider")', route)

    def test_client_uses_server_error_code_instead_of_settings_snapshot_guess(self):
        source = _current_chat_js()

        self.assertNotIn("const checkApiKeyForModel", source)
        self.assertIn("if (e.serverCode === 'api_key_missing')", source)
        self.assertIn("showApiKeyRequiredModalAsync(missingKeyModel)", source)
        self.assertIn("if (retryAfterApiKeySetup) return sendMessage()", source)

    def test_failed_preflight_preserves_prompt_and_attachments_for_retry(self):
        source = _current_chat_js()
        fetch_at = source.index("const r = await fetchChatStreamWithUnavailableRetry(")
        accepted_at = source.index("requestAccepted = true", fetch_at)

        self.assertGreater(source.index("resetUploadState();", fetch_at), accepted_at)
        self.assertGreater(source.index("get('prompt-input').value = ''", fetch_at), accepted_at)
        self.assertIn("if (!requestAccepted)", source[fetch_at:])
        self.assertIn("optimisticUserMessageEl.remove()", source[fetch_at:])

    def test_google_tts_uses_google_auth_resolution(self):
        resolver = APP_SOURCE[APP_SOURCE.index("def _resolve_chat_model_auth") :]
        resolver = resolver[: resolver.index("def _closest_aspect_ratio")]

        self.assertIn('if "google-tts" in mk_l:', resolver)
        self.assertIn('provider = "google"', resolver)
        self.assertIn('user_or_admin_env("google_api_key", "GOOGLE_API_KEY")', resolver)


if __name__ == "__main__":
    unittest.main()
