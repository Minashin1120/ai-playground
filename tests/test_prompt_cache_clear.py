"""PromptCache clear rotates the provider key without touching an uncleared thread."""
import secrets
import unittest

from tests.app_source import read_app_source

APP_SOURCE = read_app_source()


def _load_helpers():
    start = APP_SOURCE.index("def _thread_prompt_cache_key(thread):")
    end = APP_SOURCE.index("@app.route('/chat_stream', methods=['POST'])", start)
    namespace = {"secrets": secrets}
    exec(APP_SOURCE[start:end], namespace)
    return namespace["_thread_prompt_cache_key"], namespace["_forget_prompt_cache"]


class _Thread:
    def __init__(self):
        self.public_id = "pub1"
        self.id = 9
        self.enable_prompt_caching = True
        self.prompt_cache_provider = "google"
        self.prompt_cache_nonce = None


class PromptCacheClearTests(unittest.TestCase):
    def test_route_decorator_stays_on_chat_stream(self):
        marker = "@app.route('/chat_stream', methods=['POST'])"
        window = APP_SOURCE[APP_SOURCE.index(marker):APP_SOURCE.index(marker) + 160]
        self.assertIn("@login_required", window)
        self.assertIn("def chat_stream():", window)
        self.assertNotIn("def _thread_prompt_cache_key", window)

    def test_clear_rotates_the_key_and_drops_the_lock(self):
        key_for, forget = _load_helpers()
        thread = _Thread()
        self.assertEqual(key_for(thread), "thread-pub1")
        forget(thread)
        self.assertFalse(thread.enable_prompt_caching)
        self.assertIsNone(thread.prompt_cache_provider)
        self.assertEqual(len(thread.prompt_cache_nonce), 16)
        self.assertEqual(key_for(thread), f"thread-pub1-{thread.prompt_cache_nonce}")

    def test_blank_nonce_and_missing_public_id(self):
        key_for, _forget = _load_helpers()
        thread = _Thread()
        thread.prompt_cache_nonce = "  "
        self.assertEqual(key_for(thread), "thread-pub1")
        thread.public_id = None
        thread.prompt_cache_nonce = None
        self.assertEqual(key_for(thread), "thread-9")

    def test_settings_clear_uses_the_helper(self):
        settings = APP_SOURCE[APP_SOURCE.index("def update_thread_settings"):]
        settings = settings[:settings.index("\ndef ", 1)]
        self.assertIn("if 'enable_prompt_caching' in d:", settings)
        self.assertIn("_forget_prompt_cache(t)", settings)
        self.assertIn("'enable_prompt_caching':", settings)
