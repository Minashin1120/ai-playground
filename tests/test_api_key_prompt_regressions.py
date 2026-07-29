from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


def _current_chat_js():
    assets = sorted((APP_ROOT / "static/js").glob("chat_core.v4.8.*.js"))
    assert len(assets) == 1, "Expected only the latest versioned chat JS asset"
    return assets[0].read_text(encoding="utf-8")


class ApiKeyPromptRegressionTests(unittest.TestCase):
    def test_saved_model_specific_key_is_checked_from_initial_settings_snapshot(self):
        source = _current_chat_js()
        checker = source[source.index("const checkApiKeyForModel = async") :]
        checker = checker[: checker.index("const setModelApiKeyPanelOpen")]

        self.assertIn("settings && settings.model_api_keys", checker)
        self.assertIn("Object.entries(savedModelApiKeys).some", checker)
        self.assertIn("String(savedModelId || '').toLowerCase().trim() === id", checker)
        self.assertIn("if (hasSavedModelKey) return true", checker)


if __name__ == "__main__":
    unittest.main()
