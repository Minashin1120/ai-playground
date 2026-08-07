import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class WelcomeQuickStartRegressionTests(unittest.TestCase):
    def _js(self) -> str:
        assets = list((ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1)
        return assets[0].read_text(encoding="utf-8")

    def test_welcome_buttons_are_dynamic_and_limited_to_five_recent_models(self):
        chat = (ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        js = self._js()

        self.assertIn('id="welcome-quick-start"', chat)
        self.assertNotIn("onclick=\"quickStart(", chat)
        self.assertIn("const WELCOME_QUICK_START_LIMIT = 5", js)
        self.assertIn("getRecentModelsForQuickStart", js)
        self.assertIn("renderWelcomeQuickStart", js)
        self.assertIn("!m.deprecated && m.implementedAt", js)

        # Parse implementedAt metadata for non-deprecated models and compute top 5.
        items = []
        for m in re.finditer(
            r'\{\s*id:\s*"([^"]+)",\s*implementedAt:\s*"([^"]+)",\s*implementedRank:\s*(\d+)',
            js,
        ):
            mid, date, rank = m.group(1), m.group(2), int(m.group(3))
            window = js[m.start() : m.start() + 500]
            deprecated = "deprecated: true" in window
            items.append((date, rank, mid, deprecated))

        active = [it for it in items if not it[3]]
        self.assertGreaterEqual(len(active), 5)
        active.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        top5 = [mid for _, _, mid, _ in active[:5]]

        # Most recently implemented set (from changelogs through V4.8.661).
        self.assertEqual(
            top5,
            [
                "deepseek-v4-flash-0731",
                "gpt-5.6-luna",
                "gpt-5.6-terra",
                "gpt-5.6-sol",
                "gemini-3.1-flash-lite",
            ],
        )

        # Every model entry should carry implementedAt for maintainability going forward.
        model_ids = re.findall(r'\{\s*id:\s*"([^"]+)"', js[js.index("const MODELS = [") : js.index("const WELCOME_QUICK_START_LIMIT")])
        self.assertGreater(len(model_ids), 20)
        for mid in model_ids:
            self.assertRegex(
                js,
                rf'id:\s*"{re.escape(mid)}",\s*implementedAt:\s*"\d{{4}}-\d{{2}}-\d{{2}}"',
                msg=f"missing implementedAt for {mid}",
            )


if __name__ == "__main__":
    unittest.main()
