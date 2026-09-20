import tempfile
import unittest
from pathlib import Path
from unittest import mock

import app as target


class AndroidChangelogRegressionTests(unittest.TestCase):
    def test_android_markdown_endpoint_aggregates_versions_newest_first(self):
        with tempfile.TemporaryDirectory(prefix="android-changelog-") as directory:
            root = Path(directory)
            (root / "v1.9.0.md").write_text("# Android版更新履歴 - 1.9.0\n", encoding="utf-8")
            (root / "v1.10.0.md").write_text("# Android版更新履歴 - 1.10.0\n", encoding="utf-8")
            (root / "ignored.txt").write_text("not markdown\n", encoding="utf-8")
            with mock.patch.dict(target.app.config, ANDROID_CHANGELOG_FOLDER=str(root)):
                response = target.app.test_client().get(
                    "/android/release-notes.md", base_url="https://localhost"
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Content-Type"], "text/markdown; charset=utf-8")
        body = response.get_data(as_text=True)
        self.assertLess(body.index("1.10.0"), body.index("1.9.0"))
        self.assertNotIn("not markdown", body)

    def test_web_changelog_does_not_render_android_changelog_source(self):
        response = target.app.test_client().get("/changelog", base_url="https://localhost")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("AI Playground for Android", response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
