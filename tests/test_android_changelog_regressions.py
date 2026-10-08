import tempfile
import unittest
from pathlib import Path
from unittest import mock

import app as target


class AndroidChangelogRegressionTests(unittest.TestCase):
    def test_android_markdown_endpoint_default_folder_is_relative_to_app_root(self):
        with tempfile.TemporaryDirectory(prefix="android-app-root-") as directory:
            root = Path(directory)
            changelog_dir = root / "android" / "ci" / "changelogs"
            changelog_dir.mkdir(parents=True)
            (changelog_dir / "v1.10.0.md").write_text(
                "# Android版更新履歴 - 1.10.0\n\n- 変更点\n", encoding="utf-8"
            )
            missing = object()
            previous = target.app.config.pop("ANDROID_CHANGELOG_FOLDER", missing)
            try:
                with mock.patch.dict(target.app.config, TRUSTED_HOSTS=["localhost"]):
                    with mock.patch.object(target.app, "root_path", directory):
                        response = target.app.test_client().get(
                            "/android/release-notes.md", base_url="https://localhost"
                        )
            finally:
                if previous is not missing:
                    target.app.config["ANDROID_CHANGELOG_FOLDER"] = previous

        self.assertEqual(response.status_code, 200)
        self.assertIn("1.10.0", response.get_data(as_text=True))

    def test_android_markdown_endpoint_aggregates_versions_newest_first(self):
        with tempfile.TemporaryDirectory(prefix="android-changelog-") as directory:
            root = Path(directory)
            (root / "v1.9.0.md").write_text("# Android版更新履歴 - 1.9.0\n\n- 古い変更\n", encoding="utf-8")
            (root / "v1.10.0.md").write_text("- 新しい変更\n", encoding="utf-8")
            (root / "ignored.txt").write_text("not markdown\n", encoding="utf-8")
            with mock.patch.dict(
                target.app.config,
                ANDROID_CHANGELOG_FOLDER=str(root),
                TRUSTED_HOSTS=["localhost"],
            ):
                response = target.app.test_client().get(
                    "/android/release-notes.md", base_url="https://localhost"
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Content-Type"], "text/markdown; charset=utf-8")
        body = response.get_data(as_text=True)
        self.assertLess(body.index("1.10.0"), body.index("1.9.0"))
        self.assertNotIn("not markdown", body)

    def test_android_markdown_endpoint_adds_version_heading_from_file_name(self):
        # The app splits versions on "# Android版更新履歴 - X.Y.Z" and drops text before the first one.
        with tempfile.TemporaryDirectory(prefix="android-changelog-") as directory:
            root = Path(directory)
            (root / "v1.13.56.md").write_text("# Android版更新履歴 - 1.13.56\n\n- 見出しつき\n", encoding="utf-8")
            (root / "v1.13.57.md").write_text("# AI Playground Android v1.13.57\n\n- 古い題名つき\n", encoding="utf-8")
            (root / "v1.47.0.md").write_text("- 見出しなし\n", encoding="utf-8")
            (root / "v1.48.0.md").write_text("\n", encoding="utf-8")
            with mock.patch.dict(
                target.app.config,
                ANDROID_CHANGELOG_FOLDER=str(root),
                TRUSTED_HOSTS=["localhost"],
            ):
                response = target.app.test_client().get(
                    "/android/release-notes.md", base_url="https://localhost"
                )

        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertEqual(
            body,
            "# AI Playground for Android\n\n"
            "# Android版更新履歴 - 1.47.0\n\n- 見出しなし\n\n"
            "# Android版更新履歴 - 1.13.57\n\n- 古い題名つき\n\n"
            "# Android版更新履歴 - 1.13.56\n\n- 見出しつき\n",
        )

    def test_web_changelog_does_not_render_android_changelog_source(self):
        with mock.patch.dict(target.app.config, TRUSTED_HOSTS=["localhost"]):
            response = target.app.test_client().get(
                "/changelog", base_url="https://localhost"
            )
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("AI Playground for Android", response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
