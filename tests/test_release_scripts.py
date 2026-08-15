import datetime as dt
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_release_common():
    spec = importlib.util.spec_from_file_location(
        "release_common", SCRIPTS / "_release_common.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


COMMON = load_release_common()


def read(name: str) -> str:
    return (SCRIPTS / name).read_text(encoding="utf-8")


class ReleaseCommonTests(unittest.TestCase):
    def test_next_system_version_increments_patch(self):
        self.assertEqual(COMMON.next_system_version("V4.8.813"), "V4.8.814")

    def test_next_app_version_resets_on_new_day(self):
        self.assertEqual(
            COMMON.next_app_version("2026-08-15-004", dt.date(2026, 8, 16)),
            "2026-08-16-001",
        )

    def test_next_app_version_increments_same_day(self):
        self.assertEqual(
            COMMON.next_app_version("2026-08-16-001", dt.date(2026, 8, 16)),
            "2026-08-16-002",
        )

    def test_parse_versions_reads_app_py(self):
        versions = COMMON.parse_versions()
        self.assertRegex(versions["system_version"], r"^V4\.8\.\d+$")
        self.assertRegex(versions["app_version"], r"^\d{4}-\d{2}-\d{2}-\d{3}$")
        self.assertEqual(versions["system_lower"], versions["system_version"].lower())

    def test_git_allowlist_blocks_handoff_and_secrets(self):
        blocked = COMMON.classify_git_paths(
            [
                "app.py",
                "scripts/verify_changes.sh",
                "引き継ぎ資料.txt",
                "secret.key",
                ".env",
                "debug.log",
                "cookie.txt",
                "chat_core.bak.js",
                "instance/uploads/x",
            ]
        )
        self.assertIn("app.py", blocked["allowed"])
        self.assertIn("scripts/verify_changes.sh", blocked["allowed"])
        self.assertIn("引き継ぎ資料.txt", blocked["blocked"])
        self.assertIn("secret.key", blocked["blocked"])
        self.assertIn(".env", blocked["blocked"])
        self.assertIn("debug.log", blocked["blocked"])
        self.assertIn("cookie.txt", blocked["blocked"])
        self.assertIn("chat_core.bak.js", blocked["blocked"])
        self.assertIn("instance/uploads/x", blocked["blocked"])

    def test_git_allowlist_rejects_unknown_roots(self):
        classified = COMMON.classify_git_paths(["about_.env.txt", "console.log"])
        self.assertIn("about_.env.txt", classified["unknown"])
        self.assertIn("console.log", classified["blocked"])

    def test_notes_reject_user_request_phrasing(self):
        self.assertIsNotNone(COMMON.notes_are_forbidden("ユーザーの要望で追加しました"))
        self.assertIsNone(COMMON.notes_are_forbidden("版確認スクリプトを追加しました。"))

    def test_changelog_complete_requires_body_and_version(self):
        text = "# 更新履歴 - V4.8.814 (2026-08-16)\n\n確認スクリプトを追加しました。\n"
        self.assertIsNone(COMMON.changelog_is_complete(text, "V4.8.814"))
        self.assertIsNotNone(COMMON.changelog_is_complete("# 更新履歴\n", "V4.8.814"))
        self.assertIsNotNone(
            COMMON.changelog_is_complete(
                "# 更新履歴 - V4.8.814\n\nTODO ここに変更内容\n", "V4.8.814"
            )
        )


class ReleaseScriptContractTests(unittest.TestCase):
    def test_expected_scripts_exist(self):
        for name in (
            "_release_common.py",
            "_release_lib.sh",
            "verify_changes.sh",
            "prepare_version.sh",
            "publish_version.sh",
        ):
            path = SCRIPTS / name
            self.assertTrue(path.is_file(), name)
            if name.endswith(".sh"):
                self.assertTrue(path.stat().st_mode & 0o111, name)

    def test_verify_does_not_mutate_or_publish(self):
        source = read("verify_changes.sh")
        self.assertIn("pytest", source)
        self.assertIn("node --check", source)
        self.assertIn("check-assets", source)
        self.assertNotIn("prepare_version.sh", source)
        self.assertNotIn("publish_version.sh", source)
        self.assertNotIn("restart_services.sh", source)
        self.assertNotIn("purge_cloudflare_cache.sh", source)
        self.assertNotIn("git add", source)

    def test_prepare_requires_notes_and_does_not_publish(self):
        source = read("prepare_version.sh")
        self.assertIn("--notes", source)
        self.assertIn("--dry-run", source)
        self.assertIn("build_frontend.sh", source)
        self.assertIn("verify_changes.sh", source)
        self.assertNotIn("restart_services.sh", source)
        self.assertNotIn("purge_cloudflare_cache.sh", source)
        self.assertNotIn("git add", source)
        self.assertNotIn("git push", source)

    def test_publish_requires_preflight_and_stops_on_restart_failure(self):
        source = read("publish_version.sh")
        self.assertIn("--message", source)
        self.assertIn("--confirm", source)
        self.assertIn("review the plan", source)
        self.assertIn("restart_services.sh", source)
        self.assertIn("purge_cloudflare_cache.sh", source)
        self.assertIn("dump_restart_logs", source)
        self.assertIn("journalctl", source)
        self.assertIn("add --", source)
        self.assertNotIn("git add -A", source)
        self.assertNotIn("git add .", source)
        self.assertIn("tag -a", source)
        self.assertIn('refs/tags/$TAG^{}', source)
        restart_at = source.index("restart_services.sh")
        purge_at = source.index("purge_cloudflare_cache.sh")
        confirm_at = source.index("--confirm")
        self.assertLess(confirm_at, restart_at)
        self.assertLess(restart_at, purge_at)
        self.assertIn('CONFIRM" != "$SYSTEM_VERSION"', source)

    def test_scripts_readme_documents_the_three_entry_points(self):
        readme = (SCRIPTS / "README.md").read_text(encoding="utf-8")
        self.assertIn("verify_changes.sh", readme)
        self.assertIn("prepare_version.sh", readme)
        self.assertIn("publish_version.sh", readme)


if __name__ == "__main__":
    unittest.main()
