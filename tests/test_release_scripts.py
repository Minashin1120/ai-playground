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
                "server/models.py",
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
        self.assertIn("server/models.py", blocked["allowed"])
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

    def test_update_handoff_keeps_toc_at_top(self):
        import tempfile

        block = (
            "**最終更新:** 2026-09-01\n"
            "**システム状態:** **正常稼働中 (V4.8.900)**\n"
            "**バージョン:** V4.8.900\n"
            "**特記事項:** テスト\n\n"
        )
        toc = (
            "====\n"
            "【引き継ぎ資料 目次】\n"
            "1. 概要\n"
            "※ バージョンの更新履歴\n"
            "\n"
            "【現行のサービス構成】\n"
            "本文\n"
        )
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "handoff.txt"
            p.write_text(toc, encoding="utf-8")
            COMMON.update_handoff(p, block)
            result = p.read_text(encoding="utf-8")
        self.assertTrue(result.startswith("====\n【引き継ぎ資料 目次】"))
        note_pos = result.index("※ バージョンの更新履歴")
        block_pos = result.index("**最終更新:**")
        body_pos = result.index("【現行のサービス構成】")
        self.assertLess(note_pos, block_pos)
        self.assertLess(block_pos, body_pos)

    def test_update_handoff_prepends_when_no_toc_marker(self):
        import tempfile

        block = "**最終更新:** 2026-09-01\n**バージョン:** V4.8.900\n\n"
        previous = "【現行のサービス構成】\n本文\n"
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "handoff.txt"
            p.write_text(previous, encoding="utf-8")
            COMMON.update_handoff(p, block)
            result = p.read_text(encoding="utf-8")
        self.assertTrue(result.startswith("**最終更新:**"))
        self.assertIn("【現行のサービス構成】", result)


class ReleaseScriptContractTests(unittest.TestCase):
    def test_expected_scripts_exist(self):
        for name in (
            "_release_common.py",
            "_release_lib.sh",
            "verify_changes.sh",
            "prepare_version.sh",
            "publish_version.sh",
            "record_changes.sh",
            "wait_for_restart_headroom.sh",
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

    def test_static_assets_must_be_readable_by_the_web_server(self):
        # Apache serves /static directly: a private (0600) versioned asset makes the site answer 403
        # while gunicorn, which owns the file, still passes the live checks.
        self.assertIn("chmod a+r", read("prepare_version.sh"))
        self.assertIn("-perm -004", read("verify_changes.sh"))

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

    def test_record_changes_is_scoped_and_does_not_deploy_web(self):
        source = read("record_changes.sh")
        self.assertIn("--target", source)
        self.assertIn("classify-record", source)
        self.assertIn("Android Actions", source)
        self.assertIn("git_in add --", source)
        self.assertIn("git_in push origin HEAD", source)
        self.assertNotIn("restart_services.sh", source)
        self.assertNotIn("purge_cloudflare_cache.sh", source)
        self.assertNotIn("tag -a", source)
        self.assertNotIn("git add -A", source)

    def test_record_target_classification_separates_build_and_docs(self):
        android = COMMON.classify_record_target(
            ["android/app/src/main/AndroidManifest.xml", "android/version.properties"],
            "android",
        )
        self.assertEqual(android["outside_target"], [])
        self.assertIn("android/app/src/main/AndroidManifest.xml", android["android_build"])

        docs = COMMON.classify_record_target(
            ["android/README.md", "android/ci/changelogs/v1.13.32.md"], "android"
        )
        self.assertEqual(docs["android_build"], [])

        operations = COMMON.classify_record_target(
            [
                "scripts/record_changes.sh",
                ".github/workflows/android.yml",
                "tests/conftest.py",
                "tests/test_production_schema_guard_regressions.py",
            ],
            "operations",
        )
        self.assertEqual(operations["outside_target"], [])
        self.assertEqual(operations["android_build"], [])

        mixed = COMMON.classify_record_target(
            ["scripts/record_changes.sh", "server/models.py"], "operations"
        )
        self.assertIn("server/models.py", mixed["outside_target"])

    def test_deploy_server_restarts_without_version_bump(self):
        source = read("deploy_server.sh")
        self.assertIn("classify-record --target server", source)
        self.assertIn('"$CONFIRM" == "SERVER"', source)
        self.assertIn("git_in add --", source)
        self.assertIn("git_in push origin HEAD", source)
        self.assertNotIn("scripts/prepare_version.sh\"", source)
        self.assertNotIn("purge_cloudflare_cache.sh", source)
        self.assertNotIn("tag -a", source)
        self.assertNotIn("git add -A", source)
        confirm_at = source.index('"$CONFIRM" == "SERVER"')
        verify_at = source.index('"$ROOT/scripts/verify_changes.sh"')
        restart_at = source.index('"$ROOT/scripts/restart_services.sh"')
        commit_at = source.index("git_in commit")
        self.assertLess(confirm_at, verify_at)
        self.assertLess(verify_at, restart_at)
        self.assertLess(restart_at, commit_at)

    def test_server_target_excludes_web_ui_versions_and_android(self):
        server = COMMON.classify_record_target(
            [
                "server/routes_pages.py",
                "server/README.md",
                "tests/test_android_changelog_regressions.py",
                "worker.py",
                "deploy/ANDROID_CLIENT.md",
            ],
            "server",
        )
        self.assertEqual(server["outside_target"], [])
        for path in (
            "app.py",
            "templates/chat.html",
            "static/js/chat_core.v4.8.1069.js",
            "android/app/build.gradle.kts",
            "scripts/deploy_server.sh",
            "deploy/systemd/ai-chat.service",
            "requirements.txt",
        ):
            mixed = COMMON.classify_record_target(["server/routes_pages.py", path], "server")
            self.assertIn(path, mixed["outside_target"], path)

    def test_android_version_bump_requires_new_changelog(self):
        import tempfile

        previous = "VERSION_CODE=1\nVERSION_NAME=1.0.0\n"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "android" / "ci" / "changelogs").mkdir(parents=True)
            (root / "android" / "version.properties").write_text(
                "VERSION_CODE=2\nVERSION_NAME=1.0.1\n", encoding="utf-8"
            )
            bump = ["android/version.properties", "android/app/build.gradle.kts"]
            missing = COMMON.android_release_notes_errors(bump, previous, root)
            self.assertTrue(any("v1.0.1.md is missing" in e for e in missing))

            notes = root / "android" / "ci" / "changelogs" / "v1.0.1.md"
            notes.write_text("- 画面表示を修正しました。\n", encoding="utf-8")
            reused = COMMON.android_release_notes_errors(bump, previous, root)
            self.assertTrue(any("already existed" in e for e in reused))

            with_notes = bump + ["android/ci/changelogs/v1.0.1.md"]
            self.assertEqual(
                COMMON.android_release_notes_errors(with_notes, previous, root), []
            )

            notes.write_text("# v1.0.1\n", encoding="utf-8")
            empty = COMMON.android_release_notes_errors(with_notes, previous, root)
            self.assertTrue(any("bullet" in e for e in empty))

            phrase = COMMON.FORBIDDEN_CHANGELOG_PHRASES[0]
            notes.write_text(f"- {phrase}修正しました。\n", encoding="utf-8")
            forbidden = COMMON.android_release_notes_errors(with_notes, previous, root)
            self.assertTrue(any("forbidden phrase" in e for e in forbidden))

            same = "VERSION_CODE=1\nVERSION_NAME=1.0.1\n"
            self.assertEqual(COMMON.android_release_notes_errors(bump, same, root), [])
            self.assertEqual(
                COMMON.android_release_notes_errors(["server/models.py"], previous, root),
                [],
            )

        self.assertIn("check-android-notes", read("record_changes.sh"))
        self.assertIn("check-android-notes", read("publish_version.sh"))

    def test_android_workflow_paths_match_release_classification(self):
        workflow = (ROOT / ".github" / "workflows" / "android.yml").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("'android/**'", workflow)
        for path in COMMON.ANDROID_BUILD_EXACT:
            self.assertIn(f"'{path}'", workflow, path)
        for prefix in COMMON.ANDROID_BUILD_PREFIXES:
            self.assertIn(f"'{prefix}**'", workflow, prefix)

    def test_android_release_uses_versioned_changelog_notes(self):
        workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn('NOTES_FILE="android/ci/changelogs/v$VERSION.md"', workflow)
        self.assertIn('[[ -f "$NOTES_FILE" ]]', workflow)
        self.assertIn('--notes-file "$NOTES_FILE"', workflow)
        self.assertNotIn('--notes-file android/ci/release-notes.md', workflow)

    def test_restart_waits_for_resource_headroom_before_systemd(self):
        source = read("restart_services.sh")
        headroom_at = source.index("wait_for_restart_headroom.sh")
        restart_at = source.index("sudo systemctl restart --no-block")
        self.assertLess(headroom_at, restart_at)
        gate = read("wait_for_restart_headroom.sh")
        self.assertIn("/proc/pressure/memory", gate)
        self.assertIn("/proc/pressure/io", gate)
        self.assertIn("MemAvailable", gate)
        self.assertIn("pswpin", gate)

    def test_scripts_readme_documents_the_three_entry_points(self):
        readme = (SCRIPTS / "README.md").read_text(encoding="utf-8")
        self.assertIn("verify_changes.sh", readme)
        self.assertIn("prepare_version.sh", readme)
        self.assertIn("publish_version.sh", readme)
        self.assertIn("record_changes.sh", readme)
        self.assertIn("deploy_server.sh", readme)


if __name__ == "__main__":
    unittest.main()
