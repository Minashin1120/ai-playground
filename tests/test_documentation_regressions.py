import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DocumentationRegressionTests(unittest.TestCase):
    def test_public_project_documents_exist(self):
        required = (
            "README.md",
            "LICENSE",
            "THIRD_PARTY_NOTICES.md",
            "SECURITY.md",
            "CONTRIBUTING.md",
            ".env.example",
        )
        for relative in required:
            with self.subTest(relative=relative):
                self.assertTrue((ROOT / relative).is_file())

    def test_major_directories_have_readmes(self):
        required = (
            "deploy/README.md",
            "deploy/apache/README.md",
            "deploy/systemd/README.md",
            "scripts/README.md",
            "static/README.md",
            "static/css/README.md",
            "static/js/README.md",
            "static/changelogs/README.md",
            "static/legal/README.md",
            "static/pwa/README.md",
            "static/vendor/README.md",
            "templates/README.md",
            "tests/README.md",
        )
        for relative in required:
            with self.subTest(relative=relative):
                self.assertTrue((ROOT / relative).is_file())

    def test_documented_version_matches_application_and_assets(self):
        app_source = (ROOT / "app.py").read_text(encoding="utf-8")
        match = re.search(r"SYSTEM_VERSION'\]\s*=\s*'V(\d+\.\d+\.\d+)'", app_source)
        self.assertIsNotNone(match)
        version = match.group(1)
        version_lower = f"v{version}"

        self.assertIn(f"V{version}", (ROOT / "README.md").read_text(encoding="utf-8"))
        self.assertIn(f"V{version}", (ROOT / "MODELS.md").read_text(encoding="utf-8"))
        self.assertTrue((ROOT / f"static/js/chat_core.{version_lower}.js").is_file())
        self.assertTrue((ROOT / f"static/css/chat.custom.{version_lower}.css").is_file())
        self.assertTrue((ROOT / f"static/css/chat.tailwind.{version_lower}.css").is_file())

    def test_deployment_examples_cover_required_runtime_components(self):
        web_unit = (ROOT / "deploy/systemd/ai-chat.service").read_text(encoding="utf-8")
        worker_unit = (ROOT / "deploy/systemd/ai-chat-worker@.service").read_text(encoding="utf-8")
        apache = (ROOT / "deploy/apache/ai-playground.conf").read_text(encoding="utf-8")

        self.assertIn("127.0.0.1:3111", web_unit)
        self.assertIn("EnvironmentFile=/opt/ai-playground/.env", web_unit)
        self.assertIn('Environment="WORKER_INSTANCE=%i"', worker_unit)
        self.assertIn("TimeoutStopSec=660", worker_unit)
        self.assertIn("ProxyPreserveHost On", apache)
        self.assertIn('X-Forwarded-Proto "https"', apache)
        self.assertIn("RequestReadTimeout", apache)
        self.assertIn("ProxyTimeout 660", apache)

    def test_third_party_notices_cover_vendored_libraries(self):
        notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        for vendored in (ROOT / "static/vendor").glob("*.js"):
            with self.subTest(vendored=vendored.name):
                package_name = vendored.name.split("-")[0].lower()
                self.assertIn(package_name, notices.lower())

    def test_public_documents_exclude_internal_coordination_notes(self):
        documents = list(ROOT.glob("*.md"))
        for directory in ("deploy", "scripts", "static", "templates", "tests"):
            documents.extend((ROOT / directory).rglob("*.md"))

        forbidden = (
            "AGENTS.md",
            "引き継ぎ資料",
            "AIエージェント",
            "AI エージェント",
            "Gemini CLI",
            "ユーザークエリ",
            "アルファベット回答",
            "ユーザーの選択",
            "ユーザーの指示",
            "git push",
            "git commit",
            "関連コミット",
            "リリース作業",
            "/home/ai-chat",
        )
        for document in documents:
            source = document.read_text(encoding="utf-8")
            for marker in forbidden:
                with self.subTest(document=document.relative_to(ROOT), marker=marker):
                    self.assertNotIn(marker, source)

        for changelog in (ROOT / "static/changelogs").glob("*.md"):
            with self.subTest(changelog=changelog.name):
                self.assertNotIn("minashin1120.com", changelog.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
