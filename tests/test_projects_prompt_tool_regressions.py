import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests.app_source import read_app_source

os.environ.setdefault("FLASK_SECRET_KEY", "projects-prompt-tool-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-projects-prompt-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target


class ProjectsPromptToolRegressionTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.orig_upload_folder = target.app.config.get("UPLOAD_FOLDER")
        target.app.config.update(
            TESTING=True,
            MAINTENANCE_MODE=False,
            TRUSTED_HOSTS=["localhost"],
            UPLOAD_FOLDER=self.tmpdir.name,
        )
        self.addCleanup(lambda: target.app.config.update(UPLOAD_FOLDER=self.orig_upload_folder))
        target._ensure_temp_chat_monitor_running = lambda: None
        patcher = mock.patch.object(target, "_bot_turnstile_active", return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_project_files_auto_notice_registered_and_synced(self):
        self.assertIn("project_files", target.AUTO_SYSTEM_PROMPT_NOTICE_KEYS)
        self.assertIn("project_files", target.AUTO_SYSTEM_PROMPT_NOTICE_LABELS)
        default_text = target.AUTO_SYSTEM_PROMPT_NOTICE_DEFAULTS["project_files"]
        self.assertIn("{{project_files}}", default_text)
        self.assertIn("read_project_file", default_text)

        # Verify serverless-defaults.json also has project_files synced.
        asset_path = Path(target.app.root_path) / "android" / "app" / "src" / "main" / "assets" / "serverless-defaults.json"
        asset = json.loads(asset_path.read_text(encoding="utf-8"))
        self.assertIn("project_files", asset["auto_system_prompt_notices_config"])

    def test_catalog_and_render_notice_show_vision_status(self):
        files = [
            {"name": "notes.txt", "mime": "text/plain", "size": 120, "is_image": False, "note": "Architecture notes"},
            {"name": "diagram.png", "mime": "image/png", "size": 4096, "is_image": True, "note": ""},
        ]
        cat_off = target.build_project_files_catalog(files, use_vision_for_images=False)
        self.assertIn("notes.txt", cat_off)
        self.assertIn("Architecture notes", cat_off)
        self.assertIn("Vision: OFF (metadata only)", cat_off)

        cat_on = target.build_project_files_catalog(files, use_vision_for_images=True)
        self.assertIn("Vision: ON", cat_on)

        proj = SimpleNamespace(name="Demo", use_vision_for_images=False)
        rendered = target.render_project_files_notice(
            "Project {{project_name}} ({{project_file_count}}):\n{{project_files}}",
            proj,
            files,
        )
        self.assertIn("Project Demo (2):", rendered)
        self.assertIn("diagram.png", rendered)

    def test_execute_project_file_tool_text_chunking_and_unknown_name(self):
        user_dir = Path(self.tmpdir.name) / "1"
        user_dir.mkdir(parents=True, exist_ok=True)
        long_text = "A" * 1200 + "B" * 1200
        (user_dir / "long.txt").write_text(long_text, encoding="utf-8")

        project = SimpleNamespace(uuid="p1", name="Demo", use_vision_for_images=False, vision_model=None)
        files = [{
            "id": 1,
            "rel_path": "1/long.txt",
            "name": "long.txt",
            "display_name": "long.txt",
            "mime": "text/plain",
            "size": len(long_text),
            "is_image": False,
            "note": "",
        }]

        chunk1, meta1 = target.execute_project_file_tool(1, project, files, name="long.txt", offset=0, limit=1000)
        self.assertTrue(meta1["ok"])
        self.assertIn("chars 0-1000 of 2400", chunk1)
        self.assertIn("offset=1000", chunk1)
        self.assertIn("A" * 1000, chunk1)

        chunk2, meta2 = target.execute_project_file_tool(1, project, files, name="long.txt", offset=1000, limit=1500)
        self.assertTrue(meta2["ok"])
        self.assertIn("chars 1000-2400 of 2400", chunk2)
        self.assertIn("B" * 1200, chunk2)

        missing, meta_m = target.execute_project_file_tool(1, project, files, name="nonexistent.txt")
        self.assertFalse(meta_m["ok"])
        self.assertIn("Error: no project file named", missing)
        self.assertIn("long.txt", missing)

    def test_execute_project_file_tool_image_vision_off_never_calls_vision(self):
        user_dir = Path(self.tmpdir.name) / "1"
        user_dir.mkdir(parents=True, exist_ok=True)
        (user_dir / "ui.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")

        project = SimpleNamespace(uuid="p1", name="Demo", use_vision_for_images=False, vision_model="gemini-3.6-flash")
        files = [{
            "id": 2,
            "rel_path": "1/ui.png",
            "name": "ui.png",
            "display_name": "ui.png",
            "mime": "image/png",
            "size": 12,
            "is_image": True,
            "note": "",
        }]
        vision_fn = mock.Mock(return_value=("should not be called", None))
        out, meta = target.execute_project_file_tool(
            1, project, files, name="ui.png", vision_fn=vision_fn,
        )
        vision_fn.assert_not_called()
        self.assertTrue(meta["ok"])
        self.assertFalse(meta["vision_used"])
        self.assertIn("Vision model analysis is disabled", out)
        self.assertIn("ui.png", out)

    def test_execute_project_file_tool_image_vision_on_calls_selected_or_default_model(self):
        user_dir = Path(self.tmpdir.name) / "1"
        user_dir.mkdir(parents=True, exist_ok=True)
        (user_dir / "ui.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")

        files = [{
            "id": 2,
            "rel_path": "1/ui.png",
            "name": "ui.png",
            "display_name": "ui.png",
            "mime": "image/png",
            "size": 12,
            "is_image": True,
            "note": "",
        }]

        # Pick two distinct valid models from ALL_VALID_MODEL_IDS.
        valid_models = sorted(target.ALL_VALID_MODEL_IDS)
        m_override = "gemini-3.6-flash" if "gemini-3.6-flash" in target.ALL_VALID_MODEL_IDS else valid_models[0]
        m_user_default = next(m for m in valid_models if m != m_override)

        # Project overrides vision_model.
        user = SimpleNamespace(default_vision_model=m_user_default)
        project_override = SimpleNamespace(uuid="p1", name="Demo", use_vision_for_images=True, vision_model=m_override)
        self.assertEqual(target.resolve_project_vision_model(project_override, user), m_override)

        # Project leaves vision_model empty -> falls back to user's default_vision_model.
        project_default = SimpleNamespace(uuid="p2", name="Demo2", use_vision_for_images=True, vision_model=None)
        self.assertEqual(target.resolve_project_vision_model(project_default, user), m_user_default)

        called = {}
        def fake_vision_fn(img_bytes, img_mime, display_name):
            called["args"] = (len(img_bytes), img_mime, display_name)
            return "Blue button labeled Submit at center.", None

        out, meta = target.execute_project_file_tool(
            1, project_override, files, name="ui.png", vision_fn=fake_vision_fn,
        )
        self.assertTrue(meta["ok"])
        self.assertTrue(meta["vision_used"])
        self.assertIn("args", called)
        self.assertEqual(called["args"][2], "ui.png")
        self.assertIn("Blue button labeled Submit at center.", out)

    def test_provider_definitions_and_stream_wiring(self):
        self.assertEqual(target.build_project_file_tool_openai_chat()["function"]["name"], "read_project_file")
        self.assertEqual(target.build_project_file_tool_openai_responses()["name"], "read_project_file")
        self.assertEqual(target.build_project_file_tool_anthropic()["name"], "read_project_file")
        self.assertEqual(target.build_project_file_tool_definition()["name"], "read_project_file")

        self.assertTrue(target.project_file_tool_supported("gemini-3.6-flash"))
        self.assertFalse(target.project_file_tool_supported("gemini-3.6-flash", batch_mode=True))
        self.assertFalse(target.project_file_tool_supported("gpt-4o-search-preview"))

        app_source = read_app_source()
        self.assertIn("[Project Instructions]:", app_source)
        self.assertIn("include_project_instruction", app_source)
        self.assertIn("PROJECT_FILE_TOOL_NAME", app_source)
        self.assertIn("_execute_project_file_tool", app_source)

    def test_browser_fast_bootstrap_and_project_file_read_endpoint(self):
        with target.app.app_context():
            target.db.session.remove()
            target.db.drop_all()
            target.db.create_all()
            target.ensure_project_schema()
            u = target.User(username="fast-proj", is_setup_completed=True)
            u.set_password("pw")
            u.gemini_api_key = target.encrypt_val("dummy-gemini-key")
            target.db.session.add(u)
            target.db.session.commit()
            uid = u.id

        client = target.app.test_client()
        with client.session_transaction() as sess:
            sess["_user_id"] = str(uid)
            sess["_fresh"] = True
            sess["csrf_token"] = "csrf-fast"

        user_dir = Path(self.tmpdir.name) / str(uid)
        user_dir.mkdir(parents=True, exist_ok=True)
        (user_dir / "readme.md").write_text("# Fast Mode Project File\nHello!", encoding="utf-8")

        proj = client.post(
            "/api/projects",
            json={"name": "FastProj", "instruction": "Always reply in haiku."},
            headers={"X-CSRF-Token": "csrf-fast"},
            base_url="https://localhost",
        ).get_json()
        puuid = proj["uuid"]
        client.post(
            f"/api/projects/{puuid}/files",
            json={"filepaths": [f"{uid}/readme.md"]},
            headers={"X-CSRF-Token": "csrf-fast"},
            base_url="https://localhost",
        )
        tid = client.post(
            "/api/threads",
            json={"project_id": puuid},
            headers={"X-CSRF-Token": "csrf-fast"},
            base_url="https://localhost",
        ).get_json()["id"]

        boot = client.post(
            "/api/browser_fast_mode/bootstrap",
            json={"thread_id": tid, "model": "gemini-3.6-flash", "message": "Hi"},
            headers={"X-CSRF-Token": "csrf-fast"},
            base_url="https://localhost",
        )
        self.assertEqual(boot.status_code, 200, boot.get_data(as_text=True))
        data = boot.get_json()
        self.assertEqual(data["project_uuid"], puuid)
        self.assertIn("Always reply in haiku.", data["project_instruction"])
        self.assertIn("readme.md", data["project_files_notice"])
        self.assertEqual(data["project_file_tool_declaration"]["name"], "read_project_file")

        read_res = client.post(
            f"/api/projects/{puuid}/files/read",
            json={"name": "readme.md", "offset": 0, "limit": 5000},
            headers={"X-CSRF-Token": "csrf-fast"},
            base_url="https://localhost",
        )
        self.assertEqual(read_res.status_code, 200)
        self.assertIn("Fast Mode Project File", read_res.get_json()["content"])


if __name__ == "__main__":
    unittest.main()
