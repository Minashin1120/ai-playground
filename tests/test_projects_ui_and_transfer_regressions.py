import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("FLASK_SECRET_KEY", "projects-ui-transfer-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-projects-ui-transfer-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target


class ProjectsUiAndTransferRegressionTests(unittest.TestCase):
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

        with target.app.app_context():
            target.db.session.remove()
            target.db.drop_all()
            target.db.create_all()
            target.ensure_project_schema()
            u = target.User(username="transfer-proj-user", is_setup_completed=True)
            u.set_password("pw")
            target.db.session.add(u)
            target.db.session.commit()
            self.uid = u.id

    def test_web_templates_and_js_contracts(self):
        root = Path(target.app.root_path)
        sidebar = (root / "templates" / "chat" / "sidebar.html").read_text(encoding="utf-8")
        self.assertIn("project-list", sidebar)
        self.assertLess(sidebar.index("project-list"), sidebar.index("skill-list"))

        stage = (root / "templates" / "chat" / "main_stage.html").read_text(encoding="utf-8")
        self.assertIn("project-view", stage)
        self.assertIn("project-chat-banner", stage)
        self.assertIn("project-view-vision-toggle", stage)
        self.assertIn("project-view-vision-model", stage)

        overlay = (root / "templates" / "chat" / "overlay_project.html").read_text(encoding="utf-8")
        self.assertIn("project-modal", overlay)
        self.assertIn("project-use-vision-toggle", overlay)
        self.assertIn("project-vision-model-select", overlay)
        self.assertIn("move-thread-project-modal", overlay)

        thread_modal = (root / "templates" / "chat" / "overlay_thread.html").read_text(encoding="utf-8")
        self.assertIn("thread-include-project-instruction", thread_modal)
        self.assertIn("thread-project-select", thread_modal)

        js_part = (root / "static" / "js" / "chat_core_parts" / "chat_core.part33_projects.js").read_text(encoding="utf-8")
        self.assertIn("loadProjects", js_part)
        self.assertIn("openProjectView", js_part)
        self.assertIn("use_vision_for_images", js_part)

    def test_account_export_dedupe_and_mobile_sync_include_projects(self):
        self.assertIn("projects", target.ACCOUNT_IMPORT_CATEGORIES)

        with target.app.app_context():
            user = target.db.session.get(target.User, self.uid)
            rel_path = f"{self.uid}/shared.txt"
            user_dir = Path(self.tmpdir.name) / str(self.uid)
            user_dir.mkdir(parents=True, exist_ok=True)
            (user_dir / "shared.txt").write_text("data", encoding="utf-8")

            p1 = target.Project(
                user_id=self.uid,
                name="Exported Proj",
                description="desc",
                instruction="inst",
                use_vision_for_images=True,
                vision_model="gemini-3.6-flash",
            )
            target.db.session.add(p1)
            target.db.session.flush()
            pf = target.ProjectFile(project_id=p1.id, user_id=self.uid, rel_path=rel_path, display_name="shared.txt", note="memo")
            target.db.session.add(pf)
            t = target.Thread(
                user_id=self.uid,
                title="Proj Thread",
                project_id=p1.id,
                include_project_instruction=False,
            )
            target.db.session.add(t)
            target.db.session.commit()

            exported = target._account_export_projects(self.uid)
            self.assertEqual(len(exported), 1)
            self.assertEqual(exported[0]["name"], "Exported Proj")
            self.assertTrue(exported[0]["use_vision_for_images"])
            self.assertEqual(exported[0]["vision_model"], "gemini-3.6-flash")
            self.assertEqual(exported[0]["files"][0]["rel_path"], rel_path)

            # Dedupe file reference collector protects ProjectFile.rel_path.
            refs = target._collect_referenced_file_paths(self.uid)
            self.assertIn(rel_path, refs)

            # Mobile sync thread row exposes project fields.
            row = target._sync_thread_row(t, None, project_map={p1.id: p1})
            self.assertEqual(row["project_id"], p1.uuid)
            self.assertEqual(row["project_name"], "Exported Proj")
            self.assertFalse(row["include_project_instruction"])

            # Feedback chat copy row serialization for project.
            proj_fb_row = target._feedback_chat_row(p1, "project")
            self.assertEqual(proj_fb_row["kind"], "project")
            self.assertEqual(proj_fb_row["name"], "Exported Proj")
            self.assertTrue(proj_fb_row["use_vision_for_images"])


if __name__ == "__main__":
    unittest.main()
