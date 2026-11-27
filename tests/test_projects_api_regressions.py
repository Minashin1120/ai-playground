import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("FLASK_SECRET_KEY", "projects-api-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-projects-api-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target


class ProjectsApiRegressionTests(unittest.TestCase):
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
            u1 = target.User(username="proj-owner", is_setup_completed=True)
            u1.set_password("pw1")
            u2 = target.User(username="proj-other", is_setup_completed=True)
            u2.set_password("pw2")
            target.db.session.add_all([u1, u2])
            target.db.session.commit()
            self.u1_id = u1.id
            self.u2_id = u2.id

    def _client_for(self, user_id):
        client = target.app.test_client()
        with client.session_transaction() as sess:
            sess["_user_id"] = str(user_id)
            sess["_fresh"] = True
            sess["csrf_token"] = "test-csrf"
        return client

    def _headers(self):
        return {"X-CSRF-Token": "test-csrf"}

    def _create_upload(self, user_id, filename, content=b"hello project"):
        user_dir = Path(self.tmpdir.name) / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        path = user_dir / filename
        path.write_bytes(content)
        return f"{user_id}/{filename}", path

    def test_project_crud_and_ownership_isolation(self):
        c1 = self._client_for(self.u1_id)
        c2 = self._client_for(self.u2_id)

        res = c1.post(
            "/api/projects",
            json={
                "name": "Alpha Project",
                "description": "Main spec",
                "instruction": "Be concise.",
                "use_vision_for_images": True,
                "vision_model": "gemini-3.6-flash",
            },
            headers=self._headers(),
            base_url="https://localhost",
        )
        self.assertEqual(res.status_code, 201, res.get_data(as_text=True))
        proj = res.get_json()
        uuid = proj["uuid"]
        self.assertEqual(proj["name"], "Alpha Project")
        self.assertTrue(proj["use_vision_for_images"])
        self.assertEqual(proj["vision_model"], "gemini-3.6-flash")

        # List returns the created project for owner, empty for other user.
        self.assertEqual(len(c1.get("/api/projects", base_url="https://localhost").get_json()), 1)
        self.assertEqual(len(c2.get("/api/projects", base_url="https://localhost").get_json()), 0)

        # Ownership isolation on detail/update/delete.
        self.assertIn(c2.get(f"/api/projects/{uuid}", base_url="https://localhost").status_code, (403, 404))
        self.assertIn(
            c2.put(f"/api/projects/{uuid}", json={"name": "Hijack"}, headers=self._headers(), base_url="https://localhost").status_code,
            (403, 404),
        )
        self.assertIn(
            c2.delete(f"/api/projects/{uuid}", headers=self._headers(), base_url="https://localhost").status_code,
            (403, 404),
        )

        # Update toggles Vision off and clears model.
        upd = c1.put(
            f"/api/projects/{uuid}",
            json={"use_vision_for_images": False, "vision_model": None, "instruction": "Updated prompt"},
            headers=self._headers(),
            base_url="https://localhost",
        )
        self.assertEqual(upd.status_code, 200)
        self.assertFalse(upd.get_json()["use_vision_for_images"])
        self.assertEqual(upd.get_json()["vision_model"], "")
        self.assertEqual(upd.get_json()["instruction"], "Updated prompt")

    def test_project_files_attach_protection_and_library_delete(self):
        c1 = self._client_for(self.u1_id)
        rel1, disk1 = self._create_upload(self.u1_id, "spec.txt", b"project specification")
        other_rel, _ = self._create_upload(self.u2_id, "secret.txt", b"other user secret")

        proj_uuid = c1.post(
            "/api/projects", json={"name": "Files Proj"}, headers=self._headers(), base_url="https://localhost"
        ).get_json()["uuid"]

        # Reject path belonging to another user / traversal.
        bad = c1.post(
            f"/api/projects/{proj_uuid}/files",
            json={"filepaths": [other_rel, f"{self.u1_id}/../{self.u2_id}/secret.txt"]},
            headers=self._headers(),
            base_url="https://localhost",
        )
        self.assertEqual(bad.status_code, 400)

        # Attach own file.
        ok = c1.post(
            f"/api/projects/{proj_uuid}/files",
            json={"filepaths": [rel1], "note": "Primary spec"},
            headers=self._headers(),
            base_url="https://localhost",
        )
        self.assertEqual(ok.status_code, 201, ok.get_data(as_text=True))
        files = ok.get_json()["added"]
        self.assertEqual(len(files), 1)
        file_id = files[0]["id"]
        self.assertEqual(files[0]["note"], "Primary spec")

        # Update note.
        note_res = c1.put(
            f"/api/projects/{proj_uuid}/files/{file_id}",
            json={"note": "Updated note"},
            headers=self._headers(),
            base_url="https://localhost",
        )
        self.assertEqual(note_res.status_code, 200)
        self.assertEqual(note_res.get_json()["note"], "Updated note")

        # Usage API reports project reference.
        usage = c1.get(f"/api/files/usage?filepath={rel1}", base_url="https://localhost")
        self.assertEqual(usage.status_code, 200)
        usage_projects = usage.get_json().get("projects", [])
        self.assertEqual(len(usage_projects), 1)
        self.assertEqual(usage_projects[0]["uuid"], proj_uuid)

        # Deleting a thread that also references rel1 must NOT delete disk1 because ProjectFile references it.
        t_res = c1.post("/api/threads", json={"project_id": proj_uuid}, headers=self._headers(), base_url="https://localhost")
        tid = t_res.get_json()["id"]
        with target.app.app_context():
            t = target.resolve_thread_for_user(tid, self.u1_id)
            msg = target.Message(thread_id=t.id, role="user", content="check", image_url=f'["{rel1}"]')
            target.db.session.add(msg)
            target.db.session.commit()
        del_t = c1.delete(f"/api/threads/{tid}", headers=self._headers(), base_url="https://localhost")
        self.assertEqual(del_t.status_code, 200)
        self.assertTrue(disk1.exists(), "Project shared file must not be deleted when a thread is deleted")

        # Explicit library delete removes the file AND cleans up the ProjectFile row.
        lib_del = c1.post(
            "/api/files/delete",
            json={"filenames": [rel1]},
            headers=self._headers(),
            base_url="https://localhost",
        )
        self.assertEqual(lib_del.status_code, 200)
        self.assertFalse(disk1.exists())
        detail = c1.get(f"/api/projects/{proj_uuid}", base_url="https://localhost").get_json()
        self.assertEqual(detail["file_count"], 0)

    def test_threads_scoping_move_branch_and_project_delete_unassign(self):
        c1 = self._client_for(self.u1_id)
        proj_uuid = c1.post(
            "/api/projects",
            json={"name": "Work", "instruction": "Project prompt text"},
            headers=self._headers(),
            base_url="https://localhost",
        ).get_json()["uuid"]

        # Create one unassigned thread and one project thread.
        t_free = c1.post("/api/threads", json={}, headers=self._headers(), base_url="https://localhost").get_json()["id"]
        t_proj = c1.post(
            "/api/threads", json={"project_id": proj_uuid}, headers=self._headers(), base_url="https://localhost"
        ).get_json()["id"]
        c1.put(f"/api/threads/{t_proj}/title", json={"title": "Project Chat Alpha"}, headers=self._headers(), base_url="https://localhost")
        with target.app.app_context():
            t_row = target.resolve_thread_for_user(t_proj, self.u1_id)
            m0 = target.Message(thread_id=t_row.id, role="user", content="hello")
            target.db.session.add(m0)
            target.db.session.commit()
            msg_id = m0.id

        # project_scope=unassigned excludes project chats when q is empty.
        unassigned = c1.get("/api/threads?project_scope=unassigned", base_url="https://localhost").get_json()["threads"]
        self.assertEqual([t["id"] for t in unassigned], [t_free])

        # Searching with q includes project chats and includes project_name badge.
        searched = c1.get("/api/threads?project_scope=unassigned&q=Alpha", base_url="https://localhost").get_json()["threads"]
        self.assertEqual([t["id"] for t in searched], [t_proj])
        self.assertEqual(searched[0]["project_name"], "Work")

        # Project threads endpoint returns only project chats.
        p_threads = c1.get(f"/api/projects/{proj_uuid}/threads", base_url="https://localhost").get_json()["threads"]
        self.assertEqual([t["id"] for t in p_threads], [t_proj])

        # Thread settings toggle include_project_instruction & return project preview.
        upd = c1.put(
            f"/api/threads/{t_proj}/settings",
            json={"include_project_instruction": False},
            headers=self._headers(),
            base_url="https://localhost",
        )
        self.assertEqual(upd.status_code, 200)
        got = c1.get(f"/api/threads/{t_proj}/settings", base_url="https://localhost").get_json()
        self.assertFalse(got["include_project_instruction"])
        self.assertEqual(got["project_id"], proj_uuid)
        self.assertEqual(got["project_instruction"], "Project prompt text")

        # Branching inherits project_id and include_project_instruction.
        with mock.patch.object(target.redis_conn, "get", return_value=None):
            br = c1.post(
                f"/api/threads/{t_proj}/branch",
                json={"message_id": msg_id, "request_id": "req-branch-proj-1"},
                headers=self._headers(),
                base_url="https://localhost",
            )
        self.assertIn(br.status_code, (200, 201), br.get_data(as_text=True))
        br_id = br.get_json()["id"]
        br_settings = c1.get(f"/api/threads/{br_id}/settings", base_url="https://localhost").get_json()
        self.assertEqual(br_settings["project_id"], proj_uuid)
        self.assertFalse(br_settings["include_project_instruction"])

        # Deleting the project unassigns both threads without deleting them.
        del_p = c1.delete(f"/api/projects/{proj_uuid}", headers=self._headers(), base_url="https://localhost")
        self.assertEqual(del_p.status_code, 200)
        unassigned_after = c1.get("/api/threads?project_scope=unassigned", base_url="https://localhost").get_json()["threads"]
        self.assertEqual(set(t["id"] for t in unassigned_after), {t_free, t_proj, br_id})

    def test_mobile_auth_allowlist_contains_project_endpoints(self):
        expected = {
            "handle_projects",
            "handle_project_item",
            "handle_project_threads",
            "handle_project_files",
            "handle_project_file_item",
            "read_project_file_api",
        }
        for ep in expected:
            self.assertIn(ep, target.MOBILE_ENDPOINT_METHODS)


if __name__ == "__main__":
    unittest.main()
