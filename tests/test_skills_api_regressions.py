import json
import os
import unittest
from unittest import mock

from sqlalchemy import inspect, text

os.environ.setdefault("FLASK_SECRET_KEY", "skills-api-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-skills-api-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target


class SkillsApiRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        target.app.config.update(TESTING=True, MAINTENANCE_MODE=False, TRUSTED_HOSTS=["localhost"])
        target._ensure_temp_chat_monitor_running = lambda: None

    def setUp(self):
        self._turnstile_gate_patcher = mock.patch.object(
            target, "_bot_turnstile_active", return_value=False
        )
        self._turnstile_gate_patcher.start()
        self.addCleanup(self._turnstile_gate_patcher.stop)
        with target.app.app_context():
            target.db.session.remove()
            target.db.drop_all()
            target.db.create_all()
            self.user = target.User(username="skills-owner", is_setup_completed=True)
            self.user.set_password("test-password")
            target.db.session.add(self.user)
            target.db.session.commit()
            self.user_id = self.user.id
            self.other = target.User(username="skills-other", is_setup_completed=True)
            self.other.set_password("test-password")
            target.db.session.add(self.other)
            target.db.session.commit()
            self.other_id = self.other.id

    def tearDown(self):
        with target.app.app_context():
            target.db.session.remove()

    def authenticated_client(self, user_id=None):
        client = target.app.test_client()
        with client.session_transaction() as sess:
            sess["_user_id"] = str(user_id or self.user_id)
            sess["_fresh"] = True
            sess["csrf_token"] = "csrf-test-token"
        return client

    def call(self, client, method, path, **kwargs):
        headers = kwargs.pop("headers", {})
        headers.setdefault("X-CSRF-Token", "csrf-test-token")
        return client.open(path, method=method, headers=headers, base_url="https://localhost", **kwargs)

    def test_skill_crud_and_owner_scope(self):
        client = self.authenticated_client()
        created = self.call(client, "POST", "/api/skills", json={
            "name": "Summariser",
            "description": "Summarise long text",
            "instruction": "Always answer in three bullets.",
        })
        self.assertEqual(created.status_code, 200, created.get_data(as_text=True))
        skill_uuid = created.get_json()["uuid"]

        listed = self.call(client, "GET", "/api/skills")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual([row["name"] for row in listed.get_json()], ["Summariser"])

        fetched = self.call(client, "GET", f"/api/skills/{skill_uuid}")
        self.assertEqual(fetched.status_code, 200)
        self.assertEqual(fetched.get_json()["instruction"], "Always answer in three bullets.")
        self.assertNotIn("fixed_prompts", fetched.get_json())
        self.assertNotIn("default_model", fetched.get_json())

        updated = self.call(client, "PUT", f"/api/skills/{skill_uuid}", json={
            "name": "Summariser 2",
            "description": "Short description",
            "instruction": "Use two bullets.",
        })
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.get_json()["name"], "Summariser 2")

        foreign = self.authenticated_client(self.other_id)
        self.assertEqual(self.call(foreign, "GET", f"/api/skills/{skill_uuid}").status_code, 403)
        self.assertEqual(self.call(foreign, "DELETE", f"/api/skills/{skill_uuid}").status_code, 403)

        deleted = self.call(client, "DELETE", f"/api/skills/{skill_uuid}")
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(self.call(client, "GET", f"/api/skills/{skill_uuid}").status_code, 404)

    def test_skill_validation_and_unique_name(self):
        client = self.authenticated_client()
        self.assertEqual(self.call(client, "POST", "/api/skills", json={
            "name": "", "instruction": "x",
        }).status_code, 400)
        self.assertEqual(self.call(client, "POST", "/api/skills", json={
            "name": "x" * 101, "instruction": "x",
        }).status_code, 400)
        self.assertEqual(self.call(client, "POST", "/api/skills", json={
            "name": "Dup", "instruction": "x",
        }).status_code, 200)
        duplicate = self.call(client, "POST", "/api/skills", json={
            "name": "dup", "instruction": "y",
        })
        self.assertEqual(duplicate.status_code, 400)
        self.assertEqual(duplicate.get_json()["error"], "name_taken")

    def test_legacy_gem_alias_keeps_legacy_fields(self):
        client = self.authenticated_client()
        created = self.call(client, "POST", "/api/gems", json={
            "name": "Legacy Gem",
            "instruction": "legacy instruction",
            "default_model": "gemini-3.6-flash",
            "fixed_prompts": [{"name": "Format", "content": "Use tables"}],
        })
        self.assertEqual(created.status_code, 200, created.get_data(as_text=True))
        gem_uuid = created.get_json()["uuid"]

        legacy = self.call(client, "GET", f"/api/gems/{gem_uuid}").get_json()
        self.assertEqual(legacy["fixed_prompts"], json.dumps(
            [{"name": "Format", "content": "Use tables"}],
            ensure_ascii=False, separators=(",", ":"),
        ))
        self.assertEqual(legacy["default_model"], "gemini-3.6-flash")

        # Editing through the new Skill API must not erase the legacy columns.
        self.assertEqual(self.call(client, "PUT", f"/api/skills/{gem_uuid}", json={
            "name": "Legacy Gem Renamed",
        }).status_code, 200)
        legacy_after = self.call(client, "GET", f"/api/gems/{gem_uuid}").get_json()
        self.assertEqual(legacy_after["name"], "Legacy Gem Renamed")
        self.assertEqual(legacy_after["default_model"], "gemini-3.6-flash")
        self.assertIsNotNone(legacy_after["fixed_prompts"])

        # The new endpoint reports the same record under the Skill name.
        skill = self.call(client, "GET", f"/api/skills/{gem_uuid}").get_json()
        self.assertEqual(skill["uuid"], gem_uuid)
        self.assertEqual(skill["name"], "Legacy Gem Renamed")

    def test_legacy_last_gem_uuid_is_accepted_and_new_field_is_returned(self):
        client = self.authenticated_client()
        with target.app.app_context():
            skill = target.Skill(uuid="55555555-5555-5555-5555-555555555555",
                                 user_id=self.user_id, name="Thread Skill", instruction="x")
            thread = target.Thread(user_id=self.user_id, public_id=target.generate_thread_public_id())
            target.db.session.add_all([skill, thread])
            target.db.session.commit()
            thread_public_id = thread.public_id

        saved = self.call(client, "POST", "/api/settings", json={
            "last_gem_uuid": "55555555-5555-5555-5555-555555555555",
            "thread_id": thread_public_id,
        })
        self.assertEqual(saved.status_code, 200, saved.get_data(as_text=True))
        with target.app.app_context():
            thread = target.Thread.query.filter_by(public_id=thread_public_id).one()
            self.assertEqual(thread.last_skill_uuid, "55555555-5555-5555-5555-555555555555")
            user = target.db.session.get(target.User, self.user_id)
            self.assertIsNone(user.last_skill_uuid)

        payload = self.call(client, "GET", f"/api/threads/{thread_public_id}").get_json()
        self.assertEqual(payload["last_skill_uuid"], "55555555-5555-5555-5555-555555555555")
        # Legacy alias remains for older clients.
        self.assertEqual(payload["last_gem_uuid"], "55555555-5555-5555-5555-555555555555")

    def test_legacy_gem_database_is_migrated_idempotently(self):
        with target.app.app_context():
            target.db.drop_all()
            with target.db.engine.begin() as conn:
                conn.execute(text("CREATE TABLE user (id INTEGER PRIMARY KEY, last_gem_uuid VARCHAR(36))"))
                conn.execute(text("CREATE TABLE thread (id INTEGER PRIMARY KEY, last_gem_uuid VARCHAR(36))"))
                conn.execute(text(
                    "CREATE TABLE message (id INTEGER PRIMARY KEY, gem_uuid VARCHAR(36), gem_name VARCHAR(100))"
                ))
                conn.execute(text(
                    "CREATE TABLE gem (id INTEGER PRIMARY KEY, uuid VARCHAR(36), user_id INTEGER, "
                    "name VARCHAR(100), description TEXT, instruction TEXT, fixed_prompts_json TEXT, "
                    "default_model VARCHAR(64), created_at DATETIME, import_signature VARCHAR(64))"
                ))
                conn.execute(text(
                    "INSERT INTO gem (uuid, user_id, name, instruction) VALUES ('legacy-uuid', 1, 'Legacy', 'x')"
                ))
                conn.execute(text(
                    "INSERT INTO thread (last_gem_uuid) VALUES ('legacy-uuid')"
                ))
                conn.execute(text(
                    "INSERT INTO message (gem_uuid, gem_name) VALUES ('legacy-uuid', 'Legacy')"
                ))

            target.ensure_skill_schema()
            target.ensure_skill_schema()  # idempotent

            inspector = inspect(target.db.engine)
            table_names = set(inspector.get_table_names())
            self.assertIn("skill", table_names)
            self.assertNotIn("gem", table_names)
            self.assertIn("last_skill_uuid", {c["name"] for c in inspector.get_columns("thread")})
            self.assertNotIn("last_gem_uuid", {c["name"] for c in inspector.get_columns("thread")})
            message_columns = {c["name"] for c in inspector.get_columns("message")}
            self.assertIn("skill_uuid", message_columns)
            self.assertIn("skill_name", message_columns)
            self.assertNotIn("gem_uuid", message_columns)
            self.assertNotIn("gem_name", message_columns)
            with target.db.engine.connect() as conn:
                self.assertEqual(conn.execute(text("SELECT COUNT(*) FROM skill")).scalar(), 1)
                self.assertEqual(
                    conn.execute(text("SELECT name FROM skill WHERE uuid='legacy-uuid'")).scalar(),
                    "Legacy",
                )


if __name__ == "__main__":
    unittest.main()
