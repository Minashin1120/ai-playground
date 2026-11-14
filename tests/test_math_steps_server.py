import json
import os
import unittest
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("FLASK_SECRET_KEY", "math-steps-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-math-steps-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target


class MathStepsServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        target.app.config.update(TESTING=True, MAINTENANCE_MODE=False, TRUSTED_HOSTS=["localhost"])
        target._ensure_temp_chat_monitor_running = lambda: None

    def setUp(self):
        self._turnstile_gate_patcher = mock.patch.object(target, "_bot_turnstile_active", return_value=False)
        self._turnstile_gate_patcher.start()
        self.addCleanup(self._turnstile_gate_patcher.stop)
        with target.app.app_context():
            target.db.session.remove()
            target.db.drop_all()
            target.db.create_all()
            self.user = target.User(username="math-owner", is_setup_completed=True)
            self.user.set_password("test-password")
            self.other = target.User(username="math-other", is_setup_completed=True)
            self.other.set_password("test-password")
            target.db.session.add_all([self.user, self.other])
            target.db.session.commit()
            self.user_id, self.other_id = self.user.id, self.other.id
            step_content = "## ステップ 1 整理する\n本文1\n\n## ステップ 2 計算する\n本文2"
            self.thread = target.Thread(
                user_id=self.user_id,
                public_id=target.generate_thread_public_id(),
                title="Math",
                include_global_instruction=False,
            )
            target.db.session.add(self.thread)
            target.db.session.flush()
            self.message = target.Message(
                thread_id=self.thread.id, role="assistant", model="gemini-3.6-flash",
                content=step_content, job_id="job_math_1",
            )
            target.db.session.add(self.message)
            other_thread = target.Thread(
                user_id=self.other_id,
                public_id=target.generate_thread_public_id(),
                title="Other",
                include_global_instruction=False,
            )
            target.db.session.add(other_thread)
            target.db.session.flush()
            self.other_message = target.Message(
                thread_id=other_thread.id, role="assistant", model="gemini-3.6-flash",
                content=step_content,
            )
            target.db.session.add(self.other_message)
            target.db.session.commit()
            self.thread_public_id = self.thread.public_id
            self.message_id = self.message.id
            self.other_message_id = self.other_message.id

    def tearDown(self):
        with target.app.app_context():
            target.db.session.remove()

    def client(self, user_id=None):
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

    def test_config_defaults_and_prompt_builder(self):
        self.assertEqual(target.normalize_math_steps_config(None), target.MATH_STEPS_DEFAULTS)
        normalized = target.normalize_math_steps_config({
            "basics": "true", "level": "bogus", "step_size": "large",
            "diagrams": "off", "animation": 0,
        })
        self.assertEqual(normalized, {
            "basics": True, "level": "easy", "step_size": "large",
            "diagrams": "off", "animation": False,
        })
        prompt = target.build_math_steps_system_prompt(normalized)
        self.assertIn("## ステップ 1", prompt)
        self.assertIn("mathsvg", prompt)
        self.assertNotIn("{{math_", prompt)
        self.assertIn("図は使わない", prompt)

    def test_background_and_browser_fast_inject_the_math_prompt(self):
        background = Path("server/background.py").read_text()
        browser_fast = Path("server/routes_browser_fast.py").read_text()
        self.assertIn('options.get("math_steps")', background)
        self.assertIn("build_math_steps_system_prompt", background)
        self.assertIn("math_steps_system_prompt", browser_fast)

    def test_extract_math_steps_ignores_fenced_headings(self):
        text = (
            "前置き\n## ステップ 1 整理する\n本文1\n```mathsvg\n"
            "## ステップ 9 図の中\n```\n### Step 2 計算する\n本文2\n"
        )
        steps = target.extract_math_steps(text)
        self.assertEqual([step["title"] for step in steps], ["整理する", "計算する"])
        self.assertEqual([step["index"] for step in steps], [0, 1])
        self.assertIn("本文1", steps[0]["body"])
        self.assertEqual(steps[0]["title"], "整理する")
        self.assertIn("本文2", steps[1]["body"])

    def test_encrypted_math_steps_state_round_trip(self):
        with target.app.app_context():
            message = target.db.session.get(target.Message, self.message_id)
            message.is_encrypted = True
            message.content = target.encrypt_val(message.content)
            message.math_steps_state = target.encrypt_val(json.dumps({
                "current": 2, "max_reached": 3, "simplified": {"0": "暗号化された説明"},
                "expanded": {"1": [{"title": "やさしい", "body": "暗号化された分割"}]},
            }, ensure_ascii=False))
            target.db.session.commit()

        payload = self.call(self.client(), "GET", f"/api/threads/{self.thread_public_id}")
        self.assertEqual(payload.status_code, 200)
        row = next(r for r in payload.get_json()["messages"] if r["id"] == self.message_id)
        self.assertEqual(row["math_steps_state"]["current"], 2)
        self.assertEqual(row["math_steps_state"]["simplified"]["0"], "暗号化された説明")
        self.assertEqual(
            row["math_steps_state"]["expanded"]["1"][0]["body"], "暗号化された分割"
        )

        saved = self.call(self.client(), "POST", "/api/math_steps/progress", json={
            "message_id": self.message_id, "current": 3, "max_reached": 4,
        })
        self.assertEqual(saved.status_code, 200, saved.get_data(as_text=True))
        self.assertEqual(saved.get_json()["math_steps_state"]["current"], 3)
        with target.app.app_context():
            message = target.db.session.get(target.Message, self.message_id)
            self.assertTrue(message.math_steps_state)
            self.assertNotIn("max_reached\", 4", message.math_steps_state)
            self.assertEqual(
                json.loads(target.decrypt_val(message.math_steps_state))["max_reached"], 4
            )

    def test_settings_get_and_post_math_steps(self):
        client = self.client()
        got = self.call(client, "GET", "/api/settings")
        self.assertEqual(got.status_code, 200)
        payload = got.get_json()
        self.assertFalse(payload["math_steps_enabled"])
        self.assertEqual(payload["math_steps_config"], target.MATH_STEPS_DEFAULTS)

        saved = self.call(client, "POST", "/api/settings", json={
            "math_steps_enabled": True,
            "math_steps_config": {"basics": True, "level": "bogus", "animation": False},
            "default_math_steps": True,
        })
        self.assertEqual(saved.status_code, 200, saved.get_data(as_text=True))
        payload = self.call(client, "GET", "/api/settings").get_json()
        self.assertTrue(payload["math_steps_enabled"])
        self.assertTrue(payload["default_math_steps"])
        self.assertEqual(payload["math_steps_config"], {
            "basics": True, "level": "easy", "step_size": "normal",
            "diagrams": "auto", "animation": False,
        })

    def test_progress_api_ownership_and_clamping(self):
        client = self.client()
        ok = self.call(client, "POST", "/api/math_steps/progress", json={
            "message_id": self.message_id, "current": 1, "max_reached": 2,
        })
        self.assertEqual(ok.status_code, 200, ok.get_data(as_text=True))
        state = ok.get_json()["math_steps_state"]
        self.assertEqual((state["current"], state["max_reached"]), (1, 2))

        back = self.call(client, "POST", "/api/math_steps/progress", json={
            "message_id": self.message_id, "current": 99, "max_reached": 1,
        })
        state = back.get_json()["math_steps_state"]
        self.assertEqual((state["current"], state["max_reached"]), (50, 50))

        by_job = self.call(client, "POST", "/api/math_steps/progress", json={
            "job_id": "job_math_1", "current": 0, "max_reached": 5,
        })
        self.assertEqual(by_job.status_code, 200)

        foreign = self.call(self.client(self.other_id), "POST", "/api/math_steps/progress", json={
            "message_id": self.message_id, "current": 1, "max_reached": 1,
        })
        self.assertEqual(foreign.status_code, 404)

    def test_simplify_validation_and_success(self):
        client = self.client()
        no_step = self.call(client, "POST", "/api/math_steps/simplify", json={
            "message_id": self.message_id, "step_index": 9,
        })
        self.assertEqual(no_step.status_code, 400)

        foreign = self.call(self.client(self.other_id), "POST", "/api/math_steps/simplify", json={
            "message_id": self.message_id, "step_index": 0,
        })
        self.assertEqual(foreign.status_code, 404)

        with target.app.app_context():
            empty = target.Message(
                thread_id=target.Thread.query.filter_by(user_id=self.user_id).first().id,
                role="assistant", model="gemini-3.6-flash", content="",
            )
            target.db.session.add(empty)
            target.db.session.commit()
            empty_id = empty.id
        empty_resp = self.call(client, "POST", "/api/math_steps/simplify", json={
            "message_id": empty_id, "step_index": 0,
        })
        self.assertEqual(empty_resp.status_code, 409)

        with mock.patch.object(
            target, "_call_math_steps_simplify_model",
            return_value=(
                "## ステップ 1 やさしく整理\nやさしい本文1\n\n"
                "## ステップ 2 やさしく計算\nやさしい本文2",
                None,
            ),
        ):
            done = self.call(client, "POST", "/api/math_steps/simplify", json={
                "message_id": self.message_id, "step_index": 1,
            })
        self.assertEqual(done.status_code, 200, done.get_data(as_text=True))
        body = done.get_json()
        self.assertEqual([step["title"] for step in body["steps"]], ["やさしく整理", "やさしく計算"])
        self.assertIn("やさしい本文1", body["text"])
        self.assertIn("やさしい本文2", body["text"])
        self.assertNotIn("1", body["math_steps_state"]["simplified"])
        self.assertEqual(
            body["math_steps_state"]["expanded"]["1"][1]["body"], "やさしい本文2"
        )

        # Splitting one of the new sub-steps again replaces only that part.
        with mock.patch.object(
            target, "_call_math_steps_simplify_model",
            return_value=(
                "## ステップ 1 さらに分ける\nさらに本文A\n\n"
                "## ステップ 2 続き\nさらに本文B",
                None,
            ),
        ):
            again = self.call(client, "POST", "/api/math_steps/simplify", json={
                "message_id": self.message_id, "step_index": 1, "part_index": 0,
            })
        self.assertEqual(again.status_code, 200, again.get_data(as_text=True))
        parts = again.get_json()["math_steps_state"]["expanded"]["1"]
        self.assertEqual(len(parts), 3)
        self.assertEqual(parts[0]["body"], "さらに本文A")
        self.assertEqual(parts[2]["body"], "やさしい本文2")

        invalid_part = self.call(client, "POST", "/api/math_steps/simplify", json={
            "message_id": self.message_id, "step_index": 1, "part_index": 9,
        })
        self.assertEqual(invalid_part.status_code, 400)

        # A model that ignores the heading rule still yields one easier step.
        with mock.patch.object(
            target, "_call_math_steps_simplify_model", return_value=("見出しなしの本文", None)
        ):
            fallback = self.call(client, "POST", "/api/math_steps/simplify", json={
                "message_id": self.message_id, "step_index": 0,
            })
        self.assertEqual(fallback.status_code, 200, fallback.get_data(as_text=True))
        self.assertEqual(fallback.get_json()["steps"][0]["body"], "見出しなしの本文")

        restored = self.call(client, "POST", "/api/math_steps/progress", json={
            "message_id": self.message_id,
            "current": 1,
            "max_reached": 1,
            "remove_simplified": [1],
            "remove_expanded": [1],
        })
        self.assertEqual(restored.status_code, 200, restored.get_data(as_text=True))
        self.assertNotIn("1", restored.get_json()["math_steps_state"]["simplified"])
        self.assertNotIn("1", restored.get_json()["math_steps_state"]["expanded"])

    def test_prompt_options_and_message_payload(self):
        record = target._build_prompt_options_record({"math_steps": True})
        self.assertIn('"math_steps":true', record)

        with target.app.app_context():
            message = target.db.session.get(target.Message, self.message_id)
            message.math_steps_state = json.dumps({"current": 1, "max_reached": 2, "simplified": {"0": "簡単"}})
            target.db.session.commit()
        payload = self.call(self.client(), "GET", f"/api/threads/{self.thread_public_id}")
        self.assertEqual(payload.status_code, 200)
        message_payload = next(row for row in payload.get_json()["messages"] if row["id"] == self.message_id)
        self.assertEqual(message_payload["math_steps_state"]["current"], 1)
        self.assertEqual(message_payload["math_steps_state"]["simplified"]["0"], "簡単")

    def test_mobile_preferences_math_steps(self):
        client = self.client()

        def fake_mobile_session():
            now = datetime.utcnow()
            target.g.mobile_session = SimpleNamespace(
                created_at=now, last_seen_at=now, ip_address=None, user_agent=None,
            )

        target.app.before_request_funcs.setdefault(None, []).insert(0, fake_mobile_session)
        self.addCleanup(
            target.app.before_request_funcs[None].remove, fake_mobile_session
        )
        with mock.patch.object(target, "MOBILE_ENDPOINT_METHODS", {}):
            got = client.get("/api/mobile/v1/preferences", base_url="https://localhost")
            self.assertEqual(got.status_code, 200, got.get_data(as_text=True))
            self.assertEqual(got.get_json()["math_steps_config"], target.MATH_STEPS_DEFAULTS)

            saved = client.put(
                "/api/mobile/v1/preferences", base_url="https://localhost",
                json={
                    "math_steps_enabled": True,
                    "default_math_steps": True,
                    "math_steps_config": {"level": "detailed", "diagrams": "off"},
                },
            )
        self.assertEqual(saved.status_code, 200, saved.get_data(as_text=True))
        payload = saved.get_json()
        self.assertTrue(payload["math_steps_enabled"])
        self.assertTrue(payload["default_math_steps"])
        self.assertEqual(payload["math_steps_config"]["level"], "detailed")
        self.assertEqual(payload["math_steps_config"]["diagrams"], "off")

    def test_browser_fast_bootstrap_returns_math_prompt(self):
        client = self.client()
        with mock.patch.object(
            target, "_get_browser_fast_mode_user_key", return_value=("fake-key", "gemini_common")
        ):
            resp = self.call(client, "POST", "/api/browser_fast_mode/bootstrap", json={
                "model": "gemini-3.6-flash", "math_steps": True,
            })
        self.assertEqual(resp.status_code, 200, resp.get_data(as_text=True))
        prompt = resp.get_json()["math_steps_system_prompt"]
        self.assertIn("## ステップ 1", prompt)
        self.assertIn("mathsvg", prompt)


if __name__ == "__main__":
    unittest.main()
