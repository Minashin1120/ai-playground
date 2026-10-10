import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.chat_template import read_chat_markup

os.environ.setdefault("FLASK_SECRET_KEY", "bot-evidence-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-bot-evidence-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6398/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target
from tests.test_turnstile_bot_detection_regressions import _FakeRedis


APP_ROOT = Path(__file__).resolve().parents[1]
PARTS = APP_ROOT / "static" / "js" / "chat_core_parts"


class BotEvidenceAdminRegressionTests(unittest.TestCase):
    """Bot-detection records: what is logged, that they outlive accounts, and the admin API."""

    @classmethod
    def setUpClass(cls):
        target.app.config.update(TESTING=True, MAINTENANCE_MODE=False, TRUSTED_HOSTS=["localhost"])
        target._ensure_temp_chat_monitor_running = lambda: None

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        target.app.config["UPLOAD_FOLDER"] = self.temp_dir.name
        self.fake = _FakeRedis()
        patches = [
            mock.patch.object(target, "redis_conn", self.fake),
            mock.patch.object(target, "verify_turnstile", return_value=True),
            mock.patch.dict(os.environ, {"TURNSTILE_SITE_KEY": "k", "TURNSTILE_SECRET_KEY": "s"}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        with target.app.app_context():
            target.db.session.remove()
            target.db.drop_all()
            target.db.create_all()
            ids = {}
            for name, is_admin in (("locker", False), ("newcomer", False), ("site-admin", True)):
                user = target.User(username=name, is_setup_completed=True, is_admin=is_admin)
                user.set_password("test-password")
                target.db.session.add(user)
                target.db.session.commit()
                ids[name] = user.id
        self.ids = ids

    def tearDown(self):
        with target.app.app_context():
            target.db.session.remove()
        self.temp_dir.cleanup()

    def client_for(self, name):
        client = target.app.test_client()
        with client.session_transaction() as sess:
            sess["_user_id"] = str(self.ids[name])
            sess["_fresh"] = True
            sess["csrf_token"] = "csrf-test-token"
        return client

    def post_json(self, client, url, payload):
        return client.post(
            url,
            data=json.dumps(payload),
            content_type="application/json",
            headers={"X-CSRF-Token": "csrf-test-token"},
        )

    def events(self, name):
        with target.app.app_context():
            rows = target.BotEvidenceLog.query.filter_by(user_id=self.ids[name])\
                .order_by(target.BotEvidenceLog.id).all()
            return [(r.event_type, json.loads(r.details) if (r.details or "").startswith("{") else r.details)
                    for r in rows]

    def test_inherited_lock_block_is_logged_with_origin_once_per_minute(self):
        locker = self.client_for("locker")
        res = self.post_json(locker, "/api/bot/lock", {"reason": "連打検出"})
        self.assertEqual(res.get_json().get("status"), "locked")
        lock_events = [d for t, d in self.events("locker") if t == "lock"]
        self.assertEqual(len(lock_events), 1)
        self.assertIn("ip", lock_events[0]["applied_to"])
        self.assertEqual(lock_events[0]["lock_count"], 1)

        newcomer = self.client_for("newcomer")
        for _ in range(3):
            blocked = self.post_json(newcomer, "/chat_stream", {"message": "hi", "turnstile_token": "valid"})
            self.assertEqual(blocked.status_code, 403)
            self.assertEqual(blocked.get_json().get("error"), "account_locked")
        blocks = [d for t, d in self.events("newcomer") if t == "lock_blocked"]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["lock_source"], "ip")
        self.assertEqual(blocks[0]["origin_username"], "locker")
        self.assertEqual(blocks[0]["origin_client"], "web")
        self.assertEqual(blocks[0]["endpoint"], "chat_stream")

    def test_records_outlive_account_deletion_and_stay_viewable(self):
        locker = self.client_for("locker")
        self.post_json(locker, "/api/bot/lock", {"reason": "連打検出"})
        res = self.post_json(locker, "/api/account/delete", {})
        self.assertEqual(res.status_code, 200)
        types = [t for t, _ in self.events("locker")]
        self.assertIn("lock", types)
        self.assertEqual(types[-1], "account_deleted")
        self.assertEqual(self.events("locker")[-1][1], {"by": "self"})
        # The IP lock stays after the account is gone.
        self.assertTrue(any(
            (k if isinstance(k, str) else k.decode()).startswith("bot:lock:ip:") for k in self.fake._d
        ))

        admin = self.client_for("site-admin")
        listing = admin.get("/api/bot/users").get_json()
        deleted = {d["user_id"]: d for d in listing["deleted_users"]}
        self.assertIn(self.ids["locker"], deleted)
        self.assertEqual(deleted[self.ids["locker"]]["username"], "locker")
        self.assertEqual(deleted[self.ids["locker"]]["evidence_count"], 2)

        detail = admin.get(f"/api/bot/evidence?user_id={self.ids['locker']}").get_json()
        self.assertFalse(detail["user"]["exists"])
        self.assertEqual(detail["user"]["username"], "locker")
        self.assertEqual(detail["total"], 2)
        self.assertNotIn("state", detail)
        self.assertEqual(detail["items"][0]["event_type"], "account_deleted")

    def test_admin_deletes_selected_and_all_records(self):
        locker = self.client_for("locker")
        self.post_json(locker, "/api/bot/lock", {"reason": "連打検出"})
        admin = self.client_for("site-admin")
        self.post_json(admin, "/api/bot/update", {"username": "locker", "action": "toggle_detection", "enabled": True})
        items = admin.get(f"/api/bot/evidence?user_id={self.ids['locker']}").get_json()["items"]
        self.assertEqual(len(items), 2)

        res = self.post_json(admin, "/api/bot/evidence/delete", {"user_id": self.ids["locker"], "ids": [items[0]["id"]]})
        self.assertEqual(res.get_json()["deleted"], 1)
        self.assertEqual(len(self.events("locker")), 1)
        # Another account's ids are never deleted through this account.
        other = self.post_json(admin, "/api/bot/evidence/delete", {"user_id": self.ids["newcomer"], "ids": [items[1]["id"]]})
        self.assertEqual(other.get_json()["deleted"], 0)
        res = self.post_json(admin, "/api/bot/evidence/delete", {"user_id": self.ids["locker"], "all": True})
        self.assertEqual(res.get_json()["deleted"], 1)
        self.assertEqual(self.events("locker"), [])

    def test_non_admin_cannot_read_or_delete_records(self):
        normal = self.client_for("newcomer")
        self.assertEqual(normal.get(f"/api/bot/evidence?user_id={self.ids['locker']}").status_code, 403)
        res = self.post_json(normal, "/api/bot/evidence/delete", {"user_id": self.ids["locker"], "all": True})
        self.assertEqual(res.status_code, 403)

    def test_admin_state_shows_locks_and_unlock_clears_them(self):
        locker = self.client_for("locker")
        self.post_json(locker, "/api/bot/lock", {"reason": "連打検出"})
        admin = self.client_for("site-admin")
        state = admin.get(f"/api/bot/evidence?user_id={self.ids['locker']}").get_json()["state"]
        sources = {lock["source"] for lock in state["locks"]}
        self.assertIn("account", sources)
        self.assertIn("ip", sources)
        ip_lock = next(lock for lock in state["locks"] if lock["source"] == "ip")
        self.assertEqual(ip_lock["origin"]["username"], "locker")
        self.assertEqual(state["lock_count"], 1)

        res = self.post_json(admin, "/api/bot/update", {"username": "locker", "action": "unlock"})
        self.assertEqual(res.status_code, 200)
        self.assertFalse(any(
            (k if isinstance(k, str) else k.decode()).startswith("bot:lock:") for k in self.fake._d
        ))
        state = admin.get(f"/api/bot/evidence?user_id={self.ids['locker']}").get_json()["state"]
        self.assertEqual(state["locks"], [])
        last_type, last_details = self.events("locker")[-1]
        self.assertEqual(last_type, "admin_action")
        self.assertEqual(last_details, {"action": "unlock", "admin": "site-admin"})

    def test_related_ban_is_logged_on_linked_account(self):
        admin = self.client_for("site-admin")
        # Both accounts used the same IP address.
        with target.app.app_context():
            for name in ("locker", "newcomer"):
                target.db.session.add(target.UserSession(
                    user_id=self.ids[name], session_id=f"sid-{name}", ip_address="203.0.113.7"
                ))
            target.db.session.commit()
        self.post_json(admin, "/api/bot/update", {"username": "locker", "action": "ban", "reason": "Admin ban"})
        related = [d for t, d in self.events("newcomer") if t == "related_ban"]
        self.assertEqual(related, [{"source_user_id": self.ids["locker"], "source_username": "locker"}])
        # A manual ban is a BAN event on the account itself, attributed to the admin.
        self.assertEqual(
            [d for t, d in self.events("locker") if t == "ban"],
            [{"by": "admin", "admin": "site-admin"}],
        )
        detail = admin.get(f"/api/bot/evidence?user_id={self.ids['locker']}").get_json()
        self.assertEqual(detail["items"][0]["event_type"], "ban")
        self.assertEqual(detail["items"][0]["client"], "admin")

        # Lifting the linked ban records an unban on every account it released.
        self.post_json(admin, "/api/bot/update", {"username": "locker", "action": "unban_linked"})
        self.assertEqual(
            [d for t, d in self.events("locker") if t == "unban"],
            [{"by": "admin", "admin": "site-admin", "previous_reason": "Admin ban"}],
        )
        self.assertEqual(
            [d for t, d in self.events("newcomer") if t == "unban"],
            [{"by": "admin", "admin": "site-admin", "previous_reason": "Admin ban", "linked_from": "locker"}],
        )
        # Unbanning an account that is not banned records nothing.
        self.post_json(admin, "/api/bot/unban", {"username": "locker"})
        self.assertEqual(len([t for t, _ in self.events("locker") if t == "unban"]), 1)

    def test_admin_ui_wires_log_view(self):
        markup = read_chat_markup()
        self.assertIn('id="bot-admin-list-view"', markup)
        self.assertIn('id="bot-admin-detail"', markup)
        part08 = (PARTS / "chat_core.part08_domcontent_account_transfer.js").read_text(encoding="utf-8")
        part16 = (PARTS / "chat_core.part16_gems_branch_debug.js").read_text(encoding="utf-8")
        self.assertIn("bot-open-log", part08)
        self.assertIn("data.deleted_users", part08)
        self.assertIn("window.BotAdminLog = BotAdminLog", part16)
        self.assertIn("/api/bot/evidence/delete", part16)
        self.assertIn("action: 'unlock'", part16)


if __name__ == "__main__":
    unittest.main()
