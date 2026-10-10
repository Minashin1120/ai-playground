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
        self.fake.set(f"bot:score:{self.ids['locker']}", "3.5")
        with target.app.app_context():
            target.db.session.add(target.BanAppeal(user_id=self.ids["locker"], username="locker", message="appeal"))
            target.db.session.add(target.BannedIdentifier(
                kind="ip", value="198.51.100.9", reason="Linked ban",
                source_user_id=self.ids["locker"], source_username="locker",
            ))
            target.db.session.commit()
        res = self.post_json(locker, "/api/account/delete", {})
        self.assertEqual(res.status_code, 200)
        types = [t for t, _ in self.events("locker")]
        self.assertIn("lock", types)
        self.assertEqual(types[-1], "account_deleted")
        self.assertEqual(self.events("locker")[-1][1], {"by": "self"})
        # Locks, the score, ban appeals and IP/device bans all stay after the account is gone.
        keys = [(k if isinstance(k, str) else k.decode()) for k in self.fake._d]
        self.assertTrue(any(k.startswith("bot:lock:ip:") for k in keys))
        self.assertEqual(self.fake.get(f"bot:score:{self.ids['locker']}"), "3.5")
        with target.app.app_context():
            self.assertIsNone(target.db.session.get(target.User, self.ids["locker"]))
            self.assertEqual(target.BanAppeal.query.filter_by(user_id=self.ids["locker"]).count(), 1)
            self.assertEqual(target.BannedIdentifier.query.filter_by(source_user_id=self.ids["locker"]).count(), 1)

        admin = self.client_for("site-admin")
        accounts = {a["user_id"]: a for a in admin.get("/api/bot/evidence/accounts").get_json()["accounts"]}
        self.assertIn(self.ids["locker"], accounts)
        self.assertFalse(accounts[self.ids["locker"]]["exists"])
        self.assertEqual(accounts[self.ids["locker"]]["username"], "locker")
        self.assertEqual(accounts[self.ids["locker"]]["evidence_count"], 2)

        detail = admin.get(f"/api/bot/evidence?user_id={self.ids['locker']}").get_json()
        self.assertFalse(detail["user"]["exists"])
        self.assertEqual(detail["user"]["username"], "locker")
        self.assertEqual(detail["total"], 2)
        self.assertEqual(detail["items"][0]["event_type"], "account_deleted")
        state = detail["state"]
        self.assertFalse(state["exists"])
        self.assertEqual(state["appeal_count"], 1)
        self.assertEqual([i["identifier"] for i in state["banned_identifiers"]], ["198.51.100.9"])
        self.assertIn("ip", {lock["source"] for lock in state["locks"]})

        # The deleted account's locks and IP/device bans can still be lifted.
        res = self.post_json(admin, "/api/bot/account/clear-lock", {"user_id": self.ids["locker"]})
        self.assertEqual(res.status_code, 200)
        self.assertFalse(any((k if isinstance(k, str) else k.decode()).startswith("bot:lock:") for k in self.fake._d))
        res = self.post_json(admin, "/api/bot/account/clear-identifiers", {"user_id": self.ids["locker"]})
        self.assertEqual(res.get_json()["removed"], 1)
        with target.app.app_context():
            self.assertEqual(target.BannedIdentifier.query.filter_by(source_user_id=self.ids["locker"]).count(), 0)
        actions = [d["action"] for t, d in self.events("locker") if t == "admin_action"]
        self.assertEqual(actions, ["unlock", "unblock_identifiers"])
        self.assertTrue(all(r.username == "locker" for r in self._rows("locker")))

    def _rows(self, name):
        with target.app.app_context():
            return target.BotEvidenceLog.query.filter_by(user_id=self.ids[name]).all()

    def test_admin_bulk_deletes_records_of_several_accounts(self):
        for name in ("locker", "newcomer"):
            self.post_json(self.client_for(name), "/api/bot/lock", {"reason": "連打検出"})
        admin = self.client_for("site-admin")
        self.post_json(admin, "/api/bot/update", {"username": "locker", "action": "toggle_detection", "enabled": True})
        res = self.post_json(admin, "/api/bot/evidence/delete", {"user_ids": [self.ids["locker"], self.ids["newcomer"]]})
        self.assertEqual(res.get_json()["deleted"], 3)
        self.assertEqual(self.events("locker"), [])
        self.assertEqual(self.events("newcomer"), [])
        self.assertEqual(admin.get("/api/bot/evidence/accounts").get_json()["accounts"], [])
        normal = self.client_for("newcomer")
        res = self.post_json(normal, "/api/bot/evidence/delete", {"user_ids": [self.ids["locker"]]})
        self.assertEqual(res.status_code, 403)

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

    def test_account_management_cannot_delete_admin_accounts(self):
        with target.app.app_context():
            other = target.User(username="other-admin", is_setup_completed=True, is_admin=True)
            other.set_password("test-password")
            target.db.session.add(other)
            target.db.session.commit()
            other_id = other.id
        admin = self.client_for("site-admin")
        users = {u["username"]: u for u in admin.get("/api/bot/users").get_json()["users"]}
        self.assertTrue(users["other-admin"]["is_admin"])
        self.assertFalse(users["locker"]["is_admin"])
        for name in ("other-admin", "site-admin"):
            res = self.post_json(admin, "/api/bot/update", {"username": name, "action": "delete_account"})
            self.assertEqual(res.status_code, 403)
            self.assertEqual(res.get_json()["error"], "admin_account")
        res = self.post_json(admin, "/api/bot/update", {"username": "locker", "action": "delete_account"})
        self.assertEqual(res.status_code, 200)
        with target.app.app_context():
            self.assertIsNotNone(target.db.session.get(target.User, other_id))
            self.assertIsNotNone(target.db.session.get(target.User, self.ids["site-admin"]))
            self.assertIsNone(target.db.session.get(target.User, self.ids["locker"]))
        part08 = (PARTS / "chat_core.part08_domcontent_account_transfer.js").read_text(encoding="utf-8")
        self.assertIn("${u.is_admin ? '' : `<button class=\"bot-delete-account", part08)

    def test_admin_ui_wires_log_view(self):
        markup = read_chat_markup()
        # The log screen is separate from account management, with its own button and URL.
        self.assertIn('id="bot-log-open"', markup)
        self.assertIn('id="bot-log-modal"', markup)
        self.assertIn('id="bot-log-account-list"', markup)
        self.assertIn('id="bot-log-detail"', markup)
        part08 = (PARTS / "chat_core.part08_domcontent_account_transfer.js").read_text(encoding="utf-8")
        part16 = (PARTS / "chat_core.part16_gems_branch_debug.js").read_text(encoding="utf-8")
        self.assertIn("'/admin-bot-logs': { id: 'bot-log-modal'", part08)
        self.assertIn("case 'bot-log-modal'", part08)
        self.assertNotIn("bot-open-log", part08)
        self.assertIn("window.BotAdminLog = BotAdminLog", part16)
        self.assertIn("/api/bot/evidence/accounts", part16)
        self.assertIn("user_ids: ids", part16)
        self.assertIn("/api/bot/account/clear-lock", part16)
        self.assertIn("/api/bot/account/clear-identifiers", part16)
        page = self.client_for("site-admin").get("/admin-bot-logs")
        self.assertEqual(page.status_code, 200)


if __name__ == "__main__":
    unittest.main()
