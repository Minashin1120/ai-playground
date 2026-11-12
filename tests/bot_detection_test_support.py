"""Shared fixtures for Turnstile / bot-detection regression tests."""
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from tests.chat_template import read_chat_markup

from tests.app_source import read_app_source
os.environ.setdefault("FLASK_SECRET_KEY", "turnstile-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-turnstile-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6398/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target


APP_ROOT = Path(__file__).resolve().parents[1]


class _FakeRedis:
    """Minimal dict-backed Redis stand-in for tests (keys with TTL)."""

    def __init__(self):
        self._d = {}

    def exists(self, key):
        entry = self._d.get(key)
        if entry is None:
            return 0
        value, expires_at = entry
        if expires_at is not None and time.time() >= expires_at:
            self._d.pop(key, None)
            return 0
        return 1

    def set(self, key, value, ex=None, nx=False, xx=False):
        exists = self.exists(key)
        if nx and exists:
            return False
        if xx and not exists:
            return False
        self._d[key] = (value, time.time() + ex if ex else None)
        return True

    def get(self, key):
        entry = self._d.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and time.time() >= expires_at:
            self._d.pop(key, None)
            return None
        return value

    def delete(self, *keys):
        removed = 0
        for key in keys:
            if key in self._d:
                del self._d[key]
                removed += 1
        return removed

    def expire(self, key, ttl):
        entry = self._d.get(key)
        if entry is None:
            return True
        self._d[key] = (entry[0], time.time() + ttl)
        return True

    def ttl(self, key):
        entry = self._d.get(key)
        if entry is None:
            return -2
        value, expires_at = entry
        if expires_at is None:
            return -1
        return int(expires_at - time.time())

    def incr(self, key):
        current = self.get(key)
        new_value = (int(current) + 1) if current is not None else 1
        self._d[key] = (new_value, None)
        return new_value

    def incrbyfloat(self, key, amount):
        current = self.get(key)
        new_value = (float(current) + float(amount)) if current is not None else float(amount)
        self._d[key] = (new_value, None)
        return new_value

    def __getattr__(self, name):
        def _missing(*args, **kwargs):
            raise NotImplementedError(name)
        return _missing


class BotDetectionTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        target.app.config.update(TESTING=True, MAINTENANCE_MODE=False, TRUSTED_HOSTS=["localhost"])
        target._ensure_temp_chat_monitor_running = lambda: None

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        target.app.config["UPLOAD_FOLDER"] = self.temp_dir.name
        with target.app.app_context():
            target.db.session.remove()
            target.db.drop_all()
            target.db.create_all()
            user = target.User(username="turnstile-test", is_setup_completed=True)
            user.set_password("test-password")
            target.db.session.add(user)
            target.db.session.commit()
            self.user_id = user.id

    def tearDown(self):
        with target.app.app_context():
            target.db.session.remove()
        self.temp_dir.cleanup()

    def authenticated_client(self):
        client = target.app.test_client()
        with client.session_transaction() as sess:
            sess["_user_id"] = str(self.user_id)
            sess["_fresh"] = True
            sess["csrf_token"] = "csrf-test-token"
        return client

    def post_telemetry(self, client, payload):
        return client.post(
            "/api/bot-telemetry",
            data=target.json.dumps(payload),
            content_type="application/json",
            headers={"X-CSRF-Token": "csrf-test-token"},
        )

    def post_chat_stream(self, client, payload):
        return client.post(
            "/chat_stream",
            data=target.json.dumps(payload),
            content_type="application/json",
            headers={"X-CSRF-Token": "csrf-test-token"},
        )

    def post_turnstile_verify(self, client, payload):
        return client.post(
            "/api/bot/turnstile-verify",
            data=target.json.dumps(payload),
            content_type="application/json",
            headers={"X-CSRF-Token": "csrf-test-token"},
        )

    def post_bot_lock(self, client, payload=None):
        return client.post(
            "/api/bot/lock",
            data=target.json.dumps(payload or {}),
            content_type="application/json",
            headers={"X-CSRF-Token": "csrf-test-token"},
        )

    def clear_fail_cooldown(self, fake):
        """Allow the next failure to count (tests fire faster than the 15s cooldown)."""
        fake.delete(f"bot:tst:fail_cd:{self.user_id}")

    def clear_verified_marker(self, fake):
        fake.delete(f"bot:tst:v:{self.user_id}")

    def turnstile_env(self):
        # Turnstile keys are only enabled for the duration of the API-level gate tests,
        # so other test modules are not affected by the gate.
        return mock.patch.dict(
            os.environ,
            {"TURNSTILE_SITE_KEY": "test-site-key", "TURNSTILE_SECRET_KEY": "test-secret-key"},
        )
