import io
import os
import tempfile
import unittest
from unittest import mock

from cryptography.fernet import Fernet


os.environ.setdefault("FLASK_SECRET_KEY", "key-rotation-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-key-rotation-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target  # noqa: E402
import scripts.rotate_encryption_key as rot  # noqa: E402


class MemoryRedis:
    def __init__(self):
        self.data = {}

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.data:
            return False
        self.data[key] = value
        return True

    def setex(self, key, ttl, value):
        self.data[key] = value
        return True

    def get(self, key):
        return self.data.get(key)

    def exists(self, key):
        return key in self.data

    def delete(self, *keys):
        count = 0
        for key in keys:
            if key in self.data:
                del self.data[key]
                count += 1
        return count

    def expire(self, key, ttl):
        return key in self.data


class KeyRotationRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        target.app.config.update(TESTING=True, MAINTENANCE_MODE=False, TRUSTED_HOSTS=["localhost"])
        target._ensure_temp_chat_monitor_running = lambda: None

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        target.app.config["UPLOAD_FOLDER"] = self.temp_dir.name
        self.memory_redis = MemoryRedis()
        self.redis_patcher = mock.patch.object(target, "redis_conn", self.memory_redis)
        self.enqueue_patcher = mock.patch.object(target.task_queue, "enqueue", return_value=mock.Mock(id="j"))
        self.redis_patcher.start()
        self.enqueue_patcher.start()
        self.addCleanup(self.redis_patcher.stop)
        self.addCleanup(self.enqueue_patcher.stop)
        with target.app.app_context():
            target.db.session.remove()
            target.db.engine.dispose()
            target.db.drop_all()
            target.db.create_all()
        # Save the real key ring so we can restore it (these tests mutate it).
        self._saved_ring = list(target._KEY_RING)
        self._saved_cipher = target.cipher

    def tearDown(self):
        target._KEY_RING[:] = self._saved_ring
        target.cipher = self._saved_cipher
        with target.app.app_context():
            target.db.session.remove()
            target.db.engine.dispose()
        self.temp_dir.cleanup()

    def _make_user(self, enable_e2ee=True):
        with target.app.app_context():
            user = target.User(username="rot-user", is_setup_completed=True, enable_e2ee=enable_e2ee)
            user.set_password("pw")
            target.db.session.add(user)
            target.db.session.commit()
            return user.id

    def test_decrypt_falls_back_to_historical_key_in_ring(self):
        hist = Fernet(Fernet.generate_key())
        target._KEY_RING.append(hist)
        value = hist.encrypt(b"historical secret").decode()
        self.assertEqual(target.decrypt_val(value), "historical secret")

    def test_encrypt_uses_active_key(self):
        value = target.encrypt_val("active-key value")
        # Active key (target.cipher) must be able to read it.
        self.assertEqual(target.cipher.decrypt(value.encode()).decode(), "active-key value")

    def test_rotation_migrates_readable_and_skips_orphan(self):
        old_cipher = Fernet(Fernet.generate_key())
        new_cipher = Fernet(Fernet.generate_key())
        user_id = self._make_user(enable_e2ee=True)
        # One readable value and one orphan-like value that the old key cannot read.
        with target.app.app_context():
            user = target.db.session.get(target.User, user_id)
            user.openai_api_key = old_cipher.encrypt(b"readable-key").decode()
            user.system_prompt = old_cipher.encrypt(b"readable-prompt").decode()
            orphan = Fernet(Fernet.generate_key())
            user.gemini_api_key = orphan.encrypt(b"unknown-key-data").decode()
            target.db.session.commit()

        # scan: the orphan (gemini_api_key) is reported, the others are readable.
        stats, orphans = rot._scan(old_cipher)
        self.assertIn(f"user#{user_id}.gemini_api_key", orphans)

        mig = rot._migrate(old_cipher, new_cipher)
        self.assertEqual(mig["updated_users"], 1)
        self.assertEqual(mig["orphan_users"], 1)

        # verify: migrated values decrypt with new key; orphan left on its own key.
        counts, problems, vorphans = rot._verify(new_cipher, old_cipher)
        self.assertEqual(problems, [])
        self.assertEqual(vorphans, 1)
        with target.app.app_context():
            user = target.db.session.get(target.User, user_id)
            self.assertEqual(new_cipher.decrypt(user.openai_api_key.encode()).decode(), "readable-key")
            self.assertEqual(new_cipher.decrypt(user.system_prompt.encode()).decode(), "readable-prompt")
            # The orphan is unchanged (still decryptable only by its own key).
            with self.assertRaises(Exception):
                new_cipher.decrypt(user.gemini_api_key.encode())


if __name__ == "__main__":
    unittest.main()
