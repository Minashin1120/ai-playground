import hashlib
import io
import json
import os
import tempfile
import unittest
import zipfile
from unittest import mock

from cryptography.fernet import Fernet


os.environ.setdefault("FLASK_SECRET_KEY", "import-restore-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-import-restore-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target  # noqa: E402


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


class ImportRestoreRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        target.app.config.update(TESTING=True, MAINTENANCE_MODE=False, TRUSTED_HOSTS=["localhost"])
        target._ensure_temp_chat_monitor_running = lambda: None

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        target.app.config["UPLOAD_FOLDER"] = self.temp_dir.name
        self.memory_redis = MemoryRedis()
        self.redis_patcher = mock.patch.object(target, "redis_conn", self.memory_redis)
        self.rate_limit_patcher = mock.patch.object(target, "rate_limit", return_value=True)
        self.turnstile_patcher = mock.patch.object(target, "_bot_turnstile_active", return_value=False)
        self.enqueue_patcher = mock.patch.object(
            target.task_queue, "enqueue", return_value=mock.Mock(id="job")
        )
        self.redis_patcher.start()
        self.rate_limit_patcher.start()
        self.turnstile_patcher.start()
        self.enqueue_patcher.start()
        self.addCleanup(self.redis_patcher.stop)
        self.addCleanup(self.rate_limit_patcher.stop)
        self.addCleanup(self.turnstile_patcher.stop)
        self.addCleanup(self.enqueue_patcher.stop)
        with target.app.app_context():
            target.db.session.remove()
            target.db.engine.dispose()
            target.db.drop_all()
            target.db.create_all()
            user = target.User(
                username="restore-owner",
                is_setup_completed=True,
                enable_e2ee=True,
            )
            user.set_password("restore-password")
            target.db.session.add(user)
            target.db.session.commit()
            self.user_id = user.id
        self.plain = b"restored-file-bytes"
        self.rel_path = f"{self.user_id}/img.jpg"

    def tearDown(self):
        with target.app.app_context():
            target.db.session.remove()
            target.db.engine.dispose()
        self.temp_dir.cleanup()

    def client_for(self, user_id):
        client = target.app.test_client()
        with client.session_transaction() as session:
            session["_user_id"] = str(user_id)
            session["_fresh"] = True
            session["csrf_token"] = "csrf-test-token"
        return client

    def export_archive(self, mtime=None):
        sha = hashlib.sha256(self.plain).hexdigest()
        file_item = {
            "rel_path": self.rel_path,
            "archive_path": "files/000001.bin",
            "display_name": "img.jpg",
            "size_bytes": len(self.plain),
            "sha256": sha,
        }
        if mtime is not None:
            file_item["mtime"] = mtime
        manifest = {
            "format": target.ACCOUNT_EXPORT_FORMAT,
            "format_version": target.ACCOUNT_EXPORT_VERSION,
            "exported_at": target._portable_datetime(target.datetime.utcnow()),
            "data": {
                "settings": {},
                "api_credentials": {},
                "chats": [],
                "gems": [],
                "files": [file_item],
                "feedback": [],
                "diagnostics": {},
                "unreadable_files": [],
            },
        }
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as archive:
            archive.writestr("account_data.json", json.dumps(manifest, ensure_ascii=False))
            archive.writestr("files/000001.bin", self.plain)
        return buf.getvalue()

    def seed_broken_enc(self, rel_path=None):
        """Create an .enc at the path encrypted with a *different* key, so the
        current cipher cannot read it (the exact condition after a key loss)."""
        rel_path = rel_path or self.rel_path
        bogus = Fernet(Fernet.generate_key())
        disk = os.path.join(target.app.config["UPLOAD_FOLDER"], rel_path + ".enc")
        os.makedirs(os.path.dirname(disk), exist_ok=True)
        with open(disk, "wb") as fh:
            fh.write(bogus.encrypt(b"old-broken-content"))
        # Drop any cached bytes for this path so the broken content is read from
        # disk instead of a stale (possibly same-size/same-second) cache entry.
        target._media_bytes_cache_evict_path(rel_path)
        return disk

    def test_import_inplace_restores_file_at_original_path(self):
        self.seed_broken_enc()
        response = self.client_for(self.user_id).post(
            "/api/account/import",
            data={
                "categories": "files",
                "restore_inplace": "1",
                "file": (io.BytesIO(self.export_archive()), "account.zip"),
            },
            headers={"X-CSRF-Token": "csrf-test-token"},
            content_type="multipart/form-data",
            base_url="https://localhost",
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        self.assertEqual(response.get_json()["imported"]["files"], 1)
        with target.app.app_context():
            # The original broken .enc must be overwritten and now readable.
            info = target._get_file_disk_info(self.rel_path)
            self.assertTrue(info.get("exists"))
            self.assertTrue(info.get("is_encrypted"))
            self.assertEqual(target._load_user_file_bytes(self.rel_path, info), self.plain)
            # No "import-" duplicate should exist for this user's file.
            user_dir = os.path.join(target.app.config["UPLOAD_FOLDER"], str(self.user_id))
            names = os.listdir(user_dir)
            self.assertTrue(any(n == "img.jpg.enc" for n in names), names)
            self.assertFalse(any(n.startswith("import-") for n in names), names)

    def test_import_without_inplace_keeps_unique_path(self):
        self.seed_broken_enc()
        response = self.client_for(self.user_id).post(
            "/api/account/import",
            data={
                "categories": "files",
                "file": (io.BytesIO(self.export_archive()), "account.zip"),
            },
            headers={"X-CSRF-Token": "csrf-test-token"},
            content_type="multipart/form-data",
            base_url="https://localhost",
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        self.assertEqual(response.get_json()["imported"]["files"], 1)
        with target.app.app_context():
            # The broken .enc must remain untouched (still not readable by cipher).
            info = target._get_file_disk_info(self.rel_path)
            self.assertTrue(info.get("exists"))
            with open(os.path.join(target.app.config["UPLOAD_FOLDER"], self.rel_path + ".enc"), "rb") as fh:
                raw = fh.read()
            with self.assertRaises(Exception):
                target.cipher.decrypt(raw)
            # A separate import-* file was added instead.
            names = os.listdir(os.path.join(target.app.config["UPLOAD_FOLDER"], str(self.user_id)))
            self.assertTrue(any(n.startswith("import-") for n in names), names)

    def test_serve_file_409_for_key_mismatch_and_404_for_missing(self):
        self.seed_broken_enc()
        client = self.client_for(self.user_id)
        # Key mismatch: file exists but cannot be decrypted with the current key.
        ok_resp = client.get(
            f"/files/{self.rel_path}",
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )
        self.assertEqual(ok_resp.status_code, 409, ok_resp.get_data(as_text=True))
        self.assertEqual(ok_resp.get_json()["error"], "encryption_key_mismatch")
        # Thumbnail route returns 409 as well.
        thumb_resp = client.get(
            f"/files/thumb/{self.rel_path}",
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )
        self.assertEqual(thumb_resp.status_code, 409, thumb_resp.get_data(as_text=True))
        # Genuinely missing file still 404.
        missing_resp = client.get(
            f"/files/{self.user_id}/nope.jpg",
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )
        self.assertEqual(missing_resp.status_code, 404, missing_resp.get_data(as_text=True))


    def test_import_inplace_preserves_mtime(self):
        self.seed_broken_enc()
        original_mtime = 1700000000  # fixed past timestamp
        response = self.client_for(self.user_id).post(
            "/api/account/import",
            data={
                "categories": "files",
                "restore_inplace": "1",
                "file": (io.BytesIO(self.export_archive(mtime=original_mtime)), "account.zip"),
            },
            headers={"X-CSRF-Token": "csrf-test-token"},
            content_type="multipart/form-data",
            base_url="https://localhost",
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        disk = os.path.join(target.app.config["UPLOAD_FOLDER"], self.rel_path + ".enc")
        self.assertEqual(int(os.path.getmtime(disk)), original_mtime)

    def test_import_without_mtime_keeps_default_timestamp(self):
        # Backward compatibility: manifests without "mtime" (old exports) must
        # not crash and simply leave the current timestamp.
        self.seed_broken_enc()
        response = self.client_for(self.user_id).post(
            "/api/account/import",
            data={
                "categories": "files",
                "restore_inplace": "1",
                "file": (io.BytesIO(self.export_archive(mtime=None)), "account.zip"),
            },
            headers={"X-CSRF-Token": "csrf-test-token"},
            content_type="multipart/form-data",
            base_url="https://localhost",
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        with target.app.app_context():
            info = target._get_file_disk_info(self.rel_path)
            self.assertTrue(info.get("exists"))
            self.assertEqual(target._load_user_file_bytes(self.rel_path, info), self.plain)


if __name__ == "__main__":
    unittest.main()
