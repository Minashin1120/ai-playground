import os
import json
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("FLASK_SECRET_KEY", "performance-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-performance-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")

import app as target


APP_ROOT = Path(__file__).resolve().parents[1]


class PerformanceRegressionTests(unittest.TestCase):
    def test_storage_usage_scan_is_reused_within_one_request(self):
        with tempfile.TemporaryDirectory() as upload_root:
            user_dir = Path(upload_root) / "7"
            chunk_dir = Path(upload_root) / ".chunks" / "7" / "up_1752156000_deadbeef"
            user_dir.mkdir(parents=True)
            chunk_dir.mkdir(parents=True)
            (user_dir / "stored.bin").write_bytes(b"a" * 11)
            (chunk_dir / "data.part").write_bytes(b"b" * 13)

            old_root = target.app.config["UPLOAD_FOLDER"]
            target.app.config["UPLOAD_FOLDER"] = upload_root
            try:
                with target.app.test_request_context("/"):
                    first = target._get_user_storage_usage_bytes(7)
                    (user_dir / "late.bin").write_bytes(b"c" * 17)
                    second = target._get_user_storage_usage_bytes(7)
                with target.app.test_request_context("/"):
                    next_request = target._get_user_storage_usage_bytes(7)
            finally:
                target.app.config["UPLOAD_FOLDER"] = old_root

        self.assertEqual(first, 24)
        self.assertEqual(second, 24)
        self.assertEqual(next_request, 41)

    def test_disabled_chat_cache_does_not_register_pwa_worker(self):
        pwa_source = (APP_ROOT / "static/js/pwa_install.js").read_text(encoding="utf-8")
        chat_files = list((APP_ROOT / "static/js").glob("chat_core.v*.js"))
        self.assertEqual(len(chat_files), 1)
        chat_source = chat_files[0].read_text(encoding="utf-8")

        self.assertIn("window.CHAT_CONFIG.useSwCache !== true", pwa_source)
        self.assertIn("SW_CACHE_MODE_STORAGE_KEY", chat_source)
        self.assertIn("previousMode !== 'disabled'", chat_source)

    def test_upload_webp_encoding_uses_fast_method(self):
        source = (APP_ROOT / "app.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("quality=80, method=2"), 2)
        self.assertEqual(source.count("quality=80,\n                            method=2"), 1)
        self.assertEqual(source.count("quality=80,\n                        method=2"), 1)

    def test_stale_chunk_cleanup_removes_transient_data_without_rewriting(self):
        with tempfile.TemporaryDirectory() as upload_root:
            stale_dir = Path(upload_root) / ".chunks" / "7" / "up_1752156000_deadbeef"
            stale_dir.mkdir(parents=True)
            (stale_dir / "data.part").write_bytes(b"x" * (1024 * 1024))
            (stale_dir / "meta.json").write_text(
                json.dumps({"created": int(time.time()) - target._CHUNK_UPLOAD_MAX_AGE_SECONDS - 1}),
                encoding="utf-8",
            )

            old_root = target.app.config["UPLOAD_FOLDER"]
            target.app.config["UPLOAD_FOLDER"] = upload_root
            try:
                active = target._cleanup_stale_chunk_uploads(7)
                stale_exists = stale_dir.exists()
            finally:
                target.app.config["UPLOAD_FOLDER"] = old_root

        self.assertEqual(active, 0)
        self.assertFalse(stale_exists)


if __name__ == "__main__":
    unittest.main()
