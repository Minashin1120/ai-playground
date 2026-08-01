import os
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("FLASK_SECRET_KEY", "performance-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-performance-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")

import app as target


APP_ROOT = Path(__file__).resolve().parents[1]


class PerformanceRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        target.app.config.update(TESTING=True, TRUSTED_HOSTS=["localhost"])
        target._ensure_temp_chat_monitor_running = lambda: None

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

    def test_versioned_static_assets_do_not_touch_session_or_client_cookie(self):
        client = target.app.test_client()
        response = client.get(
            "/static/js/pwa_install.js?v=performance-test",
            base_url="https://localhost",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("Cache-Control"), "public, max-age=31536000, immutable")
        self.assertNotIn("Set-Cookie", response.headers)
        self.assertNotIn("Cookie", response.headers.get("Vary", ""))
        response.close()

    def test_server_upload_paths_do_not_reencode_images(self):
        source = (APP_ROOT / "app.py").read_text(encoding="utf-8")
        upload_source = source[source.index("def upload():"):source.index("def upload_init():")]
        complete_source = source[source.index("def upload_complete():"):source.index("def get_storage_usage():")]
        self.assertNotIn("Image.open", upload_source)
        self.assertNotIn("Image.open", complete_source)
        self.assertNotIn("WEBP", upload_source)
        self.assertNotIn("WEBP", complete_source)

    def test_browser_owns_format_conversion_and_quality_100_mode(self):
        chat_files = list((APP_ROOT / "static/js").glob("chat_core.v*.js"))
        self.assertEqual(len(chat_files), 1)
        source = chat_files[0].read_text(encoding="utf-8")
        template = (APP_ROOT / "templates/chat.html").read_text(encoding="utf-8")
        self.assertIn("convertImageFormatOnly", source)
        self.assertIn("quality: 1", source)
        self.assertIn("imageFilenameForMime", source)
        self.assertIn("品質100・リサイズ無効", template)

    def test_small_image_attachments_are_eligible_for_low_latency_execution(self):
        with tempfile.TemporaryDirectory() as upload_root:
            user_dir = Path(upload_root) / "7"
            user_dir.mkdir(parents=True)
            (user_dir / "one.png").write_bytes(b"png-image")
            (user_dir / "two.webp").write_bytes(b"webp-image")
            (user_dir / "notes.txt").write_text("not an image", encoding="utf-8")

            old_root = target.app.config["UPLOAD_FOLDER"]
            target.app.config["UPLOAD_FOLDER"] = upload_root
            try:
                self.assertTrue(target._is_low_latency_image_attachment_set(["7/one.png", "7/two.webp"]))
                self.assertFalse(target._is_low_latency_image_attachment_set(["7/notes.txt"]))
                self.assertFalse(target._is_low_latency_image_attachment_set(["7/missing.png"]))
            finally:
                target.app.config["UPLOAD_FOLDER"] = old_root

    def test_gemini_small_images_are_inlined_before_files_api_fallback(self):
        source = (APP_ROOT / "app.py").read_text(encoding="utf-8")
        image_branch = source[
            source.index("if mime.startswith('image/'):"):
            source.index("if mime.startswith('audio/'):", source.index("if mime.startswith('image/'):"))
        ]
        inline_pos = image_branch.index("inline_image_bytes + img_size <= _GEMINI_INLINE_IMAGE_MAX_BYTES")
        files_api_pos = image_branch.index("elif gemini_files_api_enabled")
        self.assertLess(inline_pos, files_api_pos)
        self.assertIn("types.Part.from_bytes", image_branch)

    def test_small_images_can_use_direct_and_fast_chat_paths(self):
        source = (APP_ROOT / "app.py").read_text(encoding="utf-8")
        route = source[source.index("def chat_stream():"):source.index("def estimate_prompt_tokens_api():")]
        self.assertIn("low_latency_image_attachments = _is_low_latency_image_attachment_set", route)
        self.assertGreaterEqual(route.count("(no_attachments or low_latency_image_attachments)"), 2)

    def test_browser_fast_mode_streams_directly_then_persists(self):
        chat_files = list((APP_ROOT / "static/js").glob("chat_core.v*.js"))
        self.assertEqual(len(chat_files), 1)
        source = chat_files[0].read_text(encoding="utf-8")
        template = (APP_ROOT / "templates/chat.html").read_text(encoding="utf-8")
        fast_source = source[
            source.index("async function sendBrowserFastMessage"):
            source.index("async function sendMessage()")
        ]

        direct_pos = fast_source.index("generativelanguage.googleapis.com")
        upload_pos = fast_source.index("await uploadBrowserFastLocalFiles()")
        save_pos = fast_source.index("/api/browser_fast_mode/save")
        self.assertLess(direct_pos, upload_pos)
        self.assertLess(upload_pos, save_pos)
        self.assertIn("sessionStorage.setItem(BROWSER_FAST_KEY_STORAGE", source)
        self.assertNotIn("browserFastApiKey,", fast_source[save_pos:])
        self.assertIn('id="browser-fast-mode-ignore-warning"', template)
        self.assertIn("生成中に再読み込み・タブ終了・通信切断", template)
        self.assertIn("回答完了後に画像をサーバーへアップロードしてDB保存", template)

    def test_browser_fast_mode_keeps_local_images_out_of_upload_until_completion(self):
        chat_files = list((APP_ROOT / "static/js").glob("chat_core.v*.js"))
        source = chat_files[0].read_text(encoding="utf-8")
        upload_branch = source[
            source.index("if (browserFastModeEnabled) {", source.index("async function handleFiles")):
            source.index("return await uploadFileWithProgress(t, rowObj);")
        ]

        self.assertIn("browserFastLocalFiles.set", upload_branch)
        self.assertIn("ローカル保持（未保存）", upload_branch)
        self.assertIn("return true", upload_branch)

    def test_stale_chunk_cleanup_removes_transient_data_without_rewriting(self):
        with tempfile.TemporaryDirectory() as upload_root:
            stale_dir = Path(upload_root) / ".chunks" / "7" / "up_1752156000_deadbeef"
            stale_dir.mkdir(parents=True)
            (stale_dir / "data.part").write_bytes(b"x" * (1024 * 1024))
            (stale_dir / "meta.json").write_text(
                json.dumps({"created": int(time.time()) - target._CHUNK_UPLOAD_MAX_AGE_SECONDS - 1}),
                encoding="utf-8",
            )
            old_time = int(time.time()) - target._CHUNK_UPLOAD_MAX_AGE_SECONDS - 1
            os.utime(stale_dir / "meta.json", (old_time, old_time))

            old_root = target.app.config["UPLOAD_FOLDER"]
            target.app.config["UPLOAD_FOLDER"] = upload_root
            try:
                active = target._cleanup_stale_chunk_uploads(7)
                stale_exists = stale_dir.exists()
            finally:
                target.app.config["UPLOAD_FOLDER"] = old_root

        self.assertEqual(active, 0)
        self.assertFalse(stale_exists)

    def test_recent_chunk_activity_prevents_automatic_cleanup(self):
        with tempfile.TemporaryDirectory() as upload_root:
            chunk_dir = Path(upload_root) / ".chunks" / "7" / "up_1752156000_deadbeef"
            chunk_dir.mkdir(parents=True)
            (chunk_dir / "data.part").write_bytes(b"x")
            (chunk_dir / "meta.json").write_text(
                json.dumps({
                    "created": int(time.time()) - target._CHUNK_UPLOAD_MAX_AGE_SECONDS - 100,
                    "updated": int(time.time()),
                }),
                encoding="utf-8",
            )

            old_root = target.app.config["UPLOAD_FOLDER"]
            target.app.config["UPLOAD_FOLDER"] = upload_root
            try:
                active = target._cleanup_stale_chunk_uploads(7)
                still_exists = chunk_dir.exists()
            finally:
                target.app.config["UPLOAD_FOLDER"] = old_root

        self.assertEqual(active, 1)
        self.assertTrue(still_exists)

    def test_chunk_sweep_is_rate_limited_and_runs_for_the_leader(self):
        with mock.patch.object(target.redis_conn, "set", return_value=True) as acquire:
            with mock.patch.object(target, "_cleanup_all_stale_chunk_uploads") as cleanup:
                target._maybe_sweep_stale_chunk_uploads()

        acquire.assert_called_once_with(
            target._CHUNK_SWEEP_REDIS_KEY,
            mock.ANY,
            nx=True,
            ex=target._CHUNK_SWEEP_INTERVAL_SECONDS,
        )
        cleanup.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
