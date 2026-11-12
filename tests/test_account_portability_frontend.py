"""Settings import confirmation, progress, and dedupe UI regressions."""
import hashlib
import io
import json
import os
import re
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock
import os

from tests.account_portability_test_support import (
    AccountPortabilityTestBase,
    MemoryRedis,
    read_app_source,
    read_chat_markup,
    target,
)


class AccountPortabilityFrontendTests(AccountPortabilityTestBase):
    def test_frontend_restores_background_export_with_download_and_cancel(self):
        root = os.path.dirname(os.path.dirname(__file__))
        js_assets = sorted((Path(root) / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(js_assets), 1)
        with open(js_assets[0], encoding="utf-8") as handle:
            source = handle.read()
        template = read_chat_markup()
        self.assertNotIn("await res.blob()", source)
        self.assertNotIn("frame.src = `/api/account/export?job_id=", source)
        self.assertIn("apiFetch('/api/account/export'", source)
        self.assertIn("keepalive: true", source)
        self.assertIn("/api/account/export/latest", source)
        self.assertIn("payload.download_url", source)
        self.assertIn("pollAccountTransfer", source)
        self.assertIn("/cancel`, manualSpinnerRequestOptions", source)
        self.assertIn("Number(payload.unreadable_count)", source)
        self.assertIn("downloading: 'ダウンロード中'", source)
        self.assertIn('id="account-transfer-progress-bar"', template)
        self.assertIn('id="account-transfer-cancel-btn"', template)
        self.assertIn('id="account-export-download-btn"', template)
        # The download button must re-verify availability before navigating so a
        # stale/expired URL never sends the user to a full-page 404.
        self.assertIn("accountExportDownloadBtn.addEventListener('click'", source)
        self.assertIn("event.preventDefault()", source)
        self.assertIn("data.available && data.download_url", source)
        self.assertIn("window.location.assign(data.download_url)", source)
        self.assertIn("window.location.assign(href)", source)
        with open(os.path.join(root, "worker.py"), encoding="utf-8") as handle:
            worker_source = handle.read()
        self.assertIn("worker.work(with_scheduler=True)", worker_source)

    def test_frontend_shows_terminal_export_state_when_resuming(self):
        root = os.path.dirname(os.path.dirname(__file__))
        js_assets = sorted((Path(root) / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(js_assets), 1)
        with open(js_assets[0], encoding="utf-8") as handle:
            source = handle.read()
        index = source.index("const refreshLatestAccountExport = async () => {")
        section = source[index:index + 3000]
        self.assertIn("['failed', 'cancelled', 'expired'].includes(data.state)", section)
        self.assertIn("renderAccountTransferProgress(data)", section)

    def test_export_job_uses_generous_timeout_for_large_accounts(self):
        root = os.path.dirname(os.path.dirname(__file__))
        app_source = read_app_source()
        match = re.search(r"ACCOUNT_EXPORT_JOB_TIMEOUT_SECONDS = (\d+)", app_source)
        self.assertIsNotNone(match)
        self.assertGreaterEqual(int(match.group(1)), 3600)
        export_section = app_source[app_source.index("def export_account_data("):]
        self.assertIn("job_timeout=ACCOUNT_EXPORT_JOB_TIMEOUT_SECONDS", export_section)

    def test_encrypted_export_files_are_streamed_without_media_cache(self):
        root = os.path.dirname(os.path.dirname(__file__))
        app_source = read_app_source()
        section_start = app_source.index("def _write_account_export_file(")
        section_end = app_source.index("\ndef ", section_start + 1)
        section = app_source[section_start:section_end]
        self.assertNotIn("_load_user_file_bytes", section)
        self.assertIn("decrypt_bytes(token)", section)
        self.assertIn('archive.open(archive_name, "w", force_zip64=True)', section)

    def test_export_streams_large_encrypted_file_correctly(self):
        # A file larger than the 1 MiB chunk size must round-trip intact.
        big = os.urandom(2 * 1024 * 1024 + 123)
        big_rel = f"{self.source_id}/big.bin"
        big_path = os.path.join(self.temp_dir.name, str(self.source_id), "big.bin.enc")
        with open(big_path, "wb") as handle:
            handle.write(target.encrypt_bytes(big))
        with target.app.app_context():
            target.db.session.add(target.FileCache(
                user_id=self.source_id,
                rel_path=big_rel,
                provider="label",
                file_uri="big.bin",
                state="ready",
            ))
            target.db.session.commit()
        archive_bytes = self.export_archive()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            manifest = json.loads(archive.read("account_data.json"))
            big_item = next(item for item in manifest["data"]["files"] if item["rel_path"] == big_rel)
            self.assertEqual(archive.read(big_item["archive_path"]), big)
            self.assertEqual(big_item["sha256"], hashlib.sha256(big).hexdigest())
            self.assertEqual(big_item["size_bytes"], len(big))

    def test_import_chunk_upload_accepts_chunks_larger_than_the_generic_body_limit(self):
        client = self.client_for(self.destination_id)
        start = client.post(
            "/api/account/import/upload/start",
            json={"size": 20 * 1024 * 1024},
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )
        self.assertEqual(start.status_code, 200, start.get_data(as_text=True))
        upload_id = start.get_json()["upload_id"]
        chunk = io.BytesIO(b"x" * (10 * 1024 * 1024))
        response = client.post(
            f"/api/account/import/upload/{upload_id}/chunk",
            data={"index": "0", "chunk": (chunk, "account.zip")},
            headers={"X-CSRF-Token": "csrf-test-token"},
            content_type="multipart/form-data",
            base_url="https://localhost",
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        self.assertEqual(response.get_json()["received"], 10 * 1024 * 1024)

    def test_import_progress_is_overall_and_continuous_after_upload(self):
        root = os.path.dirname(os.path.dirname(__file__))
        app_source = read_app_source()
        section = app_source[app_source.index("def import_account_data(") :]
        fixed = [36, 38, 57, 59, 93, 95, 98]
        self.assertEqual(sorted(fixed), fixed)
        self.assertIn('job_id, 36, "validating", "ZIPを検証しています"', section)
        self.assertIn('job_id, 38, "validating", "データ構成を確認しています"', section)
        self.assertIn('job_id, 57, "importing_settings"', section)
        self.assertIn('job_id, 59, "importing_credentials"', section)
        self.assertIn('job_id, 93, "importing_feedback"', section)
        self.assertIn('job_id, 95, "importing_diagnostics"', section)
        self.assertIn('job_id, 98, "finalizing"', section)
        for expr in [
            "progress = 38 + int(4 *", "progress = 43 + int(12 *", "progress = 60 + int(6 *",
            "progress = 67 + int(11 *", "progress = 79 + int(12 *",
        ]:
            self.assertIn(expr, section)
        self.assertIn('job_id, "completed", 100, "completed", "インポートが完了しました"', section)

    def test_import_progress_starts_after_upload_phase(self):
        root = os.path.dirname(os.path.dirname(__file__))
        app_source = read_app_source()
        section = app_source[app_source.index("def import_account_data(") :]
        first = re.search(r"_account_transfer_checkpoint\([^,]+,[^,]+,\s*(\d+), \"validating\"", section)
        self.assertIsNotNone(first)
        if first is not None:
            self.assertEqual(int(first.group(1)), 36)
        self.assertIn('"validating", "ZIPを検証しています"', section)
        self.assertIn('"finalizing", "変更を確定しています"', section)

    def test_frontend_import_progress_uses_overall_scale_and_poll_after_upload(self):
        root = os.path.dirname(os.path.dirname(__file__))
        js_assets = sorted((Path(root) / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(js_assets), 1)
        with open(js_assets[0], encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn("Math.min(35, Math.round((uploadedChunks / totalChunks) * 35))", source)
        self.assertNotIn("3 + Math.round((uploadedChunks / totalChunks) * 32)", source)
        self.assertIn("if (data.state !== 'pending') renderAccountTransferProgress(data)", source)
        import_index = source.index("pollPromise = pollAccountTransfer(transfer)")
        upload_complete = source.index("'アップロードを完了できません'")
        self.assertGreater(import_index, upload_complete, "polling must start only after the upload completes")
        self.assertIn("{progress: 100, phase: 'completed', message: 'インポートが完了しました'}", source)
        self.assertIn("uploading: 'ZIPをアップロード中'", source)

    def test_settings_import_requires_confirmation_and_does_not_apply(self):
        archive_bytes = self.export_archive()
        response = self.client_for(self.destination_id).post(
            "/api/account/import",
            data={
                "categories": "settings",
                "file": (io.BytesIO(archive_bytes), "account.zip"),
            },
            headers={"X-CSRF-Token": "csrf-test-token"},
            content_type="multipart/form-data",
            base_url="https://localhost",
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertEqual(payload["status"], "settings_confirmation")
        self.assertEqual(payload["error"], "settings_confirmation_required")
        changes = payload["settings_changes"]
        self.assertIsInstance(changes, list)
        self.assertTrue(changes)
        fields = {change["field"] for change in changes}
        self.assertIn("theme_color", fields)
        self.assertIn("system_prompt", fields)
        theme = next(change for change in changes if change["field"] == "theme_color")
        self.assertEqual(theme["current"], "#abcdef")
        self.assertEqual(theme["incoming"], "#123456")
        # Nothing must be written until the user confirms.
        with target.app.app_context():
            destination = target.db.session.get(target.User, self.destination_id)
            self.assertEqual(destination.theme_color, "#abcdef")
            self.assertIn(destination.system_prompt, (None, ''))

    def test_settings_import_applies_after_confirmation(self):
        archive_bytes = self.export_archive()
        client = self.client_for(self.destination_id)
        preview = client.post(
            "/api/account/import",
            data={
                "categories": "settings,api_credentials",
                "file": (io.BytesIO(archive_bytes), "account.zip"),
            },
            headers={"X-CSRF-Token": "csrf-test-token"},
            content_type="multipart/form-data",
            base_url="https://localhost",
        )
        self.assertEqual(preview.status_code, 200, preview.get_data(as_text=True))
        self.assertEqual(preview.get_json()["status"], "settings_confirmation")
        confirmed = client.post(
            "/api/account/import",
            data={
                "categories": "settings,api_credentials",
                "confirm_settings": "true",
                "file": (io.BytesIO(archive_bytes), "account.zip"),
            },
            headers={"X-CSRF-Token": "csrf-test-token"},
            content_type="multipart/form-data",
            base_url="https://localhost",
        )
        self.assertEqual(confirmed.status_code, 200, confirmed.get_data(as_text=True))
        with target.app.app_context():
            destination = target.db.session.get(target.User, self.destination_id)
            self.assertEqual(destination.theme_color, "#123456")
            self.assertEqual(target.decrypt_val(destination.system_prompt), "portable system prompt")
            self.assertEqual(target.decrypt_val(destination.openai_api_key), "portable-openai-key")

    def test_settings_import_bypasses_confirmation_when_confirm_settings_true(self):
        archive_bytes = self.export_archive()
        response = self.client_for(self.destination_id).post(
            "/api/account/import",
            data={
                "categories": "settings",
                "confirm_settings": "true",
                "file": (io.BytesIO(archive_bytes), "account.zip"),
            },
            headers={"X-CSRF-Token": "csrf-test-token"},
            content_type="multipart/form-data",
            base_url="https://localhost",
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertEqual(payload.get("status"), "ok")
        self.assertNotIn("settings_confirmation", payload)
        with target.app.app_context():
            destination = target.db.session.get(target.User, self.destination_id)
            self.assertEqual(destination.theme_color, "#123456")

    def test_settings_import_confirmation_records_in_transfer_status(self):
        archive_bytes = self.export_archive()
        client = self.client_for(self.destination_id)
        upload_id = self.chunked_upload(client, archive_bytes)
        job_id = "c0ffee" * 5 + "c" * 2  # 32 hex chars
        self.assertEqual(len(job_id), 32, job_id)
        response = client.post(
            "/api/account/import",
            json={
                "upload_id": upload_id,
                "categories": "settings",
                "job_id": job_id,
            },
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        status = client.get(
            f"/api/account/transfer/{job_id}",
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )
        self.assertEqual(status.status_code, 200, status.get_data(as_text=True))
        payload = status.get_json()
        self.assertEqual(payload["state"], "needs_settings_confirmation")
        self.assertIsInstance(payload.get("settings_changes"), list)
        self.assertTrue(payload["settings_changes"])

    def test_settings_confirmation_skipped_when_nothing_would_change(self):
        # When the imported settings already match the destination's current
        # values, no confirmation is needed and the import completes directly.
        with target.app.app_context():
            destination = target.db.session.get(target.User, self.destination_id)
            destination.theme_color = "#123456"
            destination.system_prompt = target.encrypt_val("portable system prompt")
            target.db.session.commit()
        archive_bytes = self.export_archive()
        response = self.client_for(self.destination_id).post(
            "/api/account/import",
            data={
                "categories": "settings",
                "file": (io.BytesIO(archive_bytes), "account.zip"),
            },
            headers={"X-CSRF-Token": "csrf-test-token"},
            content_type="multipart/form-data",
            base_url="https://localhost",
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        self.assertEqual(response.get_json().get("status"), "ok")
        self.assertNotIn("settings_confirmation", response.get_json())


    def test_export_includes_thread_public_id(self):
        archive_bytes = self.export_archive()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            manifest = json.loads(archive.read("account_data.json"))
            chats = manifest["data"]["chats"]
            self.assertEqual(len(chats), 1)
            self.assertEqual(chats[0]["public_id"], "source-thread-public-id")

    def test_import_twice_skips_duplicates(self):
        archive_bytes = self.export_archive()
        client = self.client_for(self.destination_id)
        categories = "chats,gems,files,feedback,diagnostics"
        first = client.post(
            "/api/account/import",
            data={
                "categories": categories,
                "file": (io.BytesIO(archive_bytes), "account.zip"),
            },
            headers={"X-CSRF-Token": "csrf-test-token"},
            content_type="multipart/form-data",
            base_url="https://localhost",
        )
        self.assertEqual(first.status_code, 200, first.get_data(as_text=True))
        self.assertEqual(first.get_json()["imported"]["chats"], 1)
        self.assertEqual(first.get_json()["imported"]["files"], 1)
        second = client.post(
            "/api/account/import",
            data={
                "categories": categories,
                "file": (io.BytesIO(archive_bytes), "account.zip"),
            },
            headers={"X-CSRF-Token": "csrf-test-token"},
            content_type="multipart/form-data",
            base_url="https://localhost",
        )
        self.assertEqual(second.status_code, 200, second.get_data(as_text=True))
        payload = second.get_json()
        self.assertEqual(payload["imported"]["chats"], 0)
        self.assertEqual(payload["imported"]["skills"], 0)
        self.assertEqual(payload["imported"]["files"], 0)
        self.assertEqual(payload["imported"]["feedback"], 0)
        self.assertEqual(payload["imported"]["diagnostics"], 0)
        self.assertEqual(payload["duplicates"]["chats"], 1)
        self.assertEqual(payload["duplicates"]["skills"], 1)
        self.assertEqual(payload["duplicates"]["files"], 1)
        self.assertEqual(payload["duplicates"]["feedback"], 1)
        self.assertEqual(payload["duplicates"]["diagnostics"], 1)
        with target.app.app_context():
            self.assertEqual(target.Thread.query.filter_by(user_id=self.destination_id).count(), 1)
            self.assertEqual(target.Skill.query.filter_by(user_id=self.destination_id).count(), 1)
            self.assertEqual(target.Feedback.query.filter_by(user_id=self.destination_id).count(), 1)
            self.assertEqual(target.FileCache.query.filter_by(user_id=self.destination_id).count(), 1)
            self.assertEqual(target.FirstTokenLatencyMetric.query.filter_by(user_id=self.destination_id).count(), 1)

    def test_self_import_skips_existing_threads(self):
        archive_bytes = self.export_archive()
        response = self.client_for(self.source_id).post(
            "/api/account/import",
            data={
                "categories": "chats",
                "file": (io.BytesIO(archive_bytes), "account.zip"),
            },
            headers={"X-CSRF-Token": "csrf-test-token"},
            content_type="multipart/form-data",
            base_url="https://localhost",
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertEqual(payload["imported"]["chats"], 0)
        self.assertEqual(payload["duplicates"]["chats"], 1)
        with target.app.app_context():
            self.assertEqual(target.Thread.query.filter_by(user_id=self.source_id).count(), 1)

    def test_import_twice_skips_trace_duplicates(self):
        with target.app.app_context():
            target.db.session.add(target.ChatLatencyTrace(
                user_id=self.source_id, job_id="trace-import-test", model="gemini-2.5-flash",
                execution_path="chat", client_first_event_type="content",
                client_first_latency_ms=120, client_total_latency_ms=500,
            ))
            target.db.session.commit()
        archive_bytes = self.export_archive()
        client = self.client_for(self.destination_id)
        first = client.post(
            "/api/account/import",
            data={
                "categories": "diagnostics",
                "file": (io.BytesIO(archive_bytes), "account.zip"),
            },
            headers={"X-CSRF-Token": "csrf-test-token"},
            content_type="multipart/form-data",
            base_url="https://localhost",
        )
        self.assertEqual(first.status_code, 200, first.get_data(as_text=True))
        self.assertEqual(first.get_json()["imported"]["diagnostics"], 2)
        second = client.post(
            "/api/account/import",
            data={
                "categories": "diagnostics",
                "file": (io.BytesIO(archive_bytes), "account.zip"),
            },
            headers={"X-CSRF-Token": "csrf-test-token"},
            content_type="multipart/form-data",
            base_url="https://localhost",
        )
        self.assertEqual(second.status_code, 200, second.get_data(as_text=True))
        payload = second.get_json()
        self.assertEqual(payload["imported"]["diagnostics"], 0)
        self.assertEqual(payload["duplicates"]["diagnostics"], 2)
        with target.app.app_context():
            self.assertEqual(target.ChatLatencyTrace.query.filter_by(user_id=self.destination_id).count(), 1)

    def test_dedupe_scan_does_not_grow_shared_decrypt_cache(self):
        # The dedupe scan decrypts every message for signature hashing.  It must
        # not populate the shared 4096-entry decrypt cache, otherwise a full
        # account scan would exhaust memory on small servers (observed as an OOM
        # kill of a gunicorn worker).
        dest = self.destination_id
        fixed_ts = target.datetime(2026, 1, 1, 0, 0, 0)
        with target.app.app_context():
            thread = target.Thread(user_id=dest, public_id="cache-thread", title="Cache check")
            target.db.session.add(thread)
            target.db.session.flush()
            for index in range(20):
                target.db.session.add(target.Message(
                    thread_id=thread.id, role="user",
                    content=target.encrypt_val(f"encrypted payload {index}"),
                    timestamp=fixed_ts, is_encrypted=True,
                ))
            target.db.session.commit()
        with target.app.app_context():
            cache_before = len(target._DECRYPT_CACHE)
            plan = target._dedupe_plan_for_user(dest)
            cache_after = len(target._DECRYPT_CACHE)
            self.assertEqual(cache_after, cache_before, "dedupe scan must bypass the shared decrypt cache")
            self.assertEqual(plan["removed"]["chats"], [])

    def test_dedupe_preview_and_execute_removes_duplicates(self):
        dest = self.destination_id
        fixed_ts = target.datetime(2026, 1, 1, 0, 0, 0)
        with target.app.app_context():
            t1 = target.Thread(user_id=dest, public_id="keep-thread", title="Duplicate chat")
            t2 = target.Thread(user_id=dest, public_id="dup-thread", title="Duplicate chat")
            target.db.session.add_all([t1, t2])
            target.db.session.flush()
            for thread in (t1, t2):
                target.db.session.add(target.Message(
                    thread_id=thread.id, role="user", content="hello",
                    timestamp=fixed_ts, is_encrypted=False,
                ))
                target.db.session.add(target.Message(
                    thread_id=thread.id, role="assistant", content="hi",
                    timestamp=fixed_ts, is_encrypted=False,
                ))
            # Each duplicate thread references its own content-identical file copy,
            # mirroring what happens when the same export is imported twice.
            target.db.session.add(target.Message(
                thread_id=t1.id, role="user", content="file",
                image_url=json.dumps([f"{dest}/same.bin"]),
                timestamp=fixed_ts, is_encrypted=False,
            ))
            target.db.session.add(target.Message(
                thread_id=t2.id, role="user", content="file",
                image_url=json.dumps([f"{dest}/same-copy.bin"]),
                timestamp=fixed_ts, is_encrypted=False,
            ))
            target.db.session.add(target.Skill(uuid="gem-keep-uuid", user_id=dest, name="Dup Gem", instruction="instr"))
            target.db.session.add(target.Skill(uuid="gem-dup-uuid", user_id=dest, name="Dup Gem", instruction="instr"))
            target.db.session.add(target.Feedback(
                user_id=dest, title="Dup feedback", message="body", status="new",
                created_at=target.datetime(2026, 1, 1, 0, 0, 0),
            ))
            target.db.session.add(target.Feedback(
                user_id=dest, title="Dup feedback", message="body", status="new",
                created_at=target.datetime(2026, 1, 1, 0, 0, 0),
            ))
            dest_dir = os.path.join(self.temp_dir.name, str(dest))
            os.makedirs(dest_dir, mode=0o700, exist_ok=True)
            for name in ("same.bin", "same-copy.bin"):
                with open(os.path.join(dest_dir, name), "wb") as handle:
                    handle.write(b"same content bytes")
                target.db.session.add(target.FileCache(
                    user_id=dest, rel_path=f"{dest}/{name}", provider="label",
                    file_uri=name, state="ready",
                ))
            target.db.session.commit()

        client = self.client_for(dest)
        preview = client.post(
            "/api/account/dedupe/preview",
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )
        self.assertEqual(preview.status_code, 200, preview.get_data(as_text=True))
        payload = preview.get_json()
        self.assertTrue(payload["has_duplicates"])
        self.assertEqual(payload["duplicates"]["chats"], 1)
        self.assertEqual(payload["duplicates"]["skills"], 1)
        self.assertEqual(payload["duplicates"]["files"], 1)
        self.assertEqual(payload["duplicates"]["feedback"], 1)
        self.assertEqual(payload["kept_referenced_files"], 0)

        execute = client.post(
            "/api/account/dedupe/execute",
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )
        self.assertEqual(execute.status_code, 200, execute.get_data(as_text=True))
        removed = execute.get_json()["removed"]
        self.assertEqual(removed["chats"], 1)
        self.assertEqual(removed["skills"], 1)
        self.assertEqual(removed["files"], 1)
        self.assertEqual(removed["feedback"], 1)

        with target.app.app_context():
            self.assertEqual(target.Thread.query.filter_by(user_id=dest).count(), 1)
            self.assertEqual(target.Thread.query.filter_by(user_id=dest).one().public_id, "keep-thread")
            self.assertEqual(target.Skill.query.filter_by(user_id=dest).count(), 1)
            self.assertEqual(target.Feedback.query.filter_by(user_id=dest).count(), 1)
            self.assertEqual(target.FileCache.query.filter_by(user_id=dest).count(), 1)
            self.assertTrue(target._get_file_disk_info(f"{dest}/same.bin")["exists"])
            self.assertFalse(target._get_file_disk_info(f"{dest}/same-copy.bin")["exists"])
            thread = target.Thread.query.filter_by(user_id=dest).one()
            ref_msg = target.Message.query.filter_by(thread_id=thread.id, role="user", content="file").one()
            self.assertEqual(json.loads(ref_msg.image_url), [f"{dest}/same.bin"])

    def test_dedupe_preview_reports_none_when_no_duplicates(self):
        client = self.client_for(self.destination_id)
        preview = client.post(
            "/api/account/dedupe/preview",
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )
        self.assertEqual(preview.status_code, 200, preview.get_data(as_text=True))
        payload = preview.get_json()
        self.assertFalse(payload["has_duplicates"])
        self.assertEqual(payload["total"], 0)

    def test_frontend_dedupe_button_and_handler_present(self):
        root = os.path.dirname(os.path.dirname(__file__))
        js_assets = sorted((Path(root) / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(js_assets), 1)
        with open(js_assets[0], encoding="utf-8") as handle:
            source = handle.read()
        template = read_chat_markup()
        self.assertIn("/api/account/dedupe/preview", source)
        self.assertIn("/api/account/dedupe/execute", source)
        self.assertIn("重複データは見つかりませんでした", source)
        self.assertIn('id="account-dedupe-btn"', template)
        self.assertIn('id="account-dedupe-result"', template)
        self.assertIn("重複データを確認して修復", template)



if __name__ == "__main__":
    unittest.main()
