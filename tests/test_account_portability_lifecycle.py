"""Account export/import archive lifecycle regressions."""
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

from tests.account_portability_test_support import (
    AccountPortabilityTestBase,
    MemoryRedis,
    read_app_source,
    read_chat_markup,
    target,
)


class AccountPortabilityTests(AccountPortabilityTestBase):
    def test_export_omits_identity_authentication_and_moderation_secrets(self):
        with target.app.app_context():
            rows = target._account_file_rows(self.source_id)
            self.assertTrue(rows)
            self.assertNotIn("data", rows[0])
            self.assertIn("info", rows[0])
        archive_bytes = self.export_archive()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            manifest = json.loads(archive.read("account_data.json"))
            serialized = json.dumps(manifest, ensure_ascii=False)
            self.assertEqual(manifest["format"], target.ACCOUNT_EXPORT_FORMAT)
            self.assertNotIn("username", manifest["data"])
            self.assertNotIn("source-account", serialized)
            self.assertNotIn("must-not-export", serialized)
            self.assertNotIn("private-admin-name", serialized)
            self.assertEqual(
                manifest["data"]["api_credentials"]["openai_api_key"],
                "portable-openai-key",
            )
            self.assertEqual(manifest["data"]["settings"]["system_prompt"], "portable system prompt")
            file_item = manifest["data"]["files"][0]
            self.assertEqual(archive.read(file_item["archive_path"]), b"portable file body")

    def test_export_preserves_unreadable_encrypted_files_without_failing(self):
        corrupt_bytes = b"not-a-valid-fernet-token"
        corrupt_rel = f"{self.source_id}/corrupt.bin"
        corrupt_path = os.path.join(self.temp_dir.name, str(self.source_id), "corrupt.bin.enc")
        with open(corrupt_path, "wb") as handle:
            handle.write(corrupt_bytes)

        archive_bytes = self.export_archive()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            manifest = json.loads(archive.read("account_data.json"))
            self.assertEqual(len(manifest["data"]["files"]), 1)
            self.assertEqual(len(manifest["data"]["unreadable_files"]), 1)
            recovery = manifest["data"]["unreadable_files"][0]
            self.assertEqual(recovery["rel_path"], corrupt_rel)
            self.assertFalse(recovery["importable"])
            self.assertTrue(recovery["encrypted_source"])
            self.assertEqual(archive.read(recovery["archive_path"]), corrupt_bytes)

        response = self.client_for(self.destination_id).post(
            "/api/account/import",
            data={
                "categories": "files",
                "file": (io.BytesIO(archive_bytes), "account.zip"),
            },
            headers={"X-CSRF-Token": "csrf-test-token"},
            content_type="multipart/form-data",
            base_url="https://localhost",
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        self.assertEqual(response.get_json()["imported"]["files"], 1)

    def test_selective_import_adds_content_without_changing_username_or_unselected_settings(self):
        archive_bytes = self.export_archive()
        response = self.client_for(self.destination_id).post(
            "/api/account/import",
            data={
                "categories": "chats,gems,files,feedback,diagnostics",
                "file": (io.BytesIO(archive_bytes), "account.zip"),
            },
            headers={"X-CSRF-Token": "csrf-test-token"},
            content_type="multipart/form-data",
            base_url="https://localhost",
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        imported = response.get_json()["imported"]
        self.assertEqual(imported["chats"], 1)
        self.assertEqual(imported["skills"], 1)
        self.assertEqual(imported["files"], 1)

        with target.app.app_context():
            destination = target.db.session.get(target.User, self.destination_id)
            self.assertEqual(destination.username, "destination-account")
            self.assertEqual(destination.theme_color, "#abcdef")
            self.assertIsNone(destination.openai_api_key)
            self.assertFalse(destination.is_admin)
            self.assertFalse(destination.is_bot_banned)

            skill = target.Skill.query.filter_by(user_id=destination.id).one()
            self.assertNotEqual(skill.uuid, "11111111-1111-1111-1111-111111111111")
            thread = target.Thread.query.filter_by(user_id=destination.id).one()
            self.assertEqual(thread.last_skill_uuid, skill.uuid)
            messages = target.Message.query.filter_by(thread_id=thread.id).order_by(target.Message.id).all()
            self.assertEqual(target.decrypt_val(messages[0].content), "portable question")
            self.assertEqual(target.decrypt_val(messages[1].content), "portable answer")
            self.assertEqual(messages[1].parent_id, messages[0].id)
            imported_ref = json.loads(messages[0].image_url)[0]
            self.assertTrue(imported_ref.startswith(f"{destination.id}/import-"))
            info = target._get_file_disk_info(imported_ref)
            self.assertTrue(info["is_encrypted"])
            self.assertEqual(target._load_user_file_bytes(imported_ref, info), b"portable file body")
            feedback = target.Feedback.query.filter_by(user_id=destination.id).one()
            self.assertEqual(feedback.status, "resolved_unreleased")
            self.assertIsNone(feedback.handled_by)
            self.assertEqual(
                [(r.role, r.message) for r in target.FeedbackReply.query.filter_by(feedback_id=feedback.id).order_by(target.FeedbackReply.id)],
                [("admin", "Reply body"), ("user", "Sender answer")])
            self.assertEqual(target.FirstTokenLatencyMetric.query.filter_by(user_id=destination.id).count(), 1)

    def test_settings_and_credentials_can_be_imported_without_authentication_state(self):
        archive_bytes = self.export_archive()
        response = self.client_for(self.destination_id).post(
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
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        with target.app.app_context():
            destination = target.db.session.get(target.User, self.destination_id)
            self.assertEqual(destination.username, "destination-account")
            self.assertEqual(destination.theme_color, "#123456")
            self.assertEqual(target.decrypt_val(destination.system_prompt), "portable system prompt")
            self.assertEqual(target.decrypt_val(destination.openai_api_key), "portable-openai-key")
            self.assertFalse(destination.is_2fa_enabled)
            self.assertIsNone(destination.totp_secret)
            self.assertEqual(target.Thread.query.filter_by(user_id=destination.id).count(), 0)

    def test_import_rejects_unsupported_archives_and_empty_selection(self):
        client = self.client_for(self.destination_id)
        empty = client.post(
            "/api/account/import",
            data={"categories": "", "file": (io.BytesIO(b"not-a-zip"), "bad.zip")},
            headers={"X-CSRF-Token": "csrf-test-token"},
            content_type="multipart/form-data",
            base_url="https://localhost",
        )
        self.assertEqual(empty.status_code, 400)
        self.assertEqual(empty.get_json()["error"], "categories_required")

        invalid = client.post(
            "/api/account/import",
            data={"categories": "settings", "file": (io.BytesIO(b"not-a-zip"), "bad.zip")},
            headers={"X-CSRF-Token": "csrf-test-token"},
            content_type="multipart/form-data",
            base_url="https://localhost",
        )
        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(invalid.get_json()["error"], "invalid_zip")

    def test_encrypted_import_uses_stored_size_for_storage_limit(self):
        for size in (0, 1, 15, 16, 17, 1024):
            self.assertEqual(target._fernet_encrypted_size(size), len(target.encrypt_bytes(b"x" * size)))
        archive_bytes = self.export_archive()
        # Plaintext size (~18B) fits under 50B but the Fernet-stored size does
        # not; the import must therefore ask for a file selection instead of
        # silently importing the file.
        with mock.patch.object(target, "_get_user_storage_limit_bytes", return_value=50):
            response = self.client_for(self.destination_id).post(
                "/api/account/import",
                data={
                    "job_id": "a" * 32,
                    "categories": "files",
                    "file": (io.BytesIO(archive_bytes), "account.zip"),
                },
                headers={"X-CSRF-Token": "csrf-test-token"},
                content_type="multipart/form-data",
                base_url="https://localhost",
            )
        self.assertEqual(response.status_code, 409, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertEqual(payload["error"], "storage_limit_files")
        self.assertEqual(len(payload["files"]), 1)
        self.assertIn("used_bytes", payload)
        self.assertIn("limit_bytes", payload)
        with target.app.app_context():
            self.assertEqual(target.FileCache.query.filter_by(user_id=self.destination_id).count(), 0)
            destination_dir = os.path.join(self.temp_dir.name, str(self.destination_id))
            self.assertFalse(os.path.exists(destination_dir))

    def test_import_storage_limit_returns_selection_and_keeps_upload(self):
        archive_bytes = self.export_archive()
        client = self.client_for(self.destination_id)
        upload_id = self.chunked_upload(client, archive_bytes)
        upload_dir = target._account_import_upload_dir(self.destination_id, upload_id)
        self.assertTrue(os.path.isdir(upload_dir))
        with mock.patch.object(target, "_get_user_storage_limit_bytes", return_value=50):
            response = client.post(
                "/api/account/import",
                json={
                    "upload_id": upload_id,
                    "categories": "files",
                    "job_id": "1" * 32,
                },
                headers={"X-CSRF-Token": "csrf-test-token"},
                base_url="https://localhost",
            )
        self.assertEqual(response.status_code, 409, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertEqual(payload["error"], "storage_limit_files")
        self.assertEqual(len(payload["files"]), 1)
        self.assertEqual(payload["files"][0]["archive_path"], "files/000001.bin")
        self.assertGreaterEqual(payload["available_bytes"], 0)
        self.assertTrue(os.path.isdir(upload_dir), "upload must be kept for the follow-up selection")

    def test_import_storage_limit_records_selection_in_transfer_status(self):
        archive_bytes = self.export_archive()
        client = self.client_for(self.destination_id)
        upload_id = self.chunked_upload(client, archive_bytes)
        job_id = "a" * 32
        with mock.patch.object(target, "_get_user_storage_limit_bytes", return_value=50):
            response = client.post(
                "/api/account/import",
                json={
                    "upload_id": upload_id,
                    "categories": "files",
                    "job_id": job_id,
                },
                headers={"X-CSRF-Token": "csrf-test-token"},
                base_url="https://localhost",
            )
        self.assertEqual(response.status_code, 409, response.get_data(as_text=True))
        status = client.get(
            f"/api/account/transfer/{job_id}",
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )
        self.assertEqual(status.status_code, 200, status.get_data(as_text=True))
        payload = status.get_json()
        self.assertEqual(payload["state"], "needs_selection")
        self.assertEqual(len(payload["files"]), 1)
        self.assertEqual(payload["files"][0]["archive_path"], "files/000001.bin")
        self.assertIn("available_bytes", payload)
        self.assertIn("limit_bytes", payload)
        self.assertIn("used_bytes", payload)

    def test_import_with_selected_files_imports_only_selection(self):
        archive_bytes = self.export_archive()
        client = self.client_for(self.destination_id)
        upload_id = self.chunked_upload(client, archive_bytes)
        with mock.patch.object(target, "_get_user_storage_limit_bytes", return_value=10_000):
            response = client.post(
                "/api/account/import",
                json={
                    "upload_id": upload_id,
                    "categories": "files",
                    "job_id": "2" * 32,
                    "selected_files": "files/000001.bin",
                },
                headers={"X-CSRF-Token": "csrf-test-token"},
                base_url="https://localhost",
            )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        self.assertEqual(response.get_json()["imported"]["files"], 1)
        self.assertFalse(os.path.isdir(target._account_import_upload_dir(self.destination_id, upload_id)))
        with target.app.app_context():
            self.assertEqual(target.FileCache.query.filter_by(user_id=self.destination_id).count(), 1)
            destination_dir = os.path.join(self.temp_dir.name, str(self.destination_id))
            self.assertTrue(any("portable" in name for _, _, names in os.walk(destination_dir) for name in names))

    def test_import_with_selected_none_imports_no_files(self):
        archive_bytes = self.export_archive()
        client = self.client_for(self.destination_id)
        upload_id = self.chunked_upload(client, archive_bytes)
        response = client.post(
            "/api/account/import",
            json={
                "upload_id": upload_id,
                "categories": "files",
                "job_id": "3" * 32,
                "selected_files": "__none__",
            },
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        self.assertEqual(response.get_json()["imported"]["files"], 0)
        with target.app.app_context():
            self.assertEqual(target.FileCache.query.filter_by(user_id=self.destination_id).count(), 0)

    def test_import_selected_files_still_over_limit_returns_selection_again(self):
        archive_bytes = self.export_archive()
        client = self.client_for(self.destination_id)
        upload_id = self.chunked_upload(client, archive_bytes)
        with mock.patch.object(target, "_get_user_storage_limit_bytes", return_value=10):
            response = client.post(
                "/api/account/import",
                json={
                    "upload_id": upload_id,
                    "categories": "files",
                    "job_id": "4" * 32,
                    "selected_files": "files/000001.bin",
                },
                headers={"X-CSRF-Token": "csrf-test-token"},
                base_url="https://localhost",
            )
        self.assertEqual(response.status_code, 409, response.get_data(as_text=True))
        self.assertEqual(response.get_json()["error"], "storage_limit_files")

    def test_import_chunked_upload_removed_on_failure(self):
        client = self.client_for(self.destination_id)
        upload_id = self.chunked_upload(client, b"not-a-zip")
        upload_dir = target._account_import_upload_dir(self.destination_id, upload_id)
        self.assertTrue(os.path.isdir(upload_dir))
        response = client.post(
            "/api/account/import",
            json={
                "upload_id": upload_id,
                "categories": "files",
                "job_id": "5" * 32,
            },
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )
        self.assertEqual(response.status_code, 400, response.get_data(as_text=True))
        self.assertEqual(response.get_json()["error"], "invalid_zip")
        self.assertFalse(os.path.isdir(upload_dir), "failed import must delete the chunked upload")

    def test_cancel_import_upload_endpoint_removes_upload(self):
        archive_bytes = self.export_archive()
        client = self.client_for(self.destination_id)
        upload_id = self.chunked_upload(client, archive_bytes)
        upload_dir = target._account_import_upload_dir(self.destination_id, upload_id)
        self.assertTrue(os.path.isdir(upload_dir))
        response = client.delete(
            f"/api/account/import/upload/{upload_id}",
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(os.path.isdir(upload_dir))

    def test_export_honors_server_side_cancellation(self):
        with mock.patch.object(target, "_account_transfer_cancelled", return_value=True), \
             mock.patch.object(target, "_set_account_transfer_status") as set_status:
            target.build_account_export_task(self.source_id, "b" * 32)
        self.assertTrue(any(call.args[2] == "cancelled" for call in set_status.call_args_list))

    def test_rq_failure_callback_removes_partial_export_and_unlocks_account(self):
        job_id = "a" * 32
        part_path = target._account_export_path(self.source_id, job_id, "part")
        with open(part_path, "wb") as handle:
            handle.write(b"partial-sensitive-export")
        self.memory_redis.set(target._account_export_active_key(self.source_id), job_id)
        job = mock.Mock(args=(self.source_id, job_id))
        target.account_export_job_failure(job, self.memory_redis, RuntimeError, RuntimeError("stopped"), "trace")
        self.assertFalse(os.path.exists(part_path))
        self.assertFalse(self.memory_redis.exists(target._account_export_active_key(self.source_id)))
        status = target._account_transfer_status_payload(self.source_id, job_id)
        self.assertEqual(status["state"], "failed")

    def test_export_survives_download_close_and_is_erased_after_one_hour(self):
        job_id = "d" * 32
        start = self.client_for(self.source_id).post(
            "/api/account/export",
            json={"job_id": job_id},
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )
        self.assertEqual(start.status_code, 202)
        self.assertIs(self.enqueue_mock.call_args.kwargs.get("on_failure"), target.account_export_job_failure)
        export_path = target._account_export_path(self.source_id, job_id)
        self.assertTrue(os.path.exists(export_path))
        latest = self.client_for(self.source_id).get(
            "/api/account/export/latest", base_url="https://localhost"
        )
        self.assertEqual(latest.status_code, 200)
        self.assertEqual(latest.get_json()["state"], "ready")
        self.assertEqual(latest.get_json()["download_url"], f"/api/account/export/{job_id}/download")
        self.assertTrue(self.enqueue_in_mock.called)
        denied = self.client_for(self.destination_id).get(
            f"/api/account/export/{job_id}/download", base_url="https://localhost"
        )
        self.assertEqual(denied.status_code, 404)
        response = self.client_for(self.source_id).get(
            f"/api/account/export/{job_id}/download",
            base_url="https://localhost",
            buffered=False,
        )
        self.assertEqual(response.status_code, 200)
        response.close()
        self.assertTrue(os.path.exists(export_path))
        metadata = json.loads(self.memory_redis.get(target._account_export_artifact_key(self.source_id, job_id)))
        with mock.patch.object(target.time, "time", return_value=metadata["expires_ts"] + 1):
            target.delete_account_export_task(self.source_id, job_id)
        self.assertFalse(os.path.exists(export_path))
        status = target._account_transfer_status_payload(self.source_id, job_id)
        self.assertEqual(status["state"], "expired")
        self.assertFalse(status["available"])

    def test_download_recovers_archive_when_redis_metadata_lapsed(self):
        job_id = "f" * 32
        client = self.client_for(self.source_id)
        start = client.post(
            "/api/account/export",
            json={"job_id": job_id},
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )
        self.assertEqual(start.status_code, 202)
        export_path = target._account_export_path(self.source_id, job_id)
        self.assertTrue(os.path.exists(export_path))
        # Simulate the Redis artifact + status records being evicted while the
        # archive file is still on disk (scheduler delay / eviction / TTL race).
        self.memory_redis.delete(target._account_export_artifact_key(self.source_id, job_id))
        self.memory_redis.delete(target._account_transfer_status_key(self.source_id, job_id))
        response = client.get(
            f"/api/account/export/{job_id}/download", base_url="https://localhost", buffered=False
        )
        self.assertEqual(response.status_code, 200)
        response.close()
        # Metadata is rebuilt from the on-disk file, so the latest status is
        # accurate again instead of "idle"/"not available".
        latest = client.get("/api/account/export/latest", base_url="https://localhost")
        self.assertEqual(latest.status_code, 200)
        payload = latest.get_json()
        self.assertEqual(payload["state"], "ready")
        self.assertTrue(payload["available"])
        self.assertEqual(payload["download_url"], f"/api/account/export/{job_id}/download")

    def test_failed_download_does_not_destroy_artifact_record(self):
        job_id = "b" * 32
        client = self.client_for(self.source_id)
        start = client.post(
            "/api/account/export",
            json={"job_id": job_id},
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )
        self.assertEqual(start.status_code, 202)
        export_path = target._account_export_path(self.source_id, job_id)
        self.assertTrue(os.path.exists(export_path))
        artifact_key = target._account_export_artifact_key(self.source_id, job_id)
        self.assertTrue(self.memory_redis.exists(artifact_key))
        # The archive file disappears (external cleanup), but the Redis record
        # must survive a failed download so the UI can show an accurate state
        # instead of silently dropping the record (which used to make the file
        # look like it disappeared after reloading).
        os.remove(export_path)
        response = client.get(
            f"/api/account/export/{job_id}/download", base_url="https://localhost"
        )
        self.assertEqual(response.status_code, 404)
        self.assertTrue(self.memory_redis.exists(artifact_key))
        latest = client.get("/api/account/export/latest", base_url="https://localhost")
        self.assertEqual(latest.status_code, 200)
        payload = latest.get_json()
        self.assertFalse(payload["available"])
        self.assertEqual(payload["state"], "expired")
        # The record is intentionally left intact for the scheduled cleanup.
        self.assertTrue(self.memory_redis.exists(artifact_key))

    def test_cancelled_import_removes_files_and_rolls_back_database(self):
        archive_bytes = self.export_archive()
        original_checkpoint = target._account_transfer_checkpoint

        def cancel_after_files(user_id, job_id, progress, phase, message=""):
            if phase == "importing_settings":
                raise target.AccountTransferCancelled()
            return original_checkpoint(user_id, job_id, progress, phase, message)

        with mock.patch.object(target, "_account_transfer_checkpoint", side_effect=cancel_after_files):
            response = self.client_for(self.destination_id).post(
                "/api/account/import",
                data={
                    "job_id": "c" * 32,
                    "categories": "files,settings",
                    "confirm_settings": "true",
                    "file": (io.BytesIO(archive_bytes), "account.zip"),
                },
                headers={"X-CSRF-Token": "csrf-test-token"},
                content_type="multipart/form-data",
                base_url="https://localhost",
            )
        self.assertEqual(response.status_code, 409, response.get_data(as_text=True))
        with target.app.app_context():
            destination = target.db.session.get(target.User, self.destination_id)
            self.assertEqual(destination.theme_color, "#abcdef")
            self.assertEqual(target.FileCache.query.filter_by(user_id=self.destination_id).count(), 0)
        destination_dir = os.path.join(self.temp_dir.name, str(self.destination_id))
        remaining = []
        if os.path.isdir(destination_dir):
            remaining = [name for _, _, names in os.walk(destination_dir) for name in names]
        self.assertEqual(remaining, [])



if __name__ == "__main__":
    unittest.main()
