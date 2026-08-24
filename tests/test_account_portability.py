import io
import hashlib
import json
import os
import re
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


os.environ.setdefault("FLASK_SECRET_KEY", "account-portability-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-account-portability-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target


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


class AccountPortabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        target.app.config.update(TESTING=True, MAINTENANCE_MODE=False, TRUSTED_HOSTS=["localhost"])
        target._ensure_temp_chat_monitor_running = lambda: None

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        target.app.config["UPLOAD_FOLDER"] = self.temp_dir.name
        self.export_dir = os.path.join(self.temp_dir.name, "account-exports")
        os.makedirs(self.export_dir, mode=0o700)
        self.memory_redis = MemoryRedis()

        def immediate_enqueue(func, *args, **kwargs):
            func(*args)
            return mock.Mock(id="test-job")

        self.redis_patcher = mock.patch.object(target, "redis_conn", self.memory_redis)
        self.export_dir_patcher = mock.patch.object(target, "_account_export_dir", return_value=self.export_dir)
        self.enqueue_patcher = mock.patch.object(target.task_queue, "enqueue", side_effect=immediate_enqueue)
        self.enqueue_in_patcher = mock.patch.object(target.task_queue, "enqueue_in", return_value=mock.Mock(id="cleanup-job"))
        self.redis_patcher.start()
        self.export_dir_patcher.start()
        self.enqueue_mock = self.enqueue_patcher.start()
        self.enqueue_in_mock = self.enqueue_in_patcher.start()
        self.addCleanup(self.redis_patcher.stop)
        self.addCleanup(self.export_dir_patcher.stop)
        self.addCleanup(self.enqueue_patcher.stop)
        self.addCleanup(self.enqueue_in_patcher.stop)
        with target.app.app_context():
            target.db.session.remove()
            target.db.engine.dispose()
            target.db.drop_all()
            target.db.create_all()
            source = target.User(
                username="source-account",
                is_setup_completed=True,
                enable_e2ee=True,
                theme_color="#123456",
                system_prompt=target.encrypt_val("portable system prompt"),
                openai_api_key=target.encrypt_val("portable-openai-key"),
                totp_secret=target.encrypt_val("must-not-export"),
                is_2fa_enabled=True,
                is_admin=True,
                is_bot_banned=True,
            )
            source.set_password("source-password")
            destination = target.User(
                username="destination-account",
                is_setup_completed=True,
                enable_e2ee=True,
                theme_color="#abcdef",
            )
            destination.set_password("destination-password")
            target.db.session.add_all([source, destination])
            target.db.session.flush()
            self.source_id = source.id
            self.destination_id = destination.id

            gem = target.Gem(
                uuid="11111111-1111-1111-1111-111111111111",
                user_id=source.id,
                name="Portable Gem",
                description="Description",
                instruction="Gem instruction",
                default_model="gemini-2.5-flash",
            )
            target.db.session.add(gem)
            thread = target.Thread(
                user_id=source.id,
                public_id="source-thread-public-id",
                title="Portable chat",
                is_bookmarked=True,
                last_gem_uuid=gem.uuid,
            )
            target.db.session.add(thread)
            target.db.session.flush()
            first = target.Message(
                thread_id=thread.id,
                role="user",
                content=target.encrypt_val("portable question"),
                image_url=json.dumps([f"{source.id}/portable.txt"]),
                is_encrypted=True,
                gem_uuid=gem.uuid,
                gem_name=gem.name,
            )
            target.db.session.add(first)
            target.db.session.flush()
            target.db.session.add(target.Message(
                thread_id=thread.id,
                role="assistant",
                content=target.encrypt_val("portable answer"),
                thought_data=target.encrypt_val("portable thought"),
                is_encrypted=True,
                parent_id=first.id,
                model="gemini-2.5-flash",
            ))
            target.db.session.add(target.Feedback(
                user_id=source.id,
                title="Portable feedback",
                message="Feedback body",
                status="replied",
                admin_reply="Reply body",
                handled_by="private-admin-name",
            ))
            target.db.session.add(target.FirstTokenLatencyMetric(
                user_id=source.id,
                job_id="source-job",
                model="gemini-2.5-flash",
                first_event_type="content",
                latency_seconds=1.25,
                latency_ms=1250,
            ))
            target.db.session.commit()

            source_dir = os.path.join(self.temp_dir.name, str(source.id))
            os.makedirs(source_dir, mode=0o700)
            with open(os.path.join(source_dir, "portable.txt.enc"), "wb") as handle:
                handle.write(target.encrypt_bytes(b"portable file body"))
            target.db.session.add(target.FileCache(
                user_id=source.id,
                rel_path=f"{source.id}/portable.txt",
                provider="label",
                file_uri="Friendly portable.txt",
                state="ready",
            ))
            target.db.session.commit()

        self.rate_limit_patcher = mock.patch.object(target, "rate_limit", return_value=True)
        self.turnstile_patcher = mock.patch.object(target, "_bot_turnstile_active", return_value=False)
        self.rate_limit_patcher.start()
        self.turnstile_patcher.start()
        self.addCleanup(self.rate_limit_patcher.stop)
        self.addCleanup(self.turnstile_patcher.stop)

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

    def export_archive(self):
        job_id = "e" * 32
        start = self.client_for(self.source_id).post(
            "/api/account/export",
            json={"job_id": job_id},
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )
        self.assertEqual(start.status_code, 202, start.get_data(as_text=True))
        response = self.client_for(self.source_id).get(
            f"/api/account/export/{job_id}/download", base_url="https://localhost"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/zip")
        payload = response.get_data()
        response.close()
        return payload

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
        self.assertEqual(imported["gems"], 1)
        self.assertEqual(imported["files"], 1)

        with target.app.app_context():
            destination = target.db.session.get(target.User, self.destination_id)
            self.assertEqual(destination.username, "destination-account")
            self.assertEqual(destination.theme_color, "#abcdef")
            self.assertIsNone(destination.openai_api_key)
            self.assertFalse(destination.is_admin)
            self.assertFalse(destination.is_bot_banned)

            gem = target.Gem.query.filter_by(user_id=destination.id).one()
            self.assertNotEqual(gem.uuid, "11111111-1111-1111-1111-111111111111")
            thread = target.Thread.query.filter_by(user_id=destination.id).one()
            self.assertEqual(thread.last_gem_uuid, gem.uuid)
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
            self.assertIsNone(feedback.handled_by)
            self.assertEqual(target.FirstTokenLatencyMetric.query.filter_by(user_id=destination.id).count(), 1)

    def test_settings_and_credentials_can_be_imported_without_authentication_state(self):
        archive_bytes = self.export_archive()
        response = self.client_for(self.destination_id).post(
            "/api/account/import",
            data={
                "categories": "settings,api_credentials",
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

    def chunked_upload(self, client, archive_bytes):
        start = client.post(
            "/api/account/import/upload/start",
            json={"size": len(archive_bytes)},
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )
        self.assertEqual(start.status_code, 200, start.get_data(as_text=True))
        upload_id = start.get_json()["upload_id"]
        chunk = client.post(
            f"/api/account/import/upload/{upload_id}/chunk",
            data={"index": "0", "chunk": (io.BytesIO(archive_bytes), "account.zip")},
            headers={"X-CSRF-Token": "csrf-test-token"},
            content_type="multipart/form-data",
            base_url="https://localhost",
        )
        self.assertEqual(chunk.status_code, 200, chunk.get_data(as_text=True))
        complete = client.post(
            f"/api/account/import/upload/{upload_id}/complete",
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )
        self.assertEqual(complete.status_code, 200, complete.get_data(as_text=True))
        return upload_id

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

    def test_frontend_restores_background_export_with_download_and_cancel(self):
        root = os.path.dirname(os.path.dirname(__file__))
        js_assets = sorted((Path(root) / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(js_assets), 1)
        with open(js_assets[0], encoding="utf-8") as handle:
            source = handle.read()
        with open(os.path.join(root, "templates", "chat.html"), encoding="utf-8") as handle:
            template = handle.read()
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
        with open(os.path.join(root, "app.py"), encoding="utf-8") as handle:
            app_source = handle.read()
        match = re.search(r"ACCOUNT_EXPORT_JOB_TIMEOUT_SECONDS = (\d+)", app_source)
        self.assertIsNotNone(match)
        self.assertGreaterEqual(int(match.group(1)), 3600)
        export_section = app_source[app_source.index("def export_account_data("):]
        self.assertIn("job_timeout=ACCOUNT_EXPORT_JOB_TIMEOUT_SECONDS", export_section)

    def test_encrypted_export_files_are_streamed_without_media_cache(self):
        root = os.path.dirname(os.path.dirname(__file__))
        with open(os.path.join(root, "app.py"), encoding="utf-8") as handle:
            app_source = handle.read()
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
        with open(os.path.join(root, "app.py"), encoding="utf-8") as handle:
            app_source = handle.read()
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
        with open(os.path.join(root, "app.py"), encoding="utf-8") as handle:
            app_source = handle.read()
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


if __name__ == "__main__":
    unittest.main()
