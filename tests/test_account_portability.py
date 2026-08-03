import io
import json
import os
import tempfile
import unittest
import zipfile
from unittest import mock


os.environ.setdefault("FLASK_SECRET_KEY", "account-portability-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-account-portability-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target


class AccountPortabilityTests(unittest.TestCase):
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
        self.temp_dir.cleanup()

    def client_for(self, user_id):
        client = target.app.test_client()
        with client.session_transaction() as session:
            session["_user_id"] = str(user_id)
            session["_fresh"] = True
            session["csrf_token"] = "csrf-test-token"
        return client

    def export_archive(self):
        response = self.client_for(self.source_id).get(
            "/api/account/export", base_url="https://localhost"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/zip")
        payload = response.get_data()
        response.close()
        return payload

    def test_export_omits_identity_authentication_and_moderation_secrets(self):
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


if __name__ == "__main__":
    unittest.main()
