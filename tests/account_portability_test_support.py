"""Shared fixtures for account export/import portability tests."""
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

from tests.chat_template import read_chat_markup

from tests.app_source import read_app_source
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


class AccountPortabilityTestBase(unittest.TestCase):
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

            skill = target.Skill(
                uuid="11111111-1111-1111-1111-111111111111",
                user_id=source.id,
                name="Portable Skill",
                description="Description",
                instruction="Skill instruction",
            )
            target.db.session.add(skill)
            thread = target.Thread(
                user_id=source.id,
                public_id="source-thread-public-id",
                title="Portable chat",
                is_bookmarked=True,
                last_skill_uuid=skill.uuid,
            )
            target.db.session.add(thread)
            target.db.session.flush()
            first = target.Message(
                thread_id=thread.id,
                role="user",
                content=target.encrypt_val("portable question"),
                image_url=json.dumps([f"{source.id}/portable.txt"]),
                is_encrypted=True,
                skill_uuid=skill.uuid,
                skill_name=skill.name,
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
            portable_feedback = target.Feedback(
                user_id=source.id,
                title="Portable feedback",
                message="Feedback body",
                status="resolved_unreleased",
                admin_reply="Reply body",
                handled_by="private-admin-name",
            )
            target.db.session.add(portable_feedback)
            target.db.session.flush()
            for role, body in (("admin", "Reply body"), ("user", "Sender answer")):
                target.db.session.add(target.FeedbackReply(
                    feedback_id=portable_feedback.id, role=role, message=body,
                    author="private-admin-name" if role == "admin" else "source"))
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

