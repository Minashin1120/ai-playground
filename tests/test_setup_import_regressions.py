import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from tests.chat_template import read_chat_markup

from tests.app_source import read_app_source
os.environ.setdefault("FLASK_SECRET_KEY", "setup-import-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-setup-import-tests.db")
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


class SetupImportRegressionTests(unittest.TestCase):
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
            target.task_queue, "enqueue", return_value=mock.Mock(id="queued-job")
        )
        self.redis_patcher.start()
        self.rate_limit_patcher.start()
        self.turnstile_patcher.start()
        self.enqueue_mock = self.enqueue_patcher.start()
        self.addCleanup(self.redis_patcher.stop)
        self.addCleanup(self.rate_limit_patcher.stop)
        self.addCleanup(self.turnstile_patcher.stop)
        self.addCleanup(self.enqueue_patcher.stop)
        with target.app.app_context():
            target.db.session.remove()
            target.db.engine.dispose()
            target.db.drop_all()
            target.db.create_all()
            self.setup_user = target.User(
                username="setup-import-user",
                is_setup_completed=False,
                enable_e2ee=False,
            )
            self.setup_user.set_password("setup-password")
            target.db.session.add(self.setup_user)
            target.db.session.commit()
            self.setup_user_id = self.setup_user.id

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

    def make_export_archive(self):
        manifest = {
            "format": target.ACCOUNT_EXPORT_FORMAT,
            "format_version": target.ACCOUNT_EXPORT_VERSION,
            "exported_at": target._portable_datetime(target.datetime.utcnow()),
            "data": {
                "settings": {},
                "api_credentials": {},
                "chats": [
                    {
                        "title": "Imported setup chat",
                        "is_bookmarked": False,
                        "updated_at": target._portable_datetime(target.datetime.utcnow()),
                        "messages": [
                            {"role": "user", "content": "portable question", "export_id": 1},
                            {"role": "assistant", "content": "portable answer", "export_id": 2, "parent_export_id": 1},
                        ],
                    }
                ],
                "gems": [],
                "files": [],
                "feedback": [],
                "diagnostics": {},
                "unreadable_files": [],
            },
        }
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as archive:
            archive.writestr("account_data.json", json.dumps(manifest, ensure_ascii=False))
        return buf.getvalue()

    def seed_unencrypted_chat(self, user_id):
        with target.app.app_context():
            thread = target.Thread(user_id=user_id, public_id="setup-import-thread", title="Plaintext chat")
            target.db.session.add(thread)
            target.db.session.flush()
            target.db.session.add(target.Message(
                thread_id=thread.id,
                role="user",
                content="plaintext content",
                is_encrypted=False,
            ))
            target.db.session.commit()

    def test_setup_page_renders_import_section_with_three_steps(self):
        response = self.client_for(self.setup_user_id).get("/setup", base_url="https://localhost")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        for expected in [
            'id="step-1"', 'id="step-2"', 'id="step-3"',
            '/ 3',
            'id="setup-account-import-file"',
            'id="setup-account-import-categories"',
            'id="setup-account-import-btn"',
            'id="setup-account-transfer-progress"',
            'id="setup-account-transfer-cancel-btn"',
            'value="settings"', 'value="api_credentials"', 'value="chats"',
            'value="gems"', 'value="files"', 'value="feedback"', 'value="diagnostics"',
            'onclick="nextStep(3)"', '#step-3 input',
        ]:
            self.assertIn(expected, html)

    def test_setup_post_with_e2ee_and_unencrypted_data_enqueues_migration(self):
        self.seed_unencrypted_chat(self.setup_user_id)
        response = self.client_for(self.setup_user_id).post(
            "/setup",
            data={
                "csrf_token": "csrf-test-token",
                "default_model": "gemini-3.6-flash",
                "enable_e2ee": "on",
            },
            base_url="https://localhost",
        )
        self.assertEqual(response.status_code, 302, response.get_data(as_text=True))
        self.assertEqual(response.headers["Location"], "/")
        self.assertTrue(self.enqueue_mock.called)
        self.assertEqual(self.enqueue_mock.call_args.args[0], target.migrate_e2ee_task)
        self.assertEqual(self.enqueue_mock.call_args.args[1], self.setup_user_id)
        self.assertIs(self.enqueue_mock.call_args.args[2], True)
        with target.app.app_context():
            user = target.db.session.get(target.User, self.setup_user_id)
            self.assertTrue(user.is_setup_completed)
            self.assertTrue(user.enable_e2ee)

    def test_setup_post_without_e2ee_does_not_enqueue_migration(self):
        self.seed_unencrypted_chat(self.setup_user_id)
        response = self.client_for(self.setup_user_id).post(
            "/setup",
            data={
                "csrf_token": "csrf-test-token",
                "default_model": "gemini-3.6-flash",
            },
            base_url="https://localhost",
        )
        self.assertEqual(response.status_code, 302, response.get_data(as_text=True))
        self.assertFalse(self.enqueue_mock.called)
        with target.app.app_context():
            user = target.db.session.get(target.User, self.setup_user_id)
            self.assertTrue(user.is_setup_completed)
            self.assertFalse(user.enable_e2ee)

    def test_setup_post_with_e2ee_but_no_data_does_not_enqueue_migration(self):
        response = self.client_for(self.setup_user_id).post(
            "/setup",
            data={
                "csrf_token": "csrf-test-token",
                "default_model": "gemini-3.6-flash",
                "enable_e2ee": "on",
            },
            base_url="https://localhost",
        )
        self.assertEqual(response.status_code, 302, response.get_data(as_text=True))
        self.assertFalse(self.enqueue_mock.called)

    def test_setup_post_preserves_imported_credentials_when_fields_empty(self):
        with target.app.app_context():
            user = target.db.session.get(target.User, self.setup_user_id)
            user.openai_api_key = target.encrypt_val("imported-openai-key")
            user.gemini_api_key = target.encrypt_val("imported-gemini-key")
            target.db.session.commit()
        response = self.client_for(self.setup_user_id).post(
            "/setup",
            data={
                "csrf_token": "csrf-test-token",
                "default_model": "gemini-3.6-flash",
            },
            base_url="https://localhost",
        )
        self.assertEqual(response.status_code, 302, response.get_data(as_text=True))
        with target.app.app_context():
            user = target.db.session.get(target.User, self.setup_user_id)
            self.assertEqual(target.decrypt_val(user.openai_api_key), "imported-openai-key")
            self.assertEqual(target.decrypt_val(user.gemini_api_key), "imported-gemini-key")

    def test_setup_post_overrides_credentials_when_fields_filled(self):
        with target.app.app_context():
            user = target.db.session.get(target.User, self.setup_user_id)
            user.openai_api_key = target.encrypt_val("old-key")
            target.db.session.commit()
        response = self.client_for(self.setup_user_id).post(
            "/setup",
            data={
                "csrf_token": "csrf-test-token",
                "default_model": "gemini-3.6-flash",
                "openai_key": "new-key",
            },
            base_url="https://localhost",
        )
        self.assertEqual(response.status_code, 302, response.get_data(as_text=True))
        with target.app.app_context():
            user = target.db.session.get(target.User, self.setup_user_id)
            self.assertEqual(target.decrypt_val(user.openai_api_key), "new-key")

    def test_setup_page_prefills_imported_default_model_and_backend(self):
        with target.app.app_context():
            user = target.db.session.get(target.User, self.setup_user_id)
            user.default_model = "claude-sonnet-4-6"
            user.gemini_backend = "vertex_ai"
            target.db.session.commit()
        response = self.client_for(self.setup_user_id).get("/setup", base_url="https://localhost")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("const setupDefaultModel = 'claude-sonnet-4-6';", html)
        self.assertIn("const setupGeminiBackend = 'vertex_ai';", html)
        self.assertNotIn('<option value="claude-sonnet-4-6" selected>', html)
        self.assertIn('value="claude-sonnet-4-6">Claude Sonnet 4.6</option>', html)

    def test_setup_page_falls_back_to_default_model_when_unknown(self):
        with target.app.app_context():
            user = target.db.session.get(target.User, self.setup_user_id)
            user.default_model = "totally-unknown-model"
            target.db.session.commit()
        response = self.client_for(self.setup_user_id).get("/setup", base_url="https://localhost")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('value="totally-unknown-model" selected>totally-unknown-model</option>', html)
        self.assertIn("const setupDefaultModel = 'totally-unknown-model';", html)
        self.assertNotIn('<option value="gemini-3.6-flash" selected>', html)

    def test_import_api_works_for_setup_stage_user(self):
        archive_bytes = self.make_export_archive()
        response = self.client_for(self.setup_user_id).post(
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
        self.assertEqual(response.get_json()["imported"]["chats"], 1)
        with target.app.app_context():
            thread = target.Thread.query.filter_by(user_id=self.setup_user_id).one()
            messages = target.Message.query.filter_by(thread_id=thread.id).order_by(target.Message.id).all()
            self.assertEqual(len(messages), 2)
            self.assertEqual(messages[1].parent_id, messages[0].id)

    def test_setup_import_chunked_upload_flow_works(self):
        client = self.client_for(self.setup_user_id)
        archive_bytes = self.make_export_archive()
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
        self.assertEqual(complete.get_json()["status"], "ready")
        imported = client.post(
            "/api/account/import",
            json={
                "upload_id": upload_id,
                "categories": "chats",
                "job_id": "f" * 32,
            },
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )
        self.assertEqual(imported.status_code, 200, imported.get_data(as_text=True))
        self.assertEqual(imported.get_json()["imported"]["chats"], 1)

    def test_setup_import_frontend_uses_shared_progress_pattern(self):
        root = os.path.dirname(os.path.dirname(__file__))
        with open(os.path.join(root, "templates", "setup.html"), encoding="utf-8") as handle:
            template = handle.read()
        self.assertIn("Step <span id=\"step-num\" class=\"text-white font-bold\">1</span> / 3", template)
        self.assertIn("[1,2,3].forEach", template)
        self.assertIn("setupImportActive", template)
        self.assertIn("window.ProgressSpinner\n                ? window.ProgressSpinner.manualRequestOptions(options)", template)
        self.assertIn("Math.min(35, Math.round((uploadedChunks / totalChunks) * 35))", template)
        self.assertIn("pollPromise = pollTransfer(transfer)", template)
        self.assertIn("{progress: 100, phase: 'completed', message: 'インポートが完了しました'}", template)
        self.assertIn("uploading: 'ZIPをアップロード中'", template)
        self.assertIn("'アップロードを完了できません'", template)
        self.assertGreater(
            template.index("pollPromise = pollTransfer(transfer)"),
            template.index("'アップロードを完了できません'"),
            "polling must start only after the upload completes",
        )

    def test_setup_post_enqueues_migration_only_with_unencrypted_data(self):
        root = os.path.dirname(os.path.dirname(__file__))
        source = read_app_source()
        self.assertIn("def _user_has_unencrypted_data(user):", source)
        self.assertIn("task_queue.enqueue(migrate_e2ee_task, current_user.id, True)", source)

    def test_setup_import_handles_storage_limit_file_selection(self):
        root = Path(__file__).resolve().parents[1]
        with open(root / "templates" / "setup.html", encoding="utf-8") as handle:
            template = handle.read()
        for expected in [
            'id="setup-import-files-modal"',
            'id="setup-import-files-grid"',
            'id="setup-import-files-summary"',
            'id="setup-import-files-confirm"',
            'id="setup-import-files-select-all"',
            'id="setup-import-files-none"',
            "data.error === 'storage_limit_files'",
            "selected_files: selectedFiles",
            "__none__",
            "showSetupImportFileSelection(data)",
            "setupImportActive",
            "modal-open",
            "modal-prep",
            "modal-close",
        ]:
            self.assertIn(expected, template)

    def test_settings_import_handles_storage_limit_file_selection(self):
        root = Path(__file__).resolve().parents[1]
        chat_template = read_chat_markup()
        js_assets = list((root / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(js_assets), 1)
        js_source = js_assets[0].read_text(encoding="utf-8")
        for expected in [
            'id="import-files-modal"',
            'id="import-files-grid"',
            'id="import-files-summary"',
            'id="import-files-confirm"',
            "showImportFileSelection(data)",
            "data.error === 'storage_limit_files'",
            "selected_files: selectedFiles",
            "'__none__'",
            "showModal('import-files-modal')",
            "hideModal('import-files-modal')",
        ]:
            self.assertIn(expected, chat_template if expected.startswith('id=') else js_source)

    def test_setup_import_reconciles_unreadable_response_with_transfer_status(self):
        root = Path(__file__).resolve().parents[1]
        with open(root / "templates" / "setup.html", encoding="utf-8") as handle:
            template = handle.read()
        for expected in [
            "fetchImportStatus",
            "settleUnreadableImport",
            "outcome.state === 'completed'",
            "outcome.state === 'needs_selection'",
            "outcome.state === 'running'",
            "status === 'done'",
            "status === 'reselect'",
            "status === 'cancelled'",
            "parseFailures < 2",
            "showSetupImportFileSelection({",
            "available_bytes: outcome.available_bytes",
        ]:
            self.assertIn(expected, template)

    def test_settings_import_reconciles_unreadable_response_with_transfer_status(self):
        root = Path(__file__).resolve().parents[1]
        js_assets = list((root / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(js_assets), 1)
        js_source = js_assets[0].read_text(encoding="utf-8")
        for expected in [
            "fetchImportStatus",
            "settleUnreadableImport",
            "outcome.state === 'completed'",
            "outcome.state === 'needs_selection'",
            "outcome.state === 'running'",
            "status === 'done'",
            "status === 'reselect'",
            "status === 'cancelled'",
            "parseFailures < 2",
            "showImportFileSelection({",
            "available_bytes: outcome.available_bytes",
        ]:
            self.assertIn(expected, js_source)

    def test_settings_import_refreshes_settings_modal_after_success(self):
        # Imported settings / API credentials must be reflected after the import
        # completes; otherwise stale pre-import form values remain and a later
        # save would overwrite the imported values.  The page is reloaded after
        # the import so every client-side setting is rebuilt from the server.
        root = Path(__file__).resolve().parents[1]
        js_assets = list((root / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(js_assets), 1)
        js_source = js_assets[0].read_text(encoding="utf-8")
        for expected in [
            "const populateSettingsFormFromData = (d) => {",
            "const refreshSettingsFormAfterImport = async () => {",
            "populateSettingsFormFromData(data)",
            "categories.includes('settings') || categories.includes('api_credentials')",
            "refreshSettingsFormAfterImport()",
            "setMinimalPromptMode(true)",
            "setCompactPromptMode(!!data.compact_prompt_mode)",
            "scheduleReload()",
            "location.reload()",
        ]:
            self.assertIn(expected, js_source)
        # The settings-import success message must report the imported counts.
        self.assertIn("`設定 ${imported.settings || 0}件`", js_source)
        self.assertIn("`API認証 ${imported.api_credentials || 0}件`", js_source)
        # openSettingsModal must delegate its population to the shared helper.
        self.assertIn("apiFetch(CHAT_CONFIG.urls.handleSettingsQuery).then(r=>r.json()).then(d=>{", js_source)
        refresh_index = js_source.index("const refreshSettingsFormAfterImport")
        finish_index = js_source.index("const finishImportSuccess")
        self.assertLess(
            refresh_index, finish_index,
            "the settings refresh helper must be defined before the import success handler uses it",
        )
        # The reload must be scheduled inside the settings refresh helper.
        refresh_body = js_source[refresh_index:finish_index]
        self.assertIn("location.reload()", refresh_body)
        self.assertIn("setTimeout(() => { location.reload(); }, 1100)", refresh_body)

    def test_setup_import_refreshes_step3_form_after_settings_import(self):
        # The setup form is rendered before the import runs.  After importing
        # settings, the step-3 form (default model / Gemini backend / Vertex
        # location) must be updated from the imported values so the setup POST
        # does not overwrite them with the pre-import defaults.
        root = Path(__file__).resolve().parents[1]
        with open(root / "templates" / "setup.html", encoding="utf-8") as handle:
            template = handle.read()
        for expected in [
            "const refreshSetupFormAfterSettingsImport = async () => {",
            "refreshSetupFormAfterSettingsImport()",
            "categories.indexOf('settings') !== -1",
            "select[name=\"default_model\"]",
            "setup-gemini-backend",
            "input[name=\"gemini_vertex_location\"]",
            "toggleSetupGeminiBackend()",
            "'設定 ' + (imported.settings || 0) + '件'",
            "'API認証 ' + (imported.api_credentials || 0) + '件'",
        ]:
            self.assertIn(expected, template)
        refresh_index = template.index("const refreshSetupFormAfterSettingsImport")
        finish_index = template.index("const finishImportSuccess")
        self.assertLess(
            refresh_index, finish_index,
            "the setup form refresh helper must be defined before the import success handler uses it",
        )


    def test_setup_import_settings_confirmation_ui_and_flow(self):
        root = Path(__file__).resolve().parents[1]
        with open(root / "templates" / "setup.html", encoding="utf-8") as handle:
            template = handle.read()
        for expected in [
            'id="setup-settings-confirmation-modal"',
            'id="setup-settings-confirmation-list"',
            'id="setup-settings-confirmation-count"',
            'id="setup-settings-confirmation-close"',
            'id="setup-settings-confirmation-cancel"',
            'id="setup-settings-confirmation-confirm"',
            'id="setup-account-import-settings-bypass"',
            "showSetupSettingsConfirmation",
            "data.status === 'settings_confirmation'",
            "outcome.state === 'needs_settings_confirmation'",
            "confirm_settings: (settingsConfirmed || settingsBypass)",
        ]:
            self.assertIn(expected, template)

    def test_setup_import_modals_have_explicit_zindex(self):
        root = Path(__file__).resolve().parents[1]
        with open(root / "static" / "css" / "app-design.css", encoding="utf-8") as handle:
            css = handle.read()
        block = css[css.index("#setup-import-files-modal,") :]
        block = block[: block.index("}") + 1]
        self.assertIn("#setup-settings-confirmation-modal", block)
        self.assertIn("z-index: 200", block)
        self.assertIn("position: fixed", block)


if __name__ == "__main__":
    unittest.main()
