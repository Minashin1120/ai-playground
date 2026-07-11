import io
import os
import tempfile
import unittest
import zipfile
from unittest import mock


os.environ.setdefault("FLASK_SECRET_KEY", "security-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-security-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target


def make_docx(document_xml):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", document_xml)
    return buf.getvalue()


class SecurityRegressionTests(unittest.TestCase):
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
            user = target.User(username="security-test", is_setup_completed=True)
            user.set_password("test-password")
            target.db.session.add(user)
            target.db.session.commit()
            self.user_id = user.id

    def tearDown(self):
        with target.app.app_context():
            target.db.session.remove()
        self.temp_dir.cleanup()

    def authenticated_client(self):
        client = target.app.test_client()
        with client.session_transaction() as sess:
            sess["_user_id"] = str(self.user_id)
            sess["_fresh"] = True
            sess["csrf_token"] = "csrf-test-token"
        return client

    def test_chunk_upload_id_rejects_path_traversal(self):
        self.assertTrue(target._is_valid_chunk_upload_id("up_1752156000_deadbeef"))
        self.assertIsNone(target._chunk_session_dir(self.user_id, "../../app"))
        self.assertIsNone(target._chunk_session_dir(self.user_id, "up_1_deadbeef/../../app"))

    def test_chunk_upload_rejects_oversized_and_replayed_chunks(self):
        client = self.authenticated_client()
        headers = {"X-CSRF-Token": "csrf-test-token"}
        init_response = client.post(
            "/upload/init",
            json={"filename": "small.txt", "size": 3},
            headers=headers,
            base_url="https://localhost",
        )
        self.assertEqual(init_response.status_code, 200, init_response.get_data(as_text=True))
        upload_id = init_response.get_json()["upload_id"]

        bad_response = client.post(
            "/upload/chunk",
            data={
                "upload_id": upload_id,
                "index": "0",
                "total": "1",
                "chunk": (io.BytesIO(b"four"), "chunk.bin"),
            },
            headers=headers,
            base_url="https://localhost",
        )
        self.assertEqual(bad_response.status_code, 400)

        good_response = client.post(
            "/upload/chunk",
            data={
                "upload_id": upload_id,
                "index": "0",
                "total": "1",
                "chunk": (io.BytesIO(b"abc"), "chunk.bin"),
            },
            headers=headers,
            base_url="https://localhost",
        )
        self.assertEqual(good_response.status_code, 200, good_response.get_data(as_text=True))

        replay_response = client.post(
            "/upload/chunk",
            data={
                "upload_id": upload_id,
                "index": "0",
                "total": "1",
                "chunk": (io.BytesIO(b"abc"), "chunk.bin"),
            },
            headers=headers,
            base_url="https://localhost",
        )
        self.assertEqual(replay_response.status_code, 409)

    def test_docx_parser_rejects_entities(self):
        malicious = b'''<?xml version="1.0"?>
<!DOCTYPE document [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body><w:p><w:r><w:t>&xxe;</w:t></w:r></w:p></w:body>
</w:document>'''
        self.assertIsNone(target._extract_text_from_docx(make_docx(malicious)))

    def test_docx_parser_accepts_normal_content(self):
        normal = b'''<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body><w:p><w:r><w:t>safe text</w:t></w:r></w:p></w:body>
</w:document>'''
        self.assertEqual(target._extract_text_from_docx(make_docx(normal)), "safe text")

    def test_rich_paste_pdf_does_not_fetch_remote_images(self):
        with target.app.test_request_context("/api/rich-paste/pdf"):
            with mock.patch.object(target.requests, "get", side_effect=AssertionError("network fetch")) as get_mock:
                pdf = target._build_rich_paste_pdf_bytes(
                    "test",
                    '<p>safe</p><img src="http://127.0.0.1:80/secret" alt="blocked">',
                )
        self.assertTrue(pdf.startswith(b"%PDF"))
        get_mock.assert_not_called()

    def test_plain_labels_remove_markup_delimiters(self):
        self.assertEqual(target._normalize_thread_title('<img src=x onerror="alert(1)">'), 'img src=x onerror="alert(1)"')
        payload = target._normalize_gem_payload({
            "name": "<svg onload=alert(1)>",
            "instruction": "safe",
            "fixed_prompts": [{"name": "<b>run</b>", "content": "hello"}],
        })
        self.assertNotIn("<", payload["name"])
        self.assertNotIn("<", payload["fixed_prompts_json"])

    def test_csrf_logout_and_security_headers(self):
        client = self.authenticated_client()
        get_logout = client.get("/logout", base_url="https://localhost")
        self.assertEqual(get_logout.status_code, 405)
        post_without_token = client.post("/logout", base_url="https://localhost")
        self.assertEqual(post_without_token.status_code, 403)

        version_response = client.get("/api/version", base_url="https://localhost")
        self.assertEqual(version_response.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(version_response.headers.get("X-Frame-Options"), "DENY")
        self.assertIn("frame-ancestors 'none'", version_response.headers.get("Content-Security-Policy", ""))
        self.assertIn("max-age=31536000", version_response.headers.get("Strict-Transport-Security", ""))
        self.assertIn("no-store", version_response.headers.get("Cache-Control", ""))

        untrusted_host = client.get("/api/version", base_url="https://evil.example")
        self.assertEqual(untrusted_host.status_code, 400)

    def test_client_ip_uses_proxy_appended_address(self):
        with target.app.test_request_context(
            "/", headers={"X-Forwarded-For": "198.51.100.50, 203.0.113.20"}
        ):
            self.assertEqual(target.get_client_ip(), "203.0.113.20")

    def test_settings_never_return_saved_api_key_material(self):
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            user.openai_api_key = target.encrypt_val("sk-sensitive-test-value")
            user.gemini_vertex_credentials_json = target.encrypt_val('{"private_key":"secret"}')
            target._save_user_model_api_key_map(user, {"gpt-test": "model-secret"})
            target.db.session.commit()
        client = self.authenticated_client()
        response = client.get("/api/settings", base_url="https://localhost")
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        body = response.get_data(as_text=True)
        self.assertNotIn("sk-sensitive-test-value", body)
        self.assertNotIn("private_key", body)
        self.assertNotIn("model-secret", body)
        payload = response.get_json()
        self.assertEqual(payload["openai_key"], target._SECRET_MASK)
        self.assertEqual(payload["gemini_vertex_credentials_json"], target._SECRET_MASK)
        self.assertEqual(payload["model_api_keys"]["gpt-test"], target._SECRET_MASK)

    def test_liquid_glass_setting_round_trip_and_initial_render(self):
        client = self.authenticated_client()
        response = client.post(
            "/api/settings",
            json={"liquid_glass_enabled": True},
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))

        settings_response = client.get("/api/settings", base_url="https://localhost")
        self.assertTrue(settings_response.get_json()["liquid_glass_enabled"])

        chat_response = client.get("/", base_url="https://localhost")
        self.assertEqual(chat_response.status_code, 200, chat_response.get_data(as_text=True))
        self.assertIn("liquid-glass-mode", chat_response.get_data(as_text=True))

    def test_liquid_glass_column_is_repaired_when_migrations_are_disabled(self):
        with target.app.app_context():
            target.db.session.remove()
            target.db.session.execute(target.text("ALTER TABLE user DROP COLUMN liquid_glass_enabled"))
            target.db.session.commit()
            before = {column["name"] for column in target.inspect(target.db.engine).get_columns("user")}
            self.assertNotIn("liquid_glass_enabled", before)

            target.ensure_user_liquid_glass_column()

            after = {column["name"] for column in target.inspect(target.db.engine).get_columns("user")}
            self.assertIn("liquid_glass_enabled", after)

    def test_liquid_glass_uses_functional_layer_and_dynamic_highlights(self):
        root = os.path.dirname(os.path.dirname(__file__))
        version = target.app.config["SYSTEM_VERSION"].lower()
        css_path = os.path.join(root, "static", "css", f"chat.custom.{version}.css")
        js_path = os.path.join(root, "static", "js", f"chat_core.{version}.js")
        with open(css_path, encoding="utf-8") as css_file:
            css = css_file.read()
        with open(js_path, encoding="utf-8") as js_file:
            script = js_file.read()

        self.assertIn("Functional glass floats above a standard content layer", css)
        self.assertIn("Standard content materials: deliberately not Liquid Glass", css)
        self.assertIn("body.liquid-glass-mode .message-bubble.bg-gray-700", css)
        self.assertIn("backdrop-filter: none !important", css)
        self.assertIn("LIQUID_GLASS_SURFACE_SELECTOR", script)
        self.assertIn("--glass-light-x", script)
        self.assertIn("document.addEventListener('pointermove'", script)
        surface_rule = css.split("body.liquid-glass-mode .liquid-glass-surface {", 1)[1].split("}", 1)[0]
        self.assertNotIn("position: relative", surface_rule)
        self.assertIn("body.liquid-glass-mode #sidebar {\n        position: fixed;", css)
        self.assertIn("body.liquid-glass-mode .composer-dock {\n        margin: 0;", css)
        self.assertIn("body.liquid-glass-mode > .flex-1 > header {\n        margin: 0;", css)

    def test_stop_chat_requires_owned_pending_job(self):
        with target.app.app_context():
            thread = target.Thread(user_id=self.user_id, public_id=target.generate_thread_public_id())
            target.db.session.add(thread)
            target.db.session.commit()
            thread_id = thread.public_id
        client = self.authenticated_client()
        headers = {"X-CSRF-Token": "csrf-test-token"}
        owned_job = f"job_1752156000_{self.user_id}_deadbeefdeadbeef"
        other_job = f"job_1752156000_{self.user_id}_feedfacefeedface"
        pending = ('{"job_id":"' + owned_job + '"}').encode()
        with mock.patch.object(target.redis_conn, "get", return_value=pending), mock.patch.object(target.redis_conn, "set") as redis_set:
            mismatch = client.post(
                "/api/stop_chat",
                json={"thread_id": thread_id, "job_id": other_job},
                headers=headers,
                base_url="https://localhost",
            )
            self.assertEqual(mismatch.status_code, 404)
            stopped = client.post(
                "/api/stop_chat",
                json={"thread_id": thread_id, "job_id": owned_job},
                headers=headers,
                base_url="https://localhost",
            )
            self.assertEqual(stopped.status_code, 200)
            redis_set.assert_called_once_with(f"stop_job:{owned_job}", "1", ex=300)

    def test_frontend_escapes_stored_values_and_forbids_iframes(self):
        root = os.path.dirname(os.path.dirname(__file__))
        script_version = target.app.config["SYSTEM_VERSION"].lower()
        script_path = os.path.join(root, "static", "js", f"chat_core.{script_version}.js")
        with open(script_path, encoding="utf-8") as script_file:
            script = script_file.read()
        with open(os.path.join(root, "templates", "chat.html"), encoding="utf-8") as template_file:
            template = template_file.read()
        self.assertIn('${escapeHtml(t.title || "No Title")}', script)
        self.assertIn('${escapeHtml(g.name)}', script)
        self.assertIn('Error: ${escapeHtml(j.content)}', script)
        self.assertIn("FORBID_TAGS: ['iframe', 'object', 'embed']", template)
        self.assertIn("dompurify/3.4.11/", template)
        self.assertIn('sandbox="allow-scripts allow-forms allow-modals allow-popups"', template)
        self.assertIn("openSandboxedHtmlTab(safe)", script)
        self.assertIn("frame.setAttribute('sandbox', '')", script)


if __name__ == "__main__":
    unittest.main()
