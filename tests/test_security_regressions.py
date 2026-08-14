import base64
import glob
import hashlib
import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from sqlalchemy.dialects import mysql
from sqlalchemy.schema import CreateTable


os.environ.setdefault("FLASK_SECRET_KEY", "security-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-security-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target


APP_ROOT = Path(__file__).resolve().parents[1]


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
        # The API-level Turnstile gate (V4.8.701) is out of scope for these
        # security regression tests; disable it so they exercise the target logic.
        self._turnstile_gate_patcher = mock.patch.object(target, "_bot_turnstile_active", return_value=False)
        self._turnstile_gate_patcher.start()
        self.addCleanup(self._turnstile_gate_patcher.stop)

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

    def test_chat_auth_resolver_covers_saved_model_and_admin_fallback_keys(self):
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)

            missing = target._resolve_chat_model_auth(user, "gpt-5.4")
            self.assertEqual(missing["error_code"], "api_key_missing")
            self.assertEqual(missing["provider"], "openai")

            user.openai_api_key = target.encrypt_val("saved-openai-key")
            saved = target._resolve_chat_model_auth(user, "gpt-5.4")
            self.assertIsNone(saved["error_code"])
            self.assertEqual(saved["api_key"], "saved-openai-key")

            user.openai_api_key = None
            target._save_user_model_api_key_map(user, {"gpt-5.4": "model-key"})
            model_specific = target._resolve_chat_model_auth(user, "gpt-5.4")
            self.assertIsNone(model_specific["error_code"])
            self.assertEqual(model_specific["api_key"], "model-key")

            user.model_api_keys = None
            user.is_admin = True
            user.admin_api_key_mode = "env_fallback"
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "admin-env-key"}):
                admin_fallback = target._resolve_chat_model_auth(user, "gpt-5.4")
            self.assertIsNone(admin_fallback["error_code"])
            self.assertEqual(admin_fallback["api_key"], "admin-env-key")

    def test_chat_auth_failure_is_returned_before_thread_or_message_is_saved(self):
        client = self.authenticated_client()
        response = client.post(
            "/chat_stream",
            json={"message": "hello", "model": "gpt-5.4"},
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertEqual(payload["code"], "api_key_missing")
        self.assertEqual(payload["model"], "gpt-5.4")
        self.assertEqual(payload["provider"], "openai")
        with target.app.app_context():
            self.assertEqual(target.Thread.query.count(), 0)
            self.assertEqual(target.Message.query.count(), 0)

    def test_browser_fast_mode_saves_completed_turn_atomically(self):
        client = self.authenticated_client()
        response = client.post(
            "/api/browser_fast_mode/save",
            json={
                "message": "fast prompt",
                "assistant_content": "fast answer",
                "thought_content": "private reasoning",
                "model": "gemini-2.5-flash",
                "image_urls": [],
            },
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )

        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertTrue(payload["thread_id"])
        with target.app.app_context():
            thread = target.Thread.query.one()
            messages = target.Message.query.order_by(target.Message.id.asc()).all()
            user = target.db.session.get(target.User, self.user_id)
            self.assertEqual(thread.public_id, payload["thread_id"])
            self.assertFalse(thread.include_global_instruction)
            self.assertEqual(thread.last_model, "gemini-2.5-flash")
            self.assertEqual(user.last_model, "gemini-2.5-flash")
            self.assertEqual([message.role for message in messages], ["user", "assistant"])
            self.assertEqual(messages[0].content, "fast prompt")
            self.assertEqual(messages[1].content, "fast answer")
            self.assertEqual(target.extract_reasoning_text(messages[1].thought_data), "private reasoning")
            self.assertEqual(messages[1].parent_id, messages[0].id)

    def test_browser_fast_mode_rejects_unsupported_or_unknown_chat_requests(self):
        client = self.authenticated_client()
        headers = {"X-CSRF-Token": "csrf-test-token"}
        base = {
            "message": "prompt",
            "assistant_content": "answer",
            "model": "gpt-5.6",
        }
        unsupported = client.post(
            "/api/browser_fast_mode/save", json=base, headers=headers, base_url="https://localhost"
        )
        unknown = client.post(
            "/api/browser_fast_mode/save",
            json={**base, "model": "gemini-2.5-flash", "thread_id": "existing"},
            headers=headers,
            base_url="https://localhost",
        )

        self.assertEqual(unsupported.status_code, 400)
        self.assertEqual(unknown.status_code, 403)
        with target.app.app_context():
            self.assertEqual(target.Thread.query.count(), 0)
            self.assertEqual(target.Message.query.count(), 0)

    def test_browser_fast_mode_uses_selected_branch_and_appends_to_existing_chat(self):
        with target.app.app_context():
            thread = target.Thread(user_id=self.user_id, public_id="fast-existing-thread", title="Existing")
            target.db.session.add(thread)
            target.db.session.flush()
            first_user = target.Message(thread_id=thread.id, role="user", content="first prompt", model="gemini-2.5-flash")
            target.db.session.add(first_user)
            target.db.session.flush()
            first_assistant = target.Message(
                thread_id=thread.id,
                role="assistant",
                content="first answer",
                model="gemini-2.5-flash",
                parent_id=first_user.id,
            )
            target.db.session.add(first_assistant)
            target.db.session.commit()
            parent_id = first_assistant.id

        client = self.authenticated_client()
        headers = {"X-CSRF-Token": "csrf-test-token"}
        signature = base64.b64encode(b"gemini-thought-signature").decode("ascii")
        response = client.post(
            "/api/browser_fast_mode/save",
            json={
                "message": "follow-up prompt",
                "assistant_content": "follow-up answer",
                "model": "gemini-2.5-flash",
                "thread_id": "fast-existing-thread",
                "parent_id": parent_id,
                "thought_signatures": [signature],
            },
            headers=headers,
            base_url="https://localhost",
        )

        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        self.assertFalse(response.get_json()["created_thread"])
        with target.app.app_context():
            self.assertEqual(target.Thread.query.count(), 1)
            messages = target.Message.query.order_by(target.Message.id.asc()).all()
            self.assertEqual(len(messages), 4)
            self.assertEqual(messages[2].parent_id, parent_id)
            self.assertEqual(messages[3].parent_id, messages[2].id)
            self.assertEqual(json.loads(messages[3].thought_signature), [signature])

    def test_browser_fast_mode_bootstrap_returns_user_model_key_and_branch_history(self):
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            user.gemini_api_key = target.encrypt_val("common-user-gemini-key")
            target._save_user_model_api_key_map(user, {"gemini-2.5-flash": "model-specific-gemini-key"})
            thread = target.Thread(user_id=self.user_id, public_id="fast-bootstrap-thread")
            target.db.session.add(thread)
            target.db.session.flush()
            first_user = target.Message(thread_id=thread.id, role="user", content="history prompt", model="gemini-2.5-flash")
            target.db.session.add(first_user)
            target.db.session.flush()
            first_assistant = target.Message(
                thread_id=thread.id,
                role="assistant",
                content="history answer",
                model="gemini-2.5-flash",
                parent_id=first_user.id,
            )
            target.db.session.add(first_assistant)
            target.db.session.commit()
            parent_id = first_assistant.id

        client = self.authenticated_client()
        response = client.post(
            "/api/browser_fast_mode/bootstrap",
            json={
                "model": "gemini-2.5-flash",
                "thread_id": "fast-bootstrap-thread",
                "parent_id": parent_id,
            },
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )

        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertEqual(payload["api_key"], "model-specific-gemini-key")
        self.assertEqual(payload["key_source"], "model_specific")
        self.assertEqual([item["role"] for item in payload["history"]], ["user", "model"])
        self.assertEqual([item["text"] for item in payload["history"]], ["history prompt", "history answer"])
        self.assertIn("no-store", response.headers.get("Cache-Control", ""))

        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            target._save_user_model_api_key_map(user, {})
            target.db.session.commit()
        common_response = client.post(
            "/api/browser_fast_mode/bootstrap",
            json={"model": "gemini-2.5-flash"},
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )
        self.assertEqual(common_response.status_code, 200)
        self.assertEqual(common_response.get_json()["api_key"], "common-user-gemini-key")
        self.assertEqual(common_response.get_json()["key_source"], "gemini_common")

    def test_browser_fast_mode_never_exposes_admin_environment_key(self):
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            user.is_admin = True
            user.admin_api_key_mode = "env_fallback"
            user.gemini_api_key = None
            target.db.session.commit()

        client = self.authenticated_client()
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "shared-admin-secret"}):
            response = client.post(
                "/api/browser_fast_mode/bootstrap",
                json={"model": "gemini-2.5-flash"},
                headers={"X-CSRF-Token": "csrf-test-token"},
                base_url="https://localhost",
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["code"], "user_api_key_missing")
        self.assertNotIn("api_key", response.get_json())

    def test_browser_fast_mode_rolls_back_if_atomic_save_fails(self):
        client = self.authenticated_client()
        with mock.patch.object(target, "safe_db_commit", side_effect=RuntimeError("forced failure")):
            response = client.post(
                "/api/browser_fast_mode/save",
                json={
                    "message": "prompt",
                    "assistant_content": "answer",
                    "model": "gemini-2.5-flash",
                },
                headers={"X-CSRF-Token": "csrf-test-token"},
                base_url="https://localhost",
            )

        self.assertEqual(response.status_code, 500)
        with target.app.app_context():
            self.assertEqual(target.Thread.query.count(), 0)
            self.assertEqual(target.Message.query.count(), 0)

    def test_google_tts_auth_does_not_depend_on_openai_key(self):
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            user.google_api_key = target.encrypt_val("google-tts-key")
            user.openai_api_key = None

            resolved = target._resolve_chat_model_auth(user, "google-tts-studio")

            self.assertIsNone(resolved["error_code"])
            self.assertEqual(resolved["provider"], "google")
            self.assertEqual(resolved["api_key"], "google-tts-key")

    def test_chunk_upload_id_rejects_path_traversal(self):
        self.assertTrue(target._is_valid_chunk_upload_id("up_1752156000_deadbeef"))
        self.assertIsNone(target._chunk_session_dir(self.user_id, "../../app"))
        self.assertIsNone(target._chunk_session_dir(self.user_id, "up_1_deadbeef/../../app"))

    def test_image_upload_is_saved_byte_for_byte_without_server_reencoding(self):
        client = self.authenticated_client()
        original = b"not-decoded-by-server\x00\xffPNG-payload"
        response = client.post(
            "/upload",
            data={"file": (io.BytesIO(original), "browser-output.png")},
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        rel_path = response.get_json()["filename"]
        saved = Path(target.app.config["UPLOAD_FOLDER"]) / rel_path
        self.assertEqual(saved.suffix, ".png")
        self.assertEqual(saved.read_bytes(), original)

    def test_encrypted_image_upload_encrypts_original_browser_bytes(self):
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            user.enable_e2ee = True
            target.db.session.commit()

        client = self.authenticated_client()
        original = b"browser-webp-output\x00\x01\x02"
        response = client.post(
            "/upload",
            data={"file": (io.BytesIO(original), "browser-output.webp")},
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        rel_path = response.get_json()["filename"]
        encrypted = Path(target.app.config["UPLOAD_FOLDER"]) / f"{rel_path}.enc"
        self.assertTrue(encrypted.exists())
        self.assertEqual(target.decrypt_bytes(encrypted.read_bytes()), original)

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

    def test_rich_paste_sanitizer_preserves_safe_layout_styles(self):
        safe = target._sanitize_rich_paste_html(
            '<style>.x{position:fixed}</style>'
            '<script>alert(1)</script>'
            '<div class="x" onclick="alert(2)" '
            'style="display:grid;grid-template-columns:1fr 2fr;gap:12px;'
            'background:linear-gradient(#fff,#ddd);padding:10px;'
            'border:2px solid oklch(50% .2 250);position:fixed;'
            'background-image:url(http://127.0.0.1/private)">'
            '<p style="color:#dc2626;text-decoration:underline">Styled text</p>'
            '<img src="http://127.0.0.1/private" alt="blocked remote">'
            '</div>'
        )
        self.assertNotIn("<script", safe)
        self.assertNotIn("<style", safe)
        self.assertNotIn("onclick", safe)
        self.assertNotIn('class="x"', safe)
        self.assertNotIn("position:", safe)
        self.assertNotIn("url(", safe)
        self.assertNotIn("127.0.0.1", safe)
        self.assertIn("display: grid", safe)
        self.assertIn("grid-template-columns: 1fr 2fr", safe)
        self.assertIn("linear-gradient", safe)
        self.assertIn("padding: 10px", safe)
        self.assertIn("oklch", safe)
        self.assertIn("[Image: blocked remote]", safe)

    def test_rich_paste_sanitizer_extracts_dominant_article_from_full_page_copy(self):
        navigation = "<nav>" + "".join(f"<span>nav{i}</span>" for i in range(130)) + "</nav>"
        article_text = "Primary documentation content " * 120
        safe = target._sanitize_rich_paste_html(
            f"<main>{navigation}<article><h1>Documentation</h1><p>{article_text}</p></article></main>"
        )
        self.assertIn("Documentation", safe)
        self.assertIn("Primary documentation content", safe)
        self.assertNotIn("nav129", safe)

    def test_rich_paste_theme_preserves_dark_and_light_contrast(self):
        dark = target._resolve_rich_paste_theme(
            '<main style="background:#000;color:rgb(220,220,220)">'
            '<p style="color:rgb(220,220,220)">Dark document text</p></main>'
        )
        inferred_dark = target._resolve_rich_paste_theme(
            '<p style="color:rgb(245,245,245)">Light text with a missing copied page background</p>'
        )
        light = target._resolve_rich_paste_theme(
            '<main style="background:#fff;color:#111827"><p>Light document text</p></main>'
        )
        modern_dark = target._resolve_rich_paste_theme(
            '<main style="background:oklch(12% 0.02 260);color:oklch(92% 0.01 260)">'
            'Modern CSS color document</main>'
        )

        self.assertEqual(dark["mode"], "dark")
        self.assertEqual(dark["background"], "rgb(0, 0, 0)")
        self.assertEqual(inferred_dark["mode"], "dark")
        self.assertEqual(light["mode"], "light")
        self.assertEqual(modern_dark["mode"], "dark")
        dark_background = target._parse_rich_paste_css_color(dark["background"])
        dark_foreground = target._parse_rich_paste_css_color(dark["foreground"])
        self.assertGreaterEqual(target._rich_paste_color_contrast(dark_background, dark_foreground), 4.5)

    def test_rich_paste_print_layout_removes_screen_offsets_without_losing_theme(self):
        normalized = target._normalize_rich_paste_print_layout(
            '<main style="display:flex;width:1030px;padding:0 0 0 240px;'
            'background-color:#000;color:#eee;font-size:16px">'
            '<article style="width:750px"><p>Readable document</p></article></main>'
        )
        self.assertIn("background-color: #000", normalized)
        self.assertIn("color: #eee", normalized)
        self.assertIn("font-size: 16px", normalized)
        self.assertIn("display: block", normalized)
        self.assertNotIn("1030px", normalized)
        self.assertNotIn("750px", normalized)
        self.assertNotIn("240px", normalized)

    def test_rich_paste_weasyprint_uses_detected_page_theme(self):
        completed = mock.Mock(returncode=0, stdout=b"%PDF-theme-test", stderr=b"")
        with mock.patch.object(target.subprocess, "run", return_value=completed) as run_mock:
            pdf = target._build_rich_paste_pdf_bytes_weasyprint(
                "Dark export",
                '<main style="background:#000;color:#eee"><p>Visible text</p></main>',
            )
        self.assertEqual(pdf, b"%PDF-theme-test")
        document_html = run_mock.call_args.kwargs["input"].decode("utf-8")
        self.assertIn("color-scheme: dark", document_html)
        self.assertIn("background: rgb(0, 0, 0)", document_html)
        self.assertIn("color: rgb(238, 238, 238)", document_html)

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

    def test_csrf_token_endpoint_returns_current_session_token_without_cache(self):
        client = self.authenticated_client()

        response = client.get("/api/csrf_token", base_url="https://localhost")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["csrf_token"], "csrf-test-token")
        self.assertIn("no-store", response.headers.get("Cache-Control", ""))
        self.assertEqual(response.headers.get("Pragma"), "no-cache")

    def test_client_refreshes_stale_csrf_token_and_retries_once(self):
        assets = list((APP_ROOT / "static/js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(assets), 1)
        source = assets[0].read_text(encoding="utf-8")

        self.assertIn("let csrfToken =", source)
        self.assertIn("const refreshCsrfToken = async () =>", source)
        self.assertIn("fetch('/api/csrf_token'", source)
        self.assertIn("response.status === 403 || response.status === 404", source)
        self.assertIn("headers['X-CSRF-Token'] = csrfToken", source)
        self.assertEqual(source.count("const refreshed = await refreshCsrfToken();"), 1)

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

    def test_message_payload_columns_compile_as_mysql_longtext(self):
        ddl = str(CreateTable(target.Message.__table__).compile(dialect=mysql.dialect()))

        for column_name in ("content", "thought_data", "quote_text", "thought_signature"):
            with self.subTest(column_name=column_name):
                self.assertRegex(ddl, rf"\b{column_name} LONGTEXT\b")

    def test_large_encrypted_message_round_trip_exceeds_legacy_text_limit(self):
        plaintext = "長文回答" * 25_000
        encrypted = target.encrypt_val(plaintext)
        self.assertGreater(len(encrypted.encode("utf-8")), 65_535)

        with target.app.app_context():
            thread = target.Thread(
                user_id=self.user_id,
                public_id=target.generate_thread_public_id(),
            )
            target.db.session.add(thread)
            target.db.session.flush()
            message = target.Message(
                thread_id=thread.id,
                role="assistant",
                content=encrypted,
                is_encrypted=True,
            )
            target.db.session.add(message)
            target.db.session.commit()

            saved = target.db.session.get(target.Message, message.id)
            self.assertEqual(target.decrypt_val(saved.content), plaintext)

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
        self.assertIn("document.addEventListener('pointerout'", script)
        self.assertIn("liquid-glass-pressed", script)
        self.assertIn("liquid-glass-clear", script)
        self.assertIn("Liquid Glass V4: adaptive background-aware material", css)
        self.assertIn("--glass-caustic-opacity", css)
        self.assertIn("prefers-reduced-transparency: reduce", css)
        self.assertIn("liquid-glass-no-background", script)
        self.assertIn("liquid-glass-scrolling", script)
        self.assertIn("body.liquid-glass-mode .liquid-glass-no-background", css)
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
        # Stream/history errors are rendered via buildChatErrorBubbleHtml (escapeHtml).
        self.assertIn("function buildChatErrorBubbleHtml(errorText)", script)
        self.assertIn("Error: ${escapeHtml(msg)}", script)
        self.assertIn("buildChatErrorBubbleHtml(j.content)", script)
        self.assertIn("FORBID_TAGS: ['iframe', 'object', 'embed']", template)
        self.assertIn("filename='vendor/dompurify-3.4.11.min.js'", template)
        self.assertNotIn("cdnjs.cloudflare.com/ajax/libs/dompurify", template)
        self.assertIn('sandbox="allow-scripts allow-forms allow-modals allow-popups"', template)
        self.assertIn("openSandboxedHtmlTab(safe)", script)
        self.assertIn("frame.setAttribute('sandbox', '')", script)

    def test_critical_markdown_libraries_are_self_hosted_with_matching_sri(self):
        root = os.path.dirname(os.path.dirname(__file__))
        expected = {
            "marked-4.3.0.min.js": "QsSpx6a0USazT7nK7w8qXDgpSAPhFsb2XtpoLFQ5+X2yFN6hvCKnwEzN8M5FWaJb",
            "dompurify-3.4.11.min.js": "o44XUELLEnv/iSlA1NWxBweqbD4TSR0qgq2VzVsxtkHS989JJjGKSE9vkfo5MN4K",
            "html2canvas-pro-2.3.2.min.js": "M073WWOJRzkLDop7Z6uvyVa8j05SMm0z5YU8Iv8BoA5wa/jzs0M9s87NQ5aQvVJE",
            "jspdf-2.5.1.umd.min.js": "JcnsjUPPylna1s1fvi1u12X5qjY5OL56iySh75FdtrwhO/SWXgMjoVqcKyIIWOLk",
        }
        for filename, sri_hash in expected.items():
            with self.subTest(filename=filename):
                asset_path = os.path.join(root, "static", "vendor", filename)
                with open(asset_path, "rb") as asset_file:
                    actual = base64.b64encode(hashlib.sha384(asset_file.read()).digest()).decode("ascii")
                self.assertEqual(actual, sri_hash)

        for template_name in ("chat.html", "landing.html", "login.html", "changelog.html", "help.html"):
            with self.subTest(template=template_name):
                with open(os.path.join(root, "templates", template_name), encoding="utf-8") as template_file:
                    template = template_file.read()
                self.assertIn("filename='vendor/marked-4.3.0.min.js'", template)
                self.assertIn("filename='vendor/dompurify-3.4.11.min.js'", template)
                self.assertNotIn("cdn.jsdelivr.net/npm/marked@4.3.0", template)
                self.assertNotIn("cdnjs.cloudflare.com/ajax/libs/dompurify", template)

        chat_core_assets = glob.glob(os.path.join(root, "static", "js", "chat_core.v4.8.*.js"))
        self.assertEqual(len(chat_core_assets), 1)
        with open(chat_core_assets[0], encoding="utf-8") as script_file:
            script = script_file.read()
        self.assertIn("/static/vendor/html2canvas-pro-2.3.2.min.js", script)
        self.assertIn("/static/vendor/jspdf-2.5.1.umd.min.js", script)
        self.assertNotIn("cdnjs.cloudflare.com/ajax/libs/html2canvas", script)
        self.assertNotIn("cdnjs.cloudflare.com/ajax/libs/jspdf", script)

    def test_help_page_is_public_and_documents_fast_mode(self):
        # ヘルプページはログイン不要で閲覧できる必要がある（匿名アクセスで200）。
        client = target.app.test_client()
        response = client.get("/help", base_url="https://localhost")
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("高速モード", body)
        self.assertIn("マルチモデル対応", body)
        self.assertIn("ヘルプセンター", body)

        # ランディングページにもヘルプへのリンクが存在する
        landing = client.get("/", base_url="https://localhost")
        self.assertEqual(landing.status_code, 200)
        self.assertIn('href="/help"', landing.get_data(as_text=True))

    def test_delete_user_account_removes_latency_metric_rows(self):
        # チャット遅延診断テーブル（first_token_latency_metric / chat_latency_trace）は
        # user.id への外部キー制約（RESTRICT）を持つ。これらの行を削除しないと
        # db.session.delete(user) が IntegrityError で失敗しアカウント削除ができないため、
        # _delete_user_account_immediately が確実に削除することを検証する。
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            thread = target.Thread(user_id=user.id, public_id="del-thread-1", title="t")
            target.db.session.add(thread)
            target.db.session.flush()
            target.db.session.add(target.Message(thread_id=thread.id, role="user", content="hello"))
            target.db.session.add(
                target.FirstTokenLatencyMetric(
                    user_id=user.id, thread_public_id="del-thread-1",
                    latency_seconds=1.0, latency_ms=1000,
                )
            )
            target.db.session.add(
                target.ChatLatencyTrace(user_id=user.id, thread_public_id="del-thread-1", job_id="job_del_1")
            )
            target.db.session.commit()
            uid = user.id
            target._delete_user_account_immediately(user)
            self.assertIsNone(target.db.session.get(target.User, uid))
            self.assertEqual(target.FirstTokenLatencyMetric.query.filter_by(user_id=uid).count(), 0)
            self.assertEqual(target.ChatLatencyTrace.query.filter_by(user_id=uid).count(), 0)
            self.assertEqual(target.Thread.query.filter_by(user_id=uid).count(), 0)
            self.assertEqual(target.Message.query.count(), 0)

    def test_delete_account_endpoint_removes_user(self):
        # 設定画面からの自己削除（POST /api/account/delete）が成功し、
        # ユーザーが DB から消えることを検証する。
        client = self.authenticated_client()
        response = client.post(
            "/api/account/delete",
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "ok")
        with target.app.app_context():
            self.assertIsNone(target.db.session.get(target.User, self.user_id))

    def _seed_admin_and_target(self):
        """Seed an admin with own encrypted chat, plus another user's encrypted chat."""
        with target.app.app_context():
            admin = target.User(
                username="enc-admin",
                is_setup_completed=True,
                is_admin=True,
                enable_e2ee=True,
            )
            admin.set_password("admin-password")
            target.db.session.add(admin)
            target.db.session.commit()
            self.admin_id = admin.id

            # Admin's own encrypted thread (the only target of admin encryption APIs).
            admin_thread = target.Thread(
                user_id=admin.id,
                public_id="enc-admin-thread",
                title="admin own chat",
            )
            target.db.session.add(admin_thread)
            target.db.session.flush()
            target.db.session.add(target.Message(
                thread_id=admin_thread.id, role="user",
                content=target.encrypt_val("secret question"), is_encrypted=True,
            ))
            target.db.session.add(target.Message(
                thread_id=admin_thread.id, role="assistant",
                content=target.encrypt_val("secret answer"),
                thought_data=target.encrypt_val("hidden reasoning"), is_encrypted=True,
            ))

            # Another user's encrypted thread must never be listed/toggled by admin APIs.
            user = target.db.session.get(target.User, self.user_id)
            user.enable_e2ee = True
            other_thread = target.Thread(
                user_id=user.id,
                public_id="enc-other-thread",
                title="other user chat",
            )
            target.db.session.add(other_thread)
            target.db.session.flush()
            target.db.session.add(target.Message(
                thread_id=other_thread.id, role="user",
                content=target.encrypt_val("other secret"), is_encrypted=True,
            ))
            target.db.session.commit()

    def _client_for_user(self, user_id):
        client = target.app.test_client()
        with client.session_transaction() as sess:
            sess["_user_id"] = str(user_id)
            sess["_fresh"] = True
            sess["csrf_token"] = "csrf-test-token"
        return client

    def test_admin_thread_list_requires_admin(self):
        self._seed_admin_and_target()
        plain_client = self.authenticated_client()
        response = plain_client.get("/api/admin/threads", base_url="https://localhost")
        self.assertEqual(response.status_code, 403)

    def test_admin_thread_list_and_per_thread_decrypt_encrypt_roundtrip(self):
        self._seed_admin_and_target()
        admin_client = self._client_for_user(self.admin_id)

        # No username param: lists only the admin's own threads.
        listed = admin_client.get("/api/admin/threads", base_url="https://localhost")
        self.assertEqual(listed.status_code, 200, listed.get_data(as_text=True))
        payload = listed.get_json()
        self.assertEqual(payload["user"]["username"], "enc-admin")
        self.assertTrue(payload["user"]["enable_e2ee"])
        self.assertEqual(len(payload["threads"]), 1)
        thread = payload["threads"][0]
        self.assertEqual(thread["thread_id"], "enc-admin-thread")
        self.assertEqual(thread["encrypted_count"], 2)
        self.assertTrue(thread["encrypted"])
        # Other users' threads must not appear.
        self.assertFalse(any(t["thread_id"] == "enc-other-thread" for t in payload["threads"]))

        # Admin decrypts own thread.
        decrypted = admin_client.post(
            "/api/admin/threads/enc-admin-thread/encryption",
            json={"enable": False},
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )
        self.assertEqual(decrypted.status_code, 200, decrypted.get_data(as_text=True))
        decrypted_payload = decrypted.get_json()
        self.assertEqual(decrypted_payload["changed"], 2)
        self.assertFalse(decrypted_payload["enable"])

        with target.app.app_context():
            thread_row = target.db.session.query(target.Thread).filter_by(public_id="enc-admin-thread").first()
            msgs = target.db.session.query(target.Message).filter_by(thread_id=thread_row.id).all()
            self.assertEqual(len(msgs), 2)
            for m in msgs:
                self.assertFalse(m.is_encrypted)
                self.assertNotEqual(m.content, target.encrypt_val(m.content))
            assistant = [m for m in msgs if m.role == "assistant"][0]
            self.assertEqual(assistant.content, "secret answer")
            self.assertEqual(assistant.thought_data, "hidden reasoning")

        # Admin re-encrypts the same thread.
        reencrypted = admin_client.post(
            "/api/admin/threads/enc-admin-thread/encryption",
            json={"enable": True},
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )
        self.assertEqual(reencrypted.status_code, 200, reencrypted.get_data(as_text=True))
        self.assertEqual(reencrypted.get_json()["changed"], 2)

        with target.app.app_context():
            thread_row = target.db.session.query(target.Thread).filter_by(public_id="enc-admin-thread").first()
            msgs = target.db.session.query(target.Message).filter_by(thread_id=thread_row.id).all()
            for m in msgs:
                self.assertTrue(m.is_encrypted)
                self.assertNotEqual(m.content, target.decrypt_val(m.content))
            assistant = [m for m in msgs if m.role == "assistant"][0]
            self.assertEqual(target.decrypt_val(assistant.content), "secret answer")
            self.assertEqual(target.decrypt_val(assistant.thought_data), "hidden reasoning")

    def test_admin_thread_encryption_rejects_other_users_thread(self):
        self._seed_admin_and_target()
        admin_client = self._client_for_user(self.admin_id)
        response = admin_client.post(
            "/api/admin/threads/enc-other-thread/encryption",
            json={"enable": False},
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )
        self.assertEqual(response.status_code, 404)
        # Other user's messages must remain encrypted and untouched.
        with target.app.app_context():
            thread_row = target.db.session.query(target.Thread).filter_by(public_id="enc-other-thread").first()
            msgs = target.db.session.query(target.Message).filter_by(thread_id=thread_row.id).all()
            self.assertEqual(len(msgs), 1)
            self.assertTrue(msgs[0].is_encrypted)
            self.assertEqual(target.decrypt_val(msgs[0].content), "other secret")

    def test_admin_thread_encryption_rejects_unknown_thread(self):
        self._seed_admin_and_target()
        admin_client = self._client_for_user(self.admin_id)
        response = admin_client.post(
            "/api/admin/threads/does-not-exist/encryption",
            json={"enable": True},
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )
        self.assertEqual(response.status_code, 404)

    def test_admin_thread_encryption_requires_admin(self):
        self._seed_admin_and_target()
        plain_client = self.authenticated_client()
        response = plain_client.post(
            "/api/admin/threads/enc-admin-thread/encryption",
            json={"enable": False},
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )
        self.assertEqual(response.status_code, 403)

    def test_admin_encryption_ui_has_bubble_toggle_and_open_chat(self):
        """Bubble lock modal and settings list expose admin decrypt/open controls."""
        from pathlib import Path
        root = Path(target.app.root_path)
        js_assets = sorted((root / "static" / "js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(js_assets), 1)
        js = js_assets[0].read_text(encoding="utf-8")
        self.assertIn("toggleThreadEncryptionFromModal", js)
        self.assertIn("window.__setAdminThreadEncryption", js)
        self.assertIn("admin-enc-open", js)
        self.assertIn("encryption-status-admin-toggle", js)
        self.assertIn("このチャットを復号化", js)
        self.assertIn("このチャットを再暗号化", js)

        html = (root / "templates" / "chat.html").read_text(encoding="utf-8")
        self.assertIn('id="encryption-status-admin-actions"', html)
        self.assertIn('id="encryption-status-admin-toggle"', html)
        self.assertIn('id="admin-enc-card"', html)
        self.assertNotIn("admin-enc-username", html)

    def test_frontend_escapes_gem_suggestions_and_branch_tree(self):
        # V4.8.799: Gem候補ドロップダウンとブランチツリー表示は、保存値を
        # escapeHtml に通してから innerHTML へ埋め込む必要がある。
        root = os.path.dirname(os.path.dirname(__file__))
        version = target.app.config["SYSTEM_VERSION"].lower()
        script_path = os.path.join(root, "static", "js", f"chat_core.{version}.js")
        with open(script_path, encoding="utf-8") as script_file:
            script = script_file.read()

        suggestions = script[script.index("function showGemSuggestions"):]
        self.assertIn("${escapeHtml(gem.name)}", suggestions)
        self.assertIn("${escapeHtml(gem.description)}", suggestions)

        branch = script[script.index("function renderBranchTreeVisualization"):]
        self.assertIn("${escapeHtml(name)}", branch)
        self.assertIn("${escapeHtml(node.model || '-')}", branch)

    def test_google_login_never_links_accounts_by_username(self):
        # V4.8.799: Googleログイン時のアカウント照合は google_id / google_email のみとし、
        # ユーザー名で既存アカウントへ紐付けしない。これにより、他ユーザーのメールアドレスを
        # ユーザー名として先に登録してGoogleログインを乗っ取る経路を遮断する。
        root = os.path.dirname(os.path.dirname(__file__))
        with open(os.path.join(root, "app.py"), encoding="utf-8") as src_file:
            source = src_file.read()
        self.assertNotIn("User.username == email", source)
        self.assertIn("def _resolve_or_create_google_user", source)
        self.assertIn("filter_by(google_email=email)", source)
        self.assertIn("if '@' in username", source)

    def test_feedback_rejects_oversized_message(self):
        client = self.authenticated_client()
        response = client.post(
            "/api/feedback",
            json={"title": "t", "message": "x" * 100_001},
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "Message is too long")

    def test_easy_login_uses_high_entropy_token(self):
        client = self.authenticated_client()
        response = client.post(
            "/api/easy_login",
            json={"minutes": 5},
            headers={"X-CSRF-Token": "csrf-test-token"},
            base_url="https://localhost",
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertGreaterEqual(len(payload["temp_password"]), 20)

    def test_in_site_system_logs_endpoint_is_removed(self):
        client = self.authenticated_client()
        with target.app.app_context():
            user = target.db.session.get(target.User, self.user_id)
            user.is_admin = True
            target.db.session.commit()
        response = client.get("/api/debug/log", base_url="https://localhost")
        self.assertEqual(response.status_code, 404)
        self.assertIsNone(target.app.view_functions.get("debug_log"))


if __name__ == "__main__":
    unittest.main()
