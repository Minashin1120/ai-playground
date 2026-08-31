"""Uploaded files from the user's device must accept every file type the
create_file tool can produce (text / markdown / code / PDF / Word / Excel).

V4.8.885 added the File tool: the model can create .md / .csv / .json / .xlsx /
.py / .js / .html ... files on the server.  But the upload endpoints still
rejected every extension outside the old media allowlist, so a user who
downloaded such a file could not upload it again from their device.  These
tests guard the allowlist + text-detection changes that close that gap.
"""

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("FLASK_SECRET_KEY", "upload-file-types-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-upload-file-types-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target


class UploadFileTypesRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        target.app.config.update(TESTING=True, MAINTENANCE_MODE=False, TRUSTED_HOSTS=["localhost"])

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        target.app.config["UPLOAD_FOLDER"] = self.temp_dir.name
        self.rate_limit_patcher = mock.patch.object(target, "rate_limit", return_value=True)
        self.rate_limit_patcher.start()
        self.addCleanup(self.rate_limit_patcher.stop)
        self.turnstile_patcher = mock.patch.object(target, "_bot_turnstile_active", return_value=False)
        self.turnstile_patcher.start()
        self.addCleanup(self.turnstile_patcher.stop)
        with target.app.app_context():
            target.db.session.remove()
            target.db.engine.dispose()
            target.db.drop_all()
            target.db.create_all()
            user = target.User(
                username="uploader",
                is_setup_completed=True,
                enable_e2ee=False,
            )
            user.set_password("test-password")
            target.db.session.add(user)
            target.db.session.flush()
            self.user_id = user.id
            target.db.session.commit()

    def tearDown(self):
        with target.app.app_context():
            target.db.session.remove()
            target.db.engine.dispose()
        self.temp_dir.cleanup()

    def client_for(self):
        client = target.app.test_client()
        with client.session_transaction() as session:
            session["_user_id"] = str(self.user_id)
            session["_fresh"] = True
            session["csrf_token"] = "csrf-test-token"
        return client

    def test_upload_allowlist_covers_every_create_file_extension(self):
        """Every extension the File tool can create must be uploadable."""
        createable = (
            set(target._CREATE_FILE_TEXT_EXTS)
            | set(target._CREATE_FILE_CODE_EXTS)
            | set(target._CREATE_FILE_DOC_EXTS)
        )
        for ext in sorted(createable):
            with self.subTest(ext=ext):
                self.assertIn(
                    ext,
                    target._UPLOAD_ALLOWED_EXTENSIONS,
                    f"create_file extension {ext} is not uploadable",
                )

    def test_upload_accepts_created_document_types(self):
        """A user can upload .md / .json / .csv / .xlsx / .py / .html files."""
        samples = {
            "memo.md": b"# Title\n\nBody\n",
            "data.json": b'{"a": 1}\n',
            "data.csv": b"name,value\napple,3\n",
            "sheet.xlsx": b"\x50\x4b\x03\x04",  # ZIP container prefix
            "script.py": b"def f():\n    return 1\n",
            "page.html": b"<html><body>hi</body></html>\n",
        }
        client = self.client_for()
        for fname, content in samples.items():
            with self.subTest(filename=fname):
                resp = client.post(
                    "/upload",
                    data={"file": (io.BytesIO(content), fname)},
                    content_type="multipart/form-data",
                    headers={"X-CSRF-Token": "csrf-test-token"},
                )
                self.assertEqual(resp.status_code, 200, resp.get_data(as_text=True))
                payload = resp.get_json()
                self.assertTrue(payload.get("filename"))
                saved = os.path.join(self.temp_dir.name, payload["filename"])
                self.assertTrue(os.path.exists(saved))

    def test_upload_init_accepts_created_document_types(self):
        """The chunked-upload init route must accept the same extensions."""
        client = self.client_for()
        for fname in ("memo.md", "data.json", "sheet.xlsx", "script.py"):
            with self.subTest(filename=fname):
                resp = client.post(
                    "/upload/init",
                    json={"filename": fname, "size": 10},
                    headers={"X-CSRF-Token": "csrf-test-token"},
                )
                self.assertEqual(resp.status_code, 200, resp.get_data(as_text=True))
                self.assertIn("upload_id", resp.get_json())

    def test_text_like_detection_includes_create_file_types(self):
        """Markdown / JSON / YAML / CSV / code uploads count as readable text."""
        for fname in ("memo.md", "data.json", "conf.yaml", "data.csv", "app.py", "page.html", "notes.txt"):
            with self.subTest(filename=fname):
                ext = os.path.splitext(fname)[1].lower()
                self.assertIn(ext, target._TEXT_LIKE_UPLOAD_EXTS)

    def test_estimate_tokens_marks_new_text_files_as_countable(self):
        rel_path = f"{self.user_id}/memo.md"
        ud = os.path.join(self.temp_dir.name, str(self.user_id))
        os.makedirs(ud, mode=0o700)
        with open(os.path.join(ud, "memo.md"), "wb") as handle:
            handle.write(b"# Title\n\nBody text\n")
        result = target._estimate_attachment_prompt_tokens(rel_path, "gpt-4o")
        self.assertTrue(result.get("countable"))

    def test_files_route_forces_download_for_html_svg(self):
        """Uploadable HTML/SVG files must not render inline when served."""
        ud = os.path.join(self.temp_dir.name, str(self.user_id))
        os.makedirs(ud, mode=0o700)
        rel = f"{self.user_id}/page.html"
        with open(os.path.join(ud, "page.html"), "wb") as handle:
            handle.write(b"<html><body>hi</body></html>\n")
        client = self.client_for()
        resp = client.get(f"/files/{rel}")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.mimetype, "application/octet-stream")
        self.assertIn("attachment", resp.headers.get("Content-Disposition", ""))


if __name__ == "__main__":
    unittest.main()
