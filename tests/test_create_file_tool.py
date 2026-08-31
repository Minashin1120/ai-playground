import os
from pathlib import Path
import unittest
from unittest.mock import patch

os.environ.setdefault("FLASK_SECRET_KEY", "create-file-tool-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-create-file-tool-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target


APP_ROOT = Path(__file__).resolve().parents[1]


class CreateFileToolTests(unittest.TestCase):
    def test_text_file_bytes_preserve_newlines(self):
        filename = "memo.txt"
        data = target._create_file_generated_bytes(
            filename, "text", "line1\nline2\n\nline4", user_id=1
        )
        self.assertIsInstance(data, bytes)
        self.assertEqual(data.decode("utf-8"), "line1\nline2\n\nline4")

    def test_code_file_bytes_preserve_newlines(self):
        filename = "script.py"
        data = target._create_file_generated_bytes(
            filename, "code", "def f():\n    return 1\n", user_id=1
        )
        self.assertEqual(data.decode("utf-8"), "def f():\n    return 1\n")

    def test_csv_tsv_json_are_text_bytes(self):
        for fname, fmt in [("a.csv", "csv"), ("a.tsv", "tsv"), ("a.json", "json")]:
            data = target._create_file_generated_bytes(fname, fmt, "a\nb\n", user_id=1)
            self.assertEqual(data.decode("utf-8"), "a\nb\n")

    def test_xlsx_bytes_are_valid(self):
        data = target._create_file_generated_bytes("data.xlsx", "xlsx", "name\tvalue\napple\t3\n", user_id=1)
        self.assertIsInstance(data, bytes)
        self.assertGreater(len(data), 500)
        self.assertTrue(data.startswith(b"PK"))  # ZIP container

    def test_docx_bytes_are_valid(self):
        data = target._create_file_generated_bytes("doc.docx", "docx", "# Title\n\nBody text", user_id=1)
        self.assertIsInstance(data, bytes)
        self.assertGreater(len(data), 500)
        self.assertTrue(data.startswith(b"PK"))

    def test_pdf_bytes_start_with_pdf_magic(self):
        with patch("subprocess.run") as run:
            run.return_value = unittest.mock.Mock(
                returncode=0, stdout=b"%PDF-1.7 test", stderr=b""
            )
            data = target._create_file_generated_bytes("doc.pdf", "pdf", "# Title\n\nBody", user_id=1)
        self.assertTrue(data.startswith(b"%PDF"))

    def test_extension_inference(self):
        self.assertEqual(target._infer_create_file_format("report.txt"), "text")
        self.assertEqual(target._infer_create_file_format("script.py"), "code")
        self.assertEqual(target._infer_create_file_format("doc.pdf"), "pdf")
        self.assertEqual(target._infer_create_file_format("doc.docx"), "docx")
        self.assertEqual(target._infer_create_file_format("data.xlsx"), "xlsx")
        self.assertEqual(target._infer_create_file_format("memo.md"), "markdown")

    def test_invalid_extension_rejected(self):
        result = target._execute_create_file_tool(
            1, {"filename": "evil.sh.exe", "content": "x"}, encrypt=False
        )
        self.assertFalse(result.get("ok"))

    def test_missing_filename_rejected(self):
        result = target._execute_create_file_tool(1, {"content": "x"}, encrypt=False)
        self.assertFalse(result.get("ok"))

    def test_missing_content_rejected(self):
        result = target._execute_create_file_tool(1, {"filename": "a.txt"}, encrypt=False)
        self.assertFalse(result.get("ok"))

    def test_execute_saves_to_library_and_returns_url(self):
        with patch("app._save_user_generated_bytes_verified") as save_mock:
            save_mock.return_value = ("memo_123.txt", "/files/1/memo_123.txt")
            result = target._execute_create_file_tool(
                1, {"filename": "memo.txt", "content": "hello\nworld"}, encrypt=False
            )
        self.assertTrue(result.get("ok"))
        self.assertEqual(result["display_name"], "memo.txt")
        self.assertEqual(result["url"], "/files/1/memo_123.txt")
        save_mock.assert_called_once()

    def test_result_text_contains_url(self):
        result = {"ok": True, "filename": "memo_1.txt", "display_name": "memo.txt", "url": "/files/1/memo_1.txt", "size": 5}
        text = target._create_file_tool_result_text(result)
        self.assertIn("/files/1/memo_1.txt", text)
        self.assertIn("memo.txt", text)

    def test_library_image_ref_is_inlined_to_data_uri(self):
        md = "![alt](/files/7/photo.png)"
        png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"  # minimal PNG header
        with patch("app._get_file_disk_info", return_value={"exists": True, "size": len(png)}), \
             patch("app._load_user_file_bytes", return_value=png):
            result, warnings = target._inline_library_images_for_create_file(md, 7)
        self.assertEqual(warnings, [])
        self.assertIn("data:image/png;base64,", result)
        self.assertNotIn("(photo.png)", result)

    def test_missing_library_image_gives_warning_and_keeps_ref(self):
        md = "![alt](/files/7/missing.png)"
        with patch("app._get_file_disk_info", return_value={"exists": False}):
            result, warnings = target._inline_library_images_for_create_file(md, 7)
        self.assertEqual(len(warnings), 1)
        self.assertIn("missing.png", result)

    def test_data_uri_image_left_untouched(self):
        md = "![alt](data:image/png;base64,iVBORw0KGgo=)"
        result, warnings = target._inline_library_images_for_create_file(md, 7)
        self.assertEqual(warnings, [])
        self.assertIn("data:image/png;base64,iVBORw0KGgo=", result)

    def test_tool_schema_shape(self):
        schema = target._build_create_file_tool_schema()
        self.assertEqual(schema["function"]["name"], "create_file")
        props = schema["function"]["parameters"]["properties"]
        self.assertIn("filename", props)
        self.assertIn("content", props)
        self.assertIn("format", props)
        self.assertIn("required", schema["function"]["parameters"])
        self.assertIn("filename", schema["function"]["parameters"]["required"])

    def test_deepseek_branch_registers_create_file_tool(self):
        app_source = (APP_ROOT / "app.py").read_text(encoding="utf-8")
        deepseek_branch = app_source[
            app_source.index('log_force("Routing: DeepSeek V4 Branch (Chat Completions)")'):
            app_source.index('elif is_kimi:', app_source.index('log_force("Routing: DeepSeek V4 Branch (Chat Completions)")'))
        ]
        self.assertIn('options.get("enable_file_creation")', deepseek_branch)
        self.assertIn("_build_create_file_tool_schema()", deepseek_branch)
        self.assertIn('call_name == "create_file"', deepseek_branch)
        self.assertIn("_execute_create_file_tool(", deepseek_branch)
        self.assertIn("_create_file_tool_result_text(create_result)", deepseek_branch)

    def test_responses_api_branch_registers_create_file_tool(self):
        app_source = (APP_ROOT / "app.py").read_text(encoding="utf-8")
        responses_branch = app_source[
            app_source.index('log_force("Routing: Responses API Branch")'):
            app_source.index('elif is_deepseek:', app_source.index('log_force("Routing: Responses API Branch")'))
        ]
        self.assertIn("options.get('enable_file_creation')", responses_branch)
        self.assertIn("_build_create_file_tool_schema()", responses_branch)
        self.assertIn('call_name in ("execute_python", "create_file")', responses_branch)

    def test_gemini_branch_registers_create_file_callable(self):
        app_source = (APP_ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("_gemini_create_file_tool.__name__ = \"create_file\"", app_source)
        self.assertIn("conf['tools'].append(_gemini_create_file_tool)", app_source)
        self.assertIn("options.get('enable_file_creation')", app_source)

    def test_frontend_wires_enable_file_creation(self):
        js_assets = list((APP_ROOT / "static/js").glob("chat_core.v4.8.*.js"))
        self.assertEqual(len(js_assets), 1)
        js_source = js_assets[0].read_text(encoding="utf-8")
        self.assertIn("enable_file_creation", js_source)
        self.assertIn("enable-file-creation", js_source)

    def test_options_dict_contains_enable_file_creation(self):
        app_source = (APP_ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("'enable_file_creation': data.get('enable_file_creation')", app_source)
        self.assertIn("current_user.last_enable_file_creation = bool(data.get('enable_file_creation'))", app_source)

    def test_db_columns_and_migrations(self):
        app_source = (APP_ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("default_enable_file_creation = db.Column(db.Boolean, default=True)", app_source)
        self.assertIn("last_enable_file_creation = db.Column(db.Boolean, default=True)", app_source)
        self.assertIn("ALTER TABLE user ADD COLUMN default_enable_file_creation BOOLEAN DEFAULT 1", app_source)
        self.assertIn("ALTER TABLE user ADD COLUMN last_enable_file_creation BOOLEAN DEFAULT 1", app_source)


if __name__ == "__main__":
    unittest.main()
