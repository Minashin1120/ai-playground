import os
from pathlib import Path
import unittest
from unittest.mock import patch

os.environ.setdefault("FLASK_SECRET_KEY", "edit-file-tool-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-edit-file-tool-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target


APP_ROOT = Path(__file__).resolve().parents[1]


class XlsxTextExtractionTests(unittest.TestCase):
    def _make_xlsx(self, content_tsv, sheet_name="Sheet1"):
        return target._build_created_xlsx_bytes(content_tsv)

    def test_extract_single_sheet_xlsx_to_tsv(self):
        data = self._make_xlsx("name\tvalue\napple\t3\nbanana\t5\n")
        text = target._extract_xlsx_as_tsv(data)
        self.assertIsNotNone(text)
        self.assertIn("name\tvalue", text)
        self.assertIn("apple\t3", text)
        self.assertIn("banana\t5", text)

    def test_extract_multi_sheet_xlsx_emits_sheet_markers(self):
        data = self._make_xlsx(
            "a\tb\n1\t2\n# Sheet: Sheet2\nx\ty\n3\t4\n"
        )
        text = target._extract_xlsx_as_tsv(data)
        self.assertIsNotNone(text)
        self.assertIn("# Sheet: Sheet2", text)
        self.assertIn("3\t4", text)

    def test_extract_round_trips_through_builder(self):
        data = self._make_xlsx(
            "a\tb\n1\t2\n# Sheet: Sheet2\nx\ty\n3\t4\n"
        )
        text = target._extract_xlsx_as_tsv(data)
        rebuilt = target._build_created_xlsx_bytes(text)
        self.assertTrue(rebuilt.startswith(b"PK"))
        from openpyxl import load_workbook
        from io import BytesIO
        wb = load_workbook(BytesIO(rebuilt), read_only=True, data_only=True)
        self.assertEqual(wb.sheetnames, ["Sheet1", "Sheet2"])

    def test_extract_invalid_bytes_returns_none(self):
        self.assertIsNone(target._extract_xlsx_as_tsv(b"not an xlsx file"))
        self.assertIsNone(target._extract_xlsx_as_tsv(None))

    def test_multiline_cells_escape_and_round_trip(self):
        data = self._make_xlsx("a\tb\none\ttwo\\nlines\nthree\ttab\\tsep\n")
        text = target._extract_xlsx_as_tsv(data)
        self.assertIsNotNone(text)
        self.assertIn("two\\nlines", text)
        self.assertIn("tab\\tsep", text)
        rebuilt = target._build_created_xlsx_bytes(text)
        from openpyxl import load_workbook
        from io import BytesIO
        wb = load_workbook(BytesIO(rebuilt), read_only=True, data_only=True)
        ws = wb.worksheets[0]
        rows = list(ws.iter_rows(values_only=True))
        self.assertEqual(rows[1][1], "two\nlines")
        self.assertEqual(rows[2][1], "tab\tsep")

    def test_build_xlsx_unescapes_cells(self):
        data = target._build_created_xlsx_bytes("a\tb\nx\ty\\\\z\\nq\n")
        from openpyxl import load_workbook
        from io import BytesIO
        wb = load_workbook(BytesIO(data), read_only=True, data_only=True)
        ws = wb.worksheets[0]
        rows = list(ws.iter_rows(values_only=True))
        self.assertEqual(rows[1][1], "y\\z\nq")

    def test_build_xlsx_handles_sheet_marker(self):
        data = target._build_created_xlsx_bytes(
            "a\tb\n1\t2\n# Sheet: Second\nc\td\n"
        )
        from openpyxl import load_workbook
        from io import BytesIO
        wb = load_workbook(BytesIO(data), read_only=True, data_only=True)
        self.assertEqual(wb.sheetnames, ["Sheet1", "Second"])


class EditFileToolTests(unittest.TestCase):
    def test_resolve_matches_send_name_path_and_basename(self):
        loaded = [{
            "send_name": "予定表.xlsx",
            "name": "1/1234_abcd.xlsx",
            "path": "1/1234_abcd.xlsx",
        }]
        self.assertEqual(
            target._resolve_attached_file_ref("予定表.xlsx", loaded), loaded[0]
        )
        self.assertEqual(
            target._resolve_attached_file_ref("1/1234_abcd.xlsx", loaded), loaded[0]
        )
        self.assertEqual(
            target._resolve_attached_file_ref("1234_abcd.xlsx", loaded), loaded[0]
        )
        self.assertEqual(
            target._resolve_attached_file_ref("予定表.XLSX", loaded), loaded[0]
        )

    def test_resolve_returns_none_when_no_match(self):
        loaded = [{"send_name": "予定表.xlsx", "name": "1/1234_abcd.xlsx"}]
        self.assertIsNone(target._resolve_attached_file_ref("missing.csv", loaded))
        self.assertIsNone(target._resolve_attached_file_ref("", loaded))
        self.assertIsNone(target._resolve_attached_file_ref("a.csv", None))

    def test_missing_source_rejected(self):
        result = target._execute_edit_file_tool(1, {"content": "x"}, encrypt=False, loaded_files=[])
        self.assertFalse(result.get("ok"))
        self.assertIn("source", result.get("error", ""))

    def test_unknown_source_rejected(self):
        result = target._execute_edit_file_tool(
            1, {"source": "nope.csv", "content": "x"}, encrypt=False,
            loaded_files=[{"send_name": "a.csv", "name": "1/x.csv", "path": "1/x.csv"}],
        )
        self.assertFalse(result.get("ok"))
        self.assertIn("見つかりません", result.get("error", ""))

    def test_missing_content_rejected(self):
        result = target._execute_edit_file_tool(
            1, {"source": "a.csv"}, encrypt=False,
            loaded_files=[{"send_name": "a.csv", "name": "1/x.csv", "path": "1/x.csv"}],
        )
        self.assertFalse(result.get("ok"))
        self.assertIn("content", result.get("error", ""))

    def test_text_file_edit_preserves_extension_and_display_name(self):
        loaded = [{"send_name": "data.csv", "name": "1/abc123.csv", "path": "1/abc123.csv"}]
        with patch("app._save_user_generated_bytes_verified") as save_mock:
            save_mock.return_value = ("data_123.csv", "/files/1/data_123.csv")
            result = target._execute_edit_file_tool(
                1, {"source": "data.csv", "content": "a,b\n1,2\n"}, encrypt=False,
                loaded_files=loaded,
            )
        self.assertTrue(result.get("ok"))
        self.assertEqual(result["display_name"], "data.csv")
        self.assertEqual(result["url"], "/files/1/data_123.csv")
        saved_bytes = save_mock.call_args[0][1]
        self.assertEqual(saved_bytes.decode("utf-8"), "a,b\n1,2\n")
        # Output filename keeps the original extension.
        made_name = save_mock.call_args[0][2]()
        self.assertTrue(made_name.endswith(".csv"))
        self.assertTrue(made_name.startswith("data_"))

    def test_xlsx_edit_rebuilds_from_tsv(self):
        loaded = [{"send_name": "予定表.xlsx", "name": "1/abc123.xlsx", "path": "1/abc123.xlsx"}]
        with patch("app._save_user_generated_bytes_verified") as save_mock:
            save_mock.return_value = ("edited_123.xlsx", "/files/1/edited_123.xlsx")
            result = target._execute_edit_file_tool(
                1,
                {"source": "予定表.xlsx", "content": "name\tvalue\napple\t3\n"},
                encrypt=False,
                loaded_files=loaded,
            )
        self.assertTrue(result.get("ok"))
        self.assertEqual(result["display_name"], "予定表.xlsx")
        saved_bytes = save_mock.call_args[0][1]
        self.assertTrue(saved_bytes.startswith(b"PK"))
        made_name = save_mock.call_args[0][2]()
        self.assertTrue(made_name.endswith(".xlsx"))

    def test_binary_extension_rejected(self):
        loaded = [{"send_name": "evil.exe", "name": "1/abc123.exe", "path": "1/abc123.exe"}]
        result = target._execute_edit_file_tool(
            1, {"source": "evil.exe", "content": "x"}, encrypt=False, loaded_files=loaded
        )
        self.assertFalse(result.get("ok"))
        self.assertIn("対応していない", result.get("error", ""))

    def test_edit_file_schema_shape(self):
        schema = target._build_edit_file_tool_schema()
        self.assertEqual(schema["function"]["name"], "edit_file")
        props = schema["function"]["parameters"]["properties"]
        self.assertIn("source", props)
        self.assertIn("content", props)
        self.assertIn("source", schema["function"]["parameters"]["required"])
        self.assertIn("content", schema["function"]["parameters"]["required"])

    def test_create_file_schema_mentions_edit_file(self):
        schema = target._build_create_file_tool_schema()
        self.assertIn("edit_file", schema["function"]["description"])


if __name__ == "__main__":
    unittest.main()
