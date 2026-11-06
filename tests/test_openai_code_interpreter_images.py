"""OpenAI thinking with images uses the hosted code interpreter, not the local sandbox."""
from pathlib import Path
import ast
import base64
import re
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock
from urllib.parse import urlparse
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


class OpenAICodeInterpreterImageTests(unittest.TestCase):
    def test_image_downloads_work_without_entering_auto_search_branches(self):
        source = (APP_ROOT / "server/background.py").read_text(encoding="utf-8")
        task = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == "background_chat_task")
        helpers = {n.name: n for n in ast.walk(task) if isinstance(n, ast.FunctionDef)
                   and n.name in ("_download_openai_api_bytes", "_save_container_file")}
        # Keep the entire task body for Python's real lexical scope, including imports
        # inside search branches, but return its image helpers before app/DB/network setup.
        leading = 0
        while isinstance(task.body[leading], (ast.Import, ast.ImportFrom)):
            leading += 1
        setup = ast.parse("""
client = test_client
ci_saved_files = {}
_emit_code_interpreter_image = test_emit
return _download_openai_api_bytes, _save_container_file
""").body
        task.body = task.body[:leading] + list(helpers.values()) + setup + task.body[leading:]
        http = MagicMock()
        response = http.Client.return_value.__enter__.return_value.get.return_value
        response.status_code = 200
        response.content = b"image bytes"
        emit = Mock(return_value="/files/test/result.png")
        scope = dict(base64=base64, re=re, urlparse=urlparse, httpx=http, log_force=Mock(),
                     test_client=SimpleNamespace(api_key="test-key"), test_emit=emit)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[task], type_ignores=[])),
                     "<task-image-scope>", "exec"), scope)
        download, save = scope[task.name]("job", None, "gpt-test", None, {"enable_search": False}, 1, {})
        self.assertEqual(download("data:image/png;base64,AQID"), b"\x01\x02\x03")
        http.Client.assert_not_called()
        self.assertEqual(save("cntr_test", "cfile_test", "result.png"), "/files/test/result.png")
        emit.assert_called_once_with(b"image bytes", "result.png")
        self.assertEqual(save("cntr_test", "cfile_test", "result.png"), "/files/test/result.png")
        http.Client.return_value.__enter__.return_value.get.assert_called_once()

    def test_inline_interpreter_images_are_decoded_without_an_http_request(self):
        source = (APP_ROOT / "server/background.py").read_text(encoding="utf-8")
        node = next(n for n in ast.walk(ast.parse(source))
                    if isinstance(n, ast.FunctionDef) and n.name == "_download_openai_api_bytes")
        http = Mock()
        scope = dict(base64=base64, re=re, httpx=http, log_force=Mock(), urlparse=urlparse)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<image-download>", "exec"), scope)
        download = scope[node.name]
        self.assertEqual(download("data:image/png;base64,AQID"), b"\x01\x02\x03")
        self.assertEqual(download("data:image/jpeg;base64,AQ\r\nID"), b"\x01\x02\x03")
        for invalid in ("data:text/html;base64,AQID", "data:image/png;base64,%%%%", "data:image/png;base64,A"):
            self.assertIsNone(download(invalid))
        http.Client.assert_not_called()

    def test_responses_branch_mounts_images_on_the_hosted_python_tool(self):
        background = (APP_ROOT / "server/background.py").read_text(encoding="utf-8")
        start = background.index('log_force("Routing: Responses API Branch")')
        source = background[start:background.index("if (is_gem or is_deepseek)", start)]
        python_tool = source.index("if options.get('enable_python'):")
        grok_tool = source.index('"name": "execute_python"', python_tool)
        hosted_tool = source.index('{"type": "code_interpreter", "container": container}', python_tool)
        self.assertLess(grok_tool, hosted_tool)
        self.assertIn("if is_grok:", source[python_tool:hosted_tool])
        self.assertIn('container["file_ids"] = openai_ci_file_ids', source)
        self.assertIn("def _queue_openai_code_interpreter_image", source)
        self.assertIn("upload_name=filename", source)
        self.assertIn("input_images=response_python_images", source)
        self.assertIn('call_name in ("execute_python", "create_file", "edit_file", "list_branch_images", "import_branch_images")', source)
        self.assertIn("list_branch_images", source)
        self.assertIn("import_branch_images", source)
        self.assertIn("container_file_citation", source)
        self.assertIn('!= "api.openai.com"', source)
        self.assertIn("https://api.openai.com/v1/containers/", source)
        # History images must not be queued into the code interpreter container or local sandbox.
        # Only the current prompt's loaded_files should be uploaded/passed as input images.
        history_loop = source[source.index("for m in history:"):source.index("curr_content = []")]
        self.assertNotIn("_queue_openai_code_interpreter_image", history_loop)
        self.assertNotIn("_remember_python_image", history_loop)

    def test_named_upload_uses_the_files_api(self):
        background = (APP_ROOT / "server/background.py").read_text(encoding="utf-8")
        start = background.index("def _openai_upload_with_retry")
        source = background[start:background.index("def ", start + 10)]
        self.assertIn('purpose="user_data"', source)
        self.assertIn("file=(upload_name, tmp, mime or \"application/octet-stream\")", source)


if __name__ == "__main__":
    unittest.main()
