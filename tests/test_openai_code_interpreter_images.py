"""OpenAI thinking with images uses the hosted code interpreter, not the local sandbox."""
from pathlib import Path
import ast
import base64
import hashlib
import json
import os
import re
import time
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock
from urllib.parse import urlparse
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]
IMAGES_PART = APP_ROOT / "server" / "background_stream_openai_images.py"


def load_outputs_class(scope):
    """Exec the extracted part and return its `OpenAICodeInterpreterOutputs` class."""
    source = IMAGES_PART.read_text(encoding="utf-8")
    exec(compile(source, str(IMAGES_PART), "exec"), scope)
    return scope["OpenAICodeInterpreterOutputs"]


def make_outputs(scope, client, **kwargs):
    return load_outputs_class(scope)(
        client,
        kwargs.get("pub", Mock()),
        kwargs.get("user_id", "user"),
        kwargs.get("user_config", {}),
        kwargs.get("agentic_image_digests", set()),
        kwargs.get("agentic_saved_urls", []),
        kwargs.get("agentic_filename_url_map", {}),
        kwargs.get("append_content", Mock()),
        kwargs.get("append_text", Mock()),
        kwargs.get("get_full_res", lambda: ""),
    )


class OpenAICodeInterpreterImageTests(unittest.TestCase):
    def test_image_downloads_work_without_entering_auto_search_branches(self):
        # The Responses branch's image helpers live on the extracted part now.
        http = MagicMock()
        response = http.Client.return_value.__enter__.return_value.get.return_value
        response.status_code = 200
        response.content = b"image bytes"
        scope = dict(base64=base64, re=re, urlparse=urlparse, httpx=http, log_force=Mock(),
                     hashlib=hashlib, time=time, os=os, json=json,
                     _prepare_agentic_image_bytes=lambda data, mime=None: (data, "png"),
                     _save_user_generated_bytes_verified=Mock(
                         return_value=(None, "/files/test/result.png")))
        outputs = make_outputs(scope, SimpleNamespace(api_key="test-key"))
        self.assertEqual(outputs.download_api_bytes("data:image/png;base64,AQID"), b"\x01\x02\x03")
        http.Client.assert_not_called()
        self.assertEqual(
            outputs.save_container_file("cntr_test", "cfile_test", "result.png"),
            "/files/test/result.png")
        self.assertEqual(
            outputs.save_container_file("cntr_test", "cfile_test", "result.png"),
            "/files/test/result.png")
        http.Client.return_value.__enter__.return_value.get.assert_called_once()

    def test_inline_interpreter_images_are_decoded_without_an_http_request(self):
        http = Mock()
        scope = dict(base64=base64, re=re, httpx=http, log_force=Mock(), urlparse=urlparse)
        download = make_outputs(scope, None).download_api_bytes
        self.assertEqual(download("data:image/png;base64,AQID"), b"\x01\x02\x03")
        self.assertEqual(download("data:image/jpeg;base64,AQ\r\nID"), b"\x01\x02\x03")
        for invalid in ("data:text/html;base64,AQID", "data:image/png;base64,%%%%", "data:image/png;base64,A"):
            self.assertIsNone(download(invalid))
        http.Client.assert_not_called()

    def test_responses_branch_mounts_images_on_the_hosted_python_tool(self):
        source = (APP_ROOT / "server/background_stream_openai.py").read_text(encoding="utf-8")
        images_source = IMAGES_PART.read_text(encoding="utf-8")
        python_tool = source.index("if options.get('enable_python'):")
        grok_tool = source.index('"name": "execute_python"', python_tool)
        hosted_tool = source.index('{"type": "code_interpreter", "container": container}', python_tool)
        self.assertLess(grok_tool, hosted_tool)
        self.assertIn("if is_grok:", source[python_tool:hosted_tool])
        self.assertIn('container["file_ids"] = openai_ci_file_ids', source)
        self.assertIn("def _queue_openai_code_interpreter_image", source)
        self.assertIn("upload_name=filename", source)
        self.assertIn("input_images=response_python_images", source)
        self.assertIn('call_name in (SKILL_TOOL_NAME, "execute_python", "create_file", "edit_file", "list_branch_images", "import_branch_images")', source)
        self.assertIn("list_branch_images", source)
        self.assertIn("import_branch_images", source)
        self.assertIn("container_file_citation", images_source)
        self.assertIn('!= "api.openai.com"', images_source)
        self.assertIn("https://api.openai.com/v1/containers/", images_source)
        self.assertIn("code_interpreter = OpenAICodeInterpreterOutputs(", source)
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
