"""Exercise production error/vision helpers without live providers or private account data."""
import ast
import re
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]


def function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def test_proxy_errors_keep_status_without_rendering_html():
    readable = function("server/token_utils.py", "readable_provider_error", {"re": re})
    assert "413" in readable("DeepSeek Error: <html><head>413 Request Entity Too Large</head></html>")
    assert "<html>" not in readable("DeepSeek Error: <html>502 Bad Gateway</html>")
    assert readable("API key invalid") == "API key invalid"


def test_native_vision_compresses_large_images_and_keeps_original_on_failure():
    encode = Mock(return_value=b"compact")
    compact = function("server/image_tools.py", "_compact_vision_image", {"_make_thumbnail_webp": encode})
    original = b"x" * (300 * 1024)
    assert compact(original, "image/png") == (b"compact", "image/webp")
    encode.assert_called_once_with(original, 1536, 85)
    encode.return_value = None
    assert compact(original, "image/png") == (original, "image/png")
    encode.reset_mock()
    assert compact(b"small", "image/jpeg") == (b"small", "image/jpeg")
    encode.assert_not_called()


def test_saved_error_exits_before_normal_answer_storage():
    tree = ast.parse((ROOT / "server/background.py").read_text())
    worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "background_chat_task")
    guards = [n for n in ast.walk(worker) if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
              and n.test.id == "stream_error_persisted" and any(isinstance(x, ast.Return) for x in n.body)]
    assert any(n.lineno > 7000 for n in guards)
