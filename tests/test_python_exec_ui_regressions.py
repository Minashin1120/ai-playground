#!/usr/bin/env python3
"""Regressions for completed-answer Python execution UI (footer button + modal)."""

from __future__ import annotations

import re
import subprocess
import tempfile
import unittest
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]


def _current_chat_core() -> Path:
    assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
    if not assets:
        raise AssertionError("chat_core.v4.8.*.js not found")
    return assets[-1]


def _current_custom_css() -> Path:
    assets = sorted((APP_ROOT / "static" / "css").glob("chat.custom.v4.8.*.css"))
    if not assets:
        raise AssertionError("chat.custom.v4.8.*.css not found")
    return assets[-1]


class PythonExecUiRegressionTests(unittest.TestCase):
    def test_chat_core_exposes_footer_button_and_modal_flow(self):
        source = _current_chat_core().read_text(encoding="utf-8")
        self.assertIn("function extractPythonExecutionsFromContent(rawText)", source)
        self.assertIn("function openPythonExecDetail(id)", source)
        self.assertIn("function closePythonExecDetail(skipHistory = false)", source)
        self.assertIn("window.openPythonExecDetail = openPythonExecDetail", source)
        self.assertIn("python_executions:", source)
        self.assertIn('class="python-exec-btn"', source)
        self.assertIn("openPythonExecDetail('${id}')", source)
        self.assertIn("buildAiMarkdownHtml(displayText)", source)
        self.assertIn("function showPythonExecDetailModal(messageId = null)", source)
        self.assertIn("history.pushState(state, '', '/python-execution')", source)
        self.assertIn("if (!skipHistory && location.pathname === '/python-execution')", source)
        self.assertIn("'/python-execution': { id: 'python-exec-modal'", source)
        self.assertIn("case 'python-exec-modal': closePythonExecDetail(skipHistory); break;", source)
        # Completed answers must not render pyexec fences inline.
        pyexec_block = source[source.index("if (l === 'pyexec')") : source.index("if (l === 'pyexec')") + 280]
        self.assertIn("return '';", pyexec_block)

    def test_chat_template_and_css_include_python_exec_modal(self):
        template = (APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        css = _current_custom_css().read_text(encoding="utf-8")
        self.assertIn('id="python-exec-modal"', template)
        self.assertIn('id="python-exec-modal-body"', template)
        self.assertIn('id="python-exec-modal-title"', template)
        modal_open = template[template.index('<div id="python-exec-modal"') :]
        modal_open = modal_open[: modal_open.index(">") + 1]
        self.assertIn("modal-overlay", modal_open)
        self.assertNotIn("onclick=", modal_open)
        self.assertIn("@app.route('/python-execution')", (APP_ROOT / "app.py").read_text(encoding="utf-8"))
        self.assertIn(".python-exec-btn", css)
        self.assertIn("#python-exec-modal-body", css)

    def test_extract_python_executions_strips_visible_tool_blocks(self):
        source = _current_chat_core().read_text(encoding="utf-8")
        start = source.index("function normalizeMarkdownNewlines(text)")
        end = source.index("function openPythonExecDetail(id)")
        helpers = source[start:end]
        harness = f"""
function escapeHtml(s) {{ return String(s); }}
function hashString(s) {{ return String(s.length); }}
{helpers}
const sample = `Answer text.

\\`\\`\\`python
print(1+1)
\\`\\`\\`

**Output:**
\\`\\`\\`
2
\\`\\`\\`

\\`\\`\\`pyexec
{{"code":"print(1+1)","output":"2"}}
\\`\\`\\`

Done.`;
const r = extractPythonExecutionsFromContent(sample);
if (r.executions.length !== 1) {{ console.error(r); process.exit(2); }}
if (r.executions[0].code !== 'print(1+1)' || r.executions[0].output !== '2') process.exit(3);
if (r.text.includes('pyexec') || r.text.includes('print(1+1)') || r.text.includes('**Output:**')) process.exit(4);
if (!r.text.includes('Answer text.') || !r.text.includes('Done.')) process.exit(5);

const plain = '```python\\nprint(9)\\n```\\n';
const r2 = extractPythonExecutionsFromContent(plain);
if (r2.executions.length !== 0 || !r2.text.includes('print(9)')) process.exit(6);
console.log('OK');
"""
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
            fh.write(harness)
            path = fh.name
        try:
            proc = subprocess.run(
                ["node", path],
                capture_output=True,
                text=True,
                check=False,
            )
        finally:
            Path(path).unlink(missing_ok=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("OK", proc.stdout)

    def test_system_version_assets_exist_for_current_release(self):
        app_py = (APP_ROOT / "app.py").read_text(encoding="utf-8")
        m = re.search(r"SYSTEM_VERSION'\]\s*=\s*'V(4\.8\.\d+)'", app_py)
        self.assertIsNotNone(m)
        ver = m.group(1).lower()
        self.assertTrue((APP_ROOT / "static" / "js" / f"chat_core.v{ver}.js").exists())
        self.assertTrue((APP_ROOT / "static" / "css" / f"chat.custom.v{ver}.css").exists())
        self.assertTrue((APP_ROOT / "static" / "css" / f"chat.tailwind.v{ver}.css").exists())
        self.assertTrue(list((APP_ROOT / "static" / "changelogs").glob(f"*v{ver}.md")))


if __name__ == "__main__":
    unittest.main()
