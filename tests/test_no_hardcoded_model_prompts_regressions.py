"""Model-facing instructions must be auto system prompt notices, not hard-coded strings.

A hard-coded instruction cannot be seen, turned off or edited by the user. The rule and the places to
update are in ~/docs/auto_system_prompt_notices.md (AGENTS.md: "モデルへ渡す指示文は直書きしない").

This test parses server/*.py and mcp_service/*.py and fails on a long string literal that is
  - assigned to a name that looks like a prompt (PROMPT / GUIDANCE / INSTRUCTION / NOTICE / HINT / PREAMBLE), or
  - assigned to, or passed as, system_prompt / system_instruction / instructions / combined_prompt, or
  - the content of a {"role": "system" | "developer"} message.
Texts that are not chat instructions go into ALLOWED with the reason. Do not add a chat instruction there.
"""
import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_GLOBS = ("server/*.py", "mcp_service/*.py")
MIN_CHARS = 40
NAME_PATTERN = re.compile(r"(PROMPT|GUIDANCE|INSTRUCTION|NOTICE|HINT|PREAMBLE)", re.I)
SYSTEM_KEYS = {"system_prompt", "system_instruction", "instructions", "combined_prompt", "sys_prompt"}
# The mechanism itself: defaults of the auto system prompt notices.
MECHANISM_PREFIX = "AUTO_SYSTEM_PROMPT_NOTICE_"

# (file, name, text it contains) -> why it is allowed. A new literal with the same name in the same
# file is still reported. Keep this list short; add to it only for text that is not a chat instruction
# (a feature's own definition, an internal prompt of another feature).
ALLOWED = {
    ("server/background.py", "CODING_MODE_SYSTEM_PROMPT", "[Coding Mode]"):
        "Coding Mode's own definition: the answer format is parsed, so the text is not user-editable.",
    ("server/background.py", "retry_prompt", "Return an image for this request"):
        "Internal retry of an image generation request, not a chat instruction.",
    ("server/agentic_media.py", "DEFAULT_LLM_TRANSCRIBE_PROMPT", "この音声を正確に"):
        "Default of the user's own transcription prompt setting.",
    ("server/agentic_media.py", "DEFAULT_IMAGE_ANALYSIS_PROMPT", "Describe this image in extreme detail"):
        "Default of the image_analysis notice (kept for the Vision Model path).",
    ("server/settings_ai.py", "transcription_prompt", "聞き取れない、無音"):
        "Speech-to-text request, not a chat.",
    ("server/routes_settings.py", "sys_prompt", "あなたはチャットアプリの『設定アシスタント』"):
        "Prompt of the settings assistant (/settings), not a chat.",
    ("server/routes_chat.py", "system message", "Generate a short title"):
        "Chat title generation, not a chat instruction.",
    ("mcp_service/execution.py", "_MCP_GUIDANCE_DEFAULT_PREAMBLE", "You have Model Context Protocol"):
        "Fallback of the mcp notice when the user's text is empty (same text as the notice default).",
}


def _literal(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _literal(node.left), _literal(node.right)
        return None if left is None or right is None else left + right
    return None


def _target_name(target):
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
        return target.slice.value
    return None


def find_hardcoded_prompts(source, filename):
    """Returns [(file, line, name, text)] of long prompt-like string literals in [source]."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign):
            text = _literal(node.value)
            if text and len(text) >= MIN_CHARS:
                for target in node.targets:
                    name = _target_name(target)
                    if name and (NAME_PATTERN.search(name) or name in SYSTEM_KEYS):
                        found.append((filename, node.lineno, name, text))
        elif isinstance(node, ast.keyword) and node.arg in SYSTEM_KEYS:
            text = _literal(node.value)
            if text and len(text) >= MIN_CHARS:
                found.append((filename, node.value.lineno, node.arg, text))
        elif isinstance(node, ast.Dict):
            entries = {k.value: v for k, v in zip(node.keys, node.values)
                       if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            role = _literal(entries["role"]) if "role" in entries else None
            if role in ("system", "developer") and "content" in entries:
                text = _literal(entries["content"])
                if text and len(text) >= MIN_CHARS:
                    found.append((filename, node.lineno, f"{role} message", text))
    return found


def _is_allowed(file, name, text):
    return any(file == f and name == n and start in text for f, n, start in ALLOWED)


def _is_allowed_by(entry, file, name, text):
    return entry[0] == file and entry[1] == name and entry[2] in text


def _scan_repository():
    found = []
    for pattern in SCAN_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            relative = path.relative_to(ROOT).as_posix()
            found += find_hardcoded_prompts(path.read_text(encoding="utf-8"), relative)
    return found


class NoHardcodedModelPromptsTests(unittest.TestCase):
    def test_model_instructions_are_not_hardcoded(self):
        offenders = [
            f"{file}:{line} {name}" for file, line, name, text in _scan_repository()
            if not name.startswith(MECHANISM_PREFIX) and not _is_allowed(file, name, text)
        ]
        self.assertEqual(
            offenders, [],
            "Model-facing instructions must be auto system prompt notices (user-editable), not hard-coded. "
            "See ~/docs/auto_system_prompt_notices.md. Offenders: " + ", ".join(offenders),
        )

    def test_allowed_entries_still_exist(self):
        found = _scan_repository()
        stale = sorted(
            entry for entry in ALLOWED
            if not any(_is_allowed_by(entry, file, name, text) for file, _, name, text in found)
        )
        self.assertEqual(stale, [], f"Remove entries that no longer match any literal: {stale}")

    def test_detector_flags_known_patterns(self):
        long_text = "x" * MIN_CHARS
        samples = {
            "GUIDE_NOTICE = '%s'\n" % long_text: "GUIDE_NOTICE",
            "conf['system_instruction'] = '%s'\n" % long_text: "system_instruction",
            "call(system_instruction='%s')\n" % long_text: "system_instruction",
            "options['system_prompt'] = f'{a}' '%s'\n" % long_text: "system_prompt",
            "m = {'role': 'system', 'content': '%s'}\n" % long_text: "system message",
            "x = ('%s' + '%s')\nmy_prompt = ('%s' + '%s')\n" % (long_text, long_text, long_text, long_text): "my_prompt",
        }
        for source, name in samples.items():
            with self.subTest(name=name):
                self.assertIn(name, [n for _, _, n, _ in find_hardcoded_prompts(source, "sample.py")])

    def test_detector_ignores_short_and_non_prompt_strings(self):
        long_text = "x" * MIN_CHARS
        source = (
            "PROMPT_BAR_MODE = 'compact'\n"
            "error_message = '%s'\n"
            "system_prompt = build_prompt()\n"
            "m = {'role': 'user', 'content': '%s'}\n"
        ) % (long_text, long_text)
        self.assertEqual(find_hardcoded_prompts(source, "sample.py"), [])


if __name__ == "__main__":
    unittest.main()
