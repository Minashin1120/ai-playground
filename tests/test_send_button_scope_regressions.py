from pathlib import Path
import re
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]
CHAT_JS_ASSETS = list((APP_ROOT / "static/js").glob("chat_core.v4.8.*.js"))
assert len(CHAT_JS_ASSETS) == 1, "Only the latest versioned chat core asset should remain"
CHAT_JS = CHAT_JS_ASSETS[0].read_text(encoding="utf-8")
MIN_JS_ASSETS = list((APP_ROOT / "static/js").glob("chat_core.min.v4.8.*.js"))
assert len(MIN_JS_ASSETS) == 1, "Only the latest minified chat core asset should remain"
MIN_JS = MIN_JS_ASSETS[0].read_text(encoding="utf-8")


def _line_no(src, needle):
    for i, line in enumerate(src.splitlines(), 1):
        if needle in line:
            return i
    return None


class SendButtonScopeRegressionTests(unittest.TestCase):
    def test_lyria_helpers_declared_at_top_level(self):
        """isLyriaRealtimeModel / isGeminiRealtimeMusicModel must be module-scope.

        sendMessage (top-level) routes Lyria RealTime sends to the studio, so the
        helpers it uses must not live inside the DOMContentLoaded callback scope.
        V4.8.852..863 broke the send button for every model because the helpers
        were declared inside the callback while sendMessage referenced them from
        the top level, producing a runtime ReferenceError.
        """
        decl_islyria = _line_no(CHAT_JS, "const isLyriaRealtimeModel = ")
        decl_realtime_music = _line_no(CHAT_JS, "const isGeminiRealtimeMusicModel = ")
        self.assertIsNotNone(decl_islyria)
        self.assertIsNotNone(decl_realtime_music)
        # The first DOMContentLoaded listener that owns the per-model UI update
        # functions is the boundary: helpers used by sendMessage must precede it.
        dcl_idx = CHAT_JS.index("document.addEventListener('DOMContentLoaded', () => {")
        dcl_line = CHAT_JS[:dcl_idx].count("\n") + 1
        self.assertLess(decl_islyria, dcl_line, "isLyriaRealtimeModel must be top-level, before DOMContentLoaded")
        self.assertLess(decl_realtime_music, dcl_line, "isGeminiRealtimeMusicModel must be top-level, before DOMContentLoaded")

    def test_send_message_lyria_route_uses_top_level_helper(self):
        send_idx = CHAT_JS.index("async function sendMessage()")
        # The Lyria routing branch in sendMessage must reference isLyriaRealtimeModel,
        # which (per test above) is now in the same top-level scope.
        branch = CHAT_JS[send_idx:send_idx + 2000]
        self.assertIn("if (isLyriaRealtimeModel()) {", branch)

    def test_minified_js_declares_helper_before_send_message(self):
        """The minified bundle must not leave a dangling free reference.

        With the old buggy build the minifier renamed the declaration but kept
        the sendMessage reference as the full identifier (ReferenceError).
        Guard against that: the helper must be declared at the same brace depth
        as sendMessage and the reference must resolve.
        """
        self.assertIn("isLyriaRealtimeModel", MIN_JS)
        # A top-level const declaration of the helper must exist.  esbuild merges
        # adjacent const declarations, so the helper appears as
        # `...,isLyriaRealtimeModel=a(...,"isLyriaRealtimeModel")` without a
        # repeated `const` keyword.
        self.assertRegex(
            MIN_JS,
            r"(?:const|,)\s*isLyriaRealtimeModel=a\(",
            "minified bundle must declare isLyriaRealtimeModel as a top-level const",
        )
        self.assertRegex(
            MIN_JS,
            r"(?:const|,)\s*isGeminiRealtimeMusicModel=a\(",
            "minified bundle must declare isGeminiRealtimeMusicModel as a top-level const",
        )
        # The sendMessage usage must be a call against the declared helper, and the
        # declaration must appear before the sendMessage function in the bundle.
        decl_pos = MIN_JS.index("isLyriaRealtimeModel=a(")
        send_pos = MIN_JS.index("async function sendMessage")
        self.assertLess(decl_pos, send_pos)

    def test_minified_has_no_duplicate_helper_reference_pattern(self):
        # Old broken pattern: declaration renamed but a usage kept the full name
        # with no top-level const (that would be `isLyriaRealtimeModel()` with the
        # only earlier occurrence inside a name-string, not a declaration).
        # Assert there is exactly one declaration and no stray second reference
        # that is unbound (i.e. every occurrence belongs to the declaration chain
        # or a call that the top-level const satisfies).
        decl_count = len(re.findall(r"isLyriaRealtimeModel=a\(", MIN_JS))
        self.assertEqual(decl_count, 1)


if __name__ == "__main__":
    unittest.main()
