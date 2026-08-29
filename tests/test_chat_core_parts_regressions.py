import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JS_DIR = ROOT / "static" / "js"
PARTS_DIR = JS_DIR / "chat_core_parts"


def _combined_source() -> Path:
    assets = sorted(p for p in JS_DIR.glob("chat_core.v4.8.*.js") if ".min." not in p.name)
    assert len(assets) == 1, f"expected exactly one combined source, found {[p.name for p in assets]}"
    return assets[0]


class ChatCorePartsRegressionTests(unittest.TestCase):
    """chat_core.v4.8.*.js is edited as ordered parts under static/js/chat_core_parts/.
    The versioned combined source and its .min counterpart are rebuilt from the
    parts by scripts/build_frontend.sh.  The parts must concatenate to exactly
    the combined source so edits to a part cannot be silently lost."""

    def test_parts_exist_and_are_ordered(self):
        parts = sorted(PARTS_DIR.glob("chat_core.part*.js"))
        self.assertGreaterEqual(len(parts), 2, "chat_core should be split into multiple parts")
        names = [p.name for p in parts]
        self.assertEqual(names, sorted(names), "part filenames must sort in concatenation order")
        for idx, part in enumerate(parts, 1):
            self.assertIn(f"chat_core.part{idx:02d}_", part.name, part.name)

    def test_parts_concatenate_to_combined_source(self):
        combined = _combined_source()
        parts = sorted(PARTS_DIR.glob("chat_core.part*.js"))
        self.assertTrue(parts, "chat_core_parts/ must contain part files")
        joined = b"".join(p.read_bytes() for p in parts)
        self.assertEqual(combined.read_bytes(), joined)

    def test_parts_are_reasonably_sized_for_reading(self):
        # Keep each part small enough that agents and editors can read it in one
        # pass.  The big DOMContentLoaded callback is still split into chunks.
        for part in sorted(PARTS_DIR.glob("chat_core.part*.js")):
            with self.subTest(part=part.name):
                line_count = part.read_text(encoding="utf-8").count("\n")
                self.assertLessEqual(line_count, 2200, part.name)


if __name__ == "__main__":
    unittest.main()
