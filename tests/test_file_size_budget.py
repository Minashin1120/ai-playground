"""Keep single files small enough to read and edit without loading the whole file.

The line budgets come from `scripts/_release_common.py` (`size_budget`), the same
numbers `scripts/verify_changes.sh -- size-report` prints, so the warning report
and this guard cannot drift apart.  Background is in
`~/docs/context_cost_reduction_plan.md` (P2).

A file over its split limit must be split; until it is, it is frozen in ALLOWED
at its current size so it cannot grow further.  Bundled/minified frontend output
(`chat_core.v*.js`, `*.min.*.js`) is not source and is excluded by size_report().
"""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

# chat_core parts use the same 1,000 warn / 1,500 split budget as the rest of
# the source; scripts/rebuild_chat_core_parts.sh keeps them near 1,000 lines.


def load_release_common():
    spec = importlib.util.spec_from_file_location(
        "release_common", SCRIPTS / "_release_common.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


COMMON = load_release_common()

# path -> current line count.  Split the file, then remove the entry.
ALLOWED = {}


def _budget(relative):
    """Return the split limit for a source path, or None to ignore it."""
    budget = COMMON.size_budget(relative)
    return budget[1] if budget else None


def _line_count(path):
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _over_budget():
    found = {}
    for pattern in COMMON.SIZE_SOURCE_PATTERNS:
        for path in sorted(ROOT.glob(pattern)):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT).as_posix()
            limit = _budget(relative)
            if limit is None:
                continue
            lines = _line_count(path)
            if lines > limit:
                found[relative] = (lines, limit)
    return found


class FileSizeBudgetTests(unittest.TestCase):
    def test_no_new_oversized_file(self):
        offenders = []
        for relative, (lines, limit) in sorted(_over_budget().items()):
            if relative not in ALLOWED:
                offenders.append(
                    f"{relative}: {lines} > {limit} (split it or add a documented exception)"
                )
        self.assertEqual(offenders, [], "Oversized files: " + "; ".join(offenders))

    def test_allowlisted_files_do_not_grow(self):
        offenders = []
        for relative, frozen in sorted(ALLOWED.items()):
            path = ROOT / relative
            if not path.is_file():
                offenders.append(f"{relative}: missing (remove from ALLOWED)")
                continue
            lines = _line_count(path)
            if lines > frozen:
                offenders.append(f"{relative}: {lines} > frozen {frozen}")
        self.assertEqual(
            offenders,
            [],
            "Files on the size exception list must not grow. Extract the touched part first: "
            + "; ".join(offenders),
        )

    def test_allowlist_entries_are_still_oversized(self):
        over = _over_budget()
        stale = sorted(relative for relative in ALLOWED if relative not in over)
        self.assertEqual(
            stale, [], "Remove split or shrunk files from ALLOWED: " + ", ".join(stale)
        )

    def test_guard_uses_the_same_budgets_as_size_report(self):
        # The numbers live in scripts/_release_common.py; the guard must not keep a copy.
        for relative in (
            "server/background.py",
            "server/routes_example.py",
            "tests/test_example.py",
            "android/app/src/main/java/com/minashin1120/aiplayground/ChatViewModel.kt",
            "static/js/chat_core_parts/chat_core.part01_nativeconsolelog.js",
        ):
            self.assertEqual(_budget(relative), COMMON.size_budget(relative)[1], relative)
        self.assertIsNone(COMMON.size_budget("android/app/build.gradle.kts"))


if __name__ == "__main__":
    unittest.main()
