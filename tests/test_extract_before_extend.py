"""Enforce "extract before extend" for files already over their line budget.

`scripts/_release_common.py` grows a single set of line budgets for `size-report`
and `tests/test_file_size_budget.py`.  Those only say how big a file may be; they
do not stop an oversized file from absorbing a whole feature.  This guard checks
the delta: a file past its warn budget may not grow by `EXTRACT_ADDED_LINES`
(150) net lines in one working-tree change, so the region being touched has to
move to its own file first.

Background: `~/docs/context_cost_reduction_plan.md` ("再発防止の仕組み" 2).
Rules for authors: `~/docs/file_layout_rules.md`.
"""
import contextlib
import importlib.util
import io
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_release_common():
    spec = importlib.util.spec_from_file_location(
        "release_common", SCRIPTS / "_release_common.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


COMMON = load_release_common()

# (lines, path, warn, split) as printed by size_report().
OVER_WARN = (1300, "server/background_example.py", 1200, 1500)
UNDER_WARN = (900, "server/routes_example.py", 900, 1200)


class ExtractViolationTests(unittest.TestCase):
    def violations(self, rows, stats, threshold=COMMON.EXTRACT_ADDED_LINES):
        return COMMON.extract_violations(rows, stats, threshold)

    def test_growth_at_the_limit_is_a_violation(self):
        found = self.violations([OVER_WARN], {OVER_WARN[1]: (400, 250)})
        self.assertEqual(found, [(OVER_WARN[1], 150, 1300, 1200)])

    def test_growth_below_the_limit_is_allowed(self):
        self.assertEqual(self.violations([OVER_WARN], {OVER_WARN[1]: (400, 251)}), [])

    def test_large_addition_with_a_larger_extraction_is_allowed(self):
        # Moving 500 lines out while adding 300 keeps the file shrinking; that is
        # the behaviour the rule wants to reward.
        self.assertEqual(self.violations([OVER_WARN], {OVER_WARN[1]: (300, 500)}), [])

    def test_files_within_budget_may_grow(self):
        self.assertEqual(self.violations([UNDER_WARN], {UNDER_WARN[1]: (500, 0)}), [])

    def test_unchanged_files_are_not_reported(self):
        self.assertEqual(self.violations([OVER_WARN], {}), [])

    def test_threshold_is_the_documented_150_lines(self):
        self.assertEqual(COMMON.EXTRACT_ADDED_LINES, 150)

    def test_new_or_renamed_files_are_ignored_by_git_numstat(self):
        # Only modified paths are measured: a split produces new files whose
        # whole content would otherwise look like growth.
        self.assertEqual(COMMON.git_numstat("does/not/exist.py"), (0, 0))


class ExtractCommandTests(unittest.TestCase):
    def run_command(self, violations):
        original = COMMON.extract_before_extend_violations
        COMMON.extract_before_extend_violations = lambda: violations
        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = COMMON.cmd_check_extract(None)
        finally:
            COMMON.extract_before_extend_violations = original
        return code, stdout.getvalue(), stderr.getvalue()

    def test_clean_tree_passes(self):
        code, out, err = self.run_command([])
        self.assertEqual(code, 0)
        self.assertIn("extract before extend ok", out)
        self.assertEqual(err, "")

    def test_violation_fails_with_the_path_and_limit(self):
        code, _out, err = self.run_command([("server/big.py", 220, 1400, 1200)])
        self.assertEqual(code, 1)
        self.assertIn("server/big.py", err)
        self.assertIn("1400", err)
        self.assertIn("limit 150", err)


class ExtractContractTests(unittest.TestCase):
    def test_verify_changes_runs_the_check(self):
        script = (SCRIPTS / "verify_changes.sh").read_text(encoding="utf-8")
        self.assertIn("run_common check-extract", script)

    def test_layout_rules_document_the_rule(self):
        rules = (ROOT.parent / "docs" / "file_layout_rules.md").read_text(encoding="utf-8")
        self.assertIn("Extract before extend", rules)
        self.assertIn("150", rules)


if __name__ == "__main__":
    unittest.main()
