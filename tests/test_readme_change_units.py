"""分割構成の README が「変更単位の表」のまま維持されているかを確認する回帰テスト。

背景は `~/docs/context_cost_reduction_plan.md` の「4. コード構造の修正案」にある。
README の主旨を何でも入る表現に戻したり、部品を追加したまま README を更新し忘れたりすると、
次に触る人が巨大なファイルを読み直すことになるため、表の形と被覆をここで固定する。
"""

import fnmatch
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# 「ファイル名 / 主な内容 / いつ開くか」の表だけで分割構成を案内する README。
CHANGE_UNIT_READMES = (
    "server/README.md",
    "templates/chat/README.md",
    "static/js/chat_core_parts/README.md",
    "tests/README.md",
    "scripts/README.md",
    "android/README.md",
)

HEADER_COLUMNS = ("主な内容", "いつ開くか")
MINIMUM_ROWS = 5
_SEPARATOR = re.compile(r"^\|[-: |]+\|$")


def table_rows(document):
    """変更単位の表の見出し行と、その下のデータ行を返す。"""
    lines = document.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        if not all(column in line for column in HEADER_COLUMNS):
            continue
        if index + 1 >= len(lines) or not _SEPARATOR.match(lines[index + 1].strip()):
            continue
        rows = []
        for row in lines[index + 2:]:
            if not row.startswith("|"):
                break
            rows.append(row)
        return line, rows
    return None, []


def first_cell(row):
    return row.strip().strip("|").split("|")[0]


def backticked_names(row):
    return [name for name in re.findall(r"`([^`]+)`", row)]


def is_glob(name):
    return any(character in name for character in "*?[")


class ReadmeChangeUnitTests(unittest.TestCase):
    def test_change_unit_readmes_keep_the_table(self):
        for relative in CHANGE_UNIT_READMES:
            with self.subTest(readme=relative):
                document = ROOT / relative
                self.assertTrue(document.is_file(), f"{relative} がありません")
                header, rows = table_rows(document)
                self.assertIsNotNone(
                    header, f"{relative} に「主な内容 / いつ開くか」の表がありません"
                )
                self.assertIn("ファイル", header)
                self.assertGreaterEqual(
                    len(rows), MINIMUM_ROWS, f"{relative} の表の行が少なすぎます"
                )

    def test_server_readme_lists_every_server_part(self):
        document = ROOT / "server" / "README.md"
        _, rows = table_rows(document)
        listed = {name for row in rows for name in backticked_names(row)}

        parts = sorted(
            path.name
            for path in (ROOT / "server").glob("*.py")
            if path.name != "__init__.py"
        )
        self.assertGreater(len(parts), 10)
        for part in parts:
            with self.subTest(part=part):
                self.assertIn(part, listed, f"server/README.md に {part} の行がありません")

    def test_server_readme_points_to_existing_files(self):
        document = ROOT / "server" / "README.md"
        _, rows = table_rows(document)
        for row in rows:
            for name in re.findall(r"`([^`]+)`", first_cell(row)):
                if not name.endswith((".py", ".json")):
                    continue
                with self.subTest(name=name):
                    self.assertTrue(
                        (document.parent / name).resolve().is_file(),
                        f"server/README.md の {name} が存在しません",
                    )

    def test_android_readme_lists_every_main_source_file(self):
        document = ROOT / "android" / "README.md"
        _, rows = table_rows(document)
        listed = {name.split("/")[-1] for row in rows for name in backticked_names(row)}

        source_root = ROOT / "android" / "app" / "src" / "main"
        self.assertTrue(source_root.is_dir())
        for path in sorted(source_root.rglob("*.kt")):
            with self.subTest(source=path.name):
                self.assertIn(
                    path.name, listed, f"android/README.md に {path.name} の行がありません"
                )

    def test_android_readme_points_to_existing_files(self):
        document = ROOT / "android" / "README.md"
        _, rows = table_rows(document)
        known = {path.name for path in (ROOT / "android").rglob("*.kt")}
        for row in rows:
            for name in backticked_names(row):
                if not name.endswith(".kt"):
                    continue
                path = document.parent / name
                basename = name.split("/")[-1]
                exists = path.is_file()
                if not exists and is_glob(basename):
                    exists = any(fnmatch.fnmatch(candidate, basename) for candidate in known)
                elif not exists:
                    exists = basename in known
                with self.subTest(name=name):
                    self.assertTrue(
                        exists, f"android/README.md の {name} が存在しません"
                    )


if __name__ == "__main__":
    unittest.main()
