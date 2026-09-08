import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import app as target


class _EmptyFileCacheQuery:
    def with_entities(self, *_columns):
        return self

    def filter_by(self, **_filters):
        return self

    def yield_per(self, _size):
        return iter(())


class LibraryApiRegressionTests(unittest.TestCase):
    def test_files_endpoint_returns_a_bounded_sorted_page(self):
        with tempfile.TemporaryDirectory() as upload_root:
            user_dir = Path(upload_root) / "7"
            user_dir.mkdir()
            for index in range(5):
                path = user_dir / f"file-{index}.txt"
                path.write_text(str(index), encoding="utf-8")
                os.utime(path, (1_700_000_000 + index, 1_700_000_000 + index))

            old_root = target.app.config["UPLOAD_FOLDER"]
            target.app.config["UPLOAD_FOLDER"] = upload_root
            try:
                with mock.patch.object(target, "current_user", SimpleNamespace(id=7)), \
                     mock.patch.object(target, "FileCache", query=_EmptyFileCacheQuery()), \
                     mock.patch.object(target, "_get_user_file_label_map", return_value={}), \
                     mock.patch.object(target, "url_for", side_effect=lambda endpoint, filename: f"/{filename}"):
                    with target.app.test_request_context(
                        "/api/files?limit=2&sort=newest"
                    ):
                        response = target.get_files_lib.__wrapped__()
                    payload = response.get_json()
            finally:
                target.app.config["UPLOAD_FOLDER"] = old_root

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["total"], 5)
        self.assertEqual(len(payload["files"]), 2)
        self.assertTrue(payload["has_more"])
        self.assertEqual(
            [item["original_filename"] for item in payload["files"]],
            ["file-4.txt", "file-3.txt"],
        )


if __name__ == "__main__":
    unittest.main()
