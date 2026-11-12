"""The Gemini file parts moved to `background_stream_gemini_files.py` stay wired.

The upload / Files API cache helpers used to be nested closures that read
`g_client`, `user_id`, and `gemini_files_api_enabled` from `background_chat_task`.
They are module-level functions now, so those three values arrive as arguments.
A missing or mis-ordered argument would only show up during a real upload, so the
wiring is checked here with a fake client and cache.
"""
import unittest
from types import SimpleNamespace

import app as app_module

FILE_URI = "https://generativelanguage.googleapis.com/v1beta/files/abc123"


class _FakeFiles:
    def __init__(self, fetched):
        self.fetched = fetched
        self.get_calls = []
        self.upload_calls = []

    def get(self, name=None):
        self.get_calls.append(name)
        return self.fetched

    def upload(self, file=None, config=None):
        self.upload_calls.append(config)
        return self.fetched


class GeminiFilePartWiringTests(unittest.TestCase):
    def setUp(self):
        self._saved = (
            app_module._get_file_cache,
            app_module._upsert_file_cache,
            app_module.safe_db_commit,
        )
        self.writes = []
        app_module._get_file_cache = lambda user_id, rel_path, provider: self.cache
        app_module._upsert_file_cache = lambda *args, **kwargs: self.writes.append((args, kwargs))
        app_module.safe_db_commit = lambda: None

    def tearDown(self):
        (
            app_module._get_file_cache,
            app_module._upsert_file_cache,
            app_module.safe_db_commit,
        ) = self._saved

    def _cache(self, **overrides):
        values = dict(
            file_id=None,
            file_uri=FILE_URI,
            state="ACTIVE",
            size_bytes=10,
            mtime=100,
            mime_type="application/pdf",
            last_checked_at=None,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_cached_part_uses_the_cache_and_client_arguments(self):
        self.cache = self._cache(file_id="files/abc123")
        client = SimpleNamespace(files=_FakeFiles(SimpleNamespace(
            name="files/abc123",
            uri=FILE_URI,
            state="ACTIVE",
        )))

        part = app_module._gemini_get_cached_part_impl(
            7, True, client, "docs/report.pdf", "application/pdf", size=10, mtime=100
        )

        self.assertIsNotNone(part, "an ACTIVE cached file must produce a Part")
        self.assertEqual(["files/abc123"], client.files.get_calls)
        self.assertTrue(self.writes, "the cache should be refreshed after a successful lookup")
        self.assertEqual(7, self.writes[-1][0][0])

    def test_cached_part_is_skipped_when_the_files_api_is_disabled(self):
        self.cache = self._cache()
        client = SimpleNamespace(files=_FakeFiles(SimpleNamespace(uri=FILE_URI)))

        part = app_module._gemini_get_cached_part_impl(
            7, False, client, "docs/report.pdf", "application/pdf", size=10, mtime=100
        )

        self.assertIsNone(part)
        self.assertEqual([], client.files.get_calls)

    def test_cache_mismatch_returns_none(self):
        self.cache = self._cache(size_bytes=11)
        client = SimpleNamespace(files=_FakeFiles(SimpleNamespace(uri=FILE_URI)))

        part = app_module._gemini_get_cached_part_impl(
            7, True, client, "docs/report.pdf", "application/pdf", size=10, mtime=100
        )

        self.assertIsNone(part)
        self.assertEqual([], client.files.get_calls)

    def test_upload_reports_vertex_mode_instead_of_calling_the_api(self):
        self.cache = self._cache()
        client = SimpleNamespace(files=_FakeFiles(SimpleNamespace(uri=FILE_URI)))

        uploaded, state, error = app_module._gemini_upload_with_retry_impl(
            7, False, client, b"data", "application/pdf", ".pdf", "docs/report.pdf"
        )

        self.assertIsNone(uploaded)
        self.assertIsNone(state)
        self.assertIn("Vertex AI", error)
        self.assertEqual([], client.files.upload_calls)

    def test_file_uri_reads_dicts_and_objects(self):
        self.assertEqual(
            FILE_URI, app_module._gemini_file_uri({"name": "files/abc123", "uri": FILE_URI})
        )
        self.assertEqual(
            FILE_URI, app_module._gemini_file_uri(SimpleNamespace(uri=FILE_URI))
        )
        self.assertIsNone(app_module._gemini_file_uri(None))


if __name__ == "__main__":
    unittest.main()
