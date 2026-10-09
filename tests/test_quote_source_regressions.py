"""A quoted selection also tells the model whose message it came from and where."""
from collections import namedtuple
from unittest import mock
import unittest

import app as target

Row = namedtuple("Row", "id parent_id role")


def _rows_query(rows):
    query = mock.MagicMock()
    query.filter.return_value.all.return_value = rows
    return query


class QuoteSourceTests(unittest.TestCase):
    def test_notice_is_registered_and_enabled_by_default(self):
        self.assertIn("quote_source", target.AUTO_SYSTEM_PROMPT_NOTICE_KEYS)
        self.assertIn("{{quote_source}}", target.AUTO_SYSTEM_PROMPT_NOTICE_DEFAULTS["quote_source"])
        config = target._build_default_auto_system_prompt_notices_config()
        self.assertTrue(config["quote_source"]["enabled"])

    def test_render_fills_role_and_position(self):
        template = target.AUTO_SYSTEM_PROMPT_NOTICE_DEFAULTS["quote_source"]
        self.assertEqual(target._render_quote_source_notice(template, "assistant", 5), "引用元: assistant (message #5)")
        self.assertEqual(target._render_quote_source_notice(template, "user", None), "引用元: user")
        self.assertEqual(target._render_quote_source_notice("From {{ quote_source }}!", "user", 2), "From user (message #2)!")
        self.assertEqual(target._render_quote_source_notice(template, "system", 1), "")

    def test_resolve_numbers_the_path_to_the_new_message(self):
        rows = [Row(1, None, "user"), Row(2, 1, "assistant"), Row(3, 2, "user"), Row(4, 3, "assistant"), Row(5, 3, "assistant")]
        with mock.patch.object(target.db.session, "query", return_value=_rows_query(rows)):
            self.assertEqual(target._resolve_quote_source(9, 4, 2), {"role": "assistant", "number": 2})
            self.assertEqual(target._resolve_quote_source(9, 4, "3"), {"role": "user", "number": 3})
            # Another branch of the same thread: only the speaker is known.
            self.assertEqual(target._resolve_quote_source(9, 4, 5), {"role": "assistant", "number": None})
            # Unknown, malformed or missing ids add nothing.
            self.assertIsNone(target._resolve_quote_source(9, 4, 99))
            self.assertIsNone(target._resolve_quote_source(9, 4, "abc"))
            self.assertIsNone(target._resolve_quote_source(9, 4, None))


if __name__ == "__main__":
    unittest.main()
