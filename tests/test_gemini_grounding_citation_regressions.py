import textwrap
import unittest
from types import SimpleNamespace

from tests.app_source import read_app_source

APP_SOURCE = read_app_source()


def _load_grounding_helpers():
    start = APP_SOURCE.index("def _chunk_grounding_info(chunk):")
    end = APP_SOURCE.index("def _extract_gemini_thought_text(part):", start)
    # 関数の中で定義されているので、行頭のインデントをそろえてから実行する。
    line_start = APP_SOURCE.rindex("\n", 0, start) + 1
    block = textwrap.dedent(APP_SOURCE[line_start:end])
    namespace = {}
    exec(block, namespace)
    return namespace


HELPERS = _load_grounding_helpers()


def _support(end_index, indices, text=None):
    return SimpleNamespace(
        segment=SimpleNamespace(end_index=end_index, text=text),
        grounding_chunk_indices=indices,
    )


def _chunk(uri):
    return SimpleNamespace(web=SimpleNamespace(uri=uri, title=None), maps=None)


class GeminiGroundingCitationRegressionTests(unittest.TestCase):
    def setUp(self):
        self.add = HELPERS["_add_gemini_citations"]
        self.chunks = [_chunk("https://a.example/"), _chunk("https://b.example/")]

    def test_end_index_is_a_utf8_byte_offset(self):
        text = "東京は大きい。雨が降る。"
        end = len("東京は大きい。".encode("utf-8"))
        result = self.add(text, [_support(end, [0])], self.chunks)
        self.assertEqual(result, "東京は大きい。[1](https://a.example/)雨が降る。")

    def test_last_segment_of_multibyte_answer_is_cited(self):
        text = "東京は大きい。雨が降る。"
        end = len(text.encode("utf-8"))
        result = self.add(text, [_support(end, [1])], self.chunks)
        self.assertEqual(result, text + "[2](https://b.example/)")

    def test_inserted_code_blocks_do_not_shift_citations(self):
        model_text = "計算します。結果は42です。"
        saved = "計算します。\n```python\nprint(42)\n```\n結果は42です。"
        end = len(model_text.encode("utf-8"))
        result = self.add(
            saved, [_support(end, [0], text="結果は42です。")], self.chunks, model_text
        )
        self.assertEqual(result, saved + "[1](https://a.example/)")

    def test_context_is_used_when_segment_text_is_missing(self):
        model_text = "前半です。後半です。"
        saved = "前半です。\n**Output:**\n```\nx\n```\n後半です。"
        end = len("前半です。".encode("utf-8"))
        result = self.add(saved, [_support(end, [0])], self.chunks, model_text)
        self.assertEqual(result, "前半です。[1](https://a.example/)\n**Output:**\n```\nx\n```\n後半です。")

    def test_supports_at_the_same_position_keep_their_order(self):
        text = "Tokyo is big. It rains."
        result = self.add(text, [_support(13, [0]), _support(13, [1])], self.chunks)
        self.assertEqual(
            result, "Tokyo is big.[1](https://a.example/)[2](https://b.example/) It rains."
        )

    def test_out_of_range_segment_is_skipped(self):
        text = "Tokyo is big."
        self.assertEqual(self.add(text, [_support(99, [0])], self.chunks), text)


class UrlRetrievalStatusLabelTests(unittest.TestCase):
    def setUp(self):
        self.label = HELPERS["_url_retrieval_status_label"]

    def test_success_is_not_shown(self):
        from google.genai import types

        self.assertEqual(self.label(types.UrlRetrievalStatus.URL_RETRIEVAL_STATUS_SUCCESS), "")
        self.assertEqual(self.label("URL_RETRIEVAL_STATUS_SUCCESS"), "")
        self.assertEqual(self.label(None), "")

    def test_failure_shows_the_status_value_without_the_enum_name(self):
        from google.genai import types

        self.assertEqual(
            self.label(types.UrlRetrievalStatus.URL_RETRIEVAL_STATUS_ERROR),
            "URL_RETRIEVAL_STATUS_ERROR",
        )
        self.assertEqual(
            self.label("UrlRetrievalStatus.URL_RETRIEVAL_STATUS_PAYWALL"),
            "URL_RETRIEVAL_STATUS_PAYWALL",
        )

    def test_url_context_list_uses_the_label(self):
        block_start = APP_SOURCE.index('url_context_chunks and (options.get')
        block = APP_SOURCE[block_start:APP_SOURCE.index("**URL Context:**", block_start)]
        self.assertIn("_url_retrieval_status_label(status)", block)
        self.assertNotIn('str(status) != "ACTIVE"', block)


if __name__ == "__main__":
    unittest.main()
