import base64
import io
import os
import unittest

from PIL import Image


os.environ.setdefault("FLASK_SECRET_KEY", "agentic-image-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-agentic-image-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target


class AgenticImageRegressionTests(unittest.TestCase):
    def test_svg_inline_output_is_rasterized_to_a_real_png(self):
        svg = b"""
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 80">
          <rect width="120" height="80" fill="#2563eb"/>
          <text x="10" y="45" fill="white">chart</text>
        </svg>
        """

        image_bytes, extension = target._prepare_agentic_image_bytes(
            base64.b64encode(svg).decode("ascii"),
            "image/svg+xml",
        )

        self.assertEqual(extension, "png")
        self.assertTrue(image_bytes.startswith(b"\x89PNG\r\n\x1a\n"))
        with Image.open(io.BytesIO(image_bytes)) as image:
            self.assertEqual(image.format, "PNG")
            self.assertEqual(image.size, (120, 80))

    def test_svg_disallows_active_or_external_content(self):
        unsafe_samples = [
            b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>',
            b'<svg xmlns="http://www.w3.org/2000/svg"><image href="https://example.com/a.png"/></svg>',
            b'<svg xmlns="http://www.w3.org/2000/svg"><rect onclick="alert(1)"/></svg>',
            b'<svg xmlns="http://www.w3.org/2000/svg"><style>@import "https://example.com/a.css";</style></svg>',
        ]

        for svg in unsafe_samples:
            with self.subTest(svg=svg):
                with self.assertRaises(ValueError):
                    target._prepare_agentic_image_bytes(svg, "image/svg+xml")

    def test_non_image_inline_output_is_not_published_as_an_image(self):
        with self.assertRaisesRegex(ValueError, "not a supported image"):
            target._prepare_agentic_image_bytes(b'{"kind":"tool-output"}', "application/json")

    def test_raster_extension_comes_from_actual_bytes(self):
        buffer = io.BytesIO()
        Image.new("RGB", (4, 3), "red").save(buffer, format="JPEG")

        image_bytes, extension = target._prepare_agentic_image_bytes(
            buffer.getvalue(),
            "image/png",
        )

        self.assertEqual(extension, "jpg")
        self.assertEqual(image_bytes, buffer.getvalue())

    def test_sandbox_image_ref_streamed_with_saved_url(self):
        buffer_state = [""]
        saved_urls = ["/files/1/agentic_abc.png"]
        consumed = []
        out = target._rewrite_streamed_sandbox_refs(
            "![cropped](sandbox:/mnt/data/cropped_image.jpeg)",
            buffer_state,
            saved_urls,
            consumed,
        )
        self.assertEqual(out, "![cropped](/files/1/agentic_abc.png)")
        self.assertEqual(saved_urls, [])
        self.assertEqual(consumed, ["/files/1/agentic_abc.png"])
        self.assertEqual(buffer_state, [""])

    def test_sandbox_image_ref_streamed_without_saved_url_kept_for_final_pass(self):
        buffer_state = [""]
        saved_urls = []
        consumed = []
        out = target._rewrite_streamed_sandbox_refs(
            "![cropped](sandbox:/mnt/data/cropped_image.jpeg)",
            buffer_state,
            saved_urls,
            consumed,
        )
        self.assertEqual(out, "![cropped](sandbox:/mnt/data/cropped_image.jpeg)")
        self.assertEqual(saved_urls, [])
        self.assertEqual(consumed, [])
        self.assertEqual(buffer_state, [""])

    def test_sandbox_image_ref_split_across_chunks(self):
        buffer_state = [""]
        saved_urls = ["/files/1/agentic_xyz.png"]
        consumed = []
        first = target._rewrite_streamed_sandbox_refs(
            "![cro", buffer_state, saved_urls, consumed
        )
        self.assertEqual(first, "")
        self.assertEqual(buffer_state, ["![cro"])
        second = target._rewrite_streamed_sandbox_refs(
            "pped](sandbox:/mnt/data/cropped_image.jpeg)",
            buffer_state,
            saved_urls,
            consumed,
        )
        self.assertEqual(second, "![cropped](/files/1/agentic_xyz.png)")
        self.assertEqual(buffer_state, [""])

    def test_sandbox_image_ref_final_pass_replaces_unresolved_with_note(self):
        saved_urls = []
        consumed = []
        out = target._rewrite_sandbox_image_refs(
            "text ![a](sandbox:/mnt/data/x.png) tail", saved_urls, consumed
        )
        self.assertNotIn("sandbox:", out)
        self.assertIn("画像データを取得できませんでした", out)

    def test_non_sandbox_text_is_untouched(self):
        buffer_state = [""]
        saved_urls = []
        consumed = []
        out = target._rewrite_streamed_sandbox_refs(
            "hello ![img](https://example.com/a.png)", buffer_state, saved_urls, consumed
        )
        self.assertEqual(out, "hello ![img](https://example.com/a.png)")
        self.assertEqual(buffer_state, [""])

    def test_extract_sandbox_image_filenames_from_code(self):
        code = (
            'img = Image.open("input.png")\n'
            'img.save("output.png")\n'
            'cv2.imwrite("cropped.jpg", result)\n'
            'fig.savefig("/mnt/data/chart.png")\n'
            'data.tofile("raw.bin")\n'
            'img.save(variable_name)\n'
        )
        names = target._extract_sandbox_image_filenames(code)
        self.assertIn("output.png", names)
        self.assertIn("cropped.jpg", names)
        self.assertIn("chart.png", names)
        self.assertNotIn("raw.bin", names)
        self.assertNotIn("variable_name", names)

    def test_bare_filename_ref_resolved_via_filename_map(self):
        filename_url_map = {
            "result.png": "/files/1/agentic_abc.png",
        }
        out = target._rewrite_sandbox_image_refs(
            "編集後 ![背景透過ロゴ](result.png)",
            [],
            [],
            filename_url_map,
        )
        self.assertIn("![背景透過ロゴ](/files/1/agentic_abc.png)", out)
        self.assertNotIn("result.png", out)

    def test_mnt_data_ref_resolved_via_filename_map(self):
        filename_url_map = {
            "chart.png": "/files/1/agentic_xyz.png",
        }
        out = target._rewrite_sandbox_image_refs(
            "![chart](/mnt/data/chart.png)",
            [],
            [],
            filename_url_map,
        )
        self.assertIn("![chart](/files/1/agentic_xyz.png)", out)

    def test_bare_filename_ref_consumes_saved_urls_when_no_map(self):
        out = target._rewrite_sandbox_image_refs(
            "![img](result.png)",
            ["/files/1/agentic_first.png"],
            [],
        )
        self.assertIn("![img](/files/1/agentic_first.png)", out)

    def test_unresolved_bare_filename_ref_is_left_unchanged(self):
        out = target._rewrite_sandbox_image_refs(
            "![img](result.png)",
            [],
            [],
        )
        self.assertEqual(out, "![img](result.png)")

    def test_unresolved_mnt_data_ref_is_replaced_with_note(self):
        out = target._rewrite_sandbox_image_refs(
            "![img](/mnt/data/result.png)",
            [],
            [],
        )
        self.assertNotIn("/mnt/data/", out)
        self.assertIn("画像データを取得できませんでした", out)

    def test_streamed_bare_ref_with_map_resolves(self):
        buffer_state = [""]
        filename_url_map = {"result.png": "/files/1/agentic_abc.png"}
        out = target._rewrite_streamed_sandbox_refs(
            "![cropped](result.png)",
            buffer_state,
            [],
            [],
            filename_url_map,
        )
        self.assertEqual(out, "![cropped](/files/1/agentic_abc.png)")
        self.assertEqual(buffer_state, [""])

    def test_streamed_bare_ref_falls_back_to_saved_urls(self):
        buffer_state = [""]
        saved_urls = ["/files/1/agentic_first.png"]
        consumed = []
        out = target._rewrite_streamed_sandbox_refs(
            "![cropped](result.png)",
            buffer_state,
            saved_urls,
            consumed,
        )
        self.assertEqual(out, "![cropped](/files/1/agentic_first.png)")
        self.assertEqual(consumed, ["/files/1/agentic_first.png"])

    def test_placeholder_and_consumed_url_dedup(self):
        """The placeholder for a map-resolved URL must be removed on final pass."""
        filename_url_map = {"result.png": "/files/1/agentic_abc.png"}
        consumed = []
        text = (
            "\n![Agentic View](/files/1/agentic_abc.png)\n"
            "![背景透過ロゴ](result.png)\n"
        )
        out = target._rewrite_sandbox_image_refs(text, [], consumed, filename_url_map)
        # The caller removes the Agentic View placeholder for every consumed URL.
        for url in list(consumed):
            out = out.replace(f"![Agentic View]({url})", "")
        self.assertNotIn("![Agentic View](/files/1/agentic_abc.png)", out)
        self.assertIn("![背景透過ロゴ](/files/1/agentic_abc.png)", out)
        self.assertEqual(consumed, ["/files/1/agentic_abc.png"])


if __name__ == "__main__":
    unittest.main()
