import base64
import io
import os
import tempfile
import unittest
from unittest import mock

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

    def test_verified_save_returns_url_when_file_lands(self):
        """A successful write that is confirmed on disk must return its URL."""
        names = []
        def make_name():
            fname = f"agentic_test_{len(names)}_{target.os.urandom(3).hex()}.png"
            names.append(fname)
            return fname

        def fake_save(user_id, data, fname, encrypt):
            fdir = os.path.join(target.app.config["UPLOAD_FOLDER"], str(user_id))
            os.makedirs(fdir, mode=0o700, exist_ok=True)
            with open(os.path.join(fdir, fname), "wb") as fh:
                fh.write(data)

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(target.app.config, {"UPLOAD_FOLDER": tmp}), \
                 mock.patch.object(target, "_save_user_generated_bytes", side_effect=fake_save):
                fname, url = target._save_user_generated_bytes_verified(
                    7, b"fake-bytes", make_name, False,
                )
                landed = os.path.exists(os.path.join(tmp, "7", fname))

        self.assertIn(fname, names)
        self.assertEqual(url, f"/files/7/{fname}")
        self.assertTrue(landed)

    def test_verified_save_retries_when_write_is_silently_lost(self):
        """If the first save produces no on-disk file, a fresh filename is tried."""
        attempts = []
        def fake_save(user_id, data, fname, encrypt):
            attempts.append(fname)
            # First attempt silently loses the write; the second one lands.
            if len(attempts) == 2:
                fdir = os.path.join(target.app.config["UPLOAD_FOLDER"], str(user_id))
                os.makedirs(fdir, mode=0o700, exist_ok=True)
                with open(os.path.join(fdir, fname), "wb") as fh:
                    fh.write(data)

        def make_name():
            return f"agentic_test_{len(attempts)}.png"

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(target.app.config, {"UPLOAD_FOLDER": tmp}), \
                 mock.patch.object(target, "_save_user_generated_bytes", side_effect=fake_save):
                fname, url = target._save_user_generated_bytes_verified(
                    7, b"fake-bytes", make_name, False,
                )

        self.assertEqual(len(attempts), 2)
        self.assertNotEqual(attempts[0], attempts[1])
        self.assertEqual(fname, attempts[1])
        self.assertEqual(url, f"/files/7/{attempts[1]}")

    def test_verified_save_raises_after_all_attempts_fail(self):
        def fake_save(user_id, data, fname, encrypt):
            raise ValueError("storage limit exceeded")

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(target.app.config, {"UPLOAD_FOLDER": tmp}), \
                 mock.patch.object(target, "_save_user_generated_bytes", side_effect=fake_save):
                with self.assertRaisesRegex(ValueError, "storage limit exceeded"):
                    target._save_user_generated_bytes_verified(
                        7, b"fake-bytes", lambda: "agentic_x.png", False,
                    )

    def test_verified_save_raises_when_file_vanishes_after_every_attempt(self):
        def fake_save(user_id, data, fname, encrypt):
            pass  # reports success but never writes anything

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(target.app.config, {"UPLOAD_FOLDER": tmp}), \
                 mock.patch.object(target, "_save_user_generated_bytes", side_effect=fake_save):
                with self.assertRaisesRegex(RuntimeError, "on-disk verification"):
                    target._save_user_generated_bytes_verified(
                        7, b"fake-bytes", lambda: "agentic_x.png", False,
                    )


if __name__ == "__main__":
    unittest.main()
