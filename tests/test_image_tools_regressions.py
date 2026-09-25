import base64
import os
import re
import shutil
import struct
import subprocess
import sys
import unittest
import zlib
from pathlib import Path
from unittest import mock


os.environ.setdefault("FLASK_SECRET_KEY", "image-tools-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-image-tools-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target


APP_ROOT = Path(__file__).resolve().parents[1]
FFMPEG = shutil.which("ffmpeg")
HAS_TOOLS = bool(FFMPEG and shutil.which("cwebp") and shutil.which("dwebp"))

TINY_AVIF_16X8 = base64.b64decode(
    "AAAAIGZ0eXBhdmlmAAAAAGF2aWZtaWYxbWlhZk1BMUIAAADrbWV0YQAAAAAAAAAhaGRscgAAAAAAAAAAcGljdAAAAAAAAAAAAAAA"
    "AAAAAAAOcGl0bQAAAAAAAQAAAB5pbG9jAAAAAEQAAAEAAQAAAAEAAAETAAAAJAAAAChpaW5mAAAAAAABAAAAGmluZmUCAAAAAAEA"
    "AGF2MDFDb2xvcgAAAABqaXBycAAAAEtpcGNvAAAAFGlzcGUAAAAAAAAAEAAAAAgAAAAQcGl4aQAAAAADCAgIAAAADGF2MUOBAAwA"
    "AAAAE2NvbHJuY2x4AAEADQAGgAAAABdpcG1hAAAAAAAAAAEAAQQBAoMEAAAALG1kYXQSAAoIGAy/tEBDQaEyFhxHh4Xd0000wsAA"
    "ALCfzl51i9urL8A="
)


def _rgba_png(width, height):
    return target._encode_png(width, height, 6, 8, [bytes([200, 30, 90, 255]) * width] * height)


def _riff_webp(chunk_type, payload):
    chunk = chunk_type + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", 4 + len(chunk)) + b"WEBP" + chunk


def _ffmpeg_sample(codec, muxer, size="640x480", pix_fmt=None):
    args = [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c=red:s={size}", "-frames:v", "1"]
    if pix_fmt:
        args += ["-pix_fmt", pix_fmt]
    args += ["-c:v", codec, "-f", muxer, "pipe:1"]
    return subprocess.run(args, capture_output=True, check=True, timeout=60).stdout


class ImageProbeTests(unittest.TestCase):
    def assert_probe(self, data, image_format, width, height):
        info = target._probe_image(data)
        self.assertIsNotNone(info)
        self.assertEqual((info["format"], info["width"], info["height"]), (image_format, width, height))
        return info

    def test_png_header_and_rgba_flag(self):
        info = self.assert_probe(_rgba_png(3, 2), "PNG", 3, 2)
        self.assertTrue(info["rgba8"])
        rgb = target._encode_png(3, 2, 2, 8, [b"\x00" * 9] * 2)
        self.assertFalse(target._probe_image(rgb)["rgba8"])

    def test_jpeg_skips_leading_segments_until_sof(self):
        jpeg = (
            b"\xff\xd8"
            + b"\xff\xe1\x00\x06Exif"
            + b"\xff\xc2\x00\x11\x08\x00\x07\x00\x0b"
            + b"\x00" * 12
        )
        self.assert_probe(jpeg, "JPEG", 11, 7)

    def test_gif_bmp_and_webp_headers(self):
        self.assert_probe(b"GIF89a" + struct.pack("<HH", 7, 5) + b"\x00" * 8, "GIF", 7, 5)
        bmp = b"BM" + b"\x00" * 12 + struct.pack("<Iii", 40, 9, -4) + b"\x00" * 8
        self.assert_probe(bmp, "BMP", 9, 4)
        lossless = _riff_webp(b"VP8L", b"\x2f" + struct.pack("<I", (12 - 1) | ((6 - 1) << 14)) + b"\x00" * 8)
        self.assert_probe(lossless, "WEBP", 12, 6)
        lossy = _riff_webp(b"VP8 ", b"\x00\x00\x00\x9d\x01\x2a" + struct.pack("<HH", 30, 20) + b"\x00" * 8)
        self.assert_probe(lossy, "WEBP", 30, 20)

    def test_animated_webp_is_flagged_and_never_sent_to_tools(self):
        extended = _riff_webp(b"VP8X", bytes([0x02, 0, 0, 0]) + (99).to_bytes(3, "little") + (49).to_bytes(3, "little"))
        info = self.assert_probe(extended, "WEBP", 100, 50)
        self.assertTrue(info["animated"])
        with mock.patch.object(target, "_run_image_command") as run:
            self.assertIsNone(target._make_thumbnail_webp(extended, 320, 78))
            self.assertIsNone(target._convert_image_to_png(extended))
        run.assert_not_called()

    def test_avif_and_heic_dimensions_come_from_ispe(self):
        self.assert_probe(TINY_AVIF_16X8, "AVIF", 16, 8)
        heic = TINY_AVIF_16X8.replace(b"ftypavif", b"ftypheic", 1).replace(b"avifmif1", b"heicmif1", 1)
        self.assert_probe(heic, "HEIC", 16, 8)

    def test_unknown_and_truncated_data_are_rejected(self):
        for data in (b"", b"not an image at all", _rgba_png(3, 2)[:20], b"\xff\xd8\xff\xda\x00\x02", None, "text"):
            with self.subTest(data=data):
                self.assertIsNone(target._probe_image(data))

    def test_pixel_limit_blocks_work_before_any_command(self):
        huge = target._PNG_SIGNATURE + b"\x00\x00\x00\x0dIHDR" + struct.pack(">IIBBBBB", 10000, 5000, 8, 6, 0, 0, 0) + b"\x00" * 8
        info = target._probe_image(huge)
        self.assertFalse(target._image_pixels_ok(info))
        with mock.patch.object(target, "_run_image_command") as run:
            self.assertIsNone(target._make_thumbnail_webp(huge, 320, 78))
            self.assertIsNone(target._convert_image_to_png(huge))
        run.assert_not_called()

    def test_png_structure_validation_checks_crc(self):
        png = _rgba_png(4, 4)
        info = target._probe_image(png)
        self.assertTrue(target._validate_image_structure(png, info))
        idat = png.index(b"IDAT")
        corrupted = png[: idat + 6] + bytes([png[idat + 6] ^ 0xFF]) + png[idat + 7:]
        self.assertFalse(target._validate_image_structure(corrupted, info))
        self.assertFalse(target._validate_image_structure(png[:-12], info))


class QrCodeTests(unittest.TestCase):
    def test_totp_qr_is_a_one_bit_png_with_white_quiet_zone(self):
        import qrcode

        uri = "otpauth://totp/AI%20Chat%20Playground:user?secret=JBSWY3DPEHPK3PXP&issuer=AI%20Chat%20Playground"
        png = target._qr_png_bytes(uri)
        info = target._probe_image(png)
        qr = qrcode.QRCode(box_size=10, border=4)
        qr.add_data(uri)
        qr.make(fit=True)
        expected_side = len(qr.get_matrix()) * 10
        self.assertEqual((info["format"], info["width"], info["height"]), ("PNG", expected_side, expected_side))
        self.assertEqual(png[24:26], b"\x01\x00")
        self.assertTrue(target._validate_image_structure(png, info))

        idat_start = png.index(b"IDAT") + 4
        idat_length = int.from_bytes(png[idat_start - 8:idat_start - 4], "big")
        raw = zlib.decompress(png[idat_start:idat_start + idat_length])
        row_length = 1 + (expected_side + 7) // 8
        first_row = raw[:row_length]
        self.assertEqual(first_row[0], 0)
        self.assertTrue(all(byte == 0xFF for byte in first_row[1:]))


@unittest.skipUnless(HAS_TOOLS, "cwebp/dwebp/ffmpeg are required")
class ImageCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = {
            "PNG": _ffmpeg_sample("png", "image2pipe", pix_fmt="rgba"),
            "JPEG": _ffmpeg_sample("mjpeg", "image2pipe", pix_fmt="yuvj420p"),
            "WEBP": _ffmpeg_sample("libwebp", "webp"),
            "GIF": _ffmpeg_sample("gif", "gif", pix_fmt="pal8"),
            "BMP": _ffmpeg_sample("bmp", "image2pipe"),
        }

    def test_thumbnails_are_webp_within_the_requested_box(self):
        for image_format, data in self.samples.items():
            with self.subTest(image_format=image_format):
                self.assertEqual(target._probe_image(data)["format"], image_format)
                thumb = target._make_thumbnail_webp(data, 320, 78)
                info = target._probe_image(thumb)
                self.assertEqual((info["format"], info["width"], info["height"]), ("WEBP", 320, 240))

    def test_avif_thumbnail_and_small_images_are_not_upscaled(self):
        thumb = target._make_thumbnail_webp(TINY_AVIF_16X8, 320, 78)
        info = target._probe_image(thumb)
        self.assertEqual((info["format"], info["width"], info["height"]), ("WEBP", 16, 8))

    def test_conversions_keep_dimensions(self):
        for image_format in ("WEBP", "GIF", "BMP"):
            with self.subTest(image_format=image_format):
                png = target._convert_image_to_png(self.samples[image_format])
                info = target._probe_image(png)
                self.assertEqual((info["format"], info["width"], info["height"]), ("PNG", 640, 480))
        avif_png = target._convert_image_to_png(TINY_AVIF_16X8)
        self.assertEqual(target._probe_image(avif_png)["width"], 16)

    def test_mask_conversion_produces_rgba_png(self):
        png = target._convert_image_to_png(self.samples["JPEG"], rgba=True)
        info = target._probe_image(png)
        self.assertEqual(info["format"], "PNG")
        self.assertTrue(info["rgba8"])

    def test_broken_input_returns_none(self):
        broken = self.samples["JPEG"][:200]
        self.assertIsNone(target._convert_image_to_png(broken))


class PillowIsolationTests(unittest.TestCase):
    def test_resident_process_never_loads_pillow_cairo_or_reportlab(self):
        self.assertIn("PIL", sys.modules)
        self.assertIsNone(sys.modules["PIL"])
        self.assertNotIn("cairosvg", sys.modules)
        self.assertNotIn("reportlab", sys.modules)

    def test_server_sources_do_not_import_pillow_dependent_modules(self):
        pattern = re.compile(r"^\s*(?:from|import)\s+(?:PIL|cairosvg|reportlab)\b", re.M)
        sources = [APP_ROOT / "app.py", *sorted((APP_ROOT / "server").glob("*.py"))]
        offenders = [str(path.relative_to(APP_ROOT)) for path in sources if pattern.search(path.read_text(encoding="utf-8"))]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
