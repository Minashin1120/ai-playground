"""Gemini input token estimates must count attached images, PDFs, audio and video.

Gemini receives these attachments as media, so `/api/token_estimate` reported
them as "非テキスト" and counted 0 tokens. The estimate now applies Gemini's
fixed media rates (Gemini 3 media_resolution defaults, legacy rates for 2.x).
"""

import io
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("FLASK_SECRET_KEY", "gemini-media-token-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-gemini-media-token-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target

FFMPEG = shutil.which("ffmpeg")


def _ffmpeg(args):
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", *args, "pipe:1"]
    return subprocess.run(cmd, capture_output=True, check=True, timeout=60).stdout


def _pdf_bytes(pages):
    writer = target.pypdf.PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class GeminiMediaTokenRulesTests(unittest.TestCase):
    def test_profile_by_model_generation(self):
        self.assertEqual(target.gemini_media_token_profile("gemini-3.8-flash"), "gemini3")
        self.assertEqual(target.gemini_media_token_profile("gemini-nano-banana-2.1"), "gemini3")
        self.assertEqual(target.gemini_media_token_profile("gemini-2.5-flash"), "legacy")
        self.assertIsNone(target.gemini_media_token_profile("gpt-5"))

    def test_rates(self):
        self.assertEqual(target.gemini_image_tokens("gemini3", 4000, 3000), 1120)
        self.assertEqual(target.gemini_image_tokens("legacy", 300, 200), 258)
        self.assertEqual(target.gemini_image_tokens("legacy", 1000, 800), 258 * 4)
        self.assertEqual(target.gemini_pdf_tokens("gemini3", 3), 1680)
        self.assertEqual(target.gemini_pdf_tokens("legacy", 3), 774)
        self.assertEqual(target.gemini_audio_tokens("gemini3", 10), 250)
        self.assertEqual(target.gemini_audio_tokens("legacy", 10), 320)
        self.assertEqual(target.gemini_video_tokens("gemini3", 10, has_audio=True), 950)
        self.assertEqual(target.gemini_video_tokens("gemini3", 10, has_audio=False), 700)
        self.assertEqual(target.gemini_video_tokens("legacy", 10), 2630)


@unittest.skipUnless(FFMPEG, "ffmpeg is required")
class GeminiAttachmentEstimateTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        target.app.config["UPLOAD_FOLDER"] = self.temp_dir.name
        os.makedirs(os.path.join(self.temp_dir.name, "7"), mode=0o700)
        patcher = mock.patch.object(target, "_token_file_tokens_cache_get", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def put(self, name, data):
        with open(os.path.join(self.temp_dir.name, "7", name), "wb") as handle:
            handle.write(data)
        return f"7/{name}"

    def estimate(self, rel_path, model="gemini-3.8-flash"):
        return target._estimate_attachment_prompt_tokens(rel_path, model_key=model)

    def assert_tokens(self, result, tokens):
        self.assertTrue(result.get("countable"), result)
        self.assertEqual(result.get("tokens"), tokens)

    def test_image(self):
        rel = self.put("photo.png", target._encode_png(1000, 800, 2, 8, [bytes(3000)] * 800))
        self.assert_tokens(self.estimate(rel), 1120)
        self.assert_tokens(self.estimate(rel, "gemini-2.5-flash"), 258 * 4)
        self.assertEqual(self.estimate(rel, "gpt-5").get("reason"), "non_text")

    def test_pdf_counts_pages(self):
        rel = self.put("doc.pdf", _pdf_bytes(3))
        self.assert_tokens(self.estimate(rel), 1680)

    def test_audio_wav_and_m4a(self):
        tone = ["-f", "lavfi", "-i", "sine=frequency=440:duration=4"]
        wav = self.put("voice.wav", _ffmpeg([*tone, "-f", "wav"]))
        self.assert_tokens(self.estimate(wav), 100)
        m4a = self.put("voice.m4a", _ffmpeg([*tone, "-c:a", "aac", "-f", "ipod", "-movflags", "frag_keyframe+empty_moov"]))
        result = self.estimate(m4a)
        self.assertTrue(result.get("countable"), result)
        self.assertIn(result.get("tokens"), (100, 101))

    def test_recorded_webm_without_duration_header(self):
        data = _ffmpeg([
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-c:a", "libopus", "-f", "webm", "-live", "1",
        ])
        result = self.estimate(self.put("rec.webm", data))
        self.assertTrue(result.get("countable"), result)
        self.assertGreaterEqual(result.get("tokens"), 75)

    def test_mp4_video_with_audio(self):
        data = _ffmpeg([
            "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=5",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
            "-shortest", "-c:v", "mpeg4", "-c:a", "aac", "-f", "mp4",
            "-movflags", "frag_keyframe+empty_moov",
        ])
        result = self.estimate(self.put("clip.mp4", data))
        self.assertTrue(result.get("countable"), result)
        # 5 frames at 1 fps (70 each) plus 5 seconds of audio (25 each).
        self.assertGreaterEqual(result.get("tokens"), 5 * 70 + 5 * 25)
        self.assertLessEqual(result.get("tokens"), 6 * 70 + 6 * 25)

    def test_regular_mp4_with_moov_at_end(self):
        out = os.path.join(self.temp_dir.name, "regular.mp4")
        subprocess.run([
            FFMPEG, "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=4", "-c:v", "mpeg4", out,
        ], capture_output=True, check=True, timeout=60)
        with open(out, "rb") as handle:
            data = handle.read()
        self.assertIsNotNone(target._probe_isobmff_media(data))
        # No audio track: only the 4 sampled frames count.
        self.assert_tokens(self.estimate(self.put("regular_copy.mp4", data)), 4 * 70)

    def test_unreadable_media_is_reported_not_zero_counted(self):
        result = self.estimate(self.put("broken.mp3", b"not audio at all" * 10))
        self.assertFalse(result.get("countable"))
        self.assertEqual(result.get("reason"), "media_unknown")


if __name__ == "__main__":
    unittest.main()
