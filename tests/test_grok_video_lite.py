"""Exercise the worker's request construction without loading the app or calling xAI."""
import base64
from pathlib import Path
import textwrap
import unittest


SOURCE = (Path(__file__).resolve().parents[1] / "server/background.py").read_text()
VIDEO_BRANCH = SOURCE.split("# --- 1.6 Grok Imagine Video Generation ---", 1)[1]
REQUEST_SOURCE = textwrap.dedent(
    VIDEO_BRANCH.split("# Prepare params\n", 1)[1].split("# Send request", 1)[0]
)


class GrokVideoLiteTests(unittest.TestCase):
    def payload(self, resolution, images=(), model="grok-imagine-video-1.5-lite"):
        namespace = {
            "model_key": model,
            "options": {
                "grok_video_duration": "15",
                "grok_video_aspect": "9:16",
                "grok_video_resolution": resolution,
                # Lite must never forward full-model reference controls.
                "reference_images": ["https://example.com/ref.png"],
                "reference_audios": [{"voice": "ursa"}],
                "last_frame": {"url": "https://example.com/last.png"},
                "keyframes": [{"timestamp": 3}],
            },
            "key": "test-key",
            "_XAI_API_HOST": "api.x.ai",
            "final_message_text": "Animate a sunrise",
            "loaded_files": list(images),
            "base64": base64,
            "_probe_image": lambda _: None,
        }
        exec(REQUEST_SOURCE, namespace)
        self.assertEqual("https://api.x.ai/v1/videos/generations", namespace["endpoint"])
        return namespace["payload"]

    def test_text_to_video_accepts_all_lite_resolutions_without_reference_controls(self):
        for resolution in ("480p", "720p", "1080p"):
            with self.subTest(resolution=resolution):
                self.assertEqual({
                    "model": "grok-imagine-video-1.5-lite",
                    "prompt": "Animate a sunrise",
                    "duration": 15,
                    "aspect_ratio": "9:16",
                    "resolution": resolution,
                }, self.payload(resolution))

    def test_image_to_video_sends_a_single_data_uri(self):
        payload = self.payload("1080p", [{"bytes": b"image", "mime": "image/png"}])
        self.assertEqual({"url": "data:image/png;base64,aW1hZ2U="}, payload["image"])
        self.assertEqual("1080p", payload["resolution"])

    def test_legacy_1080p_fallback_is_preserved(self):
        self.assertEqual("720p", self.payload("1080p", model="grok-imagine-video")["resolution"])
        self.assertEqual("1080p", self.payload("1080p", model="grok-imagine-video-1.5")["resolution"])
