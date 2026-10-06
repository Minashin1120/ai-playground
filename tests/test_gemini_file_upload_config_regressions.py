import re
import unittest

from google.genai import types

from tests.app_source import read_app_source

APP_SOURCE = read_app_source()


class GeminiFileUploadConfigRegressionTests(unittest.TestCase):
    """Videos and large files go to the Gemini Files API; a rejected config fails every such send."""

    def _upload_block(self):
        match = re.search(r'config = \{"mime_type": mime\}.*?g_client\.files\.upload\(file=tmp\.name, config=config\)', APP_SOURCE, re.S)
        self.assertIsNotNone(match, "Gemini Files API upload config not found")
        return match.group(0)

    def test_upload_config_gives_each_field_once(self):
        block = self._upload_block()
        self.assertNotIn('"displayName"', block)
        self.assertNotIn('"mimeType"', block)
        self.assertNotIn('config["name"]', block)

    def test_sdk_accepts_the_upload_config(self):
        types.UploadFileConfig.model_validate({"mime_type": "video/mp4", "display_name": "clip.mp4"})
        with self.assertRaises(Exception):
            types.UploadFileConfig.model_validate({"mimeType": "video/mp4", "display_name": "clip.mp4", "displayName": "clip.mp4"})


if __name__ == "__main__":
    unittest.main()
