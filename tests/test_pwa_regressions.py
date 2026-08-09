import json
from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = APP_ROOT / "static" / "manifest.webmanifest"


class PwaRegressionTests(unittest.TestCase):
    def test_manifest_locks_pwa_to_portrait_orientation(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

        self.assertEqual(manifest.get("display"), "standalone")
        self.assertEqual(manifest.get("orientation"), "portrait")


if __name__ == "__main__":
    unittest.main()
