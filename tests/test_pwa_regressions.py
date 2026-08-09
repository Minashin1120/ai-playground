import json
from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = APP_ROOT / "static" / "manifest.webmanifest"
PWA_INSTALL_PATH = APP_ROOT / "static" / "js" / "pwa_install.js"


class PwaRegressionTests(unittest.TestCase):
    def test_manifest_locks_pwa_to_portrait_orientation(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

        self.assertEqual(manifest.get("display"), "standalone")
        self.assertEqual(manifest.get("orientation"), "portrait-primary")

    def test_installed_pwa_reapplies_runtime_orientation_lock(self):
        source = PWA_INSTALL_PATH.read_text(encoding="utf-8")

        self.assertIn("const lockPwaOrientation = async () =>", source)
        self.assertIn("if (!isStandalone()", source)
        self.assertIn("await orientation.lock('portrait-primary')", source)
        self.assertIn("document.addEventListener('visibilitychange'", source)
        self.assertIn("window.addEventListener('pageshow', lockPwaOrientation)", source)


if __name__ == "__main__":
    unittest.main()
