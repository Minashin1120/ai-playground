import unittest
from pathlib import Path
from tests.app_source import read_chat_core_source

ROOT = Path(__file__).resolve().parents[1]
PARTS_DIR = ROOT / "static" / "js" / "chat_core_parts"


class DropOverlayRegressionTests(unittest.TestCase):
    """Ensure drop-overlay is properly suppressed and dismissed during upload modal interactions."""

    def test_upload_modal_suppresses_drop_overlay(self):
        part8 = read_chat_core_source()
        self.assertIn("isUploadModalOpen", part8)
        self.assertIn("showDropOverlay = () => {", part8)
        self.assertIn("if (isUploadModalOpen()) return;", part8)

    def test_drop_handlers_reset_drop_overlay(self):
        part8 = read_chat_core_source()
        # hideDropOverlay must reset dragCounter
        self.assertIn("const hideDropOverlay = () => {", part8)
        self.assertIn("dragCounter = 0;", part8)
        # dropzone drop listener must call hideDropOverlay
        self.assertIn("dropzone.addEventListener('drop'", part8)
        # window drop listener must call hideDropOverlay before handling
        self.assertIn("window.addEventListener('drop'", part8)
        self.assertIn("window.addEventListener('dragend'", part8)

    def test_upload_modal_open_close_dismisses_drop_overlay(self):
        part12 = read_chat_core_source()
        self.assertIn("function openUploadModal()", part12)
        self.assertIn("function closeUploadModal(", part12)
        self.assertIn("window.hideDropOverlay", part12)


if __name__ == "__main__":
    unittest.main()
