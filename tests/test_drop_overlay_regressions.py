import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTS_DIR = ROOT / "static" / "js" / "chat_core_parts"


class DropOverlayRegressionTests(unittest.TestCase):
    """Ensure drop-overlay is properly suppressed and dismissed during upload modal interactions."""

    def test_upload_modal_suppresses_drop_overlay(self):
        part8 = (PARTS_DIR / "chat_core.part08_domcontent_account_transfer.js").read_text(encoding="utf-8")
        self.assertIn("isUploadModalOpen", part8)
        self.assertIn("showDropOverlay = () => {", part8)
        self.assertIn("if (isUploadModalOpen()) return;", part8)

    def test_drop_handlers_reset_drop_overlay(self):
        part8 = (PARTS_DIR / "chat_core.part08_domcontent_account_transfer.js").read_text(encoding="utf-8")
        # hideDropOverlay must reset dragCounter
        self.assertIn("const hideDropOverlay = () => {", part8)
        self.assertIn("dragCounter = 0;", part8)
        # dropzone drop listener must call hideDropOverlay
        self.assertIn("dropzone.addEventListener('drop'", part8)
        # window drop listener must call hideDropOverlay before handling
        self.assertIn("window.addEventListener('drop'", part8)
        self.assertIn("window.addEventListener('dragend'", part8)

    def test_upload_modal_open_close_dismisses_drop_overlay(self):
        part11 = (PARTS_DIR / "chat_core.part11_file_preview_upload.js").read_text(encoding="utf-8")
        self.assertIn("function openUploadModal()", part11)
        self.assertIn("function closeUploadModal(", part11)
        self.assertIn("window.hideDropOverlay", part11)


if __name__ == "__main__":
    unittest.main()
