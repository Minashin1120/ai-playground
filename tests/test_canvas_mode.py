from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


class CanvasModeRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assets = list((APP_ROOT / "static/js").glob("chat_core.v4.8.*.js"))
        if len(assets) != 1:
            raise AssertionError(f"expected one versioned chat JS asset, got {assets}")
        cls.source = assets[0].read_text(encoding="utf-8")
        cls.template = (APP_ROOT / "templates/chat.html").read_text(encoding="utf-8")

    def test_default_canvas_source_is_the_last_code_block(self):
        self.assertIn("let nextIndex = blocks.length - 1;", self.source)
        self.assertIn("selectionMode: 'auto'", self.source)
        self.assertIn("canvasPreviewState.selectionMode === 'manual'", self.source)
        self.assertNotIn("for (let i = list.length - 1; i >= 0; i--)", self.source)
        self.assertIn("const fenceStartRe = /^(\\s*)(`{3,}|~{3,})(.*)$/;", self.source)

    def test_source_tab_can_override_the_default_block(self):
        self.assertIn('id="canvas-source-select"', self.template)
        self.assertIn("renderCanvasSourceOptions()", self.source)
        self.assertIn("applyCanvasSelection(index, { view: 'source' })", self.source)
        self.assertIn("canvasPreviewState.selectionMode = 'manual';", self.source)

    def test_stream_updates_preserve_canvas_scroll(self):
        self.assertIn("instrumentCanvasPreviewDocument", self.source)
        self.assertIn("canvas-preview-scroll", self.source)
        self.assertIn("canvas-preview-restore-scroll", self.source)
        self.assertIn("const sourceScrollTop = els.sourceScroll ? els.sourceScroll.scrollTop", self.source)
        refresh_start = self.source.index("function refreshCanvasPreviewPanel()")
        refresh_end = self.source.index("function applyCanvasSelection(", refresh_start)
        refresh_source = self.source[refresh_start:refresh_end]
        self.assertNotIn("els.sourceScroll.scrollTop = 0", refresh_source)

    def test_block_tab_uses_full_width_rows_with_code_previews(self):
        css_assets = list((APP_ROOT / "static/css").glob("chat.custom.v4.8.*.css"))
        self.assertEqual(len(css_assets), 1)
        css_source = css_assets[0].read_text(encoding="utf-8")

        self.assertIn("flex-direction: column;", css_source)
        self.assertIn(".canvas-block-chip-main", css_source)
        self.assertIn(".canvas-block-chip-preview", css_source)
        self.assertIn("width: 100%;", css_source)
        self.assertIn("const firstCodeLine =", self.source)
        self.assertIn('class="canvas-block-chip-main"', self.source)
        self.assertIn('class="canvas-block-chip-preview"', self.source)

    def test_block_selection_animates_into_the_preview_tab(self):
        css_assets = list((APP_ROOT / "static/css").glob("chat.custom.v4.8.*.css"))
        self.assertEqual(len(css_assets), 1)
        css_source = css_assets[0].read_text(encoding="utf-8")

        self.assertIn("animateCanvasMobileViewEntry", self.source)
        self.assertIn("animateView: true", self.source)
        self.assertIn("transitionFrom: 'blocks'", self.source)
        self.assertIn("canvas-view-enter-from-left", self.source)
        self.assertIn("@keyframes canvasViewEnterFromLeft", css_source)
        self.assertIn(".canvas-view-enter-from-left", css_source)


if __name__ == "__main__":
    unittest.main()
