from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


def _current_asset(folder, pattern):
    assets = sorted((APP_ROOT / "static" / folder).glob(pattern))
    assert len(assets) == 1, f"Expected only the latest asset for {pattern}"
    return assets[0].read_text(encoding="utf-8")


class ImageViewerSwipeRegressionTests(unittest.TestCase):
    def test_swipe_handlers_and_binding_exist(self):
        # The enlarged image viewer must support horizontal swipe navigation
        # between grouped images (chat messages, library, attachment thumbs).
        script = _current_asset("js", "chat_core.v4.8.*.js")

        self.assertIn("let viewerSwipe = null", script)
        self.assertIn("function onViewerTouchStart(e)", script)
        self.assertIn("function onViewerTouchMove(e)", script)
        self.assertIn("function onViewerTouchEnd()", script)
        self.assertIn("function finishSwipeNav(dir)", script)
        self.assertIn("function getViewerAdjacent(dir)", script)
        self.assertIn("function openViewerWithItems(items, index)", script)
        self.assertIn("function renderViewerChrome()", script)

        # Touch handlers must be attached to the viewer content area.
        self.assertIn(
            "viewerContentEl.addEventListener('touchstart', onViewerTouchStart, { passive: false })",
            script,
        )
        self.assertIn(
            "viewerContentEl.addEventListener('touchmove', onViewerTouchMove, { passive: false })",
            script,
        )
        self.assertIn(
            "viewerContentEl.addEventListener('touchend', onViewerTouchEnd)",
            script,
        )
        self.assertIn(
            "viewerContentEl.addEventListener('touchcancel', onViewerTouchEnd)",
            script,
        )

    def test_swipe_direction_and_adjacent_image(self):
        script = _current_asset("js", "chat_core.v4.8.*.js")

        # Swiping right goes to the previous image, swiping left to the next.
        self.assertIn("viewerSwipe.dir = dx > 0 ? -1 : 1", script)
        self.assertIn("const dir = swipe.dir || (swipe.dx > 0 ? -1 : 1)", script)
        # A drag toward an edge (first/last image) bounces back instead of
        # navigating: the current image follows with resistance.
        self.assertIn("viewerSwipe.resist = true", script)
        self.assertIn("const effDx = viewerSwipe.resist ? dx * 0.3 : dx", script)
        # The adjacent image is revealed behind the current one during the drag.
        self.assertIn("adj.className = 'viewer-adjacent'", script)
        self.assertIn("adj.style.transform = `translate(-50%, -50%) translateX(${adjDir * stageWidth + dx}px) scale(0.97)`", script)

    def test_swipe_settle_respects_reduced_motion(self):
        script = _current_asset("js", "chat_core.v4.8.*.js")

        self.assertIn("window.matchMedia('(prefers-reduced-motion: reduce)')", script)
        self.assertIn("if (reducedMotion) {\n                finishSwipeNav(dir);", script)

    def test_close_click_suppressed_after_swipe(self):
        script = _current_asset("js", "chat_core.v4.8.*.js")

        self.assertIn("let suppressViewerCloseClick = false", script)
        self.assertIn("suppressViewerCloseClick = true", script)
        self.assertIn(
            "if (suppressViewerCloseClick) { suppressViewerCloseClick = false; return; }",
            script,
        )

    def test_library_images_open_as_grouped_viewer(self):
        script = _current_asset("js", "chat_core.v4.8.*.js")

        # The library open button must route image files to the grouped viewer
        # (swipeable across library images) while other files keep the file viewer.
        self.assertIn("function openLibraryImage(f)", script)
        self.assertIn("if (f.type === 'image') {\n                        openLibraryImage(f);", script)
        self.assertIn("const images = filtered.filter((x) => x.type === 'image')", script)
        self.assertIn("openViewerWithItems(items, idx)", script)
        self.assertIn("const filtered = q ? ordered.filter((x) => fileNameForSearch(x).includes(q)) : ordered", script)

    def test_swipe_css_and_adjacent_overlay(self):
        css = _current_asset("css", "chat.custom.v4.8.*.css")
        script = _current_asset("js", "chat_core.v4.8.*.js")

        # The viewer content area must hand horizontal gestures to JS.
        self.assertIn("touch-action: pan-y;", css)
        # The adjacent image is an absolutely-positioned overlay centered on the
        # same spot as the main image so slides stay aligned.
        self.assertIn(".viewer-adjacent {", css)
        self.assertIn("position: absolute;", css)
        self.assertIn("pointer-events: none;", css)
        self.assertIn("will-change: transform, opacity;", css)
        # Reduced-motion CSS must also cover the image and its adjacent overlay.
        self.assertIn("#image-viewer-img,", css)
        self.assertIn("#image-viewer .viewer-adjacent,", css)

    def test_chat_template_viewer_structure(self):
        template = (APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")

        self.assertIn('<div id="image-viewer">', template)
        self.assertIn('<img id="image-viewer-img" alt="image preview">', template)
        self.assertIn('class="viewer-content"', template)
        self.assertIn('onclick="navImage(-1)"', template)
        self.assertIn('onclick="navImage(1)"', template)


if __name__ == "__main__":
    unittest.main()
