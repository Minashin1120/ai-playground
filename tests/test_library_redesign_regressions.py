from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


def _current_asset(folder, pattern):
    assets = sorted((APP_ROOT / "static" / folder).glob(pattern))
    assert len(assets) == 1, f"Expected only the latest asset for {pattern}"
    return assets[0].read_text(encoding="utf-8")


class LibraryRedesignRegressionTests(unittest.TestCase):
    def test_template_has_modern_library_modal_structure(self):
        template = Path(APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        # Required IDs must remain for the JS to bind behavior.
        for ident in (
            "id=\"lib-modal\"",
            "id=\"lib-grid\"",
            "id=\"lib-search\"",
            "id=\"lib-sort\"",
            "id=\"lib-total-count\"",
            "id=\"lib-attach-btn\"",
            "id=\"lib-download-btn\"",
            "id=\"lib-rename-btn\"",
            "id=\"lib-del-btn\"",
        ):
            self.assertIn(ident, template, f"Missing library element: {ident}")
        # The redesigned modal must use the modern design-system classes.
        for cls in (
            "lib-modal-panel",
            "lib-modal-header",
            "lib-modal-title",
            "lib-toolbar",
            "lib-search-box",
            "lib-action-btn",
            "lib-grid-responsive",
        ):
            self.assertIn(cls, template, f"Missing modern library class: {cls}")
        # The old Tailwind-heavy card grid style must be gone.
        self.assertNotIn("grid-template-columns:repeat(auto-fit,minmax(160px,1fr))", template)

    def test_js_has_skeleton_loading_and_modern_card(self):
        script = _current_asset("js", "chat_core.v4.8.*.js")
        # Skeleton loading screen is rendered while the library fetch is in flight.
        self.assertIn("function renderLibrarySkeleton(grid)", script)
        self.assertIn("lib-skeleton-card", script)
        self.assertIn("lib-skeleton-thumb", script)
        self.assertIn("renderLibrarySkeleton(grid)", script)
        # The card is drawn with the modern markup and a teal selection state.
        self.assertIn("className = 'library-thumb-card'", script)
        self.assertIn("classList.add('is-selected')", script)
        self.assertIn("lib-thumb-media-wrap", script)
        self.assertIn("lib-thumb-bar", script)
        self.assertIn("function libraryFileIcon(ext)", script)
        # Legacy card visuals are no longer applied to library items.
        self.assertNotIn("model-list-animate overflow-hidden", script)
        self.assertNotIn("classList.remove('ring-2', 'ring-blue-500', 'border-blue-500')", script)
        # The entry-point call replaces the old "読み込み中..." placeholder.
        self.assertNotIn("grid.innerHTML = '<div class=\"text-xs text-gray-500\">読み込み中...</div>'", script)
        # Empty / no-result / error states render as styled empty states.
        self.assertIn("lib-empty-state", script)
        self.assertIn("lib-empty-icon", script)

    def test_css_has_modern_library_and_skeleton_shimmer(self):
        css = _current_asset("css", "chat.custom.v4.8.*.css")
        for cls in (
            ".lib-modal-panel",
            ".lib-modal-header",
            ".lib-toolbar",
            ".lib-search-box",
            ".library-thumb-card",
            ".lib-thumb-media-wrap",
            ".lib-thumb-bar",
            ".lib-empty-state",
            ".lib-skeleton-card",
            ".lib-skeleton-thumb",
            ".lib-skeleton-line",
            ".library-thumb-card.is-selected",
        ):
            self.assertIn(cls, css, f"Missing modern library CSS rule: {cls}")
        self.assertIn("@keyframes libShimmer", css)
        self.assertIn("animation: libShimmer 1.4s infinite", css)
        # Skeleton cards participate in the staggered entrance animation.
        self.assertIn("#lib-grid .lib-skeleton-card", css)
        # Reduced motion must disable the shimmer and entrance animations.
        self.assertIn("#lib-grid .lib-skeleton-card,", css)
        self.assertIn(".lib-skeleton-thumb::after,", css)
        self.assertIn(".lib-skeleton-line::after,", css)

    def test_mobile_selecting_hides_card_actions(self):
        css = _current_asset("css", "chat.custom.v4.8.*.css")
        # On touch screens the top-right actions are always visible unless the
        # modal is in multi-select mode, where they must hide.
        self.assertIn("@media (max-width: 768px)", css)
        self.assertIn(".lib-thumb-actions", css)
        self.assertIn("opacity: 1", css)
        self.assertIn("#lib-modal.lib-selecting #lib-grid .lib-thumb-actions", css)


if __name__ == "__main__":
    unittest.main()
