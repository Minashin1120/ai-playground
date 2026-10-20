"""Render the Android preview shell in Chromium, using synthetic HTML only."""
import json
import re
import shutil
import unittest
from pathlib import Path
from playwright.sync_api import sync_playwright

ANDROID = Path(__file__).resolve().parents[1]
SOURCE = ANDROID / 'app/src/main/java/com/minashin1120/aiplayground/ui/HtmlCodePreviewDialog.kt'


class HtmlPreviewTests(unittest.TestCase):
    def test_html_css_scripts_and_sandbox_survive_shell_loading(self):
        source = SOURCE.read_text()
        template = re.search(r'"""(<!doctype html>.*?)""".trimIndent', source, re.S)[1]
        security = (ANDROID / 'app/src/main/assets/html-preview-security.js').read_text()
        code = '''<!doctype html><html><head><style>h1 { color: #123456; width: 100%; }</style></head><body>
        <h1>日本語のプレビュー 😀</h1><a href="/example">relative link</a>
        <script>document.querySelector('h1').dataset.ready='yes';
        try { parent.document.body.dataset.escaped='yes'; } catch(e) { document.body.dataset.isolated='yes'; }
        </script><script src="https://cdn.polyfill.io/v3/polyfill.min.js"></script></body></html>'''
        def quoted(value):
            return json.dumps(value, ensure_ascii=False).replace('<', r'\u003c').replace('\u2028', r'\u2028').replace('\u2029', r'\u2029')
        shell = template.replace('${scriptString(base)}', quoted('https://example.com/')).replace('$security', security).replace('${scriptString(code)}', quoted(code))
        errors = []
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=shutil.which('chromium'))
            page = browser.new_page()
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.route('https://html-preview.invalid/preview', lambda route: route.fulfill(body=shell, content_type='text/html; charset=utf-8'))
            page.goto('https://html-preview.invalid/preview')
            frame = page.frame_locator('#preview')
            frame.locator('h1[data-ready="yes"]').wait_for()
            self.assertEqual(frame.locator('h1').inner_text(), '日本語のプレビュー 😀')
            self.assertEqual(frame.locator('h1').evaluate('(el) => getComputedStyle(el).color'), 'rgb(18, 52, 86)')
            self.assertEqual(frame.locator('a').evaluate('(el) => el.href'), 'https://example.com/example')
            self.assertEqual(frame.locator('body').get_attribute('data-isolated'), 'yes')
            self.assertIsNone(page.locator('body').get_attribute('data-escaped'))
            self.assertEqual(frame.locator('script[src*="polyfill.io"]').count(), 0)
            self.assertEqual(frame.locator('[data-blocked-script]').count(), 1)
            self.assertEqual(page.locator('#status').count(), 0)
            self.assertEqual(errors, [])
            browser.close()


if __name__ == '__main__':
    unittest.main()
