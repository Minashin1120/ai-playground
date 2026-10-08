"""chat_core is served as several files (scripts/build_frontend.sh cuts it at `//! @chat-core-bundle-split`).

Classic scripts share their top-level names, so the pieces behave like the single file unless a
piece calls, while it loads, a function declared in a later piece (function declarations are only
hoisted within their own file). The chat page is loaded in Chromium three ways - the single
combined source, the same text cut into pieces, and the minified pieces the page really serves -
and must raise the same errors each time."""
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import urlparse

APP_ROOT = Path(__file__).resolve().parents[1]
ORIGIN = 'https://chat.example.test'
SPLIT_RE = re.compile(r'^[ \t]*//! @chat-core-bundle-split', re.MULTILINE)
BUNDLE_TAG_RE = re.compile(r'<script src="/static/js/chat_core\.min\.[^"]+" defer></script>\s*')


def split_source(source):
    starts = [0] + [m.start() for m in SPLIT_RE.finditer(source)]
    return [source[a:b] for a, b in zip(starts, starts[1:] + [len(source)])]


class ChatCoreBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise unittest.SkipTest('playwright is not installed')
        cls._pw = sync_playwright().start()
        try:
            cls.browser = cls._pw.chromium.launch()
        except Exception as e:
            cls._pw.stop()
            raise unittest.SkipTest(f'Chromium is not available: {e}')

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls._pw.stop()

    def setUp(self):
        import app as target
        self.target = target
        target.app.config.update(TESTING=True, MAINTENANCE_MODE=False, TRUSTED_HOSTS=['localhost'])
        target._ensure_temp_chat_monitor_running = lambda: None
        with target.app.app_context():
            target.db.session.remove()
            target.db.drop_all()
            target.db.create_all()
            user = target.User(username='bundle-test', is_setup_completed=True)
            user.set_password('test-password')
            target.db.session.add(user)
            target.db.session.commit()
            user_id = user.id
        patcher = mock.patch.object(target, '_bot_turnstile_active', return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        client = target.app.test_client()
        with client.session_transaction() as sess:
            sess['_user_id'] = str(user_id)
            sess['_fresh'] = True
        response = client.get('/', base_url='https://localhost')
        self.assertEqual(response.status_code, 200)
        self.html = response.get_data(as_text=True)
        self.version = target.app.config['SYSTEM_VERSION'].lower()
        self.source = (APP_ROOT / f'static/js/chat_core.{self.version}.js').read_text(encoding='utf-8')

    def bundle_tags(self):
        tags = BUNDLE_TAG_RE.findall(self.html)
        return tags

    def page_errors(self, html, extra_files):
        context = self.browser.new_context()
        page = context.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error).splitlines()[0]))

        def handle(route):
            url = urlparse(route.request.url)
            if f'{url.scheme}://{url.netloc}' != ORIGIN:
                return route.abort()
            if url.path == '/':
                return route.fulfill(status=200, content_type='text/html', body=html)
            if url.path in extra_files:
                return route.fulfill(status=200, content_type='application/javascript', body=extra_files[url.path])
            if url.path.startswith('/static/'):
                path = (APP_ROOT / url.path.lstrip('/')).resolve()
                if path.is_file() and (APP_ROOT / 'static') in path.parents:
                    return route.fulfill(status=200, path=str(path))
                return route.fulfill(status=404, body='')
            return route.fulfill(status=200, content_type='application/json', body='{}')

        page.route('**/*', handle)
        page.goto(ORIGIN + '/', wait_until='load')
        page.wait_for_timeout(1500)
        defined = page.evaluate("['sendMessage', 'loadMessages', 'updateFilePreview', 'hideModal']"
                                ".filter(name => typeof window[name] === 'function')")
        context.close()
        return errors, defined

    def with_scripts(self, paths):
        tags = self.bundle_tags()
        first = self.html.index(tags[0])
        html = BUNDLE_TAG_RE.sub('', self.html)
        scripts = ''.join(f'<script src="{p}" defer></script>\n' for p in paths)
        return html[:first] + scripts + html[first:]

    def test_split_files_behave_like_the_single_file(self):
        tags = self.bundle_tags()
        pieces = split_source(self.source)
        self.assertEqual(len(tags), len(pieces), tags)
        self.assertGreaterEqual(len(pieces), 2)

        single_errors, single_defined = self.page_errors(
            self.with_scripts(['/single.js']), {'/single.js': self.source})
        piece_files = {f'/piece{i}.js': text for i, text in enumerate(pieces, 1)}
        piece_errors, piece_defined = self.page_errors(self.with_scripts(list(piece_files)), piece_files)
        served_errors, served_defined = self.page_errors(self.html, {})

        expected = ['sendMessage', 'loadMessages', 'updateFilePreview', 'hideModal']
        # No variant may raise: a hoisting break shows up as "… is not defined", a global name
        # clash between scripts as "Identifier … has already been declared".
        self.assertEqual((single_errors, single_defined), ([], expected))
        self.assertEqual((piece_errors, piece_defined), ([], expected))
        self.assertEqual((served_errors, served_defined), ([], expected))

    def test_standalone_scripts_keep_their_helpers_inside(self):
        # esbuild's --keep-names helper must not become a global (pwa_install.min.js once declared
        # `var p`, which clashed with chat_core's `let p` and kept the whole script from running).
        for name in ('activity_log', 'progress_spinner', 'connection_monitor', 'pwa_install', 'landing_demo'):
            text = (APP_ROOT / f'static/js/{name}.min.js').read_text(encoding='utf-8')
            self.assertTrue(text.startswith('(()=>{'), name)
        first = (APP_ROOT / f'static/js/chat_core.min.{self.version}.1.js').read_text(encoding='utf-8')
        later = [(APP_ROOT / f'static/js/chat_core.min.{self.version}.{i}.js').read_text(encoding='utf-8')
                 for i in range(2, len(split_source(self.source)) + 1)]
        self.assertIn('Object.defineProperty', first.split(';', 1)[0])
        for text in later:
            self.assertNotIn('=Object.defineProperty;var ', text[:200])


if __name__ == '__main__':
    unittest.main()
