"""数学ステップ解説（Web）の描画・操作・mathsvg・進捗保存の回帰テスト。

Chromium がない環境ではブラウザー部をスキップし、ソース上の契約だけを検証する。
"""
import importlib.util
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import urlparse

APP_ROOT = Path(__file__).resolve().parents[1]
ORIGIN = 'https://chat.example.test'
STEP_TEXT = '\n'.join([
    '## ステップ 1 問題を整理する',
    'まず、わかっていることを書き出します。',
    '',
    '```mathsvg',
    '<svg viewBox="0 0 120 60" xmlns="http://www.w3.org/2000/svg">',
    '<rect x="5" y="5" width="110" height="50" fill="#e0f2fe"/>',
    '</svg>',
    '```',
    '',
    '## ステップ 2 式を立てる',
    '次に式を作ります。$x + 2 = 5$',
    '',
    '## ステップ 3 答えを確認する',
    '最後に代入して確かめます。',
])


def _read(relative):
    return (APP_ROOT / relative).read_text(encoding='utf-8')


class MathStepsSourceContractTests(unittest.TestCase):
    def test_new_part_is_last_and_wired(self):
        parts = sorted((APP_ROOT / 'static/js/chat_core_parts').glob('chat_core.part*.js'))
        self.assertTrue(parts[-1].name.startswith('chat_core.part30_'), parts[-1].name)
        source = _read('static/js/chat_core_parts/chat_core.part30_mathsteps.js')
        for name in ('parseMathSteps', 'buildMathStepsHtml', 'renderMathStepsInto',
                     'buildMathSvgRenderHtml', 'buildMathStepsPdfHtml', 'prepareMathStepsCopyText',
                     'openMathStepsModal', 'saveMathStepsConfig', 'applyMathStepsChatDefaults'):
            self.assertIn(f'window.{name} =', source, name)

    def test_mathsvg_stays_image_only_and_svg_is_unchanged(self):
        renderer = _read('static/js/chat_core_parts/chat_core.part12_toggleoptions.js')
        self.assertIn("lowerLang === 'mathsvg'", renderer)
        self.assertIn('window.buildMathSvgRenderHtml', renderer)
        self.assertIn("token !== 'svg' && token !== 'xml' && token !== 'image/svg+xml' && token !== 'mathsvg'",
                      _read('static/js/chat_core_parts/chat_core.part20_initmarkercanvas.js'))

    def test_settings_and_composer_controls_exist(self):
        settings = _read('templates/chat/overlay_settings.html')
        self.assertIn('id="set-math-steps-enabled"', settings)
        self.assertIn('id="set-default-math-steps"', settings)
        controls = _read('templates/chat/composer_controls.html')
        self.assertIn('id="enable-math-steps"', controls)
        self.assertIn('id="math-steps-container"', controls)
        self.assertIn('openMathStepsModal()', controls)
        modal = _read('templates/chat/overlay_dialogs.html')
        for control in ('math-steps-config-modal', 'math-steps-cfg-basics', 'math-steps-cfg-level',
                        'math-steps-cfg-step-size', 'math-steps-cfg-diagrams', 'math-steps-cfg-animation'):
            self.assertIn(f'id="{control}"', modal, control)

    def test_message_state_and_send_options_are_passed(self):
        self.assertIn('m.math_steps_state', _read('static/js/chat_core_parts/chat_core.part25_updatethreadhighlighting.js'))
        self.assertIn('math_steps: (typeof window.collectMathStepsSendOption',
                      _read('static/js/chat_core_parts/chat_core.part24_proceed.js'))
        self.assertIn('math_steps: (typeof window.collectMathStepsSendOption',
                      _read('static/js/chat_core_parts/chat_core.part23_selectslashcommand.js'))


@unittest.skipUnless(importlib.util.find_spec('playwright'), 'playwright is not installed')
class MathStepsBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import sync_playwright
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
            user = target.User(username='math-steps-test', is_setup_completed=True)
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
        self.html = response.get_data(as_text=True).replace('"globalEnabled": true', '"globalEnabled": false')

    def assertNoPageErrors(self, errors):
        relevant = [e for e in errors if 'serviceWorker' not in e]
        self.assertEqual(relevant, [])

    def open_page(self, progress_requests=None, settings_requests=None):
        context = self.browser.new_context(service_workers='block')
        self.addCleanup(context.close)
        page = context.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error).splitlines()[0]))
        progress_requests = progress_requests if progress_requests is not None else []
        settings_requests = settings_requests if settings_requests is not None else []

        def handle(route):
            url = urlparse(route.request.url)
            if f'{url.scheme}://{url.netloc}' != ORIGIN:
                return route.abort()
            if url.path == '/':
                return route.fulfill(status=200, content_type='text/html', body=self.html)
            if url.path.startswith('/static/'):
                path = (APP_ROOT / url.path.lstrip('/')).resolve()
                if path.is_file() and (APP_ROOT / 'static') in path.parents:
                    return route.fulfill(status=200, path=str(path))
                return route.fulfill(status=404, body='')
            if url.path == '/api/math_steps/progress':
                progress_requests.append(route.request.post_data or '')
                return route.fulfill(status=200, content_type='application/json', body='{"status":"ok","math_steps_state":{"current":1,"max_reached":1}}')
            if url.path == '/api/math_steps/simplify':
                body = '{"status":"ok","text":"やさしい説明です","math_steps_state":{"current":0,"max_reached":2,"simplified":{"0":"やさしい説明です"}}}'
                return route.fulfill(status=200, content_type='application/json', body=body)
            if url.path == '/api/settings' and route.request.method == 'POST':
                settings_requests.append(route.request.post_data or '')
                return route.fulfill(status=200, content_type='application/json', body='{"status":"ok"}')
            return route.fulfill(status=200, content_type='application/json', body='{}')

        page.route('**/*', handle)
        page.goto(ORIGIN + '/', wait_until='load')
        page.wait_for_timeout(1200)
        return page, errors

    def render(self, page, text, streaming=False, message_id=42):
        page.evaluate(
            "([text, streaming, messageId]) => {"
            "  const welcome = document.getElementById('welcome-screen');"
            "  if (welcome) welcome.classList.add('hidden');"
            "  const host = document.createElement('div');"
            "  host.id = 'math-steps-test-host';"
            "  document.getElementById('chat-container').appendChild(host);"
            "  return renderAiMarkdownInto(host, text, { streaming, mathSteps: { messageId, streaming } });"
            "}",
            [text, streaming, message_id],
        )

    def test_steps_render_one_at_a_time_with_image_only_mathsvg(self):
        page, errors = self.open_page()
        self.render(page, STEP_TEXT, streaming=True)
        self.assertEqual(page.locator('#math-steps-test-host .math-steps').count(), 1)
        self.assertIn('1 / 3+', page.inner_text('#math-steps-test-host .math-steps-counter'))
        self.assertEqual(page.locator('#math-steps-test-host .math-steps-body .math-steps-prose').count(), 1)
        body = page.inner_text('#math-steps-test-host .math-steps-prose')
        self.assertIn('問題を整理する', page.inner_text('#math-steps-test-host .math-steps-title'))
        self.assertIn('まず、わかっていること', body)
        self.assertNotIn('式を作ります', page.inner_text('#math-steps-test-host .math-steps'))
        # mathsvg is an image without the code wrapper/preview/download UI.
        self.assertEqual(page.locator('#math-steps-test-host .mathsvg-render img').count(), 1)
        self.assertEqual(page.locator('#math-steps-test-host .code-wrapper').count(), 0)
        self.assertEqual(page.locator('#math-steps-test-host .mathsvg-render .copy-btn').count(), 0)
        # While streaming the last parsed step cannot be advanced yet.
        self.assertNoPageErrors(errors)

    def test_next_back_restart_and_finish(self):
        page, errors = self.open_page()
        self.render(page, STEP_TEXT, streaming=False)
        self.assertIn('1 / 3', page.inner_text('#math-steps-test-host .math-steps-counter'))
        page.click('#math-steps-test-host .math-steps-next')
        self.assertIn('2 / 3', page.inner_text('#math-steps-test-host .math-steps-counter'))
        self.assertIn('式を作ります', page.inner_text('#math-steps-test-host .math-steps-prose'))
        page.click('#math-steps-test-host .math-steps-next')
        self.assertIn('3 / 3', page.inner_text('#math-steps-test-host .math-steps-counter'))
        next_btn = page.locator('#math-steps-test-host .math-steps-next')
        self.assertEqual(next_btn.inner_text(), '完了')
        self.assertTrue(next_btn.is_disabled())
        page.click('#math-steps-test-host .math-steps-prev')
        self.assertIn('2 / 3', page.inner_text('#math-steps-test-host .math-steps-counter'))
        page.click('#math-steps-test-host .math-steps-restart')
        self.assertIn('1 / 3', page.inner_text('#math-steps-test-host .math-steps-counter'))
        self.assertNoPageErrors(errors)

    def test_progress_is_posted_after_step_change(self):
        progress = []
        page, errors = self.open_page(progress_requests=progress)
        self.render(page, STEP_TEXT, streaming=False)
        page.click('#math-steps-test-host .math-steps-next')
        page.wait_for_timeout(900)
        self.assertTrue(progress, 'progress request was not sent')
        self.assertIn('"current":1', progress[-1])
        self.assertIn('"message_id":42', progress[-1])
        self.assertNoPageErrors(errors)

    def test_simplify_replaces_only_the_current_step_and_can_be_restored(self):
        progress = []
        page, errors = self.open_page(progress_requests=progress)
        self.render(page, STEP_TEXT, streaming=False)
        page.click('#math-steps-test-host .math-steps-simplify')
        page.wait_for_timeout(400)
        body = page.inner_text('#math-steps-test-host .math-steps-prose')
        self.assertIn('やさしい説明です', body)
        self.assertNotIn('まず、わかっていること', body)
        self.assertEqual(page.locator('#math-steps-test-host .math-steps-restore').count(), 1)
        page.click('#math-steps-test-host .math-steps-restore')
        self.assertIn('まず、わかっていること', page.inner_text('#math-steps-test-host .math-steps-prose'))
        page.wait_for_timeout(300)
        self.assertTrue(any('remove_simplified' in body for body in progress),
                        'restore must remove the saved simplified variant')
        self.assertNoPageErrors(errors)

    def test_streaming_single_step_shows_disabled_preparing_button(self):
        page, errors = self.open_page()
        self.render(page, '## ステップ 1 準備\n本文', streaming=True)
        self.assertIn('1 / 1+', page.inner_text('#math-steps-test-host .math-steps-counter'))
        next_btn = page.locator('#math-steps-test-host .math-steps-next')
        self.assertIn('次のステップを準備中', next_btn.inner_text())
        self.assertTrue(next_btn.is_disabled())
        self.assertNoPageErrors(errors)

    def test_without_step_headings_normal_markdown_is_unchanged(self):
        page, errors = self.open_page()
        self.render(page, '見出しのない普通の回答です。\n\n```svg\n<svg viewBox="0 0 10 10"><rect width="10" height="10"/></svg>\n```')
        self.assertEqual(page.locator('#math-steps-test-host .math-steps').count(), 0)
        self.assertEqual(page.locator('#math-steps-test-host .svg-code-render').count(), 1)
        self.assertNoPageErrors(errors)

    def test_pdf_and_copy_helpers_expand_steps_and_diagrams(self):
        page, errors = self.open_page()
        pdf = page.evaluate(
            "text => buildMathStepsPdfHtml(text, { current: 1, max_reached: 2, simplified: { '1': 'やさしい式の説明' } })",
            STEP_TEXT,
        )
        self.assertIn('ステップ 1', pdf)
        self.assertIn('ステップ 2', pdf)
        self.assertIn('やさしい式の説明', pdf)
        self.assertIn('mathsvg-render', pdf)
        copy_text = page.evaluate('text => prepareMathStepsCopyText(text)', STEP_TEXT)
        self.assertIn('## ステップ 1', copy_text)
        self.assertIn('［図］', copy_text)
        self.assertNotIn('mathsvg', copy_text)
        self.assertNotIn('<svg', copy_text)
        self.assertNoPageErrors(errors)

    def test_master_off_hides_chip_and_gear_modal_saves_partial_config(self):
        saved = []
        page, errors = self.open_page(settings_requests=saved)
        # The mocked settings payload has no math_steps_enabled, so the master is OFF.
        self.assertIn('hidden', page.get_attribute('#math-steps-container', 'class') or '')
        page.evaluate("openMathStepsModal()")
        page.wait_for_timeout(150)
        self.assertIn('modal-open', page.get_attribute('#math-steps-config-modal', 'class') or '')
        page.check('#math-steps-cfg-basics')
        page.select_option('#math-steps-cfg-level', 'detailed')
        page.select_option('#math-steps-cfg-step-size', 'small')
        page.select_option('#math-steps-cfg-diagrams', 'off')
        page.uncheck('#math-steps-cfg-animation')
        page.click('#math-steps-config-save')
        page.wait_for_timeout(300)
        self.assertTrue(saved, 'settings were not saved')
        self.assertIn('"math_steps_config"', saved[-1])
        self.assertIn('"basics":true', saved[-1])
        self.assertIn('"level":"detailed"', saved[-1])
        self.assertIn('"step_size":"small"', saved[-1])
        self.assertIn('"diagrams":"off"', saved[-1])
        self.assertIn('"animation":false', saved[-1])
        # Master is still OFF, so the prompt-bar chip must stay hidden.
        self.assertIn('hidden', page.get_attribute('#math-steps-container', 'class') or '')
        self.assertNoPageErrors(errors)


if __name__ == '__main__':
    unittest.main()
