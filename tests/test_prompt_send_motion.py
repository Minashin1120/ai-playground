"""Exercise the real send presentation helpers in Chromium, without network or account data."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOTION = ROOT / 'static/js/chat_core_parts/chat_core.part19_prompt_send_motion.js'


class PromptSendMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise unittest.SkipTest('playwright is not installed')
        cls.pw = sync_playwright().start()
        try:
            cls.browser = cls.pw.chromium.launch()
        except Exception as error:
            cls.pw.stop()
            raise unittest.SkipTest(f'Chromium is not available: {error}')

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def setUp(self):
        self.page = self.browser.new_page(viewport={'width': 900, 'height': 800})
        self.addCleanup(self.page.close)
        self.page.set_content('''
            <style>
            * { box-sizing: border-box; }
            #chat-container { position: absolute; top: 0; width: 850px; height: 600px; overflow: auto; }
            .message-bubble { width: 300px; margin-left: 500px; padding: 16px; border-radius: 20px; background: blue; }
            .composer-dock { position: absolute; bottom: 0; width: 850px; }
            textarea { width: 800px; height: 110px; }
            .prompt-send-hidden { visibility: hidden; }
            .prompt-send-overlay { position: fixed !important; z-index: 35; margin: 0 !important;
              transform-origin: top left; pointer-events: none; }
            </style>
            <div id="chat-container"><div id="row"><div class="message-bubble">
              <div id="content">A complete prompt</div><div>Quote</div><div>attachment.pdf</div>
            </div></div></div>
            <div class="composer-dock"><div id="quote-bar"><span id="quote-text-display">Quote</span></div>
              <div id="file-preview"><div id="file-preview-thumbs"></div></div>
              <div id="input-row"><textarea id="prompt-input">A complete prompt</textarea></div></div>
            <div id="upload-list"><div class="upload-row" data-filename="test.pdf"></div></div>
        ''')
        self.page.add_script_tag(content='''
            const get = (id) => document.getElementById(id);
            let currentImageUrls = ['test.pdf'], currentMaskImage = 'mask.png';
            let currentQuote = 'Quote', currentQuoteMessageId = 7, userAutoScroll = true, threadLoadSequence = 0;
            const browserFastLocalFiles = new Map(), markerAppliedUploads = new Set(['test.pdf']);
            let uploadProgressState = { total: 1, completed: 1, active: 0, perFilePct: {} };
            function clearQuote() { currentQuote = ''; currentQuoteMessageId = null; get('quote-bar').classList.remove('visible'); }
            function updateFilePreview() {}
            function updateMaskPreview() {}
            function schedulePromptTokenEstimate() {}
        ''')
        self.page.add_script_tag(content=MOTION.read_text())

    def start(self, follow=True):
        self.page.evaluate('''follow => {
            window.tx = takePromptForSend();
            window.stopMotion = animatePromptToMessage(get('row'), tx, follow);
        }''', follow)

    def test_take_and_restore_preserve_content_rows_and_attachment_state(self):
        self.start(False)
        self.assertEqual(self.page.input_value('#prompt-input'), '')
        self.assertEqual(self.page.locator('#upload-list .upload-row').count(), 0)
        self.assertEqual(self.page.locator('#file-preview-thumbs').count(), 1)
        self.page.evaluate('tx.restore(); tx.restore()')
        self.assertEqual(self.page.input_value('#prompt-input'), 'A complete prompt')
        self.assertEqual(self.page.locator('#upload-list .upload-row').count(), 1)
        self.assertEqual(self.page.evaluate('[currentImageUrls, currentMaskImage, currentQuote, currentQuoteMessageId, [...markerAppliedUploads]]'),
                         [['test.pdf'], 'mask.png', 'Quote', 7, ['test.pdf']])

    def test_failure_does_not_overwrite_new_draft_or_attachments(self):
        self.start(False)
        self.page.fill('#prompt-input', 'Next prompt')
        self.page.evaluate("currentImageUrls.push('next.png'); currentQuote = 'Next quote'; tx.restore()")
        self.assertEqual(self.page.input_value('#prompt-input'), 'Next prompt')
        self.assertEqual(self.page.evaluate('[currentImageUrls, currentQuote]'), [['test.pdf', 'next.png'], 'Next quote'])

    def test_accept_does_not_clear_next_draft_and_prevents_restore(self):
        self.start(False)
        self.page.fill('#prompt-input', 'Next prompt')
        self.page.evaluate('tx.accept(); tx.restore()')
        self.assertEqual(self.page.input_value('#prompt-input'), 'Next prompt')
        self.assertEqual(self.page.evaluate('currentImageUrls'), [])

    def test_overlay_contains_full_content_without_duplicate_ids_or_accessibility(self):
        self.page.evaluate("get('content').textContent = 'long prompt '.repeat(1000)")
        self.start()
        self.page.wait_for_selector('.prompt-send-overlay', state='attached')
        self.page.evaluate("document.getAnimations().forEach(a => a.pause())")
        overlay = self.page.locator('.prompt-send-overlay')
        self.assertIn('attachment.pdf', overlay.inner_text())
        self.assertEqual(overlay.locator('[id]').count(), 0)
        self.assertEqual(overlay.get_attribute('aria-hidden'), 'true')
        self.assertTrue(overlay.evaluate('(el) => el.inert'))
        self.assertGreater(len(overlay.inner_text()), 10000)
        self.page.evaluate('stopMotion()')
        self.assertEqual(overlay.count(), 0)
        self.assertNotIn('prompt-send-hidden', self.page.locator('#row').get_attribute('class') or '')

    def test_normal_finish_removes_overlay_and_reveals_real_bubble(self):
        self.start()
        self.page.wait_for_selector('.prompt-send-overlay', state='attached')
        self.page.wait_for_selector('.prompt-send-overlay', state='detached')
        self.assertTrue(self.page.locator('#row').is_visible())
        self.assertEqual(self.page.locator('#row .message-bubble').evaluate('(el) => getComputedStyle(el).opacity'), '1')

    def test_reduced_motion_and_paused_reader_skip_movement(self):
        for reduced, follow in [('reduce', True), ('no-preference', False)]:
            self.page.emulate_media(reduced_motion=reduced)
            self.start(follow)
            self.page.evaluate('tx.accept()')
            self.assertEqual(self.page.locator('.prompt-send-overlay').count(), 0)
            self.assertTrue(self.page.locator('#row').is_visible())

    def test_scroll_navigation_resize_and_removed_target_clean_up(self):
        for event in ['wheel', 'popstate', 'resize', 'remove']:
            with self.subTest(event=event):
                self.start()
                self.page.wait_for_selector('.prompt-send-overlay', state='attached')
                if event == 'wheel':
                    self.page.dispatch_event('#chat-container', 'wheel')
                elif event == 'remove':
                    self.page.evaluate("get('row').remove()")
                else:
                    self.page.evaluate('(event) => window.dispatchEvent(new Event(event))', event)
                self.page.wait_for_selector('.prompt-send-overlay', state='detached')
                self.assertEqual(self.page.locator('.prompt-send-hidden').count(), 0)


    def test_navigation_prevents_restoring_an_old_chats_composer(self):
        self.start(False)
        self.page.evaluate('threadLoadSequence++; tx.restore()')
        self.assertEqual(self.page.input_value('#prompt-input'), '')
        self.assertEqual(self.page.evaluate('currentImageUrls'), [])
        self.assertFalse(self.page.evaluate('tx.canRetry()'))
