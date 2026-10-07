"""ログの収集を強化: the browser activity log (static/js/activity_log.js) and the feedback endpoint that stores it."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

APP_ROOT = Path(__file__).resolve().parents[1]


def _run_node(body):
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const store = new Map();
const localStorage = {
  getItem(k) { return store.has(k) ? store.get(k) : null; },
  setItem(k, v) { store.set(k, String(v)); },
  removeItem(k) { store.delete(k); },
};
const listeners = {};
const docListeners = {};
const window = {
  location: new URL('https://example.test/c/abc'),
  localStorage, innerWidth: 800, innerHeight: 600, devicePixelRatio: 1,
  history: { pushState() {}, replaceState() {} },
  console: { error() {}, warn() {}, log() {} },
  setTimeout(fn) { return 0; }, clearTimeout() {}, setInterval() { return 1; },
  addEventListener(name, fn) { (listeners[name] = listeners[name] || []).push(fn); },
  fetch(url) { return Promise.resolve({ status: 200, redirected: false, headers: { get() { return 'application/json'; } } }); },
  CHAT_CONFIG: { currentUsername: 'alice', appVersion: '1.0' },
};
const document = {
  visibilityState: 'visible', body: null,
  addEventListener(name, fn) { (docListeners[name] = docListeners[name] || []).push(fn); },
  getElementsByTagName() { return []; },
};
const ctx = vm.createContext({ window, document, navigator: { onLine: true, userAgent: 'test' }, URL, JSON, Date, Math,
  setTimeout: window.setTimeout, clearTimeout: window.clearTimeout, setInterval: window.setInterval });
const load = () => vm.runInContext(fs.readFileSync('static/js/activity_log.js', 'utf8'), ctx);
""" + body
    subprocess.run(['node', '-e', script], cwd=APP_ROOT, check=True, timeout=10)


def test_activity_log_records_only_when_enabled_and_without_user_text():
    _run_node(r"""
load();
const log = window.ActivityLog;
assert.equal(log.isEnabled(), false);
window.fetch('/api/threads?q=secret');
log.flush();
assert.equal(store.get('aip_activity_log_v1'), undefined);
log.setEnabled(true);
assert.equal(store.get('aip_activity_log_enabled'), '1');
(async () => {
  await window.fetch('/api/threads?q=secret-search');
  await Promise.resolve();
  const input = { tagName: 'TEXTAREA', type: 'textarea', value: 'my private prompt', className: 'x', id: 'prompt-input', parentElement: null };
  docListeners.change.forEach(fn => fn({ target: input }));
  window.console.error('failed', { prompt: 'my private prompt' });
  const entries = log.recent();
  const text = JSON.stringify(entries);
  assert.ok(entries.some(e => e.ev === 'fetch' && e.path === '/api/threads' && e.status === 200), text);
  assert.ok(!text.includes('secret'), text);
  assert.ok(!text.includes('my private prompt'), text);
  assert.ok(entries.some(e => e.ev === 'change' && e.length === 17), text);
  assert.ok(log.stats().count > 0);
  log.clear();
  assert.equal(log.recent().length, 0);
  log.log('x');
  assert.equal(log.recent().length, 1);
  log.setEnabled(false);
  assert.equal(store.get('aip_activity_log_v1'), undefined);
})().catch(e => { console.error(e); process.exitCode = 1; });
""")


def test_activity_log_of_another_account_is_dropped_on_load():
    _run_node(r"""
store.set('aip_activity_log_enabled', '1');
store.set('aip_activity_log_owner', 'bob');
store.set('aip_activity_log_v1', JSON.stringify([{ t: Date.now(), ev: 'from-bob', seq: 1 }]));
load();
const entries = window.ActivityLog.recent();
assert.ok(!entries.some(e => e.ev === 'from-bob'));
assert.equal(store.get('aip_activity_log_owner'), 'alice');
""")


def test_activity_log_script_loads_before_chat_core():
    template = (APP_ROOT / 'templates' / 'chat.html').read_text(encoding='utf-8')
    scripts = (APP_ROOT / 'templates' / 'chat' / 'scripts.html').read_text(encoding='utf-8')
    assert "filename='js/activity_log.min.js') }}?v={{ app_version }}\" defer" in template
    assert 'chat_core.min.' in scripts


class FeedbackClientLogTests(unittest.TestCase):
    def setUp(self):
        import app as target
        self.target = target
        target.app.config.update(TESTING=True, MAINTENANCE_MODE=False, TRUSTED_HOSTS=['localhost'])
        target._ensure_temp_chat_monitor_running = lambda: None
        log_dir = tempfile.TemporaryDirectory(prefix='feedback-logs-')
        self.addCleanup(log_dir.cleanup)
        self.log_dir = log_dir.name
        with target.app.app_context():
            target.db.session.remove()
            target.db.drop_all()
            target.db.create_all()
            user = target.User(username='feedback-log-test', is_setup_completed=True)
            user.set_password('test-password')
            target.db.session.add(user)
            target.db.session.commit()
            self.user_id = user.id
        for patcher in [
            mock.patch.object(target, '_bot_turnstile_active', return_value=False),
            mock.patch.object(target, 'rate_limit', return_value=True),
            mock.patch.dict(target.app.config, FEEDBACK_LOGS_DIR=self.log_dir),
        ]:
            patcher.start()
            self.addCleanup(patcher.stop)

    def post(self, payload):
        client = self.target.app.test_client()
        with client.session_transaction() as sess:
            sess['_user_id'] = str(self.user_id)
            sess['_fresh'] = True
            sess['csrf_token'] = 'csrf-test-token'
        return client.post('/api/feedback', json=payload, headers={'X-CSRF-Token': 'csrf-test-token'},
                           base_url='https://localhost')

    def test_feedback_stores_client_logs_under_logs_dir(self):
        self.assertEqual(self.post({'message': 'm', 'client_logs': 'x'}).status_code, 400)
        plain = self.post({'message': 'no logs'})
        self.assertEqual(plain.status_code, 200)
        self.assertNotIn('logs_saved', plain.get_json())
        self.assertEqual(os.listdir(self.log_dir), [])
        entries = [{'t': 1, 'ev': 'click', 'target': 'button#fb-submit'}, 'not an object',
                   {'t': 2, 'ev': 'huge', 'blob': 'x' * 70000}]
        response = self.post({'message': 'with logs', 'client_logs': {
            'client': 'web', 'version': 'V1', 'window_seconds': 3600, 'entries': entries}})
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertIs(response.get_json()['logs_saved'], True)
        names = os.listdir(self.log_dir)
        self.assertEqual(len(names), 1)
        self.assertRegex(names[0], r'^feedback-\d+-web-\d{8}T\d{6}Z\.jsonl$')
        path = os.path.join(self.log_dir, names[0])
        with open(path, encoding='utf-8') as handle:
            rows = [json.loads(line) for line in handle]
        self.assertEqual(rows[0]['kind'], 'meta')
        self.assertEqual(rows[0]['user_id'], self.user_id)
        self.assertEqual((rows[0]['entries'], rows[0]['dropped']), (2, 1))
        self.assertEqual(rows[1]['ev'], 'click')
        self.assertEqual(set(rows[2]), {'t', 'ev', 'truncated'})
        self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
        with self.target.app.app_context():
            index = self.target._feedback_log_file_index()
        self.assertEqual(list(index.values()), [names[0]])
