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


def test_feedback_report_names_logs_and_chat_copy():
    _run_node(r"""
load();
const toasts = [];
window.showToast = (text, type) => toasts.push([text, type]);
const reply = (ok, data) => ({ ok, json: () => Promise.resolve(data) });
(async () => {
  const logs = { entries: [1, 2] };
  assert.equal(await window.ActivityLog.reportFeedback(reply(true, {}), null, false), true);
  assert.equal(await window.ActivityLog.reportFeedback(reply(true, { logs_saved: true, chat_copy_saved: true }), logs, true), true);
  assert.equal(await window.ActivityLog.reportFeedback(reply(true, { chat_copy_saved: true }), null, true), true);
  assert.equal(await window.ActivityLog.reportFeedback(reply(true, { logs_saved: true, chat_copy_saved: false }), logs, true), true);
  assert.equal(await window.ActivityLog.reportFeedback(reply(false, { error: 'rate_limit' }), null, true), false);
  assert.deepEqual(toasts.map(t => t[0]), [
    'フィードバックを送信しました',
    'フィードバック、直近1時間のログ（2件）、チャットのコピーを送信しました',
    'フィードバックとチャットのコピーを送信しました',
    'フィードバックを送信しました（チャットのコピーは保存できませんでした）',
    'rate_limit',
  ]);
})().catch(e => { console.error(e); process.exitCode = 1; });
""")


def test_chat_copy_payload_carries_storage_keyed_by_the_chat():
    _run_node(r"""
const entries = [['fixed_branch_abc', '7'], ['branch_names_abc', '{}'], ['theme', 'dark']];
localStorage.length = entries.length;
localStorage.key = i => entries[i][0];
localStorage.getItem = k => (entries.find(e => e[0] === k) || [])[1];
load();
const copy = window.ActivityLog.chatCopyPayload('abc');
assert.equal(copy.client, 'web');
assert.equal(copy.thread_id, 'abc');
assert.equal(copy.version, '1.0');
assert.deepEqual(copy.client_state.local_storage, { fixed_branch_abc: '7', branch_names_abc: '{}' });
assert.deepEqual(copy.client_state.session_storage, {});
assert.equal(copy.client_state.url, '/c/abc');
""")


def test_activity_log_script_loads_before_chat_core():
    template = (APP_ROOT / 'templates' / 'chat.html').read_text(encoding='utf-8')
    scripts = (APP_ROOT / 'templates' / 'chat' / 'scripts.html').read_text(encoding='utf-8')
    assert "filename='js/activity_log.min.js') }}?v={{ app_version }}\" defer" in template
    assert 'chat_core_bundles' in scripts


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
            mock.patch.dict(target.app.config, FEEDBACK_DIR=self.log_dir),
        ]:
            patcher.start()
            self.addCleanup(patcher.stop)

    def post(self, payload, user_agent=None):
        client = self.target.app.test_client()
        with client.session_transaction() as sess:
            sess['_user_id'] = str(self.user_id)
            sess['_fresh'] = True
            sess['csrf_token'] = 'csrf-test-token'
        headers = {'X-CSRF-Token': 'csrf-test-token'}
        if user_agent:
            headers['User-Agent'] = user_agent
        return client.post('/api/feedback', json=payload, headers=headers, base_url='https://localhost')

    def delete(self, feedback_id, user_id=None):
        client = self.target.app.test_client()
        with client.session_transaction() as sess:
            sess['_user_id'] = str(user_id or self.user_id)
            sess['_fresh'] = True
            sess['csrf_token'] = 'csrf-test-token'
        return client.delete(f'/api/feedback/{feedback_id}', headers={'X-CSRF-Token': 'csrf-test-token'},
                             base_url='https://localhost')

    def add_user(self, name, is_admin=False):
        target = self.target
        with target.app.app_context():
            user = target.User(username=name, is_setup_completed=True, is_admin=is_admin)
            user.set_password('test-password')
            target.db.session.add(user)
            target.db.session.commit()
            return user.id

    def test_sender_or_administrator_deletes_feedback_and_its_directory(self):
        logs = {'client': 'web', 'entries': [{'t': 1, 'ev': 'click'}]}
        first = self.post({'message': 'one', 'client_logs': logs}).get_json()['feedback_id']
        second = self.post({'message': 'two'}).get_json()['feedback_id']
        self.assertEqual(sorted(os.listdir(os.path.join(self.log_dir, str(first)))), ['feedback.json', 'logs'])
        other = self.add_user('feedback-other')
        self.assertEqual(self.delete(first, other).status_code, 404)
        self.assertTrue(os.path.isdir(os.path.join(self.log_dir, str(first))))
        self.assertEqual(self.delete(first).status_code, 200)
        self.assertFalse(os.path.exists(os.path.join(self.log_dir, str(first))))
        self.assertEqual(self.delete(first).status_code, 404)
        admin = self.add_user('feedback-admin', is_admin=True)
        self.assertEqual(self.delete(second, admin).status_code, 200)
        self.assertEqual(os.listdir(self.log_dir), [])
        with self.target.app.app_context():
            self.assertEqual(self.target.Feedback.query.count(), 0)

    def read_info(self, feedback_id):
        path = os.path.join(self.log_dir, str(feedback_id), 'feedback.json')
        self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
        with open(path, encoding='utf-8') as handle:
            return json.load(handle)

    def test_feedback_records_its_client_without_logs(self):
        web = self.post({'message': 'web', 'client': 'web', 'version': 'V4'}).get_json()['feedback_id']
        android = self.post({'message': 'app'}, user_agent='AIPlayground-Android/1.2.3').get_json()['feedback_id']
        other = self.post({'message': 'other', 'client': 'not valid!'}).get_json()['feedback_id']
        self.assertEqual([(info['client'], info['client_version']) for info in map(self.read_info, (web, android, other))],
                         [('web', 'V4'), ('android', '1.2.3'), ('unknown', None)])

    def test_feedback_stores_info_and_client_logs_in_its_directory(self):
        self.assertEqual(self.post({'message': 'm', 'client_logs': 'x'}).status_code, 400)
        plain = self.post({'title': 'Plain', 'message': 'no logs'})
        self.assertEqual(plain.status_code, 200)
        self.assertNotIn('logs_saved', plain.get_json())
        plain_id = plain.get_json()['feedback_id']
        self.assertEqual(os.listdir(os.path.join(self.log_dir, str(plain_id))), ['feedback.json'])
        info = self.read_info(plain_id)
        self.assertEqual((info['feedback_id'], info['user_id'], info['username'], info['title'], info['message'], info['status']),
                         (plain_id, self.user_id, 'feedback-log-test', 'Plain', 'no logs', 'new'))
        self.assertIsNone(info['activity_log_saved'])
        entries = [{'t': 1, 'ev': 'click', 'target': 'button#fb-submit'}, 'not an object',
                   {'t': 2, 'ev': 'huge', 'blob': 'x' * 70000}]
        response = self.post({'title': 'Bug', 'message': 'with logs', 'client_logs': {
            'client': 'web', 'version': 'V1', 'window_seconds': 3600, 'entries': entries}})
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertIs(response.get_json()['logs_saved'], True)
        feedback_id = response.get_json()['feedback_id']
        directory = os.path.join(self.log_dir, str(feedback_id))
        self.assertEqual(sorted(os.listdir(directory)), ['feedback.json', 'logs'])
        path = os.path.join(directory, 'logs', 'activity.jsonl')
        with open(path, encoding='utf-8') as handle:
            rows = [json.loads(line) for line in handle]
        self.assertEqual(rows[0]['kind'], 'meta')
        self.assertEqual(rows[0]['user_id'], self.user_id)
        self.assertEqual((rows[0]['entries'], rows[0]['dropped']), (2, 1))
        self.assertEqual(rows[1]['ev'], 'click')
        self.assertEqual(set(rows[2]), {'t', 'ev', 'truncated'})
        self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
        info = self.read_info(feedback_id)
        self.assertEqual((info['title'], info['message'], info['client'], info['client_version'], info['activity_log_saved']),
                         ('Bug', 'with logs', 'web', 'V1', True))
        with self.target.app.app_context():
            self.assertEqual(self.target._feedback_activity_log_path(feedback_id), path)
            fb = self.target.db.session.get(self.target.Feedback, feedback_id)
            fb.status, fb.admin_reply = 'replied', 'thanks'
            self.target.db.session.commit()
            self.target._write_feedback_info(fb)
        info = self.read_info(feedback_id)
        self.assertEqual((info['status'], info['admin_reply'], info['client']), ('replied', 'thanks', 'web'))
