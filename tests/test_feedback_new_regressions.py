"""Behavioral coverage for new-feedback settings retries and release planning."""
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class FeedbackNewRegressions(unittest.TestCase):
    def test_settings_retry_recovers_timeout_but_does_not_retry_auth_failure(self):
        source = (ROOT / 'static/js/chat_core_parts/chat_core.part01_bootstrap_utils.js').read_text()
        block = source.split('        const SETTINGS_LOAD_TIMEOUT_MS =', 1)[1].split('        // ユーザーシステムプロンプト', 1)[0]
        script = r'''
const assert = require('node:assert/strict');
async function scenario(failure, retryable = true) {
    let userSettingsSnapshot = null, userSettingsSnapshotPromise = null, calls = 0;
    const CHAT_CONFIG = {urls: {handleSettingsQuery: '/api/settings'}};
    const cacheUserSettings = data => userSettingsSnapshot = data;
    const setTimeout = (_, ms) => ms === 500 ? (queueMicrotask(_), 1) : 1;
    const clearTimeout = () => {};
    const apiFetch = async () => {
        calls++;
        if (calls === 1) {
            if (failure === 'network') throw new TypeError('network');
            return {ok: false, status: retryable ? 503 : 401};
        }
        return {ok: true, json: async () => ({default_model: 'saved'})};
    };
    BLOCK
    const [a, b] = await Promise.all([ensureUserSettingsSnapshot(), ensureUserSettingsSnapshot()]);
    assert.equal(calls, retryable ? 2 : 1);
    assert.deepEqual(a, retryable ? {default_model: 'saved'} : null);
    assert.equal(a, b);
    if (a) { await ensureUserSettingsSnapshot(); assert.equal(calls, 2); }
    assert.equal(userSettingsSnapshotPromise, null);
}
(async () => { await scenario('network'); await scenario('http'); await scenario('http', false); })()
.catch(e => { console.error(e); process.exit(1); });
'''.replace('    BLOCK', '    const SETTINGS_LOAD_TIMEOUT_MS =' + block)
        result = subprocess.run(['node', '-e', script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_record_plan_matches_missing_existing_and_failed_tag_lookup(self):
        source = (ROOT / 'scripts/record_changes.sh').read_text()
        block = source.split('        TAG_STATUS=0', 1)[1].split('\n    fi', 1)[0]
        for code, expected in [(0, 'remote tag already exists'), (2, 'unfinished release'), (128, 'lookup failed')]:
            with self.subTest(code=code):
                script = 'VERSION_NAME=1.2.3\ngit_in() { return ' + str(code) + '; }\ndie() { echo "$*"; exit 1; }\nTAG_STATUS=0' + block
                result = subprocess.run(['bash', '-c', script], capture_output=True, text=True)
                self.assertIn(expected, result.stdout)
                self.assertEqual(result.returncode, 1 if code == 128 else 0)

    def test_search_box_survives_completion_and_can_start_another_search(self):
        source = (ROOT / 'static/js/chat_core_parts/chat_core.part13_canvas_coding_stream.js').read_text()
        block = source[source.index('        const SEARCH_COMPLETE_HTML'):source.index('        function renderMessage(')]
        script = """
const assert = require('node:assert/strict');
BLOCK
const box = {innerHTML: '', pulsing: false, classList: {toggle: (_, on) => box.pulsing = on}};
let inserted = 0;
const bubble = {insertAdjacentHTML: () => inserted++, querySelector: () => box};
let active = updateSearchBox(bubble, null, 'searching');
assert.equal(box.pulsing, true);
active = updateSearchBox(bubble, active, 'done');
assert.equal(box.pulsing, false);
assert.ok(box.innerHTML.includes('Search complete'));
active = updateSearchBox(bubble, active, 'searching');
assert.equal(box.pulsing, true);
assert.equal(inserted, 1);
""".replace('BLOCK', block)
        result = subprocess.run(['node', '-e', script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_android_prompt_option_allowlist_matches_server(self):
        import ast
        import re
        source = (ROOT / 'server/routes_chat.py').read_text()
        keys = ast.literal_eval('(' + re.search(r'_PROMPT_OPTION_KEYS = \((.*?)\n\)', source, re.S).group(1) + ')')
        android = (ROOT / 'android/app/src/main/java/com/minashin1120/aiplayground/data/Models.kt').read_text().split('internal fun promptOptionsRecord', 1)[1].split('internal fun reasoningSearchUsed', 1)[0]
        self.assertEqual(set(keys), set(re.findall(r'"([a-z_0-9]+)"', android)))
