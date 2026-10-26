"""Exercise rename and feedback status filtering with synthetic DOM objects."""
import subprocess
from pathlib import Path

PARTS = Path(__file__).resolve().parents[1] / 'static/js/chat_core_parts'


def source(part, start, end):
    text = next(PARTS.glob(f'chat_core.part{part}_*.js')).read_text()
    return text[text.index(start):text.index(end, text.index(start))]


def run_js(script):
    result = subprocess.run(['node', '-e', script], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_rename_prefill_cancel_failure_retry_and_header_update():
    run_js('''
const assert = require('node:assert/strict');
const nodes = Object.fromEntries(['modal', 'form', 'input', 'save', 'cancel', 'error'].map(k =>
    ['thread-rename-' + k, {value: '', disabled: false, focus() {}}]));
nodes['thread-rename-modal'] = nodes['thread-rename-modal'];
const get = id => nodes[id];
let shown = 0, hidden = 0, calls = 0, loads = 0, header = '', fail = true;
const showModal = () => shown++, hideModal = () => hidden++;
const loadThreads = () => loads++, setCurrentChatHeaderTitle = s => header = s;
const currentThreadId = 'synthetic-id', CHAT_CONFIG = {urls: {updateTitle: '/api/threads/0/title'}};
const apiFetch = async (url, options) => {
    calls++; assert.equal(url, '/api/threads/synthetic-id/title');
    assert.deepEqual(JSON.parse(options.body), {title: 'new title'});
    return {ok: !fail, json: async () => ({title: 'normalized title'})};
};
const event = {stopPropagation() {}, currentTarget: {focus() {}, closest: () => ({dataset: {threadTitle: 'old <title>'}})}};
const submit = {preventDefault() {}};
''' + source('15', '        function renameThread', '        async function deleteThread') + '''
(async () => {
    renameThread(event, 'synthetic-id');
    assert.equal(get('thread-rename-input').value, 'old <title>');
    get('thread-rename-cancel').onclick();
    assert.equal(calls, 0); assert.equal(hidden, 1);
    renameThread(event, 'synthetic-id');
    get('thread-rename-input').value = '   ';
    await get('thread-rename-form').onsubmit(submit); assert.equal(calls, 0);
    get('thread-rename-input').value = ' new title ';
    await get('thread-rename-form').onsubmit(submit);
    assert.equal(hidden, 1); assert.ok(get('thread-rename-error').textContent);
    assert.equal(get('thread-rename-input').value, ' new title ');
    assert.equal(get('thread-rename-save').disabled, false);
    fail = false;
    await get('thread-rename-form').onsubmit(submit);
    assert.equal(hidden, 2); assert.equal(loads, 1); assert.equal(header, 'normalized title');
})().catch(e => { console.error(e); process.exitCode = 1; });
''')


def test_filter_all_unresolved_individual_empty_and_updated_status():
    run_js('''
const assert = require('node:assert/strict');
const statuses = ['new', 'in_review', 'replied', 'rejected', 'resolved', 'resolved_unreleased'];
const rows = statuses.map(status => ({dataset: {feedbackStatus: status}, hidden: false}));
let emptyHidden = true;
const filter = {value: 'all'};
const get = id => ({'fb-admin-status-filter': filter, 'fb-admin-list': {querySelectorAll: () => rows},
    'fb-admin-empty': {classList: {toggle: (name, hidden) => {emptyHidden = hidden;}}}})[id];
''' + source('09', '            function filterAdminFeedback', '            async function loadFeedback') + '''
filter.onchange(); assert.equal(rows.filter(r => !r.hidden).length, 6);
filter.value = 'unresolved'; filter.onchange();
assert.deepEqual(rows.filter(r => !r.hidden).map(r => r.dataset.feedbackStatus), statuses.slice(0, 3));
for (const status of statuses) {
    filter.value = status; filter.onchange();
    assert.deepEqual(rows.filter(r => !r.hidden).map(r => r.dataset.feedbackStatus), [status]);
}
filter.value = 'new'; rows[0].dataset.feedbackStatus = 'resolved'; filter.onchange();
assert.equal(emptyHidden, false); assert.ok(rows.every(r => r.hidden));
filter.value = 'all'; filter.onchange(); assert.equal(emptyHidden, true);
''')
