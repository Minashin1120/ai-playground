"""Run composer interactions with a small DOM harness and synthetic files."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / 'static/js/chat_core_parts'


def section(part, start, end):
    source = next(PARTS.glob(f'chat_core.part{part}_*.js')).read_text()
    return source[source.index(start):source.index(end, source.index(start))]


def run_js(source):
    result = subprocess.run(['node', '-e', source], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_reset_clears_rows_before_preview_and_invalidates_pending_uploads():
    source = section('11', 'let uploadProgressState', 'async function uploadMaskFile')
    run_js('''
const assert = require('node:assert/strict');
let currentImageUrls = ['synthetic.png'], currentMaskImage = null;
const browserFastLocalFiles = new Map(), markerAppliedUploads = new Set();
const list = {innerHTML: 'old attachment row'};
const get = id => id === 'upload-list' ? list : null;
function updateFilePreview() { assert.ok(!list.innerHTML.includes('old attachment row')); }
function updateMaskPreview() {}
''' + source + '''
const pending = {generation: uploadGeneration, uploadId: 'old', row: {isConnected: true}};
assert.equal(isUploadCancelled(pending), false);
resetUploadState();
assert.deepEqual(currentImageUrls, []);
assert.equal(isUploadCancelled(pending), true);
assert.equal(isUploadCancelled({uploadId: 'removed', row: {isConnected: false}}), true);
assert.equal(isUploadCancelled({generation: uploadGeneration, row: {isConnected: true}}), false);
''')


def test_restoring_prompt_cache_setting_never_sends_manual_change():
    source = section('04', 'const PROMPT_OPTION_INPUT_IDS', '// Kept in memory') + section('04', 'function restorePromptOptionsForEdit', 'function beginEditMessage')
    run_js('''
const assert = require('node:assert/strict');
let modelChosenManually = false, refreshed = 0, changed = 0;
const cache = {type: 'checkbox', checked: false, dispatchEvent() { changed++; }};
const get = id => id === 'enable-prompt-cache' ? cache : null;
const window = {}, updatePromptCacheUi = () => refreshed++, applyMcpPromptChipUi = () => {};
''' + source + '''
for (const enabled of [false, true, false]) {
    restorePromptOptionsForEdit({prompt_options: {enable_prompt_caching: enabled}});
    assert.equal(cache.checked, enabled);
}
assert.equal(refreshed, 3);
assert.equal(changed, 0);
''')


def test_minimal_thinking_opens_slider_without_toggling_and_off_is_selectable():
    source = section('03', 'function activeThinkingSelect', 'function buildMinimalOptionItem')
    raw = next(PARTS.glob('chat_core.part03_*.js')).read_text()
    handler = raw[raw.index('function handleMinimalOptionClick'):raw.index('function handleMinimalOptionClick') + raw[raw.index('function handleMinimalOptionClick'):].index('            if (minimalOptionDisabled(item)) return;')] + '}'
    run_js('''
const assert = require('node:assert/strict');
const chk = {checked: true, disabled: false};
const select = {value: 'high', options: ['low', 'high'].map(value => ({value, textContent: value, classList: {contains: () => false}}))};
const opts = {dataset: {thinkingMode: 'thinking'}}, budget = {disabled: true};
const get = id => ({'enable-thinking': chk, 'thinking-options': opts, 'thinking-level': select, 'thinking-budget': budget})[id];
let opened = 0;
function closeMinimalOptions() {}
function showThinkingSlider() { opened++; }
''' + source + handler + '''
handleMinimalOptionClick({special: 'thinking', checkboxId: 'enable-thinking'});
assert.equal(chk.checked, true);
assert.equal(opened, 1);
assert.deepEqual(thinkingSliderOptions().map(o => o.value), ['off', 'low', 'high']);
chk.checked = false;
assert.equal(thinkingSliderValue(), 'off');
assert.equal(currentThinkingLevelLabel(), 'Off');
chk.disabled = true;
assert.deepEqual(thinkingSliderOptions().map(o => o.value), ['low', 'high']);
budget.disabled = false;
assert.equal(thinkingUsesLevelSlider(), false);
''')


def test_late_upload_response_cannot_reattach_after_reset():
    state = section('11', 'let uploadProgressState', 'async function uploadMaskFile')
    upload = section('12', 'function uploadFileWithProgress', 'async function')
    run_js('''
const assert = require('node:assert/strict');
let currentImageUrls = [], currentMaskImage = null;
const browserFastLocalFiles = new Map(), markerAppliedUploads = new Set();
const get = () => null, updateFilePreview = () => {}, updateMaskPreview = () => {};
const window = {}, CHAT_CONFIG = {urls: {upload: '/upload'}}, csrfToken = 'synthetic';
const CHUNK_THRESHOLD_BYTES = 200;
let xhr;
class XMLHttpRequest {
 constructor() { xhr = this; this.upload = {}; }
 open() {} setRequestHeader() {} send() {}
}
class FormData { append() {} }
''' + state + upload + '''
(async () => {
 const pending = uploadFileWithProgress({size: 1}, {generation: uploadGeneration, uploadId: 'old', row: {isConnected: true}});
 resetUploadState();
 xhr.status = 200; xhr.responseText = '{"filename":"synthetic.png"}'; xhr.onload();
 assert.equal(await pending, false);
 assert.deepEqual(currentImageUrls, []);
})().catch(e => { console.error(e); process.exitCode = 1; });
''')


def test_slider_can_disable_and_reenable_thinking_without_extra_panel_visit():
    helpers = section('03', 'function activeThinkingSelect', 'function buildMinimalOptionItem')
    script = next(PARTS.glob('chat_core.part04_*.js')).read_text()
    start = script.index("                slider.addEventListener('input'")
    handler = script[start:script.index("            const closeSlider", start)].rsplit('            }', 1)[0]
    run_js('''
const assert = require('node:assert/strict');
const chk = {checked: true, disabled: false, dispatchEvent() {}};
const select = {value: 'high', options: ['low', 'high'].map(value => ({value, classList: {contains: () => false}})), dispatchEvent() {}};
const get = id => ({'enable-thinking': chk, 'thinking-options': {dataset: {thinkingMode: 'thinking'}}, 'thinking-level': select})[id];
let input;
const slider = {value: '0', addEventListener(event, callback) { input = callback; }};
function allowedThinkingValues() { return thinkingSliderOptions().map(o => o.value); }
function syncThinkingSliderUi() {} function scheduleThinkingSliderHide() {}
''' + helpers + handler + '''
input(); assert.equal(chk.checked, false); assert.equal(select.value, 'high');
slider.value = '1'; input(); assert.equal(chk.checked, true); assert.equal(select.value, 'low');
slider.value = '2'; input(); assert.equal(chk.checked, true); assert.equal(select.value, 'high');
''')
