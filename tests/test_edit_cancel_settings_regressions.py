"""Exercise the real edit/cancel functions against changed, unsent composer settings."""
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PART = ROOT / 'static/js/chat_core_parts/chat_core.part04_minimal_options_thinking.js'


class EditCancelSettingsTests(unittest.TestCase):
    def test_cancel_returns_to_before_first_edit_instead_of_latest_prompt(self):
        source = PART.read_text()
        block = source[source.index('        const PROMPT_OPTION_INPUT_IDS'):source.index('        function playSendAnimation()')]
        send_source = (ROOT / 'static/js/chat_core_parts/chat_core.part14_send_message_browser_fast.js').read_text()
        send_start = send_source[send_source.index('            const editingId = editingMessageId;'):send_source.index('            playSendAnimation();', send_source.index('            const editingId = editingMessageId;'))]
        failure_start = send_source.index('                if (!requestAccepted) {\n                    const restoreEditing')
        failure_restore = send_source[failure_start:send_source.index("                // Restore old message", failure_start)]
        send_finally = next(line.strip() for line in send_source.splitlines() if 'abortController=null; currentJobId=null; setEditUi' in line)
        script = r'''
const assert = require('node:assert/strict');
const controls = new Map();
function control(id, value = '', checked = false, type = 'text') {
    const el = {value, checked, type, dataset: {}, style: {}, scrollHeight: 24,
        focus() {}, dispatchEvent() {}, classList: {add() {}, remove() {}}};
    controls.set(id, el); return el;
}
const get = id => controls.get(id);
control('model-select', 'before-model');
control('thinking-level', 'low'); control('reasoning-effort', 'medium');
control('enable-thinking', '', true, 'checkbox');
control('enable-search', '', false, 'checkbox');
control('enable-prompt-cache', '', true, 'checkbox');
control('enable-batch-mode', '', false, 'checkbox');
control('enable-sys-prompt', '', true, 'checkbox').dataset.restoreChecked = '1';
control('tts-speed', '1.25');
control('tts-style', 'warm'); control('ideogram-image-negative', 'blur'); control('xai-stop', 'END');
control('prompt-input'); control('file-preview'); control('file-input');
let modelChosenManually = false, editingMessageId = null, currentParentId = null;
let abortController = null, currentJobId = null;
let currentLeafId = 33, currentImageUrls = [], currentQuote = '', currentQuoteMessageId = null;
let currentVisionModel = 'before-vision', currentMaskImage = 'before-mask';
let canvasModeEnabled = true, codingModeEnabled = false, browserFastModeEnabled = false;
let codingTargetSelection = {key: 'before-target'};
const allMessages = [
 {id: 10, role: 'user', model: 'old-model', content: 'old', prompt_options: {
   thinking_level: 'high', reasoning_effort: 'high', enable_search: true, enable_prompt_caching: false,
   canvas_mode: false, coding_mode: true, batch_mode: true, tts_speed: '0.5', image_vision_model: 'old-vision'}},
 {id: 32, role: 'user', model: 'latest-model', prompt_options: {thinking_level: 'high', enable_search: true}},
 {id: 33, role: 'assistant', parent_id: 32},
];
const messageStore = {10: 'old', 32: 'latest'}, messageMeta = {};
const window = {toggleOptions() {}};
const selectModelById = model => {get('model-select').value = model; get('enable-sys-prompt').checked = false;};
const syncCanvasModeUi = on => canvasModeEnabled = on;
const syncCodingModeUi = on => codingModeEnabled = on;
const setBrowserFastModeEnabled = on => browserFastModeEnabled = on;
const updatePromptCacheUi = () => {}, applyMcpPromptChipUi = () => {};
const updateMaskPreview = () => {}, schedulePromptTokenEstimate = () => {}, setEditUi = () => {};
const clearQuote = () => { currentQuote = ''; currentQuoteMessageId = null; };
BLOCK
beginEditMessage(10);
assert.equal(get('thinking-level').value, 'high');
assert.equal(get('model-select').value, 'old-model');
get('thinking-level').value = 'minimal';
get('tts-style').value = 'bright'; get('ideogram-image-negative').value = 'noise'; get('xai-stop').value = 'STOP';
get('enable-search').checked = false;
beginEditMessage(32); // Do not replace the original snapshot when choosing a second edit target.
cancelEdit();
assert.equal(get('thinking-level').value, 'low');
assert.equal(get('reasoning-effort').value, 'medium');
assert.equal(get('model-select').value, 'before-model');
assert.equal(get('enable-search').checked, false);
assert.equal(get('enable-prompt-cache').checked, true);
assert.equal(get('enable-batch-mode').checked, false);
assert.equal(get('tts-speed').value, '1.25');
assert.equal(get('tts-style').value, 'warm');
assert.equal(get('ideogram-image-negative').value, 'blur');
assert.equal(get('xai-stop').value, 'END');
assert.equal(get('enable-sys-prompt').checked, true);
assert.equal(get('enable-sys-prompt').dataset.restoreChecked, '1');
assert.equal(currentVisionModel, 'before-vision');
assert.equal(currentMaskImage, 'before-mask');
assert.equal(canvasModeEnabled, true);
assert.equal(codingModeEnabled, false);
assert.equal(editingMessageId, null);
assert.equal(promptSettingsBeforeEdit, null);
assert.equal(get('prompt-input').value, ''); // Draft clearing keeps its existing behavior.
assert.equal(allMessages[0].prompt_options.thinking_level, 'high');

// A fresh edit captures fresh settings, and cancel also restores the prior manual-model flag.
modelChosenManually = true;
get('model-select').value = 'manual-before';
get('thinking-level').value = 'medium';
beginEditMessage(10);
assert.equal(get('model-select').value, 'manual-before');
get('model-select').value = 'manual-during-edit';
get('enable-prompt-cache').checked = false;
cancelEdit();
assert.equal(get('model-select').value, 'manual-before');
assert.equal(get('thinking-level').value, 'medium');
assert.equal(get('enable-prompt-cache').checked, true);
assert.equal(modelChosenManually, true);
get('thinking-level').value = 'new-choice';
cancelEdit();
assert.equal(get('thinking-level').value, 'new-choice');
beginEditMessage(999); // An invalid edit must not capture settings.
assert.equal(promptSettingsBeforeEdit, null);
function sendStart() { SEND_START; return {editingId, settingsBeforeSubmission, capturedParentId}; }
modelChosenManually = false;
get('thinking-level').value = 'low';
beginEditMessage(10);
assert.ok(promptSettingsBeforeEdit);
sendStart();
assert.equal(promptSettingsBeforeEdit, null);
assert.equal(editingMessageId, null);
get('thinking-level').value = 'after-send';
cancelEdit();
assert.equal(get('thinking-level').value, 'after-send');
// If the server rejects the send, the edit and its original settings snapshot are resumed.
get('thinking-level').value = 'low';
beginEditMessage(10);
const failedSubmission = sendStart();
assert.equal(editingMessageId, null);
function failSend(canRetry = true) {
    const {editingId, settingsBeforeSubmission, capturedParentId} = failedSubmission;
    const requestAccepted = false;
    const sentPrompt = {canRetry: () => canRetry, restore() {}};
    FAILURE_RESTORE
}
failSend(false);
assert.equal(editingMessageId, null); // A navigation or new input wins over the old failure.
failSend();
SEND_FINALLY
assert.equal(editingMessageId, 10);
assert.ok(promptSettingsBeforeEdit);
cancelEdit();
assert.equal(get('thinking-level').value, 'low');
'''.replace('BLOCK', block).replace('SEND_START', send_start).replace('FAILURE_RESTORE', failure_restore).replace('SEND_FINALLY', send_finally)
        result = subprocess.run(['node', '-e', script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
