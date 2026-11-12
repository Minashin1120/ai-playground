"""Exercise camera upload failures without a server or real user files."""
import json
from pathlib import Path
from tests.app_source import read_chat_core_source
import shutil
import subprocess
import unittest

PARTS = Path(__file__).resolve().parents[1] / 'static/js/chat_core_parts'


def source_between(filename, start, end):
    source = read_chat_core_source()
    return source[source.index(start):source.index(end, source.index(start))]


class CameraUploadRecoveryTests(unittest.TestCase):
    def run_js(self, source, scenario):
        if not shutil.which('node'):
            self.skipTest('Node.js is required')
        result = subprocess.run(
            ['node', '-e', source + '\n' + scenario], capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])

    def test_proxy_failure_uses_chunks_only_for_camera_and_preserves_errors(self):
        source = source_between('chat_core.part13_upload_camera_canvas.js',
                                'function uploadFileWithProgress(', 'function isVideoFile(')
        self.run_js(source, r'''
const assert = require('node:assert/strict');
const CHUNK_THRESHOLD_BYTES = 20 * 1024 * 1024;
const CHAT_CONFIG = {urls: {upload: '/upload'}}, csrfToken = 'test';
let status, response, cancelled = false, chunks = 0, started = 0, ended = 0;
const window = {ConnectionMonitor: {
    operationStarted() { started++; }, operationEnded() { ended++; }
}};
function isUploadCancelled() { return cancelled; }
function showToast() {}
async function uploadFileChunked() { chunks++; return true; }
class XMLHttpRequest {
    constructor() { this.upload = {}; }
    open() {} setRequestHeader() {}
    send() {
        this.status = status; this.responseText = response;
        queueMicrotask(() => this.onload());
    }
}
class FormData { append() {} }
(async () => {
    const row = {status: {}}, file = {size: 200};
    status = 520; response = '<html>proxy failure</html>';
    assert.equal(await uploadFileWithProgress(file, row, {cameraCapture: true}), true);
    assert.equal(chunks, 1);
    assert.equal(await uploadFileWithProgress(file, row), false);
    assert.equal(chunks, 1);
    status = 413; response = JSON.stringify({error: 'Storage limit exceeded'});
    assert.equal(await uploadFileWithProgress(file, row, {cameraCapture: true}), false);
    status = 503; response = JSON.stringify({error: 'Application failure'});
    assert.equal(await uploadFileWithProgress(file, row, {cameraCapture: true}), false);
    assert.equal(chunks, 1);
    cancelled = true;
    assert.equal(await uploadFileWithProgress(file, row, {cameraCapture: true}), false);
    assert.equal(chunks, 1);
    assert.equal(started, ended);
})().catch(error => { console.error(error); process.exitCode = 1; });
''')

    def test_camera_reports_only_successful_files(self):
        source = source_between('chat_core.part12_file_preview_upload.js',
                                'async function attachCameraCapturedFiles(', 'function openUploadModal(')
        # Run each result independently, including complete and partial failure.
        for results in ([False, False], [True, False], [True, True]):
            with self.subTest(results=results):
                self.run_js(source, r'''
const assert = require('node:assert/strict');
let cameraCaptureBusy = false, opened = 0, cleared = 0;
const cameraCapturePendingFiles = [{name: 'one'}, {name: 'two'}];
const toasts = [];
function get(id) { return {checked: false, classList: {contains() { return true; }}}; }
function closeCameraCaptureModal() {}
function setCameraCaptureStatus() {}
function updateCameraCapturePendingUi() {}
function resetCameraCapturePending() { cleared++; }
function openUploadModal() { opened++; }
function showToast(message, kind) { toasts.push({message, kind}); }
''' + f'const results = {json.dumps(results)};\n' + r'''
async function handleFiles(files, opts) {
    assert.equal(opts.cameraCapture, true);
    return results;
}
(async () => {
    await attachCameraCapturedFiles();
    const count = results.filter(Boolean).length;
    const successes = toasts.filter(t => t.kind === 'success');
    assert.equal(successes.length, count ? 1 : 0);
    if (count) assert.equal(successes[0].message, `${count}枚の画像を添付しました`);
    assert.equal(opened, count < results.length ? 1 : 0);
    assert.equal(cleared, 1);
    assert.equal(cameraCaptureBusy, false);
})().catch(error => { console.error(error); process.exitCode = 1; });
''')

    def test_failed_camera_keeps_bytes_for_retry_without_reuploading_successes(self):
        source = source_between('chat_core.part13_upload_camera_canvas.js',
                                'async function handleFiles(', "get('clear-file-btn').onclick")
        self.run_js(source, r'''
const assert = require('node:assert/strict');
let uploadGeneration = 1, browserFastModeEnabled = false, compressionCount = 0;
const UPLOAD_CONCURRENCY = 2;
const uploadProgressState = {total: 0, active: 0, completed: 0, perFilePct: {}};
const rows = [], sent = [], revoked = [];
const URL = {revokeObjectURL(url) { revoked.push(url); }};
const document = {createElement() { return {}; }};
function get() { return {checked: false, value: 'test-model'}; }
function openUploadModal() {} function syncUploadRowsFromCurrent() {}
function updateFilePreview() {} function showToast() {}
function isAudioFile() { return false; } function isVideoFile() { return false; }
function getModelMediaSupport() { return {audio: true, video: true}; }
function readComposerCompression() { return {}; }
async function applyImageCompression(file) { compressionCount++; return {...file, compressed: true}; }
function updateUploadRowFile() {}
function isUploadCancelled(row) { return row.generation !== uploadGeneration; }
function addUploadRow(file) {
    const row = {uploadId: String(rows.length), file, status: {}, bar: {}, buttons: [], removed: false};
    row.row = {
        getAttribute() { return 'blob:test'; }, remove() { row.removed = true; },
        appendChild(button) { row.buttons.push(button); }
    };
    rows.push(row);
    return row;
}
async function uploadFileWithProgress(file) { sent.push(file); return file.ok; }
(async () => {
    const results = await handleFiles([
        {name: 'good', type: 'image/jpeg', ok: true},
        {name: 'bad', type: 'image/jpeg', ok: false}
    ], {cameraCapture: true, openModal: false});
    assert.deepEqual(results, [true, false]);
    assert.equal(rows[0].buttons.length, 0);
    assert.equal(rows[1].buttons.length, 1);
    const retained = sent[1];
    retained.ok = true;
    rows[1].buttons[0].onclick();
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(sent.length, 3);
    assert.equal(sent[2], retained);
    assert.equal(sent.filter(f => f.name === 'good').length, 1);
    assert.equal(compressionCount, 2);
    assert.equal(rows[1].removed, true);
    assert.equal(rows[2].buttons.length, 0);
    assert.equal(uploadProgressState.active, 0);
    assert.deepEqual(revoked, ['blob:test']);
    uploadGeneration++;
    rows[1].buttons[0].onclick();
    assert.equal(sent.length, 3);
})().catch(error => { console.error(error); process.exitCode = 1; });
''')
