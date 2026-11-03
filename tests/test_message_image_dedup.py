"""Render complete bubbles so inline images cannot reappear in the attachment grid."""
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "static/js/chat_core_parts"


def rendering_harness():
    source = (PARTS / "chat_core.part13_canvas_coding_stream.js").read_text()
    settings = (PARTS / "chat_core.part05_settings_modal.js").read_text()
    files = (PARTS / "chat_core.part11_file_preview_upload.js").read_text()
    helpers = source[source.index("function normalizeMarkdownNewlines(text)"):source.index("function openPythonExecDetail(id)")]
    render = source[source.index("function renderMessage("):source.index("function showTokenDetailModal(")]
    normalize = settings[settings.index("const normalizeAttachmentPath ="):settings.index("const isGeminiImageModelKey =")]
    urls = files[files.index("function buildFileUrl("):files.index("window.closeMarkerModal =")]
    return r'''
const assert = require('node:assert/strict');
const window = {location: {origin: 'https://playground.test'}};
const FILE_BASE_URL = 'https://playground.test/files/';
const FILE_THUMB_BASE_URL = 'https://playground.test/files/thumb/';
const messageStore = {}, messageMeta = {};
const document = {createElement: () => ({innerHTML: '', isConnected: false})};
const get = () => null;
const escapeHtml = value => String(value);
const normalizeAttachmentSource = () => 'unknown';
const setAttachmentSourceForPath = () => {};
const isImagePath = path => /.(png|jpg|jpeg|gif|webp)$/i.test(path);
let attached = [], markdown = '';
const buildChatImageHtml = src => {attached.push(src); return '<img data-attachment>';};
const buildAiMarkdownHtml = text => {markdown = text; return '<div class="content-area">' + text + '</div>';};
''' + normalize + urls + helpers + render + r'''
function bubble(text, files, role = 'assistant') {
    attached = []; markdown = '';
    const element = renderMessage('1', role, text, JSON.stringify(files), null, 'model');
    return {element, attached: [...attached], markdown, runs: messageMeta['1'].python_executions};
}
'''


class MessageImageDedupTests(unittest.TestCase):
    def test_inline_and_python_detail_images_are_not_rendered_as_attachments(self):
        script = rendering_harness() + r'''
const image = '/files/1/result.png';
const run = images => '\n```pyexec\n' + JSON.stringify({code: 'display(result)', output: '', images}) + '\n```';
// A late citation: empty execution images, a download link, then Agentic View.
const raw = '[編集済み画像をダウンロード](' + image + ')\n![Agentic View](' + image + ')\n' + run([]);
let result = bubble(raw, ['1/result.png']);
assert.equal((result.markdown.match(/!\[/g) || []).length, 1);
assert.equal(result.attached.length, 0);
assert.ok(!result.element.innerHTML.includes('image-grid'));
result = bubble('![result](' + image + ')', ['1/result.png', {filepath: '1/other.png'}]);
assert.equal(result.attached.length, 1);
assert.ok(result.attached[0].endsWith('other.png'));
result = bubble(run(['/files/1/step.png']) + run([image]), ['1/step.png', '1/result.png']);
assert.equal(result.attached.length, 0);
assert.equal(result.runs.length, 2);
assert.deepEqual(result.runs[0].images, ['/files/1/step.png']);
const three = [1,2,3].map(i => '/files/1/result' + i + '.png');
result = bubble(run(three), three.map(path => path.replace('/files/', '')));
assert.equal(result.attached.length, 0);
assert.equal((result.markdown.match(/!\[/g) || []).length, 3);
assert.equal(bubble('', ['1/result.png']).attached.length, 1);
assert.equal(bubble('![user](' + image + ')', ['1/result.png'], 'user').attached.length, 1);
assert.equal(bubble('```md\n![example](' + image + ')\n```', ['1/result.png']).attached.length, 1);
assert.equal(bubble('![external](https://other.test/files/1/result.png)', ['1/result.png']).attached.length, 1);
assert.equal(bubble('![invalid](https://%)', ['1/result.png']).attached.length, 1);
'''
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
