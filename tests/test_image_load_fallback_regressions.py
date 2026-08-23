#!/usr/bin/env python3
"""Regressions for the file / thumbnail image load fallback.

Uploaded images are served from /files/ and /files/thumb/.  A document-level
error listener turns a failed image into an explicit warning.  The listener must
NOT replace a genuinely-available image with "ファイルがありません" just because
the first load failed transiently (or a thumbnail could not be decoded): it
retries the load and, for a thumbnail URL, falls back to the original full
/files/ image that still opens.  Only a real miss (404/410/403) or a key
mismatch (409) may produce the warning.
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]


def _current_chat_core() -> Path:
    assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
    if not assets:
        raise AssertionError("chat_core.v4.8.*.js not found")
    return assets[-1]


class ImageLoadFallbackRegressionTests(unittest.TestCase):
    def test_listener_retries_available_file_and_falls_back_to_full_image(self):
        source = _current_chat_core().read_text(encoding="utf-8")
        start = source.index("// --- File/image load fallback")
        end = source.index("const isAdminSidebarDebugEnabled")
        iife = source[start:end]

        # --- Node harness --------------------------------------------------
        # Runs the real IIFE in a mock DOM/fetch, then triggers synthetic
        # image 'error' events and inspects the resulting replacement.
        harness = r"""
class FakeEl {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase();
    this.style = {};
    this._innerHTML = '';
    this.attrs = {};
    this.replacedWith = null;
  }
  get innerHTML() { return this._innerHTML; }
  set innerHTML(v) { this._innerHTML = v; }
  getAttribute(n) { return this.attrs[n] !== undefined ? this.attrs[n] : null; }
  setAttribute(n, v) { this.attrs[n] = String(v); }
  replaceWith(node) { this.replacedWith = node; }
}
class FakeImg extends FakeEl {
  constructor(src) { super('img'); this.src = src || ''; this.currentSrc = this.src; }
  getAttribute(n) { if (n === 'src') return this.src; return this.attrs[n] !== undefined ? this.attrs[n] : null; }
  setAttribute(n, v) {
    if (n === 'src') { this.src = String(v); this.currentSrc = String(v); return; }
    this.attrs[n] = String(v);
  }
  cloneNode() { const c = new FakeImg(this.src); c.attrs = Object.assign({}, this.attrs); c.replacedWith = null; return c; }
}

let errorHandler = null;
global.document = {
  addEventListener(type, handler) { if (type === 'error') errorHandler = handler; },
  createElement(tag) { return new FakeEl(tag); },
};
let statusQueue = [];
global.fetch = function () { const s = statusQueue.shift(); return Promise.resolve({ status: (s === undefined ? 200 : s) }); };

""" + iife + r"""

function fail(code, msg) { console.error('FAIL ' + code + ': ' + msg); process.exit(code); }
const flush = () => new Promise((r) => setTimeout(r, 30));
async function dispatch(img, statuses) {
  statusQueue = statuses;
  errorHandler({ target: img, stopImmediatePropagation() {}, preventDefault() {} });
  await flush();
}

(async () => {
  // 200 (already served) => retry the load, never "ファイルがありません".
  let img = new FakeImg('/files/thumb/1/a.png');
  await dispatch(img, [200]);
  if (!img.replacedWith || img.replacedWith.tagName !== 'IMG') fail(10, '200 status must retry with a fresh <img>');
  if (img.replacedWith.getAttribute('data-file-retry') !== '1') fail(11, 'retry count not recorded');
  if (!/retry=/.test(img.replacedWith.src)) fail(12, 'cache-buster missing on retry');

  // Repeated thumbnail failure (still 200) => fall back to the full /files/ image.
  img = new FakeImg('/files/thumb/1/a.png');
  await dispatch(img, [200]);            // retry 1
  img = img.replacedWith;
  await dispatch(img, [200]);            // retry 2
  img = img.replacedWith;
  await dispatch(img, [200]);            // exhausted -> full-file fallback
  img = img.replacedWith;
  if (!img || img.tagName !== 'IMG') fail(20, 'thumbnail must fall back to an <img>');
  if (!img.src.startsWith('/files/1/a.png')) fail(21, 'full-file fallback URL not used: ' + img.src);
  if (String(img.src).includes('/files/thumb/')) fail(22, 'fallback still points at /files/thumb/');

  // 404 (genuinely missing) => "ファイルがありません", no retry.
  img = new FakeImg('/files/1/b.png');
  await dispatch(img, [404]);
  if (!img.replacedWith || img.replacedWith.tagName !== 'DIV') fail(30, '404 must show the warning box');
  if (img.replacedWith.innerHTML.includes('暗号キー')) fail(31, '404 must not show key-mismatch');
  if (!img.replacedWith.innerHTML.includes('ファイルがありません')) fail(32, '404 must say ファイルがありません');

  // 409 (key mismatch) => key-mismatch warning, no retry.
  img = new FakeImg('/files/1/c.png');
  await dispatch(img, [409]);
  if (!img.replacedWith || !String(img.replacedWith.innerHTML).includes('暗号キー')) fail(40, '409 must show key-mismatch warning');

  console.log('OK');
})().catch((e) => { console.error('UNCAUGHT ' + (e && e.stack || e)); process.exit(99); });
"""
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
            fh.write(harness)
            path = fh.name
        try:
            proc = subprocess.run(
                ["node", path],
                capture_output=True,
                text=True,
                check=False,
            )
        finally:
            Path(path).unlink(missing_ok=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("OK", proc.stdout)


if __name__ == "__main__":
    unittest.main()
