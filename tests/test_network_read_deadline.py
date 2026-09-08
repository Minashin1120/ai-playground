"""Execute the browser wrappers with stalled fetches, not source assertions."""
import subprocess
from pathlib import Path


def test_stalled_reads_release_spinner_and_offline_is_not_misreported():
    root = Path(__file__).resolve().parents[1]
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const timers = new Map(); let id = 0;
const text = {textContent: ''};
const element = {classList: {add(){}, remove(){}}, textContent: ''};
const window = {
  location: new URL('https://example.test/'),
  setTimeout(fn, ms) { timers.set(++id, {fn, ms}); return id; },
  clearTimeout(i) { timers.delete(i); },
  setInterval() {return 1;}, clearInterval(){}, addEventListener(){},
  fetch(url, opts) {
    return new Promise((resolve, reject) => {
      opts.signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
    });
  }
};
const navigator = {onLine: true};
const document = {readyState: 'complete', addEventListener(){}, body: element,
  getElementById(id) {return id === 'offline-banner-text' ? text : element;}};
const ctx = vm.createContext({window, document, navigator, URL, AbortController, DOMException, Element: class {}});
vm.runInContext(fs.readFileSync('static/js/progress_spinner.js', 'utf8'), ctx);
(async () => {
  const pending = window.fetch('/api/sessions').catch(e => e.name);
  assert.equal(window.ProgressSpinner.getActiveCount(), 1);
  const deadline = [...timers.values()].find(t => t.ms === 30000);
  assert.ok(deadline);
  deadline.fn();
  assert.equal(await pending, 'AbortError');
  assert.equal(window.ProgressSpinner.getActiveCount(), 0);
  assert.ok(![...timers.values()].some(t => t.ms === 30000));
  const controller = new AbortController();
  const cancelled = window.fetch('/api/settings', {signal: controller.signal}).catch(e => e.name);
  controller.abort();
  assert.equal(await cancelled, 'AbortError');
  assert.equal(window.ProgressSpinner.getActiveCount(), 0);
  vm.runInContext(fs.readFileSync('static/js/connection_monitor.js', 'utf8'), ctx);
  window.ConnectionMonitor.setUnavailable('offline');
  assert.ok(!text.textContent.includes('インターネット接続が切断'));
  assert.ok(text.textContent.includes('このサイト'));
  navigator.onLine = false;
  window.ConnectionMonitor.setUnavailable('offline');
  assert.ok(text.textContent.includes('インターネット接続が切断'));
})().catch(e => {console.error(e); process.exitCode = 1;});
"""
    subprocess.run(['node', '-e', script], cwd=root, check=True, timeout=10)
