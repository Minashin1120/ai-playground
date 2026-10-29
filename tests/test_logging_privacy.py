"""Boundary tests use synthetic secrets only; never inspect production logs."""
import json
import logging
from pathlib import Path
from unittest import mock

from server.logging_policy import SCHEMA, safe_entry, ContentFreeFilter
from tests.test_activity_log_regressions import _run_node

ROOT = Path(__file__).resolve().parents[1]
SECRET = 'PRIVATE_PROMPT_ANSWER_TITLE_FILENAME_KEY'


def test_schema_copies_match():
    web = (ROOT / 'static/js/activity_log.js').read_text().split('var POLICY = ', 1)[1].split(';\n', 1)[0]
    assert json.loads(web) == SCHEMA
    android = (ROOT / 'android/app/src/main/java/com/minashin1120/aiplayground/data/LogPolicy.kt').read_text()
    literal = android.split('private const val POLICY_JSON = """', 1)[1].split('"""', 1)[0]
    assert json.loads(literal.replace("${'$'}", '$')) == SCHEMA


def test_no_free_text_nested_objects_or_unknown_values():
    safe = safe_entry(dict(ev='send.error', t=123, ms=40, failed=True, status=500,
        message=SECRET, title=SECRET, filename=SECRET, model=SECRET, path='/' + SECRET,
        op=SECRET, stack=[SECRET], detail={'bytes': SECRET}, **{SECRET: SECRET}))
    assert safe == {'ev': 'send.error', 't': 123, 'ms': 40, 'failed': True, 'status': 500}
    assert safe_entry({'ev': SECRET, 'ms': 10}) is None
    assert safe_entry({'ev': 'http', 'ms': float('nan'), 'status': '200'}) == {'ev': 'http'}


def test_dns_fallback_records_phases_without_names_addresses_or_notifications():
    for phase in ('start', 'done', 'error'):
        entry = safe_entry(dict(ev='dns.fallback', phase=phase, ms=12, count=2,
            hostname=SECRET, address=SECRET, message=SECRET, response=SECRET))
        assert entry == {'ev': 'dns.fallback', 'phase': phase, 'ms': 12, 'count': 2}


def test_operations_log_never_formats_args_or_exception():
    class SensitiveObject:
        def __str__(self):
            raise AssertionError('must not format arguments')
    try:
        raise ValueError(SECRET)
    except ValueError:
        import sys
        record = logging.LogRecord('sdk', logging.ERROR, __file__, 41, SECRET + '%s', (SensitiveObject(),), sys.exc_info())
    ContentFreeFilter().filter(record)
    assert SECRET not in logging.Formatter('%(message)s').format(record)
    assert record.exc_info is None
    assert json.loads(record.getMessage())['frames']


def test_web_filters_console_errors_toasts_legacy_storage_and_idle_expiry():
    _run_node(r"""
load();
const log = window.ActivityLog;
log.setEnabled(true);
const secret = 'PRIVATE_PROMPT_ANSWER_TITLE_FILENAME_KEY';
window.console.error(secret, new Error(secret));
listeners.error.forEach(fn => fn({ message: secret, filename: secret, error: new Error(secret), lineno: 4 }));
listeners.unhandledrejection.forEach(fn => fn({ reason: new Error(secret) }));
log.log('http', { title: secret, message: secret, path: '/api/threads/' + secret, status: 500, ms: 7 });
let entries = log.recent();
assert.ok(!JSON.stringify(entries).includes(secret));
assert.ok(entries.some(e => e.ev === 'http' && e.status === 500));
store.set('aip_activity_log_v1', JSON.stringify([
  { t: Date.now() - 3600001, ev: 'http', status: 200 },
  { t: Date.now(), ev: 'error', message: secret, title: secret }
]));
log.stats();
const stored = store.get('aip_activity_log_v1');
assert.ok(!stored.includes(secret));
assert.equal(JSON.parse(stored).length, 1);
assert.ok(!('message' in JSON.parse(stored)[0]));
""")


def test_stack_and_window_details_are_bounded_and_exclude_arbitrary_text():
    raw = {'ev': 'ui.selection_windows', 'windows': [
        {'window': 'DecorView', 'x': 1, 'y': 2, 'w': 300, 'h': 600, 'title': SECRET},
        {'window': SECRET, 'x': 1, 'y': 2, 'w': 3, 'h': 4}],
        'stack': [{'code_file': 'ChatViewModel.kt', 'line': 20, 'message': SECRET},
                  {'code_file': [SECRET], 'line': 99}]}
    safe = safe_entry(raw)
    assert SECRET not in json.dumps(safe)
    assert len(safe['windows']) == 1
    assert safe['stack'] == [{'code_file': 'ChatViewModel.kt', 'line': 20}]


def test_actual_job_and_thread_identifiers_remain_correlatable():
    job = 'job_1792500000_12_abcdef0123456789'
    thread = 'aBcD_0123456789-abcdefghijklmnopqrstuvxyzABCD'
    thread = thread[:43].ljust(43, 'x')
    safe = safe_entry({'ev': 'server.worker.start', 'job': job, 'thread_id': thread})
    assert safe['job'] == job
    assert safe['thread_id'] == thread
    assert 'job' not in safe_entry({'ev': 'server.worker.start', 'job': SECRET})


def test_literal_events_cannot_silently_disappear_from_the_allowlist():
    import re
    pattern = r'''(?:Diagnostics|ActivityLog)(?:\?\.|\.)(?:log|failure)\(\s*['"]([A-Za-z0-9_.]+)['"]\s*[,)]'''
    files = list((ROOT / 'android/app/src/main/java').rglob('*.kt'))
    files += list((ROOT / 'static/js/chat_core_parts').glob('*.js'))
    unknown = [(str(path.relative_to(ROOT)), event) for path in files
               for event in re.findall(pattern, path.read_text()) if event not in SCHEMA['events']]
    assert not unknown, unknown


def test_web_can_attach_server_logs_with_local_collection_disabled():
    _run_node(r"""
const toggle = { checked: true };
document.readyState = 'loading';
document.getElementById = id => id === 'set-server-diagnostic-log' ? toggle : null;
load();
assert.equal(window.ActivityLog.isEnabled(), false);
assert.equal(window.ActivityLog.canAttach(), true);
assert.deepEqual(JSON.parse(JSON.stringify(window.ActivityLog.feedbackPayload().entries)), []);
window.ActivityLog.log('click');
assert.equal(window.ActivityLog.recent().length, 0);
toggle.checked = false;
assert.equal(window.ActivityLog.feedbackPayload(), null);
""")


def test_versions_keep_web_asset_versions_without_accepting_arbitrary_text():
    assert safe_entry({'ev': 'page.load', 'version': '2026-10-21-001'})['version'] == '2026-10-21-001'
    assert safe_entry({'ev': 'app.start', 'version': '1.70.0'})['version'] == '1.70.0'
    assert 'version' not in safe_entry({'ev': 'page.load', 'version': SECRET})


def test_interaction_diagnostics_keep_only_safe_state_and_random_ids():
    op = '12345678-1234-1234-1234-123456789abc'
    raw = dict(ev='ui.interaction', op=op, parent_op=op, gesture=op,
               ui_element='math_block', interaction_stage='settled', phase='done',
               scroll_x=42, scroll_max=120, content_width=600, toolbar_requests=1,
               focused=True, action_mode_active=False, after_ms=400,
               text=SECRET, tex=SECRET, selection_text=SECRET, title=SECRET)
    safe = safe_entry(raw)
    assert safe['gesture'] == op and safe['parent_op'] == op
    assert safe['scroll_x'] == 42 and safe['scroll_max'] == 120
    assert safe['focused'] is True and safe['action_mode_active'] is False
    assert SECRET not in json.dumps(safe)
    invalid = safe_entry(dict(ev='ui.interaction', parent_op=SECRET, gesture=SECRET,
                             interaction_stage=SECRET, ui_element=SECRET, scroll_x=SECRET))
    assert invalid == {'ev': 'ui.interaction'}


def test_web_configurable_retention_expires_and_persists_without_resurrecting():
    _run_node(r"""
load();
const log = window.ActivityLog;
log.setEnabled(true);
assert.equal(log.WINDOW_MS, 3600000);
assert.equal(log.setRetentionHours(4), true);
store.set('aip_activity_log_v1', JSON.stringify([
  {t: Date.now() - 7200000, ev: 'http', status: 201},
  {t: Date.now() - 1800000, ev: 'http', status: 202}
]));
assert.ok(log.recent().some(e => e.status === 201));
assert.equal(log.feedbackPayload().window_seconds, 14400);
assert.equal(log.setRetentionHours(1), true);
assert.ok(!log.recent().some(e => e.status === 201));
assert.equal(log.setRetentionHours(8), true);
assert.ok(!log.recent().some(e => e.status === 201));
for (const value of [0, -1, 1.5, 8761, 'bad', Infinity]) assert.equal(log.setRetentionHours(value), false);
assert.equal(store.get('aip_activity_log_retention_hours'), '8');
delete window.ActivityLog;
load();
assert.equal(window.ActivityLog.WINDOW_MS, 8 * 3600000);
""")
