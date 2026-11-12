"""Offline regression checks; never send requests to a model provider."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def helpers():
    tree = ast.parse((ROOT / 'server/request_security_notices.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
             ('_prompt_context_record', '_append_turn_time_notice', '_saved_prompt_text')]
    ns = {'json': json}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<helpers>', 'exec'), ns)
    return ns


def test_saved_turn_time_replays_without_using_current_clock():
    render = helpers()['_append_turn_time_notice']
    record = {'turn_time_notice': 'Current time: 2026-11-03 12:00:00 JST (UTC+0900)'}
    first = render('hello', record)
    assert render('hello', json.dumps(record)) == first
    assert render('hello', record) == first
    assert first.startswith('hello\n\n[Turn context]\nCurrent time:')
    other = dict(record, turn_time_notice='Current time: 2026-11-03 13:00:00 JST (UTC+0900)')
    assert render('hello', other) != first
    assert render('hello', record) == first


def test_legacy_and_invalid_metadata_do_not_invent_history():
    render = helpers()['_append_turn_time_notice']
    for raw in (None, '', '{', '[]', '{}', {'turn_time_notice': False},
                {'turn_time_notice': 'untrusted'}, {'turn_time_notice': 'Current time: ' + 'x' * 100}):
        assert render('original', raw) == 'original'


def test_mcp_definitions_have_stable_order_without_mutating_catalog():
    tree = ast.parse((ROOT / 'mcp_service/execution.py').read_text())
    method = next(n for cls in tree.body if isinstance(cls, ast.ClassDef)
                  for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'tool_metas')
    ns = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<tools>', 'exec'), ns)
    a, b = SimpleNamespace(internal_name='a'), SimpleNamespace(internal_name='b')
    runtime = SimpleNamespace(load=lambda: None, servers=[{'tools': [b, a]}])
    assert ns['tool_metas'](runtime) == [a, b]
    assert runtime.servers[0]['tools'] == [b, a]
    runtime.servers.reverse()
    runtime.servers[0]['tools'].reverse()
    assert ns['tool_metas'](runtime) == [a, b]


def test_saved_quote_replays_only_on_unchanged_turn():
    replay = helpers()['_saved_prompt_text']
    record = {'prompt_text': 'Context (User Quote): quoted\nUser Message: original',
              'prompt_source': 'original', 'prompt_quote': 'quoted'}
    assert replay(json.dumps(record), 'original', 'quoted') == record['prompt_text']
    assert replay(record, 'edited', 'quoted') is None
    assert replay(record, 'original', 'different quote') is None
    assert replay({}, 'original', 'quoted') is None
