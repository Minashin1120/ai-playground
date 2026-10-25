"""Exercise counting with fabricated inputs; never load accounts or call a provider."""
import ast
import base64
import hashlib
import json
import mimetypes
from pathlib import Path
import secrets
import threading
import time
from types import SimpleNamespace
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]


def namespace():
    scope = dict(threading=threading, json=json, time=time, hashlib=hashlib, base64=base64,
                 secrets=secrets, mimetypes=mimetypes, os=__import__('os'), requests=mock.Mock())
    exec(compile((ROOT / 'server/token_estimation.py').read_text(), 'token_estimation.py', 'exec'), scope)
    scope.update(_record_server_diagnostic=mock.Mock(), _TEXT_LIKE_UPLOAD_EXTS={'.txt'},
                 _normalize_media_mime=lambda name, mime: mime)
    return scope


@pytest.mark.parametrize('provider,response,endpoint', [
    ('gemini', {'totalTokens': 12}, ':countTokens'),
    ('anthropic', {'input_tokens': 12}, '/v1/messages/count_tokens'),
    ('openai', {'input_tokens': 12}, '/v1/responses/input_tokens'),
    ('xai', {'token_ids': [1] * 12}, '/v1/tokenize-text'),
    ('kimi', {'data': {'total_tokens': 12}}, '/v1/tokenizers/estimate-token-count'),
])
def test_official_endpoints_parse_and_cache_counts(provider, response, endpoint):
    scope = namespace()
    reply = mock.Mock()
    reply.json.return_value = response
    scope['requests'].post.return_value = reply
    auth = {'provider': provider, 'api_key': 'FAKE_SECRET'}
    user = SimpleNamespace(id=7)
    call = scope['_draft_count_call']
    assert call(user, auth, 'test-model', '秘密の入力', []) == 12
    assert call(user, auth, 'test-model', '秘密の入力', []) == 12
    assert scope['requests'].post.call_count == 1
    args, kwargs = scope['requests'].post.call_args
    assert args[0].endswith(endpoint)
    assert kwargs['allow_redirects'] is False
    assert kwargs['timeout'] == (5, 15)
    call(SimpleNamespace(id=8), auth, 'test-model', '秘密の入力', [])
    assert scope['requests'].post.call_count == 2  # account isolation
    for key, value in scope['_DRAFT_COUNT_CACHE'].items():
        assert isinstance(key, bytes)
        assert value[1] == 12


@pytest.mark.parametrize('value', [None, -1, True, '12'])
def test_invalid_counts_cannot_be_reported_as_zero(value):
    scope = namespace()
    scope['requests'].post.return_value.json.return_value = {'input_tokens': value}
    with pytest.raises(ValueError):
        scope['_draft_count_call'](SimpleNamespace(id=1), {'provider': 'openai', 'api_key': 'fake'}, 'gpt-5', 'test', [])


def setup_estimate(scope):
    scope.update(get_model_api_provider=lambda _: 'openai',
                 _resolve_chat_model_auth=lambda *_: {'provider': 'openai', 'api_key': 'fake'},
                 _get_file_disk_info=lambda _: {'exists': True, 'size': 4},
                 _load_user_file_bytes=lambda *_: b'fake',
                 _draft_count_call=mock.Mock(side_effect=[8, 1108]))


def test_images_are_counted_with_text_and_breakdown_is_consistent():
    scope = namespace()
    setup_estimate(scope)
    result = scope['_provider_draft_estimate'](SimpleNamespace(id=1), 'gpt-5', 'test', ['1/photo.png'])
    assert result['tokens_prompt'] == 8
    assert result['tokens_files'] == 1100
    assert result['tokens_total'] == 1108
    assert result['files_counted'] == 1
    assert result['estimate_method'] == 'provider'
    f = scope['_draft_count_call'].call_args.args[-1][0]
    assert f['mime'] == 'image/png'
    assert f['bytes'] == b'fake'


def test_provider_failure_falls_back_without_logging_content():
    scope = namespace()
    setup_estimate(scope)
    scope['_draft_count_call'].side_effect = RuntimeError('SECRET-PROMPT SECRET-KEY')
    assert scope['_provider_draft_estimate'](SimpleNamespace(id=1), 'gpt-5', 'SECRET-PROMPT', []) is None
    logs = str(scope['_record_server_diagnostic'].call_args_list)
    assert 'SECRET' not in logs
    assert "phase='start'" in logs
    assert "phase='error'" in logs


@pytest.mark.parametrize('info', [{'exists': False}, {'exists': True, 'size': 20 * 1024 * 1024}])
def test_missing_or_large_files_do_not_get_silently_omitted(info):
    scope = namespace()
    setup_estimate(scope)
    scope['_get_file_disk_info'] = lambda _: info
    scope['_load_user_file_bytes'] = mock.Mock()
    assert scope['_provider_draft_estimate'](SimpleNamespace(id=1), 'gpt-5', 'test', ['1/photo.png']) is None
    scope['_load_user_file_bytes'].assert_not_called()
    scope['_draft_count_call'].assert_not_called()


def test_request_formats_include_native_media_without_persistent_upload():
    scope = namespace()
    f = {'name': 'document.pdf', 'mime': 'application/pdf', 'bytes': b'pdf'}
    for provider in ('openai', 'gemini', 'anthropic'):
        payload = scope['_draft_counter_request'](provider, 'model', 'test', [f])
        assert base64.b64encode(b'pdf').decode() in json.dumps(payload)
        assert 'file_id' not in json.dumps(payload)


def test_route_uses_provider_count_and_labels_fallback():
    tree = ast.parse((ROOT / 'server/routes_chat.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'estimate_prompt_tokens_api')
    fn.decorator_list = []
    scope = dict(request=SimpleNamespace(get_json=lambda **_: {'model': 'claude-test', 'message': 'hello', 'image_urls': []}),
                 current_user=SimpleNamespace(id=1), jsonify=lambda value: value,
                 _normalize_attachment_list=lambda refs, _: refs, should_count_tokens_for_display=lambda _: True,
                 count_tokens=lambda *_: 3, _provider_draft_estimate=lambda *_, **_kw: {'tokens_total': 9, 'estimate_method': 'provider'})
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'route', 'exec'), scope)
    assert scope['estimate_prompt_tokens_api']()['tokens_total'] == 9
    scope['_provider_draft_estimate'] = lambda *_, **_kw: None
    result = scope['estimate_prompt_tokens_api']()
    assert result['tokens_total'] == 3
    assert result['estimate_method'] == 'approximate'
    assert result['estimate_scope'] == 'draft'


def test_all_chat_providers_are_included_in_display_counts():
    source = (ROOT / 'server/token_utils.py').read_text()
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in {'LLM_TOKEN_MARKERS', 'PROMPT_TOKEN_MARKERS', 'NON_COUNTABLE_TOKEN_MARKERS'} for t in n.targets)]
    nodes += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'is_token_countable_model']
    scope = {'is_sts_model': lambda _: False}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'token_rules', 'exec'), scope)
    for model in ('claude-sonnet-4-6', 'deepseek-v4.1-flash', 'kimi-k2', 'mistral-large', 'glm-5', 'o3', 'gpt-5', 'gemini-3-pro'):
        assert scope['is_token_countable_model'](model), model
    for model in ('gpt-realtime', 'whisper-1', 'gpt-audio'):
        assert not scope['is_token_countable_model'](model)


def test_web_shows_unknown_attachments_and_counting_scope():
    import subprocess
    source = (ROOT / 'static/js/chat_core_parts/chat_core.part03_security_token_promptbar.js').read_text()
    renderer = source[source.index('        function renderPromptTokenEstimate('):source.index('        function schedulePromptTokenEstimate(')]
    script = '''const assert = require('node:assert/strict');
let text;
function setPromptTokenEstimateText(value) { text = value; }
''' + renderer + '''
renderPromptTokenEstimate({countable:true,tokens_total:123,tokens_prompt:23,tokens_files:100,
files_non_text:1,estimate_method:'approximate',estimate_scope:'draft'}, {message:'text'});
assert(text.includes('未計測1件'));
assert(text.includes('概算'));
assert(text.includes('履歴・SysPrompt除外'));
assert(!text.includes('0換算'));
renderPromptTokenEstimate({countable:true,tokens_total:123,tokens_prompt:23,tokens_files:100,
estimate_method:'provider',estimate_scope:'draft'}, {message:'text'});
assert(text.includes('プロバイダー計測'));
'''
    subprocess.run(['node', '-e', script], check=True, capture_output=True, text=True)


def test_web_reverting_to_cached_input_cancels_a_different_pending_estimate():
    import subprocess
    source = (ROOT / 'static/js/chat_core_parts/chat_core.part03_security_token_promptbar.js').read_text()
    scheduler = source[source.index('        function schedulePromptTokenEstimate('):source.index('        function updatePromptPlaceholder(')]
    script = '''const assert = require('node:assert/strict');
let timers = new Map(), nextTimer = 1;
const setTimeout = (fn) => { const id = nextTimer++; timers.set(id, fn); return id; };
const clearTimeout = id => timers.delete(id);
let promptTokenEstimateTimer = null, promptTokenEstimateAbort = null, promptTokenEstimateSeq = 0;
let promptTokenEstimateLastData = {tokens_total:1};
let payload = {model:'gpt-5',message:'cached',quote_text:'',image_urls:[]};
let promptTokenEstimateLastKey = JSON.stringify([payload.model,payload.message,'',[]]);
const PROMPT_TOKEN_ESTIMATE_DEBOUNCE_MS = 700;
const buildPromptTokenEstimatePayload = () => payload;
const renderPromptTokenEstimate = () => {};
''' + scheduler + '''
payload = {...payload,message:'other'};
schedulePromptTokenEstimate();
assert.equal(timers.size, 1);
payload = {...payload,message:'cached'};
schedulePromptTokenEstimate();
assert.equal(timers.size, 0);
'''
    subprocess.run(['node', '-e', script], check=True, capture_output=True, text=True)


def test_failed_tokenizer_does_not_return_zero_or_log_prompt_text():
    tree = ast.parse((ROOT / 'server/token_utils.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'count_tokens')
    scope = {'_get_token_encoder': lambda _: SimpleNamespace(encode=lambda *_a, **_k: []), 'log_force': mock.Mock()}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'count_tokens', 'exec'), scope)
    assert scope['count_tokens']('PRIVATE_PROMPT') > 0
    assert 'PRIVATE_PROMPT' not in str(scope['log_force'].call_args_list)


def test_openai_text_documents_use_the_same_native_file_input_as_generation():
    scope = namespace()
    setup_estimate(scope)
    result = scope['_provider_draft_estimate'](SimpleNamespace(id=1), 'gpt-5', 'test', ['1/document.txt'])
    assert result['estimate_method'] == 'provider'
    attachment = scope['_draft_count_call'].call_args.args[-1][0]
    assert attachment['bytes'] == b'fake'
    assert 'text' not in attachment
    body = scope['_draft_counter_request']('openai', 'gpt-5', 'test', [attachment])
    assert any(part['type'] == 'input_file' for part in body['input'][0]['content'])


def test_openai_quote_and_message_match_generation_content_blocks():
    scope = namespace()
    setup_estimate(scope)
    result = scope['_provider_draft_estimate'](SimpleNamespace(id=1), 'gpt-5', 'normalized', [],
                                               message_text='question', quote_text='quoted')
    assert result['estimate_method'] == 'provider'
    text = scope['_draft_count_call'].call_args.args[-2]
    body = scope['_draft_counter_request']('openai', 'gpt-5', text, [])
    assert body['input'][0]['content'] == [{'type': 'input_text', 'text': 'User Quote:\nquoted\n---'},
                                         {'type': 'input_text', 'text': 'question'}]


def test_media_only_input_does_not_add_a_synthetic_empty_text_part():
    scope = namespace()
    setup_estimate(scope)
    scope['_draft_count_call'].side_effect = [1120]
    result = scope['_provider_draft_estimate'](SimpleNamespace(id=1), 'gpt-5', '', ['1/photo.png'])
    assert result['tokens_prompt'] == 0
    assert result['tokens_files'] == result['tokens_total'] == 1120
    assert scope['_draft_count_call'].call_count == 1
    body = scope['_draft_counter_request']('gemini', 'gemini-3-pro', '',
        [{'name': 'photo.png', 'mime': 'image/png', 'bytes': b'fake'}])
    assert all('text' not in part for part in body['contents'][0]['parts'])
