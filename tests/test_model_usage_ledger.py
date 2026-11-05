"""Synthetic HTTP/SDK calls only; verifies per-request accounting without reading account data."""
import json
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]


def usage_scope(handler=None):
    scope = {}
    exec(compile((ROOT / 'server/model_usage.py').read_text(), 'model_usage.py', 'exec'), scope)
    exec(compile((ROOT / 'server/model_usage_pricing.py').read_text(), 'model_usage_pricing.py', 'exec'), scope)
    handler = handler or (lambda request: httpx.Response(200, json={}))
    clients = [httpx.Client(transport=httpx.MockTransport(handler)) for _ in range(2)]
    scope.update(_OPENAI_HTTPX_CLIENT=clients[0], _GEMINI_HTTPX_CLIENT=clients[1])
    exec(compile((ROOT / 'server/model_usage_http.py').read_text(), 'model_usage_http.py', 'exec'), scope)
    return scope


class Chunks(httpx.SyncByteStream):
    def __init__(self, content, error=False):
        self.content = content
        self.error = error

    def __iter__(self):
        for index in range(0, len(self.content), 11):
            yield self.content[index:index + 11]
        if self.error:
            raise httpx.ReadError('synthetic interruption')


def response_event(identity, inputs, outputs, thought=0):
    return {'type': 'response.completed', 'response': {'id': identity, 'model': 'gpt-6.1-sol',
            'usage': {'input_tokens': inputs, 'output_tokens': outputs, 'output_tokens_details': {'reasoning_tokens': thought}}}}


def test_stream_events_are_cumulative_but_separate_requests_are_added():
    counter = 0
    def handle(request):
        nonlocal counter
        counter += 1
        event = response_event(f'response_{counter}', 10000 + counter * 10, 100, 80)
        body = ('data: ' + json.dumps(event) + '\n\ndata: ' + json.dumps(event) + '\n\ndata: [DONE]\n\n').encode()
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=Chunks(body))
    scope = usage_scope(handle)
    ledger = scope['ModelUsageLedger']()
    token = scope['_MODEL_USAGE_CONTEXT'].set(ledger)
    try:
        for _ in range(2):
            with scope['_OPENAI_HTTPX_CLIENT'].stream('POST', 'https://api.openai.com/v1/responses', json={'model': 'gpt-6.1-sol', 'input': 'PRIVATE_PROMPT'}) as response:
                assert b'[DONE]' in response.read()
    finally:
        scope['_MODEL_USAGE_CONTEXT'].reset(token)
    assert ledger.token_totals() == {'input_tokens': 20030, 'output_tokens': 200, 'thought_tokens': 160}
    assert len(ledger.snapshot()) == 2
    assert all(row['complete'] for row in ledger.snapshot())
    serialized = json.dumps(ledger.snapshot())
    assert 'PRIVATE_PROMPT' not in serialized and 'response_1' not in serialized


def test_openai_sdk_retry_keeps_usage_from_the_first_attempt():
    from openai import OpenAI
    count = 0
    def handle(request):
        nonlocal count
        count += 1
        if count == 1:
            return httpx.Response(500, json={'error': {'message': 'synthetic failure'}, 'usage': {'input_tokens': 50, 'output_tokens': 5}})
        return httpx.Response(200, json={'id': 'reply', 'object': 'response', 'created_at': 1,
            'status': 'completed', 'model': 'gpt-6.1-sol', 'output': [], 'usage': {'input_tokens': 100, 'output_tokens': 10}})
    scope = usage_scope(handle)
    ledger = scope['ModelUsageLedger']()
    token = scope['_MODEL_USAGE_CONTEXT'].set(ledger)
    try:
        client = OpenAI(api_key='SYNTHETIC_KEY', http_client=scope['_OPENAI_HTTPX_CLIENT'], max_retries=1)
        client.responses.create(model='gpt-6.1-sol', input='synthetic')
    finally:
        scope['_MODEL_USAGE_CONTEXT'].reset(token)
    assert count == 2
    assert ledger.token_totals()['input_tokens'] == 150
    assert ledger.token_totals()['output_tokens'] == 15
    assert not ledger.snapshot()[0]['complete']


def test_gemini_sdk_automatic_tool_rounds_are_recorded_separately():
    from google import genai
    from google.genai import types
    count = 0
    def handle(request):
        nonlocal count
        count += 1
        part = {'functionCall': {'name': 'lookup', 'args': {}}} if count == 1 else {'text': 'done'}
        return httpx.Response(200, json={'candidates': [{'content': {'role': 'model', 'parts': [part]}, 'finishReason': 'STOP'}],
            'usageMetadata': {'promptTokenCount': 10000 + count * 10, 'candidatesTokenCount': 20, 'thoughtsTokenCount': 5}})
    def lookup() -> str:
        """Return synthetic data."""
        return 'synthetic result'
    scope = usage_scope(handle)
    ledger = scope['ModelUsageLedger']()
    token = scope['_MODEL_USAGE_CONTEXT'].set(ledger)
    try:
        client = genai.Client(api_key='SYNTHETIC_KEY', http_options=types.HttpOptions(httpx_client=scope['_GEMINI_HTTPX_CLIENT']))
        response = client.models.generate_content(model='gemini-3.6-flash', contents='synthetic', config=types.GenerateContentConfig(tools=[lookup]))
        assert response.text == 'done'
    finally:
        scope['_MODEL_USAGE_CONTEXT'].reset(token)
    assert count == 2 and len(ledger.snapshot()) == 2
    assert ledger.token_totals() == {'input_tokens': 20030, 'output_tokens': 50, 'thought_tokens': 10}


def test_claude_usage_events_merge_without_adding_cumulative_output_twice():
    scope = usage_scope()
    call = scope['ModelUsageCall']('claude-sonnet-4-6', 'anthropic')
    call.observe({'message': {'id': 'synthetic', 'usage': {'input_tokens': 100, 'cache_read_input_tokens': 20, 'cache_creation_input_tokens': 30, 'output_tokens': 3}}})
    call.observe({'usage': {'output_tokens': 1000}})
    call.finished = True
    assert call.snapshot()['input_tokens'] == 150
    assert call.snapshot()['output_tokens'] == 1000


def test_retrieval_does_not_add_another_generation_and_context_is_isolated():
    payload = response_event('same_response', 100, 20)['response']
    scope = usage_scope(lambda request: httpx.Response(200, json=payload))
    first, second = scope['ModelUsageLedger'](), scope['ModelUsageLedger']()
    token = scope['_MODEL_USAGE_CONTEXT'].set(first)
    scope['_OPENAI_HTTPX_CLIENT'].post('https://api.openai.com/v1/responses', json={'model': 'gpt-6.1-sol'})
    scope['_OPENAI_HTTPX_CLIENT'].get('https://api.openai.com/v1/responses/same_response')
    scope['_MODEL_USAGE_CONTEXT'].reset(token)
    token = scope['_MODEL_USAGE_CONTEXT'].set(second)
    scope['_OPENAI_HTTPX_CLIENT'].post('https://api.openai.com/v1/responses', json={'model': 'gpt-6.1-sol'})
    scope['_MODEL_USAGE_CONTEXT'].reset(token)
    assert len(first.snapshot()) == len(second.snapshot()) == 1


def test_explicit_rejection_and_missing_usage_are_distinct():
    scope = usage_scope(lambda request: httpx.Response(429 if request.url.host == 'rejected.test' else 200, json={}))
    ledger = scope['ModelUsageLedger']()
    token = scope['_MODEL_USAGE_CONTEXT'].set(ledger)
    for host in ['rejected.test', 'unknown.test']:
        scope['_OPENAI_HTTPX_CLIENT'].post(f'https://{host}/v1/responses', json={'model': 'gpt-6.1-sol'})
    scope['_MODEL_USAGE_CONTEXT'].reset(token)
    assert len(ledger.snapshot()) == 1 and not ledger.snapshot()[0]['complete']


def test_model_extraction_skips_large_nested_prompt_fields():
    scope = usage_scope()
    request = httpx.Request('POST', 'https://api.openai.com/v1/responses', json={'input': [{'model': 'not_the_model', 'text': 'x' * 1000000}], 'model': 'gpt-6.1-sol'})
    assert scope['_usage_model_from_request'](request) == 'gpt-6.1-sol'


def test_sync_metadata_is_numeric_and_does_not_accept_arbitrary_content():
    scope = usage_scope()
    rows = scope['_sanitize_pricing_calls']([{'model': 'gpt-6.1-sol', 'input_tokens': 100, 'output_tokens': 20,
        'thought_tokens': 10, 'complete': True, 'prompt': 'PRIVATE_PROMPT', 'api_key': 'PRIVATE_KEY', 'created_at': '2026-11-06T00:00:00Z'}])
    serialized = json.dumps(rows)
    assert 'PRIVATE' not in serialized and 'api_key' not in serialized and 'prompt"' not in serialized
    assert rows[0]['thought_tokens'] == 10


def test_interrupted_stream_keeps_reported_usage_as_incomplete():
    event = response_event('partial', 100, 30, 10)
    body = ('data: ' + json.dumps(event) + '\n\n').encode()
    scope = usage_scope(lambda request: httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=Chunks(body, error=True)))
    ledger = scope['ModelUsageLedger']()
    token = scope['_MODEL_USAGE_CONTEXT'].set(ledger)
    try:
        with pytest.raises(httpx.ReadError):
            scope['_OPENAI_HTTPX_CLIENT'].post('https://api.openai.com/v1/responses', json={'model': 'gpt-6.1-sol'})
    finally:
        scope['_MODEL_USAGE_CONTEXT'].reset(token)
    assert ledger.token_totals()['thought_tokens'] == 10
    assert not ledger.snapshot()[0]['complete']
