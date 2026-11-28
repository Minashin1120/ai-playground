"""Official-rate boundaries and saved-model accounting, without calling provider APIs."""
import json
import subprocess
from pathlib import Path
from tests.app_source import read_chat_core_source

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRICING = json.loads((ROOT / 'static/model_pricing.json').read_text())
SOURCE = read_chat_core_source()
SOURCE = SOURCE[SOURCE.index("// Costs use each saved answer's model"):]
if '// 数学ステップ解説' in SOURCE:
    SOURCE = SOURCE[:SOURCE.index('// 数学ステップ解説')]


def estimate(messages, context=None):
    script = ('const fetch = () => new Promise(() => {});\n' + SOURCE
              + '\nconsole.log(JSON.stringify(estimateConversationCost('
              + json.dumps(messages) + ',' + json.dumps(PRICING) + ','
              + json.dumps(context if context is not None else messages) + ')));')
    result = subprocess.run(['node', '-e', script], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def answer(model, input_count, output=1000, **extras):
    return dict(id=2, role='assistant', model=model, content='test', tokens_in=input_count,
                tokens_out=output, **extras)


@pytest.mark.parametrize('model,boundary,short_in,short_out,long_in,long_out,inclusive', [
    ('gpt-6.1-sol', 272000, 2, 10, 4, 15, False),
    ('gemini-3.1-pro-preview', 200000, 2, 12, 4, 18, False),
    ('gemini-2.5-pro', 200000, 1.25, 10, 2.5, 15, False),
    ('grok-4.6', 200000, 2, 6, 4, 12, True),
    ('grok-4.3', 200000, 1.25, 2.5, 2.5, 5, True),
])
def test_threshold_is_per_request_and_applies_to_whole_request(model, boundary, short_in, short_out, long_in, long_out, inclusive):
    for n in [boundary - 1, boundary, boundary + 1]:
        long = n >= boundary if inclusive else n > boundary
        result = estimate([answer(model, n)])
        expected = (n * (long_in if long else short_in) + 1000 * (long_out if long else short_out)) / 1e6
        assert result['low'] == pytest.approx(expected)
        assert result['long'] == int(long)


def test_mixed_models_charge_each_answer_and_repeated_context_once_per_answer():
    rows = [dict(id=1, role='user', tokens_in=100),
            answer('gpt-4o-mini', None, 20, parent_id=1),
            dict(id=3, role='user', tokens_in=50, parent_id=2),
            dict(answer('gpt-6-astra', None, 10, parent_id=3), id=4)]
    cost = estimate(rows)
    assert cost['low'] == pytest.approx((100 * .15 + 20 * .6 + 170 * 10 + 10 * 50) / 1e6)
    assert cost['inferred'] == 2
    assert cost['unpriced'] == 0
    # No current selected model participates in the estimator or reprices saved answers.
    assert 'model-select' not in SOURCE
    assert 'currentModel' not in SOURCE


def test_siblings_do_not_enter_request_context_and_missing_ancestors_are_marked():
    rows = [dict(id=1, role='user', tokens_in=100),
            answer('gpt-4o-mini', None, 20, parent_id=1),
            dict(answer('gpt-6-astra', None, 30, parent_id=1), id=3)]
    path = estimate(rows[:2], rows)
    all_cost = estimate(rows)
    assert all_cost['low'] - path['low'] == pytest.approx((100 * 10 + 30 * 50) / 1e6)
    assert estimate([rows[1]])['unpriced'] == 1
    assert estimate(rows + [rows[1]]) == all_cost


def test_unknown_models_and_missing_usage_are_not_free():
    assert estimate([answer('unknown', 10)])['unpriced'] == 1
    assert estimate([answer('gpt-4o', 10, None)])['unpriced'] == 1
    free = estimate([answer('glm-4.7-flash', 10)])
    assert free['low'] == 0 and free['priced'] == 1 and free['unpriced'] == 0


def test_thought_is_already_in_output_and_not_added_again():
    result = estimate([answer('gpt-6.1-sol', 100, 1000, tokens_thought=800)])
    assert result['low'] == pytest.approx((100 * 2 + 1000 * 10) / 1e6)


def test_official_prompt_usage_overrides_display_counts_and_reconstructed_context():
    row = answer('gpt-6.1-sol', 100, thought_data=json.dumps({'pricing_input_tokens': 272001}))
    result = estimate([row])
    assert result['low'] == pytest.approx((272001 * 4 + 1000 * 15) / 1e6)
    assert result['inferred'] == 0 and result['long'] == 1


@pytest.mark.parametrize('usage,expected', [
    ({'promptTokenCount': 200001, 'secret': 'DO_NOT_KEEP'}, 200001),
    ({'prompt_token_count': 272001}, 272001),
    ({'prompt_tokens': 100}, 100),
    ({'input_tokens': 100, 'cache_read_input_tokens': 20, 'cache_creation_input_tokens': 30}, 150),
    ({'input_tokens': True}, None), ({'input_tokens': -1}, None),
    ({'input_tokens': '100'}, None), ({'input_tokens': 0}, 0), ({}, None),
])
def test_provider_usage_retains_only_valid_counts(usage, expected):
    import ast
    tree = ast.parse((ROOT / 'server/token_utils.py').read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_cost_input_tokens')
    scope = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'cost-usage', 'exec'), scope)
    assert scope['_cost_input_tokens'](usage) == expected


@pytest.mark.parametrize('encrypted', [False, True])
def test_batch_usage_is_saved_with_existing_encryption_without_changing_display_totals(encrypted):
    import ast
    from types import SimpleNamespace
    utils = ast.parse((ROOT / 'server/token_utils.py').read_text())
    route = ast.parse((ROOT / 'server/routes_batch.py').read_text())
    usage = next(n for n in utils.body if isinstance(n, ast.FunctionDef) and n.name == '_cost_input_tokens')
    payload = next(n for n in ast.walk(route) if isinstance(n, ast.FunctionDef) and n.name == '_set_message_payload')
    scope = dict(json=json, encrypt_val=lambda value: 'ENCRYPTED:' + value,
                 sum_token_counts=lambda a, b: (a or 0) + (b or 0))
    exec(compile((ROOT / 'server/model_usage.py').read_text(), 'model_usage.py', 'exec'), scope)
    exec(compile((ROOT / 'server/model_usage_pricing.py').read_text(), 'model_usage_pricing.py', 'exec'), scope)
    exec(compile(ast.Module(body=[usage, payload], type_ignores=[]), 'cost-batch', 'exec'), scope)
    message = SimpleNamespace(is_encrypted=encrypted, model='gpt-6.1-sol')
    scope['_set_message_payload'](message, 'answer', thought='reasoning', tokens_out=1000,
                                  tokens_thought=500, usage={'input_tokens': 272001, 'output_tokens': 1000,
                                      'output_tokens_details': {'reasoning_tokens': 500}, 'secret': 'DO_NOT_KEEP'}, billing_payload={'output': [{'id': 'PRIVATE_TOOL', 'type': 'web_search_call', 'status': 'completed', 'action': {'type': 'search', 'query': 'DO_NOT_KEEP'}}]})
    stored = message.thought_data.removeprefix('ENCRYPTED:')
    metadata = json.loads(stored)
    assert metadata['text'] == 'reasoning' and metadata['pricing_input_tokens'] == 272001
    assert metadata['pricing_calls'][0]['thought_tokens'] == 500
    assert metadata['pricing_calls'][0]['web_search_calls'] == 1
    assert 'PRIVATE_TOOL' not in stored
    assert metadata['pricing_calls'][0]['complete'] is True
    assert 'DO_NOT_KEEP' not in stored
    assert message.thought_data.startswith('ENCRYPTED:') == encrypted
    assert message.tokens == message.tokens_out == 1000


def test_fast_pricing_load_and_model_switch_preserve_the_bar_amount():
    script = '''
const vm = require('vm');
const pricing = PRICING;
const elements = Object.fromEntries(['total-cost-estimate'].map(id =>
    [id, {classList:{toggle(){}},textContent:'',title:''}]));
const context = vm.createContext({fetch:async()=>({ok:true,json:async()=>pricing}),get:id=>elements[id]});
vm.runInContext(SOURCE, context);
setImmediate(()=>{
  vm.runInContext(`const messageMeta={__total__:{}};
    const rows=[{id:2,role:'assistant',content:'x',model:'gpt-6.1-sol',tokens_in:1000,tokens_out:1000}];
    refreshCostBar(rows);`, context);
  const before=elements['total-cost-estimate'].textContent;
  vm.runInContext(`let currentModel='gpt-6-astra';refreshCostBar();`,context);
  console.log(JSON.stringify({before,after:elements['total-cost-estimate'].textContent}));
});
'''.replace('PRICING', json.dumps(PRICING)).replace('SOURCE', json.dumps(SOURCE))
    result = subprocess.run(['node', '-e', script], check=True, capture_output=True, text=True)
    labels = json.loads(result.stdout)
    assert labels['before'] == labels['after'] == '費用予想: $0.012000'


def test_deepseek_and_audio_input_have_explicit_ranges():
    result = estimate([answer('deepseek-v4.1-flash', 1000000, 1000000)])
    assert result['low'] == pytest.approx(.75)
    assert result['high'] == pytest.approx(1.5)
    result = estimate([answer('gemini-2.5-flash', 1000000, 0)])
    assert result['low'] == pytest.approx(.3)
    assert result['high'] == pytest.approx(1)


def test_batch_discount_is_provider_specific():
    for model, multiplier in [('gpt-6.1-sol', .5), ('grok-4.6', .8), ('claude-sonnet-4-6', .5), ('kimi-k3', .6)]:
        normal = estimate([answer(model, 300000)])
        batch = estimate([answer(model, 300000, batch_job={'state': 'JOB_STATE_SUCCEEDED'})])
        assert batch['low'] == pytest.approx(normal['low'] * multiplier)


def test_batch_cache_read_pricing_follows_each_provider():
    def batch_cost(model, input_tokens, cache_read_tokens, output_tokens):
        call = {'model': model, 'input_tokens': input_tokens, 'cache_read_tokens': cache_read_tokens,
                'output_tokens': output_tokens, 'complete': True}
        return estimate([answer(model, input_tokens, output_tokens,
                                thought_data=json.dumps({'pricing_calls': [call]}),
                                batch_job={'state': 'JOB_STATE_SUCCEEDED'})])
    # Gemini 3.1 Pro keeps standard cache-read pricing in Batch; ordinary input and output are halved.
    gemini = batch_cost('gemini-3.1-pro-preview', 100000, 50000, 1000)
    assert gemini['low'] == pytest.approx((50000 * 1 + 50000 * .2 + 1000 * 6) / 1e6)
    # OpenAI applies the Batch discount to cache reads as well.
    openai = batch_cost('gpt-6.1-sol', 100000, 50000, 1000)
    assert openai['low'] == pytest.approx((50000 * 1 + 50000 * .05 + 1000 * 5) / 1e6)


def test_dates_are_from_saved_answer_and_not_wall_clock():
    before = estimate([answer('gemini-4-argon', 1000000, 0, created_at='2026-11-30T23:59:59Z')])
    after = estimate([answer('gemini-4-argon', 1000000, 0, created_at='2026-12-01T00:00:00Z')])
    assert before['low'] == 2 and after['low'] == 4
    assert estimate([answer('gemini-4-argon', 1000000, 0)])['low'] == 2
    assert estimate([answer('gemini-4-carbon', 1000000, 0)])['low'] == 2
    assert estimate([answer('gemini-4-carbon', 1000000, 0, created_at='2026-12-01T00:00:00Z')])['low'] == 4
    assert estimate([answer('gemini-4-hydrogen', 1000000, 0)])['low'] == pytest.approx(1.5)
    assert estimate([answer('Gemini 2.5 Pro', 300000, 0)])['low'] == pytest.approx(.75)


def test_android_and_web_use_identical_official_rate_data():
    assert (ROOT / 'static/model_pricing.json').read_bytes() == (ROOT / 'android/app/src/main/assets/model-pricing.json').read_bytes()
    assert PRICING['checked_at'] == '2026-11-28'
    assert all(model['provider'] in PRICING['sources'] for model in PRICING['models'].values())


def test_every_catalog_model_is_registered_and_estimates_without_unpriced():
    part09 = (ROOT / 'static/js/chat_core_parts/chat_core.part09_models.js').read_text(encoding='utf-8')
    end_idx = part09.index('\n        ];\n')
    js = part09[:end_idx + 10] + '\nconsole.log(JSON.stringify(MODELS.flatMap(c => (c.items || []).map(m => m.id))));'
    ids = json.loads(subprocess.run(['node', '-e', js], check=True, capture_output=True, text=True).stdout)
    assert len(ids) >= 159
    for mid in ids:
        key = PRICING['aliases'].get(mid, mid)
        assert key in PRICING['models'], f'{mid} missing from model_pricing.json'
        rate = PRICING['models'][key]
        unit = rate.get('unit', 'token')
        if unit == 'image':
            call = {'model': mid, 'images': 1, 'operation': 'generate', 'quality': 'medium', 'resolution': '1k', 'size': '1024x1024', 'complete': True}
        elif unit == 'video':
            call = {'model': mid, 'duration_ms': 5000, 'resolution': '720p', 'complete': True}
        elif unit == 'song':
            call = {'model': mid, 'songs': 1, 'complete': True}
        elif unit == 'page':
            call = {'model': mid, 'pages': 3, 'complete': True}
        elif unit == 'character':
            call = {'model': mid, 'characters': 500, 'complete': True}
        elif unit == 'duration':
            call = {'model': mid, 'duration_ms': 30000, 'complete': True}
        else:
            call = {'model': mid, 'input_tokens': 1000, 'output_tokens': 500, 'images': 1, 'complete': True}
        cost = estimate([{'id': 2, 'role': 'assistant', 'model': mid, 'content': 'ok', 'thought_data': json.dumps({'pricing_calls': [call]})}])
        assert cost['priced'] == 1, f'{mid} was unpriced'


def test_new_modalities_and_unlisted_models_price_accurately():
    def call_est(call, date='2026-11-28'):
        return estimate([{'id': 2, 'role': 'assistant', 'model': call['model'], 'content': 'ok',
                         'created_at': f'{date}T00:00:00Z',
                         'thought_data': json.dumps({'pricing_calls': [call]})}])

    # Gemini image token breakdown vs per-image
    nb21 = call_est({'model': 'gemini-nano-banana-2.1', 'input_tokens': 1000, 'output_tokens': 1220, 'image_output_tokens': 1120, 'images': 1, 'complete': True})
    assert nb21['low'] == pytest.approx((1000 * 1.5 + 100 * 7.5 + 1120 * 30.0) / 1e6)
    assert nb21['unpriced'] == 0

    g25 = call_est({'model': 'gemini-2.5-flash-image', 'input_tokens': 1000, 'output_tokens': 1290, 'image_output_tokens': 1290, 'images': 1, 'complete': True})
    assert g25['low'] == pytest.approx(1000 * 0.3 / 1e6 + 0.039)
    assert g25['unpriced'] == 0

    # OpenAI GPT Image token-based vs per-image fallback
    gimg26 = call_est({'model': 'gpt-image-2.6-sunburst', 'input_tokens': 100, 'output_tokens': 1000,
                       'text_input_tokens': 40, 'image_input_tokens': 60, 'image_output_tokens': 1000, 'images': 1, 'complete': True})
    assert gimg26['low'] == pytest.approx((40 * 5.0 + 60 * 8.0 + 1000 * 30.0) / 1e6)
    gimg15 = call_est({'model': 'gpt-image-1.5', 'images': 1, 'quality': 'medium', 'size': '1024x1024', 'complete': True})
    assert gimg15['low'] == pytest.approx(0.034) and gimg15['high'] == pytest.approx(0.034)

    # Veo 3.1, Grok Imagine Image/Video, Lyria, Mistral OCR, TTS, STT, Live
    assert call_est({'model': 'veo-3.1-fast-generate-preview', 'duration_ms': 8000, 'resolution': '1080p', 'complete': True})['low'] == pytest.approx(0.96)
    assert call_est({'model': 'grok-imagine-image-2.0', 'images': 1, 'resolution': '2k', 'quality': 'medium', 'input_images': 1, 'complete': True})['low'] == pytest.approx(0.09)
    assert call_est({'model': 'grok-imagine-video-1.5', 'duration_ms': 5000, 'resolution': '1080p', 'input_images': 1, 'complete': True})['low'] == pytest.approx(1.26)
    assert call_est({'model': 'lyria-3.5', 'songs': 1, 'complete': True})['low'] == pytest.approx(0.08)
    assert call_est({'model': 'mistral-ocr-4-0', 'pages': 10, 'complete': True})['low'] == pytest.approx(0.04)
    assert call_est({'model': 'google-tts-studio', 'characters': 1000, 'complete': True})['low'] == pytest.approx(0.16)
    assert call_est({'model': 'grok-tts', 'characters': 2000, 'complete': True})['low'] == pytest.approx(0.03)
    assert call_est({'model': 'gpt-transcribe', 'duration_ms': 120000, 'complete': True})['low'] == pytest.approx(0.009)
    assert call_est({'model': 'grok-voice-3.0', 'duration_ms': 60000, 'text_turns': 2, 'complete': True})['low'] == pytest.approx(0.088)
    assert call_est({'model': 'gpt-5-search-api', 'input_tokens': 1000, 'output_tokens': 100, 'web_search_calls': 1, 'complete': True})['low'] == pytest.approx((1000 * 1.25 + 100 * 10.0) / 1e6 + 0.01)

    # Unlisted OpenAI models show a documented range instead of unpriced or $0
    m55 = call_est({'model': 'gpt-5.5-mini', 'input_tokens': 1000, 'output_tokens': 500, 'complete': True})
    assert m55['priced'] == 1 and m55['unpriced'] == 1 and m55['high'] > m55['low'] > 0

    # Grok 4-1 fast switches to 4.3 rates on 2026-05-15 (and uses 200k threshold)
    assert call_est({'model': 'grok-4-1-fast-reasoning', 'input_tokens': 100000, 'output_tokens': 0, 'complete': True}, date='2026-05-14')['low'] == pytest.approx(0.02)
    assert call_est({'model': 'grok-4-1-fast-reasoning', 'input_tokens': 100000, 'output_tokens': 0, 'complete': True}, date='2026-05-15')['low'] == pytest.approx(0.125)
    assert call_est({'model': 'grok-4-1-fast-reasoning', 'input_tokens': 200000, 'output_tokens': 0, 'complete': True}, date='2026-05-15')['low'] == pytest.approx(0.50)


def test_jpy_label_converts_usd_ranges_with_the_reference_rate():
    script = '''
const vm = require('vm');
const context = vm.createContext({fetch: async () => ({ok: true, json: async () => ({})}), get: () => null});
vm.runInContext(SOURCE, context);
console.log(vm.runInContext(`JSON.stringify([
  costEstimateJpyLabel({priced:1,low:0.75,high:0.75,unpriced:0}, {rate:159.6,date:'2026-11-16'}),
  costEstimateJpyLabel({priced:1,low:0.01,high:0.02,unpriced:0}, {rate:159.6,date:'2026-11-16'}),
  costEstimateJpyLabel({priced:1,low:0.01,high:0.01,unpriced:1,lowerBound:true}, {rate:159.6,date:''}),
  costEstimateJpyLabel({priced:1,low:0.0000001,high:0.0000001,unpriced:0}, {rate:159.6,date:''}),
  costEstimateJpyLabel({priced:0}, {rate:159.6,date:''})
])`, context));
'''.replace('SOURCE', json.dumps(SOURCE))
    result = subprocess.run(['node', '-e', script], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == [
        '日本円換算: 約119.70円（参考レート 1ドル=159.60円・2026-11-16）',
        '日本円換算: 約1.60円〜3.19円（参考レート 1ドル=159.60円・2026-11-16）',
        '日本円換算: 約1.60円以上（一部未算入）（参考レート 1ドル=159.60円）',
        '日本円換算: 約0.01円未満（参考レート 1ドル=159.60円）',
        '日本円換算: 算出不可',
    ]


def test_jpy_conversion_fetches_only_the_free_rate_and_caches_it():
    script = '''
const vm = require('vm');
const requests = [];
const stored = [];
const context = vm.createContext({
  get: () => null,
  AbortController, setTimeout, clearTimeout,
  localStorage: {getItem: () => null, setItem: (key, value) => stored.push([key, value])},
  fetch: async (url) => {
    if (url.startsWith('http')) requests.push(url);
    return {ok: true, json: async () => ([{date: '2026-11-17', base: 'USD', quote: 'JPY', rate: 150}])};
  },
});
vm.runInContext(SOURCE, context);
(async () => {
  const label = await vm.runInContext(`(async () => {
    const el = {textContent: '', title: ''};
    fillCostEstimateJpy(el, {priced: 1, low: 0.01, high: 0.01, unpriced: 0});
    await new Promise(done => setTimeout(done, 20));
    return el.textContent;
  })()`, context);
  console.log(JSON.stringify({label, requests, stored}));
})();
'''.replace('SOURCE', json.dumps(SOURCE))
    result = subprocess.run(['node', '-e', script], check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    assert data['label'] == '日本円換算: 約1.50円（参考レート 1ドル=150.00円・2026-11-17）'
    assert data['requests'] == ['https://api.frankfurter.dev/v2/rates?base=USD&quotes=JPY']
    assert data['stored'][0][0] == 'cost_jpy_rate_v1'
    assert '"rate":150' in data['stored'][0][1]
