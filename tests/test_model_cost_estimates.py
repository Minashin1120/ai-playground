"""Official-rate boundaries and saved-model accounting, without calling provider APIs."""
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRICING = json.loads((ROOT / 'static/model_pricing.json').read_text())
SOURCE = (ROOT / 'static/js/chat_core_parts/chat_core.part20_cost_estimate.js').read_text()


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
    route = ast.parse((ROOT / 'server/routes_chat.py').read_text())
    usage = next(n for n in utils.body if isinstance(n, ast.FunctionDef) and n.name == '_cost_input_tokens')
    payload = next(n for n in ast.walk(route) if isinstance(n, ast.FunctionDef) and n.name == '_set_message_payload')
    scope = dict(json=json, encrypt_val=lambda value: 'ENCRYPTED:' + value,
                 sum_token_counts=lambda a, b: (a or 0) + (b or 0))
    exec(compile(ast.Module(body=[usage, payload], type_ignores=[]), 'cost-batch', 'exec'), scope)
    message = SimpleNamespace(is_encrypted=encrypted)
    scope['_set_message_payload'](message, 'answer', thought='reasoning', tokens_out=1000,
                                  tokens_thought=500, usage={'input_tokens': 272001, 'secret': 'DO_NOT_KEEP'})
    stored = message.thought_data.removeprefix('ENCRYPTED:')
    assert json.loads(stored) == {'text': 'reasoning', 'pricing_input_tokens': 272001}
    assert message.thought_data.startswith('ENCRYPTED:') == encrypted
    assert message.tokens == message.tokens_out == 1000


def test_fast_pricing_load_and_model_switch_preserve_the_bar_amount():
    script = '''
const vm = require('vm');
const pricing = PRICING;
const elements = Object.fromEntries(['total-cost-estimate','total-cost-estimate-all'].map(id =>
    [id, {classList:{toggle(){}},textContent:'',title:''}]));
const context = vm.createContext({fetch:async()=>({ok:true,json:async()=>pricing}),get:id=>elements[id]});
vm.runInContext(SOURCE, context);
setImmediate(()=>{
  vm.runInContext(`const messageMeta={__total__:{},__total_all_branches__:{}};
    const rows=[{id:2,role:'assistant',content:'x',model:'gpt-6.1-sol',tokens_in:1000,tokens_out:1000}];
    refreshCostBar(rows,rows);`, context);
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
    for model, multiplier in [('gpt-6.1-sol', .5), ('grok-4.6', .8), ('kimi-k3', .6)]:
        normal = estimate([answer(model, 300000)])
        batch = estimate([answer(model, 300000, batch_job={'state': 'JOB_STATE_SUCCEEDED'})])
        assert batch['low'] == pytest.approx(normal['low'] * multiplier)


def test_dates_are_from_saved_answer_and_not_wall_clock():
    before = estimate([answer('gemini-4-argon', 1000000, 0, created_at='2026-11-30T23:59:59Z')])
    after = estimate([answer('gemini-4-argon', 1000000, 0, created_at='2026-12-01T00:00:00Z')])
    assert before['low'] == 2 and after['low'] == 4
    assert estimate([answer('gemini-4-argon', 1000000, 0)])['low'] == 2
    assert estimate([answer('Gemini 2.5 Pro', 300000, 0)])['low'] == pytest.approx(.75)


def test_android_and_web_use_identical_official_rate_data():
    assert (ROOT / 'static/model_pricing.json').read_bytes() == (ROOT / 'android/app/src/main/assets/model-pricing.json').read_bytes()
    assert PRICING['checked_at'] == '2026-11-05'
    assert all(model['provider'] in PRICING['sources'] for model in PRICING['models'].values())
