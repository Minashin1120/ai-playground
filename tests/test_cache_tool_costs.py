"""Official cache partitions and hosted tools, using synthetic responses only."""
import json
import subprocess
import pytest
from tests.test_model_cost_estimates import SOURCE, answer, estimate
from tests.test_model_usage_ledger import usage_scope


def cost(model, **usage):
    row = dict(dict(model=model, input_tokens=100, output_tokens=1000, complete=True), **usage)
    return estimate([answer(model, None, None, thought_data=json.dumps({'pricing_calls': [row]}))])


def test_openai_cache_reads_and_two_write_lifetimes_replace_normal_input():
    result = cost('gpt-6.1-sol', input_tokens=12000, cache_read_tokens=4000, cache_write_tokens=7000,
                  cache_write_30m_tokens=3000, cache_write_12h_tokens=4000)
    assert result['low'] == pytest.approx((1000*2 + 4000*.1 + 3000*2*1.25 + 4000*2*2.5 + 1000*10)/1e6)
    assert result['unpriced'] == 0


def test_anthropic_cache_writes_are_not_counted_as_regular_input_too():
    result = cost('claude-sonnet-4-6', input_tokens=2000, cache_read_tokens=900, cache_write_tokens=1000,
                  cache_write_5m_tokens=600, cache_write_1h_tokens=400)
    assert result['low'] == pytest.approx((100*3 + 900*.3 + 600*3*1.25 + 400*3*2 + 1000*15)/1e6)


def test_long_context_is_decided_using_all_input_including_cache():
    result = cost('gpt-6.1-sol', input_tokens=300000, cache_read_tokens=290000)
    assert result['low'] == pytest.approx((10000*4 + 290000*.2 + 1000*15)/1e6)
    assert result['long'] == 1


@pytest.mark.parametrize('model,read_rate', [('gemini-3.1-pro-preview', .2), ('glm-5.3', .26), ('kimi-k3', .3)])
def test_other_provider_cached_input(model, read_rate):
    result = cost(model, cache_read_tokens=80)
    assert result['low'] > 0 and result['unpriced'] == 0


def test_token_detail_summary_splits_cache_read_and_ttl_writes():
    thought = json.dumps({'pricing_calls': [
        {'model': 'gpt-6.1-sol', 'cache_read_tokens': 4000, 'cache_write_tokens': 7000,
         'cache_write_30m_tokens': 3000, 'cache_write_12h_tokens': 4000},
        {'model': 'gpt-6.1-sol', 'cache_read_tokens': 1000, 'cache_write_tokens': 500},
    ]})
    script = ('const fetch = () => new Promise(() => {});\n' + SOURCE
              + '\nconsole.log(JSON.stringify(costCacheUsage(' + json.dumps(thought) + ')));')
    result = subprocess.run(['node', '-e', script], check=True, capture_output=True, text=True)
    summary = json.loads(result.stdout)
    assert summary == {'read': 5000, 'write': 7500, 'ttls': {'30m': 3000, '12h': 4000}}


def test_token_detail_summary_is_hidden_without_cache_fields():
    script = ('const fetch = () => new Promise(() => {});\n' + SOURCE
              + '\nconsole.log(JSON.stringify(costCacheUsage(' + json.dumps(json.dumps({'pricing_calls': [
                  {'model': 'gpt-6.1-sol', 'input_tokens': 100, 'output_tokens': 10, 'complete': True}]})) + ')));')
    result = subprocess.run(['node', '-e', script], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) is None


def test_unknown_write_lifetime_shows_a_range_and_partial_status():
    result = cost('gpt-6.1-sol', cache_write_tokens=100)
    assert result['high'] > result['low'] and result['unpriced'] == 1


def test_web_search_fee_is_separate_from_tokens_and_batch_discount():
    row = answer('gpt-6.1-sol', None, None, batch_job={'state':'JOB_STATE_SUCCEEDED'},
        thought_data=json.dumps({'pricing_calls':[{'model':'gpt-6.1-sol','input_tokens':100,'output_tokens':1000,'web_search_calls':2,'complete':True}]}))
    result = estimate([row])
    assert result['low'] == pytest.approx(.0102*.5 + .02)


def test_preview_search_and_fixed_mini_search_content_prices():
    assert cost('gpt-4o', web_search_calls=1, search_preview=True)['low'] == pytest.approx(.01025+.025)
    assert cost('gpt-4o-mini', web_search_calls=2)['low'] == pytest.approx(.000615+.02+16000*.15/1e6)


def test_gemini_grounding_counts_queries_for_new_models_and_prompts_for_25():
    a = cost('gemini-3.1-pro-preview', grounding_queries=3)
    b = cost('gemini-2.5-pro', grounding_queries=3)
    assert a['low'] == pytest.approx(.0122+.042)
    assert b['low'] == pytest.approx(.010125+.035)


def test_xai_reported_charge_includes_tools_and_cache_without_double_counting():
    result = cost('grok-4.7', cost_in_usd_ticks=400000000, web_search_calls=99, cache_read_tokens=100)
    assert result['low'] == pytest.approx(.04) and result['high'] == pytest.approx(.04)


def test_execution_runtime_and_explicit_cache_storage_are_marked_incomplete():
    assert cost('gpt-6.1-sol', code_execution_calls=2, container_sessions=1)['unpriced'] == 1
    assert cost('gemini-3.1-pro-preview', explicit_cache=True, cache_read_tokens=50)['unpriced'] == 1
    assert cost('claude-sonnet-4-6', code_execution_calls=1)['lowerBound']
    assert cost('claude-sonnet-4-6', code_execution_calls=1, code_execution_free=True)['unpriced'] == 0


def test_observer_keeps_cache_partitions_but_no_search_text_or_tool_ids():
    scope = usage_scope(); ledger = scope['ModelUsageLedger'](); call = ledger.begin('gpt-6.1-sol', 'openai')
    event = {'item': {'id':'SECRET_TOOL_ID','type':'web_search_call','status':'completed','action':{'type':'search','query':'SECRET_QUERY'}}}
    call.observe(event); call.observe(event)
    call.observe({'item': {'id':'OPEN_PAGE','type':'web_search_call','status':'completed','action':{'type':'open_page'}}})
    call.observe({'usage':{'input_tokens':12000,'output_tokens':1000,'input_tokens_details':{'cached_tokens':4000,'cache_write_tokens':7000,'cache_write_tokens_details':{'ttl_30m_tokens':3000,'ttl_12h_tokens':4000}}}})
    row = call.snapshot()
    assert row['web_search_calls'] == 1 and row['cache_write_12h_tokens'] == 4000
    assert 'SECRET' not in json.dumps(row) and 'OPEN_PAGE' not in json.dumps(row)


def test_google_and_claude_normalization_and_mobile_sanitizing():
    scope = usage_scope()
    google = scope['ModelUsageCall']('gemini-3.1-pro-preview','google')
    evt = {'candidates':[{'groundingMetadata':{'webSearchQueries':['SECRET_QUERY','SECRET_QUERY','OTHER_QUERY']}}], 'usageMetadata':{'promptTokenCount':1000,'cachedContentTokenCount':900,'candidatesTokenCount':100}}
    google.observe(evt); google.observe(evt)
    assert google.snapshot()['grounding_queries'] == 2 and google.snapshot()['cache_read_tokens'] == 900
    claude = scope['ModelUsageCall']('claude-sonnet-4-6','anthropic')
    claude.observe({'usage':{'input_tokens':100,'cache_read_input_tokens':900,'cache_creation_input_tokens':1000,'cache_creation':{'ephemeral_5m_input_tokens':600,'ephemeral_1h_input_tokens':400},'output_tokens':1000,'server_tool_use':{'web_search_requests':2}}})
    clean = scope['_sanitize_pricing_calls']([claude.snapshot(), dict(google.snapshot(), prompt='SECRET_PROMPT')])
    assert clean[0]['input_tokens'] == 2000 and clean[0]['web_search_calls'] == 2
    assert 'SECRET' not in json.dumps(clean) and clean[1]['grounding_queries'] == 2


def test_xai_cost_ticks_preserve_values_above_32bit_in_sync():
    scope = usage_scope(); call = scope['ModelUsageCall']('grok-4.7','xai')
    call.update_usage({'prompt_tokens':100,'completion_tokens':1000,'cost_in_usd_ticks':4000000000})
    assert scope['_sanitize_pricing_calls']([call.snapshot()])[0]['cost_in_usd_ticks'] == 4000000000


def test_shared_container_within_one_generation_is_not_charged_twice():
    scope = usage_scope(); ledger = scope['ModelUsageLedger']()
    for index in range(2):
        call = ledger.begin('gpt-6.1-sol', 'openai')
        call.observe({'output':[{'id':f'code_{index}', 'type':'code_interpreter_call', 'status':'completed', 'container_id':'PRIVATE_CONTAINER'}], 'usage':{'input_tokens':100,'output_tokens':1000}})
        call.finished = True
    records = ledger.snapshot()
    assert [r['container_sessions'] for r in records] == [1, 0]
    result = estimate([answer('gpt-6.1-sol', None, None, thought_data=json.dumps({'pricing_calls':records}))])
    assert result['low'] == pytest.approx(.0102*2 + .03)
    assert 'PRIVATE_CONTAINER' not in json.dumps(records)
