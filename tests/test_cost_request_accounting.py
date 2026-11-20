"""Expense sums are per model request, including tool rounds, retries and fixed-price images."""
import json
import pytest
from tests.test_model_cost_estimates import answer, estimate


def with_calls(calls):
    return answer('gpt-6.1-sol', 1, 1, thought_data=json.dumps({'pricing_calls': calls}))


def test_each_request_uses_its_own_context_tier_and_model():
    calls = [{'model': 'gpt-6.1-sol', 'input_tokens': 200000, 'output_tokens': 1000, 'complete': True},
             {'model': 'gpt-6.1-sol', 'input_tokens': 272001, 'output_tokens': 2000, 'complete': True},
             {'model': 'gpt-4o-mini', 'input_tokens': 100, 'output_tokens': 20, 'complete': True}]
    cost = estimate([with_calls(calls)])
    assert cost['low'] == pytest.approx((200000 * 2 + 1000 * 10 + 272001 * 4 + 2000 * 15 + 100 * .15 + 20 * .6) / 1e6)
    assert cost['long'] == 1 and cost['requests'] == 3


def test_sum_of_short_requests_does_not_trigger_a_long_request_rate():
    calls = [{'model': 'grok-4.6', 'input_tokens': 150000, 'output_tokens': 1000, 'complete': True}] * 2
    cost = estimate([with_calls(calls)])
    assert cost['low'] == pytest.approx(2 * (150000 * 2 + 1000 * 6) / 1e6)
    assert cost['long'] == 0


def test_unknown_retry_is_marked_and_does_not_hide_known_requests():
    cost = estimate([with_calls([{'model': 'gpt-6.1-sol', 'complete': False},
        {'model': 'gpt-6.1-sol', 'input_tokens': 100, 'output_tokens': 1000, 'complete': True}])])
    assert cost['low'] == pytest.approx(.0102)
    assert cost['unpriced'] == 1 and cost['requests'] == 2


@pytest.mark.parametrize('operation,quality,unit', [('generate', 'low', .03), ('generate', 'medium', .06),
    ('generate', 'high', .10), ('edit', 'very_low', .008), ('edit', 'low', .03), ('edit', 'medium', .06), ('edit', 'high', .22)])
@pytest.mark.parametrize('resolution', ['1k', '2k'])
def test_ideogram_45_charges_by_actual_image_count_and_quality(operation, quality, unit, resolution):
    call = {'model': 'ideogram-4.5', 'images': 3, 'operation': operation, 'quality': quality, 'resolution': resolution, 'complete': True}
    cost = estimate([with_calls([call])])
    assert cost['low'] == pytest.approx(3 * unit) and cost['high'] == pytest.approx(3 * unit)
    assert cost['unpriced'] == 0


def test_existing_ideogram_answers_use_saved_parent_options_and_images():
    parent = {'id': 1, 'role': 'user', 'tokens_in': 10, 'prompt_options': {'ideogram_quality': 'high'}, 'image_url': '["source.png"]'}
    row = answer('ideogram-4.5', None, None, image_url='["one.png","two.png"]', parent_id=1)
    assert estimate([parent, row])['low'] == pytest.approx(.44)


def test_thinking_is_part_of_output_and_not_billed_twice():
    call = {'model': 'gpt-6.1-sol', 'input_tokens': 100, 'output_tokens': 1000, 'thought_tokens': 900, 'complete': True}
    assert estimate([with_calls([call])])['low'] == pytest.approx(.0102)


def test_breakdown_items_carry_each_requests_own_cost_and_usage():
    calls = [{'model': 'gpt-6.1-sol', 'input_tokens': 300000, 'output_tokens': 1000, 'complete': True,
              'cache_read_tokens': 20, 'web_search_calls': 2},
             {'model': 'gpt-4o-mini', 'input_tokens': 100, 'output_tokens': 20, 'complete': True}]
    cost = estimate([with_calls(calls)])
    items = cost['items']
    assert [item['model'] for item in items] == ['gpt-6.1-sol', 'gpt-4o-mini']
    assert sum(item['low'] for item in items) == pytest.approx(cost['low'])
    assert sum(item['high'] for item in items) == pytest.approx(cost['high'])
    assert items[0]['long'] is True and items[1]['long'] is False
    assert items[0]['input'] == 300000 and items[0]['output'] == 1000
    assert items[0]['cache_read'] == 20 and items[0]['tools'] == 2
    assert all(item['priced'] for item in items)
    assert all(item['reason'] == '' for item in items)


def test_breakdown_explains_which_requests_are_unpriced():
    cost = estimate([with_calls([
        {'model': 'unknown-model-x', 'input_tokens': 10, 'output_tokens': 10, 'complete': True},
        {'model': 'gpt-6.1-sol', 'complete': False},
        {'model': 'gpt-6.1-sol', 'input_tokens': 100, 'output_tokens': 1000, 'complete': True}])])
    items = cost['items']
    assert [item['priced'] for item in items] == [False, False, True]
    assert items[0]['reason'] == '単価未登録'
    assert items[1]['reason'] == '使用量未取得'
    assert items[2]['reason'] == ''
    assert cost['unpriced'] == 2


def test_breakdown_flags_usage_reconstructed_from_history():
    parent = {'id': 1, 'role': 'user', 'tokens_in': 200000}
    row = answer('gpt-6.1-sol', None, 1000, parent_id=1)
    item = estimate([parent, row])['items'][-1]
    assert item['inferred'] is True and item['input'] == 200000
    assert item['reason'] == ''

