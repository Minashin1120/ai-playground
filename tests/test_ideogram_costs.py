"""Ideogram's selectable models, saved speeds, image quantities and unpublished rates."""
import json
import subprocess
from pathlib import Path

import pytest
from tests.test_model_cost_estimates import PRICING, SOURCE, answer, estimate
from tests.test_ideogram_models import IDEOGRAM, MODEL_IDS

RATES = [('ideogram-4.0', 'turbo', .03), ('ideogram-4.0', 'default', .06), ('ideogram-4.0', 'quality', .10),
         ('ideogram-3.0', 'turbo', .03), ('ideogram-3.0', 'default', .06), ('ideogram-3.0', 'quality', .09),
         ('ideogram-2a', 'turbo', .025), ('ideogram-2a', 'default', .04),
         ('ideogram-2.0', 'turbo', .05), ('ideogram-2.0', 'default', .08)]


def row(model, **call):
    call = dict(dict(model=model, images=3, operation='generate', complete=True), **call)
    return answer(model, None, None, thought_data=json.dumps({'pricing_calls': [call]}))


def test_all_selectable_ideogram_models_have_image_prices():
    for model in MODEL_IDS:
        assert PRICING['models'][model]['unit'] == 'image'


@pytest.mark.parametrize('model,speed,unit', RATES)
def test_published_speed_prices_use_actual_output_count(model, speed, unit):
    cost = estimate([row(model, speed=speed, quality='high')])
    assert cost['low'] == pytest.approx(unit * 3)
    assert cost['high'] == pytest.approx(unit * 3) and cost['unpriced'] == 0


@pytest.mark.parametrize('model', MODEL_IDS)
def test_background_removal_uses_its_endpoint_price_for_any_model(model):
    cost = estimate([row(model, operation='remove_background', speed='quality', quality='high')])
    assert cost['low'] == pytest.approx(.03) and cost['unpriced'] == 0


def test_object_removal_public_minimum_is_not_presented_as_exact_price():
    cost = estimate([row('ideogram-3.0', operation='remove_object', speed='quality')])
    assert cost['low'] == pytest.approx(.09) and cost['lowerBound'] and cost['unpriced'] == 1
    script = 'const fetch = () => new Promise(() => {});\n' + SOURCE
    script += '\nconsole.log(costEstimateLabel(' + json.dumps(cost) + '));'
    label = subprocess.run(['node', '-e', script], check=True, capture_output=True, text=True).stdout
    assert '以上' in label and '一部未算入' in label


@pytest.mark.parametrize('model,minimum', [('ideogram-2a', .025), ('ideogram-2.0', .05)])
def test_unpublished_quality_is_an_explicit_lower_bound(model, minimum):
    cost = estimate([row(model, speed='quality')])
    assert cost['low'] == pytest.approx(3 * minimum) and cost['lowerBound'] and cost['unpriced'] == 1


def test_older_call_records_recover_speed_from_saved_parent():
    parent = dict(id=1, role='user', prompt_options={'ideogram_speed': 'quality'})
    old = row('ideogram-3.0', quality='high')
    old['parent_id'] = 1
    assert estimate([parent, old])['low'] == pytest.approx(.27)
    assert estimate([old])['lowerBound']


def test_legacy_generation_does_not_become_an_edit_due_to_ancestor_images():
    prior = dict(id=8, role='assistant', image_url='["old.png"]')
    parent = dict(id=1, parent_id=8, role='user', prompt_options={'ideogram_speed': 'turbo'})
    old = answer('ideogram-4.0', None, None, image_url='["a.png","b.png"]', parent_id=1)
    assert estimate([prior, parent, old])['low'] == pytest.approx(.06)


@pytest.mark.parametrize('model,speed', [('ideogram-4.0', 'turbo'), ('ideogram-3.0', 'quality'), ('ideogram-2a', None)])
def test_request_records_the_effective_rendering_speed(monkeypatch, model, speed):
    import httpx
    monkeypatch.setattr(httpx, 'post', lambda *args, **kwargs: httpx.Response(200, json={'data': [{'url': 'https://ideogram.ai/image.png'}]}))
    ledger = IDEOGRAM.ModelUsageLedger()
    token = IDEOGRAM._MODEL_USAGE_CONTEXT.set(ledger)
    try:
        IDEOGRAM.ideogram_request('synthetic-key', model, 'synthetic-prompt', {'ideogram_speed': speed})
    finally:
        IDEOGRAM._MODEL_USAGE_CONTEXT.reset(token)
    assert ledger.snapshot()[0]['speed'] == (speed or 'default')


def test_mobile_sync_preserves_only_valid_speeds():
    from tests.test_model_usage_ledger import usage_scope
    clean = usage_scope()['_sanitize_pricing_calls']
    assert clean([{'model': 'ideogram-3.0', 'speed': 'turbo'}])[0]['speed'] == 'turbo'
    assert 'speed' not in clean([{'model': 'ideogram-3.0', 'speed': 'PRIVATE'}])[0]
