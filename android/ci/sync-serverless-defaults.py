#!/usr/bin/env python3
"""Generate android/app/src/main/assets/serverless-defaults.json from the server.

Serverless mode and the no-account profile run without the server, so the app ships the
server's model catalog (the /api/mobile/v1/me shape), the automatic system prompt notice
texts and the Coding Mode prompt. Run from the repository root with the app's venv:

    venv/bin/python android/ci/sync-serverless-defaults.py          # rewrite the asset
    venv/bin/python android/ci/sync-serverless-defaults.py --check  # fail if it is stale
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSET = os.path.join(ROOT, 'android', 'app', 'src', 'main', 'assets', 'serverless-defaults.json')


def build():
    sys.path.insert(0, ROOT)
    import app as target  # noqa: E402  (shared namespace of all server parts)

    notices = target._build_default_auto_system_prompt_notices_config()
    return {
        'format': 'ai-playground-serverless-defaults',
        'version': 1,
        'default_model': target.User.default_model.default.arg,
        'models': [target._mobile_model_metadata(model_id) for model_id in sorted(target.ALL_VALID_MODEL_IDS)],
        'auto_system_prompt_notices_config': {
            key: {'label': item['label'], 'enabled': True, 'text': item['text']}
            for key, item in notices.items()
        },
        'coding_mode_system_prompt': target.CODING_MODE_SYSTEM_PROMPT,
    }


def render(payload):
    return json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    text = render(build())
    if args.check:
        current = open(ASSET, encoding='utf-8').read() if os.path.exists(ASSET) else ''
        if current != text:
            print('serverless-defaults.json is stale; run android/ci/sync-serverless-defaults.py', file=sys.stderr)
            return 1
        print('serverless-defaults.json is up to date')
        return 0
    with open(ASSET, 'w', encoding='utf-8') as handle:
        handle.write(text)
    print(f'wrote {ASSET}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
