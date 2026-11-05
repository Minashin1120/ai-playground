"""Run the real preparation entrypoint in an isolated tree; no release or service side effects."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tree(tmp_path):
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    (tmp_path / 'static').mkdir()
    (tmp_path / 'app.py').write_text('original version')
    (tmp_path / 'changelog.md').write_text('original notes')
    for name in ['prepare_version.sh', '_release_lib.sh', '_prepare_checks.sh']:
        shutil.copy2(ROOT / 'scripts' / name, scripts / name)
    (scripts / '_release_common.py').write_text('''
import os,sys,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
with (root/'events').open('a') as f:f.write(json.dumps(sys.argv[1:])+'\\n')
if sys.argv[1]=='prepare' and '--dry-run' not in sys.argv:
    (root/'app.py').write_text('new version')
    (root/'changelog.md').write_text('new notes')
''')
    (scripts / 'build_frontend.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
printf 'build\\n' >> events
[[ "${FAIL_BUILD:-0}" == 0 ]]
''')
    (scripts / 'verify_changes.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
printf 'verify %s\\n' "$*" >> events
if [[ "${FAIL_CHECK:-0}" == 1 ]]; then exit 1; fi
if [[ "$*" != *--skip-tests* && "${FAIL_TESTS:-0}" == 1 ]]; then exit 1; fi
''')
    for name in ['build_frontend.sh', 'verify_changes.sh']:
        (scripts / name).chmod(0o755)
    return tmp_path


def run(tree, *args, **env):
    return subprocess.run(['bash', str(tree / 'scripts/prepare_version.sh'), '--notes', 'valid notes', *args],
                          cwd=tree, env={**os.environ, **env}, text=True, capture_output=True)


def events(tree):
    return (tree / 'events').read_text().splitlines()


@pytest.mark.parametrize('failure', ['FAIL_BUILD', 'FAIL_TESTS', 'FAIL_CHECK'])
def test_preflight_failure_cannot_advance_version_or_changelog(tree, failure):
    result = run(tree, **{failure: '1'})
    assert result.returncode != 0
    assert (tree / 'app.py').read_text() == 'original version'
    assert (tree / 'changelog.md').read_text() == 'original notes'
    assert not any(row.startswith('["prepare"') for row in events(tree))


def test_full_tests_run_before_bump_and_only_static_checks_after(tree):
    assert run(tree).returncode == 0
    rows = events(tree)
    full = rows.index('verify ')
    bump = next(i for i, row in enumerate(rows) if row.startswith('["prepare"'))
    assert full < bump < rows.index('verify --skip-tests')
    assert rows.count('verify ') == 1
    assert (tree / 'app.py').read_text() == 'new version'


def test_explicit_skip_waives_pytest_but_keeps_static_checks(tree):
    assert run(tree, '--skip-verify', FAIL_TESTS='1').returncode == 0
    assert events(tree).count('verify --skip-tests') == 2
    assert run(tree, '--skip-verify', FAIL_CHECK='1').returncode != 0


def test_dry_run_does_not_build_test_or_mutate(tree):
    assert run(tree, '--dry-run').returncode == 0
    assert all(row.startswith('[') for row in events(tree))
    assert (tree / 'app.py').read_text() == 'original version'


def test_generated_assets_are_readable_before_checks(tree):
    asset = tree / 'static/private.js'
    asset.write_text('test')
    asset.chmod(0o600)
    assert run(tree).returncode == 0
    assert asset.stat().st_mode & 0o004


def test_test_catalog_covers_current_files_and_has_no_broken_links():
    import re
    documents = [ROOT / 'tests/README.md', *(ROOT / 'tests/docs').glob('*.md')]
    catalog = '\n'.join(path.read_text() for path in documents)
    for path in (ROOT / 'tests').glob('test_*.py'):
        assert path.name in catalog, f'Add {path.name} to its topic document'
    for document in documents:
        for target in re.findall(r'\]\(([^)]+)\)', document.read_text()):
            if '://' not in target:
                assert (document.parent / target.split('#')[0]).is_file(), f'{document}: {target}'
