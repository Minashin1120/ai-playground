#!/usr/bin/env python3
"""List Japanese strings in the Android sources that do not appear in the Web sources.

The Web wording is the reference (WEB_PARITY.md "文字列"). A string shows up here when one of its
literal parts (text between Kotlin `${...}` templates) is missing from the Web templates, chat JS,
shared JS, server messages or legal documents. Android-only screens listed in ANDROID_ONLY.md
(sign-in, first-run setup, APK update, device cache, notifications) are skipped.

Usage (from the repository root): python3 android/ci/audit-web-strings.py
"""
import glob
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SOURCE = os.path.join(ROOT, 'android/app/src/main/java/com/minashin1120/aiplayground')
WEB_GLOBS = ['templates/**/*.html', 'static/js/chat_core_parts/*.js', 'static/js/*.js', 'server/*.py', 'static/legal/*.md']
ANDROID_ONLY_FILES = {
    'ChatViewModel.kt', 'MainActivity.kt', 'BubbleNotifications.kt', 'BatchNotifications.kt',
    'AppUpdateViewModel.kt', 'AppChangelogViewModel.kt',
}
ANDROID_ONLY_MARKERS = ('data/App', 'Offline', 'Auth', 'Passkey', 'Integrity', 'Reauth', 'AppUpdate', 'AppChangelog')
JAPANESE = re.compile(r'[぀-ヿ一-鿿]')


def normalize(text):
    return re.sub(r'\s+', ' ', text.replace('\\n', ' ')).strip()


def main():
    web = []
    for pattern in WEB_GLOBS:
        for path in glob.glob(os.path.join(ROOT, pattern), recursive=True):
            if '.min.' not in path:
                with open(path, encoding='utf-8', errors='ignore') as handle:
                    web.append(handle.read())
    haystack = normalize('\n'.join(web))
    rows = []
    for path in sorted(glob.glob(os.path.join(SOURCE, '**/*.kt'), recursive=True)):
        rel = os.path.relpath(path, SOURCE)
        if rel in ANDROID_ONLY_FILES or any(marker in rel for marker in ANDROID_ONLY_MARKERS):
            continue
        with open(path, encoding='utf-8') as handle:
            source = handle.read()
        for match in re.finditer(r'"((?:[^"\\\n]|\\.)*)"', source):
            text = match.group(1)
            if not JAPANESE.search(text):
                continue
            parts = [p for p in re.split(r'\$\{[^}]*\}|\$[a-zA-Z_]+', text) if JAPANESE.search(p)]
            if any(normalize(p) not in haystack for p in parts):
                line = source.count('\n', 0, match.start()) + 1
                rows.append(f'{rel}:{line}: {text[:120]}')
    print('\n'.join(rows))
    print(f'{len(rows)} strings to review', file=sys.stderr)


if __name__ == '__main__':
    main()
