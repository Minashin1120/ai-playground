#!/usr/bin/env bash
# Rebuild the combined chat_core source from its ordered parts, then minify the
# versioned assets with esbuild.  The source parts stay readable for editing and
# regression tests; browsers load the .min counterparts.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v npx >/dev/null 2>&1; then
  echo "npx (Node.js) is required to minify frontend assets." >&2
  exit 1
fi

VERSION_FILE="$(ls -1 static/js/chat_core.v4.8.*.js | grep -v '\.min\.' | sort | tail -n 1)"
if [[ -z "$VERSION_FILE" ]]; then
  echo "No chat_core.v4.8.*.js source file found." >&2
  exit 1
fi
VERSION="$(basename "$VERSION_FILE" | sed -E 's/^chat_core\.(v[0-9.]+)\.js$/\1/')"

JS_SRC="static/js/chat_core.${VERSION}.js"

# chat_core is edited as ordered parts under static/js/chat_core_parts/.
# Rebuild the combined versioned source from the parts so the versioned file and
# the minified browser asset always reflect the parts.  If the parts directory
# is missing (legacy layout), fall back to minifying the existing combined
# source as before.
PARTS_DIR="static/js/chat_core_parts"
if [[ -d "$PARTS_DIR" ]] && compgen -G "$PARTS_DIR/chat_core.part*.js" > /dev/null; then
  PARTS=( $(ls -1 "$PARTS_DIR"/chat_core.part*.js | sort) )
  if [[ ${#PARTS[@]} -eq 0 ]]; then
    echo "No chat_core.part*.js files found in $PARTS_DIR." >&2
    exit 1
  fi
  : > "$JS_SRC"
  for part in "${PARTS[@]}"; do
    cat "$part" >> "$JS_SRC"
  done
  echo "Rebuilt $JS_SRC from ${#PARTS[@]} parts:"
  ls -1 "$PARTS_DIR"/chat_core.part*.js
fi

ESBUILD=(npx --yes esbuild@0.25.9)

# Standalone scripts are one IIFE each.  Wrap their output too, so esbuild's
# --keep-names helper (a short name such as `p`) stays inside instead of becoming
# a global `var` that clashes with chat_core's top-level `let p` (the page then
# drops the whole script with "Identifier 'p' has already been declared").
minify_standalone_js() {
  "${ESBUILD[@]}" "$1" \
    --minify \
    --legal-comments=none \
    --keep-names \
    --format=iife \
    --target=es2019 \
    --line-limit=100 \
    --outfile="$2"
}

CSS_SRC="static/css/chat.custom.${VERSION}.css"
CSS_MIN="static/css/chat.custom.min.${VERSION}.css"
TW_SRC="static/css/chat.tailwind.${VERSION}.css"

for required in "$JS_SRC" "$CSS_SRC" "$TW_SRC"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required source asset: $required" >&2
    exit 1
  fi
done

# Browsers load chat_core as several files in order (chat_core.min.<version>.<n>.js)
# so no single file grows without bound.  The combined source is minified once
# (one copy of esbuild's --keep-names helper, as for a single file: separately
# minified pieces would each declare it as another short global) and cut where
# esbuild kept the top-level `//! @chat-core-bundle-split` comments.  Every piece
# must parse on its own, and a piece must not call, while it loads, a function
# declared in a later piece (tests/test_chat_core_bundles.py runs them in Chromium).
BUNDLE_DIR="$(mktemp -d)"
trap 'rm -rf "$BUNDLE_DIR"' EXIT
"${ESBUILD[@]}" "$JS_SRC" \
  --minify \
  --legal-comments=inline \
  --keep-names \
  --target=es2019 \
  --line-limit=100 \
  --outfile="$BUNDLE_DIR/combined.min.js"
rm -f "static/js/chat_core.min.${VERSION}.js" static/js/chat_core.min."${VERSION}".*.js
mapfile -t JS_MINS < <(python3 - "$JS_SRC" "$BUNDLE_DIR/combined.min.js" "static/js/chat_core.min.${VERSION}" <<'PY'
import re
import sys
from pathlib import Path
source, minified, prefix = sys.argv[1:]
marker = re.compile(r"^[ \t]*//! @chat-core-bundle-split", re.MULTILINE)
expected = len(marker.findall(Path(source).read_text(encoding="utf-8")))
text = Path(minified).read_text(encoding="utf-8")
cuts = list(re.finditer(r"//! @chat-core-bundle-split[^\n]*\n?", text))
if len(cuts) != expected:
    sys.exit(f"esbuild kept {len(cuts)} of {expected} @chat-core-bundle-split comments; keep them between top-level statements")
starts = [0] + [m.end() for m in cuts]
ends = [m.start() for m in cuts] + [len(text)]
for index, (start, end) in enumerate(zip(starts, ends), 1):
    path = f"{prefix}.{index}.js"
    piece = text[start:end]
    Path(path).write_text(piece if piece.endswith("\n") else piece + "\n", encoding="utf-8")
    print(path)
PY
)
if [[ ${#JS_MINS[@]} -eq 0 ]]; then
  echo "chat_core was not split into browser files." >&2
  exit 1
fi
for piece in "${JS_MINS[@]}"; do
  if ! node --check "$piece"; then
    echo "$piece does not parse on its own; put the @chat-core-bundle-split line between top-level statements." >&2
    exit 1
  fi
done
# Do not parse-minify CSS. esbuild rewrites Tailwind/arbitrary selectors
# (especially comma escapes) and drops layout utilities.
cp "$CSS_SRC" "$CSS_MIN"

if [[ -f static/js/activity_log.js ]]; then
  minify_standalone_js static/js/activity_log.js static/js/activity_log.min.js
fi
if [[ -f static/js/progress_spinner.js ]]; then
  minify_standalone_js static/js/progress_spinner.js static/js/progress_spinner.min.js
fi
if [[ -f static/js/connection_monitor.js ]]; then
  minify_standalone_js static/js/connection_monitor.js static/js/connection_monitor.min.js
fi
if [[ -f static/js/pwa_install.js ]]; then
  minify_standalone_js static/js/pwa_install.js static/js/pwa_install.min.js
fi
if [[ -f static/js/landing_demo.js ]]; then
  minify_standalone_js static/js/landing_demo.js static/js/landing_demo.min.js
fi

python3 - "$JS_SRC" "$CSS_SRC" "$CSS_MIN" "${JS_MINS[@]}" <<'PY'
import sys
from pathlib import Path
js_src, css_src, css_min, *js_mins = sys.argv[1:]
print("Minified frontend assets:")
total = sum(Path(p).stat().st_size for p in js_mins)
print(f"  chat_core: {total} bytes in {len(js_mins)} files (from {Path(js_src).stat().st_size}, "
      f"{100 * total / Path(js_src).stat().st_size:.1f}%)")
for path in js_mins:
    print(f"    {path}: {Path(path).stat().st_size} bytes")
s = Path(css_src).stat().st_size
d = Path(css_min).stat().st_size
print(f"  {css_min}: {d} bytes (from {s}, {100*d/s:.1f}%)")
PY
