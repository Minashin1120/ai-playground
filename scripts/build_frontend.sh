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
JS_MIN="static/js/chat_core.min.${VERSION}.js"

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

minify_js() {
  local src="$1"
  local dest="$2"
  "${ESBUILD[@]}" "$src" \
    --minify \
    --legal-comments=none \
    --keep-names \
    --target=es2019 \
    --line-limit=100 \
    --outfile="$dest"
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

minify_js "$JS_SRC" "$JS_MIN"
# Do not parse-minify CSS. esbuild rewrites Tailwind/arbitrary selectors
# (especially comma escapes) and drops layout utilities.
cp "$CSS_SRC" "$CSS_MIN"

if [[ -f static/js/progress_spinner.js ]]; then
  minify_js static/js/progress_spinner.js static/js/progress_spinner.min.js
fi
if [[ -f static/js/connection_monitor.js ]]; then
  minify_js static/js/connection_monitor.js static/js/connection_monitor.min.js
fi
if [[ -f static/js/pwa_install.js ]]; then
  minify_js static/js/pwa_install.js static/js/pwa_install.min.js
fi
if [[ -f static/js/landing_demo.js ]]; then
  minify_js static/js/landing_demo.js static/js/landing_demo.min.js
fi

python3 - <<PY
from pathlib import Path
pairs = [
    ("$JS_SRC", "$JS_MIN"),
    ("$CSS_SRC", "$CSS_MIN"),
]
print("Minified frontend assets:")
for src, dest in pairs:
    s = Path(src).stat().st_size
    d = Path(dest).stat().st_size
    print(f"  {dest}: {d} bytes (from {s}, {100*d/s:.1f}%)")
PY
