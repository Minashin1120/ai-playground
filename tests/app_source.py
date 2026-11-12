"""Load app.py plus server/*.py as one source string for source-inspection tests."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_PARTS_RE = re.compile(r'_SERVER_PARTS\s*=\s*\[(.*?)\]', re.S)
_WHITESPACE_RE = re.compile(r"\s+")


def compact(text):
    """Collapse whitespace runs so indentation-only moves keep substring checks valid.

    Parts are extracted between functions, so their nested code loses a level of
    indentation.  Tests that asserted a multi-line snippet with exact spaces
    broke; compare `compact(expected)` against `compact(source)` instead.
    """
    return _WHITESPACE_RE.sub(" ", text)


def read_chat_core_source():
    """Return the rebuilt chat_core source so part boundaries can change freely.

    Tests that inspect or slice chat_core code should not depend on a specific
    ``chat_core.partNN`` filename: the parts are re-split when they grow.  The
    combined versioned source is generated from the parts by
    ``scripts/build_frontend.sh`` and is byte-identical to their concatenation.
    """
    js_dir = ROOT / "static" / "js"
    sources = sorted(
        path for path in js_dir.glob("chat_core.v4.8.*.js") if ".min." not in path.name
    )
    if len(sources) != 1:
        raise AssertionError(f"expected one chat_core source, found {[p.name for p in sources]}")
    return sources[0].read_text(encoding="utf-8")


def read_app_source():
    """Return the Flask entry plus every server part in exec order.

    Runtime still imports the `app` module.  Tests that search the old
    monolith as text need this concatenation after the split.  Order matches
    `app.py`'s `_SERVER_PARTS` so slices between markers stay valid.
    """
    app_text = (ROOT / "app.py").read_text(encoding="utf-8")
    chunks = [app_text]
    match = _PARTS_RE.search(app_text)
    names = []
    if match:
        names = re.findall(r'"([^"]+\.py)"', match.group(1))
    server = ROOT / "server"
    for name in names:
        path = server / name
        if path.is_file():
            chunks.append(path.read_text(encoding="utf-8"))
    return "\n".join(chunks)
