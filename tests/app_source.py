"""Load app.py plus server/*.py as one source string for source-inspection tests."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_PARTS_RE = re.compile(r'_SERVER_PARTS\s*=\s*\[(.*?)\]', re.S)


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
