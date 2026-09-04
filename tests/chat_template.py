"""Load chat.html plus its included fragments as one markup string."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_TEMPLATES = ROOT / "templates"
_INCLUDE_RE = re.compile(r"\{%\s*include\s+'chat/([^']+)'\s*%\}")


def read_chat_markup():
    """Return chat.html and templates/chat fragments in include order.

    Regression tests historically searched a single chat.html.  The live
    template is now split with Jinja includes; this helper keeps those
    assertions working without each test reimplementing include expansion.
    """
    main = (_TEMPLATES / "chat.html").read_text(encoding="utf-8")
    chunks = [main]
    for name in _INCLUDE_RE.findall(main):
        path = _TEMPLATES / "chat" / name
        chunks.append(path.read_text(encoding="utf-8"))
    return "\n".join(chunks)
