#!/usr/bin/env python3
"""Send a feedback as the administrator, for a problem an AI agent found but did not fix.

The feedback is stored exactly like one sent from the app (database row, ``feedback/<ID>/feedback.json``),
with ``client`` set to ``agent`` and the agent's own name (``--agent``) in ``agent_name``, which the admin
screen shows instead of the administrator account. Nothing is restarted or committed.

Usage (from the app directory)
------------------------------
    venv/bin/python scripts/send_agent_feedback.py --agent "Claude Code" --title "〇〇の不具合" --message-file body.txt
    venv/bin/python scripts/send_agent_feedback.py --agent "Claude Code" --title "〇〇の不具合" --message-file - < body.txt
    venv/bin/python scripts/send_agent_feedback.py --agent "Claude Code" --title "〇〇の不具合" --message-file body.txt --dry-run

The title gets 「（エージェント調査）」 at the end when it is missing. A feedback with the same title stops
the run (``--allow-duplicate`` sends it anyway). A body that looks like it holds a key or a password stops it.
"""

import argparse
import glob
import json
import os
import re
import sys

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TITLE_SUFFIX = "（エージェント調査）"
AGENT_PATTERN = re.compile(r"^[0-9A-Za-z][0-9A-Za-z .()_-]{0,39}$")
TITLE_LIMIT = 200
MESSAGE_LIMIT = 100_000
SECRET_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9_-]{20,}"), "API key (sk-...)"),
    (re.compile(r"AIza[0-9A-Za-z_-]{30,}"), "Google API key"),
    (re.compile(r"xai-[A-Za-z0-9]{20,}"), "xAI API key"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"), "GitHub token"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
    (re.compile(r"(?i)(password|passwd|secret|api[_-]?key|token)\s*[:=]\s*\S{8,}"), "credential assignment"),
]


def _fail(message, code=1):
    print(f"[error] {message}", file=sys.stderr)
    sys.exit(code)


def _read_message(args):
    if args.message is not None:
        return args.message
    if args.message_file == "-":
        return sys.stdin.read()
    with open(args.message_file, encoding="utf-8") as handle:
        return handle.read()


def _existing_titles(feedback_root):
    titles = {}
    for path in glob.glob(os.path.join(feedback_root, "*", "feedback.json")):
        try:
            with open(path, encoding="utf-8") as handle:
                info = json.load(handle)
        except (OSError, ValueError):
            continue
        if isinstance(info, dict) and info.get("title"):
            titles[str(info["title"]).strip()] = info.get("feedback_id") or os.path.basename(os.path.dirname(path))
    return titles


def main():
    parser = argparse.ArgumentParser(description="Send a feedback for a problem an AI agent found.")
    parser.add_argument("--agent", required=True, help="送るエージェント自身の名前（例: Claude Code）。管理画面に表示される")
    parser.add_argument("--title", required=True, help="件名（末尾に「（エージェント調査）」を自動で付ける）")
    body = parser.add_mutually_exclusive_group(required=True)
    body.add_argument("--message", help="本文")
    body.add_argument("--message-file", help="本文のファイル（- で標準入力）")
    parser.add_argument("--dry-run", action="store_true", help="確認だけして送信しない")
    parser.add_argument("--allow-duplicate", action="store_true", help="同じ件名があっても送る")
    args = parser.parse_args()

    agent = args.agent.strip()
    if not AGENT_PATTERN.match(agent):
        _fail("--agent must be 1-40 letters, digits, spaces or . ( ) _ - (for example: Claude Code)")
    title = args.title.strip()
    if not title.endswith(TITLE_SUFFIX):
        title += TITLE_SUFFIX
    if len(title) > TITLE_LIMIT:
        _fail(f"title is longer than {TITLE_LIMIT} characters")
    message = _read_message(args).replace("\x00", "").strip()
    if not message:
        _fail("message is empty")
    if len(message) > MESSAGE_LIMIT:
        _fail(f"message is longer than {MESSAGE_LIMIT} characters")
    for pattern, label in SECRET_PATTERNS:
        if pattern.search(title) or pattern.search(message):
            _fail(f"the feedback looks like it contains a secret ({label}); remove it and run again")

    sys.path.insert(0, APP_DIR)
    os.chdir(APP_DIR)
    import app as target  # noqa: E402  (loads the app the same way the services do)

    app, db = target.app, target.db
    with app.app_context():
        feedback_root = target._feedback_root_dir()
        duplicate = _existing_titles(feedback_root).get(title)
        if duplicate and not args.allow_duplicate:
            _fail(f"a feedback with the same title already exists: {duplicate}", code=3)
        user = target.User.query.filter_by(is_admin=True).order_by(target.User.id).first()
        if user is None:
            _fail("no administrator account")

        print(f"agent:   {agent}")
        print(f"title:   {title}")
        print(f"account: id={user.id}")
        print(f"message: {len(message)} characters")
        if args.dry_run:
            print("[dry-run] not sent")
            return 0

        fb = target.Feedback(user_id=user.id, title=title, message=message, agent_name=agent)
        db.session.add(fb)
        db.session.commit()
        target._reset_feedback_dir(fb.public_id)
        target._write_feedback_info(
            fb, client="agent", client_version=None,
            server_version=app.config.get("SYSTEM_VERSION"),
            user_agent=f"{agent} (AI agent)", activity_log_saved=None, chat_copy_saved=None)

        with open(os.path.join(target._feedback_dir(fb.public_id), "feedback.json"), encoding="utf-8") as handle:
            saved = json.load(handle)
        if saved.get("title") != title or saved.get("client") != "agent" or saved.get("agent_name") != agent:
            _fail(f"feedback {fb.public_id} was created but feedback.json does not match", code=2)
        print(f"[ok] created feedback {fb.public_id} (status: {saved.get('status')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
