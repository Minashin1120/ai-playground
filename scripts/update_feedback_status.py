#!/usr/bin/env python3
"""Change the status of feedbacks through the same API the admin screen uses.

Each change is a POST to ``/api/feedback/<ID>/update`` as the administrator account with the smallest ID,
so the database row, ``updated_at`` and ``feedback/<ID>/feedback.json`` are written exactly as when the
「保存」 button is pressed. ``handled_by`` gets the agent's own name (``--agent``) instead of the account's,
passed in a WSGI environ key that an HTTP request cannot set. Nothing is restarted or committed.

Usage (from the app directory)
------------------------------
    venv/bin/python scripts/update_feedback_status.py --agent "Claude Code" --set 1d933dc9c016=resolved --set 391cc54136ea=in_review
    venv/bin/python scripts/update_feedback_status.py --agent "Claude Code" --set 1d933dc9c016=resolved --dry-run
    venv/bin/python scripts/update_feedback_status.py --agent "Claude Code" --set 391cc54136ea=replied --reply-file reply.txt

The status must be one the admin screen offers. A reply is sent to the person who wrote the feedback,
so it is accepted only with a single ``--set``. The message body of a feedback is never printed.
"""

import argparse
import json
import os
import re
import secrets
import sys

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATUSES = ("new", "in_review", "replied", "rejected", "resolved")
ID_PATTERN = re.compile(r"^[0-9A-Za-z_-]{1,64}$")
AGENT_PATTERN = re.compile(r"^[0-9A-Za-z][0-9A-Za-z .()_-]{0,39}$")
REPLY_LIMIT = 20_000
TITLE_PREVIEW = 40


def _fail(message, code=1):
    print(f"[error] {message}", file=sys.stderr)
    sys.exit(code)


def _parse_changes(values):
    changes = []
    for value in values:
        fid, sep, status = value.partition("=")
        fid, status = fid.strip(), status.strip()
        if not sep or not ID_PATTERN.match(fid):
            _fail(f"--set must be ID=STATUS: {value!r}")
        if status not in STATUSES:
            _fail(f"unknown status {status!r} (use one of: {', '.join(STATUSES)})")
        if any(fid == known for known, _ in changes):
            _fail(f"feedback {fid} is given more than once")
        changes.append((fid, status))
    return changes


def _read_reply(args):
    if args.reply is not None:
        return args.reply
    if args.reply_file is None:
        return None
    if args.reply_file == "-":
        return sys.stdin.read()
    with open(args.reply_file, encoding="utf-8") as handle:
        return handle.read()


def _saved_info(target, public_id):
    try:
        with open(os.path.join(target._feedback_dir(public_id), "feedback.json"), encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def main():
    parser = argparse.ArgumentParser(description="Change the status of feedbacks as the administrator.")
    parser.add_argument("--agent", required=True, help="変更するエージェント自身の名前（例: Claude Code）。対応者として記録される")
    parser.add_argument("--set", action="append", required=True, metavar="ID=STATUS",
                        help=f"変更するフィードバックと状態（{' / '.join(STATUSES)}）。複数指定できる")
    reply = parser.add_mutually_exclusive_group()
    reply.add_argument("--reply", help="返信（フィードバックの送信者に表示される）。--set が1件のときだけ")
    reply.add_argument("--reply-file", help="返信のファイル（- で標準入力）")
    parser.add_argument("--dry-run", action="store_true", help="現在の状態を表示するだけで変更しない")
    args = parser.parse_args()

    agent = args.agent.strip()
    if not AGENT_PATTERN.match(agent):
        _fail("--agent must be 1-40 letters, digits, spaces or . ( ) _ - (for example: Claude Code)")
    changes = _parse_changes(args.set)
    reply_text = _read_reply(args)
    if reply_text is not None:
        reply_text = reply_text.replace("\x00", "").strip()
        if len(changes) != 1:
            _fail("a reply can be sent only with a single --set")
        if not reply_text:
            _fail("reply is empty")
        if len(reply_text) > REPLY_LIMIT:
            _fail(f"reply is longer than {REPLY_LIMIT} characters")

    sys.path.insert(0, APP_DIR)
    os.chdir(APP_DIR)
    import app as target  # noqa: E402  (loads the app the same way the services do)

    app = target.app
    with app.app_context():
        found = {}
        for fid, _ in changes:
            fb = target._feedback_by_ref(fid)
            if fb is None:
                _fail(f"feedback {fid} was not found")
            found[fid] = (fb.public_id or fid, fb.status or "new", (fb.title or "")[:TITLE_PREVIEW])
        admin = target.User.query.filter_by(is_admin=True).order_by(target.User.id).first()
        if admin is None:
            _fail("no administrator account")

        print(f"agent:   {agent}")
        print(f"account: id={admin.id}")
        for fid, status in changes:
            public_id, current, title = found[fid]
            print(f"{public_id}: {current} -> {status}  {title}")
        if reply_text is not None:
            print(f"reply:   {len(reply_text)} characters")
        if args.dry_run:
            print("[dry-run] not changed")
            return 0

        hosts = app.config.get("TRUSTED_HOSTS") or ["localhost"]
        base_url = f"https://{hosts[0]}"
        csrf_token = secrets.token_urlsafe(32)
        client = app.test_client()
        with client.session_transaction(base_url=base_url) as session:
            session["_user_id"] = str(admin.id)
            session["_fresh"] = True
            session["csrf_token"] = csrf_token

        failed = 0
        for fid, status in changes:
            public_id = found[fid][0]
            payload = {"status": status}
            if reply_text is not None:
                payload["admin_reply"] = reply_text
            response = client.post(
                f"/api/feedback/{public_id}/update", json=payload, base_url=base_url,
                headers={"X-CSRF-Token": csrf_token},
                environ_base={"REMOTE_ADDR": "127.0.0.1", target.FEEDBACK_AGENT_ENVIRON: agent})
            body = response.get_json(silent=True) or {}
            saved = _saved_info(target, public_id)
            ok = (response.status_code == 200 and body.get("status") == "ok" and saved.get("status") == status
                  and saved.get("handled_by") == agent and (reply_text is None or saved.get("admin_reply") == reply_text))
            if ok:
                print(f"[ok] {public_id}: {status}")
            else:
                failed += 1
                print(f"[error] {public_id}: HTTP {response.status_code} {body.get('error') or ''} "
                      f"(feedback.json status: {saved.get('status')})", file=sys.stderr)
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
