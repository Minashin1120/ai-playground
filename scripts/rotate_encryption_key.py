#!/usr/bin/env python3
"""Rotate the application encryption key safely.

This tool rotates ``app/secret.key`` (the Fernet key used to encrypt captured
data at rest) to a brand new key WITHOUT making any existing value unreadable.

Safety model
------------
* A full backup (uploads, database dump, pre-rotation key) is created first and
  is NEVER deleted automatically.
* The new key is written to ``secret.key`` and the current key is retained as
  ``secret.key.rotated.<timestamp>`` BEFORE any data is touched, so both keys
  are always available.  app.py loads the active key plus every
  ``secret.key.rotated.*`` into a key ring, so a value encrypted with either
  the old or the new key stays readable even if this run is interrupted.
* Only values that actually decrypt with the current key are migrated.  Values
  that were already unreadable (pre-existing orphans) are left untouched and
  reported -- they are already lost and rotation must not make it worse.

Usage (from the app directory; see the public ROTATION doc for full steps)
-------------------------------------------------------------------------
    # 1) stop the web + workers (avoid concurrent writes during the sweep)
    sudo systemctl stop ai-chat.service ai-chat-worker@1.service ...

    # 2) create the backups + rotate + re-encrypt + verify (offline)
    venv/bin/python scripts/rotate_encryption_key.py --execute --yes

    # 3) start services and confirm
    sudo systemctl start ai-chat.service ai-chat-worker@1.service ...
"""

import argparse
import datetime
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import time

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from cryptography.fernet import Fernet, InvalidToken  # noqa: E402

# Importing the app loads the key ring (active + retained) and the DB engine.
from app import app, db  # noqa: E402
from app import ACCOUNT_SECRET_FIELDS, User, Message  # noqa: E402

KEY_FILE = os.path.join(APP_DIR, "secret.key")
SKIP_DIR_NAMES = {".chunks", "account_exports", "account_import_uploads"}
EXTRA_ENCRYPTED_USER_FIELDS = ("totp_secret",)
CONDITIONAL_ENCRYPTED_USER_FIELDS = ("system_prompt",)


def _upload_folder():
    # Resolve at call time so tests can redirect UPLOAD_FOLDER.
    return app.config["UPLOAD_FOLDER"]


def _log(msg):
    print(f"[rotate-key] {msg}", flush=True)


def _err(msg):
    print(f"[rotate-key] ERROR: {msg}", file=sys.stderr, flush=True)


def _load_key(path):
    if os.path.islink(path):
        raise RuntimeError(f"refusing to read a symlinked key: {path}")
    os.chmod(path, 0o600)
    with open(path, "rb") as fh:
        return Fernet(fh.read().strip())


def _current_active_key():
    return _load_key(KEY_FILE)


def _user_encrypted_fields(user):
    fields = list(ACCOUNT_SECRET_FIELDS) + list(EXTRA_ENCRYPTED_USER_FIELDS)
    if getattr(user, "enable_e2ee", False):
        fields += list(CONDITIONAL_ENCRYPTED_USER_FIELDS)
    return fields


def _iter_enc_files():
    upt = _upload_folder()
    if not os.path.isdir(upt):
        return
    for root, dirs, files in os.walk(upt):
        dirs[:] = [d for d in dirs if d not in SKIP_DIR_NAMES]
        for name in files:
            if name.endswith(".enc"):
                yield os.path.join(root, name)


# --------------------------------------------------------------------------- #
# backups
# --------------------------------------------------------------------------- #
def _backup_stamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _create_backups(backup_root):
    """Backup uploads, database, and the pre-rotation key. Returns paths dict."""
    stamp = _backup_stamp()
    root = os.path.join(backup_root, f"rotate-{stamp}")
    os.makedirs(root, mode=0o700, exist_ok=True)
    _log(f"creating backups under {root}")

    # uploads
    uploads_dir = os.path.join(root, "uploads")
    os.makedirs(uploads_dir, mode=0o700, exist_ok=True)
    upt = _upload_folder()
    if os.path.isdir(upt):
        for src in os.listdir(upt):
            s = os.path.join(upt, src)
            if src in SKIP_DIR_NAMES:
                continue
            d = os.path.join(uploads_dir, src)
            if os.path.isdir(s):
                shutil.copytree(s, d, symlinks=False)
            else:
                shutil.copy2(s, d)
    _log(f"  uploads copied ({upt})")

    # database
    dump_path = os.path.join(root, "database.sql.gz")
    _dump_database(dump_path)
    _log(f"  database dumped -> {dump_path}")

    # pre-rotation key
    key_copy = os.path.join(root, "secret.key.before")
    shutil.copy2(KEY_FILE, key_copy)
    os.chmod(key_copy, 0o600)
    _log(f"  pre-rotation key -> {key_copy}")

    return root, key_copy


def _dump_database(dest):
    env = {}
    with open(os.path.join(APP_DIR, ".env")) as fh:
        for line in fh:
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    import re
    import gzip
    url = env.get("DATABASE_URL", "")
    m = re.match(
        r"^(?:[a-zA-Z0-9+]+)://([^:/@]+):([^@]*)@([^:/@]+)(?::(\d+))?/([^?]+)", url
    )
    if not m:
        raise SystemExit("cannot parse DATABASE_URL for backup")
    user, pw, host, port, dbname = m.group(1), m.group(2), m.group(3), m.group(4) or "3306", m.group(5)
    envb = dict(os.environ)
    envb["MYSQL_PWD"] = pw
    cmd = ["mysqldump", "--single-transaction", "--routines", "--triggers", "--events",
           "-h", host, "-P", port, "-u", user, dbname]
    with open(dest, "wb") as raw:
        p = subprocess.Popen(cmd, stdout=raw, stderr=subprocess.PIPE, env=envb)
        _, err = p.communicate()
    if p.returncode != 0:
        raise SystemExit(f"mysqldump failed: {err.decode(errors='replace')}")
    os.chmod(dest, 0o600)


# --------------------------------------------------------------------------- #
# pre-flight / migrate / verify
# --------------------------------------------------------------------------- #
def _scan(old_cipher):
    stats = {"user_values": 0, "message_values": 0, "enc_files": 0}
    orphans = []
    with app.app_context():
        for user in User.query.yield_per(500):
            for field in _user_encrypted_fields(user):
                value = getattr(user, field, None)
                if value is None or value == "":
                    continue
                stats["user_values"] += 1
                try:
                    old_cipher.decrypt(value.encode())
                except Exception:
                    orphans.append(f"user#{user.id}.{field}")
        for msg in Message.query.yield_per(500):
            if not msg.is_encrypted:
                continue
            for field in ("content", "thought_data"):
                value = getattr(msg, field, None)
                if value is None or value == "":
                    continue
                stats["message_values"] += 1
                try:
                    old_cipher.decrypt(value.encode())
                except Exception:
                    orphans.append(f"message#{msg.id}.{field}")
    for path in _iter_enc_files():
        stats["enc_files"] += 1
        try:
            with open(path, "rb") as fh:
                old_cipher.decrypt(fh.read())
        except Exception:
            orphans.append(path)
    return stats, orphans


def _migrate(old_cipher, new_cipher):
    """Re-encrypt readable values from old to new. Skips (reports) orphans."""
    updated_users = 0
    updated_msgs = 0
    updated_files = 0
    orphan_users = 0
    orphan_msgs = 0
    orphan_files = 0

    with app.app_context():
        for user in User.query.yield_per(500):
            for field in _user_encrypted_fields(user):
                value = getattr(user, field, None)
                if value is None or value == "":
                    continue
                try:
                    plain = old_cipher.decrypt(value.encode())
                except Exception:
                    orphan_users += 1
                    continue
                setattr(user, field, new_cipher.encrypt(plain).decode())
            updated_users += 1
        for msg in Message.query.yield_per(500):
            if not msg.is_encrypted:
                continue
            for field in ("content", "thought_data"):
                value = getattr(msg, field, None)
                if value is None or value == "":
                    continue
                try:
                    plain = old_cipher.decrypt(value.encode())
                except Exception:
                    orphan_msgs += 1
                    continue
                setattr(msg, field, new_cipher.encrypt(plain).decode())
            updated_msgs += 1
        db.session.commit()

    # Files: stage then atomically rename; unreadable ones left as-is.
    staged = []
    try:
        for path in _iter_enc_files():
            with open(path, "rb") as fh:
                data = fh.read()
            try:
                plain = old_cipher.decrypt(data)
            except Exception:
                orphan_files += 1
                continue
            fd, tmp = tempfile.mkstemp(prefix=".rot-", suffix=".enc.new", dir=os.path.dirname(path))
            os.close(fd)
            with open(tmp, "wb") as fh:
                fh.write(new_cipher.encrypt(plain))
            os.chmod(tmp, 0o600)
            staged.append((path, tmp))
        for path, tmp in staged:
            os.replace(tmp, path)
            updated_files += 1
    except Exception:
        for _, tmp in staged:
            if os.path.exists(tmp):
                os.remove(tmp)
        raise

    return {
        "updated_users": updated_users, "updated_msgs": updated_msgs, "updated_files": updated_files,
        "orphan_users": orphan_users, "orphan_msgs": orphan_msgs, "orphan_files": orphan_files,
    }


def _verify(new_cipher, old_cipher):
    """Confirm every value that was readable before now decrypts with the new key.

    * decrypts with new key -> migrated correctly (OK)
    * decrypts only with the old key -> NOT migrated (problem)
    * decrypts with neither -> was already unreadable before (orphan, OK)
    """
    problems = []
    counts = {"user_values": 0, "message_values": 0, "enc_files": 0}
    orphans = 0
    with app.app_context():
        for user in User.query.yield_per(500):
            for field in _user_encrypted_fields(user):
                value = getattr(user, field, None)
                if value is None or value == "":
                    continue
                counts["user_values"] += 1
                try:
                    new_cipher.decrypt(value.encode())
                except Exception:
                    try:
                        old_cipher.decrypt(value.encode())
                    except Exception:
                        orphans += 1
                    else:
                        problems.append(f"user#{user.id}.{field}: still encrypted with old key")
        for msg in Message.query.yield_per(500):
            if not msg.is_encrypted:
                continue
            for field in ("content", "thought_data"):
                value = getattr(msg, field, None)
                if value is None or value == "":
                    continue
                counts["message_values"] += 1
                try:
                    new_cipher.decrypt(value.encode())
                except Exception:
                    try:
                        old_cipher.decrypt(value.encode())
                    except Exception:
                        orphans += 1
                    else:
                        problems.append(f"message#{msg.id}.{field}: still encrypted with old key")
    for path in _iter_enc_files():
        counts["enc_files"] += 1
        try:
            with open(path, "rb") as fh:
                new_cipher.decrypt(fh.read())
        except Exception:
            try:
                with open(path, "rb") as fh:
                    old_cipher.decrypt(fh.read())
            except Exception:
                orphans += 1
            else:
                problems.append(f"{path}: still encrypted with old key")
    return counts, problems, orphans


# --------------------------------------------------------------------------- #
# service check
# --------------------------------------------------------------------------- #
def _app_services_active():
    active = []
    for unit in ["ai-chat.service", "ai-chat-worker@1.service", "ai-chat-worker@2.service",
                 "ai-chat-worker@3.service", "ai-chat-worker@4.service"]:
        try:
            out = subprocess.run(["systemctl", "is-active", unit], capture_output=True, text=True).stdout.strip()
        except Exception:
            out = ""
        if out == "active":
            active.append(unit)
    return active


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Safely rotate the encryption key")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="validate only, write nothing")
    g.add_argument("--execute", action="store_true", help="perform the rotation")
    parser.add_argument("--yes", action="store_true", help="confirm execute")
    parser.add_argument("--allow-running", action="store_true", help="allow while services are active (unsafe)")
    parser.add_argument("--backup-dir", default=os.path.join(os.path.dirname(APP_DIR), "backups"),
                        help="directory to place rotation backups in")
    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        raise SystemExit("need --dry-run or --execute")

    old_cipher = _current_active_key()

    if args.dry_run:
        _log("Dry run: scanning encrypted data for readability...")
        stats, orphans = _scan(old_cipher)
        _log(f"  user secret values : {stats['user_values']}")
        _log(f"  message values     : {stats['message_values']}")
        _log(f"  .enc files         : {stats['enc_files']}")
        _log(f"  already-unreadable : {len(orphans)} (left untouched by rotation)")
        _log("Dry run finished.  Services must be stopped before --execute.")
        return 0

    if not args.yes:
        raise SystemExit("refusing to execute without --yes")
    if not args.allow_running:
        active = _app_services_active()
        if active:
            _err("these services are still active; stop them first:")
            for unit in active:
                _err(f"    - {unit}")
            raise SystemExit("aborting. Stop services, then re-run (or --allow-running if you accept the risk).")

    # 1) Backups (uploads + DB + pre-rotation key) -- never auto-deleted.
    backup_root, pre_key_copy = _create_backups(args.backup_dir)
    _log(f"backups ready under {backup_root}")

    # 2) Pre-flight
    _log("pre-flight: scanning encrypted data...")
    stats, orphans = _scan(old_cipher)
    _log(f"  scope: {stats['user_values']} user values, {stats['message_values']} message values, "
         f"{stats['enc_files']} .enc files, {len(orphans)} already-unreadable (will be skipped)")
    _log("  (the pre-rotation key is retained, so nothing can be lost)")

    # 3) Persist the NEW key immediately; retain the OLD key as historical.
    new_key = Fernet.generate_key()
    hist = f"{KEY_FILE}.rotated.{_backup_stamp()}"
    shutil.copy2(KEY_FILE, hist)
    os.chmod(hist, 0o600)
    fd, tmp = tempfile.mkstemp(prefix=".secret.", suffix=".new", dir=APP_DIR)
    os.close(fd)
    with open(tmp, "wb") as fh:
        fh.write(new_key)
    os.chmod(tmp, 0o600)
    os.replace(tmp, KEY_FILE)
    os.chmod(KEY_FILE, 0o600)
    _log(f"  new key written to {KEY_FILE}; old key retained at {hist}")
    new_cipher = Fernet(new_key)

    # 4) Re-encrypt
    _log("re-encrypting readable data (old -> new)...")
    res = _migrate(old_cipher, new_cipher)
    _log(f"  migrated: {res['updated_users']} user rows, {res['updated_msgs']} messages, "
         f"{res['updated_files']} files; skipped already-unreadable "
         f"({res['orphan_users']} user + {res['orphan_msgs']} msg + {res['orphan_files']} file)")

    # 5) Verify
    _log("verifying with the new key...")
    vcounts, problems, vorphans = _verify(new_cipher, old_cipher)
    if problems:
        _err(f"verification FAILED ({len(problems)}); roll back by restoring the DB dump and "
             f"the key from {backup_root}.  Both keys are still on disk so no data is lost.")
        for p in problems[:30]:
            _err(f"    - {p}")
        raise SystemExit(4)
    _log(f"  verified: {vcounts['user_values']} user values, {vcounts['message_values']} messages, "
         f"{vcounts['enc_files']} files; pre-existing unreadable left: {vorphans}")

    # 6) Post-rotation key backup
    post_key = os.path.join(backup_root, "secret.key.after")
    shutil.copy2(KEY_FILE, post_key)
    os.chmod(post_key, 0o600)
    _log(f"  post-rotation key -> {post_key}")

    _log(f"ROTATION COMPLETE.  Backups retained under {backup_root} (do not delete until told).")
    _log("The new key is active; the previous key is retained at:")
    _log(f"  {hist}")
    _log("Restart services now so the app loads the new active key, then confirm /api/version.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
