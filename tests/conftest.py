"""Force pytest onto an isolated sqlite database.

Test modules historically used os.environ.setdefault("DATABASE_URL", sqlite...).
setdefault does not override a URL that is already in the environment, so
sourcing .env (or inheriting systemd's DATABASE_URL) made db.drop_all() run
against production MySQL.  This module is imported before test modules, and
unconditionally replaces a non-sqlite DATABASE_URL.
"""
import os

# tmpfs keeps the per-test drop_all()/create_all() resets off the disk.
_TEST_DB_DIR = "/dev/shm" if os.access("/dev/shm", os.W_OK) else "/tmp"
_DEFAULT_TEST_DB = f"sqlite:///{_TEST_DB_DIR}/ai-chat-pytest-session.db"


def _sqlite_database_uri(uri):
    return (uri or "").strip().lower().startswith("sqlite:")


forced = os.environ.get("AI_CHAT_TEST_DATABASE_URL", "").strip()
if forced:
    os.environ["DATABASE_URL"] = forced
elif not _sqlite_database_uri(os.environ.get("DATABASE_URL", "")):
    os.environ["DATABASE_URL"] = _DEFAULT_TEST_DB

os.environ.setdefault("FLASK_SECRET_KEY", "pytest-isolate-secret")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")
