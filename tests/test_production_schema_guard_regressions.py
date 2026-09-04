import os
import unittest
from pathlib import Path


from tests.app_source import read_app_source
os.environ.setdefault("FLASK_SECRET_KEY", "schema-guard-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-schema-guard-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target


ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = read_app_source()
VERIFY_SCRIPT = (ROOT / "scripts" / "verify_changes.sh").read_text(encoding="utf-8")
CONFTEST = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")


class ProductionSchemaGuardRegressionTests(unittest.TestCase):
    def test_sqlite_drop_all_still_runs(self):
        self.assertTrue(target.schema_reset_is_allowed("sqlite:////tmp/x.db"))
        with target.app.app_context():
            previous = target.app.config["SQLALCHEMY_DATABASE_URI"]
            try:
                target.app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////tmp/ai-chat-schema-guard-tests.db"
                target.db.drop_all()
                target.db.create_all()
            finally:
                target.app.config["SQLALCHEMY_DATABASE_URI"] = previous

    def test_mysql_drop_all_is_refused(self):
        previous = target.app.config["SQLALCHEMY_DATABASE_URI"]
        previous_allow = os.environ.pop("ALLOW_PRODUCTION_SCHEMA_RESET", None)
        try:
            target.app.config["SQLALCHEMY_DATABASE_URI"] = (
                "mysql+pymysql://user:pass@127.0.0.1/ai_chat_db"
            )
            self.assertFalse(target.schema_reset_is_allowed())
            with self.assertRaises(RuntimeError) as raised:
                target.db.drop_all()
            self.assertIn("refusing db.drop_all", str(raised.exception))
            with self.assertRaises(RuntimeError):
                target.db.metadata.drop_all()
        finally:
            target.app.config["SQLALCHEMY_DATABASE_URI"] = previous
            if previous_allow is not None:
                os.environ["ALLOW_PRODUCTION_SCHEMA_RESET"] = previous_allow

    def test_guard_is_installed_in_app(self):
        self.assertIn("def _install_schema_reset_guard():", APP_SOURCE)
        self.assertIn("db.drop_all = guarded_drop_all", APP_SOURCE)
        self.assertIn("db.metadata.drop_all = guarded_metadata_drop_all", APP_SOURCE)
        self.assertIn("_install_schema_reset_guard()", APP_SOURCE)

    def test_pytest_conftest_overrides_non_sqlite_url(self):
        self.assertIn("os.environ[\"DATABASE_URL\"] = _DEFAULT_TEST_DB", CONFTEST)
        self.assertIn("_sqlite_database_uri", CONFTEST)

    def test_verify_script_exports_sqlite_database_url(self):
        self.assertIn("export DATABASE_URL=", VERIFY_SCRIPT)
        self.assertIn("sqlite:////tmp/ai-chat-verify.db", VERIFY_SCRIPT)


if __name__ == "__main__":
    unittest.main()
