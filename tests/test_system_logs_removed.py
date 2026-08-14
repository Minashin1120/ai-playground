from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


def _current_asset(folder, pattern):
    assets = sorted((APP_ROOT / "static" / folder).glob(pattern))
    assert len(assets) == 1, f"Expected only the latest asset for {pattern}"
    return assets[0].read_text(encoding="utf-8")


class SystemLogsRemovedRegressionTests(unittest.TestCase):
    def test_system_logs_ui_and_backend_are_removed(self):
        template = (APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        script = _current_asset("js", "chat_core.v4.8.*.js")
        css = _current_asset("css", "chat.custom.v4.8.*.css")
        app_source = (APP_ROOT / "app.py").read_text(encoding="utf-8")

        self.assertNotIn('id="debug-console"', template)
        self.assertNotIn("System Logs", template)
        self.assertNotIn("toggleDebug()", template)
        self.assertNotIn("refreshLogs()", template)
        self.assertNotIn("fa-wrench", template)

        self.assertNotIn("window.refreshLogs", script)
        self.assertNotIn("window.toggleDebug", script)
        self.assertNotIn("/api/debug/log", script)
        self.assertNotIn("get('debug-console')", script)

        self.assertNotIn("#debug-console", css)

        self.assertNotIn("@app.route('/api/debug/log'", app_source)
        self.assertNotIn("def debug_log(", app_source)
        self.assertNotIn("sudo', 'journalctl", app_source)
        self.assertNotIn("journalctl', '-u', 'ai-chat'", app_source)


if __name__ == "__main__":
    unittest.main()
