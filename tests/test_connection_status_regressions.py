from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


def _connection_script():
    assets = list((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
    assert len(assets) == 1, "Only the latest versioned chat asset may remain"
    return assets[0].read_text(encoding="utf-8")


class ConnectionStatusRegressionTests(unittest.TestCase):
    def test_monitor_checks_often_and_does_not_hold_a_stale_state(self):
        source = _connection_script()

        self.assertIn("const CONNECTION_CHECK_INTERVAL_MS = 5000", source)
        self.assertIn("const CONNECTION_CHECK_FAST_INTERVAL_MS = 2000", source)
        self.assertIn("const CONNECTION_CHECK_TIMEOUT_MS = 3000", source)
        self.assertNotIn("CONNECTION_UNSTABLE_HOLD_MS", source)
        self.assertNotIn("connectionUnstableUntil", source)

    def test_success_immediately_replaces_offline_or_unstable_state(self):
        source = _connection_script()
        probe = source[source.index("async function probeServerConnection()") :]
        probe = probe[: probe.index("function startConnectionMonitor()")]
        success = probe[probe.index("const hbData = await heartbeatRes.json()") : probe.index("} catch (e)")]

        self.assertIn("const wasDisconnected = connectionStatus === 'offline' || connectionStatus === 'unstable'", success)
        self.assertIn("connectionStatus = 'online'", success)
        self.assertIn("showConnectionRecoveredBanner()", success)
        self.assertNotIn("hadFailure", success)
        self.assertLess(success.index("connectionStatus = 'online'"), success.index("await purgeCaches()"))

    def test_failed_heartbeat_is_not_left_as_merely_unstable(self):
        source = _connection_script()
        probe = source[source.index("async function probeServerConnection()") :]
        probe = probe[: probe.index("function startConnectionMonitor()")]
        failure = probe[probe.index("} catch (e)") :]

        self.assertIn("connectionStatus = 'offline'", failure)
        self.assertIn("setConnectionBanner('offline')", failure)
        self.assertNotIn("setConnectionBanner('unstable'", failure)

    def test_slow_response_needs_repeated_samples_and_events_refresh_immediately(self):
        source = _connection_script()

        self.assertIn("const CONNECTION_SLOW_TO_UNSTABLE = 3", source)
        self.assertIn("connectionConsecutiveSlow >= CONNECTION_SLOW_TO_UNSTABLE", source)
        self.assertIn("function cancelActiveConnectionProbe()", source)
        self.assertIn("if (probeSequence !== connectionProbeSequence) return", source)
        self.assertIn("window.addEventListener('online', () => {", source)
        self.assertIn("window.addEventListener('offline', () => {", source)
        self.assertIn("window.addEventListener('focus', probeServerConnection)", source)
        self.assertIn("if (!document.hidden) probeServerConnection()", source)


if __name__ == "__main__":
    unittest.main()
