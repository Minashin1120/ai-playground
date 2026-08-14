from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


def _connection_script():
    assets = list((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
    assert len(assets) == 1, "Only the latest versioned chat asset may remain"
    return assets[0].read_text(encoding="utf-8")


def _connection_monitor_source():
    return (APP_ROOT / "static" / "js" / "connection_monitor.js").read_text(encoding="utf-8")


def _monitor_probe():
    source = _connection_monitor_source()
    probe = source[source.index("async function probeServerConnection()") :]
    probe = probe[: probe.index("function start()")]
    return probe


class ConnectionStatusRegressionTests(unittest.TestCase):
    def test_monitor_checks_often_and_does_not_hold_a_stale_state(self):
        source = _connection_monitor_source()

        self.assertIn("const CONNECTION_CHECK_INTERVAL_MS = 5000", source)
        self.assertIn("const CONNECTION_CHECK_FAST_INTERVAL_MS = 2000", source)
        self.assertIn("const CONNECTION_CHECK_TIMEOUT_MS = 3000", source)
        self.assertNotIn("CONNECTION_UNSTABLE_HOLD_MS", source)
        self.assertNotIn("connectionUnstableUntil", source)

    def test_heartbeat_logic_lives_in_its_own_module_loaded_before_chat_core(self):
        template = (APP_ROOT / "templates" / "chat.html").read_text(encoding="utf-8")
        chat_core_line = [line for line in template.splitlines() if "chat_core.min." in line and "script src" in line][0]
        monitor_line = [line for line in template.splitlines() if "connection_monitor.min.js" in line and "script src" in line][0]

        self.assertIn("window.ConnectionMonitor", _connection_monitor_source())
        self.assertLess(template.index(monitor_line), template.index(chat_core_line))

        chat = _connection_script()
        self.assertNotIn("function setConnectionBanner(", chat)
        self.assertNotIn("async function probeServerConnection()", chat)
        self.assertNotIn("CONNECTION_CHECK_INTERVAL_MS", chat)
        self.assertIn("window.ConnectionMonitor.start()", chat)

    def test_success_immediately_replaces_offline_or_unstable_state(self):
        probe = _monitor_probe()
        success = probe[probe.index("const hbData = await heartbeatRes.json()") : probe.index("} catch (e)")]

        self.assertIn("const wasDisconnected = isDisconnectedConnectionStatus()", success)
        self.assertIn("connectionStatus = 'online'", success)
        self.assertIn("showConnectionRecoveredBanner()", success)
        self.assertNotIn("hadFailure", success)

    def test_failed_heartbeat_is_treated_as_disconnected(self):
        probe = _monitor_probe()
        failure = probe[probe.index("} catch (e)") :]

        self.assertIn("setUnavailable('offline')", failure)
        self.assertNotIn("navigator.onLine ? 'server-down' : 'offline'", failure)
        self.assertNotIn("setConnectionBanner('unstable'", failure)

    def test_active_operation_replaces_heartbeat_with_process_activity(self):
        module = _connection_monitor_source()
        probe = _monitor_probe()

        self.assertIn("function operationStarted()", module)
        self.assertIn("function operationEnded()", module)
        self.assertIn("function reportActivity()", module)
        self.assertIn("function isOperationActive()", module)
        self.assertIn("if (isOperationActive()) return;", probe)

        failure = probe[probe.index("} catch (e)") :]
        offline = failure.index("setUnavailable('offline')")
        guard = failure.index("if (isOperationActive()) return;")
        self.assertLess(guard, offline, "heartbeat failure must skip offline while an operation is active")

        self.assertIn("activeOperationCount = 0", module)
        self.assertIn("activeOperationCount += 1;", module)
        self.assertIn("if (activeOperationCount > 0) activeOperationCount -= 1;", module)
        self.assertIn("if (activeOperationCount === 0) {", module)
        self.assertIn("operationStarted,", module)
        self.assertIn("operationEnded,", module)
        self.assertIn("reportActivity,", module)
        self.assertIn("operationEnded", module)

    def test_chat_core_reports_uploads_and_streams_as_active_operations(self):
        chat = _connection_script()

        self.assertIn("window.ConnectionMonitor.operationStarted()", chat)
        self.assertIn("window.ConnectionMonitor.operationEnded()", chat)
        self.assertIn("window.ConnectionMonitor.reportActivity()", chat)
        self.assertIn("streamOpStarted", chat)
        self.assertIn("resumeOpStarted", chat)
        self.assertIn("browserFastOpStarted", chat)
        self.assertIn("uploadOpStarted", chat)

        # The dedicated per-operation flag must release the operation even on
        # error/cancel paths (finally / error handler).
        send = chat[chat.index("async function sendMessage()") : chat.index("async function resumePendingStream")]
        self.assertIn("streamOpStarted = true", send)
        self.assertIn("if (streamOpStarted && window.ConnectionMonitor) window.ConnectionMonitor.operationEnded();", send)
        upload = chat[chat.index("function uploadFileWithProgress") : chat.index("function isVideoFile")]
        self.assertIn("finishUploadOp()", upload)

    def test_offline_banner_uses_an_icon_available_in_fontawesome_6(self):
        source = _connection_monitor_source()
        banner = source[source.index("function setConnectionBanner(mode, message = '')") :]
        banner = banner[: banner.index("function isDisconnectedConnectionStatus")]
        offline = banner[banner.index("if (mode === 'offline')") :]

        self.assertIn("icon.className = 'fas fa-unlink'", offline)
        self.assertNotIn("fa-wifi-slash", offline)

    def test_maintenance_and_full_server_down_have_distinct_states(self):
        source = _connection_monitor_source()

        self.assertIn("if (heartbeatRes.status === 503)", source)
        self.assertIn("setUnavailable('maintenance')", source)
        self.assertIn("if ([502, 504, 520, 521, 522, 523, 524].includes(heartbeatRes.status))", source)
        self.assertIn("setUnavailable('server-down')", source)
        self.assertIn("b.classList.add('maintenance')", source)
        self.assertIn("b.classList.add('server-down')", source)
        self.assertIn("サーバーはメンテナンス中です（自動再接続します）", source)
        self.assertIn("サーバーが停止しているか応答していません（自動再接続します）", source)

    def test_slow_response_needs_repeated_samples_and_events_refresh_immediately(self):
        source = _connection_monitor_source()

        self.assertIn("const CONNECTION_SLOW_TO_UNSTABLE = 3", source)
        self.assertIn("connectionConsecutiveSlow >= CONNECTION_SLOW_TO_UNSTABLE", source)
        self.assertIn("function cancelProbe()", source)
        self.assertIn("if (probeSequence !== connectionProbeSequence) return", source)
        self.assertIn("function probeNow()", source)
        self.assertIn("connectionCheckTimer = window.setInterval(probeServerConnection", source)

    def test_prompt_retries_unavailable_responses_with_one_stable_request_id(self):
        source = _connection_script()

        self.assertIn("async function fetchChatStreamWithUnavailableRetry", source)
        self.assertIn("await window.ConnectionMonitor.waitForRetry(options.signal)", source)
        self.assertIn("client_request_id: createClientRequestId()", source)
        self.assertIn("fetchChatStreamWithUnavailableRetry(", source)
        self.assertIn("error.name === 'AbortError'", source)
        self.assertIn("e.serverCode === 'request_already_accepted'", source)
        self.assertIn("await loadMessages(currentThreadId, { preserveDraft: true, silent: true })", source)

    def test_server_deduplicates_retried_prompt_submissions(self):
        app_source = (APP_ROOT / "app.py").read_text(encoding="utf-8")
        route = app_source[app_source.index("def chat_stream():") :]
        route = route[: route.index("@app.route('/api/token_estimate'")]

        self.assertIn("_claim_chat_submission(current_user.id, client_request_id)", route)
        self.assertIn('"code": "request_already_accepted"', route)
        self.assertIn('"code": "submission_in_progress"', route)
        self.assertIn("_complete_chat_submission(", route)
        self.assertLess(route.index("_complete_chat_submission("), route.index("def generate():"))

    def test_browser_fast_result_save_uses_the_same_resilient_deduped_path(self):
        source = _connection_script()
        app_source = (APP_ROOT / "app.py").read_text(encoding="utf-8")
        client_save = source[source.index("const saveResponse = await") :]
        client_save = client_save[: client_save.index("const saved = await")]
        server_save = app_source[app_source.index("def save_browser_fast_mode_chat():") :]
        server_save = server_save[: server_save.index("@app.route('/chat_stream'")]

        self.assertIn("fetchChatStreamWithUnavailableRetry('/api/browser_fast_mode/save'", client_save)
        self.assertIn("client_request_id: createClientRequestId()", client_save)
        self.assertIn("_claim_chat_submission(current_user.id, client_request_id)", server_save)
        self.assertIn("_store_idempotent_submission(", server_save)
        self.assertIn('"code": "submission_in_progress"', server_save)

    def test_application_maintenance_response_is_machine_readable(self):
        app_source = (APP_ROOT / "app.py").read_text(encoding="utf-8")

        self.assertGreaterEqual(app_source.count("X-AI-Maintenance"), 2)
        version_route = app_source[app_source.index("def api_version():") :]
        version_route = version_route[: version_route.index("@app.route('/api/csrf_token'")]
        self.assertIn("request.args.get('heartbeat')", version_route)
        self.assertIn("resp.status_code = 503", version_route)


if __name__ == "__main__":
    unittest.main()
