from pathlib import Path
import re
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


def _current_chat_js():
    assets = list((APP_ROOT / "static" / "js").glob("chat_core.v4.8.*.js"))
    assert len(assets) == 1, "Only the latest chat JS asset may remain"
    return assets[0].read_text(encoding="utf-8")


class GenerationRestartResilienceTests(unittest.TestCase):
    def test_every_server_side_generation_is_dispatched_to_rq(self):
        source = (APP_ROOT / "app.py").read_text(encoding="utf-8")
        route = source[source.index("def chat_stream():") : source.index("def estimate_prompt_tokens_api():")]

        self.assertIn("enqueue_queue.enqueue(", route)
        self.assertIn('execution_path = "queued_fast" if fast_queue_eligible else "queued_heavy"', route)
        self.assertNotIn("first_turn_direct_eligible", route)
        self.assertNotIn("direct_worker_started", route)
        self.assertNotIn("threading.Thread(", route)
        self.assertNotIn("direct-chat-", route)

    def test_worker_graceful_shutdown_exceeds_the_job_timeout(self):
        app_source = (APP_ROOT / "app.py").read_text(encoding="utf-8")
        unit_source = (APP_ROOT / "ai-chat-worker@.service").read_text(encoding="utf-8")
        route = app_source[app_source.index("def chat_stream():") : app_source.index("def estimate_prompt_tokens_api():")]
        job_timeout = int(re.search(r"job_timeout=(\d+)", route).group(1))
        stop_timeout = int(re.search(r"TimeoutStopSec=(\d+)", unit_source).group(1))

        self.assertGreater(stop_timeout, job_timeout)
        self.assertIn("SimpleWorker", (APP_ROOT / "worker.py").read_text(encoding="utf-8"))

    def test_initial_stream_disconnect_schedules_automatic_resume(self):
        source = _current_chat_js()
        send = source[source.index("async function sendMessage()") : source.index("async function resumePendingStream")]

        self.assertIn("reconnectAfterStreamDisconnect", send)
        self.assertIn("requestAccepted && !manuallyStopped", send)
        self.assertIn("reconnectPendingStreamUntilAvailable(", send)

    def test_resumed_stream_disconnect_retries_until_available(self):
        source = _current_chat_js()
        reconnect = source[source.index("async function reconnectPendingStreamUntilAvailable") :]
        reconnect = reconnect[: reconnect.index("window.initTurnstileWidget")]
        resume = source[source.index("async function resumePendingStream") : source.index("function updateThreadHighlighting")]

        self.assertIn("while (!reconnectController.signal.aborted)", reconnect)
        self.assertIn("await waitForConnectionRetry(reconnectController.signal)", reconnect)
        self.assertIn("await loadMessages(reconnectThreadId", reconnect)
        self.assertIn("resumePendingStream(latestPending)", reconnect)
        self.assertIn("reconnectAfterResumeDisconnect", resume)
        self.assertIn("reconnectPendingStreamUntilAvailable(", resume)

    def test_background_thread_reload_reports_success_without_error_toast(self):
        source = _current_chat_js()
        loader = source[source.index("async function loadMessages") :]
        loader = loader[: loader.index("async function loadOlderMessages")]

        self.assertIn("return true", loader)
        self.assertIn("if (!silent) showToast('チャットの読み込みに失敗しました'", loader)
        self.assertIn("return false", loader)


if __name__ == "__main__":
    unittest.main()
