import json
import unittest

from tests.app_source import read_app_source

APP_SOURCE = read_app_source()


class _FakeRedis:
    def __init__(self, values=None, hashes=None):
        self.values = values or {}
        self.hashes = hashes or {}

    def get(self, key):
        value = self.values.get(key)
        return value.encode("utf-8") if isinstance(value, str) else value

    def hgetall(self, key):
        return {k.encode(): v.encode() for k, v in self.hashes.get(key, {}).items()}


def _load_replay(fake):
    start = APP_SOURCE.index("def _replay_finished_stream_events(job_id):")
    end = APP_SOURCE.index("@app.route('/chat_stream', methods=['POST'])", start)
    namespace = {"json": json, "redis_conn": fake}
    exec(APP_SOURCE[start:end], namespace)
    return namespace["_replay_finished_stream_events"]


def _drain(generator):
    events = []
    try:
        while True:
            events.append(json.loads(next(generator)))
    except StopIteration as stop:
        return events, stop.value


class StreamLateSubscriberRegressionTests(unittest.TestCase):
    def test_job_that_ended_before_subscribing_replays_its_error(self):
        error = json.dumps({"type": "error", "content": "添付ファイルの検証に失敗しました"})
        fake = _FakeRedis({"stream_acc:job-1:final": "error", "stream_acc:job-1:error": error})
        events, finished = _drain(_load_replay(fake)("job-1"))
        self.assertTrue(finished)
        self.assertEqual(events, [{"type": "error", "content": "添付ファイルの検証に失敗しました"}])

    def test_finished_job_replays_saved_output_then_done(self):
        fake = _FakeRedis(
            {
                "stream_acc:job-2:final": "done",
                "stream_acc:job-2:thought": "考え中",
                "stream_acc:job-2:content": "答え",
            },
            {"stream_acc:job-2:python": {"p1": json.dumps({"id": "p1", "output": "ok"})}},
        )
        events, finished = _drain(_load_replay(fake)("job-2"))
        self.assertTrue(finished)
        self.assertEqual(
            [e["type"] for e in events], ["thought", "content", "python", "done"]
        )

    def test_running_job_is_not_replayed(self):
        fake = _FakeRedis({"stream_acc:job-3:content": "途中"})
        events, finished = _drain(_load_replay(fake)("job-3"))
        self.assertFalse(finished)
        self.assertEqual(events, [])

    def test_chat_stream_checks_for_a_finished_job_before_listening(self):
        route = APP_SOURCE[APP_SOURCE.index("def chat_stream():") :]
        route = route[: route.index("@app.route('/chat_stream_resume'")]
        replay = route.index("_replay_finished_stream_events(job_id)")
        self.assertLess(route.index("pubsub.subscribe(channel)"), replay)
        self.assertLess(replay, route.index("pubsub.listen()"))


if __name__ == "__main__":
    unittest.main()
