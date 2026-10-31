"""Exercise stream lifetime without real accounts, Redis, or wall-clock waits."""
import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


class Clock:
    now = 0.0

    def monotonic(self):
        return self.now


class PubSub:
    def __init__(self, clock, messages=()):
        self.clock = clock
        self.messages = iter(messages)
        self.closed = False
        self.waits = []

    def subscribe(self, channel):
        self.channel = channel

    def get_message(self, timeout):
        self.waits.append(timeout)
        self.clock.now += timeout
        return next(self.messages, None)

    def close(self):
        self.closed = True


def transport(messages=()):
    clock = Clock()
    pubsub = PubSub(clock, messages)
    events = []
    namespace = dict(
        contextmanager=contextmanager, time=clock,
        redis_conn=SimpleNamespace(pubsub=lambda: pubsub),
        current_user=SimpleNamespace(id=1), g=SimpleNamespace(diagnostic_op="a" * 32),
        _record_server_diagnostic=lambda uid, event, **fields: events.append((event, fields)),
    )
    source = Path(__file__).resolve().parents[1] / "server/routes_chat.py"
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {"_chat_stream_subscription", "_chat_stream_messages"}]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace, clock, pubsub, events


def test_silent_stream_emits_blank_heartbeats_and_reaches_deadline():
    ns, clock, pubsub, events = transport()
    with ns["_chat_stream_subscription"]("b" * 32) as (subscription, started):
        chunks = list(ns["_chat_stream_messages"](subscription, started, timeout=46, heartbeat=15))
    assert chunks == [None, None, None]
    assert clock.now == 46
    assert max(pubsub.waits) <= 1
    assert pubsub.closed
    assert any(event == "server.stream.error" and fields["reason_code"] == 123 for event, fields in events)


def test_events_are_forwarded_without_subscription_notifications():
    event = {"type": "message", "data": b'{"type":"done"}'}
    ns, clock, pubsub, events = transport([{"type": "subscribe"}, event])
    with ns["_chat_stream_subscription"]("b" * 32) as (subscription, started):
        messages = ns["_chat_stream_messages"](subscription, started)
        assert next(messages) is event
        messages.close()
    assert pubsub.closed
    assert clock.now == 2


def test_client_disconnect_during_initial_replay_closes_subscription():
    ns, _, pubsub, events = transport()

    def response():
        with ns["_chat_stream_subscription"]("b" * 32):
            yield '{"type":"job_id"}\n'
            pytest.fail("client closed before Redis polling")

    stream = response()
    next(stream)
    stream.close()
    assert pubsub.closed
    assert events[-1][1]["phase"] == "cancelled"
    assert events[-1][1]["op"] == "a" * 32


def test_subscription_failure_still_closes_and_records_failure():
    ns, _, pubsub, events = transport()

    def fail(channel):
        raise OSError("fake secret must not be logged")

    pubsub.subscribe = fail
    with pytest.raises(OSError):
        with ns["_chat_stream_subscription"]("b" * 32):
            pytest.fail("subscribe failed")
    assert pubsub.closed
    assert events[-1][1]["phase"] == "error"
    assert "fake secret" not in str(events)
