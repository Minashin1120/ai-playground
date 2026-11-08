"""Answer processing-time helper and footer/modal wiring."""

from __future__ import annotations

import ast
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

APP_ROOT = Path(__file__).resolve().parents[1]


def _processing_helpers():
    source = (APP_ROOT / "server" / "request_identity.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted_functions = {"_trace_delta_ms", "_processing_trace_boundary", "build_message_processing_time"}
    wanted_assignments = {"_PROCESSING_TIME_PHASES"}
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted_functions:
            nodes.append(node)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted_assignments:
                    nodes.append(node)
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "request_identity.py", "exec"), namespace)
    return namespace


def _trace(**overrides):
    start = datetime(2026, 1, 1, 12, 0, 0)
    values = dict(
        client_sent_at=start,
        route_received_at=start + timedelta(milliseconds=100),
        worker_started_at=start + timedelta(milliseconds=300),
        provider_first_chunk_at=start + timedelta(milliseconds=900),
        provider_first_status_at=None,
        provider_first_thought_at=None,
        provider_first_content_at=None,
        worker_done_at=start + timedelta(milliseconds=4900),
        stream_done_at=start + timedelta(milliseconds=4950),
        client_done_at=start + timedelta(milliseconds=5000),
        client_first_latency_ms=850,
        client_total_latency_ms=5000,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_processing_time_breakdown_is_non_overlapping_and_complete():
    helpers = _processing_helpers()
    payload = helpers["build_message_processing_time"](_trace(), fallback_total_ms=9999)
    assert payload["total_ms"] == 5000
    assert payload["first_response_ms"] == 850
    assert payload["source"] == "client"
    phases = {phase["key"]: phase["ms"] for phase in payload["phases"]}
    assert phases == {"send": 100, "queue": 200, "provider": 600, "generation": 4000, "receive": 100}
    assert sum(phases.values()) == payload["total_ms"]


def test_processing_time_falls_back_to_message_timestamps_without_trace():
    helpers = _processing_helpers()
    payload = helpers["build_message_processing_time"](None, fallback_total_ms=4321)
    assert payload == {
        "total_ms": 4321,
        "first_response_ms": None,
        "phases": [],
        "source": "message",
    }


def test_processing_time_prefers_server_phases_when_client_total_is_missing():
    helpers = _processing_helpers()
    trace = _trace(client_total_latency_ms=None, client_done_at=None)
    payload = helpers["build_message_processing_time"](trace, fallback_total_ms=9999)
    phases = {phase["key"]: phase["ms"] for phase in payload["phases"]}
    assert payload["total_ms"] == 4800
    assert payload["source"] == "server"
    assert phases == {"queue": 200, "provider": 600, "generation": 4000}


def test_stale_message_timestamp_is_not_used_as_a_processing_time():
    helpers = _processing_helpers()
    assert helpers["build_message_processing_time"](None, fallback_total_ms=900001) is None


def test_processing_time_requires_a_total():
    helpers = _processing_helpers()
    assert helpers["build_message_processing_time"](None, fallback_total_ms=None) is None


def _latest_chat_core() -> Path:
    assets = sorted((APP_ROOT / "static" / "js").glob("chat_core.v*.js"))
    if not assets:
        raise AssertionError("chat_core bundle not found")
    return assets[-1]


def test_footer_and_modal_are_wired_in_the_bundle_and_template():
    source = _latest_chat_core().read_text(encoding="utf-8")
    template = (APP_ROOT / "templates" / "chat" / "scripts.html").read_text(encoding="utf-8")
    assert "function buildProcessingTimeHtml(" in source
    assert "function openProcessingTimeDetail(" in source
    assert "processing_time: processingTime" in source
    assert "openProcessingTimeDetail('${id}')" in source
    assert 'id="processing-time-modal"' in template
    assert 'id="processing-time-total"' in template
    assert 'id="processing-time-phases"' in template
