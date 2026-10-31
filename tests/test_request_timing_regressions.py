"""Measure early application hooks without loading accounts or the production app."""
import ast
from pathlib import Path
from types import SimpleNamespace


def hooks(enabled=True):
    root = Path(__file__).resolve().parents[1] / "server"
    clock = SimpleNamespace(now=10.0)
    events = []
    state = SimpleNamespace()
    namespace = dict(
        app=SimpleNamespace(before_request=lambda f: f, after_request=lambda f: f),
        g=state, time=SimpleNamespace(monotonic=lambda: clock.now),
        request=SimpleNamespace(endpoint="version", headers={}, path="/api/version", method="GET",
                                url_rule=SimpleNamespace(rule="/api/version")),
        current_user=SimpleNamespace(id=1, is_authenticated=True),
        secrets=SimpleNamespace(token_hex=lambda _: "a" * 32),
        _server_diagnostic_enabled=lambda uid: enabled,
        _record_server_diagnostic=lambda uid, event, **fields: events.append((event, fields)),
    )
    for filename, names in [
        ("mobile_auth.py", {"mobile_request_guard"}),
        ("diagnostic_logging.py", {"_diagnostic_request_start", "_diagnostic_request_end"}),
    ]:
        nodes = [n for n in ast.parse((root / filename).read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)
    return namespace, clock, state, events


def test_pre_authentication_delay_is_included_in_request_duration():
    ns, clock, state, events = hooks()
    ns["mobile_request_guard"]()
    # Model authentication and other early hooks taking two seconds.
    clock.now = 12
    ns["_diagnostic_request_start"]()
    clock.now = 12.01
    response = SimpleNamespace(status_code=200, headers={})
    assert ns["_diagnostic_request_end"](response) is response
    assert events[0][1]["ms"] == 2000
    assert events[1][1]["ms"] == 2010
    assert response.headers["X-Diagnostic-ID"] == events[0][1]["op"]


def test_early_clock_does_not_bypass_disabled_logging():
    ns, clock, state, events = hooks(enabled=False)
    ns["mobile_request_guard"]()
    ns["_diagnostic_request_start"]()
    response = SimpleNamespace(status_code=200, headers={})
    ns["_diagnostic_request_end"](response)
    assert events == []
    assert response.headers == {}
