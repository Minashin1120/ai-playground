"""Content-free logging boundary. Shared schema copies are checked by regression tests."""
import json
import logging
import math
import re
from pathlib import Path

SCHEMA = json.loads(Path(__file__).with_name('logging_schema.json').read_text())
_EVENTS = frozenset(SCHEMA['events'])
_NUMBERS = frozenset(SCHEMA['numbers'])
_BOOLEANS = frozenset(SCHEMA['booleans'])
_ENUMS = {key: frozenset(values) for key, values in SCHEMA['enums'].items()}
_ROUTES = frozenset(SCHEMA['routes'])


def safe_version(value):
    return value if isinstance(value, str) and re.fullmatch(SCHEMA['version_pattern'], value) else None


def safe_entry(raw):
    if not isinstance(raw, dict) or not isinstance(raw.get('ev'), str) or raw['ev'] not in _EVENTS:
        return None
    out = {'ev': raw['ev']}
    for key, value in raw.items():
        if key in _NUMBERS and type(value) in (int, float) and abs(value) <= 2**53 and math.isfinite(value):
            out[key] = value
        elif key in _BOOLEANS and type(value) is bool:
            out[key] = value
        elif key == 'windows' and isinstance(value, list):
            windows = []
            for frame in value[:24]:
                if isinstance(frame, dict) and isinstance(frame.get('window'), str) and frame['window'] in _ENUMS['window'] and all(type(frame.get(k)) is int and abs(frame[k]) < 2**31 for k in ('x', 'y', 'w', 'h')):
                    windows.append({k: frame[k] for k in ('window', 'x', 'y', 'w', 'h')})
            if windows:
                out[key] = windows
        elif key in ('stack', 'frames') and isinstance(value, list):
            frames = []
            for frame in value[:24]:
                if isinstance(frame, dict) and isinstance(frame.get('code_file'), str) and frame['code_file'] in _ENUMS.get('code_file', ()) and type(frame.get('line')) is int and abs(frame['line']) < 2**31:
                    frames.append({'code_file': frame['code_file'], 'line': frame['line']})
            if frames:
                out[key] = frames
        elif key == 'version' and safe_version(value) is not None:
            out[key] = value
        elif key in _ENUMS and isinstance(value, str) and value in _ENUMS[key]:
            out[key] = value
        elif key in ('route', 'path') and isinstance(value, str) and value in _ROUTES:
            out[key] = value
        elif key in SCHEMA['identifiers'] and isinstance(value, str) and re.fullmatch(SCHEMA['identifiers'][key], value):
            out[key] = value
    return out


class ContentFreeFilter(logging.Filter):
    """Legacy logs keep location/severity, never formatted values or exception text.

    Apply to output handlers (including SDK logs), not only the application's logger.
    No call to getMessage(): even object repr/exception formatting is avoided.
    """
    def filter(self, record):
        frames = []
        tb = record.exc_info[2] if record.exc_info else None
        while tb and len(frames) < 24:
            code = tb.tb_frame.f_code
            name = Path(code.co_filename).name
            frames.append({'file': name if name in _ENUMS['code_file'] else 'external', 'line': tb.tb_lineno})
            tb = tb.tb_next
        name = Path(record.pathname).name
        payload = {'ev': 'operations.log', 'file': name if name in _ENUMS['code_file'] else 'external', 'line': record.lineno,
                   'level': record.levelno}
        if frames:
            payload['frames'] = frames
        record.msg, record.args = json.dumps(payload, separators=(',', ':')), ()
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        return True


def install_safe_logging():
    # A factory also covers handlers installed later by gunicorn, RQ and SDKs.
    previous = logging.getLogRecordFactory()
    if getattr(previous, '_content_free', False):
        return
    boundary = ContentFreeFilter()
    def factory(*args, **kwargs):
        record = previous(*args, **kwargs)
        boundary.filter(record)
        return record
    factory._content_free = True
    logging.setLogRecordFactory(factory)
