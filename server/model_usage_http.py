"""Observe decoded HTTPX responses without buffering a stream or changing delivery/cancellation."""

class _ModelUsageBody:
    LIMIT = 1024 * 1024

    def __init__(self, call, sse):
        self.call = call
        self.sse = sse
        self.buffer = bytearray()
        self.discard = False
        self.closed = False

    def _line(self, line):
        if not line.startswith(b'data:'):
            return
        try:
            self.call.observe(json.loads(line[5:].strip()))
        except (ValueError, UnicodeError):
            pass

    def feed(self, chunk):
        if self.closed:
            return
        if not self.sse:
            if len(self.buffer) + len(chunk) <= self.LIMIT and not self.discard:
                self.buffer.extend(chunk)
            else:
                self.buffer.clear()
                self.discard = True
            return
        start = 0
        while start < len(chunk):
            end = chunk.find(b'\n', start)
            part = chunk[start:] if end < 0 else chunk[start:end]
            if len(self.buffer) + len(part) > self.LIMIT:
                self.discard = True
                self.buffer.clear()
            elif not self.discard:
                self.buffer.extend(part)
            if end < 0:
                break
            if not self.discard:
                self._line(bytes(self.buffer).rstrip(b'\r'))
            self.buffer.clear()
            self.discard = False
            start = end + 1

    def finish(self, complete):
        if self.closed:
            return
        self.closed = True
        if not self.discard:
            if self.sse:
                self._line(bytes(self.buffer))
            else:
                try:
                    self.call.observe(json.loads(self.buffer))
                except (ValueError, UnicodeError):
                    pass
        self.buffer.clear()
        self.call.finished = complete


def _observe_usage_request(request):
    ledger = _MODEL_USAGE_CONTEXT.get()
    if ledger is None:
        return
    if request.method == 'GET' and re.search(r'/responses/[^/]+$', request.url.path):
        call = ledger.find_response(request.url.path.rsplit('/', 1)[-1])
        if call is not None:
            request.extensions['model_usage_call'] = call
        return
    if request.method != 'POST':
        return
    model = _usage_model_from_request(request)
    if not model:
        return
    host = request.url.host
    provider = 'anthropic' if host.endswith('anthropic.com') else 'xai' if host.endswith('x.ai') else 'google' if '/models/' in request.url.path else 'openai'
    request.extensions['model_usage_call'] = ledger.begin(model, provider)


def _observe_usage_response(response):
    call = response.request.extensions.pop('model_usage_call', None)
    if call is None:
        return
    if 400 <= response.status_code < 500:
        call.rejected = True  # Explicitly rejected requests are not assumed to be charged.
        return
    observer = _ModelUsageBody(call, 'text/event-stream' in response.headers.get('content-type', ''))
    if response.is_stream_consumed:
        observer.feed(response.content)
        observer.finish(response.is_success)
        return
    original_iter = response.iter_bytes

    def iter_bytes(chunk_size=None):
        completed = False
        try:
            for chunk in original_iter(chunk_size):
                observer.feed(chunk)
                yield chunk
            completed = response.is_success
        finally:
            observer.finish(completed)

    response.iter_bytes = iter_bytes


for _usage_http_client in (_OPENAI_HTTPX_CLIENT, _GEMINI_HTTPX_CLIENT):
    _usage_http_client.event_hooks['request'].append(_observe_usage_request)
    _usage_http_client.event_hooks['response'].append(_observe_usage_response)
