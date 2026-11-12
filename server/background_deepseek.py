# --- Background: DeepSeek tool-call helpers ---

def accumulate_deepseek_tool_call_deltas(tool_call_state, delta_tool_calls):
    """Merge streamed OpenAI-compatible tool call fragments by their stable index."""
    for position, tool_delta in enumerate(delta_tool_calls or []):
        if isinstance(tool_delta, dict):
            tool_index = tool_delta.get("index")
            call_id = tool_delta.get("id")
            call_type = tool_delta.get("type")
            function_delta = tool_delta.get("function") or {}
            function_name = function_delta.get("name") if isinstance(function_delta, dict) else None
            function_args = function_delta.get("arguments") if isinstance(function_delta, dict) else None
        else:
            tool_index = getattr(tool_delta, "index", None)
            call_id = getattr(tool_delta, "id", None)
            call_type = getattr(tool_delta, "type", None)
            function_delta = getattr(tool_delta, "function", None)
            function_name = getattr(function_delta, "name", None) if function_delta else None
            function_args = getattr(function_delta, "arguments", None) if function_delta else None
        if tool_index is None:
            tool_index = position
        tool_state = tool_call_state.setdefault(
            int(tool_index),
            {"id": "", "type": "function", "name": "", "arguments": ""},
        )
        if call_id:
            tool_state["id"] = call_id
        if call_type:
            tool_state["type"] = call_type
        if function_name:
            tool_state["name"] += function_name
        if function_args:
            tool_state["arguments"] += function_args
    return tool_call_state


DEEPSEEK_SEARCH_ENDPOINT = "https://api.deepseek.com/v1/tools/search"


def _deepseek_search_request(api_key, query, count=8, timeout=30):
    """Call DeepSeek's Search API and return ``(results, error)``.

    Only a successful request with a non-empty ``results`` array is billable
    (official pricing is per successful request, not per result or token).
    """
    text = str(query or "").strip()
    if not text:
        return [], "empty query"
    payload = {
        "query": text[:128],
        "count": max(1, min(int(count or 8), 10)),
        "need_content": False,
    }
    try:
        resp = requests.post(
            DEEPSEEK_SEARCH_ENDPOINT,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return [], f"request failed: {exc}"
    if resp.status_code >= 400:
        return [], f"HTTP {resp.status_code}"
    try:
        data = resp.json()
    except ValueError:
        return [], "invalid JSON response"
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        return [], "missing results"
    return [item for item in results if isinstance(item, dict)], None


def _format_deepseek_search_results(results):
    """Render Search API results as model-readable text without extra instructions."""
    blocks = []
    for index, item in enumerate(results or [], 1):
        title = str(item.get("title") or "").strip() or "(untitled)"
        url = str(item.get("url") or "").strip()
        summary = str(item.get("summary") or item.get("snippet") or "").strip()
        published = str(item.get("date") or "").strip()
        lines = [f"[{index}] {title}"]
        if url:
            lines.append(f"URL: {url}")
        if published:
            lines.append(f"Published: {published}")
        if summary:
            lines.append(f"Summary: {summary[:1200]}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
