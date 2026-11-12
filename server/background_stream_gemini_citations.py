"""Grounding, citation, and thought helpers for the Gemini chat branch.

Extracted from `background_stream_gemini_chat.py`.  These functions are pure:
they only read the objects they are given, so they no longer need to be nested
inside `background_stream_gemini_chat` and its `locals()` snapshot.
"""

def _chunk_grounding_info(chunk):
    if not chunk:
        return None, None
    candidates = [chunk]
    if isinstance(chunk, dict):
        candidates.extend([chunk.get('web'), chunk.get('maps')])
    else:
        candidates.extend([getattr(chunk, 'web', None), getattr(chunk, 'maps', None)])
    for candidate in candidates:
        if not candidate:
            continue
        if isinstance(candidate, dict):
            title = candidate.get('title') or candidate.get('name') or candidate.get('text')
            uri = candidate.get('uri') or candidate.get('url')
        else:
            title = getattr(candidate, 'title', None) or getattr(candidate, 'name', None) or getattr(candidate, 'text', None)
            uri = getattr(candidate, 'uri', None) or getattr(candidate, 'url', None)
        if title or uri:
            return title, uri
    if isinstance(chunk, dict):
        title = chunk.get('title') or chunk.get('name') or chunk.get('text')
        uri = chunk.get('uri') or chunk.get('url')
        if not title:
            place_id = chunk.get('place_id') or chunk.get('placeId')
            if place_id:
                title = place_id
    else:
        title = getattr(chunk, 'title', None) or getattr(chunk, 'name', None) or getattr(chunk, 'text', None)
        uri = getattr(chunk, 'uri', None) or getattr(chunk, 'url', None)
        if not title:
            place_id = getattr(chunk, 'place_id', None) or getattr(chunk, 'placeId', None)
            if place_id:
                title = place_id
    return title, uri

def _segment_end_index(segment):
    if segment is None:
        return None
    end_index = getattr(segment, 'end_index', None)
    if end_index is None:
        end_index = getattr(segment, 'endIndex', None)
    return end_index

def _segment_text(segment):
    if segment is None:
        return None
    seg_text = getattr(segment, 'text', None)
    return seg_text if isinstance(seg_text, str) and seg_text else None

def _gemini_citation_position(text, model_text, segment):
    """本文中で出典を差し込む文字位置を返す（見つからなければ None）。

    segment.endIndex は UTF-8 のバイト位置で、モデルが返した本文
    （テキストのパートだけを連結したもの）に対する値。保存する本文には
    コードブロックなどが挟まるため、segment.text（無ければ直前の文脈）
    を本文から探し、推定位置に最も近い出現の末尾を使う。
    """
    model = model_text if model_text is not None else text
    end_index = _segment_end_index(segment)
    estimate = None
    try:
        end_byte = int(end_index) if end_index is not None else None
    except Exception:
        end_byte = None
    model_bytes = model.encode('utf-8')
    if end_byte is not None and 0 < end_byte <= len(model_bytes):
        estimate = len(model_bytes[:end_byte].decode('utf-8', 'ignore')) or None
    anchors = []
    seg_text = _segment_text(segment)
    if seg_text:
        anchors.append(seg_text)
    if estimate:
        context = model[max(0, estimate - 40):estimate]
        if context.strip():
            anchors.append(context)
    for anchor in anchors:
        best = None
        start = text.find(anchor)
        while start != -1:
            end = start + len(anchor)
            if estimate is None:
                best = end
                break
            if best is None or abs(end - estimate) < abs(best - estimate):
                best = end
            start = text.find(anchor, start + 1)
        if best is not None:
            return best
    return None

def _add_gemini_citations(text, supports, chunks, model_text=None):
    if not text or not supports or not chunks:
        return text
    insertions = []
    for order, support in enumerate(supports):
        segment = getattr(support, 'segment', None)
        idxs = getattr(support, 'grounding_chunk_indices', None) or getattr(support, 'groundingChunkIndices', None) or []
        if not idxs:
            continue
        citation_links = []
        for i in idxs:
            try:
                idx = int(i)
            except Exception:
                continue
            if idx < 0 or idx >= len(chunks):
                continue
            _, uri = _chunk_grounding_info(chunks[idx])
            if uri:
                citation_links.append(f"[{idx + 1}]({uri})")
        if not citation_links:
            continue
        position = _gemini_citation_position(text, model_text, segment)
        if position is None:
            continue
        insertions.append((position, order, "".join(citation_links)))
    # 後ろから差し込む。同じ位置では元の順に並ぶよう、後のものから入れる。
    for position, _order, links in sorted(insertions, reverse=True):
        text = text[:position] + links + text[position:]
    return text

def _url_retrieval_status_label(status):
    """URL Context の取得状態を表示用の文字列にする（成功・不明なら空）。"""
    if status is None:
        return ""
    value = getattr(status, 'value', None)
    if not isinstance(value, str) or not value:
        value = getattr(status, 'name', None)
    if not isinstance(value, str) or not value:
        value = str(status)
    value = value.strip()
    if value.startswith('UrlRetrievalStatus.'):
        value = value[len('UrlRetrievalStatus.'):]
    if value in ('', 'ACTIVE', 'URL_RETRIEVAL_STATUS_SUCCESS', 'URL_RETRIEVAL_STATUS_UNSPECIFIED'):
        return ""
    return value

def _extract_gemini_thought_text(part):
    if not part:
        return ""
    thought_val = getattr(part, 'thought', None)
    text_val = getattr(part, 'text', None)
    if isinstance(thought_val, str):
        return thought_val
    if isinstance(thought_val, dict):
        t_val = thought_val.get("text") or thought_val.get("content") or thought_val.get("value")
        if t_val is not None:
            return str(t_val)
    if thought_val is not None and not isinstance(thought_val, bool):
        for key in ("text", "content", "value"):
            t_val = getattr(thought_val, key, None)
            if t_val:
                return str(t_val)
    if text_val:
        return str(text_val)
    return ""
