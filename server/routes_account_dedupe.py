"""Account import deduplication routes extracted from ``routes_account.py``.

The per-user signatures for imported threads, skills, files, feedback, latency
metrics and traces, the referenced-file scan, the dedupe plan and
``/api/account/dedupe/preview`` / ``/api/account/dedupe/execute``. Loaded into
the shared ``app`` namespace by ``_SERVER_PARTS``, after
``routes_account_import.py``.
"""

# region: account import deduplication
def _import_signature(prefix, parts):
    """Deterministic per-user identity for an imported record.

    ``prefix`` is a short category tag (e.g. ``thread`` / ``skill`` / ``feedback``)
    and ``parts`` are the stable fields that define the record.  The same source
    record always yields the same signature, so re-importing it can be detected
    even though local identifiers (public_id, uuid, job_id, ...) are regenerated.
    """
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part if part is not None else "").encode("utf-8", "replace"))
        digest.update(b"\x1f")
    return f"{prefix}:{digest.hexdigest()}"


def _decrypt_dedupe_value(value, is_encrypted):
    """Return a comparable plaintext value for dedupe signatures.

    Values that cannot be decrypted (e.g. after an encryption-key change) are
    normalised to a fixed placeholder so identical unreadable records still hash
    the same way.

    This deliberately bypasses the shared decrypt cache (``decrypt_val`` keeps up
    to 4096 plaintext payloads in memory).  A full-account dedupe scan decrypts
    every message, so caching them would exhaust memory on small servers.
    """
    if value is None:
        return ""
    if not is_encrypted:
        return value
    try:
        return _ring_decrypt_bytes(value.encode()).decode()
    except Exception:
        return "\x00unreadable"


def _thread_dedupe_signature(thread):
    """Content signature of a thread, or None when the thread has no messages.

    Empty threads are never treated as duplicates: users commonly have several
    unused "New Chat" rows that only differ by their id.
    """
    digest = hashlib.sha256()
    digest.update(str(thread.title or "").encode("utf-8", "replace"))
    digest.update(b"|")
    digest.update(b"1" if thread.is_bookmarked else b"0")
    digest.update(b"|")
    message_count = 0
    message_query = (
        Message.query
        .with_entities(Message.id, Message.role, Message.content, Message.model,
                       Message.timestamp, Message.skill_uuid, Message.thought_data,
                       Message.is_encrypted)
        .filter(Message.thread_id == thread.id)
        .order_by(Message.id.asc())
        .yield_per(50)
    )
    for message in message_query:
        message_count += 1
        content = _decrypt_dedupe_value(message.content, bool(message.is_encrypted))
        thought = _decrypt_dedupe_value(message.thought_data, bool(message.is_encrypted))
        digest.update(str(message.role or "").encode("utf-8", "replace"))
        digest.update(b"\x1f")
        digest.update(str(content or "").encode("utf-8", "replace"))
        digest.update(b"\x1f")
        digest.update(str(message.model or "").encode("utf-8", "replace"))
        digest.update(b"\x1f")
        digest.update((message.timestamp.isoformat() if message.timestamp else "").encode("utf-8", "replace"))
        digest.update(b"\x1f")
        digest.update(str(message.skill_uuid or "").encode("utf-8", "replace"))
        digest.update(b"\x1f")
        digest.update(str(thought or "").encode("utf-8", "replace"))
        digest.update(b"\x1f")
    if message_count == 0:
        return None
    return digest.hexdigest()


def _skill_dedupe_signature(skill):
    parts = [
        skill.name or "", skill.description or "", skill.instruction or "",
        skill.fixed_prompts_json or "",
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8", "replace")).hexdigest()


def _file_plaintext_sha256(rel_path):
    """Compute the sha256 of a stored file's plaintext content (decrypting if needed)."""
    info = _get_file_disk_info(rel_path)
    if not info.get("exists"):
        return None
    digest = hashlib.sha256()
    try:
        if info.get("is_encrypted"):
            with open(info["disk_path"], "rb") as source:
                token = source.read()
            try:
                data = decrypt_bytes(token)
            finally:
                del token
            digest.update(data)
        else:
            with open(info["disk_path"], "rb") as source:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
    except Exception:
        return None
    return digest.hexdigest()


def _feedback_dedupe_signature(feedback):
    parts = [
        feedback.title or "", feedback.message or "", feedback.status or "",
        feedback.admin_reply or "", _portable_datetime(feedback.created_at) or "",
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8", "replace")).hexdigest()


def _metric_dedupe_signature(metric):
    parts = [
        metric.thread_public_id or "", metric.model or "", metric.first_event_type or "",
        str(int(metric.latency_ms or 0)),
        _portable_datetime(metric.client_sent_at) or "", _portable_datetime(metric.created_at) or "",
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8", "replace")).hexdigest()


def _trace_dedupe_signature(trace):
    fields = []
    for column in ChatLatencyTrace.__table__.columns:
        name = column.name
        if name in {"id", "user_id", "job_id", "import_signature", "updated_at"}:
            continue
        value = getattr(trace, name)
        if isinstance(value, datetime):
            value = _portable_datetime(value)
        fields.append(f"{name}={value}")
    return hashlib.sha256("\x1f".join(fields).encode("utf-8", "replace")).hexdigest()


def _collect_referenced_file_paths(user_id, exclude_thread_ids=None):
    """Return the set of local upload refs referenced by the user's messages.

    ``exclude_thread_ids`` (an iterable of thread ids) is used to ignore messages
    that belong to duplicate threads scheduled for deletion, so files referenced
    only by those copies become eligible for removal.
    """
    excluded = set(exclude_thread_ids or [])
    referenced = set()
    cursor = 0
    while True:
        query = db.session.query(Message.id, Message.image_url, Message.content, Message.is_encrypted).join(
            Thread, Thread.id == Message.thread_id).filter(Thread.user_id == user_id, Message.id > cursor)
        if excluded:
            query = query.filter(~Message.thread_id.in_(excluded))
        rows = query.order_by(Message.id).limit(100).all()
        if not rows:
            return referenced
        for row in rows:
            referenced.update(_message_file_refs(row, user_id))
        cursor = rows[-1].id


def _dedupe_plan_for_user(user_id):
    """Compute which duplicate records should be removed for a user.

    Returns a dict with:
      - ``removed``: list of record identifiers to delete per category
      - ``kept_referenced_files``: number of content-duplicate files that must be
        kept because they are still referenced from chat messages
    """
    removed = {"chats": [], "skills": [], "files": [], "feedback": [], "metrics": [], "traces": []}
    kept_referenced_files = 0

    thread_groups = {}
    for thread in Thread.query.filter_by(user_id=user_id).order_by(Thread.id.asc()).yield_per(200):
        signature = _thread_dedupe_signature(thread)
        if signature is not None:
            thread_groups.setdefault(signature, []).append(thread.id)
    for ids in thread_groups.values():
        if len(ids) > 1:
            removed["chats"].extend(ids[1:])

    skill_groups = {}
    for skill in Skill.query.filter_by(user_id=user_id).order_by(Skill.id.asc()).yield_per(200):
        skill_groups.setdefault(_skill_dedupe_signature(skill), []).append(skill.id)
    for ids in skill_groups.values():
        if len(ids) > 1:
            removed["skills"].extend(ids[1:])

    feedback_groups = {}
    for feedback in Feedback.query.filter_by(user_id=user_id).order_by(Feedback.id.asc()).yield_per(200):
        feedback_groups.setdefault(_feedback_dedupe_signature(feedback), []).append(feedback.id)
    for ids in feedback_groups.values():
        if len(ids) > 1:
            removed["feedback"].extend(ids[1:])

    metric_groups = {}
    for metric in FirstTokenLatencyMetric.query.filter_by(user_id=user_id).order_by(FirstTokenLatencyMetric.id.asc()).yield_per(500):
        metric_groups.setdefault(_metric_dedupe_signature(metric), []).append(metric.id)
    for ids in metric_groups.values():
        if len(ids) > 1:
            removed["metrics"].extend(ids[1:])

    trace_groups = {}
    for trace in ChatLatencyTrace.query.filter_by(user_id=user_id).order_by(ChatLatencyTrace.id.asc()).yield_per(500):
        trace_groups.setdefault(_trace_dedupe_signature(trace), []).append(trace.id)
    for ids in trace_groups.values():
        if len(ids) > 1:
            removed["traces"].extend(ids[1:])

    cache_order = {}
    for cache in FileCache.query.filter_by(user_id=user_id).order_by(FileCache.id.asc()).all():
        cache_order.setdefault(cache.rel_path, cache.id)
    file_hashes = {}
    for row in _account_file_rows(user_id):
        rel_path = row.get("rel_path")
        sha = _file_plaintext_sha256(rel_path)
        if not sha:
            continue
        file_hashes.setdefault(sha, []).append(rel_path)
    referenced = _collect_referenced_file_paths(user_id, exclude_thread_ids=removed["chats"])
    for rels in file_hashes.values():
        if len(rels) <= 1:
            continue
        ordered = sorted(rels, key=lambda rel: (cache_order.get(rel, 2 ** 62), rel))
        keep = ordered[0]
        for rel in ordered[1:]:
            if rel in referenced:
                kept_referenced_files += 1
                continue
            removed["files"].append(rel)

    return {"removed": removed, "kept_referenced_files": kept_referenced_files}


def _dedupe_count_payload(plan):
    removed = plan["removed"]
    return {
        "chats": len(removed["chats"]),
        "skills": len(removed["skills"]),
        # Legacy alias for clients that still show the old category name.
        "gems": len(removed["skills"]),
        "files": len(removed["files"]),
        "feedback": len(removed["feedback"]),
        "diagnostics": len(removed["metrics"]) + len(removed["traces"]),
        "kept_referenced_files": plan["kept_referenced_files"],
        "total": (
            len(removed["chats"]) + len(removed["skills"]) + len(removed["files"])
            + len(removed["feedback"]) + len(removed["metrics"]) + len(removed["traces"])
        ),
    }


@app.route('/api/account/dedupe/preview', methods=['POST'])
@login_required
def account_dedupe_preview():
    if not rate_limit(f"rl:account_dedupe:user:{current_user.id}", 10, 3600):
        return jsonify({'error': 'rate_limit'}), 429
    plan = _dedupe_plan_for_user(current_user.id)
    counts = _dedupe_count_payload(plan)
    response = jsonify({
        "status": "ok",
        "duplicates": {key: counts[key] for key in ("chats", "skills", "gems", "files", "feedback", "diagnostics")},
        "kept_referenced_files": counts["kept_referenced_files"],
        "total": counts["total"],
        "has_duplicates": counts["total"] > 0,
    })
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route('/api/account/dedupe/execute', methods=['POST'])
@login_required
def account_dedupe_execute():
    _lock_chat_files(current_user.id)
    if not rate_limit(f"rl:account_dedupe:user:{current_user.id}", 10, 3600):
        return jsonify({'error': 'rate_limit'}), 429
    plan = _dedupe_plan_for_user(current_user.id)
    removed = plan["removed"]
    try:
        if removed["chats"]:
            for thread_id in removed["chats"]:
                thread = db.session.get(Thread, thread_id)
                if thread:
                    db.session.delete(thread)
        if removed["skills"]:
            skill_groups = {}
            for skill in Skill.query.filter_by(user_id=current_user.id).order_by(Skill.id.asc()).all():
                skill_groups.setdefault(_skill_dedupe_signature(skill), []).append(skill)
            skill_twin = {}
            for group in skill_groups.values():
                if len(group) > 1:
                    keep = group[0]
                    for dup in group[1:]:
                        skill_twin[dup.id] = keep.uuid
            thread_ids = [tid for (tid,) in db.session.query(Thread.id).filter_by(user_id=current_user.id).all()]
            for skill_id in removed["skills"]:
                skill = db.session.get(Skill, skill_id)
                if not skill:
                    continue
                # Keep references (thread last_skill_uuid / message skill_uuid)
                # pointing at a live skill when the removed duplicate had the
                # same content.
                keep_uuid = skill_twin.get(skill.id)
                if keep_uuid and keep_uuid != skill.uuid:
                    for start in range(0, len(thread_ids), 200):
                        chunk = thread_ids[start:start + 200]
                        Message.query.filter(Message.thread_id.in_(chunk), Message.skill_uuid == skill.uuid).update(
                            {"skill_uuid": keep_uuid}, synchronize_session=False
                        )
                    Thread.query.filter_by(user_id=current_user.id, last_skill_uuid=skill.uuid).update(
                        {"last_skill_uuid": keep_uuid}, synchronize_session=False
                    )
                db.session.delete(skill)
        if removed["files"]:
            for rel_path in removed["files"]:
                info = _get_file_disk_info(rel_path)
                if info.get("exists"):
                    secure_delete(info["disk_path"])
                _delete_file_cache_for_path(current_user.id, rel_path)
        if removed["feedback"]:
            for feedback_id in removed["feedback"]:
                feedback = db.session.get(Feedback, feedback_id)
                if feedback:
                    FeedbackReply.query.filter_by(feedback_id=feedback.id).delete(synchronize_session=False)
                    db.session.delete(feedback)
        if removed["metrics"]:
            for start in range(0, len(removed["metrics"]), 500):
                chunk = removed["metrics"][start:start + 500]
                FirstTokenLatencyMetric.query.filter(
                    FirstTokenLatencyMetric.id.in_(chunk)
                ).delete(synchronize_session=False)
        if removed["traces"]:
            for start in range(0, len(removed["traces"]), 500):
                chunk = removed["traces"][start:start + 500]
                ChatLatencyTrace.query.filter(
                    ChatLatencyTrace.id.in_(chunk)
                ).delete(synchronize_session=False)
        db.session.commit()
    except Exception:
        db.session.rollback()
        logger.exception("Account dedupe failed for user %s", current_user.id)
        return jsonify({'error': 'dedupe_failed'}), 500
    counts = _dedupe_count_payload(plan)
    response = jsonify({
        "status": "ok",
        "removed": {key: counts[key] for key in ("chats", "skills", "gems", "files", "feedback", "diagnostics")},
        "kept_referenced_files": counts["kept_referenced_files"],
        "total": counts["total"],
    })
    response.headers["Cache-Control"] = "no-store"
    return response

