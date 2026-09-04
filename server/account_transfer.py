
# Account portability deliberately excludes identity, authentication, active sessions,
# privileges and moderation state.  In particular, username is never written to an
# export archive and can never be changed by an import.
ACCOUNT_EXPORT_FORMAT = "ai-playground-account-export"
ACCOUNT_EXPORT_VERSION = 1
ACCOUNT_IMPORT_CATEGORIES = frozenset({
    "settings", "api_credentials", "chats", "gems", "files", "feedback", "diagnostics"
})
ACCOUNT_SETTING_FIELDS = (
    "system_prompt", "system_prompt_enabled", "apply_global_system_prompt",
    "apply_auto_system_prompt_notices", "auto_system_prompt_notices_config",
    "gemini_backend", "gemini_vertex_location", "mic_transcribe_mode", "stt_model",
    "llm_transcribe_prompt", "enter_to_send", "use_sw_cache", "clear_cache_on_version_update",
    "theme_color", "liquid_glass_enabled", "auto_search_on_links", "compact_prompt_mode",
    "minimal_prompt_mode",
    "use_last_chat_settings", "voice_studio_ui", "temp_chat_timeout_seconds", "default_model",
    "default_enable_search", "default_enable_url_context", "default_enable_maps",
    "default_enable_python", "default_enable_file_creation", "default_enable_thinking", "default_thinking_level",
    "default_thinking_budget", "default_reasoning_effort", "default_enable_system_prompt",
    "default_enable_mcp",
    "default_safety_setting", "default_vision_model", "rich_paste_prompt_default",
    "rich_paste_prompt_use_custom_default", "last_model", "last_enable_search",
    "last_enable_url_context", "last_enable_maps", "last_enable_python",
    "last_enable_file_creation", "last_enable_thinking", "last_thinking_level", "last_thinking_budget",
    "last_reasoning_effort", "last_enable_system_prompt", "last_enable_mcp", "last_safety_setting",
    "enable_latency_metrics", "enable_client_debug_log",
)
ACCOUNT_SECRET_FIELDS = (
    "openai_api_key", "gemini_api_key", "anthropic_api_key", "deepseek_api_key",
    "kimi_api_key", "mistral_api_key", "model_api_keys", "gemini_vertex_project",
    "gemini_vertex_credentials_json", "xai_api_key", "google_api_key", "google_cloud_project",
)
ACCOUNT_BOOL_SETTING_FIELDS = frozenset({
    "system_prompt_enabled", "apply_global_system_prompt", "apply_auto_system_prompt_notices",
    "enter_to_send", "use_sw_cache", "clear_cache_on_version_update", "liquid_glass_enabled",
    "auto_search_on_links", "compact_prompt_mode", "minimal_prompt_mode", "use_last_chat_settings",
    "voice_studio_ui",
    "default_enable_search", "default_enable_url_context", "default_enable_maps",
    "default_enable_python", "default_enable_file_creation", "default_enable_thinking", "default_enable_system_prompt",
    "default_enable_mcp",
    "rich_paste_prompt_use_custom_default", "last_enable_search", "last_enable_url_context",
    "last_enable_maps", "last_enable_python", "last_enable_file_creation", "last_enable_thinking",
    "last_enable_system_prompt", "last_enable_mcp", "enable_latency_metrics", "enable_client_debug_log",
})
ACCOUNT_INT_SETTING_FIELDS = frozenset({
    "temp_chat_timeout_seconds", "default_thinking_budget", "last_thinking_budget",
})
ACCOUNT_TEXT_SETTING_LIMITS = {
    "system_prompt": 500_000,
    "auto_system_prompt_notices_config": 500_000,
    "llm_transcribe_prompt": 100_000,
    "rich_paste_prompt_default": 100_000,
}
ACCOUNT_TRANSFER_JOB_RE = re.compile(r"^[a-f0-9]{32}$")
ACCOUNT_EXPORT_RETENTION_SECONDS = 3600
ACCOUNT_TRANSFER_STATUS_TTL = ACCOUNT_EXPORT_RETENTION_SECONDS + 600
# RQ kills jobs that outlive job_timeout.  A large account (thousands of chat
# messages plus hundreds of MB / GB of uploaded files) needs to be read,
# decrypted and compressed in the background, so keep a generous budget well
# beyond the typical duration to avoid killing valid exports.
ACCOUNT_EXPORT_JOB_TIMEOUT_SECONDS = 7200
ACCOUNT_EXPORT_FILE_RE = re.compile(r"^(\d+)-([a-f0-9]{32})\.(zip|part)$")


class AccountTransferCancelled(Exception):
    pass


class AccountExportFileUnreadable(Exception):
    pass


def _account_transfer_status_key(user_id, job_id):
    return f"account_transfer:status:{int(user_id)}:{job_id}"


def _account_transfer_cancel_key(user_id, job_id):
    return f"account_transfer:cancel:{int(user_id)}:{job_id}"


def _account_export_artifact_key(user_id, job_id):
    return f"account_export:artifact:{int(user_id)}:{job_id}"


def _account_export_latest_key(user_id):
    return f"account_export:latest:{int(user_id)}"


def _account_export_active_key(user_id):
    return f"account_export:active:{int(user_id)}"


def _account_export_dir():
    path = os.path.join(app.instance_path, "account_exports")
    os.makedirs(path, mode=0o700, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path


def _account_export_path(user_id, job_id, suffix="zip"):
    if not _valid_account_transfer_job_id(job_id) or suffix not in {"zip", "part"}:
        return None
    return os.path.join(_account_export_dir(), f"{int(user_id)}-{job_id}.{suffix}")


def _decode_redis_json(raw):
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    except Exception:
        return None


def _valid_account_transfer_job_id(job_id):
    return bool(ACCOUNT_TRANSFER_JOB_RE.fullmatch(str(job_id or "")))


def _set_account_transfer_status(user_id, job_id, state, progress, phase, message="", **details):
    if not _valid_account_transfer_job_id(job_id):
        return
    payload = {
        "state": str(state),
        "progress": max(0, min(100, int(progress or 0))),
        "phase": str(phase or ""),
        "message": str(message or ""),
        "updated_at": _portable_datetime(datetime.utcnow()),
    }
    for key in (
        "expires_at", "filename", "size_bytes", "available",
        "files", "used_bytes", "limit_bytes", "available_bytes",
        "settings_changes",
    ):
        if key in details:
            payload[key] = details[key]
    try:
        redis_conn.setex(
            _account_transfer_status_key(user_id, job_id),
            ACCOUNT_TRANSFER_STATUS_TTL,
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        )
    except Exception:
        pass


def _account_transfer_cancelled(user_id, job_id):
    if not _valid_account_transfer_job_id(job_id):
        return False
    try:
        return bool(redis_conn.exists(_account_transfer_cancel_key(user_id, job_id)))
    except Exception:
        return False


def _account_transfer_checkpoint(user_id, job_id, progress, phase, message=""):
    if _account_transfer_cancelled(user_id, job_id):
        raise AccountTransferCancelled()
    _set_account_transfer_status(user_id, job_id, "running", progress, phase, message)


def _account_export_artifact(user_id, job_id, destroy_on_missing=True):
    """Return metadata for a downloadable account-export archive.

    ``destroy_on_missing`` controls whether a missing/expired archive also
    clears the Redis record and the on-disk file.  Status and download
    endpoints pass ``False`` so a single failed check never erases an archive
    the user might still be able to retry, and so the UI can show an accurate
    terminal state instead of silently discarding it.  Scheduled cleanup
    (``delete_account_export_task`` / ``_cleanup_expired_account_export_files``)
    is the only path that should actually remove archives.

    When the archive file is still on disk but the Redis metadata lapsed
    (worker/scheduler delay, eviction, TTL expiry), the record is rebuilt so
    the download keeps working as long as the file is within its retention
    window.
    """
    if not _valid_account_transfer_job_id(job_id):
        return None
    try:
        metadata = _decode_redis_json(redis_conn.get(_account_export_artifact_key(user_id, job_id)))
    except Exception:
        metadata = None
    if not isinstance(metadata, dict):
        metadata = {}
    path = _account_export_path(user_id, job_id)
    if not path:
        if destroy_on_missing:
            _delete_account_export_artifact(user_id, job_id, state="expired")
        return None
    now = int(time.time())
    try:
        file_mtime = int(os.path.getmtime(path))
    except OSError:
        file_mtime = 0
    if not os.path.isfile(path):
        if destroy_on_missing:
            _delete_account_export_artifact(user_id, job_id, state="expired")
        return None
    if file_mtime <= 0 or now - file_mtime >= ACCOUNT_EXPORT_RETENTION_SECONDS:
        # The archive is beyond its one-hour retention window.  Removal is owned
        # by scheduled cleanup, but it must no longer be served.
        if destroy_on_missing:
            _delete_account_export_artifact(user_id, job_id, state="expired")
        return None
    if not metadata or not metadata.get("expires_ts"):
        # Redis metadata lapsed / was evicted while the archive is still on
        # disk and inside the retention window.  Rebuild the record so status
        # and download keep working instead of failing with a 404.
        ready_ts = file_mtime
        metadata = {
            "job_id": job_id,
            "filename": f"ai-playground-account-{datetime.utcfromtimestamp(ready_ts).strftime('%Y%m%d-%H%M%S')}.zip",
            "size_bytes": os.path.getsize(path),
            "ready_ts": ready_ts,
            "expires_ts": ready_ts + ACCOUNT_EXPORT_RETENTION_SECONDS,
            "expires_at": _portable_datetime(datetime.utcfromtimestamp(ready_ts + ACCOUNT_EXPORT_RETENTION_SECONDS)),
            "unreadable_count": 0,
        }
        try:
            redis_conn.setex(
                _account_export_artifact_key(user_id, job_id),
                ACCOUNT_TRANSFER_STATUS_TTL,
                json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
            )
        except Exception:
            pass
    metadata["path"] = path
    return metadata


def _account_transfer_status_payload(user_id, job_id):
    try:
        payload = _decode_redis_json(redis_conn.get(_account_transfer_status_key(user_id, job_id))) or {
            "state": "pending", "progress": 0, "phase": "pending", "message": ""
        }
    except Exception:
        payload = {"state": "pending", "progress": 0, "phase": "pending", "message": ""}
    payload["job_id"] = job_id
    artifact = _account_export_artifact(user_id, job_id, destroy_on_missing=False)
    if artifact:
        payload.update({
            "state": "ready",
            "progress": 100,
            "phase": "ready",
            "available": True,
            "filename": artifact.get("filename"),
            "size_bytes": artifact.get("size_bytes"),
            "expires_at": artifact.get("expires_at"),
            "download_url": f"/api/account/export/{job_id}/download",
            "unreadable_count": int(artifact.get("unreadable_count") or 0),
        })
    else:
        payload["available"] = False
        # A job that previously reached "ready" but whose archive is no longer
        # present must surface a terminal state instead of a stale "ready"
        # without a file.  The artifact record itself is left untouched so the
        # scheduled cleanup owns the actual deletion.
        if payload.get("state") == "ready":
            payload.update({
                "state": "expired",
                "progress": 0,
                "phase": "expired",
                "message": "エクスポートZIPの保存期限が切れたか、利用できません",
            })
    return payload


def _delete_account_export_artifact(user_id, job_id, state=None):
    if not _valid_account_transfer_job_id(job_id):
        return
    for suffix in ("zip", "part"):
        path = _account_export_path(user_id, job_id, suffix)
        if path and os.path.lexists(path):
            secure_delete(path)
    try:
        redis_conn.delete(_account_export_artifact_key(user_id, job_id))
        latest_raw = redis_conn.get(_account_export_latest_key(user_id))
        latest = latest_raw.decode("utf-8") if isinstance(latest_raw, bytes) else str(latest_raw or "")
        if latest == job_id:
            if state:
                redis_conn.expire(_account_export_latest_key(user_id), 600)
            else:
                redis_conn.delete(_account_export_latest_key(user_id))
        if state:
            message = "エクスポートZIPの保存期限が切れました" if state == "expired" else "エクスポートをキャンセルしました"
            _set_account_transfer_status(user_id, job_id, state, 0, state, message, available=False)
    except Exception:
        pass


def _cleanup_expired_account_export_files(now_ts=None):
    now_ts = int(now_ts or time.time())
    export_dir = _account_export_dir()
    try:
        entries = list(os.scandir(export_dir))
    except OSError:
        return
    for entry in entries:
        match = ACCOUNT_EXPORT_FILE_RE.fullmatch(entry.name)
        if not match or not entry.is_file(follow_symlinks=False):
            continue
        max_age = ACCOUNT_EXPORT_RETENTION_SECONDS if match.group(3) == "zip" else 2 * ACCOUNT_EXPORT_RETENTION_SECONDS
        try:
            expired = now_ts - int(entry.stat(follow_symlinks=False).st_mtime) >= max_age
        except OSError:
            continue
        if expired:
            secure_delete(entry.path)


def _delete_all_account_export_artifacts(user_id):
    export_dir = _account_export_dir()
    prefix = f"{int(user_id)}-"
    try:
        for entry in os.scandir(export_dir):
            if entry.name.startswith(prefix) and ACCOUNT_EXPORT_FILE_RE.fullmatch(entry.name) and entry.is_file(follow_symlinks=False):
                secure_delete(entry.path)
    except OSError:
        pass
    try:
        latest_raw = redis_conn.get(_account_export_latest_key(user_id))
        latest = latest_raw.decode("utf-8") if isinstance(latest_raw, bytes) else str(latest_raw or "")
        keys = [_account_export_latest_key(user_id), _account_export_active_key(user_id)]
        if _valid_account_transfer_job_id(latest):
            keys.extend([
                _account_export_artifact_key(user_id, latest),
                _account_transfer_status_key(user_id, latest),
                _account_transfer_cancel_key(user_id, latest),
            ])
        redis_conn.delete(*keys)
    except Exception:
        pass


def delete_account_export_task(user_id, job_id):
    with app.app_context():
        metadata = None
        try:
            metadata = _decode_redis_json(redis_conn.get(_account_export_artifact_key(user_id, job_id)))
        except Exception:
            pass
        expires_ts = int((metadata or {}).get("expires_ts") or 0)
        remaining = expires_ts - int(time.time())
        if remaining > 0:
            task_queue.enqueue_in(
                timedelta(seconds=remaining), delete_account_export_task,
                int(user_id), str(job_id), job_timeout=300,
            )
            return
        _delete_account_export_artifact(user_id, job_id, state="expired")
        leftovers = [
            path for path in (
                _account_export_path(user_id, job_id, "zip"),
                _account_export_path(user_id, job_id, "part"),
            ) if path and os.path.lexists(path)
        ]
        if leftovers:
            task_queue.enqueue_in(
                timedelta(seconds=60), delete_account_export_task,
                int(user_id), str(job_id), job_timeout=300,
            )


def _portable_datetime(value):
    return value.isoformat() + ("Z" if value and value.tzinfo is None else "") if value else None


def _parse_portable_datetime(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
    except Exception:
        return None


def _account_export_settings(user):
    result = {}
    for field in ACCOUNT_SETTING_FIELDS:
        value = getattr(user, field, None)
        if field == "system_prompt" and value and user.enable_e2ee:
            value = decrypt_val(value)
        result[field] = value
    return result


def _account_export_secrets(user):
    result = {}
    for field in ACCOUNT_SECRET_FIELDS:
        value = getattr(user, field, None)
        if field == "model_api_keys":
            result[field] = _load_user_model_api_key_map(user)
        else:
            result[field] = decrypt_val(value) if value else None
    return result


def _account_export_threads(user_id):
    rows = []
    threads = Thread.query.filter_by(user_id=user_id).order_by(Thread.id.asc()).all()
    for thread in threads:
        messages = []
        for message in Message.query.filter_by(thread_id=thread.id).order_by(Message.id.asc()).all():
            content = decrypt_val(message.content) if message.is_encrypted else message.content
            thought = decrypt_val(message.thought_data) if message.is_encrypted and message.thought_data else message.thought_data
            messages.append({
                "export_id": message.id,
                "parent_export_id": message.parent_id,
                "role": message.role,
                "content": content,
                "model": message.model,
                "image_url": message.image_url,
                "timestamp": _portable_datetime(message.timestamp),
                "tokens": message.tokens,
                "tokens_in": message.tokens_in,
                "tokens_out": message.tokens_out,
                "tokens_thought": message.tokens_thought,
                "thought_data": thought,
                "quote_text": message.quote_text,
                "thought_signature": message.thought_signature,
                "gem_uuid": message.gem_uuid,
                "gem_name": message.gem_name,
            })
        rows.append({
            "public_id": thread.public_id,
            "title": thread.title,
            "is_bookmarked": bool(thread.is_bookmarked),
            "bookmarked_at": _portable_datetime(thread.bookmarked_at),
            "is_temporary": bool(thread.is_temporary),
            "custom_instruction": thread.custom_instruction,
            "include_global_instruction": bool(thread.include_global_instruction),
            "last_model": thread.last_model,
            "last_gem_uuid": thread.last_gem_uuid,
            "enable_prompt_caching": bool(thread.enable_prompt_caching),
            "prompt_cache_provider": thread.prompt_cache_provider,
            "updated_at": _portable_datetime(thread.updated_at),
            "messages": messages,
        })
    return rows


def _account_export_gems(user_id):
    return [{
        "uuid": row.uuid,
        "name": row.name,
        "description": row.description,
        "instruction": row.instruction,
        "fixed_prompts_json": row.fixed_prompts_json,
        "default_model": row.default_model,
        "created_at": _portable_datetime(row.created_at),
    } for row in Gem.query.filter_by(user_id=user_id).order_by(Gem.id.asc()).all()]


def _account_export_feedback(user_id):
    return [{
        "title": row.title,
        "message": row.message,
        "status": row.status,
        "admin_reply": row.admin_reply,
        "created_at": _portable_datetime(row.created_at),
        "updated_at": _portable_datetime(row.updated_at),
    } for row in Feedback.query.filter_by(user_id=user_id).order_by(Feedback.id.asc()).all()]


def _account_export_diagnostics(user_id):
    first_tokens = [{
        "thread_public_id": row.thread_public_id, "job_id": row.job_id, "model": row.model,
        "first_event_type": row.first_event_type, "latency_seconds": row.latency_seconds,
        "latency_ms": row.latency_ms, "client_sent_at": _portable_datetime(row.client_sent_at),
        "created_at": _portable_datetime(row.created_at),
    } for row in FirstTokenLatencyMetric.query.filter_by(user_id=user_id).order_by(FirstTokenLatencyMetric.id.asc()).all()]
    trace_fields = [column.name for column in ChatLatencyTrace.__table__.columns if column.name not in {"id", "user_id"}]
    traces = []
    for row in ChatLatencyTrace.query.filter_by(user_id=user_id).order_by(ChatLatencyTrace.id.asc()).all():
        item = {}
        for field in trace_fields:
            value = getattr(row, field)
            item[field] = _portable_datetime(value) if isinstance(value, datetime) else value
        traces.append(item)
    return {"first_token_metrics": first_tokens, "chat_latency_traces": traces}


def _account_file_rows(user_id):
    user_dir = os.path.join(app.config["UPLOAD_FOLDER"], str(user_id))
    labels = _get_user_file_label_map(user_id)
    rows = []
    if not os.path.isdir(user_dir):
        return rows
    for root, dirs, filenames in os.walk(user_dir):
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        for filename in sorted(filenames):
            disk_path = os.path.join(root, filename)
            is_encrypted = filename.endswith(".enc")
            logical_path = disk_path[:-4] if is_encrypted else disk_path
            rel_inside = os.path.relpath(logical_path, user_dir)
            rel_path = os.path.join(str(user_id), rel_inside).replace(os.sep, "/")
            # If both representations somehow exist, prefer the plaintext one once.
            if is_encrypted and os.path.exists(logical_path):
                continue
            info = _get_file_disk_info(rel_path)
            if info.get("exists") and not _path_is_within(app.config["UPLOAD_FOLDER"], info.get("disk_path")):
                continue
            rows.append({
                "rel_path": rel_path,
                "display_name": labels.get(rel_path),
                "mime_type": mimetypes.guess_type(logical_path)[0],
                "info": info,
            })
    return rows


def _write_account_export_file(archive, archive_name, row):
    info = row.get("info") or {}
    if not info.get("exists"):
        raise AccountExportFileUnreadable("export_file_missing")
    digest = hashlib.sha256()
    size_bytes = 0
    if info.get("is_encrypted"):
        # Decrypt the Fernet token once, then stream the plaintext into the
        # archive in bounded chunks.  This keeps the export worker's peak
        # memory near the size of the largest single file instead of also
        # retaining every decrypted file in the shared media cache, which
        # could otherwise tip a small server over its memory limit and get
        # the whole export job OOM-killed.
        try:
            with open(info["disk_path"], "rb") as source:
                token = source.read()
        except OSError:
            raise AccountExportFileUnreadable("export_file_unreadable")
        try:
            data = decrypt_bytes(token)
        except Exception:
            data = None
        finally:
            del token
        if data is None:
            raise AccountExportFileUnreadable("export_file_unreadable")
        try:
            with archive.open(archive_name, "w", force_zip64=True) as target:
                for offset in range(0, len(data), 1024 * 1024):
                    chunk = data[offset:offset + 1024 * 1024]
                    target.write(chunk)
                    digest.update(chunk)
                    size_bytes += len(chunk)
        finally:
            del data
    else:
        with open(info["disk_path"], "rb") as source, archive.open(archive_name, "w", force_zip64=True) as target:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                digest.update(chunk)
                size_bytes += len(chunk)
    return size_bytes, digest.hexdigest()


def _write_account_export_recovery_file(archive, archive_name, row):
    """Preserve an unreadable file byte-for-byte for possible future recovery."""
    info = row.get("info") or {}
    disk_path = info.get("disk_path")
    if not info.get("exists") or not disk_path or not _path_is_within(app.config["UPLOAD_FOLDER"], disk_path):
        return None
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        with open(disk_path, "rb") as source, archive.open(archive_name, "w", force_zip64=True) as target:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                digest.update(chunk)
                size_bytes += len(chunk)
    except OSError:
        return None
    return {"archive_path": archive_name, "size_bytes": size_bytes, "sha256": digest.hexdigest()}


def _build_account_export_archive(user, job_id, export_path):
    _account_transfer_checkpoint(user.id, job_id, 2, "preparing", "エクスポート対象を確認しています")
    file_rows = _account_file_rows(user.id)
    manifest = {
        "format": ACCOUNT_EXPORT_FORMAT,
        "format_version": ACCOUNT_EXPORT_VERSION,
        "exported_at": _portable_datetime(datetime.utcnow()),
        "system_version": app.config.get("SYSTEM_VERSION"),
        "excluded": [
            "username", "password", "google_identity", "two_factor_secrets", "passkeys",
            "login_sessions", "account_privileges", "moderation_and_security_state",
        ],
        "data": {
            "settings": _account_export_settings(user),
            "api_credentials": _account_export_secrets(user),
            "chats": [],
            "gems": [],
            "feedback": [],
            "diagnostics": {},
            "files": [],
            "unreadable_files": [],
        },
    }
    _account_transfer_checkpoint(user.id, job_id, 5, "preparing", "チャット履歴を読み込んでいます")
    manifest["data"]["chats"] = _account_export_threads(user.id)
    _account_transfer_checkpoint(user.id, job_id, 12, "preparing", "Gemとフィードバックを読み込んでいます")
    manifest["data"]["gems"] = _account_export_gems(user.id)
    manifest["data"]["feedback"] = _account_export_feedback(user.id)
    _account_transfer_checkpoint(user.id, job_id, 16, "preparing", "診断データを読み込んでいます")
    manifest["data"]["diagnostics"] = _account_export_diagnostics(user.id)
    total_file_weight = sum(max(1, int((row.get("info") or {}).get("size") or 0)) for row in file_rows) or 1
    completed_file_weight = 0
    with zipfile.ZipFile(export_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for index, row in enumerate(file_rows, start=1):
            progress = 18 + int(72 * completed_file_weight / total_file_weight)
            _account_transfer_checkpoint(
                user.id, job_id, progress, "exporting_files",
                f"ファイルを書き出しています（{index}/{len(file_rows)}）",
            )
            archive_name = f"files/{index:06d}.bin"
            try:
                size_bytes, sha256 = _write_account_export_file(archive, archive_name, row)
                item = {key: value for key, value in row.items() if key != "info"}
                item["archive_path"] = archive_name
                item["size_bytes"] = size_bytes
                item["sha256"] = sha256
                item["mtime"] = (row.get("info") or {}).get("mtime")
                manifest["data"]["files"].append(item)
            except AccountExportFileUnreadable as exc:
                recovery_archive_name = f"recovery_files/{index:06d}.enc"
                recovery = _write_account_export_recovery_file(archive, recovery_archive_name, row)
                recovery_item = {key: value for key, value in row.items() if key != "info"}
                recovery_item.update({
                    "reason": str(exc),
                    "importable": False,
                    "encrypted_source": bool((row.get("info") or {}).get("is_encrypted")),
                    "mtime": (row.get("info") or {}).get("mtime"),
                })
                if recovery:
                    recovery_item.update(recovery)
                manifest["data"]["unreadable_files"].append(recovery_item)
                logger.warning(
                    "Preserved unreadable account export file for user %s: %s",
                    user.id, row.get("rel_path"),
                )
            completed_file_weight += max(1, int((row.get("info") or {}).get("size") or 0))
        _account_transfer_checkpoint(user.id, job_id, 92, "finalizing", "ZIPを仕上げています")
        archive.writestr(
            "account_data.json",
            json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        )
    return manifest


def build_account_export_task(user_id, job_id):
    with app.app_context():
        part_path = _account_export_path(user_id, job_id, "part")
        final_path = _account_export_path(user_id, job_id, "zip")
        try:
            if not part_path or not final_path:
                raise ValueError("invalid_export_path")
            user = db.session.get(User, int(user_id))
            if not user:
                raise ValueError("account_not_found")
            _cleanup_expired_account_export_files()
            for path in (part_path, final_path):
                if os.path.lexists(path):
                    secure_delete(path)
            manifest = _build_account_export_archive(user, job_id, part_path)
            _account_transfer_checkpoint(user.id, job_id, 96, "finalizing", "ダウンロード用ZIPを保存しています")
            os.chmod(part_path, 0o600)
            os.replace(part_path, final_path)
            try:
                os.chmod(final_path, 0o600)
            except OSError:
                pass
            ready_ts = int(time.time())
            expires_ts = ready_ts + ACCOUNT_EXPORT_RETENTION_SECONDS
            filename = f"ai-playground-account-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.zip"
            unreadable_count = len(manifest["data"]["unreadable_files"])
            message = (
                f"エクスポートが完了しました（読取不能な{unreadable_count}件は復旧用データとして収録）"
                if unreadable_count else "エクスポートZIPの準備が完了しました"
            )
            metadata = {
                "job_id": job_id,
                "filename": filename,
                "size_bytes": os.path.getsize(final_path),
                "ready_ts": ready_ts,
                "expires_ts": expires_ts,
                "expires_at": _portable_datetime(datetime.utcfromtimestamp(expires_ts)),
                "unreadable_count": unreadable_count,
            }
            redis_conn.setex(
                _account_export_artifact_key(user.id, job_id),
                ACCOUNT_TRANSFER_STATUS_TTL,
                json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
            )
            redis_conn.setex(_account_export_latest_key(user.id), ACCOUNT_TRANSFER_STATUS_TTL, job_id)
            _set_account_transfer_status(
                user.id, job_id, "ready", 100, "ready", message,
                expires_at=metadata["expires_at"], filename=filename,
                size_bytes=metadata["size_bytes"], available=True,
            )
            task_queue.enqueue_in(
                timedelta(seconds=ACCOUNT_EXPORT_RETENTION_SECONDS),
                delete_account_export_task, user.id, job_id, job_timeout=300,
            )
        except AccountTransferCancelled:
            if part_path and os.path.lexists(part_path):
                secure_delete(part_path)
            _set_account_transfer_status(
                user_id, job_id, "cancelled", 0, "cancelled", "エクスポートをキャンセルしました",
                available=False,
            )
        except Exception:
            for path in (part_path, final_path):
                if path and os.path.lexists(path):
                    secure_delete(path)
            logger.exception("Background account export failed for user %s", user_id)
            _set_account_transfer_status(
                user_id, job_id, "failed", 0, "failed", "エクスポートに失敗しました",
                available=False,
            )
        finally:
            try:
                active_raw = redis_conn.get(_account_export_active_key(user_id))
                active = active_raw.decode("utf-8") if isinstance(active_raw, bytes) else str(active_raw or "")
                if active == job_id:
                    redis_conn.delete(_account_export_active_key(user_id))
            except Exception:
                pass


def account_export_job_failure(job, connection, exc_type, exc_value, traceback_text):
    """Clean up and publish a terminal state if RQ stops the export itself."""
    try:
        user_id, job_id = job.args[:2]
    except Exception:
        return
    with app.app_context():
        _delete_account_export_artifact(user_id, job_id)
        _set_account_transfer_status(
            user_id, job_id, "failed", 0, "failed", "バックグラウンド処理が中断されました",
            available=False,
        )
        try:
            active_raw = redis_conn.get(_account_export_active_key(user_id))
            active = active_raw.decode("utf-8") if isinstance(active_raw, bytes) else str(active_raw or "")
            if active == job_id:
                redis_conn.delete(_account_export_active_key(user_id))
        except Exception:
            pass


def _fernet_encrypted_size(plain_size):
    size = max(0, int(plain_size or 0))
    padded_ciphertext_size = 16 * ((size // 16) + 1)
    token_binary_size = 57 + padded_ciphertext_size
    return 4 * ((token_binary_size + 2) // 3)


def _coerce_account_import_categories(raw):
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",")]
    if not isinstance(raw, list):
        return set()
    return {str(item).strip() for item in raw if str(item).strip() in ACCOUNT_IMPORT_CATEGORIES}


IMPORT_SELECTED_FILES_NONE = "__none__"


def _coerce_import_selected_files(raw):
    """Parse the archive_paths the client wants to import.

    Returns ``None`` when no filter was supplied (import every file), an empty
    set when the client explicitly deselected every file, or the set of allowed
    ``files/`` archive paths.  Only entries under ``files/`` are accepted so a
    crafted value cannot smuggle another archive member into the files list.
    """
    if raw is None:
        return None
    if isinstance(raw, (list, tuple, set)):
        items = [str(x).strip() for x in raw]
    else:
        items = [part.strip() for part in str(raw or "").split(",") if part.strip()]
    if not items:
        return None
    if items == [IMPORT_SELECTED_FILES_NONE]:
        return set()
    return {item for item in items if item.startswith("files/")}


def _account_storage_limit_payload(file_items, used_bytes, limit_bytes):
    """Build the selection payload shown when an import exceeds storage.

    Shared by the HTTP response and the ``needs_selection`` transfer status so
    the client can recover the file list even when the (potentially large)
    409 response body is lost on a flaky connection.
    """
    files = []
    for item in file_items or []:
        if not isinstance(item, dict):
            continue
        files.append({
            "archive_path": str(item.get("archive_path") or ""),
            "rel_path": str(item.get("rel_path") or ""),
            "display_name": str(item.get("display_name") or os.path.basename(str(item.get("rel_path") or "")) or "file"),
            "size_bytes": int(item.get("size_bytes") or 0),
        })
    return {
        "status": "storage_limit",
        "error": "storage_limit_files",
        "message": "ストレージ容量が不足しています。インポートするファイルを選択してください。",
        "files": files,
        "used_bytes": int(used_bytes or 0),
        "limit_bytes": int(limit_bytes or 0),
        "available_bytes": max(0, int(limit_bytes or 0) - int(used_bytes or 0)),
    }


def _safe_account_import_text(value, max_chars):
    text_value = "" if value is None else str(value).replace("\x00", "")
    if len(text_value) > max_chars:
        raise ValueError("import_value_too_large")
    return text_value


def _normalize_imported_account_settings(user, values):
    """Return the canonical (plaintext) values that would be stored for import.

    Fields that are absent from ``values`` are skipped, invalid model ids are
    ignored, and every other field is normalized to the same form used when
    saving settings from the settings screen.  The ``system_prompt`` is returned
    in plaintext (encryption is applied later when storing).
    """
    if not isinstance(values, dict):
        raise ValueError("invalid_settings")
    normalized = {}
    for field in ACCOUNT_SETTING_FIELDS:
        if field not in values:
            continue
        value = values.get(field)
        if field in ACCOUNT_BOOL_SETTING_FIELDS:
            value = bool(value)
        elif field in ACCOUNT_INT_SETTING_FIELDS:
            value = int(value or 0)
            if field == "temp_chat_timeout_seconds":
                value = _normalize_temp_chat_timeout_seconds(value)
            else:
                value = max(0, min(value, 32768))
        else:
            value = _safe_account_import_text(value, ACCOUNT_TEXT_SETTING_LIMITS.get(field, 4096))
        if field in {"default_model", "default_vision_model", "last_model"} and value not in ALL_VALID_MODEL_IDS:
            continue
        if field == "stt_model" and value not in VALID_STT_MODELS:
            continue
        if field == "theme_color":
            value = normalize_theme_color(value)
        elif field == "gemini_backend":
            value = _normalize_gemini_backend(value)
        elif field == "gemini_vertex_location":
            value = _normalize_gemini_vertex_location(value)
        elif field == "mic_transcribe_mode":
            value = _normalize_mic_transcribe_mode(value)
        elif field == "llm_transcribe_prompt":
            value = _normalize_llm_transcribe_prompt(value)
        normalized[field] = value
    return normalized


def _apply_imported_account_settings(user, values):
    for field, value in _normalize_imported_account_settings(user, values).items():
        if field == "system_prompt" and user.enable_e2ee:
            value = encrypt_val(value)
        setattr(user, field, value)


def _account_setting_values_equal(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if a is None:
        a = ""
    if b is None:
        b = ""
    return a == b


def _account_import_settings_changes(user, values):
    """List the settings that would change if ``values`` were imported.

    Returns ``[{"field": ..., "current": ..., "incoming": ...}, ...]`` for every
    setting whose normalized import value differs from the current value, so the
    client can show the user exactly what would be overwritten before starting.
    """
    normalized = _normalize_imported_account_settings(user, values)
    changes = []
    for field, incoming in normalized.items():
        current = getattr(user, field, None)
        if field == "system_prompt" and current and user.enable_e2ee:
            try:
                current = decrypt_val(current)
            except Exception:
                pass
        if _account_setting_values_equal(current, incoming):
            continue
        changes.append({
            "field": field,
            "current": current,
            "incoming": incoming,
        })
    return changes


def _apply_imported_account_secrets(user, values):
    if not isinstance(values, dict):
        raise ValueError("invalid_api_credentials")
    for field in ACCOUNT_SECRET_FIELDS:
        if field not in values:
            continue
        value = values.get(field)
        if field == "model_api_keys":
            if not isinstance(value, dict) or len(value) > 500:
                raise ValueError("invalid_model_api_keys")
            clean = {}
            for model_id, secret in value.items():
                model_id = str(model_id).strip()
                secret = _safe_account_import_text(secret, 4096)
                if model_id in ALL_VALID_MODEL_IDS and secret:
                    clean[model_id] = secret
            _save_user_model_api_key_map(user, clean)
            continue
        max_chars = 100_000 if field == "gemini_vertex_credentials_json" else 4096
        value = _safe_account_import_text(value, max_chars) if value else None
        if field == "gemini_vertex_credentials_json" and value:
            value = _normalize_gemini_vertex_credentials_json(value)
        setattr(user, field, encrypt_val(value) if value else None)


def _unique_import_file_rel_path(user_id, original_rel_path):
    base_name = _sanitize_file_display_name(os.path.basename(str(original_rel_path or ""))) or "imported-file"
    stem, ext = os.path.splitext(base_name)
    stem = (stem or "imported-file")[:120]
    for _ in range(100):
        candidate = f"{user_id}/import-{secrets.token_hex(6)}-{stem}{ext}"
        base = os.path.join(app.config["UPLOAD_FOLDER"], candidate)
        if not os.path.exists(base) and not os.path.exists(base + ".enc"):
            return candidate
    raise ValueError("file_name_collision")


def _rewrite_imported_attachment_value(raw_value, file_map, target_user_id):
    if not raw_value:
        return raw_value
    try:
        parsed = json.loads(raw_value) if isinstance(raw_value, str) else raw_value
    except Exception:
        parsed = raw_value

    def rewrite(item):
        if isinstance(item, list):
            return [rewrite(child) for child in item]
        if isinstance(item, dict):
            result = dict(item)
            for key in ("path", "filepath", "file", "url", "name"):
                if key in result and result[key]:
                    result[key] = rewrite(result[key])
                    break
            return result
        norm = _normalize_upload_ref(item)
        if norm in file_map:
            return file_map[norm]
        if norm and norm.startswith(f"{target_user_id}/") and _get_file_disk_info(norm).get("exists"):
            return norm
        return item

    rewritten = rewrite(parsed)
    if isinstance(raw_value, str) and isinstance(parsed, (list, dict)):
        return json.dumps(rewritten, ensure_ascii=False, separators=(",", ":"))
    return rewritten

_LATENCY_PHASE_TO_FIELD = {
    "route_received_ms": "route_received_at",
    "route_dispatch_ms": "route_dispatch_at",
    "route_stream_open_ms": "route_stream_open_at",
    "worker_started_ms": "worker_started_at",
    "provider_request_started_ms": "provider_request_started_at",
    "provider_first_chunk_ms": "provider_first_chunk_at",
    "provider_first_status_ms": "provider_first_status_at",
    "provider_first_thought_ms": "provider_first_thought_at",
    "provider_first_content_ms": "provider_first_content_at",
    "stream_first_pubsub_ms": "stream_first_pubsub_at",
    "stream_first_status_to_client_ms": "stream_first_status_to_client_at",
    "stream_first_thought_to_client_ms": "stream_first_thought_to_client_at",
    "stream_first_content_to_client_ms": "stream_first_content_to_client_at",
    "stream_done_ms": "stream_done_at",
    "worker_done_ms": "worker_done_at",
}

