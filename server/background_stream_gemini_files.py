"""Gemini file parts, uploads, and the Files API cache.

Extracted from `background_stream_gemini_chat.py` so the chat branch keeps only
the stream flow.  The helpers used to be nested closures; they are plain
functions here because every other name they use (`types`, `os`, `time`,
`datetime`, `tempfile`, `log_force`, `_get_file_cache`, `_upsert_file_cache`,
`safe_db_commit`, ...) is provided by the `app` namespace that `app.py` builds
by exec'ing every server part.

The three request-specific values stay explicit arguments instead of hidden
globals: `user_id`, `files_enabled` (the Files API is off in Vertex AI mode),
and `client` (the `genai` client of the running task).
"""

def _gemini_file_state_name(fobj):
    if not fobj:
        return None
    st = fobj.get("state") if isinstance(fobj, dict) else getattr(fobj, "state", None)
    if isinstance(st, dict):
        return st.get("name") or st.get("state")
    return getattr(st, "name", None) or st

def _gemini_file_name(fobj):
    if not fobj:
        return None
    return fobj.get("name") if isinstance(fobj, dict) else getattr(fobj, "name", None)

def _gemini_file_uri(fobj):
    if not fobj:
        return None
    if isinstance(fobj, dict):
        return fobj.get("uri") or fobj.get("file_uri") or fobj.get("fileUri") or fobj.get("name")
    return (
        getattr(fobj, "uri", None)
        or getattr(fobj, "file_uri", None)
        or getattr(fobj, "fileUri", None)
        or getattr(fobj, "name", None)
    )

def _make_gemini_uri_part(file_uri, mime):
    if not file_uri:
        return None
    if hasattr(types.Part, "from_uri"):
        try:
            return types.Part.from_uri(file_uri, mime_type=mime)
        except TypeError:
            try:
                return types.Part.from_uri(file_uri, mime)
            except Exception:
                try:
                    return types.Part.from_uri(file_uri=file_uri, mime_type=mime)
                except Exception:
                    try:
                        return types.Part.from_uri(uri=file_uri, mime_type=mime)
                    except Exception:
                        return None
    if hasattr(types, "FileData"):
        try:
            return types.Part(file_data=types.FileData(file_uri=file_uri, mime_type=mime))
        except Exception:
            try:
                return types.Part(file_data=types.FileData(file_uri=file_uri))
            except Exception:
                return None
    return None

def _wait_gemini_file_active(client, fobj, label=""):
    state = _gemini_file_state_name(fobj)
    if not state or state == "ACTIVE":
        return fobj, state
    if state == "FAILED":
        return fobj, state
    name = _gemini_file_name(fobj)
    deadline = time.time() + 120
    while state == "PROCESSING" and time.time() < deadline:
        time.sleep(2)
        try:
            if name:
                fobj = client.files.get(name=name)
            else:
                break
        except Exception as e:
            log_force(f"Gemini file poll failed {label}: {e}")
            break
        state = _gemini_file_state_name(fobj)
    return fobj, state

def _normalize_gemini_audio(data, mime, name=""):
    if not data or not mime:
        return data, mime, name
    m = (mime or '').lower()
    ext = (os.path.splitext(name or '')[1] or '').lower()
    if m in ("audio/webm", "audio/ogg", "audio/oga", "audio/opus") or ext in (".webm", ".ogg", ".oga", ".opus"):
        try:
            src_suffix = ext if ext else ".webm"
            pcm = _convert_audio_to_pcm(data, src_suffix=src_suffix, rate=16000)
            wav = _pcm_to_wav_bytes(pcm, rate=16000)
            base = os.path.splitext(name or "audio")[0]
            return wav, "audio/wav", f"{base}.wav"
        except Exception as e:
            log_force(f"Gemini audio convert failed: {e}")
    return data, mime, name

def _gemini_cache_matches(cache, size, mtime, mime):
    if not cache or not cache.file_uri:
        return False
    if size is not None and cache.size_bytes is not None and cache.size_bytes != size:
        return False
    if mtime is not None and cache.mtime is not None and cache.mtime != mtime:
        return False
    if mime and cache.mime_type and cache.mime_type != mime:
        return False
    return True

def _gemini_get_cached_part_impl(user_id, files_enabled, client, rel_path, mime, size=None, mtime=None, label=""):
    if not files_enabled:
        return None
    cache = _get_file_cache(user_id, rel_path, "gemini")
    if not _gemini_cache_matches(cache, size, mtime, mime):
        return None
    try:
        if cache.file_id:
            fobj = client.files.get(name=cache.file_id)
            state = _gemini_file_state_name(fobj)
            cache.file_uri = _gemini_file_uri(fobj) or cache.file_uri
            cache.state = state or cache.state
            cache.last_checked_at = datetime.utcnow()
            if state and state != "ACTIVE":
                if state == "PROCESSING":
                    fobj, state = _wait_gemini_file_active(client, fobj, label=label)
                    cache.file_uri = _gemini_file_uri(fobj) or cache.file_uri
                    cache.state = state or cache.state
                if state and state != "ACTIVE":
                    _upsert_file_cache(
                        user_id,
                        rel_path,
                        "gemini",
                        state=state,
                        last_error=f"state:{state}",
                        size_bytes=size,
                        mtime=mtime,
                        mime_type=mime,
                        last_checked_at=datetime.utcnow()
                    )
                    safe_db_commit()
                    return None
        part = _make_gemini_uri_part(cache.file_uri, mime)
        if part:
            _upsert_file_cache(
                user_id,
                rel_path,
                "gemini",
                state="ACTIVE",
                last_error=None,
                size_bytes=size,
                mtime=mtime,
                mime_type=mime,
                last_checked_at=datetime.utcnow()
            )
            safe_db_commit()
            return part
    except Exception as e:
        _upsert_file_cache(
            user_id,
            rel_path,
            "gemini",
            state="FAILED",
            last_error=str(e),
            size_bytes=size,
            mtime=mtime,
            mime_type=mime,
            last_checked_at=datetime.utcnow()
        )
        safe_db_commit()
    return None

def _gemini_upload_with_retry_impl(user_id, files_enabled, client, data, mime, suffix, rel_path,
                                   label="", display_name=None):
    if not files_enabled:
        return None, None, "Vertex AI モードではこのアプリの Files API 経路を利用できません（20MB以下にするか Gemini API モードへ切替してください）。"
    max_attempts = 2
    try:
        max_attempts = int(os.getenv("GEMINI_FILE_UPLOAD_RETRIES", "2") or "2")
    except Exception:
        max_attempts = 2
    last_err = None
    for attempt in range(max_attempts):
        try:
            _upsert_file_cache(
                user_id,
                rel_path,
                "gemini",
                state="UPLOADING",
                last_error=None,
                retries=attempt + 1
            )
            safe_db_commit()
            with tempfile.NamedTemporaryFile(suffix=suffix or '.bin') as tmp:
                tmp.write(data)
                tmp.flush()
                # UploadFileConfig forbids extra keys: give each field once (an alias
                # next to its field name counts as extra), and leave `name` to the API,
                # which takes it as the `files/<id>` resource name.
                config = {"mime_type": mime}
                if display_name:
                    config["display_name"] = display_name
                up = client.files.upload(file=tmp.name, config=config)
            up, up_state = _wait_gemini_file_active(client, up, label=label)
            if up_state and up_state != "ACTIVE":
                last_err = f"state:{up_state}"
                time.sleep(1)
                continue
            return up, up_state, None
        except Exception as e:
            last_err = str(e)
            time.sleep(1)
            continue
    return None, None, last_err
