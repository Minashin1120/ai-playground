# === AI-assisted settings (in settings modal) ===
# Safe (non-secret, non-admin, low-risk) fields that can be changed via natural language prompt + tool use.
# This list defines the tool schema and what _apply_ai_settings_update will accept.
# Keep in sync with settings modal UI and handle_settings POST handling for these fields.
AI_SAFE_EDITABLE_FIELDS = {
    # Default chat behaviors
    'default_model', 'default_enable_search', 'default_enable_url_context', 'default_enable_maps',
    'default_enable_python', 'default_enable_file_creation', 'default_enable_thinking', 'default_thinking_level',
    'default_thinking_budget', 'default_reasoning_effort', 'default_enable_system_prompt',
    'default_safety_setting',
    # Vision model for image analysis
    'default_vision_model',
    # User-level instruction / system prompts
    'system_prompt', 'system_prompt_enabled', 'apply_global_system_prompt',
    'apply_auto_system_prompt_notices', 'auto_system_prompt_notices_config',
    # STT / voice transcription
    'mic_transcribe_mode', 'stt_model', 'llm_transcribe_prompt',
    # UI / behavior prefs (safe)
    'enter_to_send', 'use_sw_cache', 'compact_prompt_mode', 'minimal_prompt_mode', 'auto_search_on_links',
    'use_last_chat_settings', 'voice_studio_ui', 'temp_chat_timeout_seconds', 'theme_color', 'liquid_glass_enabled',
    # Rich paste custom prompt
    'rich_paste_prompt_default', 'rich_paste_prompt_use_custom_default',
    # Debug / metrics (user opt-in)
    'enable_latency_metrics', 'enable_client_debug_log',
    # Per-user bot detection preference
    'bot_detection_enabled',
    # Low-risk 2FA prefs (not passkey_only_login which risks lockout)
    'skip_2fa_on_google_login', 'default_2fa_method',
}

# Fields that must NEVER be settable via AI prompt (secrets, high-impact, admin-only, or runtime)
AI_NEVER_EDITABLE_FIELDS = {
    # Secrets / credentials
    'openai_key', 'gemini_key', 'anthropic_key', 'deepseek_key', 'xai_key', 'kimi_key',
    'mistral_key', 'google_key',
    'google_project', 'gemini_vertex_project', 'gemini_vertex_location',
    'gemini_vertex_credentials_json', 'model_api_keys',
    # High impact / special handling
    'new_password', 'new_username', 'enable_e2ee', 'disable_2fa', 'passkey_only_login',
    'last_gem_uuid',
    # Admin-only (global)
    'admin_api_key_mode', 'bot_detection_global_enabled',
}


def _get_ai_safe_settings_snapshot(user):
    """Return current values for AI-readable settings only; never include credentials or admin settings."""
    values = {field: getattr(user, field, None) for field in AI_SAFE_EDITABLE_FIELDS}
    system_prompt = getattr(user, 'system_prompt', None) or ''
    if getattr(user, 'enable_e2ee', False) and system_prompt:
        system_prompt = decrypt_val(system_prompt)
    values.update({
        'system_prompt': system_prompt,
        'system_prompt_enabled': user.system_prompt_enabled if user.system_prompt_enabled is not None else True,
        'apply_global_system_prompt': user.apply_global_system_prompt if user.apply_global_system_prompt is not None else True,
        'apply_auto_system_prompt_notices': get_user_auto_system_prompt_notices_enabled(user),
        'auto_system_prompt_notices_config': get_user_auto_system_prompt_notices_config(user),
        'mic_transcribe_mode': _normalize_mic_transcribe_mode(getattr(user, 'mic_transcribe_mode', None)),
        'stt_model': getattr(user, 'stt_model', None) or 'gpt-4o-mini-transcribe',
        'llm_transcribe_prompt': _normalize_llm_transcribe_prompt(getattr(user, 'llm_transcribe_prompt', None)) or '',
        'temp_chat_timeout_seconds': _get_user_temp_chat_timeout_seconds(user),
        'default_model': getattr(user, 'default_model', None) or 'gemini-3.6-flash',
        'default_vision_model': getattr(user, 'default_vision_model', None) or 'gemini-3-flash-preview',
        'default_thinking_level': getattr(user, 'default_thinking_level', None) or 'high',
        'default_thinking_budget': user.default_thinking_budget if user.default_thinking_budget is not None else 4096,
        'default_reasoning_effort': getattr(user, 'default_reasoning_effort', None) or 'medium',
        'default_safety_setting': getattr(user, 'default_safety_setting', None) or 'default',
        'rich_paste_prompt_default': getattr(user, 'rich_paste_prompt_default', None) or '',
        'rich_paste_prompt_use_custom_default': bool(getattr(user, 'rich_paste_prompt_use_custom_default', False)),
        'theme_color': getattr(user, 'theme_color', None) or '',
        'liquid_glass_enabled': bool(getattr(user, 'liquid_glass_enabled', False)),
        'compact_prompt_mode': bool(getattr(user, 'compact_prompt_mode', False)),
        'minimal_prompt_mode': bool(getattr(user, 'minimal_prompt_mode', False)),
        'voice_studio_ui': bool(getattr(user, 'voice_studio_ui', True)),
        'enable_latency_metrics': bool(getattr(user, 'enable_latency_metrics', False)),
        'enable_client_debug_log': bool(getattr(user, 'enable_client_debug_log', False)),
        'bot_detection_enabled': user.bot_detection_enabled if user.bot_detection_enabled is not None else True,
        'skip_2fa_on_google_login': bool(getattr(user, 'skip_2fa_on_google_login', False)),
        'default_2fa_method': getattr(user, 'default_2fa_method', None) or 'totp',
    })
    return {field: values.get(field) for field in sorted(AI_SAFE_EDITABLE_FIELDS)}


def _summarize_ai_settings_for_model(values):
    """Keep private/long prompt bodies out of the intent-classification context."""
    summarized = dict(values or {})
    for field in ('system_prompt', 'llm_transcribe_prompt', 'rich_paste_prompt_default'):
        summarized[field] = '設定済み' if summarized.get(field) else '未設定'
    if summarized.get('auto_system_prompt_notices_config'):
        summarized['auto_system_prompt_notices_config'] = '設定済み'
    return summarized

def _apply_ai_settings_update(current_user, delta):
    """
    Apply a delta dict (from AI tool call) to current_user, but ONLY for AI_SAFE_EDITABLE_FIELDS.
    Never touches secrets, admin-only, or high-risk fields. Mirrors the assignment logic
    from handle_settings POST for consistency ("ロジックは同じ").
    Does NOT commit. Caller must safe_db_commit() after.
    Returns dict of actually applied field->new_value (for response summary).
    """
    if not delta or not isinstance(delta, dict):
        return {}
    applied = {}
    for key, val in delta.items():
        if key not in AI_SAFE_EDITABLE_FIELDS:
            continue  # silently ignore unknown or disallowed
        if key in AI_NEVER_EDITABLE_FIELDS:
            continue
        try:
            if key == 'system_prompt':
                if current_user.enable_e2ee:
                    current_user.system_prompt = encrypt_val(val or "")
                else:
                    current_user.system_prompt = val or ""
                applied[key] = "(更新)"
            elif key == 'system_prompt_enabled':
                current_user.system_prompt_enabled = bool(val)
                applied[key] = current_user.system_prompt_enabled
            elif key == 'apply_global_system_prompt':
                current_user.apply_global_system_prompt = bool(val)
                applied[key] = current_user.apply_global_system_prompt
            elif key == 'apply_auto_system_prompt_notices':
                current_user.apply_auto_system_prompt_notices = bool(val)
                applied[key] = current_user.apply_auto_system_prompt_notices
            elif key == 'auto_system_prompt_notices_config':
                set_user_auto_system_prompt_notices_config(current_user, val)
                applied[key] = "(更新)"
            elif key in ('mic_transcribe_mode',):
                current_user.mic_transcribe_mode = _normalize_mic_transcribe_mode(val)
                applied[key] = current_user.mic_transcribe_mode
            elif key == 'stt_model':
                _raw = str(val).strip() if val else ''
                if _raw and _raw not in VALID_STT_MODELS:
                    log_force(f"AI-SETTINGS-INVALID-STT: user={current_user.id} rejected={_raw}")
                    continue
                current_user.stt_model = _raw if _raw else "gpt-4o-mini-transcribe"
                applied[key] = current_user.stt_model
            elif key == 'llm_transcribe_prompt':
                current_user.llm_transcribe_prompt = _normalize_llm_transcribe_prompt(val)
                applied[key] = "(更新)"
            elif key == 'enter_to_send':
                current_user.enter_to_send = bool(val)
                applied[key] = current_user.enter_to_send
            elif key == 'use_sw_cache':
                current_user.use_sw_cache = bool(val)
                applied[key] = current_user.use_sw_cache
            elif key == 'compact_prompt_mode':
                current_user.compact_prompt_mode = bool(val)
                if current_user.compact_prompt_mode:
                    current_user.minimal_prompt_mode = False
                applied[key] = current_user.compact_prompt_mode
            elif key == 'minimal_prompt_mode':
                current_user.minimal_prompt_mode = bool(val)
                if current_user.minimal_prompt_mode:
                    current_user.compact_prompt_mode = False
                applied[key] = current_user.minimal_prompt_mode
            elif key == 'theme_color':
                current_user.theme_color = normalize_theme_color(val)
                applied[key] = current_user.theme_color
            elif key == 'liquid_glass_enabled':
                current_user.liquid_glass_enabled = bool(val)
                applied[key] = current_user.liquid_glass_enabled
            elif key == 'auto_search_on_links':
                current_user.auto_search_on_links = bool(val)
                applied[key] = current_user.auto_search_on_links
            elif key == 'use_last_chat_settings':
                current_user.use_last_chat_settings = bool(val)
                applied[key] = current_user.use_last_chat_settings
            elif key == 'voice_studio_ui':
                current_user.voice_studio_ui = bool(val)
                applied[key] = current_user.voice_studio_ui
            elif key == 'temp_chat_timeout_seconds':
                current_user.temp_chat_timeout_seconds = _normalize_temp_chat_timeout_seconds(val)
                applied[key] = current_user.temp_chat_timeout_seconds
            elif key == 'default_model':
                _raw = str(val).strip() if val else ''
                if _raw and _raw not in ALL_VALID_MODEL_IDS:
                    log_force(f"AI-SETTINGS-INVALID-MODEL: user={current_user.id} rejected={_raw}")
                    continue
                current_user.default_model = _raw if _raw else current_user.default_model
                applied[key] = current_user.default_model
            elif key == 'default_enable_search':
                current_user.default_enable_search = bool(val)
                applied[key] = current_user.default_enable_search
            elif key == 'default_enable_url_context':
                current_user.default_enable_url_context = bool(val)
                applied[key] = current_user.default_enable_url_context
            elif key == 'default_enable_maps':
                current_user.default_enable_maps = bool(val)
                applied[key] = current_user.default_enable_maps
            elif key == 'default_enable_python':
                current_user.default_enable_python = bool(val)
                applied[key] = current_user.default_enable_python
            elif key == 'default_enable_file_creation':
                current_user.default_enable_file_creation = bool(val)
                applied[key] = current_user.default_enable_file_creation
            elif key == 'default_enable_thinking':
                current_user.default_enable_thinking = bool(val)
                applied[key] = current_user.default_enable_thinking
            elif key == 'default_thinking_level':
                _raw = str(val).strip() if val else ''
                if _raw in VALID_THINKING_LEVELS:
                    current_user.default_thinking_level = _raw
                else:
                    current_user.default_thinking_level = "high"
                applied[key] = current_user.default_thinking_level
            elif key == 'default_thinking_budget':
                try:
                    current_user.default_thinking_budget = int(val)
                    applied[key] = current_user.default_thinking_budget
                except Exception:
                    pass
            elif key == 'default_reasoning_effort':
                _raw = str(val).strip() if val else ''
                if _raw in VALID_REASONING_EFFORTS:
                    current_user.default_reasoning_effort = _raw
                else:
                    current_user.default_reasoning_effort = "medium"
                applied[key] = current_user.default_reasoning_effort
            elif key == 'default_enable_system_prompt':
                current_user.default_enable_system_prompt = bool(val)
                applied[key] = current_user.default_enable_system_prompt
            elif key == 'default_safety_setting':
                _raw = str(val).strip() if val else ''
                if _raw in VALID_SAFETY_SETTINGS:
                    current_user.default_safety_setting = _raw
                else:
                    current_user.default_safety_setting = "default"
                applied[key] = current_user.default_safety_setting
            elif key == 'rich_paste_prompt_default':
                current_user.rich_paste_prompt_default = str(val or "")
                applied[key] = current_user.rich_paste_prompt_default
            elif key == 'rich_paste_prompt_use_custom_default':
                current_user.rich_paste_prompt_use_custom_default = bool(val)
                applied[key] = current_user.rich_paste_prompt_use_custom_default
            elif key == 'enable_latency_metrics':
                current_user.enable_latency_metrics = bool(val)
                applied[key] = current_user.enable_latency_metrics
            elif key == 'enable_client_debug_log':
                current_user.enable_client_debug_log = bool(val)
                applied[key] = current_user.enable_client_debug_log
                log_force(f"SETTINGS-AI-UPDATE: user={current_user.id} enable_client_debug_log={val}")
            elif key == 'bot_detection_enabled':
                if val is not None:
                    current_user.bot_detection_enabled = bool(val)
                    applied[key] = current_user.bot_detection_enabled
            elif key == 'skip_2fa_on_google_login':
                current_user.skip_2fa_on_google_login = bool(val)
                applied[key] = current_user.skip_2fa_on_google_login
            elif key == 'default_2fa_method':
                current_user.default_2fa_method = str(val) if val in ('totp', 'webauthn') else 'totp'
                applied[key] = current_user.default_2fa_method
            # Note: auto_system_prompt_notices_config handled above; complex ones can be extended later
        except Exception as e:
            log_force(f"AI-SETTINGS-APPLY-ERR key={key} err={e}")
    return applied

def _normalize_mic_transcribe_mode(value):
    v = str(value or "").strip().lower()
    return v if v in MIC_TRANSCRIBE_MODES else "stt_api"

def _normalize_llm_transcribe_prompt(raw_text):
    if raw_text is None:
        return None
    text = str(raw_text).strip()
    if not text:
        return None
    if len(text) > LLM_TRANSCRIBE_PROMPT_MAX_CHARS:
        text = text[:LLM_TRANSCRIBE_PROMPT_MAX_CHARS]
    return text

def get_user_llm_transcribe_prompt(user):
    try:
        raw = getattr(user, "llm_transcribe_prompt", None)
    except Exception:
        raw = None
    return _normalize_llm_transcribe_prompt(raw) or DEFAULT_LLM_TRANSCRIBE_PROMPT

def _analyze_image_with_vision_model(model_id, img_data, img_mime, prompt, api_keys):
    try:
        model_l = model_id.lower()
        img_b64 = base64.b64encode(img_data).decode("utf-8")
        data_uri = f"data:{img_mime};base64,{img_b64}"

        # --- Gemini ---
        if model_l.startswith("gemini-"):
            g_key = api_keys.get("gemini")
            if not g_key:
                return None
            try:
                import google.genai.types as types
                g_client = _get_gemini_client(api_key=g_key)
                if not g_client:
                    return None
                resp = g_client.models.generate_content(
                    model=model_id,
                    contents=[prompt, types.Part.from_bytes(data=img_data, mime_type=img_mime)]
                )
                return resp.text if hasattr(resp, "text") else None
            except Exception:
                pass
            return None

        # --- OpenAI / xAI (Grok) ---
        if model_l.startswith(("gpt-", "o1-", "o3-", "grok-")):
            base_url = None
            oa_key = None
            if model_l.startswith("grok-"):
                oa_key = api_keys.get("xai")
                base_url = f"https://{_XAI_API_HOST}/v1"
            else:
                oa_key = api_keys.get("openai")
            if not oa_key:
                return None
            oa_client = _get_openai_client(oa_key, base_url=base_url)
            if not oa_client:
                return None
            resp = oa_client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": data_uri}}]}],
                max_tokens=4096,
            )
            return getattr(resp.choices[0].message, "content", None) or ""

        # --- Anthropic Claude ---
        if model_l.startswith("claude-"):
            c_key = api_keys.get("anthropic")
            if not c_key:
                return None
            try:
                from anthropic import Anthropic
            except ImportError:
                return None
            c_client = Anthropic(api_key=c_key)
            resp = c_client.messages.create(
                model=model_id,
                max_tokens=4096,
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "source": {"type": "base64", "media_type": img_mime, "data": img_b64}}]}],
            )
            text_parts = []
            for block in resp.content:
                if block.type == "text":
                    text_parts.append(block.text)
            return "".join(text_parts) or None

        return None
    except Exception as e:
        log_force(f"IMAGE_ANALYSIS_ERROR: model={model_id} error={e}")
        return None

def _extract_openai_response_text(resp):
    try:
        if isinstance(resp, dict):
            raw = resp.get("output_text")
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        else:
            raw = getattr(resp, "output_text", None)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        output = resp.get("output") if isinstance(resp, dict) else getattr(resp, "output", None)
        texts = []
        for item in output or []:
            content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
            for part in content or []:
                if isinstance(part, dict):
                    p_type = part.get("type")
                    if p_type in ("output_text", "text") and part.get("text"):
                        texts.append(str(part.get("text")))
                else:
                    p_type = getattr(part, "type", None)
                    p_text = getattr(part, "text", None)
                    if p_type in ("output_text", "text") and p_text:
                        texts.append(str(p_text))
        if texts:
            return "\n".join(texts).strip()
    except Exception:
        return ""
    return ""

def _transcribe_audio_with_llm(audio_content, fname, llm_model_key, user):
    no_speech_token = "[[NO_SPEECH]]"
    base_transcription_prompt = get_user_llm_transcribe_prompt(user)
    transcription_prompt = (
        f"{base_transcription_prompt}\n"
        f"聞き取れない、無音、音声が極端に小さい場合は推測せず {no_speech_token} のみを返してください。"
    )
    model_key = (llm_model_key or "").strip()
    model_key_l = model_key.lower()
    is_gem = is_gemini_model_key(model_key_l)
    is_deepseek = is_deepseek_model_key(model_key_l)
    is_grok = ("grok" in model_key_l) and ("gpt" not in model_key_l)
    if is_grok:
        raise ValueError("現在の xAI/Grok モデルのLLM文字起こしは未対応です。OpenAI/Gemini対応モデルに切り替えるか、STT APIを使用してください。")
    if is_deepseek:
        raise ValueError("DeepSeek モデルのLLM文字起こしは未対応です。OpenAI/Gemini対応モデルに切り替えるか、STT APIを使用してください。")
    if not model_key:
        model_key = "gpt-4o-mini"
        model_key_l = model_key.lower()

    src_ext = os.path.splitext(fname or "")[1].lower() or ".webm"
    if src_ext not in (".webm", ".wav", ".mp3", ".m4a", ".ogg", ".flac", ".opus"):
        src_ext = ".webm"
    try:
        target_rate = 16000 if is_gem else 24000
        pcm = _convert_audio_to_pcm(audio_content, src_suffix=src_ext, rate=target_rate)
        wav_bytes = _pcm_to_wav_bytes(pcm, rate=target_rate)
    except Exception as e:
        raise RuntimeError(f"Audio conversion failed (ffmpeg): {e}") from e

    metrics = _pcm_audio_metrics_mono_s16le(pcm, rate=target_rate)
    # Guard against near-silent capture in LLM mode to prevent hallucinated transcripts.
    # Relaxed thresholds to allow quieter valid inputs.
    if metrics["duration_sec"] >= 0.35 and metrics["rms"] < 30 and metrics["peak"] < 250:
        logger.warning(
            "LLM transcription rejected due to near-silent audio "
            f"(model={model_key}, dur={metrics['duration_sec']:.2f}s, rms={metrics['rms']}, peak={metrics['peak']})"
        )
        raise ValueError("録音音声が極端に小さい/無音です。マイク入力（ノイズ抑制設定含む）を確認して、もう一度お試しください。")

    if is_gem:
        gemini_runtime = _resolve_gemini_runtime(user)
        g_key = _get_model_specific_api_key(user, model_key) or gemini_runtime.get("api_key")
        backend = gemini_runtime.get("backend")
        if backend == "vertex_ai":
            if not gemini_runtime.get("vertex_project"):
                raise ValueError("Vertex AI Project ID が未設定です。Gemini設定を確認してください。")
        elif not g_key:
            raise ValueError("Gemini API Key not configured")
        g_client = _get_gemini_client(
            api_key=g_key,
            backend=backend,
            vertex_project=gemini_runtime.get("vertex_project"),
            vertex_location=gemini_runtime.get("vertex_location"),
            vertex_credentials_json=gemini_runtime.get("vertex_credentials_json"),
        )
        if not g_client:
            raise RuntimeError("Gemini client initialization failed")
        resp = g_client.models.generate_content(
            model=model_key,
            contents=[
                types.Part(text=transcription_prompt),
                types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
            ],
        )
        text_out = (getattr(resp, "text", None) or "").strip()
        if _llm_transcript_is_no_speech(text_out, no_speech_token):
            raise ValueError("音声を検出できませんでした。マイク入力（ノイズ抑制設定含む）を確認して、もう一度お試しください。")
        return text_out

    o_key = _get_model_specific_api_key(user, model_key) or decrypt_val(user.openai_api_key)
    if not o_key and _admin_env_fallback_enabled(user):
        o_key = os.getenv('OPENAI_API_KEY')
    if not o_key:
        raise ValueError("OpenAI API Key not configured")
    client = _get_openai_client(o_key, base_url=None)
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
    resp = client.responses.create(
        model=model_key,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": transcription_prompt},
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}
                ]
            }
        ]
    )
    text_out = _extract_openai_response_text(resp).strip()
    if _llm_transcript_is_no_speech(text_out, no_speech_token):
        raise ValueError("音声を検出できませんでした。マイク入力（ノイズ抑制設定含む）を確認して、もう一度お試しください。")
    return text_out

async def _openai_sts_realtime(pcm_bytes, api_key, model_key, voice="alloy", speed=None, rate=24000):
    # OpenAI Realtime currently supports 24kHz PCM audio for output; keep session aligned.
    rate = 24000
    if model_key == "gpt-realtime-translate":
        url = f"wss://api.openai.com/v1/realtime/translations?model={model_key}"
    elif model_key == "gpt-realtime-whisper":
        url = f"wss://api.openai.com/v1/realtime/transcription_sessions?model={model_key}"
    else:
        url = f"wss://api.openai.com/v1/realtime?model={model_key}"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1",
    }
    audio_out = bytearray()
    transcript_out = ""
    async with websockets.connect(url, additional_headers=headers, max_size=None) as ws:
        session_update = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": model_key,
                "output_modalities": ["audio"],
                "voice": voice,
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": rate},
                        "turn_detection": None
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": rate}
                    }
                }
            }
        }
        if speed is not None:
            session_update["session"]["speed"] = speed
        await ws.send(json.dumps(session_update))
        await ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
        for chunk in _chunk_bytes(pcm_bytes):
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode('utf-8')
            }))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        resp_cfg = {"voice": voice}
        if speed is not None:
            resp_cfg["speed"] = speed
        await ws.send(json.dumps({"type": "response.create", "response": resp_cfg}))
        while True:
            msg = json.loads(await ws.recv())
            mtype = msg.get("type")
            if mtype == "error":
                logger.error(f"OpenAI STS error event: {msg}")
            elif mtype and mtype.startswith("response."):
                logger.debug(f"OpenAI STS event: {mtype}")
            if mtype in ("response.output_audio.delta", "response.audio.delta"):
                delta = msg.get("delta")
                if delta:
                    audio_out += base64.b64decode(delta)
            elif mtype in ("response.output_audio", "response.audio"):
                delta = msg.get("audio") or msg.get("data")
                if delta:
                    audio_out += base64.b64decode(delta)
            elif mtype == "response.output_audio_transcript.delta":
                delta = msg.get("delta")
                if delta:
                    transcript_out += delta
            elif mtype in ("response.output_audio.done", "response.done"):
                break
    return bytes(audio_out), transcript_out

async def _openai_realtime_transcribe(pcm_bytes, api_key, model_key, rate=24000):
    """Transcribe one committed PCM turn through an OpenAI Realtime transcription session."""
    if model_key not in {"gpt-transcribe", "gpt-live-transcribe"}:
        raise ValueError("Unsupported OpenAI transcription model")

    rate = 24000
    # Transcription models are selected inside session.audio.input.transcription;
    # they are not Realtime conversation models for the WebSocket query string.
    url = "wss://api.openai.com/v1/realtime?intent=transcription"
    headers = {"Authorization": f"Bearer {api_key}"}
    transcript_deltas = []

    async with websockets.connect(url, additional_headers=headers, max_size=None) as ws:
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": rate},
                        "transcription": {"model": model_key},
                        "turn_detection": None,
                    }
                },
            },
        }))
        await ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
        for chunk in _chunk_bytes(pcm_bytes):
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode("ascii"),
            }))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=60)
            event = json.loads(raw)
            event_type = event.get("type")
            if event_type == "conversation.item.input_audio_transcription.delta":
                delta = event.get("delta")
                if delta:
                    transcript_deltas.append(str(delta))
            elif event_type == "conversation.item.input_audio_transcription.completed":
                completed = str(event.get("transcript") or "").strip()
                return completed or "".join(transcript_deltas).strip()
            elif event_type == "conversation.item.input_audio_transcription.failed":
                error = event.get("error") or {}
                message = error.get("message") if isinstance(error, dict) else str(error)
                raise RuntimeError(message or "Realtime transcription failed")
            elif event_type == "error":
                error = event.get("error") or {}
                message = error.get("message") if isinstance(error, dict) else str(error)
                raise RuntimeError(message or "OpenAI Realtime API error")

async def _xai_sts_realtime(pcm_bytes, api_key, model_key="grok-voice-agent", voice="Ara", rate_in=24000, rate_out=24000):
    model_key = XAI_STS_MODEL_ALIASES.get(model_key, model_key)
    url = f"wss://{_XAI_API_HOST}/v1/realtime?model={model_key}"
    headers = {"Authorization": f"Bearer {api_key}"}
    audio_out = bytearray()
    transcript_out = ""
    async with websockets.connect(url, ssl=True, additional_headers=headers, max_size=None) as ws:
        session_update = {
            "type": "session.update",
            "session": {
                "voice": voice,
                "turn_detection": {"type": None},
                "audio": {
                    "input": {"format": {"type": "audio/pcm", "rate": rate_in}},
                    "output": {"format": {"type": "audio/pcm", "rate": rate_out}}
                }
            }
        }
        await ws.send(json.dumps(session_update))
        for chunk in _chunk_bytes(pcm_bytes):
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode('utf-8')
            }))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        # Wait for commit confirmation when using client-side VAD
        try:
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
                if msg.get("type") == "input_audio_buffer.committed":
                    break
        except Exception:
            pass

        await ws.send(json.dumps({"type": "response.create", "response": {"modalities": ["audio", "text"]}}))
        while True:
            msg = json.loads(await ws.recv())
            mtype = msg.get("type")
            if mtype == "error":
                logger.error(f"xAI STS error event: {msg}")
            elif mtype and mtype.startswith("response."):
                logger.debug(f"xAI STS event: {mtype}")
            if mtype == "response.output_audio.delta":
                delta = msg.get("delta")
                if delta:
                    audio_out += base64.b64decode(delta)
            elif mtype == "response.output_audio":
                delta = msg.get("audio")
                if delta:
                    audio_out += base64.b64decode(delta)
            elif mtype == "response.output_audio_transcript.delta":
                delta = msg.get("delta")
                if delta:
                    transcript_out += delta
            elif mtype in ("response.output_audio.done", "response.done"):
                break
    return bytes(audio_out), transcript_out

