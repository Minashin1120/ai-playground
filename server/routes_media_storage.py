"""Storage, latency metric and client log routes extracted from ``routes_media.py``.

``/api/storage``, ``/api/metrics/first_token`` and ``/api/client_log``.
Loaded into the shared ``app`` namespace by ``_SERVER_PARTS``, after
``routes_media_upload.py``.
"""

# region: Storage API and upload size limit
@app.route('/api/storage', methods=['GET'])
@login_required
def get_storage_usage():
    limit = _get_user_storage_limit_bytes(current_user)
    used = _get_user_storage_usage_bytes(current_user.id)
    if limit is None:
        limit = 0
    return jsonify({
        'used_bytes': used,
        'limit_bytes': limit,
        'used_mb': _bytes_to_mb_str(used),
        'limit_mb': _bytes_to_mb_str(limit) if limit else 'unlimited',
        'is_unlimited': limit == 0
    })

@app.errorhandler(RequestEntityTooLarge)
def handle_upload_too_large(e):
    limit = getattr(request, 'max_content_length', None) or app.config.get('MAX_CONTENT_LENGTH')
    if not limit:
        return jsonify({'error': 'File too large. The server rejected the upload.'}), 413
    limit_mb = limit // (1024 * 1024)
    return jsonify({'error': f'File too large. Max {limit_mb}MB'}), 413

# region: Startup schema preparation
with app.app_context():
    # Rename the legacy gem table/columns to skill before create_all() so a new
    # empty ``skill`` table cannot be created alongside the populated one.
    ensure_skill_schema()
    try:
        db.create_all()
    except Exception as e:
        try:
            logger.error(f"db.create_all failed: {e}")
        except Exception:
            pass
    # Run column additions BEFORE any model query to avoid SQLAlchemy metadata cache staleness
    # This column is required by every authenticated User SELECT, so it must not depend on
    # RUN_SCHEMA_MIGRATIONS. Fail startup clearly instead of serving HTTP 500 to all sessions.
    ensure_user_liquid_glass_column()
    # Model output, reasoning, and Fernet ciphertext can exceed MySQL TEXT's 64 KiB
    # limit. This is correctness-critical and must run even when optional migrations
    # are disabled.
    ensure_message_payload_longtext_columns()
    try:
        try_alter("ALTER TABLE skill ADD COLUMN uuid VARCHAR(36)")
    except: pass
    try:
        import uuid as _uuid_backfill
        for skill in Skill.query.filter(Skill.uuid.is_(None)).all():
            skill.uuid = str(_uuid_backfill.uuid4())
        safe_db_commit()
    except: pass
    try:
        ensure_thread_last_model_column()
    except Exception:
        pass
    try:
        ensure_thread_temporary_column()
    except Exception:
        pass
    try:
        ensure_thread_prompt_caching_columns()
    except Exception:
        pass
    try:
        ensure_import_signature_columns()
    except Exception:
        pass
    try:
        ensure_feedback_agent_column()
    except Exception:
        pass
    try:
        ensure_ban_appeal_outlives_account()
    except Exception:
        pass
    try:
        ensure_feedback_public_ids()
    except Exception:
        pass
    try:
        ensure_user_minashin_columns()
    except Exception:
        pass
    try:
        ensure_message_token_io_columns()
    except Exception:
        pass
    try:
        ensure_message_prompt_options_column()
    except Exception:
        pass
    try:
        ensure_message_job_id_column()
    except Exception:
        pass
    try:
        ensure_user_system_prompt_columns()
    except Exception:
        pass
    try:
        ensure_user_file_creation_columns()
    except Exception:
        pass
    try:
        ensure_user_mcp_enable_columns()
    except Exception:
        pass
    try:
        ensure_user_gemini_backend_columns()
    except Exception:
        pass
    try:
        ensure_user_deepseek_api_key_column()
    except Exception:
        pass
    try:
        ensure_user_kimi_api_key_column()
    except Exception:
        pass
    try:
        ensure_user_mistral_api_key_column()
    except Exception:
        pass
    try:
        ensure_user_ideogram_api_key_column()
    except Exception:
        pass
    try:
        ensure_user_zai_api_key_column()
    except Exception:
        pass
    try:
        ensure_user_anthropic_api_key_column()
    except Exception:
        pass
    try:
        ensure_user_admin_api_key_mode_column()
    except Exception:
        pass
    try:
        ensure_user_2fa_default_columns()
    except Exception:
        pass
    try:
        ensure_user_model_api_keys_column()
    except Exception:
        pass
    try:
        ensure_user_temp_chat_timeout_column()
    except Exception:
        pass
    try:
        ensure_user_compact_prompt_mode_column()
    except Exception:
        pass
    try:
        ensure_user_minimal_prompt_mode_column()
    except Exception:
        pass
    try:
        ensure_user_voice_studio_ui_column()
    except Exception:
        pass
    try:
        ensure_skill_fixed_prompts_column()
    except Exception:
        pass
    try:
        ensure_skill_default_model_column()
    except Exception:
        pass
    try:
        ensure_user_stt_settings_columns()
    except Exception:
        pass
    try:
        ensure_chat_latency_trace_columns()
    except Exception:
        pass
    try:
        ensure_user_debug_settings_columns()
    except Exception:
        pass
    try:
        ensure_bot_evidence_columns()
    except Exception:
        pass
    try:
        ensure_user_cache_settings_columns()
    except Exception:
        pass
    try:
        ensure_user_default_model_columns()
    except Exception:
        pass
    try:
        ensure_user_vision_model_columns()
    except Exception:
        pass
    try:
        ensure_batch_job_columns()
    except Exception:
        pass
    try:
        ensure_user_google_columns()
    except Exception:
        pass
    try:
        cleanup_user_temp_system_prompt_columns()
    except Exception:
        pass
    try:
        ensure_performance_indexes()
    except Exception:
        pass
    try:
        ensure_app_setting("bot_detection_global_enabled", "1")
    except Exception:
        pass
    try:
        # MCP外部連携のプリセットサーバー行を用意する（無ければ作成）
        from mcp_service.registry import get_or_create_presets
        get_or_create_presets()
    except Exception as _mcp_preset_err:
        try:
            log_force(f"MCP preset seeding failed: {_mcp_preset_err}")
        except Exception:
            pass
    try:
        admin_user = None
        primary_admin = _get_primary_admin_username()
        if primary_admin:
            admin_user = User.query.filter_by(username=primary_admin).first()
        if admin_user and not getattr(admin_user, "is_admin", False):
            admin_user.is_admin = True
            safe_db_commit()
    except Exception:
        pass
    if RUN_SCHEMA_MIGRATIONS:
        try:
            try_alter("ALTER TABLE user ADD COLUMN is_admin BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE message ADD COLUMN thought_signature TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE message ADD COLUMN tokens_in INTEGER DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE message ADD COLUMN tokens_out INTEGER DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE message ADD COLUMN tokens_thought INTEGER DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN enable_e2ee BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE message ADD COLUMN is_encrypted BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN enable_client_debug_log BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN enable_server_diagnostic_log BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE chat_latency_trace ADD COLUMN client_done_at DATETIME")
        except: pass
        try:
            try_alter("ALTER TABLE chat_latency_trace ADD COLUMN client_total_latency_ms INTEGER")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN xai_api_key TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN gemini_backend VARCHAR(24) DEFAULT 'gemini_api'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN gemini_vertex_project TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN gemini_vertex_location VARCHAR(64) DEFAULT 'global'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN gemini_vertex_credentials_json TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN gemini_free_api_key TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN deepseek_api_key TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN kimi_api_key TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN mistral_api_key TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN ideogram_api_key TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN zai_api_key TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN admin_api_key_mode VARCHAR(24) DEFAULT 'env_fallback'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN model_api_keys TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN is_2fa_enabled BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN totp_secret VARCHAR(255)")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN webauthn_credentials TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN passkey_only_login BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN skip_2fa_on_google_login BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_2fa_method VARCHAR(16) DEFAULT 'totp'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN mic_transcribe_mode VARCHAR(16) DEFAULT 'stt_api'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN stt_model VARCHAR(64)")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN llm_transcribe_prompt TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN enter_to_send BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN use_sw_cache BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN clear_cache_on_version_update BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN theme_color VARCHAR(16)")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN auto_search_on_links BOOLEAN DEFAULT 1")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN compact_prompt_mode BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN minimal_prompt_mode BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN use_last_chat_settings BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN voice_studio_ui BOOLEAN DEFAULT 1")
        except: pass
        try:
            try_alter(f"ALTER TABLE user ADD COLUMN temp_chat_timeout_seconds INTEGER DEFAULT {_TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS}")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_enable_search BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_gemini_free_tier BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_enable_mcp BOOLEAN DEFAULT 1")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_enable_url_context BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_enable_maps BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_enable_python BOOLEAN DEFAULT 1")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_enable_file_creation BOOLEAN DEFAULT 1")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_enable_thinking BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_thinking_level VARCHAR(16) DEFAULT 'high'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_thinking_budget INTEGER DEFAULT 4096")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_reasoning_effort VARCHAR(16) DEFAULT 'medium'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_enable_system_prompt BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN system_prompt_enabled BOOLEAN DEFAULT 1")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN apply_global_system_prompt BOOLEAN DEFAULT 1")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN apply_auto_system_prompt_notices BOOLEAN DEFAULT 1")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN auto_system_prompt_notices_config TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN default_safety_setting VARCHAR(16) DEFAULT 'default'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN rich_paste_prompt_default TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN rich_paste_prompt_use_custom_default BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_enable_search BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_gemini_free_tier BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_enable_mcp BOOLEAN DEFAULT 1")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_enable_url_context BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_enable_maps BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_enable_python BOOLEAN DEFAULT 1")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_enable_file_creation BOOLEAN DEFAULT 1")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_enable_thinking BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_thinking_level VARCHAR(16) DEFAULT 'high'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_thinking_budget INTEGER DEFAULT 4096")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_reasoning_effort VARCHAR(16) DEFAULT 'medium'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_enable_system_prompt BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN last_safety_setting VARCHAR(16) DEFAULT 'default'")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN google_api_key TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN google_cloud_project TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN easy_login_hash TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN easy_login_expires_at DATETIME")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN bot_detection_enabled BOOLEAN DEFAULT 1")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN is_bot_banned BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN bot_banned_at DATETIME")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN bot_ban_reason TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN bot_unbanned_at DATETIME")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN bot_unban_notice BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN appeal_blocked BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN appeal_block_reason TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE user ADD COLUMN appeal_blocked_at DATETIME")
        except: pass
        try:
            try_alter("ALTER TABLE ban_appeal ADD COLUMN admin_reply TEXT")
        except: pass
        try:
            try_alter("ALTER TABLE ban_appeal ADD COLUMN replied_at DATETIME")
        except: pass
        try:
            try_alter("ALTER TABLE user_client_token ADD COLUMN last_seen_at DATETIME")
        except: pass
        try:
            try_alter("ALTER TABLE user_client_token ADD COLUMN ip_address VARCHAR(64)")
        except: pass
        try:
            try_alter("ALTER TABLE thread ADD COLUMN is_bookmarked BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE thread ADD COLUMN bookmarked_at DATETIME")
        except: pass
        try:
            try_alter("ALTER TABLE thread ADD COLUMN public_id VARCHAR(64)")
        except: pass
        try:
            try_alter("ALTER TABLE thread ADD COLUMN last_model VARCHAR(64)")
        except: pass
        try:
            try_alter("ALTER TABLE thread ADD COLUMN is_temporary BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE thread ADD COLUMN enable_prompt_caching BOOLEAN DEFAULT 0")
        except: pass
        try:
            try_alter("ALTER TABLE thread ADD COLUMN prompt_cache_provider VARCHAR(32)")
        except: pass
        try:
            try_alter("ALTER TABLE thread ADD COLUMN prompt_cache_nonce VARCHAR(32)")
        except: pass

# region: First-token metric
@app.route('/api/metrics/first_token', methods=['POST'])
@login_required
def first_token_metric():
    if not rate_limit(f"rl:first_token_metric:user:{current_user.id}", 240, 60):
        return jsonify({'error': 'rate_limit'}), 429
    try:
        d = request.get_json(silent=True) or {}

        try:
            latency_seconds = float(d.get('latency_seconds'))
        except Exception:
            return jsonify({'error': 'latency_seconds is required'}), 400
        if latency_seconds != latency_seconds or latency_seconds < 0 or latency_seconds > 600:
            return jsonify({'error': 'latency_seconds out of range'}), 400

        latency_ms = _coerce_int_or_none(d.get('latency_ms'))
        if latency_ms is None:
            latency_ms = int(round(latency_seconds * 1000))
        if latency_ms < 0:
            latency_ms = 0

        thread_public_id = str(d.get('thread_id') or '').strip()[:64] or None
        job_id = str(d.get('job_id') or '').strip()[:64] or None
        model = str(d.get('model') or '').strip()[:80] or None
        first_event_type = str(d.get('first_event_type') or '').strip()[:32] or None

        client_sent_at = None
        client_sent_at_ms = _coerce_int_or_none(d.get('client_sent_at_ms'))
        if client_sent_at_ms is not None and 946684800000 <= client_sent_at_ms <= 4102444800000:
            client_sent_at = datetime.utcfromtimestamp(client_sent_at_ms / 1000.0)

        is_total = bool(d.get('is_total'))
        client_done_at_ms = _coerce_int_or_none(d.get('client_done_at_ms'))

        if is_total:
            log_force(f"RECEIVED-TOTAL-REPORT: job={job_id} model={model} latency_ms={latency_ms}")
            # If it's a total latency report, we primarily update the trace
            _upsert_chat_latency_trace(
                job_id=job_id,
                user_id=current_user.id,
                thread_public_id=thread_public_id,
                model=model,
                client_sent_at_ms=client_sent_at_ms,
                client_done_at_ms=client_done_at_ms,
                client_total_latency_ms=latency_ms
            )
            log_force(
                "TOTAL-LATENCY-METRIC: "
                f"user={current_user.id} "
                f"model={model or '-'} "
                f"thread={thread_public_id or '-'} "
                f"job={job_id or '-'} "
                f"total_seconds={latency_seconds:.3f} "
                f"client_done_at={client_done_at_ms or '-'}"
            )
            return jsonify({'status': 'ok', 'type': 'total'})

        row = FirstTokenLatencyMetric(
            user_id=current_user.id,
            thread_public_id=thread_public_id,
            job_id=job_id,
            model=model,
            first_event_type=first_event_type,
            latency_seconds=round(latency_seconds, 6),
            latency_ms=latency_ms,
            client_sent_at=client_sent_at,
            ip_address=get_client_ip(),
            user_agent=get_request_user_agent()
        )
        db.session.add(row)
        safe_db_commit()
        trace = _upsert_chat_latency_trace(
            job_id=job_id,
            user_id=current_user.id,
            thread_public_id=thread_public_id,
            model=model,
            client_sent_at_ms=client_sent_at_ms,
            client_first_event_type=first_event_type,
            client_first_latency_ms=latency_ms
        )

        window_start = datetime.utcnow() - timedelta(hours=24)
        stats = db.session.query(
            func.count(FirstTokenLatencyMetric.id),
            func.avg(FirstTokenLatencyMetric.latency_seconds),
            func.min(FirstTokenLatencyMetric.latency_seconds),
            func.max(FirstTokenLatencyMetric.latency_seconds)
        ).filter(
            FirstTokenLatencyMetric.user_id == current_user.id,
            FirstTokenLatencyMetric.created_at >= window_start
        ).first()
        stats_evt = None
        if first_event_type:
            stats_evt = db.session.query(
                func.count(FirstTokenLatencyMetric.id),
                func.avg(FirstTokenLatencyMetric.latency_seconds),
                func.min(FirstTokenLatencyMetric.latency_seconds),
                func.max(FirstTokenLatencyMetric.latency_seconds)
            ).filter(
                FirstTokenLatencyMetric.user_id == current_user.id,
                FirstTokenLatencyMetric.first_event_type == first_event_type,
                FirstTokenLatencyMetric.created_at >= window_start
            ).first()

        cnt = int((stats[0] or 0)) if stats else 0
        avg_s = float(stats[1]) if stats and stats[1] is not None else latency_seconds
        min_s = float(stats[2]) if stats and stats[2] is not None else latency_seconds
        max_s = float(stats[3]) if stats and stats[3] is not None else latency_seconds
        evt_cnt = int((stats_evt[0] or 0)) if stats_evt else 0
        evt_avg_s = float(stats_evt[1]) if stats_evt and stats_evt[1] is not None else latency_seconds
        evt_min_s = float(stats_evt[2]) if stats_evt and stats_evt[2] is not None else latency_seconds
        evt_max_s = float(stats_evt[3]) if stats_evt and stats_evt[3] is not None else latency_seconds
        phase_parts = []
        if trace:
            phase_candidates = {
                "client_to_route_ms": _trace_delta_ms(trace, "client_sent_at", "route_received_at"),
                "route_to_dispatch_ms": _trace_delta_ms(trace, "route_received_at", "route_dispatch_at"),
                "dispatch_to_worker_ms": _trace_delta_ms(trace, "route_dispatch_at", "worker_started_at"),
                "worker_to_provider_req_ms": _trace_delta_ms(trace, "worker_started_at", "provider_request_started_at"),
                "provider_req_to_first_chunk_ms": _trace_delta_ms(trace, "provider_request_started_at", "provider_first_chunk_at"),
                "provider_req_to_first_content_ms": _trace_delta_ms(trace, "provider_request_started_at", "provider_first_content_at"),
                "provider_content_to_client_ms": _trace_delta_ms(trace, "provider_first_content_at", "stream_first_content_to_client_at"),
                "route_to_client_content_ms": _trace_delta_ms(trace, "route_received_at", "stream_first_content_to_client_at"),
            }
            for key, val in phase_candidates.items():
                if val is not None:
                    phase_parts.append(f"{key}={val}")
        log_force(
            "FIRST-TOKEN-METRIC: "
            f"user={current_user.id} "
            f"model={model or '-'} "
            f"thread={thread_public_id or '-'} "
            f"job={job_id or '-'} "
            f"event={first_event_type or '-'} "
            f"seconds={latency_seconds:.3f} "
            f"window24h(count={cnt},avg={avg_s:.3f},min={min_s:.3f},max={max_s:.3f}) "
            f"event24h(count={evt_cnt},avg={evt_avg_s:.3f},min={evt_min_s:.3f},max={evt_max_s:.3f}) "
            f"path={getattr(trace, 'execution_path', '-') or '-'} "
            f"phases({','.join(phase_parts)})"
        )

        return jsonify({
            'status': 'ok',
            'latency_seconds': round(latency_seconds, 3),
            'window24h': {
                'count': cnt,
                'avg_seconds': round(avg_s, 3),
                'min_seconds': round(min_s, 3),
                'max_seconds': round(max_s, 3),
            },
            'event24h': {
                'event': first_event_type or None,
                'count': evt_cnt,
                'avg_seconds': round(evt_avg_s, 3),
                'min_seconds': round(evt_min_s, 3),
                'max_seconds': round(evt_max_s, 3),
            },
            'execution_path': getattr(trace, 'execution_path', None) if trace else None
        })
    except Exception as e:
        db.session.rollback()
        log_force(f"FIRST-TOKEN-METRIC-ERROR: user={getattr(current_user, 'id', 'unknown')} err={e}")
        return jsonify({'status': 'error'}), 500

# region: Client log
@app.route('/api/client_log', methods=['POST'])
@login_required
def client_log():
    if not getattr(current_user, 'enable_client_debug_log', False):
        return jsonify({'status': 'ignored', 'reason': 'disabled'}), 200
    if not rate_limit(f"rl:client_log:user:{current_user.id}", 60, 60):
        return jsonify({'error': 'rate_limit'}), 429
    try:
        d = request.get_json(silent=True) or {}
        level = str(d.get('level') or 'info').upper()
        if level not in {'DEBUG', 'INFO', 'WARNING', 'ERROR'}:
            level = 'INFO'
        _record_server_diagnostic(current_user.id, 'server.client_log')
        return jsonify({'status': 'ok'})
    except Exception:
        return jsonify({'status': 'error'}), 500
