@app.context_processor
def inject_csrf():
    is_admin = current_user.is_authenticated and bool(getattr(current_user, "is_admin", False))
    initial_theme_color = normalize_theme_color(getattr(current_user, 'theme_color', '')) if current_user.is_authenticated else ""
    initial_liquid_glass_enabled = bool(getattr(current_user, 'liquid_glass_enabled', False)) if current_user.is_authenticated else False
    return {
        'csrf_token': get_csrf_token(),
        'app_version': app.config.get('APP_VERSION'),
        'system_version': app.config.get('SYSTEM_VERSION'),
        'is_admin': is_admin,
        'attachment_max_files': app.config.get('ATTACHMENT_MAX_FILES', 30),
        'upload_concurrency': app.config.get('UPLOAD_CONCURRENCY', 3),
        'initial_theme_color': initial_theme_color,
        'initial_theme_css': build_theme_css_vars(initial_theme_color),
        'initial_liquid_glass_enabled': initial_liquid_glass_enabled,
    }

def validate_csrf():
    token = request.headers.get('X-CSRF-Token') or request.form.get('csrf_token')
    session_token = session.get('csrf_token')
    if not token or not session_token:
        return False
    try:
        return secrets.compare_digest(str(token), str(session_token))
    except Exception:
        return False

def get_app_setting(key, default=None):
    try:
        row = AppSetting.query.get(key)
        if row is None:
            return default
        return row.value
    except Exception:
        return default

def set_app_setting(key, value):
    row = AppSetting.query.get(key)
    if row is None:
        row = AppSetting(key=key, value=str(value))
        db.session.add(row)
    else:
        row.value = str(value)
    row.updated_at = datetime.utcnow()
    safe_db_commit()

def ensure_app_setting(key, default):
    try:
        row = AppSetting.query.get(key)
        if row is None:
            db.session.add(AppSetting(key=key, value=str(default)))
            safe_db_commit()
    except Exception:
        pass

def try_alter(sql):
    """
    Executes a raw SQL command.
    WARNING: This function executes raw SQL and is potentially vulnerable to SQL injection if used with untrusted input.
    It should ONLY be used for internal schema migrations with hardcoded SQL strings.
    """
    try:
        with db.engine.connect() as conn:
            conn.execute(text("SET SESSION lock_wait_timeout=1"))
            conn.execute(text(sql))
            conn.commit()
    except Exception:
        pass

def ensure_user_liquid_glass_column():
    """Ensure the login-critical Liquid Glass preference exists before any User query."""
    table_names = set(inspect(db.engine).get_table_names())
    if 'user' not in table_names:
        return
    column_names = {column['name'] for column in inspect(db.engine).get_columns('user')}
    if 'liquid_glass_enabled' not in column_names:
        with db.engine.begin() as conn:
            if db.engine.dialect.name in ('mysql', 'mariadb'):
                conn.execute(text("SET SESSION lock_wait_timeout=5"))
            conn.execute(text(
                "ALTER TABLE user ADD COLUMN liquid_glass_enabled BOOLEAN DEFAULT 0"
            ))
    verified_columns = {column['name'] for column in inspect(db.engine).get_columns('user')}
    if 'liquid_glass_enabled' not in verified_columns:
        raise RuntimeError("Required database column user.liquid_glass_enabled is missing")

def ensure_thread_last_model_column():
    try:
        with db.engine.connect() as conn:
            res = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME='thread' "
                "AND COLUMN_NAME='last_model'"
            )).scalar()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE thread ADD COLUMN last_model VARCHAR(64)"))
    except Exception:
        pass

def ensure_user_minashin_columns():
    """Add the Minashin SSO columns to the user table.

    These are required by the minashin login/settings queries and must exist
    before any authenticated request, so they are applied unconditionally at
    startup (like the other correctness-critical ensure_* migrations).
    """
    try:
        with db.engine.connect() as conn:
            res = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME='user' "
                "AND COLUMN_NAME='minashin_sub'"
            )).scalar()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE user ADD COLUMN minashin_sub VARCHAR(128) NULL"))
            res_email = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME='user' "
                "AND COLUMN_NAME='minashin_email'"
            )).scalar()
            if not res_email:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE user ADD COLUMN minashin_email VARCHAR(128) NULL"))
            try:
                conn.execute(text(
                    "CREATE UNIQUE INDEX ux_user_minashin_sub ON user (minashin_sub)"
                ))
            except Exception:
                # Index already exists (duplicate key on index name).
                pass
            conn.commit()
    except Exception:
        pass

def ensure_thread_temporary_column():
    try:
        with db.engine.connect() as conn:
            res = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME='thread' "
                "AND COLUMN_NAME='is_temporary'"
            )).scalar()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE thread ADD COLUMN is_temporary BOOLEAN DEFAULT 0"))
    except Exception:
        pass

def ensure_thread_prompt_caching_columns():
    try:
        with db.engine.connect() as conn:
            res = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME='thread' "
                "AND COLUMN_NAME='enable_prompt_caching'"
            )).scalar()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE thread ADD COLUMN enable_prompt_caching BOOLEAN DEFAULT 0"))
            res2 = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME='thread' "
                "AND COLUMN_NAME='prompt_cache_provider'"
            )).scalar()
            if not res2:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE thread ADD COLUMN prompt_cache_provider VARCHAR(32)"))
    except Exception:
        pass

def ensure_import_signature_columns():
    """Add the import dedupe columns used by account-data imports.

    These must exist before any import query touches them, so they are applied
    unconditionally at startup (like the other correctness-critical ensure_*
    migrations) rather than gated behind RUN_SCHEMA_MIGRATIONS.
    """
    candidates = [
        ("thread", "import_signature", "ALTER TABLE thread ADD COLUMN import_signature VARCHAR(64)"),
        ("gem", "import_signature", "ALTER TABLE gem ADD COLUMN import_signature VARCHAR(64)"),
        ("feedback", "import_signature", "ALTER TABLE feedback ADD COLUMN import_signature VARCHAR(64)"),
        ("file_cache", "import_signature", "ALTER TABLE file_cache ADD COLUMN import_signature VARCHAR(64)"),
        ("first_token_latency_metric", "import_signature", "ALTER TABLE first_token_latency_metric ADD COLUMN import_signature VARCHAR(64)"),
        ("chat_latency_trace", "import_signature", "ALTER TABLE chat_latency_trace ADD COLUMN import_signature VARCHAR(64)"),
    ]
    try:
        table_names = set(inspect(db.engine).get_table_names())
    except Exception:
        table_names = set()
    try:
        with db.engine.connect() as conn:
            for table, column, ddl in candidates:
                if table not in table_names:
                    continue
                try:
                    res = conn.execute(text(
                        "SELECT COUNT(*) FROM information_schema.COLUMNS "
                        "WHERE TABLE_SCHEMA=DATABASE() "
                        "AND TABLE_NAME=:tbl AND COLUMN_NAME=:col"
                    ), {"tbl": table, "col": column}).scalar()
                except Exception:
                    res = 0
                if not res:
                    try:
                        conn.execute(text("SET SESSION lock_wait_timeout=1"))
                        conn.execute(text(ddl))
                        conn.commit()
                    except Exception:
                        try:
                            conn.rollback()
                        except Exception:
                            pass
    except Exception:
        pass

def ensure_message_token_io_columns():
    try:
        with db.engine.connect() as conn:
            res_in = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME='message' "
                "AND COLUMN_NAME='tokens_in'"
            )).scalar()
            if not res_in:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE message ADD COLUMN tokens_in INTEGER DEFAULT 0"))
            res_out = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME='message' "
                "AND COLUMN_NAME='tokens_out'"
            )).scalar()
            if not res_out:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE message ADD COLUMN tokens_out INTEGER DEFAULT 0"))
            res_thought = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME='message' "
                "AND COLUMN_NAME='tokens_thought'"
            )).scalar()
            if not res_thought:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE message ADD COLUMN tokens_thought INTEGER DEFAULT 0"))
    except Exception:
        pass

def ensure_message_payload_longtext_columns():
    """Keep large encrypted messages and reasoning payloads above MySQL TEXT's 64 KiB limit."""
    if db.engine.dialect.name not in ('mysql', 'mariadb'):
        return

    required_columns = ('content', 'thought_data', 'quote_text', 'thought_signature')
    lock_name = 'ai_chat_message_payload_longtext_v1'
    with db.engine.connect() as conn:
        acquired = conn.execute(
            text("SELECT GET_LOCK(:lock_name, 30)"),
            {'lock_name': lock_name},
        ).scalar()
        if acquired != 1:
            raise RuntimeError("Could not acquire the message payload schema migration lock")
        try:
            rows = conn.execute(text(
                "SELECT COLUMN_NAME, DATA_TYPE FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='message' "
                "AND COLUMN_NAME IN ('content', 'thought_data', 'quote_text', 'thought_signature')"
            )).mappings().all()
            types_by_name = {row['COLUMN_NAME']: str(row['DATA_TYPE']).lower() for row in rows}
            missing = [name for name in required_columns if name not in types_by_name]
            if missing:
                raise RuntimeError(
                    "Required message payload columns are missing: " + ", ".join(missing)
                )
            for column_name in required_columns:
                if types_by_name[column_name] != 'longtext':
                    conn.execute(text(
                        f"ALTER TABLE message MODIFY COLUMN `{column_name}` LONGTEXT NULL"
                    ))

            verified = conn.execute(text(
                "SELECT COLUMN_NAME, DATA_TYPE FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='message' "
                "AND COLUMN_NAME IN ('content', 'thought_data', 'quote_text', 'thought_signature')"
            )).mappings().all()
            verified_types = {
                row['COLUMN_NAME']: str(row['DATA_TYPE']).lower() for row in verified
            }
            invalid = [
                name for name in required_columns if verified_types.get(name) != 'longtext'
            ]
            if invalid:
                raise RuntimeError(
                    "Required message payload columns are not LONGTEXT: " + ", ".join(invalid)
                )
        finally:
            conn.execute(
                text("SELECT RELEASE_LOCK(:lock_name)"),
                {'lock_name': lock_name},
            )

def ensure_user_system_prompt_columns():
    try:
        with db.engine.connect() as conn:
            columns = [
                ("system_prompt_enabled", "ALTER TABLE user ADD COLUMN system_prompt_enabled BOOLEAN DEFAULT 1"),
                ("apply_global_system_prompt", "ALTER TABLE user ADD COLUMN apply_global_system_prompt BOOLEAN DEFAULT 1"),
                ("apply_auto_system_prompt_notices", "ALTER TABLE user ADD COLUMN apply_auto_system_prompt_notices BOOLEAN DEFAULT 1"),
                ("auto_system_prompt_notices_config", "ALTER TABLE user ADD COLUMN auto_system_prompt_notices_config TEXT"),
            ]
            for column_name, ddl in columns:
                res = conn.execute(text(
                    "SELECT COUNT(*) FROM information_schema.COLUMNS "
                    "WHERE TABLE_SCHEMA=DATABASE() "
                    "AND TABLE_NAME='user' "
                    "AND COLUMN_NAME=:column_name"
                ), {"column_name": column_name}).scalar()
                if not res:
                    conn.execute(text("SET SESSION lock_wait_timeout=1"))
                    conn.execute(text(ddl))
    except Exception:
        pass

def ensure_user_file_creation_columns():
    """Add the create_file tool default / last-used columns to the user table.

    The chat request flow reads these columns on every authenticated User SELECT,
    so they must exist before any request.  Applied unconditionally at startup
    (like the other correctness-critical ensure_* migrations) rather than gated
    behind RUN_SCHEMA_MIGRATIONS.
    """
    try:
        with db.engine.connect() as conn:
            columns = [
                ("default_enable_file_creation", "ALTER TABLE user ADD COLUMN default_enable_file_creation BOOLEAN DEFAULT 1"),
                ("last_enable_file_creation", "ALTER TABLE user ADD COLUMN last_enable_file_creation BOOLEAN DEFAULT 1"),
            ]
            for column_name, ddl in columns:
                res = conn.execute(text(
                    "SELECT COUNT(*) FROM information_schema.COLUMNS "
                    "WHERE TABLE_SCHEMA=DATABASE() "
                    "AND TABLE_NAME='user' "
                    "AND COLUMN_NAME=:column_name"
                ), {"column_name": column_name}).scalar()
                if not res:
                    conn.execute(text("SET SESSION lock_wait_timeout=1"))
                    conn.execute(text(ddl))
            conn.commit()
    except Exception:
        pass

def ensure_user_mcp_enable_columns():
    """Add the MCP default / last-used toggle columns to the user table.

    The chat request flow reads these columns on every authenticated User SELECT
    and on every send, so they must exist before any request.  Applied
    unconditionally at startup (like the other correctness-critical ensure_*
    migrations) rather than gated behind RUN_SCHEMA_MIGRATIONS.
    """
    try:
        with db.engine.connect() as conn:
            columns = [
                ("default_enable_mcp", "ALTER TABLE user ADD COLUMN default_enable_mcp BOOLEAN DEFAULT 1"),
                ("last_enable_mcp", "ALTER TABLE user ADD COLUMN last_enable_mcp BOOLEAN DEFAULT 1"),
            ]
            for column_name, ddl in columns:
                res = conn.execute(text(
                    "SELECT COUNT(*) FROM information_schema.COLUMNS "
                    "WHERE TABLE_SCHEMA=DATABASE() "
                    "AND TABLE_NAME='user' "
                    "AND COLUMN_NAME=:column_name"
                ), {"column_name": column_name}).scalar()
                if not res:
                    conn.execute(text("SET SESSION lock_wait_timeout=1"))
                    conn.execute(text(ddl))
            conn.commit()
    except Exception:
        pass

def ensure_user_gemini_backend_columns():
    try:
        with db.engine.connect() as conn:
            columns = [
                ("gemini_backend", "ALTER TABLE user ADD COLUMN gemini_backend VARCHAR(24) DEFAULT 'gemini_api'"),
                ("gemini_vertex_project", "ALTER TABLE user ADD COLUMN gemini_vertex_project TEXT"),
                ("gemini_vertex_location", "ALTER TABLE user ADD COLUMN gemini_vertex_location VARCHAR(64) DEFAULT 'global'"),
                ("gemini_vertex_credentials_json", "ALTER TABLE user ADD COLUMN gemini_vertex_credentials_json TEXT"),
            ]
            for column_name, ddl in columns:
                res = conn.execute(text(
                    "SELECT COUNT(*) FROM information_schema.COLUMNS "
                    "WHERE TABLE_SCHEMA=DATABASE() "
                    "AND TABLE_NAME='user' "
                    "AND COLUMN_NAME=:column_name"
                ), {"column_name": column_name}).scalar()
                if not res:
                    conn.execute(text("SET SESSION lock_wait_timeout=1"))
                    conn.execute(text(ddl))
    except Exception:
        pass

def ensure_user_deepseek_api_key_column():
    try:
        with db.engine.connect() as conn:
            res = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME='user' "
                "AND COLUMN_NAME='deepseek_api_key'"
            )).scalar()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE user ADD COLUMN deepseek_api_key TEXT"))
    except Exception:
        pass

def ensure_user_kimi_api_key_column():
    try:
        with db.engine.connect() as conn:
            res = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME='user' "
                "AND COLUMN_NAME='kimi_api_key'"
            )).scalar()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE user ADD COLUMN kimi_api_key TEXT"))
    except Exception:
        pass

def ensure_user_mistral_api_key_column():
    try:
        with db.engine.connect() as conn:
            res = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME='user' "
                "AND COLUMN_NAME='mistral_api_key'"
            )).scalar()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE user ADD COLUMN mistral_api_key TEXT"))
    except Exception:
        pass

def ensure_user_anthropic_api_key_column():
    try:
        with db.engine.connect() as conn:
            res = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME='user' "
                "AND COLUMN_NAME='anthropic_api_key'"
            )).scalar()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE user ADD COLUMN anthropic_api_key TEXT"))
    except Exception:
        pass

def ensure_user_2fa_default_columns():
    try:
        with db.engine.connect() as conn:
            columns = [
                ("skip_2fa_on_google_login", "ALTER TABLE user ADD COLUMN skip_2fa_on_google_login BOOLEAN DEFAULT 0"),
                ("default_2fa_method", "ALTER TABLE user ADD COLUMN default_2fa_method VARCHAR(16) DEFAULT 'totp'"),
            ]
            for column_name, ddl in columns:
                res = conn.execute(text(
                    "SELECT COUNT(*) FROM information_schema.COLUMNS "
                    "WHERE TABLE_SCHEMA=DATABASE() "
                    "AND TABLE_NAME='user' "
                    "AND COLUMN_NAME=:column_name"
                ), {"column_name": column_name}).scalar()
                if not res:
                    conn.execute(text("SET SESSION lock_wait_timeout=1"))
                    conn.execute(text(ddl))
    except Exception:
        pass

def ensure_user_admin_api_key_mode_column():
    try:
        with db.engine.connect() as conn:
            res = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME='user' "
                "AND COLUMN_NAME='admin_api_key_mode'"
            )).scalar()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE user ADD COLUMN admin_api_key_mode VARCHAR(24) DEFAULT 'env_fallback'"))
    except Exception:
        pass

def ensure_user_model_api_keys_column():
    try:
        with db.engine.connect() as conn:
            res = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME='user' "
                "AND COLUMN_NAME='model_api_keys'"
            )).scalar()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE user ADD COLUMN model_api_keys TEXT"))
    except Exception:
        pass

def ensure_user_temp_chat_timeout_column():
    try:
        with db.engine.connect() as conn:
            res = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME='user' "
                "AND COLUMN_NAME='temp_chat_timeout_seconds'"
            )).scalar()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text(
                    f"ALTER TABLE user ADD COLUMN temp_chat_timeout_seconds INTEGER DEFAULT {_TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS}"
                ))
    except Exception:
        pass

def ensure_user_compact_prompt_mode_column():
    try:
        with db.engine.connect() as conn:
            res = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME='user' "
                "AND COLUMN_NAME='compact_prompt_mode'"
            )).scalar()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE user ADD COLUMN compact_prompt_mode BOOLEAN DEFAULT 0"))
    except Exception:
        pass

def ensure_user_minimal_prompt_mode_column():
    try:
        with db.engine.connect() as conn:
            res = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME='user' "
                "AND COLUMN_NAME='minimal_prompt_mode'"
            )).scalar()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE user ADD COLUMN minimal_prompt_mode BOOLEAN DEFAULT 0"))
    except Exception:
        pass

def ensure_user_voice_studio_ui_column():
    try:
        with db.engine.connect() as conn:
            res = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME='user' "
                "AND COLUMN_NAME='voice_studio_ui'"
            )).scalar()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE user ADD COLUMN voice_studio_ui BOOLEAN DEFAULT 1"))
    except Exception:
        pass

def ensure_gem_fixed_prompts_column():
    try:
        from sqlalchemy import text
        db.session.execute(text("ALTER TABLE gem ADD COLUMN fixed_prompts_json TEXT"))
        db.session.commit()
        logger.info("Column fixed_prompts_json added to gem table.")
    except Exception:
        db.session.rollback()


def ensure_gem_default_model_column():
    try:
        from sqlalchemy import text
        db.session.execute(text("ALTER TABLE gem ADD COLUMN default_model VARCHAR(64)"))
        db.session.commit()
        logger.info("Column default_model added to gem table.")
    except Exception:
        db.session.rollback()
def ensure_chat_latency_trace_columns():
    try:
        with db.engine.connect() as conn:
            columns = [
                ("client_done_at", "ALTER TABLE chat_latency_trace ADD COLUMN client_done_at DATETIME"),
                ("client_total_latency_ms", "ALTER TABLE chat_latency_trace ADD COLUMN client_total_latency_ms INTEGER"),
            ]
            for column_name, ddl in columns:
                res = conn.execute(text(
                    "SELECT COUNT(*) FROM information_schema.COLUMNS "
                    "WHERE TABLE_SCHEMA=DATABASE() "
                    "AND TABLE_NAME='chat_latency_trace' "
                    "AND COLUMN_NAME=:col"
                ), {"col": column_name}).scalar()
                if not res:
                    conn.execute(text("SET SESSION lock_wait_timeout=1"))
                    conn.execute(text(ddl))
    except Exception:
        pass

def ensure_user_stt_settings_columns():
    try:
        with db.engine.connect() as conn:
            columns = [
                ("mic_transcribe_mode", "ALTER TABLE user ADD COLUMN mic_transcribe_mode VARCHAR(16) DEFAULT 'stt_api'"),
                ("stt_model", "ALTER TABLE user ADD COLUMN stt_model VARCHAR(64)"),
                ("llm_transcribe_prompt", "ALTER TABLE user ADD COLUMN llm_transcribe_prompt TEXT"),
            ]
            for column_name, ddl in columns:
                res = conn.execute(text(
                    "SELECT COUNT(*) FROM information_schema.COLUMNS "
                    "WHERE TABLE_SCHEMA=DATABASE() "
                    "AND TABLE_NAME='user' "
                    "AND COLUMN_NAME=:column_name"
                ), {"column_name": column_name}).scalar()
                if not res:
                    conn.execute(text("SET SESSION lock_wait_timeout=1"))
                    conn.execute(text(ddl))
    except Exception:
        pass

def cleanup_user_temp_system_prompt_columns():
    try:
        with db.engine.connect() as conn:
            for column_name in ("temp_system_prompt", "temp_system_prompt_enabled"):
                res = conn.execute(text(
                    "SELECT COUNT(*) FROM information_schema.COLUMNS "
                    "WHERE TABLE_SCHEMA=DATABASE() "
                    "AND TABLE_NAME='user' "
                    "AND COLUMN_NAME=:column_name"
                ), {"column_name": column_name}).scalar()
                if res:
                    conn.execute(text("SET SESSION lock_wait_timeout=1"))
                    conn.execute(text(f"ALTER TABLE user DROP COLUMN {column_name}"))
    except Exception:
        pass

def ensure_user_debug_settings_columns():
    try:
        with db.engine.connect() as conn:
            res = conn.execute(text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='user' AND TABLE_SCHEMA=DATABASE() AND COLUMN_NAME='enable_client_debug_log'")).fetchone()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE user ADD COLUMN enable_client_debug_log BOOLEAN DEFAULT 0"))
                conn.commit()
    except Exception:
        pass

def ensure_bot_evidence_columns():
    try:
        with db.engine.connect() as conn:
            res = conn.execute(text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='user' AND TABLE_SCHEMA=DATABASE() AND COLUMN_NAME='bot_evidence'")).fetchone()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE user ADD COLUMN bot_evidence TEXT"))
                conn.commit()
    except Exception:
        pass
    try:
        with db.engine.connect() as conn:
            res = conn.execute(text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='ban_appeal' AND TABLE_SCHEMA=DATABASE() AND COLUMN_NAME='evidence'")).fetchone()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE ban_appeal ADD COLUMN evidence TEXT"))
                conn.commit()
    except Exception:
        pass

def ensure_user_cache_settings_columns():
    try:
        with db.engine.connect() as conn:
            res = conn.execute(text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='user' AND TABLE_SCHEMA=DATABASE() AND COLUMN_NAME='clear_cache_on_version_update'")).fetchone()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE user ADD COLUMN clear_cache_on_version_update BOOLEAN DEFAULT 0"))
                conn.commit()
    except Exception:
        pass

def ensure_user_default_model_columns():
    try:
        with db.engine.connect() as conn:
            # check default_model
            res = conn.execute(text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='user' AND TABLE_SCHEMA=DATABASE() AND COLUMN_NAME='default_model'")).fetchone()
            if not res:
                conn.execute(text("ALTER TABLE user ADD COLUMN default_model VARCHAR(64) DEFAULT 'gemini-3.6-flash'"))
                conn.commit()
            # check last_model
            res = conn.execute(text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='user' AND TABLE_SCHEMA=DATABASE() AND COLUMN_NAME='last_model'")).fetchone()
            if not res:
                conn.execute(text("ALTER TABLE user ADD COLUMN last_model VARCHAR(64)"))
                conn.commit()
            # The preview endpoint is retired. Preserve thread-level references
            # for history, but move any remaining user settings to the GA model.
            conn.execute(text(
                "UPDATE user SET default_model='gemini-3.1-flash-lite' "
                "WHERE default_model='gemini-3.1-flash-lite-preview'"
            ))
            conn.execute(text(
                "UPDATE user SET last_model='gemini-3.1-flash-lite' "
                "WHERE last_model='gemini-3.1-flash-lite-preview'"
            ))
            conn.commit()
            # check default_enable_url_context / default_enable_maps
            res = conn.execute(text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='user' AND TABLE_SCHEMA=DATABASE() AND COLUMN_NAME='default_enable_url_context'")).fetchone()
            if not res:
                conn.execute(text("ALTER TABLE user ADD COLUMN default_enable_url_context BOOLEAN DEFAULT 0"))
                conn.commit()
            res = conn.execute(text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='user' AND TABLE_SCHEMA=DATABASE() AND COLUMN_NAME='default_enable_maps'")).fetchone()
            if not res:
                conn.execute(text("ALTER TABLE user ADD COLUMN default_enable_maps BOOLEAN DEFAULT 0"))
                conn.commit()
            # check last_enable_url_context / last_enable_maps
            res = conn.execute(text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='user' AND TABLE_SCHEMA=DATABASE() AND COLUMN_NAME='last_enable_url_context'")).fetchone()
            if not res:
                conn.execute(text("ALTER TABLE user ADD COLUMN last_enable_url_context BOOLEAN DEFAULT 0"))
                conn.commit()
            res = conn.execute(text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='user' AND TABLE_SCHEMA=DATABASE() AND COLUMN_NAME='last_enable_maps'")).fetchone()
            if not res:
                conn.execute(text("ALTER TABLE user ADD COLUMN last_enable_maps BOOLEAN DEFAULT 0"))
                conn.commit()
            res = conn.execute(text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='user' AND TABLE_SCHEMA=DATABASE() AND COLUMN_NAME='rich_paste_prompt_default'")).fetchone()
            if not res:
                conn.execute(text("ALTER TABLE user ADD COLUMN rich_paste_prompt_default TEXT"))
                conn.commit()
            res = conn.execute(text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='user' AND TABLE_SCHEMA=DATABASE() AND COLUMN_NAME='rich_paste_prompt_use_custom_default'")).fetchone()
            if not res:
                conn.execute(text("ALTER TABLE user ADD COLUMN rich_paste_prompt_use_custom_default BOOLEAN DEFAULT 0"))
                conn.commit()
    except Exception as e:
        logger.error(f"Failed to ensure user default model columns: {e}")

def ensure_user_google_columns():
    try:
        with db.engine.connect() as conn:
            # google_id
            res = conn.execute(text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='user' AND TABLE_SCHEMA=DATABASE() AND COLUMN_NAME='google_id'")).fetchone()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE user ADD COLUMN google_id VARCHAR(128) UNIQUE"))
                conn.execute(text("ALTER TABLE user ADD INDEX idx_user_google_id (google_id)"))
                conn.commit()
            # google_email
            res = conn.execute(text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='user' AND TABLE_SCHEMA=DATABASE() AND COLUMN_NAME='google_email'")).fetchone()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE user ADD COLUMN google_email VARCHAR(128)"))
                conn.commit()
    except Exception:
        pass

def ensure_db_index(table_name, index_name, ddl):
    try:
        with db.engine.connect() as conn:
            res = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.STATISTICS "
                "WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME=:table_name "
                "AND INDEX_NAME=:index_name"
            ), {
                "table_name": table_name,
                "index_name": index_name
            }).scalar()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text(ddl))
    except Exception:
        pass

def ensure_performance_indexes():
    ensure_db_index(
        "thread",
        "idx_thread_public_id",
        "CREATE INDEX idx_thread_public_id ON thread (public_id)"
    )
    ensure_db_index(
        "thread",
        "idx_thread_user_bookmark_updated",
        "CREATE INDEX idx_thread_user_bookmark_updated ON thread (user_id, is_bookmarked, bookmarked_at, updated_at)"
    )
    ensure_db_index(
        "message",
        "idx_message_thread_ts_id",
        "CREATE INDEX idx_message_thread_ts_id ON message (thread_id, timestamp, id)"
    )
    ensure_db_index(
        "message",
        "idx_message_thread_id",
        "CREATE INDEX idx_message_thread_id ON message (thread_id, id)"
    )

def ensure_user_vision_model_columns():
    try:
        with db.engine.connect() as conn:
            res = conn.execute(text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='user' AND TABLE_SCHEMA=DATABASE() AND COLUMN_NAME='default_vision_model'")).fetchone()
            if not res:
                conn.execute(text("SET SESSION lock_wait_timeout=1"))
                conn.execute(text("ALTER TABLE user ADD COLUMN default_vision_model VARCHAR(64) DEFAULT 'gemini-3-flash-preview'"))
                conn.commit()
    except Exception:
        pass

