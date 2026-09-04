class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)
    google_id = db.Column(db.String(128), unique=True, nullable=True, index=True)
    google_email = db.Column(db.String(128), nullable=True)
    minashin_sub = db.Column(db.String(128), unique=True, nullable=True, index=True)
    minashin_email = db.Column(db.String(128), nullable=True)
    is_admin = db.Column(db.Boolean, default=False)
    admin_api_key_mode = db.Column(db.String(24), default="env_fallback")
    password_hash = db.Column(db.String(255))
    system_prompt = db.Column(db.Text, default="")
    system_prompt_enabled = db.Column(db.Boolean, default=True)
    apply_global_system_prompt = db.Column(db.Boolean, default=True)
    apply_auto_system_prompt_notices = db.Column(db.Boolean, default=True)
    auto_system_prompt_notices_config = db.Column(db.Text, nullable=True)
    openai_api_key = db.Column(db.Text, nullable=True)
    gemini_api_key = db.Column(db.Text, nullable=True)
    anthropic_api_key = db.Column(db.Text, nullable=True)
    deepseek_api_key = db.Column(db.Text, nullable=True)
    kimi_api_key = db.Column(db.Text, nullable=True)
    mistral_api_key = db.Column(db.Text, nullable=True)
    model_api_keys = db.Column(db.Text, nullable=True)
    gemini_backend = db.Column(db.String(24), default="gemini_api")
    gemini_vertex_project = db.Column(db.Text, nullable=True)
    gemini_vertex_location = db.Column(db.String(64), default="global")
    gemini_vertex_credentials_json = db.Column(db.Text, nullable=True)
    xai_api_key = db.Column(db.Text, nullable=True)
    google_api_key = db.Column(db.Text, nullable=True)
    google_cloud_project = db.Column(db.Text, nullable=True)
    mic_transcribe_mode = db.Column(db.String(16), default="stt_api")
    stt_model = db.Column(db.String(64), default="gpt-4o-mini-transcribe")
    llm_transcribe_prompt = db.Column(db.Text, nullable=True)
    enter_to_send = db.Column(db.Boolean, default=False)
    use_sw_cache = db.Column(db.Boolean, default=False)
    clear_cache_on_version_update = db.Column(db.Boolean, default=False)
    theme_color = db.Column(db.String(16), default="")
    liquid_glass_enabled = db.Column(db.Boolean, default=False)
    auto_search_on_links = db.Column(db.Boolean, default=True)
    compact_prompt_mode = db.Column(db.Boolean, default=False)
    minimal_prompt_mode = db.Column(db.Boolean, default=False)
    use_last_chat_settings = db.Column(db.Boolean, default=False)
    voice_studio_ui = db.Column(db.Boolean, default=True)
    temp_chat_timeout_seconds = db.Column(db.Integer, default=_TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS)
    default_model = db.Column(db.String(64), default="gemini-3.6-flash")
    default_enable_search = db.Column(db.Boolean, default=False)
    default_enable_url_context = db.Column(db.Boolean, default=False)
    default_enable_maps = db.Column(db.Boolean, default=False)
    default_enable_python = db.Column(db.Boolean, default=True)
    default_enable_file_creation = db.Column(db.Boolean, default=True)
    default_enable_thinking = db.Column(db.Boolean, default=False)
    default_thinking_level = db.Column(db.String(16), default="high")
    default_thinking_budget = db.Column(db.Integer, default=4096)
    default_reasoning_effort = db.Column(db.String(16), default="medium")
    default_enable_system_prompt = db.Column(db.Boolean, default=False)
    default_safety_setting = db.Column(db.String(16), default="default")
    default_vision_model = db.Column(db.String(64), default="gemini-3-flash-preview")
    rich_paste_prompt_default = db.Column(db.Text, nullable=True)
    rich_paste_prompt_use_custom_default = db.Column(db.Boolean, default=False)
    last_model = db.Column(db.String(64), nullable=True)
    last_enable_search = db.Column(db.Boolean, default=False)
    last_enable_url_context = db.Column(db.Boolean, default=False)
    last_enable_maps = db.Column(db.Boolean, default=False)
    last_enable_python = db.Column(db.Boolean, default=True)
    last_enable_file_creation = db.Column(db.Boolean, default=True)
    last_enable_thinking = db.Column(db.Boolean, default=False)
    last_thinking_level = db.Column(db.String(16), default="high")
    last_thinking_budget = db.Column(db.Integer, default=4096)
    last_reasoning_effort = db.Column(db.String(16), default="medium")
    last_enable_system_prompt = db.Column(db.Boolean, default=False)
    last_safety_setting = db.Column(db.String(16), default="default")
    last_gem_uuid = db.Column(db.String(36), nullable=True)
    easy_login_hash = db.Column(db.Text, nullable=True)
    easy_login_expires_at = db.Column(db.DateTime, nullable=True)
    is_setup_completed = db.Column(db.Boolean, default=False)
    enable_e2ee = db.Column(db.Boolean, default=False)
    # 2FA Fields
    is_2fa_enabled = db.Column(db.Boolean, default=False)
    totp_secret = db.Column(db.String(255), nullable=True)  # Fernet-encrypted TOTP secret
    webauthn_credentials = db.Column(db.Text, nullable=True) # JSON list
    passkey_only_login = db.Column(db.Boolean, default=False)
    skip_2fa_on_google_login = db.Column(db.Boolean, default=False)
    default_2fa_method = db.Column(db.String(16), default='totp')
    bot_detection_enabled = db.Column(db.Boolean, default=True)
    is_bot_banned = db.Column(db.Boolean, default=False)
    bot_banned_at = db.Column(db.DateTime, nullable=True)
    bot_ban_reason = db.Column(db.Text, nullable=True)
    bot_unbanned_at = db.Column(db.DateTime, nullable=True)
    bot_unban_notice = db.Column(db.Boolean, default=False)
    bot_evidence = db.Column(db.Text, nullable=True)
    appeal_blocked = db.Column(db.Boolean, default=False)
    appeal_block_reason = db.Column(db.Text, nullable=True)
    appeal_blocked_at = db.Column(db.DateTime, nullable=True)
    enable_latency_metrics = db.Column(db.Boolean, default=False)
    enable_client_debug_log = db.Column(db.Boolean, default=False)
    threads = db.relationship('Thread', backref='user', lazy=True, cascade="all, delete-orphan")
    gems = db.relationship('Gem', backref='user', lazy=True, cascade="all, delete-orphan")
    sessions = db.relationship('UserSession', backref='user', lazy=True, cascade="all, delete-orphan")
    def set_password(self, password): self.password_hash = generate_password_hash(password)
    def check_password(self, password): return check_password_hash(self.password_hash, password)

class UserSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    session_id = db.Column(db.String(128), unique=True, index=True, nullable=False)
    user_agent = db.Column(db.Text, nullable=True)
    ip_address = db.Column(db.String(64), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_seen_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_revoked = db.Column(db.Boolean, default=False)
    revoked_at = db.Column(db.DateTime, nullable=True)

class UserClientToken(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    token = db.Column(db.String(128), index=True, nullable=False)
    ip_address = db.Column(db.String(64), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_seen_at = db.Column(db.DateTime, default=datetime.utcnow)

class BannedIdentifier(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    kind = db.Column(db.String(16), index=True, nullable=False)  # ip / cookie
    value = db.Column(db.String(255), index=True, nullable=False)
    reason = db.Column(db.Text, nullable=True)
    source_user_id = db.Column(db.Integer, nullable=True)
    source_username = db.Column(db.String(80), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Thread(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(64), unique=True, index=True, nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), default="New Chat")
    is_bookmarked = db.Column(db.Boolean, default=False)
    bookmarked_at = db.Column(db.DateTime, nullable=True)
    is_temporary = db.Column(db.Boolean, default=False)
    custom_instruction = db.Column(db.Text, nullable=True)
    include_global_instruction = db.Column(db.Boolean, default=True)
    last_model = db.Column(db.String(64), nullable=True)
    last_gem_uuid = db.Column(db.String(36), nullable=True)
    enable_prompt_caching = db.Column(db.Boolean, default=False)
    prompt_cache_provider = db.Column(db.String(32), nullable=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)
    # Stable per-user identity of the source record this thread was imported
    # from (e.g. "thread:<source-public-id>").  Used to detect duplicate
    # imports instead of relying on the globally-unique public_id.
    import_signature = db.Column(db.String(64), nullable=True, index=True)
    messages = db.relationship('Message', backref='thread', cascade="all, delete-orphan", lazy=True)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    thread_id = db.Column(db.Integer, db.ForeignKey('thread.id'), nullable=False, index=True)
    role = db.Column(db.String(20))
    content = db.Column(MESSAGE_PAYLOAD_TEXT)
    model = db.Column(db.String(50))
    image_url = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    tokens = db.Column(db.Integer, default=0)
    tokens_in = db.Column(db.Integer, default=0)
    tokens_out = db.Column(db.Integer, default=0)
    tokens_thought = db.Column(db.Integer, default=0)
    thought_data = db.Column(MESSAGE_PAYLOAD_TEXT)
    quote_text = db.Column(MESSAGE_PAYLOAD_TEXT)
    is_encrypted = db.Column(db.Boolean, default=False)
    thought_signature = db.Column(MESSAGE_PAYLOAD_TEXT, nullable=True)
    gem_uuid = db.Column(db.String(36), nullable=True)
    gem_name = db.Column(db.String(100), nullable=True)
    parent_id = db.Column(db.Integer, db.ForeignKey('message.id'), nullable=True)
    children = db.relationship('Message', backref=db.backref('parent', remote_side=[id]), lazy=True)

class FileCache(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    rel_path = db.Column(db.Text, nullable=False, index=True)
    provider = db.Column(db.String(32), nullable=False, index=True)
    size_bytes = db.Column(db.Integer, nullable=True)
    mtime = db.Column(db.Integer, nullable=True)
    mime_type = db.Column(db.String(128), nullable=True)
    file_id = db.Column(db.String(256), nullable=True)
    file_uri = db.Column(db.Text, nullable=True)
    state = db.Column(db.String(32), default="unknown")
    last_error = db.Column(db.Text, nullable=True)
    retries = db.Column(db.Integer, default=0)
    last_checked_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # Content hash of the plaintext file (sha256 hex).  Set when a file is
    # imported so repeat imports can be detected and skipped.
    import_signature = db.Column(db.String(64), nullable=True, index=True)

class Gem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    uuid = db.Column(db.String(36), unique=True, index=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    instruction = db.Column(db.Text, nullable=False)
    fixed_prompts_json = db.Column(db.Text, nullable=True)
    default_model = db.Column(db.String(64), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    # Source identity of the imported gem (e.g. "gem:<source-uuid>") so repeat
    # imports of the same gem can be detected.
    import_signature = db.Column(db.String(64), nullable=True, index=True)

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), default="")
    message = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default="new")  # new, in_review, replied, rejected, resolved
    admin_reply = db.Column(db.Text, nullable=True)
    handled_by = db.Column(db.String(80), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)
    # Stable identity of the imported feedback row so repeat imports can be
    # detected (feedback has no natural key of its own).
    import_signature = db.Column(db.String(64), nullable=True, index=True)

class BanAppeal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    username = db.Column(db.String(80), nullable=False)
    message = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default="new")  # new, in_review, replied, resolved, rejected
    admin_note = db.Column(db.Text, nullable=True)
    admin_reply = db.Column(db.Text, nullable=True)
    admin_read_at = db.Column(db.DateTime, nullable=True)
    replied_at = db.Column(db.DateTime, nullable=True)
    handled_at = db.Column(db.DateTime, nullable=True)
    handled_by = db.Column(db.String(80), nullable=True)
    ban_reason = db.Column(db.Text, nullable=True)
    ban_at = db.Column(db.DateTime, nullable=True)
    ip_address = db.Column(db.String(64), nullable=True)
    user_agent = db.Column(db.Text, nullable=True)
    evidence = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

class BotEvidenceLog(db.Model):
    """Persistent log of bot-detection events for moderation and ban review."""
    __tablename__ = 'bot_evidence_log'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False, index=True)
    username = db.Column(db.String(80), nullable=True)
    event_type = db.Column(db.String(32), nullable=False)  # telemetry, turnstile_fail, verify_ok, verify_fail, ban
    score = db.Column(db.Float, nullable=True)
    behavior_score = db.Column(db.Float, nullable=True)
    reasons = db.Column(db.Text, nullable=True)
    details = db.Column(db.Text, nullable=True)
    ip_address = db.Column(db.String(64), nullable=True)
    user_agent = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

class AppSetting(db.Model):
    key = db.Column(db.String(64), primary_key=True)
    value = db.Column(db.Text, nullable=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

class FirstTokenLatencyMetric(db.Model):
    __tablename__ = 'first_token_latency_metric'
    __table_args__ = (
        db.Index('idx_ft_latency_user_created', 'user_id', 'created_at'),
        db.Index('idx_ft_latency_user_event_created', 'user_id', 'first_event_type', 'created_at'),
    )
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    thread_public_id = db.Column(db.String(64), nullable=True, index=True)
    job_id = db.Column(db.String(64), nullable=True, index=True)
    model = db.Column(db.String(80), nullable=True)
    first_event_type = db.Column(db.String(32), nullable=True)
    latency_seconds = db.Column(db.Float, nullable=False)
    latency_ms = db.Column(db.Integer, nullable=False)
    client_sent_at = db.Column(db.DateTime, nullable=True)
    ip_address = db.Column(db.String(64), nullable=True)
    user_agent = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    # Stable identity of the imported metric row so repeat imports can be
    # detected (job_id is regenerated on every import).
    import_signature = db.Column(db.String(64), nullable=True, index=True)

class ChatLatencyTrace(db.Model):
    __tablename__ = 'chat_latency_trace'
    __table_args__ = (
        db.UniqueConstraint('job_id', name='uq_chat_latency_trace_job_id'),
        db.Index('idx_chat_latency_trace_user_created', 'user_id', 'created_at'),
        db.Index('idx_chat_latency_trace_thread_created', 'thread_public_id', 'created_at'),
    )
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    thread_public_id = db.Column(db.String(64), nullable=True, index=True)
    job_id = db.Column(db.String(64), nullable=False, index=True)
    model = db.Column(db.String(80), nullable=True)
    execution_path = db.Column(db.String(24), nullable=True)
    client_sent_at = db.Column(db.DateTime, nullable=True)
    client_first_event_type = db.Column(db.String(32), nullable=True)
    client_first_latency_ms = db.Column(db.Integer, nullable=True)
    route_received_at = db.Column(db.DateTime, nullable=True)
    route_dispatch_at = db.Column(db.DateTime, nullable=True)
    route_stream_open_at = db.Column(db.DateTime, nullable=True)
    worker_started_at = db.Column(db.DateTime, nullable=True)
    provider_request_started_at = db.Column(db.DateTime, nullable=True)
    provider_first_chunk_at = db.Column(db.DateTime, nullable=True)
    provider_first_status_at = db.Column(db.DateTime, nullable=True)
    provider_first_thought_at = db.Column(db.DateTime, nullable=True)
    provider_first_content_at = db.Column(db.DateTime, nullable=True)
    stream_first_pubsub_at = db.Column(db.DateTime, nullable=True)
    stream_first_status_to_client_at = db.Column(db.DateTime, nullable=True)
    stream_first_thought_to_client_at = db.Column(db.DateTime, nullable=True)
    stream_first_content_to_client_at = db.Column(db.DateTime, nullable=True)
    stream_done_at = db.Column(db.DateTime, nullable=True)
    worker_done_at = db.Column(db.DateTime, nullable=True)
    client_done_at = db.Column(db.DateTime, nullable=True)
    client_total_latency_ms = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # Stable identity of the imported trace row so repeat imports can be
    # detected (job_id is regenerated on every import).
    import_signature = db.Column(db.String(64), nullable=True, index=True)


# MCP外部連携（mcp_service）のモデル定義を db.create_all() より前に登録する。
# mcp_service は app/mcp_service に置き、PyPI の公式 mcp SDK との名前衝突を避けている。
from mcp_service import models as _mcp_models  # noqa: E402,F401

