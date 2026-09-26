package com.minashin1120.aiplayground.data

import com.minashin1120.aiplayground.BuildConfig
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.io.InputStream

data class ThreadItem(
    val id: String,
    val title: String,
    val model: String,
    val isBookmarked: Boolean = false,
    val isTemporary: Boolean = false,
)
data class ChatMessage(val id: String, val role: String, val content: String,
                       val thought: String = "", val files: List<String> = emptyList(),
                       val parentId: Int? = null, val model: String = "",
                       /** Web message meta: token counts, encryption, quote and Gem shown in the bubble footer. */
                       val tokens: Int? = null, val tokensIn: Int? = null, val tokensOut: Int? = null,
                       val tokensContent: Int? = null, val tokensThought: Int? = null,
                       val encrypted: Boolean? = null, val quote: String = "", val gemName: String = "")
data class Attachment(val name: String, val reference: String, val mime: String = "")
data class ModelInfo(val id: String, val name: String, val provider: String, val providerLabel: String,
                     val mode: String, val capabilities: Set<String>, val deprecated: Boolean,
                     val selectable: Boolean,
                     val description: String = "", val price: String = "", val category: String = "",
                     val implementedAt: String = "", val implementedRank: Int = 0, val emoji: String = "",
                     val tags: Set<String> = emptySet(), val webCatalogOrder: Int = Int.MAX_VALUE,
                     /** Web `MODELS` group icon (`fas fa-star text-yellow-400`) and description, for the picker headers. */
                     val categoryIcon: String = "", val categoryDescription: String = "",
                     val apiId: String = "", val agenticView: Boolean = false,
                     /** Web `getModelCapabilitySearchTerms`: extra words the picker search matches. */
                     val searchTerms: List<String> = emptyList()) {
    fun supports(capability: String) = capability in capabilities
}
data class Account(val id: Int, val name: String, val models: List<ModelInfo>, val defaultModel: String,
                   val encrypted: Boolean)
data class StoredSession(val token: String, val expiresAt: Long)

data class LibraryFile(
    val displayName: String,
    val filepath: String,
    val url: String,
    val thumbnailUrl: String,
    val type: String,
    val ext: String,
    val isFavorite: Boolean,
    val timestamp: Long,
) {
    val isImage: Boolean get() = type == "image" || thumbnailUrl.isNotBlank()
}

data class FixedPrompt(val name: String, val content: String)

data class CodingTarget(val id: String, val code: String, val language: String, val messageId: String)

fun parseFixedPrompts(value: Any?): List<FixedPrompt> {
    val rows = when (value) {
        is JSONArray -> value
        is String -> runCatching { JSONArray(value) }.getOrNull()
        else -> null
    } ?: return emptyList()
    return (0 until rows.length()).mapNotNull { index ->
        val row = rows.optJSONObject(index) ?: return@mapNotNull null
        val name = row.nullableString("name").trim()
        val content = row.nullableString("content").trim()
        if (name.isBlank() || content.isBlank()) null else FixedPrompt(name, content)
    }
}

data class Gem(
    val uuid: String,
    val name: String,
    val description: String,
    val instruction: String,
    val defaultModel: String,
    val fixedPrompts: List<FixedPrompt> = emptyList(),
)

data class Preferences(
    val username: String,
    val defaultModel: String,
    val defaultEnableThinking: Boolean,
    val defaultEnableSearch: Boolean,
    val enterToSend: Boolean,
    val lightModeEnabled: Boolean,
    val autoSearchOnLinks: Boolean,
    val themeColor: String,
    val tempChatTimeoutSeconds: Int,
    val e2eeEnabled: Boolean,
    val twoFactorEnabled: Boolean,
    val hasTotp: Boolean,
    val hasWebauthn: Boolean,
    val sessionCreatedAt: String,
    val sessionExpiresAt: String,
    val deviceName: String,
    val defaultEnableUrlContext: Boolean = false,
    val defaultEnableMaps: Boolean = false,
    val defaultEnablePython: Boolean = false,
    val defaultEnableFileCreation: Boolean = true,
    val defaultEnableSystemPrompt: Boolean = false,
    val defaultEnableMcp: Boolean = true,
    val defaultThinkingLevel: String = "high",
    val defaultThinkingBudget: Int = 4096,
    val defaultReasoningEffort: String = "medium",
    val defaultSafetySetting: String = "default",
    val defaultVisionModel: String = "gemini-3-flash-preview",
    val useLastChatSettings: Boolean = false,
    val voiceStudioUi: Boolean = true,
    val liquidGlassEnabled: Boolean = false,
    val compactPromptMode: Boolean = false,
    val minimalPromptMode: Boolean = false,
    val promptBarMode: String = "normal",
    val micTranscribeMode: String = "stt_api",
    val sttModel: String = "gpt-4o-mini-transcribe",
    val systemPrompt: String = "",
    val systemPromptEnabled: Boolean = true,
    val applyGlobalSystemPrompt: Boolean = true,
    val applyAutoSystemPromptNotices: Boolean = true,
    val globalSystemPrompt: String = "",
    val globalSystemPromptEnabled: Boolean = true,
    val richPastePromptDefault: String = "",
    val richPastePromptUseCustomDefault: Boolean = false,
    val lastModel: String = "",
    val lastEnableSearch: Boolean = false,
    val lastEnableUrlContext: Boolean = false,
    val lastEnableMaps: Boolean = false,
    val lastEnablePython: Boolean = true,
    val lastEnableFileCreation: Boolean = true,
    val lastEnableThinking: Boolean = false,
    val lastThinkingLevel: String = "high",
    val lastThinkingBudget: Int = 4096,
    val lastReasoningEffort: String = "medium",
    val lastEnableSystemPrompt: Boolean = false,
    val lastEnableMcp: Boolean = true,
    val lastSafetySetting: String = "default",
    val skip2faOnGoogleLogin: Boolean = false,
    val default2faMethod: String = "totp",
    val googleEmail: String = "",
    val minashinEmail: String = "",
    val globalSystemPromptEffective: String = "",
    val globalSystemPromptUsesTimeFallback: Boolean = false,
    val autoSystemPrompts: List<AutoSystemPrompt> = emptyList(),
    val llmTranscribePrompt: String = "",
    val llmTranscribePromptDefault: String = "",
    val enableLatencyMetrics: Boolean = false,
    val enableClientDebugLog: Boolean = false,
    val passkeyOnlyLogin: Boolean = false,
    /** Provider key fields (`openai_key` …) mapped to the server mask; blank when not set. */
    val providerKeys: Map<String, String> = emptyMap(),
    /** Model ids with a model-specific key (values are never sent to the device). */
    val modelApiKeys: Set<String> = emptySet(),
    val geminiBackend: String = "gemini_api",
    val geminiVertexProject: String = "",
    val geminiVertexLocation: String = "global",
    val geminiVertexCredentialsSet: Boolean = false,
    val googleProject: String = "",
    val isAdmin: Boolean = false,
    val googleLinked: Boolean = false,
    val minashinLinked: Boolean = false,
    /** Web `migration_status` / `migration_progress` of the E2EE switch. */
    val migrationStatus: String = "idle",
    val migrationProgress: String = "",
) {
    val effectivePromptBarMode: String
        get() = when {
            promptBarMode in listOf("normal", "compact", "minimal") -> promptBarMode
            minimalPromptMode -> "minimal"
            compactPromptMode -> "compact"
            else -> "normal"
        }
}

data class McpServerInfo(
    val id: Int,
    val name: String,
    val enabled: Boolean,
    val connectionState: String,
    val authStatus: String,
    val toolCount: Int,
    val isPreset: Boolean,
    val description: String,
    val url: String = "",
    val authType: String = "none",
    val lastError: String = "",
    val oauthClientRegistered: Boolean = false,
    val authHasToken: Boolean = false,
)

data class FeedbackItem(
    val id: Int,
    val title: String,
    val message: String,
    val status: String,
    val adminReply: String,
    val createdAt: String,
)

data class StorageUsage(
    val usedBytes: Long,
    val limitBytes: Long,
    val usedMb: String,
    val limitMb: String,
    val unlimited: Boolean,
)

fun parsePreferences(json: JSONObject): Preferences = Preferences(
    username = json.nullableString("username"),
    defaultModel = json.nullableString("default_model"),
    defaultEnableThinking = json.optBoolean("default_enable_thinking"),
    defaultEnableSearch = json.optBoolean("default_enable_search"),
    enterToSend = json.optBoolean("enter_to_send"),
    lightModeEnabled = json.optBoolean("light_mode_enabled"),
    autoSearchOnLinks = json.optBoolean("auto_search_on_links", true),
    themeColor = json.nullableString("theme_color"),
    tempChatTimeoutSeconds = json.optInt("temp_chat_timeout_seconds", 90),
    e2eeEnabled = json.optBoolean("enable_e2ee"),
    twoFactorEnabled = json.optBoolean("is_2fa_enabled"),
    hasTotp = json.optBoolean("has_totp"),
    hasWebauthn = json.optBoolean("has_webauthn"),
    sessionCreatedAt = json.nullableString("session_created_at"),
    sessionExpiresAt = json.nullableString("session_expires_at"),
    deviceName = json.nullableString("device_name"),
    defaultEnableUrlContext = json.optBoolean("default_enable_url_context"),
    defaultEnableMaps = json.optBoolean("default_enable_maps"),
    defaultEnablePython = json.optBoolean("default_enable_python"),
    defaultEnableFileCreation = json.optBoolean("default_enable_file_creation", true),
    defaultEnableSystemPrompt = json.optBoolean("default_enable_system_prompt"),
    defaultEnableMcp = json.optBoolean("default_enable_mcp", true),
    defaultThinkingLevel = json.nullableString("default_thinking_level").ifBlank { "high" },
    defaultThinkingBudget = json.optInt("default_thinking_budget", 4096),
    defaultReasoningEffort = json.nullableString("default_reasoning_effort").ifBlank { "medium" },
    defaultSafetySetting = json.nullableString("default_safety_setting").ifBlank { "default" },
    defaultVisionModel = json.nullableString("default_vision_model").ifBlank { "gemini-3-flash-preview" },
    useLastChatSettings = json.optBoolean("use_last_chat_settings"),
    voiceStudioUi = json.optBoolean("voice_studio_ui", true),
    liquidGlassEnabled = json.optBoolean("liquid_glass_enabled"),
    compactPromptMode = json.optBoolean("compact_prompt_mode"),
    minimalPromptMode = json.optBoolean("minimal_prompt_mode"),
    promptBarMode = json.nullableString("prompt_bar_mode").ifBlank {
        when {
            json.optBoolean("minimal_prompt_mode") -> "minimal"
            json.optBoolean("compact_prompt_mode") -> "compact"
            else -> "normal"
        }
    },
    micTranscribeMode = json.nullableString("mic_transcribe_mode").ifBlank { "stt_api" },
    sttModel = json.nullableString("stt_model").ifBlank { "gpt-4o-mini-transcribe" },
    systemPrompt = json.nullableString("system_prompt"),
    systemPromptEnabled = json.optBoolean("system_prompt_enabled", true),
    applyGlobalSystemPrompt = json.optBoolean("apply_global_system_prompt", true),
    applyAutoSystemPromptNotices = json.optBoolean("apply_auto_system_prompt_notices", true),
    globalSystemPrompt = json.nullableString("global_system_prompt"),
    globalSystemPromptEnabled = json.optBoolean("global_system_prompt_enabled", true),
    richPastePromptDefault = json.nullableString("rich_paste_prompt_default"),
    richPastePromptUseCustomDefault = json.optBoolean("rich_paste_prompt_use_custom_default"),
    lastModel = json.nullableString("last_model"),
    lastEnableSearch = json.optBoolean("last_enable_search"),
    lastEnableUrlContext = json.optBoolean("last_enable_url_context"),
    lastEnableMaps = json.optBoolean("last_enable_maps"),
    lastEnablePython = json.optBoolean("last_enable_python", true),
    lastEnableFileCreation = json.optBoolean("last_enable_file_creation", true),
    lastEnableThinking = json.optBoolean("last_enable_thinking"),
    lastThinkingLevel = json.nullableString("last_thinking_level").ifBlank { "high" },
    lastThinkingBudget = json.optInt("last_thinking_budget", 4096),
    lastReasoningEffort = json.nullableString("last_reasoning_effort").ifBlank { "medium" },
    lastEnableSystemPrompt = json.optBoolean("last_enable_system_prompt"),
    lastEnableMcp = json.optBoolean("last_enable_mcp", true),
    lastSafetySetting = json.nullableString("last_safety_setting").ifBlank { "default" },
    skip2faOnGoogleLogin = json.optBoolean("skip_2fa_on_google_login"),
    default2faMethod = json.nullableString("default_2fa_method").ifBlank { "totp" },
    googleEmail = json.nullableString("google_email"),
    minashinEmail = json.nullableString("minashin_email"),
    globalSystemPromptEffective = json.nullableString("global_system_prompt_effective"),
    globalSystemPromptUsesTimeFallback = json.optBoolean("global_system_prompt_uses_time_fallback"),
    autoSystemPrompts = json.optJSONObject("auto_system_prompt_notices_config")?.let(::parseAutoSystemPrompts).orEmpty(),
    llmTranscribePrompt = json.nullableString("llm_transcribe_prompt"),
    llmTranscribePromptDefault = json.nullableString("llm_transcribe_prompt_default"),
    enableLatencyMetrics = json.optBoolean("enable_latency_metrics"),
    enableClientDebugLog = json.optBoolean("enable_client_debug_log"),
    passkeyOnlyLogin = json.optBoolean("passkey_only_login"),
    providerKeys = PROVIDER_KEY_FIELDS.associateWith { json.nullableString(it) },
    modelApiKeys = json.optJSONObject("model_api_keys")?.keys()?.asSequence()?.toSet().orEmpty(),
    geminiBackend = json.nullableString("gemini_backend").ifBlank { "gemini_api" },
    geminiVertexProject = json.nullableString("gemini_vertex_project"),
    geminiVertexLocation = json.nullableString("gemini_vertex_location").ifBlank { "global" },
    geminiVertexCredentialsSet = json.nullableString("gemini_vertex_credentials_json").isNotBlank(),
    googleProject = json.nullableString("google_project"),
    isAdmin = json.optBoolean("is_admin"),
    googleLinked = json.optBoolean("google_linked", json.nullableString("google_email").isNotBlank()),
    minashinLinked = json.optBoolean("minashin_linked", json.nullableString("minashin_email").isNotBlank()),
    migrationStatus = json.nullableString("migration_status").ifBlank { "idle" },
    migrationProgress = json.nullableString("migration_progress"),
)

/** Server mask for stored secrets (Web `_SECRET_MASK`): sending it back keeps the stored value. */
const val SECRET_MASK = "********"

val PROVIDER_KEY_FIELDS = listOf(
    "openai_key", "gemini_key", "deepseek_key", "kimi_key", "mistral_key", "anthropic_key", "xai_key", "google_key",
)

/** One row of the Web "自動注入システムプロンプト" list (`AUTO_SYS_PROMPT_ITEMS`). */
data class AutoSystemPrompt(
    val key: String, val label: String, val enabled: Boolean, val text: String, val defaultText: String,
    val hint: String = "", val mcpLocked: Boolean = false,
)

/** Web order and labels; the server supplies the current value and default text. */
val AUTO_SYSTEM_PROMPT_ITEMS = listOf(
    Triple("python", "Python 実行案内", ""),
    Triple("gemini_local_python", "Gemini 音声/動画/PDF/DOCX + Python（ローカル実行）", ""),
    Triple("grok_search", "Search補助（Grok）", ""),
    Triple("openai_search", "Search補助（OpenAI/xAI Responses）", ""),
    Triple("marker", "Marker編集時", ""),
    Triple("attachment_names", "添付ファイル名（LLM入力時）", "利用可能変数: {{attachment_names}} / {{attachment_count}}"),
    Triple("mathjax", "MathJax（LaTeX数式）", ""),
    Triple("image_analysis", "画像解析（Vision Model指示文）", ""),
    Triple("mcp", "MCP（外部ツール接続）", "利用可能変数: {{mcp_tools}}（接続中のMCPツール一覧が入ります）"),
)

fun parseAutoSystemPrompts(json: JSONObject): List<AutoSystemPrompt> = AUTO_SYSTEM_PROMPT_ITEMS.map { (key, label, hint) ->
    val row = json.optJSONObject(key)
    AutoSystemPrompt(
        key = key, label = label,
        enabled = row?.optBoolean("enabled", true) ?: true,
        text = row?.nullableString("text").orEmpty(),
        defaultText = row?.nullableString("default_text").orEmpty(),
        hint = hint, mcpLocked = key == "mcp",
    )
}

fun parseMcpServers(json: JSONObject): List<McpServerInfo> {
    val rows = json.optJSONArray("servers") ?: return emptyList()
    return (0 until rows.length()).map { index ->
        val row = rows.getJSONObject(index)
        McpServerInfo(
            id = row.optInt("id"),
            name = row.nullableString("name").ifBlank { "MCP" },
            enabled = row.optBoolean("enabled"),
            connectionState = row.nullableString("connection_state"),
            authStatus = row.nullableString("auth_status"),
            toolCount = row.optInt("tool_count"),
            isPreset = row.optBoolean("is_preset"),
            description = row.nullableString("description"),
            url = row.nullableString("url"),
            authType = row.nullableString("auth_type").ifBlank { "none" },
            lastError = row.nullableString("last_error"),
            oauthClientRegistered = row.optBoolean("oauth_client_registered"),
            authHasToken = row.optBoolean("auth_has_token"),
        )
    }
}

fun parseFeedbackItems(json: JSONObject): List<FeedbackItem> {
    val rows = json.optJSONArray("items") ?: return emptyList()
    return (0 until rows.length()).map { index ->
        val row = rows.getJSONObject(index)
        FeedbackItem(
            id = row.optInt("id"),
            title = row.nullableString("title"),
            message = row.nullableString("message"),
            status = row.nullableString("status"),
            adminReply = row.nullableString("admin_reply"),
            createdAt = row.nullableString("created_at"),
        )
    }
}

fun parseStorageUsage(json: JSONObject): StorageUsage = StorageUsage(
    usedBytes = json.optLong("used_bytes"),
    limitBytes = json.optLong("limit_bytes"),
    usedMb = json.optString("used_mb", "0"),
    limitMb = json.optString("limit_mb", "unlimited"),
    unlimited = json.optBoolean("is_unlimited"),
)

fun parseLibraryFiles(json: JSONObject): List<LibraryFile> {
    val rows = json.optJSONArray("files") ?: return emptyList()
    return (0 until rows.length()).map { index ->
        val row = rows.getJSONObject(index)
        LibraryFile(
            displayName = row.nullableString("filename").ifBlank { row.nullableString("original_filename") },
            filepath = row.nullableString("filepath"),
            url = row.nullableString("url"),
            thumbnailUrl = row.nullableString("thumbnail_url"),
            type = row.optString("type", "file"),
            ext = row.optString("ext"),
            isFavorite = row.optBoolean("is_favorite"),
            timestamp = row.optLong("ts", 0L),
        )
    }
}

fun parseGems(rows: JSONArray): List<Gem> = (0 until rows.length()).map { index ->
    val row = rows.getJSONObject(index)
    Gem(
        uuid = row.nullableString("uuid"),
        name = row.nullableString("name").ifBlank { "Gem" },
        description = row.nullableString("description"),
        instruction = row.nullableString("instruction"),
        defaultModel = row.nullableString("default_model"),
        fixedPrompts = parseFixedPrompts(row.opt("fixed_prompts")),
    )
}

/** Detects a trailing `@name` mention for the Gem candidate list. */
fun gemMentionQuery(text: String): String? =
    Regex("(?:^|\\s)@([^\\s@]*)$").find(text)?.groupValues?.get(1)

/** Removes the trailing `@query` mention once a Gem is applied. */
fun replaceGemMention(text: String, query: String): String {
    val marker = "@$query"
    return if (text.endsWith(marker)) text.dropLast(marker.length).trimEnd() else text
}

/** Live-only progress cards for streamed search and tool execution. */
enum class CardKind { SEARCH, PYTHON, MCP, CODING, TOOL }

data class StatusCard(
    val id: String,
    val kind: CardKind,
    val label: String,
    val detail: String = "",
    val code: String = "",
    val output: String = "",
    val done: Boolean = false,
    /** The MCP call failed (Web `.mcp-box.mcp-error`). */
    val failed: Boolean = false,
    /** The dashed note under an MCP card: the result summary or the error message. */
    val note: String = "",
)

data class McpDecision(
    val id: String,
    val jobId: String,
    val serverName: String,
    val toolName: String,
    val argsPreview: String,
)

/** Web `showBotLockOverlay` state. */
data class AccountLock(val message: String, val untilMillis: Long)

class ApiException(val status: Int, val payload: JSONObject, val retryAfter: Long = 5) : IOException() {
    val code: String get() = payload.optString("code").ifBlank { payload.optString("error") }
    override val message: String get() = when {
        code == "passkey_unavailable" -> "パスキーでログインできません。「パスキーのみでログイン」を有効にしたアカウントか確認してください。"
        code == "invalid_auth_code" -> "外部ログインを確認できませんでした。もう一度ログインしてください。"
        code == "setup_already_completed" -> "初回設定はすでに完了しています。アプリを再起動してください。"
        status == 401 -> "ログインの有効期限が切れました。もう一度連携してください。"
        code == "invalid_credentials" -> "ユーザー名またはパスワードが正しくありません。"
        code == "username_taken" -> "そのユーザー名はすでに使われています。"
        code == "invalid_username" -> "ユーザー名は3〜80文字で、@は使えません。"
        code == "invalid_password" -> "パスワードは8〜256文字で入力してください。"
        code == "invalid_2fa" -> "2段階認証コードが正しくありません。"
        code == "setup_required" -> "初回設定を完了してください。"
        code == "invalid_vertex_credentials" -> "Vertex AIのサービスアカウントJSONを確認してください。"
        code == "turnstile_required" -> "Webで安全性の確認が必要です。「Web設定」を開いて確認してください。"
        code == "account_locked" -> payload.optString("message").ifBlank { "アカウントが一時的にロックされています。" }
        code == "banned" || code == "request_blocked" -> "アカウントの利用が制限されています。Webで状態を確認してください。"
        status == 429 -> "アクセスが集中しています。${retryAfter}秒以上待って再試行してください。"
        status == 503 -> "サービスを一時的に利用できません。しばらくしてからお試しください。"
        else -> payload.optString("error", "通信エラー（HTTP $status）").take(500)
    }
}

fun JSONArray.strings(): List<String> = (0 until length()).map { getString(it) }
fun JSONObject.nullableString(name: String): String = if (isNull(name)) "" else optString(name)
fun parseThreadItem(row: JSONObject): ThreadItem = ThreadItem(
    id = row.get("id").toString(),
    title = row.nullableString("title"),
    model = row.nullableString("last_model"),
    isBookmarked = row.optBoolean("is_bookmarked"),
    isTemporary = row.optBoolean("is_temporary"),
)
fun parseModels(json: JSONObject): List<ModelInfo> {
    val rows = json.optJSONArray("models")
    if (rows == null) return json.optJSONArray("model_ids")?.strings().orEmpty().map {
        ModelInfo(it, it, "", "", "chat", setOf("chat", "attachments"), false, true)
    }
    return (0 until rows.length()).map { index ->
        val row = rows.getJSONObject(index)
        ModelInfo(
            id = row.getString("id"), name = row.optString("name", row.getString("id")),
            provider = row.optString("provider"), providerLabel = row.optString("provider_label"),
            mode = row.optString("mode", "chat"),
            capabilities = row.optJSONArray("capabilities")?.strings()?.toSet().orEmpty(),
            deprecated = row.optBoolean("deprecated"), selectable = row.optBoolean("selectable", true),
            description = row.nullableString("description"), price = row.nullableString("price"),
            category = row.nullableString("category"), implementedAt = row.nullableString("implementedAt"),
            implementedRank = row.optInt("implementedRank"), emoji = row.nullableString("emoji"),
            tags = row.optJSONArray("tags")?.strings()?.toSet().orEmpty(),
        )
    }
}
fun parseMessages(json: JSONObject): List<ChatMessage> {
    val rows = json.optJSONArray("messages") ?: JSONArray()
    return (0 until rows.length()).map { index ->
        val row = rows.getJSONObject(index)
        val attachmentText = row.nullableString("image_url")
        val files = if (attachmentText.isBlank()) emptyList() else runCatching {
            JSONArray(attachmentText).strings()
        }.getOrElse { listOf(attachmentText) }
        ChatMessage(row.get("id").toString(), row.optString("role"), row.nullableString("content"),
            row.nullableString("thought_data"), files,
            parentId = if (row.isNull("parent_id")) null else row.optInt("parent_id"),
            model = row.nullableString("model"),
            tokens = row.nullableInt("tokens"), tokensIn = row.nullableInt("tokens_in"),
            tokensOut = row.nullableInt("tokens_out"), tokensContent = row.nullableInt("tokens_content"),
            tokensThought = row.nullableInt("tokens_thought"),
            encrypted = if (row.has("is_encrypted") && !row.isNull("is_encrypted")) row.optBoolean("is_encrypted") else null,
            quote = row.nullableString("quote_text"), gemName = row.nullableString("gem_name"))
    }
}

/** Integer field that may be absent or JSON null. */
fun JSONObject.nullableInt(name: String): Int? =
    if (!has(name) || isNull(name)) null else optDouble(name).takeIf { !it.isNaN() }?.toInt()

/** Web `buildTokenTotals`: sums per-message totals; a field is null when no message reported it. */
data class TokenTotals(val total: Int, val tokensIn: Int?, val tokensOut: Int?, val tokensContent: Int?, val tokensThought: Int?)

fun buildTokenTotals(messages: List<ChatMessage>): TokenTotals {
    var total = 0
    var sumIn = 0; var hasIn = false
    var sumOut = 0; var hasOut = false
    var sumContent = 0; var hasContent = false
    var sumThought = 0; var hasThought = false
    messages.forEach { m ->
        val rowTotal = m.tokens ?: if (m.tokensIn != null || m.tokensOut != null) (m.tokensIn ?: 0) + (m.tokensOut ?: 0) else null
        if (rowTotal != null) total += rowTotal
        m.tokensIn?.let { sumIn += it; hasIn = true }
        m.tokensOut?.let { sumOut += it; hasOut = true }
        m.tokensContent?.let { sumContent += it; hasContent = true }
        m.tokensThought?.let { sumThought += it; hasThought = true }
    }
    return TokenTotals(total, sumIn.takeIf { hasIn }, sumOut.takeIf { hasOut }, sumContent.takeIf { hasContent }, sumThought.takeIf { hasThought })
}

/** Web bubble footer token label: `In a / Out b (Thought c)`, or `N tokens`, or null. */
fun messageTokenLabel(message: ChatMessage): String? {
    val parts = buildList {
        message.tokensIn?.let { add("In $it") }
        message.tokensOut?.let { out ->
            add(if ((message.tokensThought ?: 0) > 0) "Out $out (Thought ${message.tokensThought})" else "Out $out")
        }
    }
    return when {
        parts.isNotEmpty() -> parts.joinToString(" / ")
        message.tokens != null -> "${message.tokens} tokens"
        else -> null
    }
}

/** Numeric database id, or null for local optimistic messages. */
fun numericId(message: ChatMessage): Int? = message.id.toIntOrNull()

/** Active branch path ending at [leafId], oldest first. Falls back to the newest message. */
fun activeBranchPath(messages: List<ChatMessage>, leafId: Int?): List<ChatMessage> {
    val byId = messages.mapNotNull { message -> numericId(message)?.let { it to message } }.toMap()
    if (byId.isEmpty()) return messages
    val start = leafId?.let { byId[it] } ?: byId.values.maxByOrNull { numericId(it) ?: 0 } ?: return messages
    val path = ArrayDeque<ChatMessage>()
    val seen = HashSet<Int>()
    var current: ChatMessage? = start
    while (current != null) {
        val id = numericId(current) ?: break
        if (!seen.add(id)) break
        path.addFirst(current)
        current = current.parentId?.let { byId[it] }
    }
    return path.toList()
}

/** Messages that share one parent, used for branch navigation. */
fun siblingGroup(messages: List<ChatMessage>, message: ChatMessage): List<ChatMessage> =
    messages.filter { numericId(it) != null && it.parentId == message.parentId }
        .sortedBy { numericId(it) ?: 0 }

/** Follows the highest-id child chain from [startId] to the latest leaf of that branch. */
fun latestLeafId(messages: List<ChatMessage>, startId: Int): Int {
    val children = messages.groupBy { it.parentId }
    var current = startId
    while (true) {
        val next = children[current]?.mapNotNull { numericId(it) }?.maxOrNull() ?: return current
        current = next
    }
}

/** Extracts role and plain text from the server PDF payload for native rendering. */
fun parsePdfMessages(payload: JSONObject): List<ChatMessage> {
    val rows = payload.optJSONArray("messages") ?: return emptyList()
    return (0 until rows.length()).map { index ->
        val row = rows.getJSONObject(index)
        ChatMessage(index.toString(), row.optString("role"), row.nullableString("content"),
            row.nullableString("thought_text"))
    }
}

/** Adds or updates the single Web-search card for the current stream. */
fun upsertSearchCard(cards: List<StatusCard>, content: String): List<StatusCard> {
    val done = content.trim().lowercase() in setOf("done", "complete", "completed", "finished")
    val card = StatusCard(
        id = "search", kind = CardKind.SEARCH,
        label = if (done) "Web検索が完了しました" else "Webを検索しています…",
        done = done,
    )
    val index = cards.indexOfFirst { it.kind == CardKind.SEARCH }
    return if (index >= 0) cards.toMutableList().also { it[index] = card } else cards + card
}

/** Merges the code and output events that share one Python execution id. */
fun upsertPythonCard(cards: List<StatusCard>, payload: JSONObject): List<StatusCard> {
    val id = payload.optString("id").ifBlank { "python" }
    val code = payload.optString("code")
    val output = payload.optString("output")
    val index = cards.indexOfFirst { it.kind == CardKind.PYTHON && it.id == id }
    val previous = if (index >= 0) cards[index] else null
    val card = StatusCard(
        id = id, kind = CardKind.PYTHON, label = "Python",
        code = code.ifBlank { previous?.code.orEmpty() },
        output = output.ifBlank { previous?.output.orEmpty() },
        done = output.isNotBlank(),
    )
    return if (index >= 0) cards.toMutableList().also { it[index] = card } else cards + card
}

/** Renders structured tool events without leaking raw protocol JSON into the answer text. */
/**
 * Web `handleMcpStreamEvent`: `start` adds a running card, `result` marks it done with the first line of
 * the summary, `error` marks it failed with the message. Decision events do not create cards.
 */
fun upsertMcpCard(cards: List<StatusCard>, payload: JSONObject?): List<StatusCard> {
    val type = payload?.optString("type").orEmpty()
    if (payload == null || type !in setOf("start", "result", "error")) return cards
    val id = payload.optString("id").ifBlank { "mcp_" + cards.count { it.kind == CardKind.MCP } }
    val tool = payload.optString("tool_name").ifBlank { payload.optString("internal_name") }
    val label = "${payload.optString("server_name").ifBlank { "MCP" }} / $tool"
    val index = cards.indexOfFirst { it.kind == CardKind.MCP && it.id == id }
    val card = when (type) {
        "start" -> if (index >= 0) return cards else StatusCard(id, CardKind.MCP, label)
        "result" -> StatusCard(id, CardKind.MCP, label, done = true,
            note = payload.optString("summary").lineSequence().firstOrNull().orEmpty().take(220))
        else -> StatusCard(id, CardKind.MCP, label, done = true, failed = true,
            note = payload.optString("message").ifBlank { "MCPツールの実行に失敗しました" }.take(300))
    }
    return if (index >= 0) cards.toMutableList().also { it[index] = card } else cards + card
}

fun upsertToolCard(cards: List<StatusCard>, type: String, content: Any?): List<StatusCard> {
    val payload = content as? JSONObject
    if (type == "mcp") return upsertMcpCard(cards, payload)
    if (type.startsWith("mcp")) return cards
    val kind = if (type == "coding_diff") CardKind.CODING else if (type.startsWith("mcp")) CardKind.MCP else CardKind.TOOL
    val id = payload?.optString("id")?.takeIf { it.isNotBlank() }
        ?: payload?.optString("tool_call_id")?.takeIf { it.isNotBlank() }
        ?: payload?.optString("target_id")?.takeIf { it.isNotBlank() }
        ?: "${kind.name.lowercase()}-${cards.count { it.kind == kind }}"
    val innerType = payload?.optString("type").orEmpty()
    val done = type.endsWith("resolved") || payload?.optBoolean("done") == true ||
        type == "coding_diff" || innerType in setOf("result", "error", "decision_resolved")
    val label = when (kind) {
        CardKind.MCP -> payload?.optString("tool_name")?.ifBlank { "MCPツール" } ?: "MCPツール"
        CardKind.CODING -> "Coding差分"
        else -> payload?.optString("name")?.ifBlank { "ツール" } ?: "ツール"
    }
    val detail = payload?.optString("message").orEmpty()
        .ifBlank { payload?.optString("summary").orEmpty() }
        .ifBlank { payload?.optString("status").orEmpty() }
    val output = when (kind) {
        CardKind.CODING -> payload?.optString("diff").orEmpty()
        else -> payload?.optString("output").orEmpty().ifBlank { payload?.optString("result").orEmpty() }
    }
    val card = StatusCard(id, kind, label, detail = detail, output = output, done = done)
    val index = cards.indexOfFirst { it.kind == kind && it.id == id }
    return if (index >= 0) cards.toMutableList().also { it[index] = card } else cards + card
}

private val IMAGE_EXTENSIONS = setOf("png", "jpg", "jpeg", "gif", "webp", "bmp", "heic", "heif", "avif")
private val AUDIO_EXTENSIONS = setOf("wav", "mp3", "m4a", "ogg", "flac", "aac", "opus")
private val VIDEO_EXTENSIONS = setOf("mp4", "mov", "mkv", "avi", "m4v", "webm", "3gp")
private val TEXT_EXTENSIONS = setOf("txt", "md", "markdown", "json", "yaml", "yml", "csv", "tsv", "log",
    "py", "js", "ts", "tsx", "jsx", "html", "css", "xml", "kt", "java", "c", "cpp", "h", "sh", "sql")

/** Categories used to pick a MIME-appropriate attachment preview and label. */
enum class AttachmentKind { IMAGE, AUDIO, VIDEO, PDF, TEXT, FILE }

fun attachmentKind(name: String, mime: String = ""): AttachmentKind {
    val ext = name.substringBefore('?').substringAfterLast('.', "").lowercase()
    val type = mime.substringBefore(';').trim().lowercase()
    return when {
        ext in IMAGE_EXTENSIONS || type.startsWith("image/") -> AttachmentKind.IMAGE
        ext in AUDIO_EXTENSIONS || type.startsWith("audio/") -> AttachmentKind.AUDIO
        ext in VIDEO_EXTENSIONS || type.startsWith("video/") -> AttachmentKind.VIDEO
        ext == "pdf" || type == "application/pdf" -> AttachmentKind.PDF
        ext in TEXT_EXTENSIONS || type.startsWith("text/") -> AttachmentKind.TEXT
        else -> AttachmentKind.FILE
    }
}

fun attachmentKindIcon(kind: AttachmentKind): String = when (kind) {
    AttachmentKind.IMAGE -> "🖼"
    AttachmentKind.AUDIO -> "🎵"
    AttachmentKind.VIDEO -> "🎬"
    AttachmentKind.PDF -> "📄"
    AttachmentKind.TEXT -> "📝"
    AttachmentKind.FILE -> "📎"
}

/** Best-effort extension for providers that only report a MIME type (e.g. FileProvider). */
fun extensionForMime(mime: String): String = when (mime.substringBefore(';').trim().lowercase()) {
    "image/jpeg" -> "jpg"
    "image/png" -> "png"
    "image/webp" -> "webp"
    "image/gif" -> "gif"
    "image/heic" -> "heic"
    "image/heif" -> "heif"
    "audio/mpeg" -> "mp3"
    "audio/mp4", "audio/m4a", "audio/x-m4a" -> "m4a"
    "audio/wav", "audio/x-wav" -> "wav"
    "audio/ogg" -> "ogg"
    "audio/flac" -> "flac"
    "video/mp4" -> "mp4"
    "video/quicktime" -> "mov"
    "video/webm" -> "webm"
    "application/pdf" -> "pdf"
    "text/plain" -> "txt"
    else -> ""
}

fun formatByteSize(bytes: Long): String {
    if (bytes < 0) return ""
    if (bytes < 1024) return "$bytes B"
    val units = listOf("KB", "MB", "GB")
    var value = bytes.toDouble() / 1024
    var unit = 0
    while (value >= 1024 && unit < units.lastIndex) { value /= 1024; unit++ }
    return if (value >= 10 || value % 1.0 == 0.0) "${value.toInt()} ${units[unit]}"
    else String.format(java.util.Locale.US, "%.1f %s", value, units[unit])
}

fun isImageReference(reference: String): Boolean =
    reference.substringBefore('?').substringAfterLast('.', "").lowercase() in IMAGE_EXTENSIONS

/**
 * Accepts only a same-origin attachment reference. Provider URLs, other hosts and
 * path traversal are rejected so the Bearer token is never attached to foreign hosts.
 */
fun fileReferencePath(value: String): String? {
    val raw = value.trim()
    if (raw.isEmpty()) return null
    val path = if (raw.startsWith("http://") || raw.startsWith("https://")) {
        val base = BuildConfig.BASE_URL.toHttpUrlOrNull() ?: return null
        val url = raw.toHttpUrlOrNull() ?: return null
        if (url.scheme != base.scheme || url.host != base.host || url.port != base.port) return null
        url.encodedPath
    } else raw
    val stripped = when {
        path.startsWith("/files/") -> path.removePrefix("/files/")
        path.startsWith("files/") -> path.removePrefix("files/")
        else -> path
    }
    if (stripped.isEmpty() || stripped.startsWith('/')) return null
    if (stripped.contains("://") || stripped.contains(':') || stripped.contains('?') || stripped.contains('#')) return null
    val segments = stripped.split('/')
    if (segments.any { it.isBlank() || it == ".." }) return null
    return stripped
}

/** A pending WebAuthn ceremony that the UI must run through Credential Manager. */
data class CredentialRequest(val kind: String, val transactionId: String, val publicKeyJson: String)

data class PasskeyInfo(val id: String, val name: String, val createdAt: String?)

data class SecurityInfo(
    val is2faEnabled: Boolean,
    val hasTotp: Boolean,
    val hasWebauthn: Boolean,
    val default2fa: String,
    val passkeyOnlyLogin: Boolean,
    val skip2faOnGoogle: Boolean,
    val passkeys: List<PasskeyInfo>,
)

fun parseSecurityInfo(json: JSONObject): SecurityInfo {
    val keys = json.optJSONArray("passkeys") ?: JSONArray()
    val passkeys = buildList {
        for (index in 0 until keys.length()) {
            val item = keys.optJSONObject(index) ?: continue
            val id = item.nullableString("id") ?: continue
            add(PasskeyInfo(id, item.nullableString("name") ?: "パスキー", item.nullableString("created_at")))
        }
    }
    return SecurityInfo(
        is2faEnabled = json.optBoolean("is_2fa_enabled"),
        hasTotp = json.optBoolean("has_totp"),
        hasWebauthn = json.optBoolean("has_webauthn"),
        default2fa = json.nullableString("default_2fa_method").ifBlank { "totp" },
        passkeyOnlyLogin = json.optBoolean("passkey_only_login"),
        skip2faOnGoogle = json.optBoolean("skip_2fa_on_google_login"),
        passkeys = passkeys,
    )
}

fun readBoundedUtf8(input: InputStream, limit: Int): String {
    return String(readBoundedBytes(input, limit.toLong()), Charsets.UTF_8)
}

fun readBoundedBytes(input: InputStream, limit: Long): ByteArray {
    val output = ByteArrayOutputStream()
    val buffer = ByteArray(8192)
    var total = 0L
    while (true) {
        val count = input.read(buffer)
        if (count < 0) break
        total += count
        if (total > limit) throw IOException("応答が大きすぎます。表示件数を減らしてください。")
        output.write(buffer, 0, count)
    }
    return output.toByteArray()
}

/**
 * The pending / streaming answer bubble of the Web (`sendMessage`): the skeleton with its status and
 * sub line until the first answer event, the search box, image analysis, the reasoning placeholder
 * and a stream error.
 */
data class LiveAnswer(
    val model: String = "",
    /** `.skeleton-status`; null once the first answer event arrived (`beginPendingToStreamTransition`). */
    val pendingStatus: String? = null,
    val pendingSub: String = "",
    /** `markApiAccepted` ran (the status only moves to 接続完了 once). */
    val accepted: Boolean = false,
    /** `search_status`: "searching", "done" (Search complete, removed after 2s) or "". */
    val search: String = "",
    val imageAnalysis: String? = null,
    /** The collapsed "Thinking Process" placeholder shown for reasoning requests until thoughts arrive. */
    val thoughtPlaceholder: String? = null,
    val error: String? = null,
)

/** Web `getPendingSkeletonKind`. */
fun pendingSkeletonKind(model: String): String {
    val m = model.lowercase()
    return when {
        m.contains("video") -> "video"
        m.contains("tts") || m.contains("transcribe") || m.contains("realtime") || m.contains("voice") ||
            m.contains("native-audio") || (m.contains("live") && m.contains("gemini")) -> "audio"
        m.contains("gpt-image") || m.contains("imagine-image") || (m.contains("image") && !m.contains("vision")) ||
            (m.contains("gemini") && (m.contains("image") || m.contains("nano"))) -> "image"
        m.contains("ocr") -> "text"
        m.contains("build") || m.contains("code-fast") || m.contains("coding") -> "code"
        else -> "text"
    }
}

/** Web `shouldShowReasoningProgress`: reasoning was requested for a model that streams reasoning. */
fun showsReasoningProgress(model: String, enableThinking: Boolean, effort: String): Boolean {
    val m = model.lowercase()
    val requested = enableThinking || (effort.isNotBlank() && effort.lowercase() != "none")
    val capable = m.contains("gemini") || m.contains("o1") || m.contains("o3") || m.contains("gpt-5") ||
        (m.contains("reasoning") && !m.contains("non-reasoning"))
    return requested && capable
}
