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
                       val parentId: Int? = null, val model: String = "")
data class Attachment(val name: String, val reference: String, val mime: String = "")
data class ModelInfo(val id: String, val name: String, val provider: String, val providerLabel: String,
                     val mode: String, val capabilities: Set<String>, val deprecated: Boolean,
                     val selectable: Boolean,
                     val description: String = "", val price: String = "", val category: String = "",
                     val implementedAt: String = "", val implementedRank: Int = 0, val emoji: String = "",
                     val tags: Set<String> = emptySet()) {
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
)

data class McpDecision(
    val id: String,
    val jobId: String,
    val serverName: String,
    val toolName: String,
    val argsPreview: String,
)

class ApiException(val status: Int, val payload: JSONObject, val retryAfter: Long = 5) : IOException() {
    val code: String get() = payload.optString("code").ifBlank { payload.optString("error") }
    override val message: String get() = when {
        status == 401 -> "ログインの有効期限が切れました。もう一度連携してください。"
        code == "turnstile_required" -> "Webで安全性の確認が必要です。「Web設定」を開いて確認してください。"
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
    title = row.optString("title", "新しいチャット"),
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
            model = row.nullableString("model"))
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
fun upsertToolCard(cards: List<StatusCard>, type: String, content: Any?): List<StatusCard> {
    val payload = content as? JSONObject
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
