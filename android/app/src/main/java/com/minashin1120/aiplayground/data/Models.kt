package com.minashin1120.aiplayground.data

import com.minashin1120.aiplayground.BuildConfig
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.io.InputStream

data class ThreadItem(val id: String, val title: String, val model: String)
data class ChatMessage(val id: String, val role: String, val content: String,
                       val thought: String = "", val files: List<String> = emptyList())
data class Attachment(val name: String, val reference: String)
data class ModelInfo(val id: String, val name: String, val provider: String, val providerLabel: String,
                     val mode: String, val capabilities: Set<String>, val deprecated: Boolean,
                     val selectable: Boolean) {
    fun supports(capability: String) = capability in capabilities
}
data class Account(val id: Int, val name: String, val models: List<ModelInfo>, val defaultModel: String,
                   val encrypted: Boolean)
data class StoredSession(val token: String, val expiresAt: Long)

/** Live-only progress cards for streamed search and tool execution. */
enum class CardKind { SEARCH, PYTHON }

data class StatusCard(
    val id: String,
    val kind: CardKind,
    val label: String,
    val detail: String = "",
    val code: String = "",
    val output: String = "",
    val done: Boolean = false,
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
            row.nullableString("thought_data"), files)
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

private val IMAGE_EXTENSIONS = setOf("png", "jpg", "jpeg", "gif", "webp", "bmp", "heic", "heif", "avif")

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
