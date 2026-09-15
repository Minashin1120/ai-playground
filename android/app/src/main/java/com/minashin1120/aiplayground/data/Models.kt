package com.minashin1120.aiplayground.data

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

fun readBoundedUtf8(input: InputStream, limit: Int): String {
    val output = ByteArrayOutputStream()
    val buffer = ByteArray(8192)
    var total = 0
    while (true) {
        val count = input.read(buffer)
        if (count < 0) break
        total += count
        if (total > limit) throw IOException("応答が大きすぎます。表示件数を減らしてください。")
        output.write(buffer, 0, count)
    }
    return output.toString(Charsets.UTF_8.name())
}
