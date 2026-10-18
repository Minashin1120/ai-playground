package com.minashin1120.aiplayground.data

import org.json.JSONObject
import java.time.Instant

/** A snapshot includes all routes before the chosen (timestamp, id), inclusive. */
fun branchSnapshot(rows: List<JSONObject>, selectedId: Int): List<JSONObject> {
    fun stamp(row: JSONObject): Instant = runCatching { Instant.parse(row.getString("created_at")) }.getOrNull()
        ?: Instant.ofEpochMilli(row.optLong("created_ms"))
    val selected = rows.firstOrNull { it.optInt("id") == selectedId } ?: error("メッセージがありません")
    val cutoff = stamp(selected)
    val included = rows.filter { stamp(it) < cutoff || (stamp(it) == cutoff && it.optInt("id") <= selectedId) }
        .sortedWith(compareBy<JSONObject> { stamp(it) }.thenBy { it.optInt("id") })
    val seen = HashSet<Int>()
    included.forEach { row ->
        check(!row.optBoolean("generating")) { "回答の生成が完了してから分岐してください" }
        check(row.isNull("parent_id") || row.optInt("parent_id") in seen) { "会話の親子関係が不正です" }
        seen += row.getInt("id")
    }
    return included
}

/** Local and server references appearing in attachment metadata or rendered file links. */
fun messageFileReferences(row: JSONObject): Set<String> {
    val refs = LinkedHashSet<String>()
    val raw = row.optString("image_url").takeUnless { it == "null" }.orEmpty()
    if (raw.isNotBlank()) runCatching {
        val values = org.json.JSONArray(raw)
        for (i in 0 until values.length()) {
            val value = values.get(i)
            val ref = if (value is JSONObject) listOf("filepath", "path", "url", "file").firstNotNullOfOrNull {
                key -> value.optString(key).takeIf { it.isNotBlank() && it != "null" }
            } else value.toString()
            if (ref != null) refs += ref.removePrefix("/files/")
        }
    }.onFailure { refs += raw.removePrefix("/files/") }
    Regex("""(?:/files/([0-9]+/[^\s<>"')\]}]+))|(local/[a-zA-Z0-9_.-]+)""").findAll(row.optString("content")).forEach {
        refs += it.groupValues[1].ifBlank { it.groupValues[2] }
    }
    return refs
}

const val BRANCH_SAVE_WARNING = "分岐先は通常のチャットとして保存され、自動削除されません。コピー対象の添付・生成ファイルも保存対象になります。"

/** Keep stored content intact; replace only rendered links, excluding code. */
fun missingFileMarkup(content: String, missing: Set<String>): String {
    if (missing.isEmpty()) return content
    val code = Regex("(?s)```.*?```|~~~.*?~~~|`[^`]*`")
    fun replace(segment: String): String {
        var result = segment
        missing.forEach { ref ->
            val url = if (ref.startsWith("local/")) "(?:/files/)?" + Regex.escape(ref) else Regex.escape("/files/$ref")
            result = result.replace(Regex("""!?\[[^\]]*]\($url(?:[?#][^)]*)?\)"""), "ファイルがありません")
            result = result.replace(Regex("""(?is)<(audio|video)\b[^>]*$url[^>]*>.*?</\1>"""), "ファイルがありません")
            result = result.replace(Regex("""(?is)<a\b[^>]*$url[^>]*>.*?</a>"""), "ファイルがありません")
            result = result.replace(Regex("""(?is)<img\b[^>]*$url[^>]*>"""), "ファイルがありません")
        }
        return result
    }
    val out = StringBuilder()
    var cursor = 0
    code.findAll(content).forEach { match ->
        out.append(replace(content.substring(cursor, match.range.first))).append(match.value)
        cursor = match.range.last + 1
    }
    return out.append(replace(content.substring(cursor))).toString()
}
