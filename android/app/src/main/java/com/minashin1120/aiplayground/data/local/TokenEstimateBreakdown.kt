package com.minashin1120.aiplayground.data.local

import com.minashin1120.aiplayground.data.ActivityLog
import com.minashin1120.aiplayground.data.direct.DirectAttachment
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import org.json.JSONArray
import org.json.JSONObject

internal fun tokenEstimateBreakdown(result: JSONObject, message: Int?, quote: Int?, files: JSONArray): JSONObject {
    val known = (message ?: 0) + (quote ?: 0) + (0 until files.length()).sumOf { files.getJSONObject(it).optInt("tokens") }
    return JSONObject().put("message_tokens", message ?: JSONObject.NULL).put("quote_tokens", quote ?: JSONObject.NULL)
        .put("files", files).put("adjustment_tokens", result.getInt("tokens_total") - known)
}

/** Called only after opening the detail dialog; API failures keep the combined count intact. */
internal suspend fun providerTokenBreakdown(
    result: JSONObject, hasQuote: Boolean, refs: List<String>, attachments: List<DirectAttachment>,
    countMessage: suspend () -> Int?, countFiles: suspend (List<DirectAttachment>) -> Int?,
): JSONObject {
    val deadline = System.nanoTime() + 30_000_000_000L
    suspend fun safeCount(block: suspend () -> Int?): Int? = try { block() }
        catch (e: CancellationException) { throw e }
        catch (e: Exception) {
            ActivityLog.log("state", "route" to "/api/token_estimate", "phase" to "error")
            null
        }
    val message = if (hasQuote) safeCount(countMessage) else result.getInt("tokens_prompt")
    val quote = if (!hasQuote) 0 else message?.let { maxOf(0, result.getInt("tokens_prompt") - it) }
    var previous: Int? = result.getInt("tokens_prompt")
    val rows = JSONArray()
    attachments.forEachIndexed { index, file ->
        currentCoroutineContext().ensureActive()
        val current = if (System.nanoTime() >= deadline) null else if (index == attachments.lastIndex)
            result.getInt("tokens_total") else safeCount { countFiles(attachments.take(index + 1)) }
        val prior = previous
        val tokens = if (current != null && prior != null && current >= prior) current - prior else null
        previous = current
        rows.put(JSONObject().put("reference", refs[index]).put("name", file.name).put("tokens", tokens ?: JSONObject.NULL)
            .put("method", "provider").put("reason", if (tokens == null) "unseparated" else "ok"))
    }
    return tokenEstimateBreakdown(result, message, quote, rows)
}
