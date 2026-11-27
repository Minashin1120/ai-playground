package com.minashin1120.aiplayground.data.local

import com.minashin1120.aiplayground.data.ActivityLog
import com.minashin1120.aiplayground.data.direct.AttachmentExtractor
import com.minashin1120.aiplayground.data.direct.DirectAttachment
import com.minashin1120.aiplayground.data.direct.DirectRequest
import com.minashin1120.aiplayground.data.direct.DirectRouter
import com.minashin1120.aiplayground.data.direct.DirectTurn
import com.minashin1120.aiplayground.data.direct.VertexTarget
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import org.json.JSONArray
import org.json.JSONObject

internal class LocalChatPromptEstimator(
    private val store: LocalChatStore,
    private val remote: ServerHistory?,
    private val router: DirectRouter,
    private val settings: LocalSettingsStore,
    private val modeOf: (String) -> String,
    private val vertexTarget: () -> VertexTarget?,
    private val attachmentRefs: (JSONObject) -> List<String>,
) {
    private val tokenEstimates = LinkedHashMap<String, AttachmentTokenEstimate>()

    suspend fun estimatePrompt(payload: JSONObject): JSONObject {
        val started = System.currentTimeMillis()
        ActivityLog.log("state", "route" to "/api/token_estimate", "phase" to "start")
        try {
            val model = payload.optString("model")
            val quote = payload.optString("quote_text")
            val message = payload.optString("message")
            val text = if (quote.isEmpty()) message else
                "Context (User Quote):\n\"\"\"\n$quote\n\"\"\"\n\nUser Message:\n$message"
            val prompt = LocalChatBackend.estimateTokens(text)
            val refs = attachmentRefs(payload)
            providerEstimate(model, text, refs, payload)?.let {
                ActivityLog.log("state", "route" to "/api/token_estimate", "phase" to "done", "files" to refs.size,
                    "ms" to System.currentTimeMillis() - started)
                return it
            }
            var tokens = 0
            var counted = 0
            var missing = 0
            var errors = 0
            var nonText = 0
            val detailFiles = JSONArray()
            for (ref in refs) {
                currentCoroutineContext().ensureActive()
                val key = "$model\n$ref"
                val cached = synchronized(tokenEstimates) { tokenEstimates[key] }
                val estimate = cached ?: try {
                    val local = LocalChatStore.isLocalReference(ref)
                    val info = if (local) store.fileInfo(ref) else null
                    val name = info?.optString("name") ?: ref.substringAfterLast('/')
                    val mime = info?.optString("mime") ?: java.net.URLConnection.guessContentTypeFromName(name).orEmpty()
                    val job = currentCoroutineContext()[Job]
                    val bytes = if (local) store.loadFile(ref, LocalChatBackend.MAX_ATTACHMENT_BYTES) { job?.isActive == false }
                        else remote?.file(ref, LocalChatBackend.MAX_ATTACHMENT_BYTES)
                    if (bytes == null) AttachmentTokenEstimate(reason = "missing")
                    else AttachmentTokenEstimator.estimate(model, name, mime, bytes, store::temporaryProbeFile)
                } catch (e: CancellationException) { throw e }
                catch (e: Exception) { AttachmentTokenEstimate(reason = "error") }
                if (cached == null && estimate.reason == "ok") synchronized(tokenEstimates) {
                    if (tokenEstimates.size >= 128) tokenEstimates.remove(tokenEstimates.keys.first())
                    tokenEstimates[key] = estimate
                }
                if (payload.optBoolean("include_details")) detailFiles.put(JSONObject()
                    .put("reference", ref).put("name", store.fileInfo(ref)?.optString("name") ?: ref.substringAfterLast('/'))
                    .put("tokens", if (estimate.reason == "ok") estimate.tokens else JSONObject.NULL)
                    .put("method", "approximate").put("reason", estimate.reason))
                tokens += estimate.tokens
                when (estimate.reason) {
                    "ok" -> counted++
                    "missing" -> missing++
                    "non_text" -> nonText++
                    else -> errors++
                }
            }
            ActivityLog.log("state", "route" to "/api/token_estimate", "phase" to "done",
                "files" to refs.size, "failed_files" to errors + missing, "ms" to System.currentTimeMillis() - started)
            val result = JSONObject().put("estimate_method", "approximate").put("estimate_scope", "draft")
                .put("countable", true).put("tokens_total", prompt + tokens).put("tokens_prompt", prompt)
                .put("tokens_files", tokens).put("files_total", refs.size).put("files_counted", counted)
                .put("files_non_text", nonText).put("files_missing", missing).put("files_error", errors)
            if (payload.optBoolean("include_details")) {
                val messageTokens = LocalChatBackend.estimateTokens(message)
                result.put("breakdown", tokenEstimateBreakdown(result, messageTokens,
                    if (quote.isEmpty()) 0 else maxOf(0, prompt - messageTokens), detailFiles))
            }
            return result
        } catch (e: CancellationException) {
            ActivityLog.log("state", "route" to "/api/token_estimate", "phase" to "cancelled")
            throw e
        } catch (e: Exception) {
            ActivityLog.log("state", "route" to "/api/token_estimate", "phase" to "error")
            throw e
        }
    }

    private suspend fun providerEstimate(model: String, text: String, refs: List<String>, payload: JSONObject): JSONObject? {
        if (modeOf(model) != "chat") return null
        // Automatic recounts must not resend megabytes of media for every draft edit.
        // The details dialog remains an explicit request for full provider counting.
        if (!payload.optBoolean("include_details") && refs.any { ref ->
                !LocalChatStore.isLocalReference(ref) ||
                    (store.fileInfo(ref)?.optLong("size") ?: Long.MAX_VALUE) > 1024 * 1024
            }) return null
        val found = router.route(model, modeOf(model)) ?: return null
        try {
            val vertex = if (found.provider == "gemini") vertexTarget() else null
            val key = if (vertex != null) "" else settings.apiKeyFor(model, payload.optBoolean("gemini_free_tier")) ?: return null
            var remaining = 16L * 1024 * 1024 - text.toByteArray().size
            if (remaining < 0) return null
            val attachments = refs.map { ref ->
                val local = LocalChatStore.isLocalReference(ref)
                val info = if (local) store.fileInfo(ref) else null
                if (remaining <= 0 || (info != null && info.optLong("size") > remaining)) return null
                val job = currentCoroutineContext()[Job]
                val bytes = if (local) store.loadFile(ref, remaining) { job?.isActive == false }
                    else remote?.file(ref, remaining)
                if (bytes == null) return null
                remaining -= bytes.size
                val name = info?.optString("name") ?: ref.substringAfterLast('/')
                val mime = info?.optString("mime") ?: java.net.URLConnection.guessContentTypeFromName(name).orEmpty()
                AttachmentExtractor.prepare(name, mime, bytes)
            }
            val request = DirectRequest(model, key, "", listOf(DirectTurn("user", text)), JSONObject())
            val prompt = router.tokenCounter.count(request, vertex) ?: return null
            val total = if (attachments.isEmpty()) prompt else router.tokenCounter.count(
                request.copy(turns = listOf(DirectTurn("user", text, attachments))), vertex) ?: return null
            if (total < prompt) return null
            val result = JSONObject().put("countable", true).put("tokens_total", total).put("tokens_prompt", prompt)
                .put("tokens_files", total - prompt).put("files_total", refs.size).put("files_counted", refs.size)
                .put("files_non_text", 0).put("files_missing", 0).put("files_error", 0)
                .put("estimate_method", "provider").put("estimate_scope", "draft")
            if (payload.optBoolean("include_details")) result.put("breakdown", providerTokenBreakdown(
                result, payload.optString("quote_text").isNotEmpty(), refs, attachments,
                countMessage = {
                    val message = payload.optString("message")
                    if (message.isEmpty()) 0 else router.tokenCounter.count(request.copy(turns = listOf(DirectTurn("user", message))), vertex)
                },
                countFiles = { subset -> router.tokenCounter.count(request.copy(turns = listOf(DirectTurn("user", text, subset))), vertex) },
            ))
            return result
        } catch (e: CancellationException) { throw e }
        catch (e: Exception) {
            ActivityLog.log("state", "route" to "/api/token_estimate", "phase" to "error")
            return null
        }
    }
}
