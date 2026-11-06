package com.minashin1120.aiplayground.data.direct

import android.graphics.Bitmap
import com.minashin1120.aiplayground.data.PythonExecution
import com.minashin1120.aiplayground.data.pythonExecutionMarkdown
import android.graphics.BitmapFactory
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.util.Base64

/**
 * OpenAI Responses API streaming from the device (server `background.py` Responses branch). Also used
 * for xAI Grok through its OpenAI-compatible Responses endpoint ([xai] = true). Python runs in the
 * provider's hosted code interpreter instead of the server sandbox.
 */
class OpenAiResponsesDirect(
    private val http: DirectHttp,
    private val xai: Boolean = false,
    private val baseUrl: String = if (xai) "https://api.x.ai" else "https://api.openai.com",
) : DirectEngine {
    override suspend fun run(request: DirectRequest, emit: (JSONObject) -> Unit, onProgress: (String, String) -> Unit): DirectResult {
        // input_image is vision only. The python tool (code interpreter) can crop,
        // zoom, and rotate an image only after that image is a container file.
        val containerFileIds = if (!xai && request.options.flag("enable_python")) {
            uploadContainerImages(request, emit)
        } else {
            emptyList()
        }
        val payload = buildPayload(request, containerFileIds)
        val content = StringBuilder()
        val thought = StringBuilder()
        val reasoningParts = HashMap<String, String>()
        fun reasoning(data: JSONObject, text: String, snapshot: Boolean = false, index: Int? = null) {
            if (text.isEmpty()) return
            val key = data.optString("item_id").ifBlank { data.optString("id", "reasoning") } + ":" +
                (index ?: data.optInt("summary_index", data.optInt("content_index", 0)))
            val newPart = key !in reasoningParts
            val previous = reasoningParts[key].orEmpty()
            val delta = if (snapshot) text.takeIf { it.startsWith(previous) }?.removePrefix(previous).orEmpty() else text
            reasoningParts[key] = if (snapshot) text else previous + text
            if (delta.isNotEmpty()) {
                if (newPart && thought.isNotEmpty()) { thought.append("\n\n"); emit(event("thought", "\n\n")) }
                thought.append(delta); emit(event("thought", delta))
            }
        }
        val files = mutableListOf<DirectOutputFile>()
        val sources = LinkedHashMap<String, String>()
        val remoteImages = mutableListOf<RemoteImage>()
        val seenItems = HashSet<String>()
        var containerId = ""
        fun citation(annotation: JSONObject) {
            val url = annotation.optString("url")
            if (url.startsWith("https://") || url.startsWith("http://"))
                sources[url] = annotation.optString("title").ifBlank { url }
        }
        var usage: JSONObject? = null
        var searching = false
        val codes = LinkedHashMap<String, StringBuilder>()
        val runs = LinkedHashMap<String, PythonExecution>()
        fun absorbItem(item: JSONObject) {
            when (item.optString("type")) {
                "reasoning" -> {
                    for (field in listOf("summary", "content")) {
                        val parts = item.optJSONArray(field) ?: continue
                        for (index in 0 until parts.length()) parts.optJSONObject(index)?.let { part ->
                            if (part.optString("type") in listOf("summary_text", "reasoning_text"))
                                reasoning(item, part.optString("text"), snapshot = true, index = index)
                        }
                    }
                }
                "code_interpreter_call" -> {
                    item.optString("container_id").takeIf { it.isNotBlank() }?.let { containerId = it }
                    val id = item.optString("id").ifBlank { "py" }
                    if (seenItems.add("ci:$id")) {
                        outputItem(item, codes, content, files, emit)
                        val outputs = item.optJSONArray("outputs") ?: JSONArray()
                        runs[id] = PythonExecution(item.optString("code").ifBlank { codes[id]?.toString().orEmpty() },
                            (0 until outputs.length()).mapNotNull { outputs.optJSONObject(it)?.optString("logs")?.takeIf(String::isNotBlank) }.joinToString("\n"))
                    }
                    val outputs = item.optJSONArray("outputs") ?: return
                    for (index in 0 until outputs.length()) {
                        val output = outputs.optJSONObject(index) ?: continue
                        if (output.optString("type") != "image") continue
                        val fileId = output.optString("file_id")
                        val url = output.optString("url")
                        if (fileId.isNotBlank() && containerId.isNotBlank()) {
                            remoteImages += RemoteImage(containerId = containerId, fileId = fileId, filename = "output.png", runId = id)
                        } else if (url.isNotBlank()) {
                            remoteImages += RemoteImage(url = url, filename = "output.png", runId = id)
                        }
                    }
                }
                "message" -> {
                    val parts = item.optJSONArray("content") ?: return
                    for (index in 0 until parts.length()) {
                        val annotations = parts.optJSONObject(index)?.optJSONArray("annotations") ?: continue
                        for (a in 0 until annotations.length()) annotations.optJSONObject(a)?.let { annotation ->
                            if (!collectContainerCitation(annotation, containerId, remoteImages)) citation(annotation)
                        }
                    }
                }
                else -> if (seenItems.add(item.optString("type") + ":" + item.optString("id"))) {
                    outputItem(item, codes, content, files, emit)
                }
            }
        }
        emit(event("status", "APIへ送信完了。モデルが応答を生成中です..."))
        http.postSse("$baseUrl/v1/responses", mapOf("Authorization" to "Bearer ${request.apiKey}"), payload, maxLine = 32L * 1024 * 1024) { sse ->
            if (sse.data == "[DONE]") return@postSse false
            val data = runCatching { JSONObject(sse.data) }.getOrNull() ?: return@postSse true
            when (val type = data.optString("type").ifBlank { sse.event.orEmpty() }) {
                "response.output_text.annotation.added" -> data.optJSONObject("annotation")?.let { annotation ->
                    if (!collectContainerCitation(annotation, containerId, remoteImages)) citation(annotation)
                }
                "response.output_text.delta" -> data.optString("delta").takeIf { it.isNotEmpty() }?.let {
                    content.append(it); emit(event("content", it))
                }
                "response.reasoning_summary_text.delta", "response.reasoning_text.delta" -> data.optString("delta").takeIf { it.isNotEmpty() }?.let {
                    reasoning(data, it)
                }
                "response.reasoning_text.done", "response.reasoning_summary_text.done" -> reasoning(data, data.optString("text"), snapshot = true)
                "response.reasoning_summary_part.done", "response.content_part.done" -> data.optJSONObject("part")?.let { part ->
                    if (part.optString("type") in listOf("summary_text", "reasoning_text"))
                        reasoning(data, part.optString("text"), snapshot = true)
                }
                "response.web_search_call.in_progress", "response.web_search_call.searching" ->
                    if (!searching) { searching = true; emit(event("search_status", "searching")) }
                "response.web_search_call.completed" -> if (searching) emit(event("search_status", "done"))
                "response.code_interpreter_call_code.delta" -> {
                    val id = data.optString("item_id").ifBlank { "py" }
                    codes.getOrPut(id) { StringBuilder() }.append(data.optString("delta"))
                    emit(event("python", JSONObject().put("id", id).put("code", codes[id].toString())))
                }
                "response.output_item.done" -> data.optJSONObject("item")?.let { item ->
                    absorbItem(item)
                }
                "response.completed", "response.incomplete" -> {
                    val response = data.optJSONObject("response")
                    usage = response?.optJSONObject("usage")
                    val output = response?.optJSONArray("output")
                    if (output != null) {
                        for (index in 0 until output.length()) output.optJSONObject(index)?.let { absorbItem(it) }
                    }
                }
                "response.failed", "error" -> {
                    val error = data.optJSONObject("response")?.optJSONObject("error") ?: data.optJSONObject("error") ?: data
                    throw DirectApiException(0, error.optString("message").ifBlank { type })
                }
            }
            onProgress(content.toString(), thought.toString())
            true
        }
        val savedRemote = HashMap<String, String?>()
        fun resolveCitation(remote: RemoteImage, saved: String) {
            val name = remote.filename?.substringAfterLast('/') ?: return
            val rewritten = content.toString().replace("sandbox:/mnt/data/$name", saved)
                .replace("sandbox:/mnt/data/${name.replace(" ", "%20")}", saved)
            content.setLength(0)
            content.append(com.minashin1120.aiplayground.data.inlineGeneratedImageLinks(rewritten, listOf(saved)))
        }
        for (remote in remoteImages) {
            val key = remote.fileId ?: remote.url ?: continue
            if (savedRemote.containsKey(key)) {
                savedRemote[key]?.let { saved -> resolveCitation(remote, saved); remote.runId?.let { id -> runs[id]?.let { runs[id] = it.copy(images = (it.images + saved).distinct()) } } }
                continue
            }
            val url = when {
                remote.containerId != null && remote.fileId != null &&
                    CONTAINER_ID.matches(remote.containerId) && CONTAINER_FILE_ID.matches(remote.fileId) ->
                    "$baseUrl/v1/containers/${remote.containerId}/files/${remote.fileId}/content"
                !remote.url.isNullOrBlank() -> remote.url
                else -> continue
            }
            val bytes = if (url.startsWith("data:")) decodeInterpreterImage(url)
                ?: throw DirectApiException(0, "Pythonの画像出力を読み取れませんでした。") else
                http.download(url, mapOf("Authorization" to "Bearer ${request.apiKey}"), 20L * 1024 * 1024)
            if (bytes == null || bytes.isEmpty()) continue
            val ext = remote.filename?.substringAfterLast('.', "png")?.lowercase()?.takeIf { it.length in 3..4 } ?: "png"
            val file = DirectOutputFile("openai_ci_${savedRemote.size + 1}.$ext", "image/$ext", bytes)
            val saved = request.saveOutput?.invoke(file)
            savedRemote[key] = saved
            if (saved != null) resolveCitation(remote, saved)
            if (saved == null) files += file
            else if (remote.runId != null && runs.containsKey(remote.runId)) {
                val run = runs.getValue(remote.runId)
                runs[remote.runId] = run.copy(images = (run.images + saved).distinct())
            } else {
                val markdown = "\n![Agentic View]($saved)\n"
                content.append(markdown)
                emit(event("content", markdown))
            }
        }
        for ((id, run) in runs) {
            val markdown = pythonExecutionMarkdown(run)
            content.append(markdown); emit(event("content", markdown))
            emit(event("python", JSONObject().put("id", id).put("code", run.code).put("output", run.output)
                .put("images", JSONArray(run.images))))
        }
        if (remoteImages.isNotEmpty() || runs.isNotEmpty()) onProgress(content.toString(), thought.toString())
        if (sources.isNotEmpty()) {
            fun label(text: String) = text.replace("\n", " ").replace("\r", " ").replace("[", "\\[").replace("]", "\\]")
            val sourceText = "\n\n**Sources:**\n" + sources.entries.mapIndexed { index, (url, title) ->
                "- [${index + 1}] [${label(title)}](<${url.replace(">", "%3E").replace("<", "%3C").replace("\n", "").replace("\r", "")}>)"
            }.joinToString("\n") + "\n"
            content.append(sourceText)
            emit(event("content", sourceText))
            onProgress(content.toString(), thought.toString())
        }
        val meta = usage
        val reasoningTokens = meta?.optJSONObject("output_tokens_details")?.optInt("reasoning_tokens")?.takeIf { it > 0 }
        return DirectResult(content.toString(), thought.toString(),
            tokensIn = meta?.optInt("input_tokens")?.takeIf { it > 0 },
            tokensOut = meta?.optInt("output_tokens")?.takeIf { it > 0 },
            tokensThought = reasoningTokens, files = files)
    }

    private fun outputItem(item: JSONObject, codes: Map<String, StringBuilder>, content: StringBuilder,
                           files: MutableList<DirectOutputFile>, emit: (JSONObject) -> Unit) {
        when (item.optString("type")) {
            "code_interpreter_call" -> {
                val id = item.optString("id").ifBlank { "py" }
                val code = item.optString("code").ifBlank { codes[id]?.toString().orEmpty() }
                val outputs = item.optJSONArray("outputs") ?: JSONArray()
                val output = (0 until outputs.length()).mapNotNull { outputs.optJSONObject(it)?.optString("logs")?.takeIf { log -> log.isNotBlank() } }
                    .joinToString("\n")
                content.append("\n```python\n").append(code).append("\n```\n")
                if (output.isNotBlank()) content.append("\n**Output:**\n```\n").append(output).append("\n```\n")
                emit(event("python", JSONObject().put("id", id).put("code", code).put("output", output.ifBlank { "(no output)" })))
            }
            "image_generation_call" -> item.optString("result").takeIf { it.isNotBlank() }?.let { encoded ->
                runCatching { Base64.getDecoder().decode(encoded) }.getOrNull()?.let { bytes ->
                    files += DirectOutputFile("openai_image_${files.size + 1}.png", "image/png", bytes)
                }
            }
        }
    }

    internal fun buildPayload(request: DirectRequest, containerFileIds: List<String> = emptyList()): JSONObject {
        val options = request.options
        val model = request.model.lowercase()
        val input = JSONArray()
        request.turns.forEach { turn ->
            val parts = JSONArray()
            val assistant = turn.role == "assistant"
            if (!assistant) turn.attachments.forEach { attachment ->
                val bytes = attachment.bytes
                when {
                    bytes != null && attachment.mime.startsWith("image/") -> parts.put(JSONObject().put("type", "input_image")
                        .put("image_url", "data:${attachment.mime};base64," + Base64.getEncoder().encodeToString(bytes)))
                    bytes != null && attachment.mime == "application/pdf" && !xai -> parts.put(JSONObject().put("type", "input_file")
                        .put("filename", attachment.name).put("file_data", "data:application/pdf;base64," + Base64.getEncoder().encodeToString(bytes)))
                    attachment.text != null -> parts.put(JSONObject().put("type", "input_text").put("text", attachmentText(attachment)))
                }
            }
            if (turn.text.isNotEmpty() || parts.length() == 0) {
                parts.put(JSONObject().put("type", if (assistant) "output_text" else "input_text").put("text", turn.text))
            }
            input.put(JSONObject().put("role", if (assistant) "assistant" else "user").put("content", parts))
        }
        val payload = JSONObject().put("model", request.model).put("input", input).put("stream", true).put("store", false)
        if (request.system.isNotBlank()) payload.put("instructions", request.system)
        val tools = JSONArray()
        if (options.flag("enable_search")) {
            tools.put(JSONObject().put("type", "web_search"))
            if (xai) tools.put(JSONObject().put("type", "x_search"))
        }
        if (options.flag("enable_python") && !xai) {
            val container = JSONObject().put("type", "auto")
            if (containerFileIds.isNotEmpty()) container.put("file_ids", JSONArray(containerFileIds))
            tools.put(JSONObject().put("type", "code_interpreter").put("container", container))
            payload.put("include", JSONArray().put("code_interpreter_call.outputs"))
        }
        if (tools.length() > 0) payload.put("tools", tools)
        val effortOption = options.optString("reasoning_effort").lowercase().trim()
        val reasoningRequested = options.flag("enable_thinking") || (effortOption.isNotEmpty() && effortOption != "none")
        val reasoningModel = !xai && listOf("o1", "o3", "o4", "gpt-6", "gpt-5", "reasoning").any { model.contains(it) }
        if (reasoningModel && (reasoningRequested || (model == "gpt-6.1-terra" && effortOption == "none"))) {
            var effort = effortOption.ifBlank {
                when (options.optString("thinking_level").lowercase()) { "low" -> "low"; "high" -> "high"; else -> "medium" }
            }
            if (effort == "none" && listOf("gpt-5-mini", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.5-mini", "gpt-5.5-nano").any { model.contains(it) }) effort = "minimal"
            if (effort == "none" && (model.contains("gpt-6-astra") || model.contains("gpt-6.1-sol"))) effort = "medium"
            payload.put("reasoning", JSONObject().put("effort", effort).put("summary", "auto"))
        } else if (xai && reasoningRequested && effortOption.isNotEmpty() && effortOption != "none") {
            payload.put("reasoning", JSONObject().put("effort", effortOption))
        }
        return payload
    }

    /** Uploads images so the code interpreter container can open them. Vision still uses input_image. */
    private suspend fun uploadContainerImages(request: DirectRequest, emit: (JSONObject) -> Unit): List<String> {
        val lastUserTurn = request.turns.lastOrNull { it.role == "user" }
        val images = (lastUserTurn?.attachments ?: emptyList())
            .filter { attachment ->
                val bytes = attachment.bytes
                bytes != null && bytes.isNotEmpty() && attachment.mime.startsWith("image/") && bytes.size <= 50 * 1024 * 1024
            }
            .take(16)
        val prepared = images.mapNotNull { image -> containerImage(image.bytes ?: return@mapNotNull null, image.mime) }
        if (prepared.isEmpty()) return emptyList()
        emit(event("status", "画像をコード実行環境へ送っています..."))
        val ids = mutableListOf<String>()
        prepared.forEachIndexed { index, (bytes, ext) ->
            val mime = if (ext == "jpg") "image/jpeg" else "image/$ext"
            val form = MultipartBody.Builder().setType(MultipartBody.FORM)
                .addFormDataPart("purpose", "user_data")
                .addFormDataPart("file", "input_${index + 1}.$ext", bytes.toRequestBody(mime.toMediaType()))
                .build()
            // A failed upload must not silently turn Python image analysis into vision-only input.
            // Let DirectHttp report the failure and preserve coroutine cancellation.
            val (body, _) = http.postBytes(
                "$baseUrl/v1/files",
                mapOf("Authorization" to "Bearer ${request.apiKey}"),
                form,
            )
            val uploaded = JSONObject(body.toString(Charsets.UTF_8)).optString("id")
            if (uploaded.isBlank()) throw DirectApiException(0, "画像をコード実行環境へ送れませんでした。再試行してください。")
            ids += uploaded
        }
        return ids
    }

    /** PNG, JPEG, and GIF are container files as-is. Other images are converted to PNG. */
    private fun containerImage(bytes: ByteArray, mime: String): Pair<ByteArray, String>? {
        when (mime.substringBefore(';').lowercase()) {
            "image/png" -> return bytes to "png"
            "image/jpeg", "image/jpg" -> return bytes to "jpg"
            "image/gif" -> return bytes to "gif"
        }
        val bitmap = runCatching { BitmapFactory.decodeByteArray(bytes, 0, bytes.size) }.getOrNull() ?: return null
        return try {
            val out = ByteArrayOutputStream()
            if (!bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)) return null
            val png = out.toByteArray()
            if (png.isEmpty() || png.size > 50 * 1024 * 1024) null else png to "png"
        } finally {
            runCatching { bitmap.recycle() }
        }
    }

    private data class RemoteImage(
        val url: String? = null,
        val containerId: String? = null,
        val fileId: String? = null,
        val filename: String? = null,
        val runId: String? = null,
    )

    internal companion object {
        fun decodeInterpreterImage(url: String): ByteArray? {
            val match = Regex("^data:image/(?:png|jpeg|webp|gif);base64,([A-Za-z0-9+/=\\r\\n]+)$").matchEntire(url) ?: return null
            if (match.groupValues[1].length > 28 * 1024 * 1024) return null
            return runCatching { Base64.getMimeDecoder().decode(match.groupValues[1]) }.getOrNull()
                ?.takeIf { it.isNotEmpty() && it.size <= 20 * 1024 * 1024 }
        }

        private val CONTAINER_ID = Regex("cntr_[A-Za-z0-9]+")
        private val CONTAINER_FILE_ID = Regex("cfile_[A-Za-z0-9]+")

        private fun collectContainerCitation(annotation: JSONObject, containerId: String, into: MutableList<RemoteImage>): Boolean {
            if (annotation.optString("type") != "container_file_citation") return false
            val fileId = annotation.optString("file_id")
            val container = annotation.optString("container_id").ifBlank { containerId }
            if (fileId.isNotBlank() && container.isNotBlank()) {
                into += RemoteImage(containerId = container, fileId = fileId, filename = annotation.optString("filename").ifBlank { null })
            }
            return true
        }
    }
}
