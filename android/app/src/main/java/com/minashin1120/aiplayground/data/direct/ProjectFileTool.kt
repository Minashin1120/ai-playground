package com.minashin1120.aiplayground.data.direct

import org.json.JSONArray
import org.json.JSONObject
import java.util.Locale

/**
 * Server `projects.py` equivalent for serverless mode: the model sees the current project's shared
 * file list and calls `read_project_file` on demand to load text chunks or (only when the project's
 * `use_vision_for_images` toggle is ON) Vision-analyzed image descriptions.
 */
object ProjectFileTool {
    const val NAME = "read_project_file"
    const val DEFAULT_LIMIT = 20_000
    const val MAX_LIMIT = 30_000

    private const val DESCRIPTION =
        "Read the contents of one of the current project's shared files. " +
            "Call this when the user asks about a project file or when a listed file appears relevant. " +
            "Use the exact file name from the project files list. For long files, pass offset to read subsequent chunks. " +
            "Never invent a file name that is not listed."

    private val parameters = JSONObject()
        .put("type", "object")
        .put("properties", JSONObject()
            .put("name", JSONObject()
                .put("type", "string")
                .put("description", "Exact file name from the project shared files list."))
            .put("offset", JSONObject()
                .put("type", "integer")
                .put("description", "Optional zero-based character offset to resume reading a long file."))
            .put("limit", JSONObject()
                .put("type", "integer")
                .put("description", "Optional max characters to return (default 20000, max 30000).")))
        .put("required", JSONArray().put("name"))

    fun openAiChatDefinition(): JSONObject = JSONObject()
        .put("type", "function")
        .put("function", JSONObject().put("name", NAME).put("description", DESCRIPTION).put("parameters", parameters))

    fun openAiResponsesDefinition(): JSONObject = JSONObject()
        .put("type", "function").put("name", NAME).put("description", DESCRIPTION).put("parameters", parameters)

    fun anthropicDefinition(): JSONObject = JSONObject()
        .put("name", NAME).put("description", DESCRIPTION).put("input_schema", parameters)

    fun geminiDeclaration(): JSONObject = JSONObject()
        .put("name", NAME).put("description", DESCRIPTION).put("parameters", parameters)

    fun formatSize(bytes: Long): String = when {
        bytes < 1024L -> "$bytes B"
        bytes < 1024L * 1024L -> String.format(Locale.US, "%.1f KB", bytes / 1024.0)
        else -> String.format(Locale.US, "%.1f MB", bytes / (1024.0 * 1024.0))
    }

    fun buildCatalog(files: JSONArray?, useVisionForImages: Boolean): String {
        val rows = files ?: return ""
        val lines = mutableListOf<String>()
        for (i in 0 until rows.length()) {
            val row = rows.optJSONObject(i) ?: continue
            val name = row.optString("filename").ifBlank { row.optString("name") }.trim()
            if (name.isEmpty()) continue
            val isImage = row.optBoolean("is_image") || row.optString("type") == "image"
            val sizeLabel = formatSize(row.optLong("size", 0L))
            val meta = if (isImage) {
                val visionTag = if (useVisionForImages) "Vision: ON" else "Vision: OFF (metadata only)"
                "image, $sizeLabel, $visionTag"
            } else {
                val mime = row.optString("mime").ifBlank { "file" }
                "$mime, $sizeLabel"
            }
            val note = row.optString("note").replace("\r\n", "\n").trim()
            lines += if (note.isNotEmpty()) "- $name ($meta): $note" else "- $name ($meta)"
        }
        return lines.joinToString("\n")
    }

    fun renderNotice(template: String, projectName: String, files: JSONArray?, useVisionForImages: Boolean): String {
        val rows = files ?: return ""
        if (rows.length() == 0) return ""
        val catalog = buildCatalog(rows, useVisionForImages)
        if (catalog.isBlank()) return ""
        var rendered = template.replace("\r\n", "\n").trim()
        var replacedFiles = false
        listOf("{{project_files}}", "{project_files}").forEach { token ->
            if (rendered.contains(token)) { rendered = rendered.replace(token, catalog); replacedFiles = true }
        }
        listOf("{{project_file_count}}", "{project_file_count}").forEach { token ->
            if (rendered.contains(token)) rendered = rendered.replace(token, rows.length().toString())
        }
        listOf("{{project_name}}", "{project_name}").forEach { token ->
            if (rendered.contains(token)) rendered = rendered.replace(token, projectName)
        }
        if (!replacedFiles) rendered = if (rendered.isBlank()) catalog else "$rendered\n\n$catalog"
        return rendered.trim()
    }

    suspend fun execute(
        project: JSONObject,
        name: String,
        offset: Int,
        limit: Int,
        loadBytes: suspend (filepath: String) -> ByteArray?,
        visionAnalyze: (suspend (bytes: ByteArray, mime: String, filename: String) -> String?)? = null,
    ): String {
        val files = project.optJSONArray("files") ?: JSONArray()
        val query = name.trim()
        var target: JSONObject? = null
        for (i in 0 until files.length()) {
            val row = files.optJSONObject(i) ?: continue
            val fn = row.optString("filename").ifBlank { row.optString("name") }
            if (fn == query) { target = row; break }
        }
        if (target == null && query.isNotEmpty()) {
            for (i in 0 until files.length()) {
                val row = files.optJSONObject(i) ?: continue
                val fn = row.optString("filename").ifBlank { row.optString("name") }
                if (fn.equals(query, ignoreCase = true)) { target = row; break }
            }
        }
        if (target == null) {
            val available = (0 until files.length()).mapNotNull {
                files.optJSONObject(it)?.optString("filename")?.takeIf { n -> n.isNotBlank() }
            }
            val suffix = if (available.isEmpty()) "" else " Available project files: ${available.joinToString(", ")}"
            return "Error: no project file named \"$query\".$suffix"
        }
        val filename = target.optString("filename").ifBlank { target.optString("name") }
        val filepath = target.optString("filepath")
        val mime = target.optString("mime").ifBlank { "application/octet-stream" }
        val sizeLabel = formatSize(target.optLong("size", 0L))
        val isImage = target.optBoolean("is_image") || target.optString("type") == "image" || mime.startsWith("image/")
        val safeOffset = offset.coerceAtLeast(0)
        val safeLimit = limit.coerceIn(500, MAX_LIMIT)

        if (isImage) {
            if (!project.optBoolean("use_vision_for_images", false)) {
                return "Project image file: $filename (mime=$mime, size=$sizeLabel).\n" +
                    "Vision model analysis is disabled for this project, so the image contents were not read. " +
                    "Ask the user to enable Vision model analysis in the project settings or attach the image directly to the chat."
            }
            val bytes = loadBytes(filepath) ?: return "Error: project image file \"$filename\" could not be loaded."
            val description = visionAnalyze?.invoke(bytes, mime, filename)?.trim()
                ?: return "Project image file: $filename (mime=$mime, size=$sizeLabel).\nVision model analysis failed (check Vision model API key)."
            return sliceChunk("Project image file (Vision analysis): $filename", description, safeOffset, safeLimit)
        }

        val bytes = loadBytes(filepath) ?: return "Error: project file \"$filename\" could not be loaded."
        val prepared = AttachmentExtractor.prepare(filename, mime, bytes)
        val extracted = prepared.text
            ?: return "Project file: $filename (mime=$mime, size=$sizeLabel).\nThis file format cannot be extracted as text via read_project_file."
        if (extracted.isEmpty()) {
            return "Project file: $filename ($sizeLabel) is empty or contains no extractable text."
        }
        return sliceChunk("Project file: $filename ($sizeLabel)", extracted, safeOffset, safeLimit)
    }

    private fun sliceChunk(prefix: String, fullText: String, offset: Int, limit: Int): String {
        val total = fullText.length
        val start = offset.coerceAtMost(total)
        val end = (start + limit).coerceAtMost(total)
        val chunk = fullText.substring(start, end)
        val hasMore = end < total
        val moreHint = if (hasMore) " (more available; call read_project_file with offset=$end)" else ""
        return "$prefix [chars $start-$end of $total]$moreHint\n\n$chunk"
    }
}
