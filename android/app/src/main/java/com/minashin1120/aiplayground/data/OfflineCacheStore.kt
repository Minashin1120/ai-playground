package com.minashin1120.aiplayground.data

import android.content.Context
import com.minashin1120.aiplayground.data.secure.EncryptedFileStore
import okhttp3.HttpUrl
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.io.InputStream

enum class CacheCategory { CHAT_HISTORY, FILES }

enum class HistoryCacheMode(val value: String) {
    VIEWED("viewed"),
    FULL("full");

    companion object {
        fun from(value: String?): HistoryCacheMode = entries.firstOrNull { it.value == value } ?: VIEWED
    }
}

data class OfflineCacheStats(
    val historyBytes: Long = 0L,
    val fileBytes: Long = 0L,
) {
    val totalBytes: Long get() = historyBytes + fileBytes
}

data class CachedFile(val bytes: ByteArray, val mime: String)

/**
 * Folder key of one account's offline data. The official server keeps the historical `accountId` key
 * (no migration of existing caches); other servers add the origin so user 1 on two servers never shares
 * a folder.
 */
fun offlineAccountScope(accountId: Int, origin: HttpUrl): String =
    if (ServerOrigin.isOfficial(origin)) accountId.toString() else "${originLabel(origin)}#$accountId"

/**
 * Encrypted, account-scoped storage for data that has already been displayed in the app.
 * This intentionally uses noBackupFilesDir rather than cacheDir: Android may evict cacheDir
 * at any time, while these records must remain available for offline viewing.
 */
class OfflineCacheStore(private val context: Context) {
    private val root = File(context.noBackupFilesDir, "offline-cache")
    // v1 keys were generated with Android Keystore's default randomized-IV
    // requirement, which rejects the stored IV during GCM decryption. A new
    // alias is required because that policy cannot be changed on an existing
    // Keystore key.
    private val crypto = EncryptedFileStore(MAGIC, EncryptedFileStore.keystoreKey("ai-playground-offline-cache-v2"))

    private fun accountDir(accountId: Int): File = File(root, digest(offlineAccountScope(accountId, ServerOrigin.current)).take(32))
    private fun fileDir(accountId: Int): File = File(accountDir(accountId), "files")

    @Synchronized
    fun saveAccount(accountId: Int, payload: JSONObject) {
        writeJson(File(accountDir(accountId), "account.enc"), payload)
    }

    @Synchronized
    fun loadAccount(accountId: Int): JSONObject? = readJson(File(accountDir(accountId), "account.enc"))

    @Synchronized
    fun savePreferences(accountId: Int, payload: JSONObject) {
        writeJson(File(accountDir(accountId), "preferences.enc"), payload)
    }

    @Synchronized
    fun loadPreferences(accountId: Int): JSONObject? = readJson(File(accountDir(accountId), "preferences.enc"))

    @Synchronized
    fun saveThreads(accountId: Int, threads: List<ThreadItem>, merge: Boolean = true) {
        val values = if (merge) {
            val existing = loadThreads(accountId)
            val byId = LinkedHashMap<String, ThreadItem>()
            (existing + threads).forEach { byId[it.id] = it }
            byId.values.toList()
        } else threads
        val rows = JSONArray()
        values.forEach { rows.put(threadJson(it)) }
        writeJson(File(accountDir(accountId), "threads.enc"), JSONObject().put("threads", rows))
    }

    @Synchronized
    fun loadThreads(accountId: Int): List<ThreadItem> {
        val rows = readJson(File(accountDir(accountId), "threads.enc"))?.optJSONArray("threads") ?: return emptyList()
        return (0 until rows.length()).mapNotNull { index ->
            rows.optJSONObject(index)?.let { row ->
                runCatching { parseThread(row) }.getOrNull()
            }
        }
    }

    @Synchronized
    fun saveThread(
        accountId: Int,
        thread: ThreadItem,
        messages: List<ChatMessage>,
        hasOlder: Boolean,
        oldestLoadedId: String?,
        customInstruction: String,
        includeGlobalInstruction: Boolean,
        temporaryRemainingSeconds: Long?,
        leafId: Int?,
    ) {
        val path = File(accountDir(accountId), "thread-${digest(thread.id)}.enc")
        val previous = readJson(path)
        val merged = LinkedHashMap<String, ChatMessage>()
        previous?.optJSONArray("messages")?.let { rows ->
            (0 until rows.length()).forEach { index ->
                rows.optJSONObject(index)?.let { row -> runCatching { parseMessage(row) }.getOrNull()?.let { merged[it.id] = it } }
            }
        }
        messages.forEach { merged[it.id] = it }
        val rows = JSONArray()
        merged.values.forEach { rows.put(messageJson(it)) }
        val payload = JSONObject()
            .put("thread", threadJson(thread))
            .put("messages", rows)
            .put("has_older_messages", hasOlder)
            .put("oldest_loaded_id", oldestLoadedId ?: JSONObject.NULL)
            .put("custom_instruction", customInstruction)
            .put("include_global_instruction", includeGlobalInstruction)
            .put("temp_chat_remaining_seconds", temporaryRemainingSeconds ?: JSONObject.NULL)
            .put("leaf_id", leafId ?: JSONObject.NULL)
        writeJson(path, payload)
    }

    @Synchronized
    fun loadThread(accountId: Int, threadId: String): JSONObject? =
        readJson(File(accountDir(accountId), "thread-${digest(threadId)}.enc"))

    @Synchronized
    fun deleteThread(accountId: Int, threadId: String) {
        File(accountDir(accountId), "thread-${digest(threadId)}.enc").delete()
        val remaining = loadThreads(accountId).filterNot { it.id == threadId }
        saveThreads(accountId, remaining, merge = false)
    }

    @Synchronized
    fun saveLibrary(accountId: Int, files: List<LibraryFile>, merge: Boolean = true) {
        val values = if (merge) {
            val byPath = LinkedHashMap<String, LibraryFile>()
            (loadLibrary(accountId) + files).forEach { byPath[it.filepath] = it }
            byPath.values.toList()
        } else files
        val rows = JSONArray()
        values.forEach { rows.put(libraryJson(it)) }
        writeJson(File(accountDir(accountId), "library.enc"), JSONObject().put("files", rows))
    }

    @Synchronized
    fun loadLibrary(accountId: Int): List<LibraryFile> {
        val rows = readJson(File(accountDir(accountId), "library.enc"))?.optJSONArray("files") ?: return emptyList()
        return (0 until rows.length()).mapNotNull { index ->
            rows.optJSONObject(index)?.let { row -> runCatching { parseLibraryFile(row) }.getOrNull() }
        }
    }

    @Synchronized
    fun deleteLibraryFile(accountId: Int, filepath: String) {
        saveLibrary(accountId, loadLibrary(accountId).filterNot { it.filepath == filepath }, merge = false)
    }

    @Synchronized
    fun saveFile(accountId: Int, reference: String, thumbnail: Boolean, bytes: ByteArray, mime: String) {
        if (bytes.isEmpty()) return
        val target = File(fileDir(accountId), "${digest(reference + if (thumbnail) "|thumb" else "|full")}.enc")
        writeEncrypted(target, bytes.inputStream())
        updateFileIndex(accountId, reference, thumbnail, mime, bytes.size.toLong())
    }

    @Synchronized
    fun saveFileFromFile(accountId: Int, reference: String, source: File, mime: String) {
        if (!source.isFile || source.length() == 0L) return
        val target = File(fileDir(accountId), "${digest(reference + "|full")}.enc")
        FileInputStream(source).use { writeEncrypted(target, it) }
        updateFileIndex(accountId, reference, false, mime, source.length())
    }

    @Synchronized
    fun loadFile(accountId: Int, reference: String, thumbnail: Boolean, limit: Long): CachedFile? {
        val index = loadFileIndex(accountId).firstOrNull { it.optString("reference") == reference && it.optBoolean("thumbnail") == thumbnail }
            ?: return null
        val target = File(fileDir(accountId), index.optString("filename"))
        if (!target.isFile) return null
        val bytes = runCatching { decryptToBytes(target, limit) }.getOrNull() ?: return null
        return CachedFile(bytes, index.optString("mime", "application/octet-stream"))
    }

    @Synchronized
    fun materializeFile(accountId: Int, reference: String, destination: File): String? {
        val index = loadFileIndex(accountId).firstOrNull { it.optString("reference") == reference && !it.optBoolean("thumbnail") }
            ?: return null
        val source = File(fileDir(accountId), index.optString("filename"))
        if (!source.isFile) return null
        return runCatching {
            destination.parentFile?.mkdirs()
            decryptToFile(source, destination)
            index.optString("mime", "application/octet-stream")
        }.getOrElse {
            destination.delete()
            null
        }
    }

    @Synchronized
    fun clear(accountId: Int, category: CacheCategory) {
        val dir = accountDir(accountId)
        when (category) {
            CacheCategory.CHAT_HISTORY -> {
                File(dir, "threads.enc").delete()
                dir.listFiles()?.filter { it.name.startsWith("thread-") && it.name.endsWith(".enc") }
                    ?.forEach { it.delete() }
            }
            CacheCategory.FILES -> {
                File(dir, "library.enc").delete()
                File(dir, "file-index.enc").delete()
                fileDir(accountId).deleteRecursively()
            }
        }
    }

    @Synchronized
    fun stats(accountId: Int): OfflineCacheStats {
        val dir = accountDir(accountId)
        val history = listOf(File(dir, "threads.enc")) +
            (dir.listFiles()?.filter { it.name.startsWith("thread-") && it.name.endsWith(".enc") }.orEmpty())
        val files = listOf(File(dir, "library.enc"), File(dir, "file-index.enc")) +
            (fileDir(accountId).listFiles()?.toList().orEmpty())
        return OfflineCacheStats(history.sumOf { if (it.isFile) it.length() else 0L }, files.sumOf { if (it.isFile) it.length() else 0L })
    }

    private fun loadFileIndex(accountId: Int): List<JSONObject> {
        val rows = readJson(File(accountDir(accountId), "file-index.enc"))?.optJSONArray("files") ?: return emptyList()
        return (0 until rows.length()).mapNotNull { rows.optJSONObject(it) }
    }

    private fun updateFileIndex(accountId: Int, reference: String, thumbnail: Boolean, mime: String, size: Long) {
        val entries = loadFileIndex(accountId).filterNot {
            it.optString("reference") == reference && it.optBoolean("thumbnail") == thumbnail
        }.toMutableList()
        val filename = "${digest(reference + if (thumbnail) "|thumb" else "|full")}.enc"
        entries += JSONObject().put("reference", reference).put("thumbnail", thumbnail)
            .put("mime", mime).put("size", size).put("filename", filename)
        val rows = JSONArray()
        entries.forEach { rows.put(it) }
        writeJson(File(accountDir(accountId), "file-index.enc"), JSONObject().put("files", rows))
    }

    private fun threadJson(thread: ThreadItem): JSONObject = JSONObject()
        .put("id", thread.id).put("title", thread.title).put("model", thread.model)
        .put("is_bookmarked", thread.isBookmarked).put("is_temporary", thread.isTemporary)

    private fun parseThread(row: JSONObject): ThreadItem = ThreadItem(
        row.getString("id"), row.optString("title"), row.optString("model"),
        row.optBoolean("is_bookmarked"), row.optBoolean("is_temporary"),
    )

    private fun messageJson(message: ChatMessage): JSONObject = JSONObject()
        .put("id", message.id).put("role", message.role).put("content", message.content)
        .put("thought_data", message.thought).put("image_url", JSONArray(message.files).toString())
        .put("parent_id", message.parentId ?: JSONObject.NULL).put("model", message.model)
        .put("tokens", message.tokens ?: JSONObject.NULL).put("tokens_in", message.tokensIn ?: JSONObject.NULL)
        .put("tokens_out", message.tokensOut ?: JSONObject.NULL).put("tokens_content", message.tokensContent ?: JSONObject.NULL)
        .put("tokens_thought", message.tokensThought ?: JSONObject.NULL)
        .put("is_encrypted", message.encrypted ?: JSONObject.NULL)
        .put("quote_text", message.quote).put("gem_name", message.gemName).put("created_at", message.createdAt)
        .put("prompt_options", message.promptOptions ?: JSONObject.NULL)

    private fun parseMessage(row: JSONObject): ChatMessage {
        val files = row.optJSONArray("files")?.let { values ->
            (0 until values.length()).map { values.getString(it) }
        } ?: row.optString("image_url").takeIf { it.isNotBlank() }?.let { value ->
            runCatching { JSONArray(value).let { values -> (0 until values.length()).map { values.getString(it) } } }
                .getOrElse { listOf(value) }
        }.orEmpty()
        return ChatMessage(row.getString("id"), row.optString("role"), row.optString("content"),
            row.optString("thought").ifBlank { row.optString("thought_data") }, files,
            if (row.isNull("parent_id")) null else row.optInt("parent_id"), row.optString("model"),
            tokens = row.nullableInt("tokens"), tokensIn = row.nullableInt("tokens_in"),
            tokensOut = row.nullableInt("tokens_out"), tokensContent = row.nullableInt("tokens_content"),
            tokensThought = row.nullableInt("tokens_thought"),
            encrypted = if (row.has("is_encrypted") && !row.isNull("is_encrypted")) row.optBoolean("is_encrypted") else null,
            quote = row.optString("quote_text"), gemName = row.optString("gem_name"),
            createdAt = row.optString("created_at"), promptOptions = row.optJSONObject("prompt_options"))
    }

    private fun libraryJson(file: LibraryFile): JSONObject = JSONObject()
        .put("display_name", file.displayName).put("filepath", file.filepath).put("url", file.url)
        .put("thumbnail_url", file.thumbnailUrl).put("type", file.type).put("ext", file.ext)
        .put("is_favorite", file.isFavorite).put("timestamp", file.timestamp)

    private fun parseLibraryFile(row: JSONObject): LibraryFile = LibraryFile(
        row.optString("display_name"), row.getString("filepath"), row.optString("url"),
        row.optString("thumbnail_url"), row.optString("type"), row.optString("ext"),
        row.optBoolean("is_favorite"), row.optLong("timestamp"),
    )

    private fun writeJson(file: File, payload: JSONObject) = crypto.writeJson(file, payload)

    private fun readJson(file: File): JSONObject? = crypto.readJson(file)

    private fun writeEncrypted(target: File, input: InputStream) = crypto.writeStream(target, input)

    private fun decryptToBytes(file: File, limit: Long): ByteArray = crypto.readBytes(file, limit)

    private fun decryptToFile(source: File, destination: File) = crypto.decryptToFile(source, destination)

    private fun digest(value: String): String = EncryptedFileStore.sha256Hex(value)

    private companion object {
        val MAGIC = byteArrayOf('O'.code.toByte(), 'F'.code.toByte(), 'C'.code.toByte(), 1)
    }
}
