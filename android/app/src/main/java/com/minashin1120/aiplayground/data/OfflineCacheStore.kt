package com.minashin1120.aiplayground.data

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.security.KeyStore
import java.security.MessageDigest
import java.security.SecureRandom
import javax.crypto.Cipher
import javax.crypto.CipherInputStream
import javax.crypto.CipherOutputStream
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

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
 * Encrypted, account-scoped storage for data that has already been displayed in the app.
 * This intentionally uses noBackupFilesDir rather than cacheDir: Android may evict cacheDir
 * at any time, while these records must remain available for offline viewing.
 */
class OfflineCacheStore(private val context: Context) {
    private val root = File(context.noBackupFilesDir, "offline-cache")
    private val keyAlias = "ai-playground-offline-cache-v1"
    private val random = SecureRandom()

    private fun accountDir(accountId: Int): File = File(root, digest(accountId.toString()).take(32))
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

    private fun parseMessage(row: JSONObject): ChatMessage {
        val files = row.optJSONArray("files")?.let { values ->
            (0 until values.length()).map { values.getString(it) }
        } ?: row.optString("image_url").takeIf { it.isNotBlank() }?.let { value ->
            runCatching { JSONArray(value).let { values -> (0 until values.length()).map { values.getString(it) } } }
                .getOrElse { listOf(value) }
        }.orEmpty()
        return ChatMessage(row.getString("id"), row.optString("role"), row.optString("content"),
            row.optString("thought").ifBlank { row.optString("thought_data") }, files,
            if (row.isNull("parent_id")) null else row.optInt("parent_id"), row.optString("model"))
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

    private fun writeJson(file: File, payload: JSONObject) = writeEncrypted(file, payload.toString().toByteArray(Charsets.UTF_8).inputStream())

    private fun readJson(file: File): JSONObject? = runCatching {
        if (!file.isFile) return null
        JSONObject(String(decryptToBytes(file, 16L * 1024 * 1024), Charsets.UTF_8))
    }.getOrNull()

    private fun writeEncrypted(target: File, input: InputStream) {
        target.parentFile?.mkdirs()
        val temporary = File(target.parentFile, "${target.name}.part")
        val iv = ByteArray(12).also(random::nextBytes)
        val cipher = Cipher.getInstance("AES/GCM/NoPadding").apply {
            init(Cipher.ENCRYPT_MODE, key(), GCMParameterSpec(128, iv))
        }
        try {
            FileOutputStream(temporary).buffered().use { output ->
                output.write(MAGIC)
                output.write(iv)
                CipherOutputStream(output, cipher).use { encrypted -> input.copyTo(encrypted, 64 * 1024) }
            }
            if (target.exists() && !target.delete()) throw IOException("キャッシュを更新できません。")
            if (!temporary.renameTo(target)) throw IOException("キャッシュを保存できません。")
        } finally {
            input.close()
            temporary.delete()
        }
    }

    private fun decryptToBytes(file: File, limit: Long): ByteArray {
        FileInputStream(file).use { input ->
            openDecrypted(input).use { decrypted ->
                val output = ByteArrayOutputStream()
                val buffer = ByteArray(64 * 1024)
                var total = 0L
                while (true) {
                    val count = decrypted.read(buffer)
                    if (count < 0) break
                    total += count
                    if (total > limit) throw IOException("キャッシュされた添付が上限を超えています。")
                    output.write(buffer, 0, count)
                }
                return output.toByteArray()
            }
        }
    }

    private fun decryptToFile(source: File, destination: File) {
        FileInputStream(source).use { input ->
            openDecrypted(input).use { decrypted ->
                FileOutputStream(destination).use { output -> decrypted.copyTo(output, 64 * 1024) }
            }
        }
    }

    private fun openDecrypted(input: InputStream): CipherInputStream {
        val magic = ByteArray(MAGIC.size)
        if (!readFully(input, magic) || !magic.contentEquals(MAGIC)) throw IOException("キャッシュ形式が不正です。")
        val iv = ByteArray(12)
        if (!readFully(input, iv)) throw IOException("キャッシュの暗号化情報が不正です。")
        val cipher = Cipher.getInstance("AES/GCM/NoPadding").apply {
            init(Cipher.DECRYPT_MODE, key(), GCMParameterSpec(128, iv))
        }
        return CipherInputStream(input, cipher)
    }

    private fun readFully(input: InputStream, buffer: ByteArray): Boolean {
        var offset = 0
        while (offset < buffer.size) {
            val count = input.read(buffer, offset, buffer.size - offset)
            if (count < 0) return false
            offset += count
        }
        return true
    }

    private fun key(): SecretKey {
        val store = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        (store.getKey(keyAlias, null) as? SecretKey)?.let { return it }
        return KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore").apply {
            init(KeyGenParameterSpec.Builder(keyAlias, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256)
                .build())
        }.generateKey()
    }

    private fun digest(value: String): String = MessageDigest.getInstance("SHA-256")
        .digest(value.toByteArray(Charsets.UTF_8)).joinToString("") { "%02x".format(it.toInt() and 0xff) }

    private companion object {
        val MAGIC = byteArrayOf('O'.code.toByte(), 'F'.code.toByte(), 'C'.code.toByte(), 1)
    }
}
