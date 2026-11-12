package com.minashin1120.aiplayground

import android.webkit.MimeTypeMap
import com.minashin1120.aiplayground.data.Diagnostics
import com.minashin1120.aiplayground.data.OfflineCacheStore
import com.minashin1120.aiplayground.data.PlaygroundApi
import com.minashin1120.aiplayground.data.ServerOrigin
import com.minashin1120.aiplayground.data.fileReferencePath
import com.minashin1120.aiplayground.data.local.LocalChatStore
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.util.UUID

/**
 * Reads attachments for previews and saves them to a file: same-origin files (with the session
 * token), device files of a local profile, and external https images in answers. Successful reads
 * are copied to the offline cache. This class only reads state; the view model owns updates.
 */
internal class ChatAttachmentFiles(
    private val state: StateFlow<ChatState>,
    private val store: () -> LocalChatStore?,
    private val cache: OfflineCacheStore,
    private val api: PlaygroundApi,
    private val sessionToken: () -> String?,
    private val token: () -> String,
    private val cacheDir: File,
    private val refreshCacheStats: suspend (Int?) -> Unit,
) {
    /** Account/profile-scoped decoded-image identity; safe to call from composition without IO. */
    fun imageCacheKey(reference: String): String {
        val identity = store()?.imageCacheReference(reference) ?: fileReferencePath(reference) ?: reference
        val profile = if (state.value.localProfile) "local" else state.value.account?.id.toString()
        return "${ServerOrigin.current}|$profile|$identity"
    }

    /** Loads same-origin attachment bytes for preview; returns null when unavailable. */
    suspend fun loadBytes(reference: String, thumbnail: Boolean, limit: Long = 8L * 1024 * 1024): ByteArray? {
        if (state.value.banned) return null
        // Images on other sites in answers load like Web `<img>`: without the token, https only.
        if (reference.startsWith("https://") && fileReferencePath(reference) == null) {
            if (state.value.offline) return null
            return withContext(Dispatchers.IO) { runCatching { api.loadExternalImage(reference, limit) }.getOrNull() }
        }
        if (LocalChatStore.isLocalReference(reference)) {
            val chats = store() ?: return null
            val bytes = withContext(Dispatchers.IO) { chats.loadFile(reference, maxOf(limit, 64L * 1024 * 1024)) }
            if (bytes != null) return bytes
            val remote = withContext(Dispatchers.IO) { chats.serverFileReference(reference) }
                ?.takeIf { !LocalChatStore.isLocalReference(it) && fileReferencePath(it) != null }
            Diagnostics.log("store.file_read", "ok" to false, "retry" to (remote != null))
            if (remote == null) return null
            return loadBytes(remote, thumbnail, limit)
        }
        val accountId = state.value.account?.id ?: return null
        val cached = withContext(Dispatchers.IO) { cache.loadFile(accountId, reference, thumbnail, limit) }
        if (cached != null) return cached.bytes
        val active = sessionToken() ?: return null
        return withContext(Dispatchers.IO) {
            try {
                val bytes = api.loadFileBytes(reference, active, thumbnail, limit)
                cache.saveFile(accountId, reference, thumbnail, bytes, mimeFor(reference))
                refreshCacheStats(accountId)
                bytes
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) { null }
        }
    }

    /** Attachment bytes from the offline cache (serverless mode answering while the server is out of reach). */
    fun cachedFileBytes(reference: String, limit: Long): ByteArray? {
        val accountId = state.value.account?.id ?: return null
        val target = File(cacheDir, "serverless-" + UUID.randomUUID())
        return try {
            cache.materializeFile(accountId, reference, target) ?: return null
            if (target.length() > limit) null else target.readBytes()
        } finally { target.delete() }
    }

    /** Writes an attachment to a cache file and returns it with its MIME type. */
    suspend fun download(reference: String): Pair<File, String> {
        val accountId = state.value.account?.id ?: throw IOException("アカウント情報がありません。")
        val directory = File(cacheDir, "shared").apply { mkdirs() }
        val suffix = reference.substringBefore('?').substringAfterLast('.', "bin").take(8).filter { it.isLetterOrDigit() }.ifBlank { "bin" }
        val target = File(directory, "${UUID.randomUUID()}.$suffix")
        if (LocalChatStore.isLocalReference(reference)) {
            val chats = store() ?: throw IOException("この添付は別のプロファイルの端末内ファイルです。")
            val mime = withContext(Dispatchers.IO) { chats.materializeFile(reference, target) }
            if (mime != null) return target to mime
            val remote = withContext(Dispatchers.IO) { chats.serverFileReference(reference) }
                ?.takeIf { !LocalChatStore.isLocalReference(it) && fileReferencePath(it) != null } ?: throw IOException("添付を開けません。")
            return download(remote)
        }
        val cachedMime = withContext(Dispatchers.IO) { cache.materializeFile(accountId, reference, target) }
        if (cachedMime != null) return target to cachedMime
        val mime = withContext(Dispatchers.IO) {
            val result = api.download(reference, target, token())
            cache.saveFileFromFile(accountId, reference, target, result)
            refreshCacheStats(accountId)
            result
        }
        return target to mime
    }

    private fun mimeFor(reference: String): String {
        val extension = reference.substringBefore('?').substringAfterLast('.', "").lowercase()
        return MimeTypeMap.getSingleton().getMimeTypeFromExtension(extension) ?: "application/octet-stream"
    }
}
