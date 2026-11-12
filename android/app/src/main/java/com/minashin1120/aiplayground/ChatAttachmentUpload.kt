package com.minashin1120.aiplayground

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import com.minashin1120.aiplayground.data.Attachment
import com.minashin1120.aiplayground.data.CompressionSettings
import com.minashin1120.aiplayground.data.Diagnostics
import com.minashin1120.aiplayground.data.PlaygroundApi
import com.minashin1120.aiplayground.data.local.LocalChatBackend
import com.minashin1120.aiplayground.data.compressImage
import com.minashin1120.aiplayground.data.extensionForMime
import com.minashin1120.aiplayground.data.local.LocalChatStore
import com.minashin1120.aiplayground.data.markedFileName
import com.minashin1120.aiplayground.data.sync.SyncEngine
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import okio.BufferedSink
import org.json.JSONObject
import java.io.IOException
import java.io.InputStream
import java.util.ArrayDeque

/** 添付アップロードのしきい値・一時データ・読み取り／転送補助（ChatViewModel から分離）。 */
internal const val CHAT_UPLOAD_CHUNK_THRESHOLD_BYTES = 8L * 1024 * 1024
internal const val CHAT_UPLOAD_MAX_SINGLE_BYTES = 64L * 1024 * 1024

internal data class LocalAttachment(val name: String, val size: Long, val mime: String)

internal class LocalUpload(
    val name: String,
    val size: Long,
    val mime: String,
    val opener: () -> InputStream,
)

internal fun queryLocalAttachment(resolver: android.content.ContentResolver, uri: Uri): LocalAttachment {
    var name = "attachment"
    var size = -1L
    resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE), null, null, null)?.use { cursor ->
        if (cursor.moveToFirst()) {
            val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (nameIndex >= 0) cursor.getString(nameIndex)?.takeIf { it.isNotBlank() }?.let { name = it }
            val sizeIndex = cursor.getColumnIndex(OpenableColumns.SIZE)
            if (sizeIndex >= 0 && !cursor.isNull(sizeIndex)) size = cursor.getLong(sizeIndex)
        }
    }
    val mime = resolver.getType(uri).orEmpty()
    if (!name.contains('.') && mime.isNotBlank()) {
        val extension = extensionForMime(mime)
        if (extension.isNotBlank()) name = "$name.$extension"
    }
    return LocalAttachment(name, size, mime)
}

internal fun prepareLocalUpload(
    context: Context,
    resolver: android.content.ContentResolver,
    uri: Uri,
    local: LocalAttachment,
    compression: CompressionSettings,
): LocalUpload {
    val compressed = runCatching {
        compressImage(context, uri, local.name, local.mime, compression)
    }.getOrNull()
    if (compressed != null) {
        return LocalUpload(compressed.name, compressed.file.length(), compressed.mime) { compressed.file.inputStream() }
    }
    return LocalUpload(local.name, local.size, local.mime) {
        resolver.openInputStream(uri) ?: throw IOException("添付を開けません。")
    }
}

/** 添付アップロード本体（キュー・転送・進捗）。状態は持たず、画面の更新は `ChatViewModel` へ委ねる。 */
internal class ChatAttachmentUploader(
    private val scope: CoroutineScope,
    private val state: StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val context: Context,
    private val api: PlaygroundApi,
    private val token: () -> String,
    private val uploadsLocal: () -> Boolean,
    private val localBackend: () -> com.minashin1120.aiplayground.data.backend.ChatBackend?,
    private val localChats: () -> LocalChatStore?,
    private val syncApi: () -> PlaygroundApi,
    private val report: suspend (Throwable, String) -> Unit,
) {
    private var uploadJob: Job? = null
    private val pendingUploads = ArrayDeque<Pair<Uri, Boolean>>()

    val activeJob: Job? get() = uploadJob

    fun cancelJobs() { uploadJob?.cancel() }

    /** 端末モードでは添付を端末内ストアへ保存し、サーバーへは送らない。 */
    private fun storeLocal(name: String, mime: String, size: Long, opener: () -> InputStream): String? {
        val store = localChats() ?: return null
        val started = System.currentTimeMillis()
        return opener().use { input -> store.saveFile(name, mime, input, size) }.also {
            Diagnostics.log("attach.stored", "mime" to mime, "bytes" to size, "ms" to System.currentTimeMillis() - started)
        }
    }

    /**
     * サーバーを使うサーバーレス利用（アカウントあり）で、端末保存した添付をサーバー側のツールが
     * 使う必要があるときだけ、同じ添付をサーバーへ同期する。オフライン中やアカウントなしの端末
     * モードでは同期しない（送信・実行は端末で完結する）。
     */
    private fun maybeSyncUploadedAttachment(uploaded: String) {
        val deviceBackend = localBackend() as? LocalChatBackend ?: return
        val chats = localChats() ?: return
        val current = state.value
        if (!current.serverless || current.localProfile || current.offline) return
        val options = JSONObject().put("thread_id", current.selected?.id.orEmpty()).put("model", current.model)
            .put("coding_mode", current.codingMode).put("enable_file_creation", current.enableFileCreation)
            .put("enable_mcp", current.enableMcp).put("enable_python", current.enablePython)
        scope.launch(Dispatchers.IO) {
            if (!deviceBackend.needsServerTools(options)) return@launch
            try {
                SyncEngine(chats, syncApi()) { token() }.uploadAttachment(uploaded)
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Diagnostics.failure("sync.error", e)
            }
        }
    }

    fun upload(uris: List<Uri>, skipCompression: Boolean = false) {
        if (state.value.starting) {
            scope.launch {
                state.first { !it.starting }
                upload(uris, skipCompression)
            }
            return
        }
        if (state.value.banned) return
        if (state.value.offline && !uploadsLocal()) {
            mutable.update { it.copy(notice = "オフライン中はファイルをアップロードできません。") }
            return
        }
        if (uris.isEmpty()) return
        uris.forEach { pendingUploads.addLast(it to skipCompression) }
        if (state.value.uploading) {
            mutable.update { it.copy(uploadCount = it.uploadCount + uris.size) }
            return
        }
        mutable.update {
            it.copy(
                uploading = true, uploadSent = 0, uploadTotal = 0, uploadName = "",
                uploadCompleted = 0, uploadCount = pendingUploads.size,
            )
        }
        uploadJob = scope.launch {
            try {
                do {
                    coroutineScope {
                        while (pendingUploads.isNotEmpty()) {
                            val (uri, queuedSkipCompression) = pendingUploads.removeFirst()
                            try {
                                val resolver = context.contentResolver
                                val local = withContext(Dispatchers.IO) { queryLocalAttachment(resolver, uri) }
                                mutable.update {
                                    it.copy(
                                        uploadName = local.name, uploadSent = 0,
                                        uploadTotal = if (local.size > 0) local.size else 0,
                                    )
                                }
                                val source = withContext(Dispatchers.IO) {
                                    if (queuedSkipCompression) LocalUpload(local.name, local.size, local.mime) {
                                        resolver.openInputStream(uri) ?: throw IOException("添付を開けません。")
                                    } else prepareLocalUpload(context, resolver, uri, local, state.value.compression)
                                }
                                mutable.update {
                                    it.copy(
                                        uploadName = source.name, uploadSent = 0,
                                        uploadTotal = if (source.size > 0) source.size else 0,
                                    )
                                }
                                val uploaded = withContext(Dispatchers.IO) {
                                    storeLocal(source.name, source.mime, source.size, source.opener)
                                } ?: if (source.size > CHAT_UPLOAD_CHUNK_THRESHOLD_BYTES) uploadInChunks(source) else uploadWhole(source)
                                mutable.update { it.withUploadedAttachment(Attachment(source.name, uploaded, source.mime)) }
                                maybeSyncUploadedAttachment(uploaded)
                            } catch (e: CancellationException) {
                                throw e
                            } catch (e: Exception) {
                                report(e, "添付のアップロードに失敗しました")
                            } finally {
                                mutable.update {
                                    it.copy(
                                        uploadCompleted = it.uploadCompleted + 1,
                                        uploadName = "", uploadSent = 0, uploadTotal = 0,
                                    )
                                }
                            }
                        }
                    }
                } while (pendingUploads.isNotEmpty())
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                report(e, "添付のアップロードに失敗しました")
            } finally {
                pendingUploads.clear()
                mutable.update {
                    it.copy(
                        uploading = false, uploadSent = 0, uploadTotal = 0, uploadName = "",
                        uploadCompleted = 0, uploadCount = 0,
                    )
                }
            }
        }
    }

    fun cancel() {
        if (!state.value.uploading) return
        pendingUploads.clear()
        uploadJob?.cancel()
    }

    fun reset() {
        pendingUploads.clear()
        uploadJob?.cancel()
    }

    /** Web `uploadMaskFile`: the chosen mask image is uploaded as it is. */
    fun uploadMask(uri: Uri) {
        if (state.value.banned || (state.value.offline && !uploadsLocal())) return
        scope.launch {
            try {
                val resolver = context.contentResolver
                val local = withContext(Dispatchers.IO) { queryLocalAttachment(resolver, uri) }
                val bytes = withContext(Dispatchers.IO) { resolver.openInputStream(uri)?.use { it.readBytes() } }
                    ?: throw IOException("Mask upload failed")
                val mime = local.mime.ifBlank { "image/png" }
                uploadMaskBytes(local.name, mime, bytes)
            } catch (e: CancellationException) { throw e }
            catch (e: com.minashin1120.aiplayground.data.ApiException) {
                mutable.update { it.copy(notice = e.payload.optString("error").ifBlank { "Mask upload failed" }) }
            } catch (e: Exception) {
                mutable.update { it.copy(notice = "Mask upload failed") }
            }
        }
    }

    fun uploadMask(png: ByteArray) {
        if (state.value.banned || (state.value.offline && !uploadsLocal())) return
        scope.launch {
            try { uploadMaskBytes("object_mask.png", "image/png", png) }
            catch (e: CancellationException) { throw e }
            catch (e: com.minashin1120.aiplayground.data.ApiException) {
                mutable.update { it.copy(notice = e.payload.optString("error").ifBlank { "Mask upload failed" }) }
            } catch (e: Exception) {
                mutable.update { it.copy(notice = "Mask upload failed") }
            }
        }
    }

    private suspend fun uploadMaskBytes(name: String, mime: String, bytes: ByteArray) {
        val uploaded = withContext(Dispatchers.IO) { storeLocal(name, mime, bytes.size.toLong()) { bytes.inputStream() } }
            ?: api.upload(name, bytes.toRequestBody(mime.toMediaType()), token()).getString("filename")
        mutable.update { it.copy(imageMask = uploaded) }
    }

    /**
     * Web `saveMarkerToRow`: uploads the edited PNG as `<name>_marked.png` and swaps it into the row,
     * keeping the first pre-edit file as the row's original.
     */
    fun uploadEditedAttachment(reference: String, png: ByteArray, attachOriginal: Boolean) {
        val row = state.value.attachments.firstOrNull { it.reference == reference } ?: return
        if (state.value.offline && !uploadsLocal()) {
            mutable.update { it.copy(notice = "オフライン中はファイルをアップロードできません。") }
            return
        }
        val fileName = markedFileName(row.name.ifBlank { "marked.png" })
        mutable.update { current -> current.copy(editingAttachment = reference, attachments = current.attachments.map {
            if (it.reference == reference) it.copy(attachOriginal = attachOriginal) else it
        }) }
        scope.launch {
            try {
                val uploaded = withContext(Dispatchers.IO) { storeLocal(fileName, "image/png", png.size.toLong()) { png.inputStream() } }
                    ?: api.upload(fileName, png.toRequestBody("image/png".toMediaType()), token()).getString("filename")
                val original = row.original ?: row.copy(original = null, attachOriginal = false, edited = false)
                mutable.update { current -> current.copy(attachments = current.attachments.map {
                    if (it.reference != reference) it
                    else Attachment(fileName, uploaded, "image/png", "upload", fileName, original, attachOriginal, edited = true)
                }) }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) { mutable.update { it.copy(notice = "編集画像のアップロードに失敗しました") } }
            finally { mutable.update { it.copy(editingAttachment = null) } }
        }
    }

    private suspend fun uploadWhole(local: LocalUpload): String {
        val known = local.size > 0
        val body = object : RequestBody() {
            override fun contentType() = local.mime.takeIf { it.isNotBlank() }?.toMediaTypeOrNull()
            override fun contentLength() = if (known) local.size else -1L
            override fun writeTo(sink: BufferedSink) {
                val input = local.opener()
                input.use {
                    val bytes = ByteArray(64 * 1024)
                    var sent = 0L
                    while (true) {
                        val count = it.read(bytes)
                        if (count < 0) break
                        sent += count
                        if (sent > CHAT_UPLOAD_MAX_SINGLE_BYTES) throw IOException("添付は1ファイル64MiBまでです。")
                        sink.write(bytes, 0, count)
                        mutable.update { state -> state.copy(uploadSent = sent, uploadTotal = if (known) local.size else sent) }
                    }
                }
            }
        }
        return api.upload(local.name, body, token()).getString("filename")
    }

    private suspend fun uploadInChunks(local: LocalUpload): String {
        val init = api.uploadInit(local.name, local.size, token())
        val uploadId = init.getString("upload_id")
        val chunkSize = init.optLong("chunk_size", CHAT_UPLOAD_CHUNK_THRESHOLD_BYTES).toInt().coerceAtLeast(1)
        val totalChunks = ((local.size + chunkSize - 1) / chunkSize).toInt()
        val input = local.opener()
        input.use {
            val buffer = ByteArray(chunkSize)
            var index = 0
            var sent = 0L
            while (index < totalChunks) {
                if (!currentCoroutineContext().isActive) throw CancellationException()
                val expected = minOf(chunkSize.toLong(), local.size - sent).toInt()
                var read = 0
                while (read < expected) {
                    val count = it.read(buffer, read, expected - read)
                    if (count < 0) break
                    read += count
                }
                if (read != expected) throw IOException("添付の読み込みに失敗しました。")
                api.uploadChunk(uploadId, index, totalChunks, buffer.copyOf(read), token())
                sent += read
                index++
                mutable.update { state -> state.copy(uploadSent = sent, uploadTotal = local.size) }
            }
        }
        return api.uploadComplete(uploadId, token()).getString("filename")
    }
}
