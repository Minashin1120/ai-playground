package com.minashin1120.aiplayground

import android.content.SharedPreferences
import android.net.Uri
import com.minashin1120.aiplayground.data.CompressionSettings
import com.minashin1120.aiplayground.data.saveCompressionSettings
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.io.File

/**
 * 添付の入口（Web `#file-input` / `uploadCancelTokens` / `resetUploadState`）だけを持つ協作者：
 * アップロード・中止・一覧のクリア・圧縮設定・ダウンロード・共有テキストのプレビュー・画像分割。
 * 実際の転送は [ChatAttachmentUploader]、端末キャッシュは [ChatAttachmentFiles]、画像分割は
 * [ChatImageSplitManager] に委ねる。
 */
internal class ChatAttachments(
    private val mutable: MutableStateFlow<ChatState>,
    private val scope: CoroutineScope,
    private val prefs: SharedPreferences,
    private val attachmentFiles: ChatAttachmentFiles,
    private val attachmentUploader: ChatAttachmentUploader,
    private val imageSplitManager: ChatImageSplitManager,
    private val notify: (String) -> Unit,
    private val report: suspend (Throwable) -> Unit,
) {
    /** Account/profile-scoped decoded-image identity; safe to call from composition without IO. */
    fun imageCacheKey(reference: String): String = attachmentFiles.imageCacheKey(reference)

    /** Loads same-origin attachment bytes for preview; returns null when unavailable. */
    suspend fun loadAttachmentBytes(reference: String, thumbnail: Boolean, limit: Long = 8L * 1024 * 1024): ByteArray? =
        attachmentFiles.loadBytes(reference, thumbnail, limit)

    fun upload(uris: List<Uri>) = attachmentUploader.upload(uris, skipCompression = false)

    fun upload(uris: List<Uri>, skipCompression: Boolean) = attachmentUploader.upload(uris, skipCompression)

    /** メッセージとしてプレビュー (ANDROID_ONLY.md): text from the share target, drawn as a message bubble. */
    fun openSharedTextPreview(text: String) {
        if (text.isBlank()) notify("プレビューできるテキストがありません。")
        else mutable.update { it.copy(sharedTextPreview = text) }
    }

    fun closeSharedTextPreview() { mutable.update { it.copy(sharedTextPreview = null) } }

    fun openImageSplit(uris: List<Uri>) = imageSplitManager.open(uris)
    fun closeImageSplit() = imageSplitManager.close()

    /**
     * Splits every image of the dialog, then attaches the files ([attach]) or only saves them to the
     * device (Pictures/AI Playground, or [tree] before Android 10). Saving works without an account.
     */
    fun runImageSplit(options: ImageSplitOptions, attach: Boolean, tree: Uri? = null) =
        imageSplitManager.run(options, attach, tree)

    /** Removing the uploading row (Web `uploadCancelTokens`): the rest of this batch is not uploaded. */
    fun cancelUpload() = attachmentUploader.cancel()

    /** Web `resetUploadState`: the composer ✕ and "リストをクリア" drop every attachment and the mask. */
    fun resetUploads() {
        attachmentUploader.reset()
        mutable.update { it.copy(attachments = emptyList(), imageMask = null) }
    }

    /** Persists the local image-compression settings used before image uploads. */
    fun saveCompression(settings: CompressionSettings) {
        saveCompressionSettings(prefs, settings)
        mutable.update { it.copy(compression = settings) }
    }

    suspend fun downloadAttachment(reference: String): Pair<File, String> = attachmentFiles.download(reference)

    fun openFile(reference: String, onReady: (File, String) -> Unit) { scope.launch {
        try {
            val (file, mime) = attachmentFiles.download(reference)
            onReady(file, mime)
        } catch (e: Exception) { report(e) }
    } }
}
