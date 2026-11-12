package com.minashin1120.aiplayground

import android.app.Application
import android.net.Uri
import android.os.Build
import com.minashin1120.aiplayground.data.IMAGE_SPLIT_MAX_IMAGES
import com.minashin1120.aiplayground.data.IMAGE_SPLIT_SAVE_FOLDER
import com.minashin1120.aiplayground.data.ImageSplitOptions
import com.minashin1120.aiplayground.data.ImageSplitRequest
import com.minashin1120.aiplayground.data.renderImageSplit
import com.minashin1120.aiplayground.data.saveSplitOutputs
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.UUID

/** 画像分割 (ANDROID_ONLY.md): dialog state and split/attach/save workflow (extracted from ChatViewModel). */
internal class ChatImageSplitManager(
    private val scope: CoroutineScope,
    private val app: Application,
    private val isUploading: () -> Boolean,
    private val upload: (List<Uri>, Boolean) -> Unit,
    private val notify: (String) -> Unit,
) {
    private val mutable = MutableStateFlow<ImageSplitRequest?>(null)
    val state: StateFlow<ImageSplitRequest?> = mutable.asStateFlow()

    fun open(uris: List<Uri>) {
        if (uris.isEmpty()) return
        if (uris.size > IMAGE_SPLIT_MAX_IMAGES) notify("画像の分割は一度に${IMAGE_SPLIT_MAX_IMAGES}枚までです。先頭${IMAGE_SPLIT_MAX_IMAGES}枚のみ使います。")
        mutable.value = ImageSplitRequest(uris.take(IMAGE_SPLIT_MAX_IMAGES))
    }

    fun close() {
        if (mutable.value?.busy != true) mutable.value = null
    }

    /**
     * Splits every image of the dialog, then attaches the files ([attach]) or only saves them to the
     * device (Pictures/AI Playground, or [tree] before Android 10). Saving works without an account.
     */
    fun run(options: ImageSplitOptions, attach: Boolean, tree: Uri? = null) {
        val request = mutable.value ?: return
        if (request.busy) return
        if (attach && isUploading()) { notify("アップロードが終わってから、もう一度お試しください。"); return }
        mutable.value = request.copy(busy = true, error = null)
        scope.launch {
            val root = File(app.cacheDir, "shared/image_split")
            val dir = File(root, UUID.randomUUID().toString())
            // Earlier batches are no longer read once their upload has finished.
            val clearEarlier = !isUploading()
            try {
                val outputs = withContext(Dispatchers.IO) {
                    if (clearEarlier) root.deleteRecursively()
                    request.uris.flatMap { renderImageSplit(app, it, options, dir) }
                }
                if (attach) {
                    mutable.value = null
                    upload(outputs.map { it.uploadUri }, true)
                } else {
                    val saved = withContext(Dispatchers.IO) {
                        try { saveSplitOutputs(app, outputs, tree) } finally { dir.deleteRecursively() }
                    }
                    mutable.value = null
                    notify(if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) "${saved}枚の画像を「Pictures/$IMAGE_SPLIT_SAVE_FOLDER」に保存しました。"
                        else "${saved}枚の画像を保存しました。")
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: OutOfMemoryError) {
                mutable.value = request.copy(busy = false,
                    error = "画像が大きすぎるため分割できませんでした。分割数を減らすか、小さい画像でお試しください。")
            } catch (e: Exception) {
                mutable.value = request.copy(busy = false,
                    error = "画像を分割できませんでした: ${e.message ?: e.javaClass.simpleName}")
            }
        }
    }
}
