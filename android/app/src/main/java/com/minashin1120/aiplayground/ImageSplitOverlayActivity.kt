package com.minashin1120.aiplayground

import android.content.ClipData
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.lifecycle.lifecycleScope
import com.minashin1120.aiplayground.data.IMAGE_SPLIT_MAX_IMAGES
import com.minashin1120.aiplayground.data.IMAGE_SPLIT_SAVE_FOLDER
import com.minashin1120.aiplayground.data.ImageSplitOptions
import com.minashin1120.aiplayground.data.ImageSplitRequest
import com.minashin1120.aiplayground.data.SplitOutput
import com.minashin1120.aiplayground.data.renderImageSplit
import com.minashin1120.aiplayground.data.saveSplitOutputs
import com.minashin1120.aiplayground.ui.ImageSplitDialog
import java.io.File
import java.util.UUID
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue

/** A translucent, short lived activity: the previous app stays visible under the split dialog. */
class ImageSplitOverlayActivity : ComponentActivity() {
    private var request by mutableStateOf<ImageSplitRequest?>(null)
    private var pendingSave: ImageSplitOptions? = null
    private val pickImages = registerForActivityResult(ActivityResultContracts.GetMultipleContents()) { uris ->
        if (uris.isEmpty()) { finish(); return@registerForActivityResult }
        if (uris.size > IMAGE_SPLIT_MAX_IMAGES) toast("画像の分割は一度に${IMAGE_SPLIT_MAX_IMAGES}枚までです。先頭${IMAGE_SPLIT_MAX_IMAGES}枚のみ使います。")
        request = ImageSplitRequest(uris.take(IMAGE_SPLIT_MAX_IMAGES))
    }
    private val pickFolder = registerForActivityResult(ActivityResultContracts.OpenDocumentTree()) { tree ->
        val options = pendingSave
        pendingSave = null
        if (tree != null && options != null) split(options, save = true, tree = tree)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        savedInstanceState?.getStringArrayList("split_uris")?.map(Uri::parse)?.takeIf { it.isNotEmpty() }
            ?.let { request = ImageSplitRequest(it) }
        if (savedInstanceState?.getBoolean("save_pending") == true) {
            pendingSave = ImageSplitOptions(
                savedInstanceState.getInt("save_pieces"), savedInstanceState.getInt("save_overlap"),
                savedInstanceState.getBoolean("save_original"), savedInstanceState.getBoolean("save_mark_original"),
                savedInstanceState.getBoolean("save_mark_pieces"))
        }
        setContent {
            request?.let { current ->
                ImageSplitDialog(current, onDismiss = ::finish,
                    onAttach = { split(it, save = false) },
                    onSaveOnly = { options ->
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) split(options, save = true)
                        else { pendingSave = options; pickFolder.launch(null) }
                    }, attachLabel = "分割して共有")
            }
        }
        if (savedInstanceState == null) pickImages.launch("image/*")
    }

    override fun onSaveInstanceState(outState: Bundle) {
        request?.let { outState.putStringArrayList("split_uris", ArrayList(it.uris.map(Uri::toString))) }
        pendingSave?.let {
            outState.putBoolean("save_pending", true)
            outState.putInt("save_pieces", it.pieces)
            outState.putInt("save_overlap", it.overlapPct)
            outState.putBoolean("save_original", it.includeOriginal)
            outState.putBoolean("save_mark_original", it.markOriginal)
            outState.putBoolean("save_mark_pieces", it.markPieces)
        }
        super.onSaveInstanceState(outState)
    }

    private fun split(options: ImageSplitOptions, save: Boolean, tree: Uri? = null) {
        val current = request ?: return
        if (current.busy) return
        request = current.copy(busy = true, error = null)
        lifecycleScope.launch {
            val directory = File(cacheDir, "shared/image_split_tile/${UUID.randomUUID()}")
            try {
                val outputs = withContext(Dispatchers.IO) {
                    current.uris.flatMap { renderImageSplit(this@ImageSplitOverlayActivity, it, options, directory) }
                }
                if (save) {
                    val count = withContext(Dispatchers.IO) {
                        try { saveSplitOutputs(this@ImageSplitOverlayActivity, outputs, tree) }
                        finally { directory.deleteRecursively() }
                    }
                    toast(if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q)
                        "${count}枚の画像を「Pictures/$IMAGE_SPLIT_SAVE_FOLDER」に保存しました。"
                        else "${count}枚の画像を保存しました。")
                    finish()
                } else {
                    share(outputs)
                    finish()
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: OutOfMemoryError) {
                request = current.copy(error = "画像が大きすぎるため分割できませんでした。分割数を減らすか、小さい画像でお試しください。")
            } catch (e: Exception) {
                request = current.copy(error = "画像を分割できませんでした: ${e.message ?: e.javaClass.simpleName}")
            }
        }
    }

    private fun share(outputs: List<SplitOutput>) {
        val uris = ArrayList(outputs.map { it.uploadUri })
        val send = if (uris.size == 1) {
            Intent(Intent.ACTION_SEND).setType(outputs.single().mime)
                .putExtra(Intent.EXTRA_STREAM, uris.single())
        } else {
            Intent(Intent.ACTION_SEND_MULTIPLE).setType("image/*")
                .putParcelableArrayListExtra(Intent.EXTRA_STREAM, uris)
        }
        send.clipData = ClipData.newUri(contentResolver, "分割した画像", uris.first()).apply {
            uris.drop(1).forEach { addItem(ClipData.Item(it)) }
        }
        send.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        startActivity(Intent.createChooser(send, "分割した画像を共有"))
    }

    private fun toast(message: String) = Toast.makeText(this, message, Toast.LENGTH_LONG).show()
}
