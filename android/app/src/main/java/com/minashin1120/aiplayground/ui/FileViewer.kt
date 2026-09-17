package com.minashin1120.aiplayground.ui

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.pdf.PdfRenderer
import android.media.MediaPlayer
import android.os.ParcelFileDescriptor
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.MediaController
import android.widget.VideoView
import androidx.compose.foundation.Image
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.OpenInNew
import androidx.compose.material.icons.rounded.Pause
import androidx.compose.material.icons.rounded.PlayArrow
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.minashin1120.aiplayground.data.AttachmentKind
import com.minashin1120.aiplayground.data.attachmentKind
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.charset.Charset

internal data class FileViewRequest(
    val reference: String,
    val displayName: String = "",
    val ext: String = "",
    val mime: String = "",
)

internal fun fileViewerTitle(reference: String, displayName: String = ""): String {
    val named = displayName.trim()
    if (named.isNotBlank()) return named
    val fromPath = reference.substringBefore('?').substringAfterLast('/').trim()
    return fromPath.ifBlank { "ファイル" }
}

internal fun fileViewerKind(request: FileViewRequest): AttachmentKind {
    val title = fileViewerTitle(request.reference, request.displayName)
    val hinted = if (title.substringAfterLast('.', "").isBlank() && request.ext.isNotBlank()) {
        "$title.${request.ext.trim().trimStart('.')}"
    } else title
    return attachmentKind(hinted.ifBlank { request.reference }, request.mime)
}

/** Returns UTF-8/UTF-16 text, or null when the bytes look binary. */
internal fun decodePreviewText(bytes: ByteArray): String? {
    if (bytes.isEmpty()) return ""
    var offset = 0
    var charset: Charset = Charsets.UTF_8
    if (bytes.size >= 3 && bytes[0] == 0xEF.toByte() && bytes[1] == 0xBB.toByte() && bytes[2] == 0xBF.toByte()) {
        offset = 3
    } else if (bytes.size >= 2 && bytes[0] == 0xFF.toByte() && bytes[1] == 0xFE.toByte()) {
        offset = 2
        charset = Charsets.UTF_16LE
    } else if (bytes.size >= 2 && bytes[0] == 0xFE.toByte() && bytes[1] == 0xFF.toByte()) {
        offset = 2
        charset = Charsets.UTF_16BE
    }
    val text = String(bytes, offset, bytes.size - offset, charset)
    val sample = text.take(4096)
    if (sample.indexOf('\u0000') >= 0) return null
    if (sample.isNotEmpty()) {
        val replacement = sample.count { it == '\uFFFD' }
        if (replacement * 10 > sample.length) return null
    }
    return text
}

@Composable
internal fun FileViewerDialog(
    request: FileViewRequest,
    loader: FileBytesLoader,
    download: suspend (String) -> Pair<File, String>,
    onDismiss: () -> Unit,
    onOpenExternal: (String) -> Unit,
) {
    val title = fileViewerTitle(request.reference, request.displayName)
    val kind = fileViewerKind(request)
    Dialog(onDismissRequest = onDismiss, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        Surface(
            Modifier.fillMaxWidth().fillMaxHeight(0.94f).padding(8.dp),
            shape = MaterialTheme.shapes.large,
            tonalElevation = 3.dp,
        ) {
            Column(Modifier.fillMaxSize()) {
                Row(
                    Modifier.fillMaxWidth().padding(start = 20.dp, end = 8.dp, top = 8.dp, bottom = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(title, style = MaterialTheme.typography.titleLarge, maxLines = 2,
                        modifier = Modifier.weight(1f))
                    IconButton(onClick = onDismiss) { Icon(Icons.Rounded.Close, "閉じる") }
                }
                HorizontalDivider()
                Box(Modifier.weight(1f).fillMaxWidth().padding(12.dp)) {
                    when (kind) {
                        AttachmentKind.IMAGE -> ImagePreview(request.reference, loader)
                        AttachmentKind.TEXT -> TextPreview(request.reference, loader)
                        AttachmentKind.PDF -> DownloadedPreview(request.reference, download) { file, _ ->
                            PdfPreview(file)
                        }
                        AttachmentKind.AUDIO -> DownloadedPreview(request.reference, download) { file, _ ->
                            AudioPreview(file)
                        }
                        AttachmentKind.VIDEO -> DownloadedPreview(request.reference, download) { file, _ ->
                            VideoPreview(file)
                        }
                        AttachmentKind.FILE -> UnsupportedPreview()
                    }
                }
                HorizontalDivider()
                Row(
                    Modifier.fillMaxWidth().padding(8.dp),
                    horizontalArrangement = Arrangement.End,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    TextButton(onClick = { onOpenExternal(request.reference) }) {
                        Icon(Icons.Rounded.OpenInNew, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(8.dp))
                        Text("外部アプリで開く")
                    }
                    TextButton(onClick = onDismiss) { Text("閉じる") }
                }
            }
        }
    }
}

@Composable
private fun ImagePreview(reference: String, loader: FileBytesLoader) {
    var scale by remember(reference) { mutableFloatStateOf(1f) }
    var offset by remember(reference) { mutableStateOf(Offset.Zero) }
    Box(
        Modifier.fillMaxSize().pointerInput(reference) {
            detectTransformGestures { _, pan, zoom, _ ->
                scale = (scale * zoom).coerceIn(1f, 6f)
                offset = if (scale == 1f) Offset.Zero else offset + pan
            }
        },
        contentAlignment = Alignment.Center,
    ) {
        ProtectedImage(
            reference, loader, onOpen = null,
            modifier = Modifier.fillMaxSize().graphicsLayer {
                scaleX = scale
                scaleY = scale
                translationX = offset.x
                translationY = offset.y
            },
            thumbnail = false,
            compact = false,
            maxDecodeWidth = 2048,
            limitBytes = 24L * 1024 * 1024,
            contentDescription = "画像プレビュー",
        )
    }
}

@Composable
private fun TextPreview(reference: String, loader: FileBytesLoader) {
    var text by remember(reference) { mutableStateOf<String?>(null) }
    var failed by remember(reference) { mutableStateOf(false) }
    LaunchedEffect(reference) {
        val bytes = runCatching { loader(reference, false, 2L * 1024 * 1024) }.getOrNull()
        val decoded = bytes?.let { decodePreviewText(it) }
        if (decoded != null) text = decoded else failed = true
    }
    when {
        failed -> Text("このファイルはテキストとして表示できません。外部アプリで開いてください。",
            style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        text == null -> Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            CircularProgressIndicator()
        }
        else -> SelectionContainer {
            val body = text ?: return@SelectionContainer
            Text(
                body,
                style = MaterialTheme.typography.bodySmall,
                fontFamily = FontFamily.Monospace,
                modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()),
            )
        }
    }
}

@Composable
private fun DownloadedPreview(
    reference: String,
    download: suspend (String) -> Pair<File, String>,
    content: @Composable (File, String) -> Unit,
) {
    var ready by remember(reference) { mutableStateOf<Pair<File, String>?>(null) }
    var error by remember(reference) { mutableStateOf<String?>(null) }
    LaunchedEffect(reference) {
        runCatching { download(reference) }
            .onSuccess { ready = it }
            .onFailure { error = it.message?.ifBlank { null } ?: "ファイルを読み込めませんでした。" }
    }
    val local = ready
    when {
        error != null -> Text(error!!, style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant)
        local == null -> Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            CircularProgressIndicator()
        }
        else -> content(local.first, local.second)
    }
}

@Composable
private fun PdfPreview(file: File) {
    var pages by remember(file) { mutableStateOf<List<Bitmap>>(emptyList()) }
    var truncated by remember(file) { mutableStateOf(false) }
    var error by remember(file) { mutableStateOf<String?>(null) }
    DisposableEffect(pages) {
        val held = pages
        onDispose { held.forEach { if (!it.isRecycled) it.recycle() } }
    }
    LaunchedEffect(file) {
        val rendered = withContext(Dispatchers.IO) {
            runCatching { renderPdfPages(file, 40) }.getOrElse { return@withContext null }
        }
        if (rendered == null) {
            error = "PDFを表示できませんでした。外部アプリで開いてください。"
        } else {
            truncated = rendered.second
            pages = rendered.first
        }
    }
    when {
        error != null -> Text(error!!, style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant)
        pages.isEmpty() -> Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            CircularProgressIndicator()
        }
        else -> LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            if (truncated) item(key = "truncated") {
                Text("先頭40ページまで表示しています。続きは外部アプリで開けます。",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            items(pages.size, key = { "pdf-$it" }) { index ->
                val page = pages[index]
                if (!page.isRecycled) {
                    Image(page.asImageBitmap(), "ページ ${index + 1}",
                        modifier = Modifier.fillMaxWidth(), contentScale = ContentScale.FillWidth)
                }
            }
        }
    }
}

private fun renderPdfPages(file: File, maxPages: Int): Pair<List<Bitmap>, Boolean> {
    val pfd = ParcelFileDescriptor.open(file, ParcelFileDescriptor.MODE_READ_ONLY)
    val renderer = PdfRenderer(pfd)
    try {
        val count = renderer.pageCount
        val limit = minOf(count, maxPages)
        val bitmaps = (0 until limit).map { index ->
            renderer.openPage(index).use { page ->
                val width = 1080
                val height = (page.height * (width.toFloat() / page.width.coerceAtLeast(1))).toInt().coerceAtLeast(1)
                Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also { bitmap ->
                    bitmap.eraseColor(Color.WHITE)
                    page.render(bitmap, null, null, PdfRenderer.Page.RENDER_MODE_FOR_DISPLAY)
                }
            }
        }
        return bitmaps to (count > maxPages)
    } finally {
        renderer.close()
        pfd.close()
    }
}

@Composable
private fun AudioPreview(file: File) {
    var playing by remember(file) { mutableStateOf(false) }
    var ready by remember(file) { mutableStateOf(false) }
    var durationMs by remember(file) { mutableIntStateOf(0) }
    var failed by remember(file) { mutableStateOf(false) }
    val player = remember(file) { MediaPlayer() }
    DisposableEffect(file) {
        player.setOnPreparedListener {
            durationMs = player.duration.coerceAtLeast(0)
            ready = true
        }
        player.setOnCompletionListener { playing = false }
        player.setOnErrorListener { _, _, _ -> failed = true; true }
        val started = runCatching {
            player.setDataSource(file.absolutePath)
            player.prepareAsync()
        }.isSuccess
        if (!started) failed = true
        onDispose {
            player.setOnPreparedListener(null)
            player.setOnCompletionListener(null)
            player.setOnErrorListener(null)
            runCatching { player.stop() }
            player.release()
        }
    }
    if (failed) {
        Text("音声を再生できませんでした。外部アプリで開いてください。",
            style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        return
    }
    Column(
        Modifier.fillMaxWidth().padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text("音声プレビュー", style = MaterialTheme.typography.titleMedium)
        if (durationMs > 0) {
            Text("${durationMs / 1000 / 60}:${"%02d".format((durationMs / 1000) % 60)}",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        if (!ready) CircularProgressIndicator(Modifier.size(24.dp), strokeWidth = 2.dp)
        else FilledIconButton(onClick = {
            if (player.isPlaying) {
                player.pause()
                playing = false
            } else {
                player.start()
                playing = true
            }
        }) {
            Icon(if (playing) Icons.Rounded.Pause else Icons.Rounded.PlayArrow,
                contentDescription = if (playing) "一時停止" else "再生")
        }
    }
}

@Composable
private fun VideoPreview(file: File) {
    AndroidView(
        factory = { context ->
            FrameLayout(context).apply {
                layoutParams = ViewGroup.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT,
                )
                val video = VideoView(context).apply {
                    setVideoPath(file.absolutePath)
                    setOnPreparedListener { start() }
                }
                addView(
                    video,
                    FrameLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.MATCH_PARENT,
                    ),
                )
                val controller = MediaController(context)
                controller.setAnchorView(this)
                video.setMediaController(controller)
            }
        },
        modifier = Modifier.fillMaxSize(),
        onRelease = { frame ->
            val video = frame.getChildAt(0) as? VideoView
            video?.stopPlayback()
        },
    )
}

@Composable
private fun UnsupportedPreview() {
    Column(
        Modifier.fillMaxSize().padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text("この形式はアプリ内でプレビューできません。", style = MaterialTheme.typography.bodyMedium)
        Text("外部アプリで開くか、チャットへ再利用できます。",
            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}
