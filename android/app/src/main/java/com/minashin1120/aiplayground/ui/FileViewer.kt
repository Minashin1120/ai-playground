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
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.R
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.IntSize
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
    /** Web `openImageViewer`: the chat's images the viewer steps through ("n / total"). */
    val gallery: List<String> = emptyList(),
)

/** What the Web image viewer toolbar does with the shown image. */
internal class ImageViewerActions(
    val onDownload: (String) -> Unit = {},
    val onCopyUrl: (String) -> Unit = {},
    /** Returns whether the image was added (the viewer then closes, as on Web). */
    val onReuse: (String) -> Boolean = { false },
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
    imageActions: ImageViewerActions = ImageViewerActions(),
) {
    val title = fileViewerTitle(request.reference, request.displayName)
    val kind = fileViewerKind(request)
    if (kind == AttachmentKind.IMAGE) {
        ImageViewer(request, loader, onDismiss, imageActions)
        return
    }
    // Web `#file-viewer`: a dark panel with the file name and the round close button.
    Dialog(onDismissRequest = onDismiss, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        WebModalWindow(0.dp)
        Box(Modifier.fillMaxSize().background(androidx.compose.ui.graphics.Color(2, 6, 16).copy(alpha = 0.94f))
            .clickable(interactionSource = remember { MutableInteractionSource() }, indication = null, onClick = onDismiss),
            contentAlignment = Alignment.Center) {
            ModalPanelMotion(fullScreen = false, onDismissRequest = onDismiss) {
                val web = LocalWebPalette.current
                val shape = RoundedCornerShape(16.dp)
                val screen = LocalConfiguration.current
                Column(
                    Modifier.width(minOf(screen.screenWidthDp.dp * 0.92f, 980.dp)).heightIn(max = screen.screenHeightDp.dp * 0.86f)
                        .clip(shape).background(androidx.compose.ui.graphics.Color(10, 16, 30).copy(alpha = 0.92f)).border(1.dp, web.line, shape)
                        .clickable(interactionSource = remember { MutableInteractionSource() }, indication = null) {},
                ) {
                    Row(
                        Modifier.fillMaxWidth().drawBehind {
                            drawRect(androidx.compose.ui.graphics.Color(148, 163, 184).copy(alpha = 0.12f),
                                Offset(0f, size.height - 1.dp.toPx()), androidx.compose.ui.geometry.Size(size.width, 1.dp.toPx()))
                        }.padding(horizontal = 14.dp, vertical = 10.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(title, fontSize = 12.sp, color = androidx.compose.ui.graphics.Color(0xFFE5E7EB), maxLines = 1,
                            overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f))
                        Box(Modifier.size(32.dp).clip(CircleShape).background(androidx.compose.ui.graphics.Color(13, 21, 40).copy(alpha = 0.7f))
                            .border(1.dp, web.line, CircleShape).clickable(role = Role.Button, onClick = onDismiss),
                            contentAlignment = Alignment.Center) {
                            FaIcon(R.drawable.fa_solid_times, "close", size = 12.dp, tint = androidx.compose.ui.graphics.Color(0xFFE5E7EB))
                        }
                    }
                    Box(Modifier.weight(1f, fill = false).fillMaxWidth().padding(10.dp)) {
                        when (kind) {
                            AttachmentKind.TEXT -> TextPreview(request.reference, loader)
                            AttachmentKind.PDF -> DownloadedPreview(request.reference, download) { file, _ -> PdfPreview(file) }
                            AttachmentKind.AUDIO -> DownloadedPreview(request.reference, download) { file, _ -> AudioPreview(file) }
                            AttachmentKind.VIDEO -> DownloadedPreview(request.reference, download) { file, _ -> VideoPreview(file) }
                            else -> UnsupportedPreview(onDownload = { imageActions.onDownload(request.reference) },
                                onOpen = { onOpenExternal(request.reference) })
                        }
                    }
                }
            }
        }
    }
}

/**
 * Web `#image-viewer`: the image on a dark backdrop with pinch zoom, "n / total • name" at the top, the
 * previous / next buttons for the chat's other images and the Download / Copy URL / Reuse / Close toolbar.
 */
@Composable
private fun ImageViewer(request: FileViewRequest, loader: FileBytesLoader, onDismiss: () -> Unit, actions: ImageViewerActions) {
    val items = request.gallery.takeIf { request.reference in it } ?: listOf(request.reference)
    var index by remember(request) { mutableIntStateOf(items.indexOf(request.reference).coerceAtLeast(0)) }
    val current = items[index]
    val glass = androidx.compose.ui.graphics.Color(13, 21, 40).copy(alpha = 0.45f)
    val edge = androidx.compose.ui.graphics.Color.White.copy(alpha = 0.18f)
    Dialog(onDismissRequest = onDismiss, properties = DialogProperties(usePlatformDefaultWidth = false, decorFitsSystemWindows = false)) {
        WebModalWindow(0.dp)
        Box(Modifier.fillMaxSize().background(androidx.compose.ui.graphics.Color(2, 6, 16).copy(alpha = 0.96f)).safeDrawingPadding()) {
            key(current) {
                Box(Modifier.fillMaxSize().padding(horizontal = 8.dp, vertical = 64.dp)) { ImagePreview(current, loader) }
            }
            Text(
                "${index + 1} / ${items.size} • ${fileViewerTitle(current, if (current == request.reference) request.displayName else "")}",
                fontSize = 13.sp, color = androidx.compose.ui.graphics.Color(0xFFF8FAFC), maxLines = 1, overflow = TextOverflow.Ellipsis,
                modifier = Modifier.align(Alignment.TopCenter).padding(top = 16.dp).fillMaxWidth(0.8f).wrapContentWidth()
                    .clip(CircleShape).background(glass).border(1.dp, edge, CircleShape).padding(horizontal = 20.dp, vertical = 8.dp),
            )
            if (items.size > 1) {
                listOf(-1 to R.drawable.fa_solid_chevron_left, 1 to R.drawable.fa_solid_chevron_right).forEach { (step, icon) ->
                    val enabled = index + step in items.indices
                    Box(
                        Modifier.align(if (step < 0) Alignment.CenterStart else Alignment.CenterEnd).padding(horizontal = 16.dp).size(48.dp)
                            .graphicsLayer { alpha = if (enabled) 1f else 0.3f }.clip(CircleShape)
                            .background(androidx.compose.ui.graphics.Color.White.copy(alpha = 0.08f))
                            .border(1.dp, androidx.compose.ui.graphics.Color.White.copy(alpha = 0.12f), CircleShape)
                            .clickable(enabled = enabled, role = Role.Button) { index += step },
                        contentAlignment = Alignment.Center,
                    ) { FaIcon(icon, if (step < 0) "前へ" else "次へ", size = 16.dp, tint = androidx.compose.ui.graphics.Color.White) }
                }
            }
            Row(
                Modifier.align(Alignment.BottomCenter).padding(bottom = 24.dp).clip(CircleShape).background(glass).border(1.dp, edge, CircleShape)
                    .padding(horizontal = 20.dp, vertical = 10.dp),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                listOf<Triple<Int, String, () -> Unit>>(
                    Triple(R.drawable.fa_solid_download, "Download", { actions.onDownload(current) }),
                    Triple(R.drawable.fa_solid_link, "Copy URL", { actions.onCopyUrl(current) }),
                    Triple(R.drawable.fa_solid_reply, "Reuse as Attachment", { if (actions.onReuse(current)) onDismiss() }),
                    Triple(R.drawable.fa_solid_times, "Close", onDismiss),
                ).forEach { (icon, label, action) ->
                    Box(Modifier.size(40.dp).clip(CircleShape).clickable(role = Role.Button, onClickLabel = label, onClick = action),
                        contentAlignment = Alignment.Center) { FaIcon(icon, label, size = 16.dp, tint = androidx.compose.ui.graphics.Color(0xFFF1F5F9)) }
                }
            }
        }
    }
}

@Composable
private fun ImagePreview(reference: String, loader: FileBytesLoader) {
    var scale by remember(reference) { mutableFloatStateOf(1f) }
    var offset by remember(reference) { mutableStateOf(Offset.Zero) }
    var viewport by remember(reference) { mutableStateOf(IntSize.Zero) }
    Box(
        Modifier.fillMaxSize().onSizeChanged { viewport = it }.pointerInput(reference) {
            detectTransformGestures { centroid, pan, zoom, _ ->
                val oldScale = scale
                val nextScale = (oldScale * zoom).coerceIn(1f, 6f)
                val ratio = nextScale / oldScale
                val center = Offset(viewport.width / 2f, viewport.height / 2f)
                val focalPoint = centroid - center
                val nextOffset = offset + pan + (focalPoint - offset) * (1f - ratio)
                scale = nextScale
                offset = if (nextScale == 1f) Offset.Zero
                else clampImagePreviewOffset(nextOffset, nextScale, viewport)
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

private fun clampImagePreviewOffset(offset: Offset, scale: Float, viewport: IntSize): Offset {
    val maxX = (viewport.width * (scale - 1f) / 2f).coerceAtLeast(0f)
    val maxY = (viewport.height * (scale - 1f) / 2f).coerceAtLeast(0f)
    return Offset(
        offset.x.coerceIn(-maxX, maxX),
        offset.y.coerceIn(-maxY, maxY),
    )
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
            FaIcon(if (playing) R.drawable.fa_solid_pause else R.drawable.fa_solid_play, if (playing) "一時停止" else "再生", size = 16.dp)
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

/** Web `#file-viewer .fallback`: no preview, with ダウンロード and 新しいタブで開く. */
@Composable
private fun UnsupportedPreview(onDownload: () -> Unit, onOpen: () -> Unit) {
    Column(Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 24.dp), horizontalAlignment = Alignment.CenterHorizontally) {
        Text("この形式はプレビューできません。", fontSize = 12.sp, color = androidx.compose.ui.graphics.Color(0xFFCBD5F5))
        Row(Modifier.padding(top = 12.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            listOf("ダウンロード" to onDownload, "新しいタブで開く" to onOpen).forEach { (label, action) ->
                val shape = RoundedCornerShape(4.dp)
                Text(label, fontSize = 12.sp, color = androidx.compose.ui.graphics.Color.White,
                    modifier = Modifier.clip(shape).background(Tw.gray800).border(1.dp, Tw.gray700, shape)
                        .clickable(role = Role.Button, onClick = action).padding(horizontal = 12.dp, vertical = 4.dp))
            }
        }
    }
}
