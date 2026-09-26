@file:OptIn(androidx.compose.foundation.layout.ExperimentalLayoutApi::class)

package com.minashin1120.aiplayground.ui

import android.net.Uri
import android.provider.DocumentsContract
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.DrawableRes
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.GridItemSpan
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.ChatViewModel
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.FileUsageChat
import com.minashin1120.aiplayground.data.LibraryFile
import com.minashin1120.aiplayground.data.sortLibraryFiles
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

private val LIB_SORT_OPTIONS = webOptions(
    "newest" to "新しい順", "oldest" to "古い順", "name_asc" to "名前順 (A→Z)", "name_desc" to "名前順 (Z→A)",
)

/** Web `libraryFileIcon`. */
@DrawableRes
private fun libraryFileIcon(ext: String): Int = when (ext.lowercase()) {
    "pdf" -> R.drawable.fa_solid_file_pdf
    "png", "jpg", "jpeg", "gif", "webp", "bmp", "svg", "heic" -> R.drawable.fa_solid_image
    else -> R.drawable.fa_solid_file
}

private fun LibraryFile.extName(): String = ext.ifBlank { displayName.substringAfterLast('.', "") }.lowercase()

/** `#lib-modal` (ファイルライブラリ): search, order, favorites filter, multi-select actions and the file grid. */
@Composable
internal fun LibraryDialog(state: ChatState, model: ChatViewModel, onOpenFile: (FileViewRequest) -> Unit, onDismiss: () -> Unit) {
    val web = LocalWebPalette.current
    val context = LocalContext.current
    val loader: FileBytesLoader = { reference, thumbnail, limit -> model.loadAttachmentBytes(reference, thumbnail, limit) }
    val selected = remember { mutableStateListOf<String>() }
    var renameTarget by remember { mutableStateOf<LibraryFile?>(null) }
    var deleteTarget by remember { mutableStateOf<LibraryFile?>(null) }
    var deleteSelected by remember { mutableStateOf(false) }
    var usageTarget by remember { mutableStateOf<LibraryFile?>(null) }
    var pendingSave by remember { mutableStateOf<List<LibraryFile>>(emptyList()) }
    LaunchedEffect(Unit) { model.refreshLibrary() }
    // Web clears the selection whenever the first page is loaded again.
    LaunchedEffect(state.librarySort, state.libraryFavoritesOnly, state.libraryQuery) { selected.clear() }
    val files = sortLibraryFiles(state.library, state.librarySort)
    val chosen = files.filter { it.filepath in selected }

    fun writeTo(uri: Uri, local: File) {
        context.contentResolver.openOutputStream(uri)?.use { out -> local.inputStream().use { it.copyTo(out) } }
            ?: throw java.io.IOException("保存先を開けませんでした。")
    }
    val saveOne = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/octet-stream")) { uri ->
        val targets = pendingSave
        pendingSave = emptyList()
        if (uri != null && targets.isNotEmpty()) model.saveLibraryFiles(targets) { _, local, _ ->
            withContext(Dispatchers.IO) { writeTo(uri, local) }
        }
    }
    val saveMany = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { tree ->
        val targets = pendingSave
        pendingSave = emptyList()
        if (tree != null && targets.isNotEmpty()) model.saveLibraryFiles(targets) { file, local, mime ->
            withContext(Dispatchers.IO) {
                val parent = DocumentsContract.buildDocumentUriUsingTree(tree, DocumentsContract.getTreeDocumentId(tree))
                val created = DocumentsContract.createDocument(context.contentResolver, parent, mime.ifBlank { "application/octet-stream" }, file.displayName)
                    ?: throw java.io.IOException("保存先を作成できませんでした。")
                writeTo(created, local)
            }
        }
    }
    fun download(targets: List<LibraryFile>) {
        if (targets.isEmpty()) return
        pendingSave = targets
        if (targets.size == 1) saveOne.launch(targets.first().displayName) else saveMany.launch(null)
    }
    fun open(file: LibraryFile) {
        val reference = file.url.ifBlank { file.filepath }
        onOpenFile(FileViewRequest(reference, file.displayName, file.ext, if (file.type == "image") "image/" else ""))
    }

    WebOverlayModal(onDismiss, if (web.isLight) Color(15, 23, 42).copy(alpha = 0.35f) else Color(3, 7, 16).copy(alpha = 0.72f), 10.dp) { phone ->
        val shape = RoundedCornerShape(if (phone) 0.dp else 22.dp)
        Column(
            Modifier.fillMaxSize().padding(if (phone) 0.dp else 16.dp).clip(shape)
                .background(Brush.verticalGradient(
                    if (web.isLight) listOf(Color.White, Color(0xFFF7F9FC))
                    else listOf(Color(8, 12, 24).copy(alpha = 0.99f), Color(5, 7, 15).copy(alpha = 0.99f)),
                ))
                .border(1.dp, if (web.isLight) Color(15, 23, 42).copy(alpha = 0.10f) else Color.White.copy(alpha = 0.08f), shape),
        ) {
            LibraryHeader(phone, onDismiss)
            LibraryToolbar(
                state, phone, selectedCount = selected.size,
                onAttach = {
                    model.attachLibraryFiles(chosen)
                    selected.clear()
                    onDismiss()
                },
                onDownload = { download(chosen) },
                onRename = { renameTarget = chosen.singleOrNull() },
                onUsage = { usageTarget = chosen.singleOrNull() },
                onDelete = { deleteSelected = true },
                model = model,
            )
            val gridPadding = if (phone) PaddingValues(horizontal = 16.dp, vertical = 14.dp) else PaddingValues(horizontal = 22.dp, vertical = 20.dp)
            BoxWithConstraints(Modifier.weight(1f).fillMaxWidth()) {
                val usable = maxWidth - (if (phone) 32.dp else 44.dp)
                val columns = ((usable + 16.dp) / (180.dp + 16.dp)).toInt().coerceAtLeast(1)
                val query = state.libraryQuery.trim().lowercase()
                val visible = files.filter { (!state.libraryFavoritesOnly || it.isFavorite) && (query.isEmpty() || it.displayName.lowercase().contains(query)) }
                LazyVerticalGrid(
                    columns = GridCells.Fixed(columns),
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = gridPadding,
                    horizontalArrangement = Arrangement.spacedBy(16.dp),
                    verticalArrangement = Arrangement.spacedBy(20.dp),
                ) {
                    when {
                        state.libraryBusy && state.library.isEmpty() -> items(12) { LibrarySkeletonCard() }
                        state.libraryFailed && state.library.isEmpty() -> item(span = { GridItemSpan(maxLineSpan) }) {
                            LibraryEmpty(R.drawable.fa_solid_exclamation_triangle, "ライブラリの読み込みに失敗しました", "通信状況を確認して時間をおいて再度お試しください。")
                        }
                        state.library.isEmpty() -> item(span = { GridItemSpan(maxLineSpan) }) {
                            LibraryEmpty(R.drawable.fa_solid_folder, "ファイルがまだありません", "アップロードしたファイルがここに表示されます。")
                        }
                        visible.isEmpty() -> item(span = { GridItemSpan(maxLineSpan) }) {
                            if (state.libraryFavoritesOnly && query.isEmpty()) {
                                LibraryEmpty(R.drawable.fa_solid_star, "お気に入りがありません", "ファイルの星ボタンからお気に入りに追加できます。")
                            } else {
                                LibraryEmpty(R.drawable.fa_solid_search, "一致するファイルがありません", "検索条件や並び順を変更してください。")
                            }
                        }
                        else -> items(visible, key = { it.filepath }) { file ->
                            LibraryCard(
                                file, selected = file.filepath in selected, showActions = !(phone && selected.isNotEmpty()),
                                loader = loader, offline = state.offline,
                                onToggle = { if (file.filepath in selected) selected.remove(file.filepath) else selected.add(file.filepath) },
                                onFavorite = { model.toggleLibraryFavorite(file) },
                                onOpen = { open(file) },
                                onDelete = { deleteTarget = file },
                            )
                        }
                    }
                    if (state.libraryHasMore && !state.libraryBusy) item(span = { GridItemSpan(maxLineSpan) }) {
                        Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
                            LibActionButton(R.drawable.fa_solid_chevron_down, "さらに読み込む", phone, onClick = model::moreLibrary)
                        }
                    }
                }
            }
        }
    }
    renameTarget?.let { file ->
        BrowserPromptDialog("新しいファイル名を入力してください", initial = file.displayName) { name ->
            renameTarget = null
            if (name != null) model.renameLibraryFile(file, name)
        }
    }
    deleteTarget?.let { file ->
        BrowserConfirmDialog("削除しますか？") { ok ->
            deleteTarget = null
            if (ok) { model.deleteLibraryFile(file); selected.remove(file.filepath) }
        }
    }
    if (deleteSelected) BrowserConfirmDialog("削除しますか？") { ok ->
        deleteSelected = false
        if (ok) { model.deleteLibraryFiles(selected.toList()); selected.clear() }
    }
    usageTarget?.let { file ->
        FileUsageDialog(file, load = { model.libraryFileUsage(file) }, onDismiss = { usageTarget = null }) { chat ->
            usageTarget = null
            onDismiss()
            model.openThreadId(chat.id)
        }
    }
}

/** `.lib-modal-header`: folder tile, title, subtitle and the close button over a faint theme gradient. */
@Composable
private fun LibraryHeader(phone: Boolean, onClose: () -> Unit) {
    val web = LocalWebPalette.current
    Row(
        Modifier.fillMaxWidth().background(Brush.verticalGradient(listOf(web.theme.rgb(0.10f), Color.Transparent)))
            .drawBehind { drawLine(if (web.isLight) web.lineSoft else web.line, Offset(0f, size.height), Offset(size.width, size.height), 1.dp.toPx()) }
            .padding(horizontal = if (phone) 16.dp else 22.dp, vertical = if (phone) 14.dp else 18.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Row(Modifier.weight(1f), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(14.dp)) {
            val tile = RoundedCornerShape(14.dp)
            Box(
                Modifier.size(44.dp).clip(tile)
                    .background(Brush.linearGradient(listOf(web.theme.rgb(0.30f), web.theme.rgb(0.08f)), start = Offset.Zero, end = Offset.Infinite))
                    .border(1.dp, web.theme.rgb(0.30f), tile),
                contentAlignment = Alignment.Center,
            ) { FaIcon(R.drawable.fa_solid_folder, null, size = 20.dp, tint = web.theme300) }
            Column(Modifier.weight(1f)) {
                Text("ファイルライブラリ", fontSize = 18.sp, lineHeight = 25.sp, fontWeight = FontWeight.Bold, letterSpacing = 0.18.sp, color = web.text)
                Text("アップロードしたファイルの管理・添付・ダウンロード", fontSize = 12.sp, lineHeight = 17.sp, color = web.muted,
                    modifier = Modifier.padding(top = 2.dp))
            }
        }
        val closeShape = RoundedCornerShape(12.dp)
        Box(
            Modifier.size(38.dp).clip(closeShape)
                .background(if (web.isLight) Color(15, 23, 42).copy(alpha = 0.04f) else Color.White.copy(alpha = 0.03f))
                .border(1.dp, web.lineSoft, closeShape).clickable(role = Role.Button, onClick = onClose),
            contentAlignment = Alignment.Center,
        ) { FaIcon(R.drawable.fa_solid_times, "閉じる", size = 14.dp, tint = web.muted) }
    }
}

/** `.lib-toolbar`: search, 表示順, お気に入りのみ, the file count and the selection actions. */
@Composable
private fun LibraryToolbar(
    state: ChatState,
    phone: Boolean,
    selectedCount: Int,
    onAttach: () -> Unit,
    onDownload: () -> Unit,
    onRename: () -> Unit,
    onUsage: () -> Unit,
    onDelete: () -> Unit,
    model: ChatViewModel,
) {
    val web = LocalWebPalette.current
    val left: @Composable RowScope.() -> Unit = {
        LibrarySearch(state.libraryQuery, model::librarySearch, Modifier.weight(1f).widthIn(max = if (phone) Dp.Unspecified else 360.dp))
        if (!phone) Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(5.dp)) {
            FaIcon(R.drawable.fa_solid_arrows_alt_h, null, size = 12.dp, tint = web.muted)
            Text("表示順", fontSize = 12.sp, color = web.muted, maxLines = 1)
        }
        WebSelect(
            state.librarySort, LIB_SORT_OPTIONS, model::setLibrarySort,
            fontSize = 12.5.sp, background = if (web.isLight) Color.White else Color.White.copy(alpha = 0.04f),
            borderColor = web.line, textColor = if (web.isLight) web.text else web.muted, shape = RoundedCornerShape(11.dp),
            contentPadding = PaddingValues(start = 12.dp, end = 12.dp, top = 10.dp, bottom = 10.dp), contentDescription = "表示順",
        )
        LibActionButton(
            if (state.libraryFavoritesOnly) R.drawable.fa_solid_star else R.drawable.fa_regular_star, "お気に入りのみ", phone,
            tone = if (state.libraryFavoritesOnly) LibTone.Favorite else LibTone.Plain,
            onClick = { model.setLibraryFavoritesOnly(!state.libraryFavoritesOnly) },
        )
    }
    val right: @Composable RowScope.() -> Unit = {
        val total = state.libraryTotal.takeIf { it > 0 } ?: state.library.size
        val count = if (state.library.isEmpty()) "0 files"
            else if (state.libraryHasMore || state.libraryQuery.isNotBlank() || state.libraryFavoritesOnly) "${state.library.size} / $total files"
            else "$total files"
        Text(count, fontSize = 12.sp, lineHeight = 20.sp, color = web.muted, maxLines = 1,
            modifier = Modifier.clip(CircleShape)
                .background(if (web.isLight) Color(15, 23, 42).copy(alpha = 0.04f) else Color.White.copy(alpha = 0.04f))
                .border(1.dp, if (web.isLight) Color(15, 23, 42).copy(alpha = 0.10f) else web.line, CircleShape)
                .padding(horizontal = 11.dp, vertical = 5.dp))
        FlowRow(horizontalArrangement = Arrangement.spacedBy(if (phone) 6.dp else 8.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            val online = !state.offline
            LibActionButton(R.drawable.fa_solid_paperclip, if (selectedCount > 0) "添付 ($selectedCount)" else "添付", phone,
                tone = LibTone.Accent, enabled = selectedCount > 0, onClick = onAttach)
            LibActionButton(R.drawable.fa_solid_download, if (selectedCount > 0) "ダウンロード ($selectedCount)" else "ダウンロード", phone,
                enabled = selectedCount > 0, onClick = onDownload)
            LibActionButton(R.drawable.fa_solid_pen, "名前変更", phone, enabled = selectedCount == 1 && online, onClick = onRename)
            LibActionButton(R.drawable.fa_solid_comment_dots, "使用チャット", phone, enabled = selectedCount == 1 && online, onClick = onUsage)
            LibActionButton(R.drawable.fa_solid_trash, if (selectedCount > 0) "削除 ($selectedCount)" else "削除", phone,
                tone = LibTone.Danger, enabled = selectedCount > 0 && online, onClick = onDelete)
        }
    }
    Column(
        Modifier.fillMaxWidth()
            .background(if (web.isLight) Color(15, 23, 42).copy(alpha = 0.035f) else Color(8, 12, 24).copy(alpha = 0.85f))
            .drawBehind { drawLine(if (web.isLight) web.lineSoft else web.line, Offset(0f, size.height), Offset(size.width, size.height), 1.dp.toPx()) }
            .padding(horizontal = if (phone) 16.dp else 22.dp, vertical = if (phone) 10.dp else 12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        if (phone) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp), content = left)
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp), content = right)
        } else {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Row(Modifier.weight(1f), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp), content = left)
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp), content = right)
            }
        }
    }
}

@Composable
private fun LibrarySearch(value: String, onChange: (String) -> Unit, modifier: Modifier) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(11.dp)
    val style = TextStyle(fontSize = 13.sp, lineHeight = 20.sp, color = web.text, fontFamily = WebFonts.sans)
    BasicTextField(
        value, onChange, singleLine = true, textStyle = style, cursorBrush = SolidColor(web.text),
        modifier = modifier.clip(shape).background(if (web.isLight) Color.White else Color.White.copy(alpha = 0.04f))
            .border(1.dp, web.line, shape).semantics { contentDescription = "ファイル名を検索" },
        decorationBox = { inner ->
            Row(Modifier.padding(start = 12.dp, end = 12.dp, top = 10.dp, bottom = 10.dp), verticalAlignment = Alignment.CenterVertically) {
                FaIcon(R.drawable.fa_solid_search, null, size = 13.dp, tint = web.muted)
                Box(Modifier.weight(1f).padding(start = 9.dp)) {
                    if (value.isEmpty()) Text("ファイル名を検索", style = style.copy(color = web.muted), maxLines = 1)
                    inner()
                }
            }
        },
    )
}

private enum class LibTone { Plain, Accent, Danger, Favorite }

/** `.lib-action-btn` (and its accent, danger and active filter variants); phones show only the icon. */
@Composable
private fun LibActionButton(
    @DrawableRes icon: Int,
    label: String,
    phone: Boolean,
    tone: LibTone = LibTone.Plain,
    enabled: Boolean = true,
    onClick: () -> Unit,
) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(11.dp)
    val (bg, border, fg) = when (tone) {
        LibTone.Plain -> Triple(if (web.isLight) Color(15, 23, 42).copy(alpha = 0.04f) else Color.White.copy(alpha = 0.04f), web.lineStrong, web.muted)
        LibTone.Accent -> Triple(web.theme.rgb(0.16f), web.theme.rgb(0.40f), web.theme200)
        LibTone.Danger -> Triple(Color(249, 112, 141).copy(alpha = 0.12f), Color(249, 112, 141).copy(alpha = 0.35f),
            if (web.isLight) Color(0xFFBE123C) else Color(0xFFFFD2DB))
        LibTone.Favorite -> Triple(Color(250, 204, 21).copy(alpha = 0.16f), Color(250, 204, 21).copy(alpha = 0.48f),
            if (web.isLight) Color(0xFF92400E) else Color(0xFFFDE68A))
    }
    Row(
        Modifier.alpha(if (enabled) 1f else 0.45f).clip(shape).background(bg).border(1.dp, border, shape)
            .clickable(enabled = enabled, role = Role.Button, onClick = onClick)
            .semantics { contentDescription = label }
            .padding(horizontal = if (phone) 10.dp else 13.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        FaIcon(icon, null, size = 12.dp, tint = fg)
        if (!phone) Text(label, fontSize = 12.sp, lineHeight = 14.sp, fontWeight = FontWeight.SemiBold, color = fg, maxLines = 1)
    }
}

/** `.library-thumb-card`: preview (image or file tile), the name bar and the ★ / open / delete circles. */
@Composable
private fun LibraryCard(
    file: LibraryFile,
    selected: Boolean,
    showActions: Boolean,
    loader: FileBytesLoader,
    offline: Boolean,
    onToggle: () -> Unit,
    onFavorite: () -> Unit,
    onOpen: () -> Unit,
    onDelete: () -> Unit,
) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(14.dp)
    val ring = web.theme.rgb(0.5f)
    Box(
        Modifier.fillMaxWidth().height(210.dp)
            .drawBehind {
                if (selected) {
                    val spread = 2.dp.toPx()
                    drawRoundRect(ring, topLeft = Offset(-spread, -spread),
                        size = androidx.compose.ui.geometry.Size(size.width + spread * 2, size.height + spread * 2),
                        cornerRadius = androidx.compose.ui.geometry.CornerRadius(16.dp.toPx(), 16.dp.toPx()))
                }
            }
            .clip(shape).background(web.panel)
            .background(Brush.verticalGradient(listOf(Color.White.copy(alpha = 0.04f), Color.White.copy(alpha = 0.01f))))
            .border(1.dp, if (selected) web.theme.t500 else web.line, shape)
            .clickable(role = Role.Checkbox, onClick = onToggle),
    ) {
        Column(Modifier.fillMaxSize()) {
            Box(
                Modifier.weight(1f).fillMaxWidth()
                    .background(Brush.linearGradient(0f to web.theme.rgb(0.08f), 0.6f to Color.Transparent, start = Offset.Zero, end = Offset.Infinite)),
                contentAlignment = Alignment.Center,
            ) {
                if (file.type == "image") {
                    ProtectedImage(
                        file.url.ifBlank { file.filepath }, loader, onOpen = null, modifier = Modifier.fillMaxSize(),
                        thumbnail = true, contentDescription = file.displayName, shape = RoundedCornerShape(0.dp), contentScale = ContentScale.Crop,
                    )
                } else {
                    val ext = file.extName()
                    Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(9.6.dp)) {
                        val tile = RoundedCornerShape(16.dp)
                        Box(
                            Modifier.size(58.dp).clip(tile).background(web.theme.rgb(0.10f)).border(1.dp, web.theme.rgb(0.18f), tile),
                            contentAlignment = Alignment.Center,
                        ) { FaIcon(libraryFileIcon(ext), null, size = 30.dp, tint = web.theme300) }
                        Text(ext.uppercase().ifBlank { "FILE" }, fontSize = 9.sp, lineHeight = 20.sp, fontWeight = FontWeight.ExtraBold, letterSpacing = 0.72.sp,
                            color = web.theme200,
                            modifier = Modifier.clip(CircleShape).background(web.theme.rgb(0.14f)).border(1.dp, web.theme.rgb(0.20f), CircleShape)
                                .padding(horizontal = 9.dp, vertical = 2.dp))
                    }
                }
            }
            val barTop = if (web.isLight) Color(15, 23, 42).copy(alpha = 0.08f) else Color.White.copy(alpha = 0.05f)
            Text(
                file.displayName, fontSize = 11.sp, lineHeight = 14.3.sp, color = web.text, maxLines = 1, overflow = TextOverflow.Ellipsis,
                modifier = Modifier.fillMaxWidth()
                    .background(Brush.verticalGradient(
                        if (web.isLight) listOf(Color.White.copy(alpha = 0.86f), Color(241, 245, 249).copy(alpha = 0.96f))
                        else listOf(Color(5, 7, 15).copy(alpha = 0.35f), Color(5, 7, 15).copy(alpha = 0.82f)),
                    ))
                    .drawBehind { drawLine(barTop, Offset.Zero, Offset(size.width, 0f), 1.dp.toPx()) }
                    .padding(horizontal = 10.4.dp, vertical = 8.dp),
            )
        }
        if (showActions) Row(Modifier.align(Alignment.TopEnd).padding(8.dp), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            LibCircle(
                if (file.isFavorite) R.drawable.fa_solid_star else R.drawable.fa_regular_star,
                if (file.isFavorite) "お気に入りから外す" else "お気に入りに追加",
                tint = if (file.isFavorite) Color(0xFFFACC15) else if (web.isLight) Color(0xFF64748B) else Color(0xFFCBD5E1),
                highlight = file.isFavorite, enabled = !offline, onClick = onFavorite,
            )
            LibCircle(R.drawable.fa_solid_eye, "開く", tint = web.text, onClick = onOpen)
            LibCircle(R.drawable.fa_solid_trash, "削除", tint = if (web.isLight) Color(0xFFBE123C) else web.danger, enabled = !offline, onClick = onDelete)
        }
    }
}

@Composable
private fun LibCircle(@DrawableRes icon: Int, label: String, tint: Color, highlight: Boolean = false, enabled: Boolean = true, onClick: () -> Unit) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(10.dp)
    val bg = when {
        highlight -> Color(250, 204, 21).copy(alpha = 0.16f)
        web.isLight -> Color.White.copy(alpha = 0.94f)
        else -> Color(13, 20, 38).copy(alpha = 0.94f)
    }
    val border = when {
        highlight -> Color(250, 204, 21).copy(alpha = 0.48f)
        web.isLight -> Color(15, 23, 42).copy(alpha = 0.14f)
        else -> Color.White.copy(alpha = 0.14f)
    }
    Box(
        Modifier.size(30.dp).alpha(if (enabled) 1f else 0.55f).clip(shape).background(bg).border(1.dp, border, shape)
            .clickable(enabled = enabled, role = Role.Button, onClick = onClick).semantics { contentDescription = label },
        contentAlignment = Alignment.Center,
    ) { FaIcon(icon, null, size = 12.dp, tint = tint) }
}

/** `.lib-skeleton-card` with the shimmer (`libShimmer`, 1.4s). */
@Composable
private fun LibrarySkeletonCard() {
    val web = LocalWebPalette.current
    val reduce = LocalReduceMotion.current
    val shift = if (reduce) -1f else rememberInfiniteTransition(label = "lib shimmer").animateFloat(
        -1f, 1f, infiniteRepeatable(tween(1400, easing = LinearEasing), RepeatMode.Restart), label = "lib shimmer x",
    ).value
    val shine = Color.White.copy(alpha = if (web.isLight) 0.35f else 0.08f)
    val shimmer = Modifier.drawBehind {
        val w = size.width
        drawRect(Brush.horizontalGradient(listOf(Color.Transparent, shine, Color.Transparent), startX = shift * w, endX = shift * w + w))
    }
    val shape = RoundedCornerShape(14.dp)
    Column(Modifier.fillMaxWidth().height(210.dp).clip(shape).background(web.panel).border(1.dp, web.line, shape)) {
        Box(Modifier.weight(1f).fillMaxWidth().background(Color(226, 232, 240).copy(alpha = 0.05f)).then(shimmer))
        Column(
            Modifier.fillMaxWidth().height(44.dp)
                .background(if (web.isLight) Color(15, 23, 42).copy(alpha = 0.05f) else Color(5, 7, 15).copy(alpha = 0.6f))
                .padding(horizontal = 12.dp),
            verticalArrangement = Arrangement.spacedBy(7.dp, Alignment.CenterVertically),
        ) {
            val line = if (web.isLight) Color(15, 23, 42).copy(alpha = 0.08f) else Color(226, 232, 240).copy(alpha = 0.08f)
            Box(Modifier.fillMaxWidth(0.78f).height(9.dp).clip(CircleShape).background(line).then(shimmer))
            Box(Modifier.fillMaxWidth(0.45f).height(9.dp).clip(CircleShape).background(line).then(shimmer))
        }
    }
}

/** `.lib-empty-state`. */
@Composable
private fun LibraryEmpty(@DrawableRes icon: Int, title: String, sub: String) {
    val web = LocalWebPalette.current
    Column(
        Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 60.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        val tile = RoundedCornerShape(22.dp)
        Box(
            Modifier.size(72.dp).clip(tile).background(web.theme.rgb(0.10f)).border(1.dp, web.theme.rgb(0.20f), tile),
            contentAlignment = Alignment.Center,
        ) { FaIcon(icon, null, size = 30.dp, tint = web.theme300) }
        Text(title, fontSize = 15.sp, fontWeight = FontWeight.SemiBold, color = web.text, textAlign = TextAlign.Center)
        Text(sub, fontSize = 12.sp, lineHeight = 19.2.sp, color = web.muted, textAlign = TextAlign.Center, modifier = Modifier.widthIn(max = 280.dp))
    }
}

/** `#lib-usage-modal` (使用されているチャット). */
@Composable
private fun FileUsageDialog(
    file: LibraryFile,
    load: suspend () -> Pair<List<FileUsageChat>, Boolean>,
    onDismiss: () -> Unit,
    onOpen: (FileUsageChat) -> Unit,
) {
    val web = LocalWebPalette.current
    var result by remember(file.filepath) { mutableStateOf<Pair<List<FileUsageChat>, Boolean>?>(null) }
    var failed by remember(file.filepath) { mutableStateOf(false) }
    LaunchedEffect(file.filepath) { runCatching { load() }.onSuccess { result = it }.onFailure { failed = true } }
    WebOverlayModal(onDismiss, if (web.isLight) Color(15, 23, 42).copy(alpha = 0.35f) else Color(3, 7, 16).copy(alpha = 0.72f), 10.dp) { phone ->
        BoxWithConstraints(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            val shape = RoundedCornerShape(12.dp)
            val rule = web.twBorder(Tw.gray700)
            Column(
                Modifier.padding(16.dp).widthIn(max = 576.dp).heightIn(max = maxHeight * 0.8f).fillMaxWidth().clip(shape)
                    .background(web.twBg(Tw.gray900)).border(1.dp, web.twBorder(Tw.gray700), shape).padding(20.dp),
            ) {
                Row(
                    Modifier.fillMaxWidth()
                        .drawBehind { drawLine(rule, Offset(0f, size.height), Offset(size.width, size.height), 1.dp.toPx()) }
                        .padding(bottom = 12.dp),
                    verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    Column(Modifier.weight(1f)) {
                        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            FaIcon(R.drawable.fa_solid_comment_dots, null, size = 16.dp, tint = web.twText(Tw.cyan300))
                            Text("使用されているチャット", fontSize = 16.sp, lineHeight = 24.sp, fontWeight = FontWeight.Bold, color = web.twText(Tw.white))
                        }
                        Text(file.displayName, fontSize = 12.sp, lineHeight = 16.sp, color = web.twText(Tw.gray400), maxLines = 1,
                            overflow = TextOverflow.Ellipsis, modifier = Modifier.padding(top = 4.dp))
                    }
                    val closeShape = RoundedCornerShape(12.dp)
                    Box(
                        Modifier.size(38.dp).clip(closeShape).background(Color.White.copy(alpha = 0.03f)).border(1.dp, web.lineSoft, closeShape)
                            .clickable(role = Role.Button, onClick = onDismiss),
                        contentAlignment = Alignment.Center,
                    ) { FaIcon(R.drawable.fa_solid_times, "閉じる", size = 14.dp, tint = web.muted) }
                }
                Column(Modifier.padding(top = 12.dp).verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    val data = result
                    when {
                        failed -> Row(Modifier.fillMaxWidth().padding(vertical = 32.dp), horizontalArrangement = Arrangement.Center,
                            verticalAlignment = Alignment.CenterVertically) {
                            FaIcon(R.drawable.fa_solid_exclamation_triangle, null, size = 14.dp, tint = web.twText(Tw.red300), modifier = Modifier.padding(end = 8.dp))
                            Text("使用チャットの取得に失敗しました。", fontSize = 14.sp, color = web.twText(Tw.red300))
                        }
                        data == null -> Row(Modifier.fillMaxWidth().padding(vertical = 32.dp), horizontalArrangement = Arrangement.Center,
                            verticalAlignment = Alignment.CenterVertically) {
                            FaIcon(R.drawable.fa_solid_spinner, null, size = 14.dp, tint = web.twText(Tw.gray400), modifier = Modifier.padding(end = 8.dp))
                            Text("読み込み中…", fontSize = 14.sp, color = web.twText(Tw.gray400))
                        }
                        data.first.isEmpty() -> Column(Modifier.fillMaxWidth().padding(vertical = 32.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                            FaIcon(R.drawable.fa_solid_comment_dots, null, size = 20.dp, tint = web.twText(Tw.gray400), modifier = Modifier.padding(bottom = 8.dp))
                            Text("このファイルを使用しているチャットはありません。", fontSize = 14.sp, color = web.twText(Tw.gray400), textAlign = TextAlign.Center)
                        }
                        else -> {
                            data.first.forEach { chat ->
                                val rowShape = RoundedCornerShape(8.dp)
                                Row(
                                    Modifier.fillMaxWidth().clip(rowShape).background(web.twBg(Tw.gray800).copy(alpha = 0.7f))
                                        .border(1.dp, web.twBorder(Tw.gray700), rowShape).padding(12.dp),
                                    verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp),
                                ) {
                                    Column(Modifier.weight(1f)) {
                                        Text(chat.title, fontSize = 14.sp, lineHeight = 20.sp, color = web.twText(Tw.gray200), maxLines = 1, overflow = TextOverflow.Ellipsis)
                                        Text(if (chat.updatedAt.isBlank()) "" else webLocaleTime(chat.updatedAt), fontSize = 11.sp, lineHeight = 16.sp,
                                            color = web.twText(Tw.gray500), modifier = Modifier.padding(top = 4.dp))
                                    }
                                    LibActionButton(R.drawable.fa_solid_folder, "開く", phone, tone = LibTone.Accent) { onOpen(chat) }
                                }
                            }
                            if (data.second) Text("表示できるチャットは最大100件です。", fontSize = 11.sp, color = web.twText(Tw.gray500),
                                textAlign = TextAlign.Center, modifier = Modifier.fillMaxWidth().padding(top = 8.dp))
                        }
                    }
                }
            }
        }
    }
}
