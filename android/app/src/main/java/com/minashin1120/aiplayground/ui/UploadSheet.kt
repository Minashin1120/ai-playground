package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.ChatViewModel
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.Attachment
import com.minashin1120.aiplayground.data.isImageReference

/** Web `#vision-model-info`: DeepSeek models without native image input analyse images with the Vision Model. */
internal fun needsVisionModelNotice(model: String): Boolean {
    val m = model.lowercase()
    return m.contains("deepseek") && m != "deepseek-v4.1-flash" && m != "deepseek-v4-flash-vision-exp"
}

/**
 * Web `#upload-modal` as a bottom sheet (ANDROID_ONLY.md): the same title, buttons, Vision Model row,
 * progress line and `#upload-list` rows (newest first). The drop zone is left out because there is no
 * drag and drop on Android.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
internal fun UploadSheet(
    state: ChatState,
    model: ChatViewModel,
    loader: FileBytesLoader?,
    onDismiss: () -> Unit,
    onPickFiles: () -> Unit,
    onCamera: () -> Unit,
    onPhotos: () -> Unit,
    onLibrary: () -> Unit,
    onChangeVisionModel: () -> Unit,
    onOpenFile: (String) -> Unit,
    onEditImage: (Attachment) -> Unit,
) {
    val web = LocalWebPalette.current
    val sheet = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    var renaming by remember { mutableStateOf<Attachment?>(null) }
    ModalBottomSheet(onDismissRequest = onDismiss, sheetState = sheet, containerColor = web.twBg(Tw.gray800)) {
        val columns = if (LocalConfiguration.current.screenWidthDp >= 768) 3 else 2
        Column(
            Modifier.fillMaxWidth().verticalScroll(rememberScrollState())
                .padding(start = 24.dp, end = 24.dp, bottom = 24.dp),
        ) {
            Row(Modifier.fillMaxWidth().padding(bottom = 12.dp), verticalAlignment = Alignment.CenterVertically) {
                FaIcon(R.drawable.fa_solid_paperclip, null, size = 16.dp, tint = web.twText(Tw.blue400))
                Text("ファイルアップロード", fontSize = 18.sp, fontWeight = FontWeight.Bold, color = web.text,
                    modifier = Modifier.padding(start = 8.dp).weight(1f))
                Box(
                    Modifier.size(32.dp).clip(CircleShape).clickable(role = Role.Button, onClick = onDismiss),
                    contentAlignment = Alignment.Center,
                ) { FaIcon(R.drawable.fa_solid_times, "閉じる", size = 14.dp, tint = Tw.gray400) }
            }
            val buttons = listOf<Triple<String, Color, () -> Unit>>(
                Triple("ファイルを選択", Tw.blue600, onPickFiles),
                Triple("カメラで撮影", Tw.emerald600, onCamera),
                Triple("写真", Tw.sky600, onPhotos),
                Triple("ライブラリから選択", Tw.gray700, onLibrary),
                Triple("リストをクリア", Tw.gray700, { model.resetUploads() }),
            )
            Column(Modifier.padding(bottom = 16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                buttons.chunked(columns).forEach { row ->
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        row.forEach { (label, color, action) -> UploadModalButton(label, color, Modifier.weight(1f), action) }
                        repeat(columns - row.size) { Spacer(Modifier.weight(1f)) }
                    }
                }
            }
            if (needsVisionModelNotice(state.model)) {
                val shape = RoundedCornerShape(4.dp)
                val vision = state.visionModel ?: state.preferences?.defaultVisionModel
                val display = vision?.let { id -> state.account?.models?.firstOrNull { it.id == id }?.name ?: id } ?: "設定から選択"
                Row(
                    Modifier.fillMaxWidth().padding(bottom = 12.dp).clip(shape).background(web.twBg(Tw.gray900, 0.5f))
                        .border(1.dp, web.twBorder(Tw.gray700), shape).padding(8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(Modifier.weight(1f)) {
                        Text("Vision Model（画像解析用）", fontSize = 10.sp, color = Tw.gray400)
                        Text(display, fontSize = 12.sp, fontWeight = FontWeight.Medium, color = web.twText(Tw.gray200))
                    }
                    Text("変更", fontSize = 10.sp, color = web.twText(Tw.gray300),
                        modifier = Modifier.clip(shape).background(web.twBg(Tw.gray700))
                            .clickable(role = Role.Button, onClick = onChangeVisionModel).padding(horizontal = 8.dp, vertical = 4.dp))
                }
            }
            Row(Modifier.fillMaxWidth().padding(bottom = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                val status = if (state.uploading) " (${state.uploadCompleted}/${state.uploadCount})" else ""
                Text("進行状況$status", fontSize = 12.sp, color = Tw.gray400, modifier = Modifier.weight(1f))
                if (state.uploading) {
                    Box(Modifier.width(128.dp).height(6.dp).clip(CircleShape).background(web.twBg(Tw.gray900))) {
                        Box(Modifier.fillMaxHeight().fillMaxWidth(uploadFraction(state)).background(Tw.blue500))
                    }
                }
            }
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                if (state.uploading && state.uploadName.isNotBlank()) {
                    val pct = if (state.uploadTotal > 0) (state.uploadSent * 100 / state.uploadTotal).toInt().coerceIn(0, 100) else 0
                    UploadRow(
                        name = state.uploadName, status = "$pct%", progress = pct / 100f, preview = null,
                        onRemove = model::cancelUpload,
                    )
                }
                state.attachments.asReversed().forEach { attachment ->
                    key(attachment.reference) {
                        val image = isImageReference(attachment.name) || attachment.mime.startsWith("image/")
                        val preview: (@Composable (Modifier) -> Unit)? = if (image) {
                            { modifier ->
                                ProtectedImage(attachment.reference, loader, onOpen = onOpenFile, modifier = modifier,
                                    shape = RoundedCornerShape(4.dp), contentScale = ContentScale.Crop, contentDescription = attachment.name)
                            }
                        } else null
                        UploadRow(
                            name = attachment.name,
                            status = when {
                                state.editingAttachment == attachment.reference -> "編集反映中..."
                                attachment.source == "library" -> "ready"
                                else -> "完了"
                            },
                            progress = 1f,
                            edited = attachment.edited,
                            preview = preview,
                            onOpen = { onOpenFile(attachment.reference) },
                            onEdit = if (image) ({ onEditImage(attachment) }) else null,
                            onRename = { renaming = attachment },
                            onRemove = { model.removeAttachment(attachment.reference) },
                        )
                    }
                }
                if (state.attachments.isEmpty() && !state.uploading) {
                    Text("まだアップロードがありません。", fontSize = 12.sp, color = Tw.gray500)
                }
            }
        }
    }
    renaming?.let { attachment ->
        BrowserPromptDialog("送信時のファイル名を入力してください（空欄でデフォルトに戻す）", attachment.name) { input ->
            renaming = null
            if (input != null) model.renameAttachment(attachment.reference, input)
        }
    }
}

internal fun uploadFraction(state: ChatState): Float {
    if (state.uploadCount <= 0) return 0f
    val current = if (state.uploadTotal > 0) (state.uploadSent.toFloat() / state.uploadTotal).coerceIn(0f, 1f) else 0f
    return ((state.uploadCompleted + current) / state.uploadCount).coerceIn(0f, 1f)
}

@Composable
private fun UploadModalButton(label: String, color: Color, modifier: Modifier, onClick: () -> Unit) {
    val shape = RoundedCornerShape(4.dp)
    Box(
        modifier.heightIn(min = 40.dp).clip(shape).background(color).clickable(role = Role.Button, onClick = onClick)
            .padding(horizontal = 16.dp, vertical = 10.dp),
        contentAlignment = Alignment.Center,
    ) { Text(label, fontSize = 14.sp, fontWeight = FontWeight.Bold, color = Tw.white, maxLines = 1, overflow = TextOverflow.Ellipsis) }
}

/** `.upload-row`: preview, name, status (+ "編集済み"), 画像編集 / 送信名 / 削除, and the progress bar. */
@Composable
private fun UploadRow(
    name: String,
    status: String,
    progress: Float,
    preview: (@Composable (Modifier) -> Unit)?,
    edited: Boolean = false,
    onOpen: (() -> Unit)? = null,
    onEdit: (() -> Unit)? = null,
    onRename: (() -> Unit)? = null,
    onRemove: () -> Unit,
) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(4.dp)
    Column(
        Modifier.fillMaxWidth().clip(shape).background(web.twBg(Tw.gray900, 0.6f))
            .border(1.dp, Color(148, 163, 184).copy(alpha = 0.14f), shape).padding(8.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            val thumb = Modifier.size(48.dp).clip(shape).border(1.dp, web.twBorder(Tw.gray700), shape)
            if (preview != null) preview(thumb)
            else Box(
                thumb.background(web.twBg(Tw.gray800)).then(if (onOpen != null) Modifier.clickable(onClick = onOpen) else Modifier),
                contentAlignment = Alignment.Center,
            ) { Text("FILE", fontSize = 14.sp, color = Tw.gray400) }
            Column(Modifier.weight(1f)) {
                Text(name, fontSize = 12.sp, color = web.twText(Tw.gray200), maxLines = 1, overflow = TextOverflow.Ellipsis)
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(status, fontSize = 10.sp, color = Tw.gray400)
                    if (edited) Text("編集済み", fontSize = 10.sp, color = Color(0xFFFDE68A),
                        modifier = Modifier.clip(CircleShape).background(Color(250, 204, 21).copy(alpha = 0.16f))
                            .border(1.dp, Color(250, 204, 21).copy(alpha = 0.3f), CircleShape).padding(horizontal = 6.dp, vertical = 1.dp))
                }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                if (onEdit != null) RowButton("画像編集", Color(0xFFFDE68A), Color(250, 204, 21).copy(alpha = 0.14f),
                    Color(250, 204, 21).copy(alpha = 0.35f), onEdit)
                if (onRename != null) RowButton("送信名", web.twText(Tw.gray300), Color.Transparent, web.twBorder(Tw.gray700), onRename)
                RowButton("削除", Tw.gray400, Color.Transparent, web.twBorder(Tw.gray700), onRemove)
            }
        }
        Box(
            Modifier.padding(top = 8.dp).fillMaxWidth().height(8.dp).clip(shape)
                .background(web.theme.rgb(0.2f)),
        ) {
            Box(Modifier.fillMaxHeight().fillMaxWidth(progress.coerceIn(0f, 1f))
                .background(Brush.horizontalGradient(listOf(web.theme.t300, web.theme.t500))))
        }
    }
}

@Composable
private fun RowButton(label: String, color: Color, background: Color, border: Color, onClick: () -> Unit, padding: Dp = 8.dp) {
    val shape = RoundedCornerShape(4.dp)
    Text(label, fontSize = 10.sp, color = color, maxLines = 1,
        modifier = Modifier.clip(shape).background(background).border(1.dp, border, shape)
            .clickable(role = Role.Button, onClick = onClick).padding(horizontal = padding, vertical = 4.dp))
}
