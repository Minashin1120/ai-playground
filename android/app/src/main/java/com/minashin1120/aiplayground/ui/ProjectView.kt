package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.ChatViewModel
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.ProjectFileItem
import com.minashin1120.aiplayground.data.ProjectItem
import com.minashin1120.aiplayground.data.ThreadItem
import com.minashin1120.aiplayground.data.direct.ProjectFileTool

/**
 * Dedicated Project view (Web `#project-view`): project header, system prompt preview,
 * shared files with inline per-project Vision model toggle/select, and project chats list.
 */
@Composable
internal fun ProjectView(
    state: ChatState,
    model: ChatViewModel,
    project: ProjectItem,
    onBack: () -> Unit,
    onNewChat: () -> Unit,
    onEdit: () -> Unit,
    onDelete: () -> Unit,
    onUploadFiles: () -> Unit,
    onAddFromLibrary: () -> Unit,
    onEditFileNote: (ProjectFileItem) -> Unit,
    onRemoveFile: (ProjectFileItem) -> Unit,
    onOpenFile: (ProjectFileItem) -> Unit,
    onOpenThread: (ThreadItem) -> Unit,
    onRenameThread: (ThreadItem) -> Unit,
    onMoveThread: (ThreadItem) -> Unit,
    onDeleteThread: (ThreadItem) -> Unit,
    modifier: Modifier = Modifier,
) {
    val web = LocalWebPalette.current
    val cardShape = RoundedCornerShape(14.dp)
    val cardBg = if (web.isLight) Color.White else Color(10, 15, 28).copy(alpha = 0.78f)
    val visionOptions = remember(state.account?.models, state.preferences?.defaultVisionModel) {
        projectVisionModelOptions(state)
    }

    Column(
        modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp, vertical = 14.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        // Header
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            Box(
                Modifier
                    .size(34.dp)
                    .clip(RoundedCornerShape(10.dp))
                    .background(cardBg)
                    .border(1.dp, web.line, RoundedCornerShape(10.dp))
                    .clickable(role = Role.Button, onClickLabel = "戻る", onClick = onBack),
                contentAlignment = Alignment.Center,
            ) { FaIcon(R.drawable.fa_solid_arrow_left, "戻る", size = 13.dp, tint = web.text) }
            Column(Modifier.weight(1f)) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    FaIcon(R.drawable.fa_solid_folder, null, size = 15.dp, tint = web.theme300)
                    Text(
                        project.name,
                        color = web.text, fontSize = 18.sp, fontWeight = FontWeight.Bold,
                        maxLines = 1, overflow = TextOverflow.Ellipsis,
                    )
                }
                if (project.description.isNotBlank()) {
                    Text(
                        project.description,
                        color = web.twText(Tw.gray400), fontSize = 12.sp, maxLines = 2, overflow = TextOverflow.Ellipsis,
                    )
                }
            }
            if (state.projectsBusy) {
                CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp, color = web.theme300)
            }
        }

        // Primary action bar
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
            WebButton(onClick = onNewChat, variant = WebButtonVariant.Primary, modifier = Modifier.weight(1f)) {
                FaIcon(R.drawable.fa_solid_plus, null, size = 11.dp, tint = Color.White)
                Spacer(Modifier.width(6.dp))
                Text("このプロジェクトで新しいチャット", fontSize = 12.sp)
            }
            Box(
                Modifier
                    .height(36.dp)
                    .clip(RoundedCornerShape(8.dp))
                    .background(cardBg)
                    .border(1.dp, web.line, RoundedCornerShape(8.dp))
                    .clickable(role = Role.Button, onClickLabel = "編集", onClick = onEdit)
                    .padding(horizontal = 10.dp),
                contentAlignment = Alignment.Center,
            ) { FaIcon(R.drawable.fa_solid_pencil_alt, "編集", size = 12.dp, tint = web.text) }
            Box(
                Modifier
                    .height(36.dp)
                    .clip(RoundedCornerShape(8.dp))
                    .background(cardBg)
                    .border(1.dp, web.line, RoundedCornerShape(8.dp))
                    .clickable(role = Role.Button, onClickLabel = "削除", onClick = onDelete)
                    .padding(horizontal = 10.dp),
                contentAlignment = Alignment.Center,
            ) { FaIcon(R.drawable.fa_solid_trash, "削除", size = 12.dp, tint = Tw.red400) }
        }

        // System Prompt Card
        Column(
            Modifier.fillMaxWidth().clip(cardShape).background(cardBg).border(1.dp, web.line, cardShape).padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Text("プロジェクトのシステムプロンプト", color = web.text, fontSize = 13.sp, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
                Text(
                    "編集",
                    color = web.theme300, fontSize = 12.sp, fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.clickable(role = Role.Button, onClick = onEdit).padding(horizontal = 4.dp, vertical = 2.dp),
                )
            }
            Text(
                project.instruction.ifBlank { "（未設定）プロジェクト内のチャットに共通の指示を追加できます。" },
                color = if (project.instruction.isBlank()) web.twText(Tw.gray500) else web.twText(Tw.gray300),
                fontSize = 12.sp, lineHeight = 18.sp, maxLines = 6, overflow = TextOverflow.Ellipsis,
            )
        }

        // Shared Files Card + Inline Vision Setting
        Column(
            Modifier.fillMaxWidth().clip(cardShape).background(cardBg).border(1.dp, web.line, cardShape).padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                Text(
                    "共有ファイル (${project.files.size}/20)",
                    color = web.text, fontSize = 13.sp, fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.weight(1f),
                )
                ProjectSmallActionButton(R.drawable.fa_solid_file_upload, "アップロード", onClick = onUploadFiles)
                ProjectSmallActionButton(R.drawable.fa_solid_folder, "ライブラリ", onClick = onAddFromLibrary)
            }

            // Inline Vision setting for shared images
            val visionBoxShape = RoundedCornerShape(10.dp)
            Column(
                Modifier
                    .fillMaxWidth()
                    .clip(visionBoxShape)
                    .background(web.twBg(Tw.gray900).copy(alpha = 0.45f))
                    .border(1.dp, web.line, visionBoxShape)
                    .padding(10.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f).padding(end = 8.dp)) {
                        Text("画像ファイルにVisionモデルを使う", color = web.text, fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
                        Text(
                            if (project.useVisionForImages) "ON: モデルが必要時にVisionモデルで画像を解析します"
                            else "OFF: 画像の中身は読まずメタ情報のみ渡します",
                            color = web.twText(Tw.gray400), fontSize = 10.sp, lineHeight = 14.sp,
                        )
                    }
                    WebToggle(
                        checked = project.useVisionForImages,
                        onCheckedChange = { enabled ->
                            model.updateProjectVisionSettings(project.uuid, enabled, project.visionModel)
                        },
                        activeColor = Tw.emerald500,
                    )
                }
                if (project.useVisionForImages) {
                    WebSelect(
                        value = project.visionModel.orEmpty(),
                        options = visionOptions,
                        onSelect = { chosen ->
                            model.updateProjectVisionSettings(project.uuid, true, chosen.ifBlank { null })
                        },
                        fillWidth = true,
                        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 6.dp),
                    )
                }
            }

            if (project.files.isEmpty()) {
                Text(
                    "共有ファイルはまだありません。ファイルを追加すると、モデルが必要に応じて参照できます。",
                    color = web.twText(Tw.gray500), fontSize = 12.sp,
                )
            } else {
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    project.files.forEach { file ->
                        ProjectFileRow(
                            file = file,
                            useVision = project.useVisionForImages,
                            onOpen = { onOpenFile(file) },
                            onEditNote = { onEditFileNote(file) },
                            onRemove = { onRemoveFile(file) },
                        )
                    }
                }
            }
        }

        // Project Chats Card
        Column(
            Modifier.fillMaxWidth().clip(cardShape).background(cardBg).border(1.dp, web.line, cardShape).padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Text(
                "プロジェクト内のチャット (${project.threadCount})",
                color = web.text, fontSize = 13.sp, fontWeight = FontWeight.SemiBold,
            )
            TwInput(
                value = state.projectThreadsQuery,
                onChange = model::searchProjectThreads,
                placeholder = "このプロジェクト内のチャットを検索...",
                fontSize = 12.sp,
                padding = 8.dp,
            )
            if (state.projectThreads.isEmpty()) {
                Text(
                    if (state.projectThreadsQuery.isNotBlank()) "一致するチャットがありません。"
                    else "このプロジェクトにはまだチャットがありません。",
                    color = web.twText(Tw.gray500), fontSize = 12.sp,
                )
            } else {
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    state.projectThreads.forEach { thread ->
                        ProjectThreadRow(
                            thread = thread,
                            offline = state.offline,
                            onOpen = { onOpenThread(thread) },
                            onBookmark = { model.toggleBookmark(thread) },
                            onMove = { onMoveThread(thread) },
                            onRename = { onRenameThread(thread) },
                            onDelete = { onDeleteThread(thread) },
                        )
                    }
                }
                if (state.projectThreadsNextPage != null) {
                    WebButton(
                        onClick = model::loadMoreProjectThreads,
                        variant = WebButtonVariant.Ghost,
                        modifier = Modifier.fillMaxWidth(),
                    ) { Text("さらに読み込む", fontSize = 12.sp) }
                }
            }
        }
        Spacer(Modifier.height(16.dp))
    }
}

@Composable
private fun ProjectSmallActionButton(icon: Int, label: String, onClick: () -> Unit) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(8.dp)
    Row(
        Modifier
            .clip(shape)
            .background(web.theme.rgb(0.14f))
            .border(1.dp, web.theme.rgb(0.35f), shape)
            .clickable(role = Role.Button, onClick = onClick)
            .padding(horizontal = 8.dp, vertical = 5.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        FaIcon(icon, null, size = 10.dp, tint = web.theme300)
        Text(label, color = web.theme300, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
    }
}

@Composable
private fun ProjectFileRow(
    file: ProjectFileItem,
    useVision: Boolean,
    onOpen: () -> Unit,
    onEditNote: () -> Unit,
    onRemove: () -> Unit,
) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(10.dp)
    Row(
        Modifier
            .fillMaxWidth()
            .clip(shape)
            .background(web.twBg(Tw.gray900).copy(alpha = 0.4f))
            .border(1.dp, web.line, shape)
            .clickable(onClick = onOpen)
            .padding(horizontal = 10.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        FaIcon(
            if (file.isImage) R.drawable.fa_solid_image else R.drawable.fa_solid_file_lines,
            null, size = 14.dp, tint = if (file.isImage) Tw.emerald400 else web.theme300,
        )
        Column(Modifier.weight(1f)) {
            Text(file.filename, color = web.text, fontSize = 12.sp, fontWeight = FontWeight.Medium, maxLines = 1, overflow = TextOverflow.Ellipsis)
            val meta = buildString {
                append(ProjectFileTool.formatSize(file.size))
                if (file.isImage) append(if (useVision) " · Vision: ON" else " · Vision: OFF")
            }
            Text(meta, color = web.twText(Tw.gray400), fontSize = 10.sp)
            if (file.note.isNotBlank()) {
                Text(file.note, color = web.twText(Tw.gray300), fontSize = 11.sp, maxLines = 2, overflow = TextOverflow.Ellipsis)
            }
        }
        Box(
            Modifier.clickable(role = Role.Button, onClickLabel = "メモを編集", onClick = onEditNote).padding(6.dp),
        ) { FaIcon(R.drawable.fa_solid_pen, "メモを編集", size = 11.dp, tint = web.twText(Tw.gray400)) }
        Box(
            Modifier.clickable(role = Role.Button, onClickLabel = "共有から外す", onClick = onRemove).padding(6.dp),
        ) { FaIcon(R.drawable.fa_solid_times, "共有から外す", size = 12.dp, tint = Tw.red400) }
    }
}

@Composable
private fun ProjectThreadRow(
    thread: ThreadItem,
    offline: Boolean,
    onOpen: () -> Unit,
    onBookmark: () -> Unit,
    onMove: () -> Unit,
    onRename: () -> Unit,
    onDelete: () -> Unit,
) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(10.dp)
    Row(
        Modifier
            .fillMaxWidth()
            .clip(shape)
            .background(web.twBg(Tw.gray900).copy(alpha = 0.45f))
            .border(1.dp, web.line, shape)
            .clickable(onClick = onOpen)
            .padding(horizontal = 10.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Box(Modifier.clickable(enabled = !offline, role = Role.Button, onClick = onBookmark).padding(4.dp)) {
            FaIcon(R.drawable.fa_solid_star, "ブックマーク", size = 10.dp, tint = if (thread.isBookmarked) Tw.yellow400 else web.twText(Tw.gray500))
        }
        Text(
            thread.title.ifBlank { "No Title" },
            color = web.text, fontSize = 13.sp, maxLines = 1, overflow = TextOverflow.Ellipsis,
            modifier = Modifier.weight(1f),
        )
        Box(Modifier.clickable(enabled = !offline, role = Role.Button, onClick = onMove).padding(4.dp)) {
            FaIcon(R.drawable.fa_solid_folder, "プロジェクトへ移動", size = 11.dp, tint = web.twText(Tw.gray400))
        }
        Box(Modifier.clickable(enabled = !offline, role = Role.Button, onClick = onRename).padding(4.dp)) {
            FaIcon(R.drawable.fa_solid_pen, "名前を変更", size = 11.dp, tint = web.twText(Tw.gray400))
        }
        Box(Modifier.clickable(enabled = !offline, role = Role.Button, onClick = onDelete).padding(4.dp)) {
            FaIcon(R.drawable.fa_solid_trash, "削除", size = 11.dp, tint = web.twText(Tw.gray400))
        }
    }
}

/** Compact banner shown above the chat conversation when a project chat is active (Web `#project-chat-banner`). */
@Composable
internal fun ProjectChatBanner(
    projectName: String,
    onOpenProject: () -> Unit,
    onExitProject: () -> Unit,
) {
    val web = LocalWebPalette.current
    Row(
        Modifier
            .fillMaxWidth()
            .background(web.theme.rgb(0.12f))
            .border(width = 0.5.dp, color = web.theme.rgb(0.3f))
            .padding(horizontal = 12.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Row(
            Modifier.weight(1f).clickable(role = Role.Button, onClick = onOpenProject),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            FaIcon(R.drawable.fa_solid_folder, null, size = 12.dp, tint = web.theme300)
            Text(
                projectName.ifBlank { "プロジェクト" },
                color = web.theme300, fontSize = 12.sp, fontWeight = FontWeight.SemiBold,
                maxLines = 1, overflow = TextOverflow.Ellipsis,
            )
            Text("を開く", color = web.twText(Tw.gray400), fontSize = 11.sp)
        }
        Box(
            Modifier.clickable(role = Role.Button, onClickLabel = "通常チャットへ戻る", onClick = onExitProject).padding(4.dp),
        ) { FaIcon(R.drawable.fa_solid_times, "閉じる", size = 11.dp, tint = web.twText(Tw.gray400)) }
    }
}
