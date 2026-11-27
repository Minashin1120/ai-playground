package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.ProjectFileItem
import com.minashin1120.aiplayground.data.ProjectItem
import com.minashin1120.aiplayground.data.ThreadItem

internal fun projectVisionModelOptions(state: ChatState): List<WebOption> {
    val defaultModel = state.preferences?.defaultVisionModel?.takeIf { it.isNotBlank() } ?: "gemini-3.6-flash"
    val out = mutableListOf(WebOption("", "既定のVisionモデル ($defaultModel)"))
    val seen = HashSet<String>()
    state.account?.models.orEmpty()
        .filter { it.mode == "chat" && it.selectable }
        .forEach { m ->
            if (seen.add(m.id)) out += WebOption(m.id, m.name.ifBlank { m.id })
        }
    return out
}

/**
 * `#project-modal`: Create / Edit Project (name, description, project system prompt,
 * per-project Vision model toggle & model picker for shared image files).
 */
@Composable
internal fun ProjectEditorDialog(
    project: ProjectItem?,
    state: ChatState,
    onDismiss: () -> Unit,
    onSave: (
        name: String,
        description: String,
        instruction: String,
        useVisionForImages: Boolean,
        visionModel: String?,
        done: (Boolean) -> Unit,
    ) -> Unit,
) {
    val web = LocalWebPalette.current
    var name by remember(project?.uuid) { mutableStateOf(project?.name.orEmpty()) }
    var description by remember(project?.uuid) { mutableStateOf(project?.description.orEmpty()) }
    var instruction by remember(project?.uuid) { mutableStateOf(project?.instruction.orEmpty()) }
    var useVision by remember(project?.uuid) { mutableStateOf(project?.useVisionForImages ?: false) }
    var visionModel by remember(project?.uuid) { mutableStateOf(project?.visionModel.orEmpty()) }
    var saving by remember { mutableStateOf(false) }
    var missingName by remember { mutableStateOf(false) }
    val visionOptions = remember(state.account?.models, state.preferences?.defaultVisionModel) {
        projectVisionModelOptions(state)
    }

    PlaygroundDialog(
        onDismissRequest = onDismiss,
        title = { Text(if (project == null) "新しいプロジェクト" else "プロジェクトを編集") },
        text = {
            Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(14.dp)) {
                ProjectField("プロジェクト名 *") {
                    TwInput(name, { name = it.take(120) }, "例: 新規アプリ開発")
                }
                ProjectField("説明（任意）") {
                    TwInput(description, { description = it.take(1000) }, "プロジェクトの簡単な説明（任意）")
                }
                ProjectField("プロジェクトのシステムプロンプト") {
                    TwInput(
                        instruction,
                        { instruction = it.take(100_000) },
                        "このプロジェクト内のすべてのチャットに共通で適用する指示を入力してください...",
                        singleLine = false,
                        height = 148.dp,
                    )
                    Text(
                        "全体/ユーザーシステムプロンプトの後、チャット専用プロンプトの前に適用されます。",
                        fontSize = 10.sp, lineHeight = 14.sp, color = web.twText(Tw.gray500),
                        modifier = Modifier.padding(top = 4.dp),
                    )
                }
                val shape = RoundedCornerShape(8.dp)
                Column(
                    Modifier
                        .fillMaxWidth()
                        .clip(shape)
                        .background(web.twBg(Tw.gray900).copy(alpha = 0.6f))
                        .border(1.dp, web.line, shape)
                        .padding(12.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f).padding(end = 8.dp)) {
                            Text("共有画像ファイルにVisionモデルを使う", fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = web.text)
                            Text(
                                "OFF時は画像の中身を読まずメタ情報（ファイル名・サイズ）のみ渡します。ON時は指定したVisionモデルで画像を解析します。",
                                fontSize = 10.sp, lineHeight = 14.sp, color = web.twText(Tw.gray400),
                                modifier = Modifier.padding(top = 2.dp),
                            )
                        }
                        WebToggle(useVision, { useVision = it }, activeColor = Tw.emerald500)
                    }
                    if (useVision) {
                        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                            Text("Visionモデル", fontSize = 11.sp, color = web.twText(Tw.gray400))
                            WebSelect(
                                value = visionModel,
                                options = visionOptions,
                                onSelect = { visionModel = it },
                                fillWidth = true,
                                contentPadding = PaddingValues(horizontal = 8.dp, vertical = 6.dp),
                            )
                        }
                    }
                }
            }
        },
        confirmButton = {
            WebButton(
                onClick = {
                    if (name.trim().isEmpty()) {
                        missingName = true
                    } else {
                        saving = true
                        onSave(name.trim(), description.trim(), instruction, useVision, visionModel.ifBlank { null }) {
                            saving = false
                        }
                    }
                },
                variant = WebButtonVariant.Primary,
                enabled = !saving,
            ) { Text(if (saving) "保存中..." else if (project == null) "作成" else "保存") }
        },
        dismissButton = { WebButton(onClick = onDismiss, variant = WebButtonVariant.Ghost) { Text("キャンセル") } },
        icon = R.drawable.fa_solid_folder,
        subtitle = "チャットと共有ファイルをまとめる",
        tone = WebModalTone.Info,
        panelMaxWidth = 760.dp,
    )
    if (missingName) BrowserAlertDialog("プロジェクト名を入力してください。") { missingName = false }
}

/** `#project-move-thread-modal`: assign or unassign a chat to/from a project. */
@Composable
internal fun MoveThreadProjectDialog(
    thread: ThreadItem,
    projects: List<ProjectItem>,
    onDismiss: () -> Unit,
    onMove: (projectUuid: String?) -> Unit,
) {
    val web = LocalWebPalette.current
    var selectedUuid by remember(thread.id) { mutableStateOf(thread.projectId.orEmpty()) }
    PlaygroundDialog(
        onDismissRequest = onDismiss,
        title = { Text("プロジェクトへ移動") },
        text = {
            Column(Modifier.fillMaxWidth().verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(
                    thread.title.ifBlank { "No Title" },
                    fontSize = 12.sp, color = web.twText(Tw.gray400), maxLines = 1, overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.padding(bottom = 4.dp),
                )
                MoveProjectOptionRow(
                    label = "未所属（通常のチャット一覧）",
                    subtitle = "プロジェクトから外す",
                    selected = selectedUuid.isEmpty(),
                    onClick = { selectedUuid = "" },
                )
                projects.forEach { proj ->
                    MoveProjectOptionRow(
                        label = proj.name,
                        subtitle = proj.description.ifBlank { "チャット ${proj.threadCount}件 · ファイル ${proj.fileCount}件" },
                        selected = selectedUuid == proj.uuid,
                        onClick = { selectedUuid = proj.uuid },
                    )
                }
            }
        },
        confirmButton = {
            WebButton(
                onClick = { onMove(selectedUuid.ifBlank { null }) },
                variant = WebButtonVariant.Primary,
            ) { Text("移動") }
        },
        dismissButton = { WebButton(onClick = onDismiss, variant = WebButtonVariant.Ghost) { Text("キャンセル") } },
        icon = R.drawable.fa_solid_folder,
        subtitle = "所属先プロジェクトを選択",
        tone = WebModalTone.Info,
        panelMaxWidth = 520.dp,
    )
}

@Composable
private fun MoveProjectOptionRow(label: String, subtitle: String, selected: Boolean, onClick: () -> Unit) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(10.dp)
    Row(
        Modifier
            .fillMaxWidth()
            .clip(shape)
            .background(if (selected) web.theme.rgb(0.16f) else web.twBg(Tw.gray900).copy(alpha = 0.5f))
            .border(1.dp, if (selected) web.theme.t500 else web.line, shape)
            .clickable(role = Role.RadioButton, onClick = onClick)
            .padding(horizontal = 12.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        FaIcon(R.drawable.fa_solid_folder, null, size = 14.dp, tint = if (selected) web.theme300 else web.twText(Tw.gray400))
        Column(Modifier.weight(1f)) {
            Text(label, fontSize = 13.sp, fontWeight = FontWeight.SemiBold, color = web.text, maxLines = 1, overflow = TextOverflow.Ellipsis)
            if (subtitle.isNotBlank()) {
                Text(subtitle, fontSize = 11.sp, color = web.twText(Tw.gray400), maxLines = 1, overflow = TextOverflow.Ellipsis)
            }
        }
    }
}

/** Edit optional note/memo for a project shared file. */
@Composable
internal fun ProjectFileNoteDialog(
    file: ProjectFileItem,
    onDismiss: () -> Unit,
    onSave: (note: String) -> Unit,
) {
    var note by remember(file.id) { mutableStateOf(file.note) }
    PlaygroundDialog(
        onDismissRequest = onDismiss,
        title = { Text("ファイルのメモを編集") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(file.filename, fontSize = 12.sp, color = LocalWebPalette.current.twText(Tw.gray400), maxLines = 1, overflow = TextOverflow.Ellipsis)
                TwInput(note, { note = it.take(200) }, "このファイルの用途や説明（モデルへの一覧に添えられます）", singleLine = false, height = 88.dp)
            }
        },
        confirmButton = {
            WebButton(onClick = { onSave(note.trim()) }, variant = WebButtonVariant.Primary) { Text("保存") }
        },
        dismissButton = { WebButton(onClick = onDismiss, variant = WebButtonVariant.Ghost) { Text("キャンセル") } },
        icon = R.drawable.fa_solid_file_lines,
        subtitle = "モデルに伝える補助メモ（最大200文字）",
        tone = WebModalTone.Info,
        panelMaxWidth = 520.dp,
    )
}

@Composable
private fun ProjectField(label: String, content: @Composable () -> Unit) {
    val web = LocalWebPalette.current
    Column(Modifier.fillMaxWidth()) {
        Text(label, fontSize = 12.sp, lineHeight = 16.sp, color = web.twText(Tw.gray400), modifier = Modifier.padding(bottom = 4.dp))
        content()
    }
}
