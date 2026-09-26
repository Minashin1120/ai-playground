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
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.ImportSettingChange
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.FileSelectionRequest

/** Web `ACCOUNT_SETTING_LABELS` for the settings import confirmation. */
internal val ACCOUNT_SETTING_LABELS = mapOf(
    "system_prompt" to "システムプロンプト", "system_prompt_enabled" to "システムプロンプトを使用",
    "apply_global_system_prompt" to "全体システムプロンプトを適用", "apply_auto_system_prompt_notices" to "自動注入システムプロンプトを適用",
    "auto_system_prompt_notices_config" to "自動注入システムプロンプトの種類別設定",
    "gemini_backend" to "Gemini バックエンド", "gemini_vertex_location" to "Vertex AI ロケーション",
    "mic_transcribe_mode" to "マイク文字起こし方式", "stt_model" to "STTモデル", "llm_transcribe_prompt" to "LLM文字起こしプロンプト",
    "enter_to_send" to "Enterキーで送信", "use_sw_cache" to "Service Workerキャッシュ",
    "clear_cache_on_version_update" to "バージョン更新時キャッシュ削除", "theme_color" to "テーマカラー",
    "liquid_glass_enabled" to "Liquid Glass", "light_mode_enabled" to "ライトモード", "auto_search_on_links" to "リンクで自動検索",
    "compact_prompt_mode" to "プロンプトバー表示（コンパクト）", "minimal_prompt_mode" to "プロンプトバー表示（ミニマル）",
    "use_last_chat_settings" to "直前のチャット設定を使用", "voice_studio_ui" to "音声スタジオUI",
    "temp_chat_timeout_seconds" to "一時チャットの有効時間（秒）", "default_model" to "既定のモデル",
    "default_enable_search" to "既定: Search", "default_enable_url_context" to "既定: URLコンテキスト",
    "default_enable_maps" to "既定: Maps", "default_enable_python" to "既定: Python",
    "default_enable_file_creation" to "既定: File",
    "default_enable_thinking" to "既定: Thinking", "default_thinking_level" to "既定: Thinkingレベル",
    "default_thinking_budget" to "既定: Thinking budget", "default_reasoning_effort" to "既定: Reasoning effort",
    "default_enable_system_prompt" to "既定: システムプロンプト", "default_enable_mcp" to "既定: MCP",
    "default_safety_setting" to "既定: 安全設定",
    "default_vision_model" to "Vision Model", "rich_paste_prompt_default" to "リッチ貼り付けプロンプト",
    "rich_paste_prompt_use_custom_default" to "リッチ貼り付けカスタムプロンプト既定",
    "last_model" to "直前のモデル", "last_enable_search" to "直前: Search", "last_enable_url_context" to "直前: URLコンテキスト",
    "last_enable_maps" to "直前: Maps", "last_enable_python" to "直前: Python", "last_enable_file_creation" to "直前: File",
    "last_enable_thinking" to "直前: Thinking",
    "last_thinking_level" to "直前: Thinkingレベル", "last_thinking_budget" to "直前: Thinking budget",
    "last_reasoning_effort" to "直前: Reasoning effort", "last_enable_system_prompt" to "直前: システムプロンプト", "last_enable_mcp" to "直前: MCP",
    "last_safety_setting" to "直前: 安全設定", "enable_latency_metrics" to "レスポンス速度の計測",
    "enable_client_debug_log" to "デバッグログの拡張送信",
)

/** Web `importFormatBytes`. */
internal fun importFormatBytes(bytes: Long): String = when {
    bytes >= 1024L * 1024 * 1024 -> String.format("%.2f GB", bytes / (1024.0 * 1024 * 1024))
    bytes >= 1024L * 1024 -> String.format("%.1f MB", bytes / (1024.0 * 1024))
    bytes >= 1024 -> "${Math.round(bytes / 1024.0)} KB"
    else -> "$bytes B"
}

/** The gray `bg-gray-800` header/footer modal used by both import dialogs. */
@Composable
private fun ImportModal(
    title: String,
    icon: Int,
    iconTint: Color,
    maxWidth: Dp,
    onClose: () -> Unit,
    footerNote: @Composable RowScope.() -> Unit,
    footerButtons: @Composable RowScope.() -> Unit,
    toolbar: (@Composable () -> Unit)? = null,
    content: @Composable ColumnScope.() -> Unit,
) {
    val web = LocalWebPalette.current
    val line = web.twBorder(Tw.gray700)
    WebOverlayModal(onClose, grayOverlay(), 4.dp) { _ ->
        Column(Modifier.widthIn(max = maxWidth).fillMaxWidth().fillMaxHeight(if (maxWidth > 600.dp) 1f else 0.85f)) {
            Row(
                Modifier.fillMaxWidth().background(web.twBg(Tw.gray800))
                    .drawBehind { drawLine(line, Offset(0f, size.height), Offset(size.width, size.height), 1.dp.toPx()) }.padding(16.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                FaIcon(icon, null, size = 16.dp, tint = iconTint)
                Text(title, fontSize = 18.sp, fontWeight = FontWeight.Bold, color = if (web.isLight) web.text else Color.White,
                    modifier = Modifier.padding(start = 8.dp).weight(1f))
                Text("✕", color = Tw.gray400, modifier = Modifier.clickable(role = Role.Button, onClick = onClose).padding(8.dp))
            }
            toolbar?.invoke()
            Column(Modifier.weight(1f).fillMaxWidth().background(grayOverlay()).verticalScroll(rememberScrollState()).padding(16.dp), content = content)
            Row(
                Modifier.fillMaxWidth().background(web.twBg(Tw.gray800))
                    .drawBehind { drawLine(line, Offset(0f, 0f), Offset(size.width, 0f), 1.dp.toPx()) }.padding(16.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                footerNote()
                footerButtons()
            }
        }
    }
}

@Composable
internal fun ImportSettingsConfirmDialog(changes: List<ImportSettingChange>, onResult: (Boolean) -> Unit) {
    val web = LocalWebPalette.current
    ImportModal(
        title = "設定のインポート確認", icon = R.drawable.fa_solid_sliders_h, iconTint = web.theme300, maxWidth = 512.dp,
        onClose = { onResult(false) },
        footerNote = { Text("設定以外の選択データはこの確認後に続けてインポートされます。", fontSize = 11.sp, color = Tw.gray400, modifier = Modifier.weight(1f)) },
        footerButtons = {
            SettingsSmallButton("キャンセル", { onResult(false) })
            SettingsSmallButton("この内容でインポート", { onResult(true) }, tone = SettingsButtonTone.Emerald)
        },
    ) {
        Text("${changes.size}件の設定が変更されます", fontSize = 14.sp, fontWeight = FontWeight.Bold, color = if (web.isLight) web.text else Tw.gray100,
            modifier = Modifier.padding(bottom = 8.dp))
        Text("ZIPに含まれる「設定」で、以下の項目が現在の設定から変更されます。よろしければ「この内容でインポート」を押してください。",
            fontSize = 12.sp, color = Tw.gray400, modifier = Modifier.padding(bottom = 12.dp))
        if (changes.isEmpty()) Text("変更される設定はありませんでした。", fontSize = 12.sp, color = Tw.gray400)
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            changes.forEach { change ->
                val shape = RoundedCornerShape(4.dp)
                Column(Modifier.fillMaxWidth().clip(shape).background(web.twBg(Tw.gray800).copy(alpha = 0.6f)).border(1.dp, web.twBorder(Tw.gray700), shape).padding(8.dp)) {
                    Text(ACCOUNT_SETTING_LABELS[change.field] ?: change.field, fontSize = 12.sp, fontWeight = FontWeight.Bold,
                        color = if (web.isLight) web.text else Tw.gray100)
                    Text("現在: ${change.current}", fontSize = 11.sp, color = Tw.gray400, modifier = Modifier.padding(top = 4.dp))
                    Text("→ ${change.incoming}", fontSize = 11.sp, color = Tw.emerald300)
                }
            }
        }
    }
}

/** `#import-files-modal`: choose which files fit the remaining storage. Returns the comma list, `__none__`, or null. */
@Composable
internal fun ImportFileSelectionDialog(request: FileSelectionRequest, onResult: (String?) -> Unit) {
    val web = LocalWebPalette.current
    val selection = remember(request) { mutableStateListOf<String>().apply { addAll(request.files.map { it.archivePath }) } }
    val total = request.files.filter { it.archivePath in selection }.sumOf { it.sizeBytes }
    val over = total > request.availableBytes
    ImportModal(
        title = "インポートするファイルを選択", icon = R.drawable.fa_solid_file_import, iconTint = Tw.emerald400, maxWidth = 4000.dp,
        onClose = { onResult(null) },
        toolbar = {
            Row(Modifier.fillMaxWidth().background(web.twBg(Tw.gray800).copy(alpha = 0.5f)).padding(8.dp),
                verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("${request.files.size} files", fontSize = 12.sp, color = Tw.gray400, modifier = Modifier.padding(horizontal = 8.dp).weight(1f))
                SettingsSmallButton("すべて選択", { selection.clear(); selection.addAll(request.files.map { it.archivePath }) })
                SettingsSmallButton("すべて解除", { selection.clear() })
            }
        },
        footerNote = {
            Text("選択中: ${importFormatBytes(total)} / 利用可能: ${importFormatBytes(request.availableBytes)}${if (over) " （容量超過）" else ""}",
                fontSize = 11.sp, color = if (over) Tw.red300 else Tw.gray400, modifier = Modifier.weight(1f))
        },
        footerButtons = {
            SettingsSmallButton("キャンセル", { onResult(null) })
            SettingsSmallButton("選択したファイルをインポート", {
                onResult(if (selection.isEmpty()) "__none__" else request.files.map { it.archivePath }.filter { it in selection }.joinToString(","))
            }, tone = SettingsButtonTone.Emerald)
        },
    ) {
        if (request.files.isEmpty()) Text("インポート可能なファイルがありません。", fontSize = 12.sp, color = Tw.gray500)
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            request.files.forEach { file ->
                val checked = file.archivePath in selection
                val shape = RoundedCornerShape(4.dp)
                Row(
                    Modifier.fillMaxWidth().clip(shape).background(web.twBg(Tw.gray800))
                        .border(1.dp, if (checked) Tw.blue500 else web.twBorder(Tw.gray600), shape)
                        .clickable(role = Role.Checkbox) { if (checked) selection.remove(file.archivePath) else selection.add(file.archivePath) }
                        .padding(8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    WebCheckbox(checked, onCheckedChange = null, size = 16.dp)
                    Column(Modifier.weight(1f)) {
                        Text(file.displayName, fontSize = 12.sp, color = if (web.isLight) web.text else Tw.gray200, maxLines = 1, overflow = TextOverflow.Ellipsis)
                        Text(importFormatBytes(file.sizeBytes), fontSize = 10.sp, color = Tw.gray500)
                    }
                }
            }
        }
    }
}
