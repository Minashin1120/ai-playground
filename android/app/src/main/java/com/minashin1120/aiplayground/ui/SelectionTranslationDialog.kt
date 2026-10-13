package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.SelectionTranslationState

/**
 * ANDROID_ONLY.md 「AIで翻訳」: the translation shown over the app the text was selected in (the activity is
 * translucent, so that app stays visible behind the modal).
 */
@Composable
internal fun SelectionTranslationDialog(
    state: SelectionTranslationState,
    onDismiss: () -> Unit,
    onCopy: () -> Unit,
    onRetry: () -> Unit,
    onOpenApp: () -> Unit,
) {
    val web = LocalWebPalette.current
    val setting = state.setting
    PlaygroundDialog(
        onDismissRequest = onDismiss,
        title = { Text("AIで翻訳") },
        icon = R.drawable.fa_solid_globe,
        subtitle = setting?.takeIf { it.model.isNotBlank() }?.let { "${it.modelName.ifBlank { it.model }} → ${it.language}" },
        panelMaxWidth = 640.dp,
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                var sourceExpanded by rememberSaveable { mutableStateOf(false) }
                SettingsFieldLabel("原文")
                Text(state.source, fontSize = 12.sp, lineHeight = 18.sp, color = web.muted,
                    maxLines = if (sourceExpanded) Int.MAX_VALUE else 3, overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.fillMaxWidth().clickable { sourceExpanded = !sourceExpanded })
                SettingsFieldLabel("翻訳", Modifier.settingsTopRule(web.lineStrong, 10.dp))
                when {
                    state.error != null -> Text(state.error, fontSize = 13.sp, lineHeight = 20.sp, color = Tw.red300)
                    state.result.isNotEmpty() -> SelectionContainer {
                        Text(state.result, fontSize = 15.sp, lineHeight = 24.sp, color = web.text)
                    }
                    else -> Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        CircularProgressIndicator(Modifier.size(14.dp), strokeWidth = 2.dp, color = web.theme.t500)
                        Text("翻訳中...", fontSize = 13.sp, color = web.muted)
                    }
                }
            }
        },
        confirmButton = {
            when {
                state.error != null && state.openApp -> WebButton(onOpenApp, variant = WebButtonVariant.Primary) {
                    Text(if (setting?.model.isNullOrBlank()) "設定を開く" else "アプリを開く")
                }
                state.error != null -> WebButton(onRetry, variant = WebButtonVariant.Primary) {
                    FaIcon(R.drawable.fa_solid_rotate_right, null, size = 12.dp)
                    Text("再試行")
                }
                else -> WebButton(onCopy, variant = WebButtonVariant.Primary, enabled = !state.running && state.result.isNotBlank()) {
                    FaIcon(R.drawable.fa_solid_copy, null, size = 12.dp)
                    Text("コピー", fontWeight = FontWeight.Bold)
                }
            }
        },
        dismissButton = { WebButton(onDismiss) { Text("閉じる") } },
    )
}
