package com.minashin1120.aiplayground.ui

import android.content.ClipDescription
import android.content.Context
import android.text.Html
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp

private fun clipboardRichText(context: Context): String {
    val manager = context.getSystemService(Context.CLIPBOARD_SERVICE) as? android.content.ClipboardManager ?: return ""
    val clip = manager.primaryClip ?: return ""
    if (clip.description.hasMimeType(ClipDescription.MIMETYPE_TEXT_HTML)) {
        clip.getItemAt(0).htmlText?.let { html ->
            return Html.fromHtml(html, Html.FROM_HTML_MODE_COMPACT).toString()
        }
    }
    return clip.getItemAt(0).coerceToText(context).toString()
}

/** Imports clipboard HTML/text into the native composer while keeping the Web
 * action's terminology and preview/edit flow. */
@Composable
fun RichPasteDialog(initialDraft: String, onDismiss: () -> Unit, onInsert: (String) -> Unit) {
    val context = LocalContext.current
    var value by remember { mutableStateOf(clipboardRichText(context).ifBlank { initialDraft }) }
    PlaygroundDialog(
        onDismissRequest = onDismiss,
        title = { Text("リッチ貼り付け") },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("クリップボードの書式を読み取り、入力欄へ貼り付けます。画像や複雑なレイアウトはテキストとして取り込みます。",
                    style = MaterialTheme.typography.bodySmall)
                OutlinedTextField(value, { value = it }, minLines = 8, maxLines = 18,
                    label = { Text("取り込み内容") }, modifier = Modifier.fillMaxWidth())
                TextButton(onClick = { value = clipboardRichText(context) }) { Text("クリップボードを再読み込み") }
                if (value.isBlank()) Text("クリップボードにテキストがありません。", style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        },
        confirmButton = {
            TextButton(onClick = { onInsert(value.trim()); onDismiss() }, enabled = value.isNotBlank()) { Text("入力欄へ追加") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("キャンセル") } },
    )
}
