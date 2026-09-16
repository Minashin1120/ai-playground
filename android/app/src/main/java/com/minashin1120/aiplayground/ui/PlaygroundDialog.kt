package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties

/** Web modal proportions with a fixed header/footer and an IME-aware native body. */
@Composable
internal fun PlaygroundDialog(
    onDismissRequest: () -> Unit,
    title: @Composable () -> Unit,
    text: @Composable () -> Unit,
    confirmButton: @Composable () -> Unit,
    dismissButton: @Composable (() -> Unit)? = null,
) {
    Dialog(onDismissRequest, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        Surface(Modifier.imePadding().padding(12.dp).widthIn(max = 720.dp).fillMaxWidth().heightIn(max = 760.dp),
            shape = MaterialTheme.shapes.large, border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant)) {
            Column {
                Row(Modifier.fillMaxWidth().padding(start = 20.dp, end = 8.dp, top = 8.dp, bottom = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                    Box(Modifier.weight(1f)) { ProvideTextStyle(MaterialTheme.typography.titleLarge, title) }
                    IconButton(onClick = onDismissRequest) { Icon(Icons.Rounded.Close, "閉じる") }
                }
                HorizontalDivider()
                Box(Modifier.weight(1f, fill = false).fillMaxWidth().padding(16.dp)) { text() }
                HorizontalDivider()
                Row(Modifier.fillMaxWidth().padding(8.dp), horizontalArrangement = Arrangement.End, verticalAlignment = Alignment.CenterVertically) {
                    dismissButton?.invoke()
                    confirmButton()
                }
            }
        }
    }
}
