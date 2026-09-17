package com.minashin1120.aiplayground.ui

import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.ui.unit.dp
import com.minashin1120.aiplayground.data.AppUpdate

@Composable
fun AppUpdateDialog(update: AppUpdate, onDismiss: () -> Unit, onOpenRelease: () -> Unit) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("新しいAndroid版があります") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("AI Playground ${update.versionName} が利用できます。")
                Text("GitHub Releaseを開いて、APKをダウンロードして更新してください。")
            }
        },
        confirmButton = { TextButton(onClick = onOpenRelease) { Text("更新を確認") } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("後で") } },
    )
}
