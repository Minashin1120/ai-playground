package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.minashin1120.aiplayground.AppChangelogUiState

@Composable
fun AppChangelogDialog(
    state: AppChangelogUiState,
    onDismiss: () -> Unit,
    onRetry: () -> Unit,
) {
    PlaygroundDialog(
        onDismissRequest = onDismiss,
        title = { Text("Android版の更新履歴") },
        text = {
            val content = state.content
            when {
                content != null -> Column(
                    Modifier.fillMaxSize().verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    state.errorMessage?.let { message ->
                        Text(message, color = MaterialTheme.colorScheme.error)
                        TextButton(onClick = onRetry) { Text("再試行") }
                    }
                    MarkdownText(content)
                }
                state.loading -> Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    CircularProgressIndicator()
                }
                else -> Column(
                    Modifier.fillMaxWidth().padding(vertical = 24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    Text(state.errorMessage ?: "更新履歴を取得できませんでした。")
                    TextButton(onClick = onRetry) { Text("再試行") }
                }
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text("閉じる") } },
    )
}
