package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.ChatViewModel
import com.minashin1120.aiplayground.data.BatchJob
import com.minashin1120.aiplayground.data.batchStateLabel

@Composable
fun AdvancedToolsDialog(
    state: ChatState,
    model: ChatViewModel,
    onDismiss: () -> Unit,
    onWebPath: (String) -> Unit,
    onRealtime: () -> Unit = {},
    onLyria: () -> Unit = {},
) {
    PlaygroundDialog(
        onDismissRequest = onDismiss,
        title = { Text("高度な機能") },
        text = {
            LazyColumn(Modifier.heightIn(max = 500.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                item {
                    Text("ネイティブ生成", fontWeight = FontWeight.SemiBold)
                    Text("画像・動画・OCR・TTS・文字起こしは、入力欄のモデル選択から利用できます。生成物は履歴とファイルライブラリに保存されます。",
                        style = MaterialTheme.typography.bodySmall)
                }
                item {
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text("Batch", fontWeight = FontWeight.SemiBold)
                        TextButton(onClick = model::refreshBatchJobs, enabled = !state.batchBusy) { Text("更新") }
                    }
                    if (state.batchBusy) LinearProgressIndicator(Modifier.fillMaxWidth())
                    if (state.batchJobs.isEmpty() && !state.batchBusy) {
                        Text("Batch履歴はありません。Batch対応モデルを選び、入力欄の「Batch」を有効にして送信できます。",
                            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
                items(state.batchJobs, key = { it.id }) { job -> BatchJobRow(job, model) }
                item {
                    HorizontalDivider()
                    Text("音声・管理機能", fontWeight = FontWeight.SemiBold, modifier = Modifier.padding(top = 8.dp))
                    Text("常時接続の音声・音楽はネイティブセッションで利用できます。APIキーやセキュリティ設定は、認証済みブラウザーで開きます。端末へ秘密情報は渡しません。",
                        style = MaterialTheme.typography.bodySmall)
                    Column {
                        TextButton(onClick = onRealtime, modifier = Modifier.fillMaxWidth()) { Text("Realtime音声（OpenAI／Grok／Gemini Native Audio）") }
                        TextButton(onClick = onLyria, modifier = Modifier.fillMaxWidth()) { Text("Lyria音楽（ネイティブ）") }
                        TextButton(onClick = onRealtime, modifier = Modifier.fillMaxWidth()) { Text("Gemini Live / STS（ネイティブ）") }
                        TextButton(onClick = { onWebPath("/") }, modifier = Modifier.fillMaxWidth()) { Text("Web版のCanvasを開く") }
                        TextButton(onClick = { onWebPath("/settings") }, modifier = Modifier.fillMaxWidth()) { Text("アカウント・APIキー・2FAを開く") }
                    }
                }
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text("閉じる") } },
    )
}

@Composable
private fun BatchJobRow(job: BatchJob, model: ChatViewModel) {
    Surface(color = MaterialTheme.colorScheme.surfaceContainerHigh, shape = MaterialTheme.shapes.medium) {
        Column(Modifier.fillMaxWidth().padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(job.threadTitle, fontWeight = FontWeight.SemiBold)
            Text("${job.model} · ${batchStateLabel(job)}", style = MaterialTheme.typography.labelSmall)
            if (job.error.isNotBlank()) Text(job.error.take(300), color = MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.bodySmall)
            Row {
                if (job.threadId.isNotBlank()) TextButton(onClick = { model.openThreadId(job.threadId) }) { Text("チャットを開く") }
                if (job.canCancel) TextButton(onClick = { model.cancelBatchJob(job) }) { Text("停止") }
                if (!job.active) TextButton(onClick = { model.deleteBatchJob(job) }) { Text("履歴を削除") }
            }
        }
    }
}
