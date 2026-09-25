package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.minashin1120.aiplayground.ChatViewModel
import com.minashin1120.aiplayground.data.ChatMessage
import com.minashin1120.aiplayground.data.CodingTarget

private data class CodeBlock(val id: String, val language: String, val code: String, val messageId: String)

private fun codeBlocks(messages: List<ChatMessage>): List<CodeBlock> {
    val regex = Regex("```([^\\n`]*)\\n([\\s\\S]*?)```")
    return messages.flatMap { message ->
        regex.findAll(message.content).mapIndexed { index, match ->
            CodeBlock("history-${message.id.filter { it.isLetterOrDigit() }}-$index",
                match.groupValues[1].trim().ifBlank { "text" }.take(40), match.groupValues[2], message.id)
        }.toList()
    }.filter { it.code.isNotBlank() }.takeLast(30)
}

@Composable
fun CanvasPreview(messages: List<ChatMessage>, onUse: (String) -> Unit, onClose: () -> Unit) {
    val blocks = remember(messages) { codeBlocks(messages) }
    var selected by remember(blocks) { mutableStateOf(blocks.lastOrNull()) }
    var source by remember(selected?.id) { mutableStateOf(selected?.code.orEmpty()) }
    Surface(color = MaterialTheme.colorScheme.surfaceContainerHigh, shape = MaterialTheme.shapes.medium,
        tonalElevation = 2.dp, modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 8.dp)) {
        Column(Modifier.fillMaxWidth().padding(10.dp), verticalArrangement = Arrangement.spacedBy(7.dp)) {
            Row(verticalAlignment = androidx.compose.ui.Alignment.CenterVertically) {
                Text("Canvas", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
                TextButton(onClick = onClose) { Text("閉じる") }
            }
            if (blocks.isNotEmpty()) Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(5.dp)) {
                blocks.forEach { block ->
                    FilterChip(selected?.id == block.id, { selected = block }, { Text(block.language) })
                }
            }
            if (selected == null) Text("表示できるコードブロックがありません。", style = MaterialTheme.typography.bodySmall)
            else {
                OutlinedTextField(source, { source = it }, minLines = 5, maxLines = 14,
                    label = { Text("ソース（編集可能）") }, textStyle = LocalTextStyle.current.copy(fontFamily = FontFamily.Monospace),
                    modifier = Modifier.fillMaxWidth().verticalScroll(rememberScrollState()))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(onClick = { onUse("```${selected!!.language}\n${source}\n```") }) { Text("入力へ反映") }
                    TextButton(onClick = { source = selected!!.code }) { Text("元に戻す") }
                }
            }
        }
    }
}
