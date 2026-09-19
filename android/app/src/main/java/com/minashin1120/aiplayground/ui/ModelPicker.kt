package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.data.ModelInfo

@Composable
internal fun ModelPicker(state: ChatState, onDismiss: () -> Unit, onSelect: (String) -> Unit, selectedId: String = state.model) {
    var query by remember { mutableStateOf("") }
    var category by remember { mutableStateOf("All") }
    val catalog = state.account?.models.orEmpty()
    val categories = listOf("All") + catalog.map { it.providerLabel }.distinct() +
        catalog.map { it.mode }.distinct() + catalog.flatMap { it.tags + it.capabilities }.distinct().sorted()
    val models = catalog.filter { info ->
        (category == "All" || category == info.providerLabel || category == info.mode || category in info.capabilities || category in info.tags) &&
            listOf(info.name, info.id, info.providerLabel, info.mode, info.description, info.category, (info.tags + info.capabilities).joinToString(" "))
                .any { it.contains(query.trim(), ignoreCase = true) }
    }.sortedWith(compareBy<ModelInfo> { it.webCatalogOrder }
        .thenBy { !it.selectable }
        .thenBy { it.providerLabel }
        .thenBy { it.name })
    val listState = rememberLazyListState()
    val selectedIndex = models.indexOfFirst { it.id == selectedId }
    LaunchedEffect(selectedId, models.map { it.id }) {
        if (selectedIndex >= 0 && listState.layoutInfo.visibleItemsInfo.none { it.index == selectedIndex }) {
            listState.animateScrollToItem(selectedIndex, scrollOffset = -12)
        }
    }
    val colors = MaterialTheme.colorScheme
    Dialog(onDismissRequest = onDismiss, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        Surface(Modifier.padding(12.dp).widthIn(max = 720.dp).fillMaxWidth().fillMaxHeight(0.9f),
            shape = RoundedCornerShape(20.dp), color = colors.surface,
            border = BorderStroke(1.dp, colors.outlineVariant)) {
            Column {
                Row(Modifier.fillMaxWidth().padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Rounded.SmartToy, null, tint = colors.primary)
                    Column(Modifier.weight(1f).padding(start = 12.dp)) {
                        Text("Select Model", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                        Text("使用するモデルを選択してください", style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
                    }
                    IconButton(onClick = onDismiss) { Icon(Icons.Rounded.Close, "閉じる") }
                }
                HorizontalDivider()
                OutlinedTextField(query, { query = it }, singleLine = true,
                    placeholder = { Text("モデル名・対応機能で検索") },
                    leadingIcon = { Icon(Icons.Rounded.Search, null) },
                    trailingIcon = { if (query.isNotEmpty()) IconButton(onClick = { query = "" }) { Icon(Icons.Rounded.Close, "検索をクリア") } },
                    shape = RoundedCornerShape(12.dp), modifier = Modifier.fillMaxWidth().padding(12.dp))
                Row(Modifier.horizontalScroll(rememberScrollState()).padding(horizontal = 12.dp), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    categories.distinct().forEach { tag ->
                        FilterChip(category == tag, { category = tag }, { Text(tag) })
                    }
                }
                Text("${models.size}件", style = MaterialTheme.typography.labelSmall, color = colors.onSurfaceVariant,
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp))
                LazyColumn(state = listState, modifier = Modifier.weight(1f), contentPadding = PaddingValues(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    if (models.isEmpty()) item {
                        Text("一致するモデルがありません。検索語または分類を変更してください。", modifier = Modifier.padding(16.dp))
                    }
                    items(models, key = { it.id }) { info ->
                        Surface(onClick = { onSelect(info.id) }, enabled = info.selectable,
                            shape = RoundedCornerShape(12.dp),
                            color = if (info.id == selectedId) colors.primaryContainer else colors.surfaceContainerLow,
                            border = BorderStroke(1.dp, if (info.id == selectedId) colors.primary else colors.outlineVariant)) {
                            Row(Modifier.fillMaxWidth().padding(14.dp), verticalAlignment = Alignment.CenterVertically) {
                                Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                                    Text(info.name, fontWeight = FontWeight.SemiBold)
                                    Text(info.category.ifBlank { "${info.providerLabel} · ${info.mode}" }, style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
                                    if (info.description.isNotBlank()) Text(info.description, style = MaterialTheme.typography.bodySmall)
                                    if (info.price.isNotBlank()) Text(info.price, style = MaterialTheme.typography.labelSmall, color = colors.onSurfaceVariant)
                                    if (info.capabilities.isNotEmpty()) Text(info.capabilities.joinToString(" · "), style = MaterialTheme.typography.labelSmall, color = colors.primary)
                                    if (!info.selectable) Text(if (info.deprecated) "提供終了" else "アプリ未対応・Web版で利用可能", style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
                                }
                                if (info.id == selectedId) Icon(Icons.Rounded.CheckCircle, "選択中", tint = colors.primary)
                            }
                        }
                    }
                }
            }
        }
    }
}
