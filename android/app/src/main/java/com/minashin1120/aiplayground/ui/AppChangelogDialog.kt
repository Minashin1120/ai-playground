package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.Search
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.minashin1120.aiplayground.AppChangelogUiState

internal data class AppChangelogSection(
    val version: String?,
    val markdown: String,
)

private val APP_CHANGELOG_HEADING = Regex("^#\\s+Android版更新履歴\\s+-\\s+(.+?)\\s*$")

internal fun parseAppChangelogSections(markdown: String): List<AppChangelogSection> {
    val sections = mutableListOf<AppChangelogSection>()
    var version: String? = null
    val body = mutableListOf<String>()

    fun appendSection() {
        val content = body.joinToString("\n").trim()
        if (version != null && content.isNotEmpty()) sections += AppChangelogSection(version, content)
        body.clear()
    }

    markdown.replace("\r\n", "\n").lines().forEach { line ->
        val heading = APP_CHANGELOG_HEADING.matchEntire(line)
        if (heading != null) {
            appendSection()
            version = heading.groupValues[1]
        } else if (version != null || line.isNotBlank()) {
            // The aggregate title is intentionally omitted; each version gets its own card.
            body += line
        }
    }
    appendSection()

    return sections.ifEmpty {
        listOf(AppChangelogSection(version = null, markdown = markdown.trim()))
    }
}

internal fun filterAppChangelogSections(
    sections: List<AppChangelogSection>,
    query: String,
): List<AppChangelogSection> {
    val normalizedQuery = query.trim()
    if (normalizedQuery.isBlank()) return sections

    return sections.filter { section ->
        val version = section.version.orEmpty()
        (version.isNotBlank() && (
            version.contains(normalizedQuery, ignoreCase = true) ||
                "v$version".contains(normalizedQuery, ignoreCase = true)
            )) ||
            section.markdown.contains(normalizedQuery, ignoreCase = true)
    }
}

@Composable
fun AppChangelogDialog(
    state: AppChangelogUiState,
    onDismiss: () -> Unit,
    onRetry: () -> Unit,
) {
    PlaygroundDialog(
        onDismissRequest = onDismiss,
        title = {
            Column(verticalArrangement = Arrangement.spacedBy(1.dp)) {
                Text("Android版の更新履歴", fontWeight = FontWeight.Bold)
                Text(
                    "このアプリの変更点",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        },
        text = {
            val content = state.content
            when {
                content != null -> {
                    val sections = remember(content) { parseAppChangelogSections(content) }
                    var query by remember(content) { mutableStateOf("") }
                    val filteredSections = remember(sections, query) {
                        filterAppChangelogSections(sections, query)
                    }
                    Column(Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedTextField(
                            value = query,
                            onValueChange = { query = it },
                            singleLine = true,
                            placeholder = { Text("更新履歴を検索...") },
                            leadingIcon = { Icon(Icons.Rounded.Search, contentDescription = null) },
                            trailingIcon = {
                                if (query.isNotEmpty()) {
                                    IconButton(onClick = { query = "" }) {
                                        Icon(Icons.Rounded.Close, contentDescription = "検索をクリア")
                                    }
                                }
                            },
                            shape = RoundedCornerShape(12.dp),
                            modifier = Modifier.fillMaxWidth(),
                        )
                        if (query.isNotBlank()) {
                            Text(
                                if (filteredSections.isNotEmpty()) {
                                    "${filteredSections.size} 件の更新履歴が見つかりました"
                                } else {
                                    "該当する更新履歴はありません"
                                },
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                modifier = Modifier.padding(horizontal = 4.dp),
                            )
                        }
                        LazyColumn(
                            modifier = Modifier.weight(1f),
                            contentPadding = PaddingValues(bottom = 8.dp),
                            verticalArrangement = Arrangement.spacedBy(12.dp),
                        ) {
                            item {
                                Surface(
                                    modifier = Modifier.fillMaxWidth(),
                                    shape = RoundedCornerShape(14.dp),
                                    color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.5f),
                                ) {
                                    Column(Modifier.padding(horizontal = 16.dp, vertical = 13.dp), verticalArrangement = Arrangement.spacedBy(3.dp)) {
                                        Text(
                                            "新しい順に表示しています",
                                            style = MaterialTheme.typography.titleSmall,
                                            fontWeight = FontWeight.Bold,
                                            color = MaterialTheme.colorScheme.onPrimaryContainer,
                                        )
                                        Text(
                                            "バージョンまたは変更内容を検索できます。",
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.8f),
                                        )
                                    }
                                }
                            }
                            state.errorMessage?.let { message ->
                                item {
                                    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                                        Text(message, color = MaterialTheme.colorScheme.error)
                                        TextButton(onClick = onRetry) { Text("再試行") }
                                    }
                                }
                            }
                            if (query.isNotBlank() && filteredSections.isEmpty()) {
                                item {
                                    Text(
                                        "表示できる更新履歴がありません",
                                        style = MaterialTheme.typography.bodyMedium,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        modifier = Modifier.fillMaxWidth().padding(vertical = 24.dp),
                                    )
                                }
                            }
                            itemsIndexed(filteredSections, key = { index, section -> section.version ?: "legacy-$index" }) { index, section ->
                                Surface(
                                    modifier = Modifier.fillMaxWidth(),
                                    shape = RoundedCornerShape(16.dp),
                                    color = MaterialTheme.colorScheme.surfaceContainerLow,
                                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant),
                                ) {
                                    Column(
                                        Modifier.padding(horizontal = 16.dp, vertical = 15.dp),
                                        verticalArrangement = Arrangement.spacedBy(10.dp),
                                    ) {
                                        Row(
                                            Modifier.fillMaxWidth(),
                                            verticalAlignment = Alignment.CenterVertically,
                                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                                        ) {
                                            Text(
                                                section.version?.let { "v$it" } ?: "更新履歴",
                                                style = MaterialTheme.typography.titleMedium,
                                                fontWeight = FontWeight.Bold,
                                                color = MaterialTheme.colorScheme.primary,
                                            )
                                            if (index == 0 && section.version != null) {
                                                Surface(
                                                    shape = RoundedCornerShape(50),
                                                    color = MaterialTheme.colorScheme.primaryContainer,
                                                ) {
                                                    Text(
                                                        "最新",
                                                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
                                                        style = MaterialTheme.typography.labelSmall,
                                                        fontWeight = FontWeight.Bold,
                                                        color = MaterialTheme.colorScheme.onPrimaryContainer,
                                                    )
                                                }
                                            }
                                        }
                                        MarkdownText(section.markdown)
                                    }
                                }
                            }
                        }
                    }
                }
                state.loading -> Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(12.dp)) {
                        CircularProgressIndicator()
                        Text("更新履歴を読み込んでいます…", style = MaterialTheme.typography.bodyMedium)
                    }
                }
                else -> Column(
                    Modifier.fillMaxSize().padding(vertical = 24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    Text(state.errorMessage ?: "更新履歴を取得できませんでした。", color = MaterialTheme.colorScheme.onSurfaceVariant)
                    TextButton(onClick = onRetry) { Text("再試行") }
                }
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text("閉じる") } },
    )
}
