package com.minashin1120.aiplayground.ui

import androidx.annotation.DrawableRes
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsFocusedAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.AppChangelogUiState
import com.minashin1120.aiplayground.BuildConfig
import com.minashin1120.aiplayground.R

internal data class AppChangelogSection(
    val version: String?,
    val markdown: String,
)

private val APP_CHANGELOG_HEADING = Regex("^#\\s+Android版更新履歴\\s+-\\s+(.+?)\\s*$")
private val APP_CHANGELOG_BULLET = Regex("^[-*+]\\s+(.+)$")

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

/**
 * The change items of a section written as a flat bullet list (the format of `ci/changelogs/vX.Y.Z.md`).
 * Indented lines continue the previous item. Returns null for anything else (headings, paragraphs,
 * nested lists), which is then shown as plain Markdown.
 */
internal fun appChangelogItems(markdown: String): List<String>? {
    val items = mutableListOf<String>()
    for (line in markdown.replace("\r\n", "\n").lines()) {
        if (line.isBlank()) continue
        val bullet = APP_CHANGELOG_BULLET.matchEntire(line)
        when {
            bullet != null -> items += bullet.groupValues[1].trim()
            items.isNotEmpty() && line.first().isWhitespace() && APP_CHANGELOG_BULLET.matchEntire(line.trim()) == null ->
                items[items.lastIndex] = items.last() + " " + line.trim()
            else -> return null
        }
    }
    return items.ifEmpty { null }
}

@Composable
fun AppChangelogDialog(
    state: AppChangelogUiState,
    onDismiss: () -> Unit,
    onRetry: () -> Unit,
    installedVersion: String = BuildConfig.VERSION_NAME,
) {
    PlaygroundDialog(
        onDismissRequest = onDismiss,
        title = { Text("Android版の更新履歴") },
        subtitle = "このアプリの変更点",
        icon = R.drawable.fa_solid_history,
        panelMaxWidth = 760.dp,
        fillHeight = true,
        text = {
            val content = state.content
            when {
                content != null -> ChangelogBody(content, state.errorMessage, installedVersion, onRetry)
                state.loading -> ChangelogSkeleton()
                else -> Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    ChangelogNotice(
                        icon = R.drawable.fa_solid_exclamation_triangle,
                        title = "更新履歴を取得できませんでした",
                        message = state.errorMessage,
                        danger = true,
                        onRetry = onRetry,
                    )
                }
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text("閉じる") } },
    )
}

@Composable
private fun ChangelogBody(
    content: String,
    errorMessage: String?,
    installedVersion: String,
    onRetry: () -> Unit,
) {
    val web = LocalWebPalette.current
    val reduce = LocalReduceMotion.current
    val sections = remember(content) { parseAppChangelogSections(content) }
    var query by remember(content) { mutableStateOf("") }
    val filteredSections = remember(sections, query) { filterAppChangelogSections(sections, query) }
    val searching = query.isNotBlank()
    val latestVersion = sections.firstOrNull()?.version

    Column(Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        ChangelogSearchField(query, onChange = { query = it })
        if (searching) {
            Text(
                if (filteredSections.isNotEmpty()) "${filteredSections.size} 件の更新履歴が見つかりました" else "該当する更新履歴はありません",
                color = web.muted,
                fontSize = 12.sp,
                modifier = Modifier.padding(horizontal = 4.dp),
            )
        }
        LazyColumn(
            modifier = Modifier.weight(1f).fillMaxWidth(),
            contentPadding = PaddingValues(top = 2.dp, bottom = 8.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            if (!searching && latestVersion != null) {
                item(key = "summary") {
                    ChangelogSummary(latestVersion, installedVersion, sections.count { it.version != null })
                }
            }
            errorMessage?.let { message ->
                item(key = "error") {
                    ChangelogNotice(
                        icon = R.drawable.fa_solid_exclamation_triangle,
                        title = "最新の更新履歴を取得できませんでした",
                        message = message,
                        danger = true,
                        onRetry = onRetry,
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
            }
            if (searching && filteredSections.isEmpty()) {
                item(key = "empty") {
                    ChangelogNotice(
                        icon = R.drawable.fa_solid_search,
                        title = "表示できる更新履歴がありません",
                        message = "バージョン番号や別のキーワードで検索してください。",
                        modifier = Modifier.fillMaxWidth().padding(vertical = 24.dp),
                    )
                }
            }
            itemsIndexed(filteredSections, key = { index, section -> section.version ?: "legacy-$index" }) { index, section ->
                ChangelogEntry(
                    section = section,
                    latest = section.version != null && section.version == latestVersion,
                    installed = section.version != null && section.version == installedVersion,
                    first = index == 0,
                    last = index == filteredSections.lastIndex,
                    modifier = if (reduce) Modifier else Modifier.animateItem(),
                )
            }
        }
    }
}

/** Web `.search-box`: a rounded field that takes the theme border and a soft ring while focused. */
@Composable
private fun ChangelogSearchField(value: String, onChange: (String) -> Unit) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(12.dp)
    val interaction = remember { MutableInteractionSource() }
    val focused by interaction.collectIsFocusedAsState()
    val style = TextStyle(fontSize = 14.sp, lineHeight = 20.sp, color = web.text, fontFamily = WebFonts.sans)
    BasicTextField(
        value, onChange,
        singleLine = true,
        textStyle = style,
        cursorBrush = SolidColor(web.text),
        interactionSource = interaction,
        modifier = Modifier
            .fillMaxWidth()
            .then(if (focused) Modifier.border(3.dp, web.theme.rgb(0.2f), RoundedCornerShape(15.dp)).padding(3.dp) else Modifier.padding(3.dp))
            .clip(shape)
            .background(if (web.isLight) Color.White else Color(10, 16, 35).copy(alpha = 0.7f))
            .border(1.dp, if (focused) web.theme.t500 else web.line, shape)
            .semantics { contentDescription = "更新履歴を検索" },
        decorationBox = { inner ->
            Row(Modifier.padding(start = 14.dp, end = 6.dp), verticalAlignment = Alignment.CenterVertically) {
                FaIcon(R.drawable.fa_solid_search, null, size = 13.dp, tint = if (focused) web.theme300 else web.muted)
                Box(Modifier.weight(1f).padding(start = 11.dp, top = 12.dp, bottom = 12.dp)) {
                    if (value.isEmpty()) Text("更新履歴を検索...", style = style.copy(color = web.muted), maxLines = 1)
                    inner()
                }
                if (value.isNotEmpty()) {
                    Box(
                        Modifier
                            .size(34.dp)
                            .clip(CircleShape)
                            .clickable(onClickLabel = "検索をクリア", role = Role.Button) { onChange("") }
                            .semantics { contentDescription = "検索をクリア" },
                        contentAlignment = Alignment.Center,
                    ) { FaIcon(R.drawable.fa_solid_times, null, size = 13.dp, tint = web.muted) }
                } else {
                    Spacer(Modifier.width(8.dp))
                }
            }
        },
    )
}

/** Hero card above the timeline: the newest release and how this installation compares to it. */
@Composable
private fun ChangelogSummary(latestVersion: String, installedVersion: String, releaseCount: Int) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(18.dp)
    val upToDate = installedVersion == latestVersion
    Column(
        Modifier
            .fillMaxWidth()
            .clip(shape)
            .background(Brush.linearGradient(listOf(web.theme.rgb(0.24f), web.theme.rgb(0.05f)), start = Offset.Zero, end = Offset.Infinite))
            .border(1.dp, web.theme.rgb(0.32f), shape)
            .padding(horizontal = 18.dp, vertical = 16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(14.dp)) {
            val tile = RoundedCornerShape(14.dp)
            Box(
                Modifier
                    .size(46.dp)
                    .clip(tile)
                    .background(Brush.linearGradient(listOf(web.theme.t500, web.theme.t600), start = Offset.Zero, end = Offset.Infinite))
                    .border(1.dp, web.theme.rgb(0.45f), tile),
                contentAlignment = Alignment.Center,
            ) { FaIcon(R.drawable.fa_solid_rocket, null, size = 19.dp, tint = web.textInverse) }
            Column(Modifier.weight(1f)) {
                Text("最新バージョン", color = web.theme300, fontSize = 11.sp, fontWeight = FontWeight.Bold, letterSpacing = 0.08.em)
                Text("v$latestVersion", color = web.text, fontSize = 24.sp, lineHeight = 30.sp, fontWeight = FontWeight.Bold)
            }
        }
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            if (upToDate) {
                ChangelogBadge("最新の状態です", web.success, icon = R.drawable.fa_solid_check_circle)
            } else {
                ChangelogBadge("使用中 v$installedVersion", web.theme300, icon = R.drawable.fa_solid_mobile_alt)
            }
            Text("全 $releaseCount 件の更新", color = web.muted, fontSize = 12.sp)
        }
    }
}

/** One release on the timeline: the rail with its node on the left, the card on the right. */
@Composable
private fun ChangelogEntry(
    section: AppChangelogSection,
    latest: Boolean,
    installed: Boolean,
    first: Boolean,
    last: Boolean,
    modifier: Modifier = Modifier,
) {
    val web = LocalWebPalette.current
    val railX = 9.dp
    val nodeY = 27.dp
    val gap = 14.dp
    val railColor = web.theme.rgb(0.28f)
    val nodeColor = web.theme.t500
    val haloColor = web.theme.rgb(0.22f)
    Row(
        modifier
            .fillMaxWidth()
            .drawBehind {
                val x = railX.toPx()
                val y = nodeY.toPx()
                val radius = (if (latest) 9.dp else 5.dp).toPx()
                val clearance = 4.dp.toPx()
                val stroke = 2.dp.toPx()
                if (!first) drawLine(railColor, Offset(x, 0f), Offset(x, y - radius - clearance), stroke)
                // The rail runs on through the gap to the next entry.
                if (!last) drawLine(railColor, Offset(x, y + radius + clearance), Offset(x, size.height + gap.toPx()), stroke)
                if (latest) {
                    drawCircle(haloColor, radius, Offset(x, y))
                    drawCircle(nodeColor, 5.dp.toPx(), Offset(x, y))
                } else {
                    drawCircle(nodeColor.copy(alpha = 0.7f), radius - stroke / 2, Offset(x, y), style = Stroke(stroke))
                }
            }
            .padding(start = railX * 2 + 10.dp),
    ) {
        ChangelogCard(section, latest, installed)
    }
}

/** Web `.log-card`: the version header with its badges and the list of changes. */
@Composable
private fun ChangelogCard(section: AppChangelogSection, latest: Boolean, installed: Boolean) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(16.dp)
    val items = remember(section.markdown) { appChangelogItems(section.markdown) }
    val background = if (web.isLight) Color.White else Color(10, 16, 35).copy(alpha = 0.84f)
    Column(
        Modifier
            .fillMaxWidth()
            .clip(shape)
            .background(background)
            .then(
                if (latest) Modifier.background(Brush.verticalGradient(listOf(web.theme.rgb(0.10f), Color.Transparent)))
                else Modifier,
            )
            .border(1.dp, if (latest) web.theme.rgb(0.38f) else web.line, shape)
            .padding(horizontal = 16.dp, vertical = 14.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Row(
            Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            FaIcon(R.drawable.fa_solid_tag, null, size = 12.dp, tint = if (latest) web.theme300 else web.muted)
            Text(
                section.version?.let { "v$it" } ?: "更新履歴",
                color = if (latest) web.theme300 else web.text,
                fontSize = 16.sp,
                lineHeight = 22.sp,
                fontWeight = FontWeight.Bold,
            )
            if (latest) ChangelogBadge("最新", web.theme300)
            if (installed) ChangelogBadge("使用中", web.success)
            Spacer(Modifier.weight(1f))
            if (items != null) Text("${items.size} 件の変更", color = web.muted, fontSize = 11.sp)
        }
        Box(Modifier.fillMaxWidth().height(1.dp).background(web.lineSoft))
        if (items != null) {
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                items.forEach { item ->
                    Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        Box(
                            Modifier
                                .padding(top = 10.dp)
                                .size(6.dp)
                                .clip(CircleShape)
                                .background(web.theme.rgb(if (latest) 0.9f else 0.6f)),
                        )
                        Box(Modifier.weight(1f)) { MarkdownText(item) }
                    }
                }
            }
        } else {
            MarkdownText(section.markdown)
        }
    }
}

@Composable
private fun ChangelogBadge(label: String, color: Color, @DrawableRes icon: Int? = null) {
    val shape = RoundedCornerShape(50)
    Row(
        Modifier
            .clip(shape)
            .background(color.copy(alpha = 0.14f))
            .border(1.dp, color.copy(alpha = 0.35f), shape)
            .padding(horizontal = 9.dp, vertical = 3.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(5.dp),
    ) {
        if (icon != null) FaIcon(icon, null, size = 10.dp, tint = color)
        Text(label, color = color, fontSize = 11.sp, lineHeight = 14.sp, fontWeight = FontWeight.Bold, maxLines = 1)
    }
}

/** Centered notice for the empty search result and load failures, with a retry button when given. */
@Composable
private fun ChangelogNotice(
    @DrawableRes icon: Int,
    title: String,
    message: String?,
    modifier: Modifier = Modifier,
    danger: Boolean = false,
    onRetry: (() -> Unit)? = null,
) {
    val web = LocalWebPalette.current
    val tone = if (danger) web.danger else web.theme300
    Column(
        modifier.padding(horizontal = 16.dp, vertical = 12.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Box(
            Modifier.size(48.dp).clip(CircleShape).background(tone.copy(alpha = 0.12f)).border(1.dp, tone.copy(alpha = 0.3f), CircleShape),
            contentAlignment = Alignment.Center,
        ) { FaIcon(icon, null, size = 18.dp, tint = tone) }
        Text(title, color = web.text, fontSize = 14.sp, fontWeight = FontWeight.Bold, textAlign = TextAlign.Center)
        if (!message.isNullOrBlank()) {
            Text(message, color = web.muted, fontSize = 12.sp, lineHeight = 18.sp, textAlign = TextAlign.Center)
        }
        if (onRetry != null) {
            WebButton(onClick = onRetry, variant = WebButtonVariant.Primary, modifier = Modifier.padding(top = 4.dp)) {
                FaIcon(R.drawable.fa_solid_sync, null, size = 12.dp)
                Text("再試行")
            }
        }
    }
}

/** Web `.changelog-skeleton`: placeholder cards with a shimmer while the release notes load. */
@Composable
private fun ChangelogSkeleton() {
    val web = LocalWebPalette.current
    val reduce = LocalReduceMotion.current
    val shift = if (reduce) {
        0.5f
    } else {
        val transition = rememberInfiniteTransition(label = "changelog skeleton")
        val value by transition.animateFloat(
            initialValue = 0f,
            targetValue = 1f,
            animationSpec = infiniteRepeatable(tween(1550, easing = LinearEasing), RepeatMode.Restart),
            label = "changelog shimmer",
        )
        value
    }
    val base = if (web.isLight) Color(148, 163, 184).copy(alpha = 0.16f) else Color(148, 163, 184).copy(alpha = 0.12f)
    val highlight = web.theme.rgb(0.28f)
    Column(
        Modifier.fillMaxSize().semantics { contentDescription = "更新履歴を読み込み中" },
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        repeat(4) { card ->
            val shape = RoundedCornerShape(16.dp)
            Column(
                Modifier
                    .fillMaxWidth()
                    .clip(shape)
                    .background(if (web.isLight) Color.White else Color(10, 16, 35).copy(alpha = 0.84f))
                    .border(1.dp, web.line, shape)
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                val widths = listOf(0.36f + (card % 3) * 0.08f, 0.94f, 0.78f + (card % 4) * 0.04f, 0.6f)
                widths.forEachIndexed { line, width ->
                    Box(
                        Modifier
                            .fillMaxWidth(width)
                            .height(if (line == 0) 14.dp else 11.dp)
                            .clip(RoundedCornerShape(50))
                            .drawBehind {
                                val span = size.width * 2.2f
                                val start = size.width - span * (shift - line * 0.03f)
                                drawRect(
                                    Brush.linearGradient(
                                        listOf(base, base, highlight, base, base),
                                        start = Offset(start, 0f),
                                        end = Offset(start + span, 0f),
                                    ),
                                )
                            },
                    )
                }
            }
        }
    }
}
