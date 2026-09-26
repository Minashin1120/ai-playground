@file:OptIn(androidx.compose.foundation.layout.ExperimentalLayoutApi::class)

package com.minashin1120.aiplayground.ui

import androidx.annotation.DrawableRes
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.ModelInfo
import com.minashin1120.aiplayground.data.PROVIDER_LABELS
import com.minashin1120.aiplayground.data.modelApiProvider

/** Web `#model-tag-bar` in its fixed order; the value is the lower-cased label (`activeModelTag`). */
internal val MODEL_TAGS = listOf(
    "All", "OpenAI", "Gemini", "Anthropic", "Kimi", "DeepSeek", "Mistral", "xAI", "Image", "Video", "Audio",
    "Music", "Transcription", "OCR", "Reasoning", "Fast", "Agent", "Agentic View",
)

/** Group icon of the Web `MODELS` category (`fas fa-star text-yellow-400` → drawable). */
@DrawableRes
internal fun modelGroupIcon(iconClass: String): Int? = when (iconClass.split(' ').firstOrNull { it.startsWith("fa-") }) {
    "fa-star" -> R.drawable.fa_solid_star
    "fa-bolt" -> R.drawable.fa_solid_bolt
    "fa-brain" -> R.drawable.fa_solid_brain
    "fa-clapperboard" -> R.drawable.fa_solid_clapperboard
    "fa-closed-captioning" -> R.drawable.fa_solid_closed_captioning
    "fa-file" -> R.drawable.fa_solid_file
    "fa-headset" -> R.drawable.fa_solid_headset
    "fa-history" -> R.drawable.fa_solid_history
    "fa-image" -> R.drawable.fa_solid_image
    "fa-magic" -> R.drawable.fa_solid_magic
    "fa-microphone" -> R.drawable.fa_solid_microphone
    "fa-music" -> R.drawable.fa_solid_music
    "fa-paint-brush" -> R.drawable.fa_solid_paint_brush
    "fa-robot" -> R.drawable.fa_solid_robot
    "fa-rocket" -> R.drawable.fa_solid_rocket
    else -> null
}

/** Web `renderModelList` visibility: search text, PromptCache provider lock and the active tag. */
internal fun modelVisible(model: ModelInfo, query: String, tag: String, lockedProvider: String?): Boolean {
    val searchText = listOf(model.name, model.id, model.apiId, if (model.agenticView) "agentic view" else "", model.category,
        model.tags.joinToString(" "), model.searchTerms.joinToString(" ")).joinToString(" ").lowercase()
    return searchText.contains(query.lowercase()) &&
        (lockedProvider == null || modelApiProvider(model.id) == lockedProvider) &&
        (tag == "all" || tag in model.tags)
}

/** `#model-modal` (Select Model): search, fixed tag bar, category groups and two-column cards. */
@Composable
internal fun ModelPicker(
    state: ChatState,
    onDismiss: () -> Unit,
    onSelect: (String) -> Unit,
    selectedId: String = state.model,
    /** The PromptCache lock applies to the chat model, not to other pickers. */
    lockToPromptCache: Boolean = true,
) {
    val web = LocalWebPalette.current
    var query by remember { mutableStateOf("") }
    var tag by remember { mutableStateOf("all") }
    val lockedProvider = if (lockToPromptCache && state.enablePromptCache) modelApiProvider(state.model) else null
    val lockedLabel = lockedProvider?.let { PROVIDER_LABELS[it] ?: it }.orEmpty()
    val models = state.account?.models.orEmpty().filter { !it.deprecated && it.category.isNotBlank() }
        .sortedBy { it.webCatalogOrder }
    val groups = models.filter { modelVisible(it, query.trim(), tag, lockedProvider) }.groupBy { it.category }.toList()
    val listState = rememberLazyListState()
    LaunchedEffect(Unit) {
        val index = groups.indexOfFirst { (_, items) -> items.any { it.id == selectedId } }
        if (index > 0) listState.scrollToItem(index + if (lockedProvider != null) 1 else 0)
    }
    PlaygroundDialog(
        onDismissRequest = onDismiss,
        title = { Text("Select Model") },
        icon = R.drawable.fa_solid_robot,
        subtitle = "使用するモデルを選択してください",
        fillHeight = true,
        padBody = false,
        showFooter = false,
        text = {
            Column(Modifier.fillMaxSize()) {
                ModelSearchBox(query) { query = it }
                val lineColor = if (web.isLight) Color(15, 23, 42).copy(alpha = 0.10f) else web.line
                FlowRow(
                    Modifier.fillMaxWidth().heightIn(max = 132.dp)
                        .background(if (web.isLight) Color(15, 23, 42).copy(alpha = 0.035f) else Color(8, 12, 22).copy(alpha = 0.45f))
                        .drawBehind { drawLine(lineColor, Offset(0f, 0f), Offset(size.width, 0f), 1.dp.toPx()) }
                        .verticalScroll(rememberScrollState()).padding(horizontal = 14.dp, vertical = 10.dp),
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    MODEL_TAGS.forEach { label ->
                        val value = label.lowercase()
                        ModelTagButton(label, tag == value) { tag = value }
                    }
                }
                LazyColumn(
                    state = listState,
                    modifier = Modifier.weight(1f).fillMaxWidth().drawBehind {
                        drawRect(Brush.radialGradient(listOf(web.theme.rgb(0.06f), Color.Transparent),
                            center = Offset(size.width, 0f), radius = 364.dp.toPx()))
                    },
                    contentPadding = PaddingValues(start = 12.dp, end = 12.dp, top = 12.dp, bottom = 14.dp),
                ) {
                    if (lockedProvider != null) item(key = "banner") {
                        val shape = RoundedCornerShape(12.dp)
                        Row(
                            Modifier.fillMaxWidth().padding(bottom = 16.dp).clip(shape).background(web.theme.rgb(0.08f))
                                .border(1.dp, web.theme.rgb(0.35f), shape).padding(horizontal = 14.dp, vertical = 10.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            FaIcon(R.drawable.fa_solid_database, null, size = 11.dp, tint = web.theme200, modifier = Modifier.padding(end = 6.dp))
                            Text(buildAnnotatedString {
                                append("PromptCache 有効中: ")
                                withStyle(SpanStyle(fontWeight = FontWeight.Bold)) { append(lockedLabel) }
                                append(" のモデルのみ選択できます（他APIへの切替は不可）")
                            }, fontSize = 11.sp, color = web.theme200)
                        }
                    }
                    if (groups.isEmpty()) item(key = "empty") {
                        Text(if (lockedProvider != null) "No $lockedLabel models found." else "No models found.",
                            fontSize = 13.sp, color = web.muted, modifier = Modifier.fillMaxWidth().padding(vertical = 32.dp),
                            textAlign = TextAlign.Center)
                    }
                    items(groups, key = { it.first }) { (category, items) ->
                        Column(Modifier.padding(bottom = 18.dp)) {
                            val first = items.first()
                            Row(Modifier.padding(horizontal = 2.dp).padding(bottom = 12.dp), verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                                modelGroupIcon(first.categoryIcon)?.let { FaIcon(it, null, size = 13.dp, tint = web.theme300) }
                                Column {
                                    Text(category, fontSize = 13.sp, fontWeight = FontWeight.Bold, letterSpacing = 0.26.sp, color = web.theme300)
                                    Text(first.categoryDescription, fontSize = 11.sp, color = web.muted)
                                }
                            }
                            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                items.chunked(2).forEach { row ->
                                    Row(Modifier.height(IntrinsicSize.Min), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                        row.forEach { info ->
                                            ModelCard(info, info.id == selectedId, Modifier.weight(1f).fillMaxHeight()) { onSelect(info.id) }
                                        }
                                        if (row.size == 1) Spacer(Modifier.weight(1f))
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        confirmButton = {},
    )
}

@Composable
private fun ModelSearchBox(value: String, onChange: (String) -> Unit) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(12.dp)
    val style = TextStyle(fontSize = 13.sp, lineHeight = 20.sp, color = web.text, fontFamily = WebFonts.sans)
    BasicTextField(
        value = value, onValueChange = onChange, singleLine = true, textStyle = style, cursorBrush = SolidColor(web.text),
        modifier = Modifier.fillMaxWidth().padding(start = 14.dp, end = 14.dp, bottom = 10.dp).clip(shape)
            .background(if (web.isLight) Color.White else Color.White.copy(alpha = 0.04f))
            .border(1.dp, if (web.isLight) Color(15, 23, 42).copy(alpha = 0.14f) else web.line, shape)
            .semantics { contentDescription = "Search models or capabilities..." },
        decorationBox = { inner ->
            Row(Modifier.padding(horizontal = 12.dp, vertical = 11.dp), verticalAlignment = Alignment.CenterVertically) {
                FaIcon(R.drawable.fa_solid_search, null, size = 12.dp, tint = web.muted)
                Box(Modifier.weight(1f).padding(start = 12.dp)) {
                    if (value.isEmpty()) Text("Search models or capabilities...", style = style.copy(color = web.muted))
                    inner()
                }
                if (value.isNotEmpty()) Box(
                    Modifier.size(22.dp).clip(CircleShape).background(Color.White.copy(alpha = 0.06f))
                        .clickable(role = Role.Button) { onChange("") }.semantics { contentDescription = "検索をクリア" },
                    contentAlignment = Alignment.Center,
                ) { FaIcon(R.drawable.fa_solid_times, null, size = 10.dp, tint = web.muted) }
            }
        },
    )
}

@Composable
private fun ModelTagButton(label: String, active: Boolean, onClick: () -> Unit) {
    val web = LocalWebPalette.current
    val fg = when {
        !active -> web.muted
        web.isLight -> web.theme.t700
        else -> web.theme200
    }
    val bg = when {
        active -> web.theme.rgb(0.14f)
        web.isLight -> Color(15, 23, 42).copy(alpha = 0.04f)
        else -> Color.White.copy(alpha = 0.03f)
    }
    val border = when {
        active -> web.theme.rgb(0.28f)
        web.isLight -> Color(15, 23, 42).copy(alpha = 0.10f)
        else -> web.line
    }
    Text(label, fontSize = 11.sp, fontWeight = FontWeight.SemiBold, letterSpacing = 0.11.sp, color = fg,
        modifier = Modifier.clip(CircleShape).background(bg).border(1.dp, border, CircleShape)
            .clickable(role = Role.Tab, onClick = onClick).padding(horizontal = 12.dp, vertical = 6.dp))
}

/** `.model-card`: name, Agentic View badge, description, API model and price; the selected card is tinted. */
@Composable
private fun ModelCard(info: ModelInfo, selected: Boolean, modifier: Modifier, onClick: () -> Unit) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(16.dp)
    val background = when {
        selected -> Brush.verticalGradient(listOf(web.theme.rgb(if (web.isLight) 0.14f else 0.16f), web.theme.rgb(0.05f)))
        web.isLight -> Brush.verticalGradient(listOf(Color.White, Color(0xFFF7F9FC)))
        else -> Brush.verticalGradient(listOf(Color.White.copy(alpha = 0.043f), Color.White.copy(alpha = 0.02f)))
    }
    Column(
        modifier.alpha(if (info.selectable) 1f else 0.5f).clip(shape)
            .background(if (web.isLight) Color.White else Color(10, 16, 30).copy(alpha = 0.78f))
            .background(background)
            .border(1.dp, if (selected && !web.isLight) web.theme.rgb(0.55f) else if (web.isLight) Color(15, 23, 42).copy(alpha = 0.08f)
                else Color.White.copy(alpha = 0.08f), shape)
            .clickable(enabled = info.selectable, role = Role.Button, onClick = onClick)
            .padding(horizontal = 16.dp, vertical = 14.dp),
    ) {
        Row(Modifier.fillMaxWidth().padding(bottom = 4.dp), verticalAlignment = Alignment.Top, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            FlowRow(Modifier.weight(1f), horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(info.name, fontSize = 14.sp, fontWeight = FontWeight.Bold, color = web.text)
                if (info.agenticView) Row(
                    Modifier.align(Alignment.CenterVertically).clip(CircleShape).background(Tw.teal900.copy(alpha = 0.2f)).border(1.dp, Tw.teal500.copy(alpha = 0.4f), CircleShape)
                        .padding(horizontal = 8.dp, vertical = 2.dp),
                    verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp),
                ) {
                    FaIcon(R.drawable.fa_solid_eye, null, size = 9.dp, tint = if (web.isLight) Tw.teal700 else Tw.teal200)
                    Text("Agentic View", fontSize = 9.sp, fontWeight = FontWeight.SemiBold, color = if (web.isLight) Tw.teal700 else Tw.teal200)
                }
            }
            if (selected) FaIcon(R.drawable.fa_solid_check_circle, "選択中", size = 14.dp, tint = web.theme300, modifier = Modifier.padding(top = 2.dp))
        }
        if (info.description.isNotBlank()) Text(info.description, fontSize = 10.sp, lineHeight = 15.5.sp, color = web.muted)
        if (info.apiId.isNotBlank()) Text(buildAnnotatedString {
            withStyle(SpanStyle(color = Tw.gray500, fontFamily = WebFonts.sans)) { append("API model: ") }
            append(info.apiId)
        }, fontSize = 10.sp, lineHeight = 20.sp, fontFamily = FontFamily.Monospace, color = Tw.cyan300.copy(alpha = 0.9f),
            modifier = Modifier.padding(top = 6.dp))
        if (info.price.isNotBlank()) Row(Modifier.padding(top = 6.dp), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
            FaIcon(R.drawable.fa_solid_tag, null, size = 9.dp, tint = Tw.amber400.copy(alpha = 0.63f), modifier = Modifier.padding(top = 4.dp))
            Text(info.price, fontSize = 10.sp, lineHeight = 20.sp, fontFamily = FontFamily.Monospace, color = Tw.amber400.copy(alpha = 0.9f))
        }
    }
}
