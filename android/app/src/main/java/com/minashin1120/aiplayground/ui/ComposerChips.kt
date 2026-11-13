package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.ChatViewModel
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.ComposerRules
import com.minashin1120.aiplayground.data.EFFORT_LEVELS
import com.minashin1120.aiplayground.data.OptionRule
import com.minashin1120.aiplayground.data.THINKING_LEVELS

/*
 * The prompt-bar option chips of the Web `#standard-chat-controls` /
 * `#temporary-chat-controls` (composer*.html + chat.custom.css), split out of
 * `Composer.kt` so the prompt bar stays under the Kotlin size budget.
 */

@Composable
internal fun ModeChip(label: String, checked: Boolean, tone: ChipTone, rule: OptionRule, onToggle: () -> Unit, modifier: Modifier = Modifier) {
    WebCheckChip(label, checked, { onToggle() }, tone.accent, composerChipStyle(checked, tone), modifier,
        enabled = !rule.dimmed && !rule.disabled)
}

/** `#prompt-controls-toggle-btn` (compact mode only). */
@Composable
internal fun DetailsToggle(expanded: Boolean, onClick: () -> Unit, modifier: Modifier = Modifier) {
    val web = LocalWebPalette.current
    val color = if (web.isLight) web.theme.t700 else web.theme200
    Row(
        modifier.heightIn(min = 30.4.dp).clip(CircleShape).background(web.theme.rgb(0.10f))
            .border(1.dp, web.theme.rgb(0.27f), CircleShape)
            .clickable(role = Role.Button, onClick = onClick)
            .padding(horizontal = 10.88.dp, vertical = 5.12.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Text(if (expanded) "折りたたむ" else "詳細", fontSize = 10.sp, fontWeight = FontWeight.SemiBold, letterSpacing = 0.2.sp, color = color)
        FaIcon(if (expanded) R.drawable.fa_solid_chevron_up else R.drawable.fa_solid_chevron_down, null, size = 10.dp, tint = color)
    }
}

/** `#standard-chat-controls` and `#temporary-chat-controls`. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
internal fun DetailChips(
    state: ChatState,
    model: ChatViewModel,
    rules: ComposerRules,
    onChatInstructions: () -> Unit,
    onCompressionSettings: () -> Unit,
    onTemporarySettings: () -> Unit,
) {
    val web = LocalWebPalette.current
    val light = web.isLight
    fun label(dark: Color, lightColor: Color) = if (light) lightColor else dark
    val gray = label(Color(156, 163, 175), Color(91, 102, 117))
    FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
        val center = Modifier.align(Alignment.CenterVertically)
        OptChip("Search", state.enableSearch, rules.search, model::toggleSearch, Tw.blue500, center)
        if (rules.urls.visible) OptChip("URLs", state.enableUrlContext, rules.urls, model::toggleUrlContext, Tw.indigo500, center)
        if (rules.maps.visible) OptChip("Maps", state.enableMaps, rules.maps, model::toggleMaps, Tw.emerald500, center)
        if (rules.python.visible) OptChip("Python", state.enablePython, rules.python, model::togglePython, Tw.yellow500, center,
            labelColor = label(Color(254, 240, 138), Color(133, 77, 14)))
        OptChip("File", state.enableFileCreation, rules.file, model::toggleFileCreation, Tw.orange500, center,
            labelColor = if (light) Color(154, 52, 18) else null)
        if (rules.mcp.visible) OptChip("MCP", state.enableMcp, rules.mcp, model::toggleMcp, Tw.cyan500, center,
            labelColor = label(Color(165, 243, 252), Color(14, 116, 144)))
        OptChip("Skills", state.enableSkills, OptionRule(), model::toggleSkills, Tw.blue500, center,
            labelColor = label(Color(191, 219, 254), Color(30, 64, 175)))
        OptGroup(state.enableSystemPrompt, center, dimmed = rules.sysPrompt.dimmed) {
            GroupCheck("SysPrompt", state.enableSystemPrompt, rules.sysPrompt, model::toggleSystemPrompt,
                label(Color(134, 239, 172), Color(21, 128, 61)))
            GearButton("Chat Settings", onChatInstructions)
        }
        if (rules.thinking.visible) {
            // Web と同じく reasoning-effort モデルも Thinking ボタン（ON/OFF＋レベル）で操作する。
            val thinkingKey = if (rules.effortAsThinking) "reasoning_effort" else "thinking_level"
            val thinkingOptions = if (rules.effortAsThinking) EFFORT_LEVELS else THINKING_LEVELS
            val thinkingLabels = if (rules.effortAsThinking) EFFORT_LABELS else THINKING_LABELS
            val thinkingValue = state.chipValues[thinkingKey].orEmpty()
            val thinkingChecked = if (rules.effortAsThinking) thinkingValue != "none" else state.enableThinking
            val levels = thinkingOptions.filter { it in rules.thinkingLevels }
            OptGroup(thinkingChecked, center, dimmed = rules.thinking.dimmed) {
                GroupCheck("Thinking", thinkingChecked, rules.thinking, {
                    if (rules.effortAsThinking) {
                        val next = if (thinkingChecked) "none"
                        else levels.firstOrNull { it != "none" } ?: rules.effortFallback
                        model.generationOption("reasoning_effort", next)
                    } else model.toggleThinking()
                }, label(Color(216, 180, 254), Color(126, 34, 206)))
                ChipSelect(
                    value = thinkingValue,
                    options = levels.map { WebOption(it, thinkingLabels.getValue(it)) },
                    enabled = levels.isNotEmpty(),
                    onSelect = { model.generationOption(thinkingKey, it) },
                    display = thinkingLabels,
                )
                if (!rules.effortAsThinking) {
                    Text("Budget", fontSize = 10.sp, color = label(Color(216, 180, 254).copy(alpha = 0.8f), Color(126, 34, 206)))
                    BudgetField(state.chipValues["thinking_budget"].orEmpty(), rules.budgetEnabled) { model.generationOption("thinking_budget", it) }
                }
            }
        }
        OptGroup(false, center) {
            Text("Safety:", fontSize = 10.sp, color = gray)
            ChipSelect(
                value = state.chipValues["safety_setting"].orEmpty(),
                options = listOf(WebOption("default", "Default"), WebOption("none", "None")),
                onSelect = { model.generationOption("safety_setting", it) },
            )
        }
        // Web `updatePromptCacheUi`: `ring-1 ring-teal-500/50` around the chip while PromptCache is on.
        OptGroup(state.enablePromptCache, center.then(if (state.enablePromptCache) Modifier.outerRing(Tw.teal500.copy(alpha = 0.5f)) else Modifier),
            dimmed = rules.promptCache.dimmed) {
            GroupCheck("PromptCache", state.enablePromptCache, rules.promptCache, model::togglePromptCache,
                label(Color(94, 234, 212), Color(15, 118, 110)))
        }
        OptGroup(state.compression.enabled, center) {
            GroupCheck("Compress", state.compression.enabled, OptionRule(),
                { model.saveCompressionSettings(state.compression.copy(enabled = !state.compression.enabled)) }, gray)
            GearButton("Compression Settings", onCompressionSettings)
        }
        Row(center, verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
            val temporary = state.selected?.isTemporary == true || state.newThreadTemporary
            OptChip("一時チャット", temporary, OptionRule(), model::toggleTemporaryChat, Tw.amber500,
                labelColor = label(Color(252, 211, 77), Color(146, 64, 14)))
            TemporarySettingsButton(onTemporarySettings)
        }
    }
}

internal val THINKING_LABELS = mapOf("minimal" to "Min", "low" to "Low", "medium" to "Mid", "high" to "High", "xhigh" to "XHigh")
internal val EFFORT_LABELS = mapOf("none" to "None", "low" to "Low", "medium" to "Med", "high" to "High", "xhigh" to "XHigh", "max" to "Max")

/** `<label class="composer-opt">` checkbox chip. */
@Composable
private fun OptChip(
    label: String,
    checked: Boolean,
    rule: OptionRule,
    onToggle: () -> Unit,
    accent: Color,
    modifier: Modifier = Modifier,
    labelColor: Color? = null,
) {
    WebCheckChip(label, checked, { onToggle() }, accent, composerOptStyle(checked), modifier,
        labelColor = labelColor, enabled = !rule.dimmed && !rule.disabled)
}

/** Tailwind `ring-1`: a 1dp line just outside the pill (a box-shadow, so it does not change the layout). */
private fun Modifier.outerRing(color: Color): Modifier = drawBehind {
    val width = 1.dp.toPx()
    drawRoundRect(color, topLeft = Offset(-width / 2f, -width / 2f), size = Size(size.width + width, size.height + width),
        cornerRadius = CornerRadius(size.height / 2f + width / 2f), style = Stroke(width))
}

/** `<div class="composer-opt composer-opt-group">`: a round pill that holds a checkbox label and extra controls. */
@Composable
private fun OptGroup(checked: Boolean, modifier: Modifier = Modifier, dimmed: Boolean = false, content: @Composable RowScope.() -> Unit) {
    WebPill(composerOptStyle(checked, group = true), modifier, enabled = !dimmed, content = content)
}

@Composable
private fun GroupCheck(label: String, checked: Boolean, rule: OptionRule, onToggle: () -> Unit, labelColor: Color) {
    Row(
        Modifier.toggleable(checked, enabled = !rule.dimmed && !rule.disabled, role = Role.Checkbox) { onToggle() },
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        WebCheckbox(checked, onCheckedChange = null)
        Text(label, fontSize = 10.sp, color = labelColor, maxLines = 1)
    }
}

/** The small `fa-cog` button inside SysPrompt / Compress. */
@Composable
private fun GearButton(label: String, onClick: () -> Unit) {
    val web = LocalWebPalette.current
    Box(
        Modifier.clip(RoundedCornerShape(12.dp)).clickable(role = Role.Button, onClick = onClick)
            .semantics { contentDescription = label }.padding(4.dp),
    ) {
        FaIcon(R.drawable.fa_solid_cog, null, size = 10.dp, tint = if (web.isLight) Color(105, 117, 136) else Color(107, 114, 128))
    }
}

/** The amber "設定" button next to 一時チャット. */
@Composable
private fun TemporarySettingsButton(onClick: () -> Unit) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(12.dp)
    Box(
        Modifier.heightIn(min = 22.dp).clip(shape)
            .background(Brush.verticalGradient(listOf(Color.White.copy(alpha = 0.04f), Color.Black.copy(alpha = 0.06f))))
            .border(1.dp, Color.White.copy(alpha = 0.06f), shape)
            .clickable(role = Role.Button, onClick = onClick)
            .padding(horizontal = 6.dp, vertical = 2.dp),
        contentAlignment = Alignment.Center,
    ) {
        Text("設定", fontSize = 10.sp, lineHeight = 16.sp, color = if (web.isLight) Color(146, 64, 14) else Color(252, 211, 77))
    }
}

/** A `<select>` inside a detail chip (`#thinking-level`, `#reasoning-effort`, `#safety-setting`). */
@Composable
private fun ChipSelect(
    value: String,
    options: List<WebOption>,
    onSelect: (String) -> Unit,
    enabled: Boolean = true,
    display: Map<String, String> = emptyMap(),
) {
    val web = LocalWebPalette.current
    val shown = if (options.any { it.value == value }) options else options + WebOption(value, display[value] ?: value)
    WebSelect(
        value = value, options = shown, onSelect = onSelect, enabled = enabled, fontSize = 10.sp,
        background = if (web.isLight) Color.White else Color(8, 14, 28).copy(alpha = 0.7f),
        borderColor = if (web.isLight) Color(15, 23, 42).copy(alpha = 0.14f) else Color(148, 163, 184).copy(alpha = 0.16f),
        textColor = if (web.isLight) web.text else Color(226, 232, 240),
        shape = RoundedCornerShape(6.dp),
        contentPadding = PaddingValues(horizontal = 4.dp, vertical = 4.dp),
    )
}

/** `#thinking-budget` (`type=number`, 0–32768). */
@Composable
private fun BudgetField(value: String, enabled: Boolean, onChange: (String) -> Unit) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(6.dp)
    val color = if (web.isLight) web.text else Color(226, 232, 240)
    BasicTextField(
        value = value,
        onValueChange = { next -> if (next.length <= 5 && next.all(Char::isDigit)) onChange(next) },
        enabled = enabled,
        singleLine = true,
        textStyle = TextStyle(fontSize = 10.sp, color = color, fontFamily = WebFonts.sans),
        cursorBrush = SolidColor(color),
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
        modifier = Modifier.width(80.dp).alpha(if (enabled) 1f else 0.5f).clip(shape)
            .background(if (web.isLight) Color.White else Color(8, 14, 28).copy(alpha = 0.7f))
            .border(1.dp, if (web.isLight) Color(15, 23, 42).copy(alpha = 0.14f) else Color(148, 163, 184).copy(alpha = 0.16f), shape)
            .padding(horizontal = 4.dp, vertical = 4.dp)
            .semantics { contentDescription = "Thinking Budget" },
    )
}
