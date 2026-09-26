@file:OptIn(androidx.compose.foundation.layout.ExperimentalLayoutApi::class)

package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.GenField
import com.minashin1120.aiplayground.data.GenPanel
import com.minashin1120.aiplayground.data.genFieldValue
import com.minashin1120.aiplayground.data.generationPanels
import java.util.Locale

/**
 * The Web generation panels under the prompt bar (`composer_gen_image.html` / `composer_gen_media.html`):
 * image, video, music, OCR, TTS and xAI chat options, the input-limit note and the Lyria RealTime bar.
 */
@Composable
fun GenerationOptionsPanel(
    modelId: String,
    values: Map<String, String>,
    enabled: Boolean,
    onChange: (String, String) -> Unit,
    onOpenLyriaStudio: () -> Unit = {},
    /** Minimal prompt bar: the Web moves these panels into the ＋ popup (`MINIMAL_MODEL_PANEL_IDS`). */
    minimal: Boolean = false,
) {
    val panels = generationPanels(modelId, values).filter { !minimal || it.id !in MINIMAL_MODEL_PANEL_IDS }
    if (panels.isEmpty()) return
    Column {
        panels.forEach { panel ->
            key(panel.id) {
                when {
                    panel.studioBar -> LyriaStudioBar(onOpenLyriaStudio)
                    panel.limits.isNotEmpty() -> LimitsNote(panel)
                    else -> OptionsPanel(panel, values, enabled, onChange)
                }
            }
        }
    }
}

internal val MINIMAL_MODEL_PANEL_IDS = setOf(
    "gpt-image", "gemini-image", "grok-image", "xai-chat", "grok-video", "mistral-ocr", "image-input-limits", "audio-gen",
)

@Composable
private fun panelModifier(border: Color? = null): Modifier {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(8.dp)
    return Modifier.padding(top = 8.dp).fillMaxWidth().clip(shape).background(web.twBg(Tw.gray800, 0.7f))
        .border(1.dp, border ?: web.twBorder(Tw.gray700), shape).padding(horizontal = 12.dp, vertical = 8.dp)
}

@Composable
private fun OptionsPanel(panel: GenPanel, values: Map<String, String>, enabled: Boolean, onChange: (String, String) -> Unit) {
    val web = LocalWebPalette.current
    val size = if (panel.large) 12.sp else 10.sp
    FlowRow(
        panelModifier(),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp),
        itemVerticalAlignment = Alignment.CenterVertically,
    ) {
        panel.fields.filter { !it.hidden }.forEach { field ->
            key(field.key) { PanelField(field, genFieldValue(field, values), size, enabled, panel.note, onChange) }
        }
        if (panel.note.isNotEmpty() && panel.id != "audio-gen") {
            Text(panel.note, fontSize = size, lineHeight = size * 1.5f, color = web.twText(Tw.gray500))
        }
    }
}

@Composable
private fun PanelField(field: GenField, value: String, size: TextUnit, enabled: Boolean, note: String, onChange: (String, String) -> Unit) {
    val web = LocalWebPalette.current
    val label = web.twText(Tw.gray400)
    val usable = enabled && field.enabled
    when (field.kind) {
        GenField.Kind.Check -> Row(
            Modifier.alpha(if (usable) 1f else 0.5f)
                .toggleable(value == "true", enabled = usable, role = Role.Checkbox) { onChange(field.key, it.toString()) },
            verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            WebCheckbox(value == "true", null, enabled = usable, size = 12.dp)
            Text(field.label, fontSize = size, lineHeight = size * 1.5f, color = web.twText(Tw.gray300))
        }
        else -> Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(field.label, fontSize = size, lineHeight = size * 1.5f, color = label)
            when (field.kind) {
                GenField.Kind.Select -> WebSelect(
                    value, field.options.map { WebOption(it.first, it.second) }, { onChange(field.key, it) },
                    enabled = usable, fontSize = size, contentPadding = PaddingValues(horizontal = 8.dp, vertical = 4.dp),
                    contentDescription = field.label, disabledValues = field.disabledValues,
                )
                GenField.Kind.Range -> SpeedRange(field, value, usable, note, onChange)
                else -> PanelInput(field, value, size, usable) { onChange(field.key, it) }
            }
            if (field.suffix.isNotEmpty()) Text(field.suffix, fontSize = size, color = web.twText(Tw.gray500))
        }
    }
}

/** `bg-gray-700 border border-gray-600 rounded px-2 py-1` number / text input with its `placeholder`. */
@Composable
private fun PanelInput(field: GenField, value: String, size: TextUnit, enabled: Boolean, onChange: (String) -> Unit) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(4.dp)
    val text = web.twText(Tw.white)
    val style = TextStyle(fontSize = size, lineHeight = size * 1.5f, color = text, fontFamily = WebFonts.sans)
    val invalid = field.error(value) != null
    BasicTextField(
        value, onChange, enabled = enabled, singleLine = true, textStyle = style, cursorBrush = SolidColor(text),
        keyboardOptions = KeyboardOptions(keyboardType = if (field.kind == GenField.Kind.Number) KeyboardType.Decimal else KeyboardType.Text),
        modifier = Modifier.then(if (field.width > 0) Modifier.width(field.width.dp) else Modifier.widthIn(min = 64.dp))
            .alpha(if (enabled) 1f else 0.5f).clip(shape).background(web.twBg(Tw.gray700))
            .border(1.dp, if (invalid) Tw.red500 else web.twBorder(Tw.gray600), shape)
            .semantics { contentDescription = field.label },
        decorationBox = { inner ->
            Box(Modifier.padding(horizontal = 8.dp, vertical = 4.dp)) {
                if (value.isEmpty() && field.placeholder.isNotEmpty()) Text(field.placeholder, style = style.copy(color = Tw.gray400), maxLines = 1)
                inner()
            }
        },
    )
}

/** `input[type=range].accent-blue-500` with the `1.00x` label and the provider note. */
@Composable
private fun SpeedRange(field: GenField, value: String, enabled: Boolean, note: String, onChange: (String, String) -> Unit) {
    val web = LocalWebPalette.current
    val min = (field.min ?: 0.25).toFloat()
    val max = (field.max ?: 4.0).toFloat()
    val current = (value.toFloatOrNull() ?: 1f).coerceIn(min, max)
    val steps = ((max - min) / field.step.toFloat()).toInt() - 1
    Slider(
        current, { onChange(field.key, String.format(Locale.ROOT, "%.2f", it)) }, enabled = enabled,
        valueRange = min..max, steps = steps.coerceAtLeast(0),
        colors = SliderDefaults.colors(thumbColor = Tw.blue500, activeTrackColor = Tw.blue500, inactiveTrackColor = web.twBg(Tw.gray600),
            disabledThumbColor = Tw.gray500, disabledActiveTrackColor = Tw.gray500, activeTickColor = Color.Transparent, inactiveTickColor = Color.Transparent,
            disabledActiveTickColor = Color.Transparent, disabledInactiveTickColor = Color.Transparent),
        modifier = Modifier.width(129.dp).height(24.dp),
    )
    Text(String.format(Locale.ROOT, "%.2fx", current), fontSize = 12.sp, fontFamily = FontFamily.Monospace, color = web.twText(Tw.gray200))
    if (note.isNotEmpty()) Text(note, fontSize = 12.sp, color = web.twText(Tw.gray500))
}

@Composable
private fun LimitsNote(panel: GenPanel) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(8.dp)
    Column(
        Modifier.padding(top = 8.dp).fillMaxWidth().clip(shape).background(web.twBg(Tw.gray900, 0.4f)).border(1.dp, web.twBorder(Tw.gray700), shape)
            .padding(horizontal = 12.dp, vertical = 8.dp),
    ) {
        Text(panel.limitsTitle, fontSize = 10.sp, lineHeight = 15.sp, fontWeight = FontWeight.Bold, color = web.twText(Tw.gray300),
            modifier = Modifier.padding(bottom = 4.dp))
        panel.limits.forEach { Text(it, fontSize = 10.sp, lineHeight = 15.sp, color = web.twText(Tw.gray400)) }
    }
}

/** `#lyria-realtime-studio-bar`. */
@Composable
private fun LyriaStudioBar(onOpen: () -> Unit) {
    val web = LocalWebPalette.current
    FlowRow(
        panelModifier(web.twBorder(Tw.purple700, 0.5f)),
        horizontalArrangement = Arrangement.spacedBy(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp),
        itemVerticalAlignment = Alignment.CenterVertically,
    ) {
        FaIcon(R.drawable.fa_solid_music, null, size = 12.dp, tint = web.twText(Tw.purple300))
        Text("Lyria RealTime はリアルタイム生成スタジオで使えます。プロンプトを入力して送信するとスタジオが開き、生成しながら指示・設定を変更できます。",
            fontSize = 12.sp, lineHeight = 16.sp, color = web.twText(Tw.purple200), modifier = Modifier.weight(1f, fill = false))
        Text("スタジオを開く", fontSize = 11.sp, lineHeight = 16.sp, fontWeight = FontWeight.Bold, color = Color.White,
            modifier = Modifier.clip(RoundedCornerShape(4.dp)).background(Tw.purple600).clickable(role = Role.Button, onClick = onOpen)
                .padding(horizontal = 12.dp, vertical = 6.dp))
    }
}
