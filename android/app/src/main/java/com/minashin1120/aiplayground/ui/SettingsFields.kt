package com.minashin1120.aiplayground.ui

import androidx.annotation.DrawableRes
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/*
 * Form controls inside the Web settings modal (`#settings-modal` input/select/label rules and the
 * Tailwind classes of templates/chat/overlay_settings.html). Values follow the phone layout.
 */

/** `label span` / checkbox label: 12px semibold, `rgba(148,163,184,.95)` (light `#131c2e`). */
@Composable
internal fun settingsLabelColor(radio: Boolean = false): Color {
    val web = LocalWebPalette.current
    return when {
        !web.isLight -> Color(148, 163, 184).copy(alpha = 0.95f)
        radio -> Color(63, 74, 92)
        else -> web.text
    }
}

/** `text-[10px] text-gray-500` description (rendered at 11px on phones). */
@Composable
internal fun SettingsDesc(text: String, modifier: Modifier = Modifier, color: Color? = null) {
    val web = LocalWebPalette.current
    Text(text, fontSize = 11.sp, lineHeight = 17.05.sp, color = color ?: if (web.isLight) Color(92, 103, 121) else Color(139, 149, 168),
        modifier = modifier)
}

/** `label.text-xs.text-gray-500.block`: the field caption above an input. */
@Composable
internal fun SettingsFieldLabel(text: String, modifier: Modifier = Modifier) {
    val web = LocalWebPalette.current
    Text(text, fontSize = 12.sp, lineHeight = 18.6.sp, fontWeight = FontWeight.SemiBold, letterSpacing = 0.24.sp,
        color = if (web.isLight) Color(92, 103, 121) else Color(139, 149, 168), modifier = modifier)
}

/** `<label class="flex items-center gap-2"><input type="checkbox"><span>…</span></label>`. */
@Composable
internal fun SettingsCheck(
    label: String,
    checked: Boolean,
    onChange: (Boolean) -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    labelColor: Color? = null,
    fontSize: TextUnit = 12.sp,
    boxSize: Dp = 20.dp,
) {
    Row(
        modifier.alpha(if (enabled) 1f else 0.5f).toggleable(checked, enabled = enabled, role = Role.Checkbox, onValueChange = onChange),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        WebCheckbox(checked, onCheckedChange = null, size = boxSize)
        Text(label, fontSize = fontSize, lineHeight = 16.sp, fontWeight = FontWeight.SemiBold, letterSpacing = 0.24.sp,
            color = labelColor ?: settingsLabelColor())
    }
}

/** Chrome's native radio with `accent-cyan-500`. */
@Composable
internal fun SettingsRadio(label: String, selected: Boolean, onSelect: () -> Unit, modifier: Modifier = Modifier) {
    val web = LocalWebPalette.current
    val accent = Tw.cyan500
    Row(
        modifier.selectable(selected, role = Role.RadioButton, onClick = onSelect),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Box(
            Modifier.size(13.dp).clip(CircleShape)
                .background(if (selected) accent else Color.White)
                .border(1.dp, if (selected) accent else if (web.isLight) Color(118, 118, 118) else Color(148, 163, 184), CircleShape)
                .padding(if (selected) 2.5.dp else 0.dp),
            contentAlignment = Alignment.Center,
        ) {
            if (selected) Box(Modifier.fillMaxSize().clip(CircleShape).background(Color.White).padding(2.dp)) {
                Box(Modifier.fillMaxSize().clip(CircleShape).background(accent))
            }
        }
        Text(label, fontSize = 12.sp, lineHeight = 16.sp, fontWeight = FontWeight.SemiBold, letterSpacing = 0.24.sp,
            color = settingsLabelColor(radio = true))
    }
}

/** Colors of `#settings-modal input, select, textarea`. */
internal data class SettingsFieldColors(val background: Color, val border: Color, val text: Color, val placeholder: Color)

@Composable
internal fun settingsFieldColors(deep: Boolean = false): SettingsFieldColors {
    val web = LocalWebPalette.current
    return if (web.isLight) SettingsFieldColors(
        background = if (deep) Color.White else Color(15, 23, 42).copy(alpha = 0.04f),
        border = Color(15, 23, 42).copy(alpha = 0.08f), text = web.text, placeholder = Color(92, 103, 121),
    ) else SettingsFieldColors(
        background = if (deep) Color(3, 7, 18) else Color(5, 8, 16).copy(alpha = 0.45f),
        border = Color.White.copy(alpha = 0.08f), text = Color(238, 241, 247), placeholder = Color(139, 149, 168),
    )
}

/** `<select>` in a settings card (13px, 12px radius, 9/10 padding). */
@Composable
internal fun SettingsSelect(
    value: String,
    options: List<WebOption>,
    onSelect: (String) -> Unit,
    modifier: Modifier = Modifier,
    fillWidth: Boolean = true,
    enabled: Boolean = true,
    fontSize: TextUnit = 13.sp,
) {
    val colors = settingsFieldColors()
    WebSelect(
        value = value, options = options, onSelect = onSelect, modifier = modifier, enabled = enabled,
        fontSize = fontSize, background = colors.background, borderColor = colors.border, textColor = colors.text,
        shape = RoundedCornerShape(12.dp), contentPadding = PaddingValues(horizontal = 10.dp, vertical = 11.dp),
        fillWidth = fillWidth,
    )
}

/** `<input>` / `<textarea>` in a settings card. */
@Composable
internal fun SettingsTextField(
    value: String,
    onChange: (String) -> Unit,
    modifier: Modifier = Modifier,
    placeholder: String = "",
    password: Boolean = false,
    number: Boolean = false,
    minHeight: Dp = 0.dp,
    singleLine: Boolean = true,
    readOnly: Boolean = false,
    mono: Boolean = false,
    fontSize: TextUnit = 13.sp,
    deep: Boolean = false,
    textColor: Color? = null,
) {
    val colors = settingsFieldColors(deep)
    val shape = RoundedCornerShape(12.dp)
    val style = TextStyle(
        fontSize = fontSize, lineHeight = fontSize * 1.45f, color = textColor ?: colors.text,
        fontFamily = if (mono) FontFamily.Monospace else WebFonts.sans,
    )
    BasicTextField(
        value = value,
        onValueChange = onChange,
        readOnly = readOnly,
        singleLine = singleLine,
        textStyle = style,
        cursorBrush = SolidColor(colors.text),
        visualTransformation = if (password) PasswordVisualTransformation() else VisualTransformation.None,
        keyboardOptions = KeyboardOptions(keyboardType = when {
            password -> KeyboardType.Password
            number -> KeyboardType.Number
            else -> KeyboardType.Text
        }),
        modifier = modifier.heightIn(min = minHeight).clip(shape).background(colors.background).border(1.dp, colors.border, shape),
        decorationBox = { inner ->
            Box(Modifier.padding(horizontal = 12.dp, vertical = 10.dp)) {
                if (value.isEmpty() && placeholder.isNotEmpty()) Text(placeholder, style = style.copy(color = colors.placeholder))
                inner()
            }
        },
    )
}

/** Small `bg-gray-700 … rounded text-xs font-bold` button used inside cards (リセット, 既定に戻す, …). */
@Composable
internal fun SettingsSmallButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    tone: SettingsButtonTone = SettingsButtonTone.Gray,
    fontSize: TextUnit = 12.sp,
    enabled: Boolean = true,
    @DrawableRes icon: Int? = null,
    fill: Boolean = false,
) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(if (fontSize.value <= 10f) 8.dp else 10.dp)
    val bg = tone.background(web)
    Row(
        modifier.then(if (fill) Modifier.fillMaxWidth() else Modifier).alpha(if (enabled) 1f else 0.5f).clip(shape).background(bg)
            .border(1.dp, if (tone == SettingsButtonTone.Gray) web.lineStrong else Color.Transparent, shape)
            .clickable(enabled = enabled, role = Role.Button, onClick = onClick)
            .padding(horizontal = if (fontSize.value <= 10f) 8.dp else 12.dp, vertical = if (fontSize.value <= 10f) 4.dp else 7.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(4.dp, Alignment.CenterHorizontally),
    ) {
        val fg = if (tone == SettingsButtonTone.Gray) web.text else Color.White
        if (icon != null) FaIcon(icon, null, size = fontSize.value.dp, tint = fg)
        Text(text, fontSize = fontSize, fontWeight = FontWeight.Bold, color = fg)
    }
}

internal enum class SettingsButtonTone {
    Gray, Blue, Emerald, Orange, Red, Cyan, Purple, Green;

    fun background(web: WebPalette): Color = when (this) {
        Gray -> if (web.isLight) Color(15, 23, 42).copy(alpha = 0.06f) else Color.White.copy(alpha = 0.06f)
        Blue -> web.theme.t600
        Emerald -> Tw.emerald700
        Orange -> Tw.orange700
        Red -> Tw.red700
        Cyan -> Tw.cyan700
        Purple -> Tw.purple600
        Green -> Tw.green600
    }
}

/** `rounded border border-gray-700 bg-gray-950/50 p-3` sub box inside a card. */
@Composable
internal fun SettingsSubBox(modifier: Modifier = Modifier, content: @Composable ColumnScope.() -> Unit) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(10.dp)
    Column(
        modifier.fillMaxWidth().clip(shape)
            .background(if (web.isLight) Color(15, 23, 42).copy(alpha = 0.03f) else Color(3, 7, 18).copy(alpha = 0.5f))
            .border(1.dp, if (web.isLight) Color(15, 23, 42).copy(alpha = 0.10f) else web.line, shape)
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp),
        content = content,
    )
}

/** A row with a bold title, description and the Web switch on the right (Light mode, Liquid Glass, debug). */
@Composable
internal fun SettingsSwitchRow(
    title: String,
    description: String,
    checked: Boolean,
    onChange: (Boolean) -> Unit,
    modifier: Modifier = Modifier,
    @DrawableRes icon: Int? = null,
    iconTint: Color = Color.Unspecified,
    accent: Color? = null,
) {
    val web = LocalWebPalette.current
    Row(modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(16.dp)) {
        Column(Modifier.weight(1f)) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                if (icon != null) FaIcon(icon, null, size = 12.dp, tint = iconTint)
                Text(title, fontSize = 12.sp, fontWeight = FontWeight.Bold, color = if (web.isLight) web.text else Color.White)
            }
            SettingsDesc(description, Modifier.padding(top = 4.dp))
        }
        WebToggle(checked, onChange, activeColor = accent)
    }
}

/** A dashed-free divider: `border-t border-gray-700 pt-3`. */
internal fun Modifier.settingsTopRule(color: Color, paddingTop: Dp = 12.dp): Modifier =
    drawBehind { drawLine(color, androidx.compose.ui.geometry.Offset(0f, 0f), androidx.compose.ui.geometry.Offset(size.width, 0f), 1.dp.toPx()) }
        .padding(top = paddingTop)
