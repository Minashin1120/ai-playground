package com.minashin1120.aiplayground.ui

import androidx.annotation.DrawableRes
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.isSpecified
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.luminance
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.R

/*
 * Building blocks that reproduce the Web chat UI (Tailwind utilities + chat.custom.v*.css).
 * Each component names the Web markup or selector it mirrors; keep values in sync with the stylesheet.
 */

/**
 * `<i class="fas fa-…">` at the given font size. Font Awesome glyphs are one em tall and as wide as
 * their own aspect ratio; `fixedWidth` mirrors `fa-fw` (1.25em wide, centered).
 */
@Composable
internal fun FaIcon(
    @DrawableRes icon: Int,
    contentDescription: String?,
    modifier: Modifier = Modifier,
    size: Dp = 14.dp,
    tint: Color = LocalContentColor.current,
    fixedWidth: Boolean = false,
) {
    val painter = painterResource(icon)
    val intrinsic = painter.intrinsicSize
    val ratio = if (intrinsic.isSpecified && intrinsic.height > 0f) intrinsic.width / intrinsic.height else 1f
    val width = if (fixedWidth) size * 1.25f else size * ratio
    Icon(painter, contentDescription, modifier.size(width = width, height = size), tint = tint)
}

/**
 * Settings switch: `w-11 h-6 bg-gray-700 rounded-full` with a 20px white knob that slides by its own
 * width, `peer-checked:bg-[var(--theme-600)]`, `transition-all` (150ms).
 */
@Composable
internal fun WebToggle(
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    contentDescription: String? = null,
) {
    val web = LocalWebPalette.current
    val reduce = LocalReduceMotion.current
    val track by animateColorAsState(
        if (checked) web.theme.t600 else web.twBg(Tw.gray700),
        motionTween(reduce, 150), label = "toggle track",
    )
    val knobOffset by animateDpAsState(if (checked) 20.dp else 0.dp, motionTween(reduce, 150), label = "toggle knob")
    Box(
        modifier
            .size(width = 44.dp, height = 24.dp)
            .alpha(if (enabled) 1f else 0.5f)
            .clip(CircleShape)
            .background(track)
            .toggleable(checked, enabled = enabled, role = Role.Switch, onValueChange = onCheckedChange)
            .then(if (contentDescription != null) Modifier.semanticsLabel(contentDescription) else Modifier),
    ) {
        Box(
            Modifier
                .offset(x = 2.dp + knobOffset, y = 2.dp)
                .size(20.dp)
                .clip(CircleShape)
                .background(Tw.white),
        )
    }
}

/**
 * Native `<input type="checkbox" class="accent-… w-3 h-3">` as Chrome draws it: a small rounded square
 * with a gray border, filled with the accent color and a check mark when checked.
 */
@Composable
internal fun WebCheckbox(
    checked: Boolean,
    onCheckedChange: ((Boolean) -> Unit)?,
    accent: Color,
    modifier: Modifier = Modifier,
    size: Dp = 12.dp,
    enabled: Boolean = true,
) {
    val web = LocalWebPalette.current
    val checkColor = if (accent.luminance() > 0.55f) Color(0xFF101010) else Tw.white
    val base = modifier
        .size(size)
        .alpha(if (enabled) 1f else 0.5f)
        .clip(RoundedCornerShape(2.dp))
        .background(if (checked) accent else Tw.white)
        .border(1.dp, if (checked) accent else if (web.isLight) Color(0xFF767676) else Color(0xFF858585), RoundedCornerShape(2.dp))
    val interactive = if (onCheckedChange != null) {
        base.toggleable(checked, enabled = enabled, role = Role.Checkbox, onValueChange = onCheckedChange)
    } else base
    Box(interactive, contentAlignment = Alignment.Center) {
        if (checked) {
            Canvas(Modifier.size(size * 0.72f)) {
                val path = Path().apply {
                    moveTo(this@Canvas.size.width * 0.12f, this@Canvas.size.height * 0.52f)
                    lineTo(this@Canvas.size.width * 0.40f, this@Canvas.size.height * 0.80f)
                    lineTo(this@Canvas.size.width * 0.90f, this@Canvas.size.height * 0.22f)
                }
                drawPath(path, checkColor, style = Stroke(width = this.size.width * 0.18f, cap = StrokeCap.Round, join = StrokeJoin.Round))
            }
        }
    }
}

/** An `<option>`: the stored value and the label shown by the Web select. */
internal data class WebOption(val value: String, val label: String)

internal fun webOptions(vararg pairs: Pair<String, String>): List<WebOption> = pairs.map { WebOption(it.first, it.second) }

/**
 * `<select>` with Tailwind styling (`bg-gray-700 border border-gray-600 rounded px-1 py-0.5 text-xs text-white`
 * by default). The value is the stored option value; the Web label is shown. Unknown values show as-is.
 */
@Composable
internal fun WebSelect(
    value: String,
    options: List<WebOption>,
    onSelect: (String) -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    fontSize: TextUnit = 12.sp,
    background: Color? = null,
    borderColor: Color? = null,
    textColor: Color? = null,
    shape: Shape = RoundedCornerShape(4.dp),
    contentPadding: PaddingValues = PaddingValues(horizontal = 4.dp, vertical = 2.dp),
    fillWidth: Boolean = false,
    contentDescription: String? = null,
) {
    val web = LocalWebPalette.current
    var open by remember { mutableStateOf(false) }
    val label = options.firstOrNull { it.value == value }?.label ?: value
    val fg = textColor ?: web.twText(Tw.white)
    Box(modifier) {
        Row(
            Modifier
                .then(if (fillWidth) Modifier.fillMaxWidth() else Modifier)
                .alpha(if (enabled) 1f else 0.5f)
                .clip(shape)
                .background(background ?: web.twBg(Tw.gray700))
                .border(1.dp, borderColor ?: web.twBorder(Tw.gray600), shape)
                .clickable(enabled = enabled, role = Role.DropdownList) { open = true }
                .then(if (contentDescription != null) Modifier.semanticsLabel(contentDescription) else Modifier)
                .padding(contentPadding),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            Text(
                label, color = fg, fontSize = fontSize, maxLines = 1, overflow = TextOverflow.Ellipsis,
                modifier = if (fillWidth) Modifier.weight(1f) else Modifier,
            )
            SelectArrow(fg, (fontSize.value * 0.6f).dp)
        }
        DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
            options.forEach { option ->
                DropdownMenuItem(
                    text = {
                        Text(
                            option.label,
                            fontWeight = if (option.value == value) FontWeight.Bold else FontWeight.Normal,
                            fontSize = 14.sp,
                        )
                    },
                    onClick = { open = false; if (option.value != value) onSelect(option.value) },
                )
            }
        }
    }
}

/** The small down arrow Chrome draws inside a styled `<select>`. */
@Composable
private fun SelectArrow(color: Color, size: Dp) {
    Canvas(Modifier.size(size)) {
        val w = this.size.width
        val h = this.size.height
        val path = Path().apply {
            moveTo(w * 0.1f, h * 0.3f)
            lineTo(w * 0.5f, h * 0.72f)
            lineTo(w * 0.9f, h * 0.3f)
        }
        drawPath(path, color, style = Stroke(width = w * 0.16f, cap = StrokeCap.Round, join = StrokeJoin.Round))
    }
}

/** Visual variants of the Web pill controls. */
internal data class WebPillStyle(
    val background: Color,
    val border: Color,
    val content: Color,
    val shape: Shape = CircleShape,
    val padding: PaddingValues,
    val fontSize: TextUnit,
    val fontWeight: FontWeight = FontWeight.Normal,
    val minHeight: Dp = 0.dp,
)

/**
 * `.composer-opt` detail chip (`body:not(.minimal-prompt-mode)`): 10.5px text, `padding: .15rem .5rem`,
 * a translucent border, and the theme tint while its checkbox is checked.
 */
@Composable
internal fun composerOptStyle(checked: Boolean): WebPillStyle {
    val web = LocalWebPalette.current
    return if (checked) WebPillStyle(
        background = web.theme.rgb(0.10f),
        border = web.theme.rgb(0.35f),
        content = web.theme200,
        padding = PaddingValues(horizontal = 8.dp, vertical = 2.4.dp),
        fontSize = 10.5.sp,
    ) else WebPillStyle(
        background = if (web.isLight) Color(15, 23, 42).copy(alpha = 0.04f) else Color(10, 14, 26).copy(alpha = 0.5f),
        border = if (web.isLight) Color(15, 23, 42).copy(alpha = 0.10f) else Tw.white.copy(alpha = 0.07f),
        content = if (web.isLight) web.muted else Color(0xFFA8B3C7),
        padding = PaddingValues(horizontal = 8.dp, vertical = 2.4.dp),
        fontSize = 10.5.sp,
    )
}

/**
 * `.composer-chip` (Canvas / Coding / Batch): 11px semibold pill, `min-height: 1.9rem`,
 * `padding: .32rem .62rem`. `tone` supplies the per-chip border/background/text of the checked state.
 */
@Composable
internal fun composerChipStyle(checked: Boolean, tone: ChipTone): WebPillStyle {
    val web = LocalWebPalette.current
    return WebPillStyle(
        background = if (checked) tone.checkedBackground else if (web.isLight) Color(15, 23, 42).copy(alpha = 0.04f) else Color(14, 22, 41).copy(alpha = 0.58f),
        border = if (checked) tone.checkedBorder else Color(148, 163, 184).copy(alpha = 0.16f),
        content = web.twText(tone.text),
        padding = PaddingValues(horizontal = 10.dp, vertical = 5.dp),
        fontSize = 11.sp,
        fontWeight = FontWeight.SemiBold,
        minHeight = 30.dp,
    )
}

/** Checked colors of a `.composer-chip` (e.g. `#canvas-mode-container:has(input:checked)`). */
internal data class ChipTone(val text: Color, val checkedBorder: Color, val checkedBackground: Color, val accent: Color)

internal object ChipTones {
    val Canvas = ChipTone(Tw.cyan200, Color(34, 211, 238).copy(alpha = 0.55f), Color(8, 51, 68).copy(alpha = 0.45f), Tw.cyan400)
    val Coding = ChipTone(Tw.emerald200, Tw.emerald500.copy(alpha = 0.30f), Tw.emerald900.copy(alpha = 0.15f), Tw.emerald400)
    val Batch = ChipTone(Tw.violet200, Tw.violet500.copy(alpha = 0.40f), Tw.violet900.copy(alpha = 0.20f), Tw.violet400)
}

/** A pill container shared by chips and option groups. */
@Composable
internal fun WebPill(
    style: WebPillStyle,
    modifier: Modifier = Modifier,
    onClick: (() -> Unit)? = null,
    enabled: Boolean = true,
    content: @Composable RowScope.() -> Unit,
) {
    Row(
        modifier
            .defaultMinSize(minHeight = style.minHeight)
            .alpha(if (enabled) 1f else 0.5f)
            .clip(style.shape)
            .background(style.background)
            .border(1.dp, style.border, style.shape)
            .then(if (onClick != null) Modifier.clickable(enabled = enabled, onClick = onClick) else Modifier)
            .padding(style.padding),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        CompositionLocalProvider(
            LocalContentColor provides style.content,
            androidx.compose.material3.LocalTextStyle provides TextStyle(
                fontSize = style.fontSize,
                fontWeight = style.fontWeight,
                fontFamily = WebFonts.sans,
                lineHeight = style.fontSize * 1.25f,
                color = style.content,
            ),
        ) { content() }
    }
}

/** A checkbox chip: `<label class="composer-opt"><input type="checkbox" class="accent-…"><span>Label</span></label>`. */
@Composable
internal fun WebCheckChip(
    label: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit,
    accent: Color,
    style: WebPillStyle,
    modifier: Modifier = Modifier,
    labelColor: Color? = null,
    enabled: Boolean = true,
    trailing: (@Composable RowScope.() -> Unit)? = null,
) {
    val web = LocalWebPalette.current
    WebPill(style, modifier.toggleable(checked, enabled = enabled, role = Role.Checkbox, onValueChange = onCheckedChange), enabled = enabled) {
        WebCheckbox(checked, onCheckedChange = null, accent = accent)
        Text(label, color = labelColor?.let(web::twText) ?: style.content, maxLines = 1)
        trailing?.invoke(this)
    }
}

/** `.settings-card` with its `.settings-card-title` (13px bold, `--theme-300`; danger cards use `#fda4af`). */
@Composable
internal fun WebSettingsCard(
    title: String?,
    modifier: Modifier = Modifier,
    danger: Boolean = false,
    compact: Boolean = false,
    content: @Composable ColumnScope.() -> Unit,
) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(16.dp)
    // Dark: `linear-gradient(white .045 → .018)` over `rgba(10,16,30,.78)`; phones use `rgba(8,12,24,.95)`.
    val base = when {
        web.isLight -> Color.Transparent
        compact -> Color(8, 12, 24).copy(alpha = 0.95f)
        else -> Color(10, 16, 30).copy(alpha = 0.78f)
    }
    val sheen = if (web.isLight) {
        Brush.verticalGradient(listOf(Tw.white.copy(alpha = 0.9f), Color(248, 250, 252).copy(alpha = 0.94f)))
    } else {
        Brush.verticalGradient(listOf(Tw.white.copy(alpha = 0.045f), Tw.white.copy(alpha = 0.018f)))
    }
    Column(
        modifier
            .fillMaxWidth()
            .clip(shape)
            .background(base)
            .background(sheen)
            .border(1.dp, if (web.isLight) Color(15, 23, 42).copy(alpha = 0.08f) else Tw.white.copy(alpha = 0.08f), shape)
            .padding(horizontal = 18.dp, vertical = 16.dp),
    ) {
        if (title != null) {
            Text(
                title,
                color = if (danger) Tw.rose300 else web.theme300,
                fontSize = if (compact) 12.sp else 13.sp,
                fontWeight = FontWeight.Bold,
                letterSpacing = 0.02.em,
                modifier = Modifier.padding(bottom = 10.dp),
            )
        }
        CompositionLocalProvider(LocalContentColor provides web.text) { content() }
    }
}

/** Button variants used across Web modals. */
internal enum class WebButtonVariant {
    /** `.settings-btn-primary`: theme gradient, inverse text, glow. */
    Primary,

    /** `.settings-btn-ghost`: muted text, transparent. */
    Ghost,

    /** `bg-gray-700` inside a settings card: faint white fill with `--line-strong` border. */
    Gray,

    /** `bg-emerald-700` / `bg-green-600` inside a settings card. */
    Success,

    /** `bg-red-600` destructive buttons. */
    Danger,
}

@Composable
internal fun WebButton(
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    variant: WebButtonVariant = WebButtonVariant.Gray,
    enabled: Boolean = true,
    radius: Dp = 12.dp,
    contentPadding: PaddingValues = PaddingValues(horizontal = 16.dp, vertical = 9.dp),
    fontSize: TextUnit = 13.sp,
    content: @Composable RowScope.() -> Unit,
) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(radius)
    val (background, border, fg) = when (variant) {
        WebButtonVariant.Primary -> Triple(
            Brush.linearGradient(listOf(web.theme.t500, web.theme.t600), start = Offset.Zero, end = Offset.Infinite),
            web.theme.rgb(0.45f), web.textInverse,
        )
        WebButtonVariant.Ghost -> Triple(Brush.linearGradient(listOf(Color.Transparent, Color.Transparent)), Color.Transparent, web.muted)
        WebButtonVariant.Gray -> Triple(
            Brush.linearGradient(listOf(if (web.isLight) Color(15, 23, 42).copy(alpha = 0.06f) else Tw.white.copy(alpha = 0.06f), if (web.isLight) Color(15, 23, 42).copy(alpha = 0.06f) else Tw.white.copy(alpha = 0.06f))),
            web.lineStrong, web.text,
        )
        WebButtonVariant.Success -> Triple(
            Brush.linearGradient(listOf(Color(0xFF34D399), Color(0xFF059669)), start = Offset.Zero, end = Offset.Infinite),
            Color(52, 211, 153).copy(alpha = 0.4f), Color(0xFF06281C),
        )
        WebButtonVariant.Danger -> Triple(Brush.linearGradient(listOf(Tw.red600, Tw.red600)), Tw.red600, Tw.white)
    }
    Row(
        modifier
            .alpha(if (enabled) 1f else 0.5f)
            .clip(shape)
            .background(background)
            .border(1.dp, border, shape)
            .clickable(enabled = enabled, role = Role.Button, onClick = onClick)
            .padding(contentPadding),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(7.dp, Alignment.CenterHorizontally),
    ) {
        CompositionLocalProvider(
            LocalContentColor provides fg,
            androidx.compose.material3.LocalTextStyle provides TextStyle(fontSize = fontSize, fontWeight = FontWeight.Bold, fontFamily = WebFonts.sans, color = fg),
        ) { content() }
    }
}

/**
 * Modal header (`.settings-header`): optional 44px icon tile, 18px bold title (16px on phones),
 * 12px muted subtitle and the 38px close button, over a faint theme gradient.
 */
@Composable
internal fun WebModalHeader(
    title: @Composable () -> Unit,
    onClose: () -> Unit,
    phone: Boolean,
    modifier: Modifier = Modifier,
    @DrawableRes icon: Int? = null,
    subtitle: String? = null,
) {
    val web = LocalWebPalette.current
    Row(
        modifier
            .fillMaxWidth()
            .background(Brush.verticalGradient(listOf(web.theme.rgb(0.12f), Color.Transparent)))
            .padding(start = if (phone) 14.dp else 20.dp, end = if (phone) 14.dp else 20.dp, top = if (phone) 14.dp else 16.dp, bottom = if (phone) 8.dp else 12.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Row(Modifier.weight(1f), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(14.dp)) {
            if (icon != null) {
                val tileShape = RoundedCornerShape(14.dp)
                Box(
                    Modifier
                        .size(44.dp)
                        .clip(tileShape)
                        .background(Brush.linearGradient(listOf(web.theme.rgb(0.30f), web.theme.rgb(0.08f)), start = Offset.Zero, end = Offset.Infinite))
                        .border(1.dp, web.theme.rgb(0.30f), tileShape),
                    contentAlignment = Alignment.Center,
                ) { FaIcon(icon, null, size = 18.dp, tint = web.theme300) }
            }
            Column(Modifier.weight(1f)) {
                CompositionLocalProvider(
                    LocalContentColor provides web.text,
                    androidx.compose.material3.LocalTextStyle provides TextStyle(
                        fontSize = if (phone) 16.sp else 18.sp, fontWeight = FontWeight.Bold, fontFamily = WebFonts.sans,
                        letterSpacing = 0.01.em, color = web.text,
                    ),
                ) { title() }
                if (subtitle != null) {
                    Text(subtitle, color = web.muted, fontSize = 12.sp, modifier = Modifier.padding(top = 2.dp))
                }
            }
        }
        val closeShape = RoundedCornerShape(12.dp)
        Box(
            Modifier
                .size(38.dp)
                .clip(closeShape)
                .background(if (web.isLight) Color(15, 23, 42).copy(alpha = 0.03f) else Tw.white.copy(alpha = 0.03f))
                .border(1.dp, web.lineSoft, closeShape)
                .clickable(role = Role.Button, onClick = onClose),
            contentAlignment = Alignment.Center,
        ) { FaIcon(R.drawable.fa_solid_times, "閉じる", size = 14.dp, tint = web.muted) }
    }
}

/** Modal footer (`.settings-footer`): right-aligned buttons over a fading panel gradient with a top rule. */
@Composable
internal fun WebModalFooter(phone: Boolean, modifier: Modifier = Modifier, content: @Composable RowScope.() -> Unit) {
    val web = LocalWebPalette.current
    Column(modifier.fillMaxWidth()) {
        Spacer(Modifier.fillMaxWidth().height(1.dp).background(web.line))
        Row(
            Modifier
                .fillMaxWidth()
                .background(
                    if (web.isLight) Brush.verticalGradient(listOf(Color(248, 250, 252).copy(alpha = 0.72f), Tw.white))
                    else Brush.verticalGradient(listOf(Color(8, 12, 22).copy(alpha = 0.55f), Color(8, 12, 22).copy(alpha = 0.98f))),
                )
                .padding(start = if (phone) 12.dp else 18.dp, end = if (phone) 12.dp else 18.dp, top = if (phone) 10.dp else 12.dp, bottom = if (phone) 14.dp else 16.dp),
            horizontalArrangement = Arrangement.spacedBy(10.dp, Alignment.End),
            verticalAlignment = Alignment.CenterVertically,
            content = content,
        )
    }
}

/** Spacer helper matching Tailwind `gap-*`/`w-*` in dp. */
@Composable
internal fun HSpace(width: Dp) = Spacer(Modifier.width(width))

private fun Modifier.semanticsLabel(label: String): Modifier = semantics { contentDescription = label }
