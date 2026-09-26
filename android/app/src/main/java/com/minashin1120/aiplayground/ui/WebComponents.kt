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
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.graphics.graphicsLayer
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
    /** `peer-checked:bg-*`; the theme 600 color when null. */
    activeColor: Color? = null,
) {
    val web = LocalWebPalette.current
    val reduce = LocalReduceMotion.current
    val track by animateColorAsState(
        if (checked) activeColor ?: web.theme.t600 else web.twBg(Tw.gray700),
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
 * The Web's global checkbox (`input[type="checkbox"]` in chat.custom.v*.css): a 14px rounded square
 * with a dark fill; when checked the border turns to the theme color, a 2px theme ring appears and an
 * 8px rounded square with the theme gradient scales in. The light theme keeps a white box without the ring.
 * [accent] is kept for call sites that still pass the Tailwind `accent-*` color; the Web ignores it.
 */
@Suppress("UNUSED_PARAMETER")
@Composable
internal fun WebCheckbox(
    checked: Boolean,
    onCheckedChange: ((Boolean) -> Unit)?,
    accent: Color = Color.Unspecified,
    modifier: Modifier = Modifier,
    size: Dp = 14.dp,
    enabled: Boolean = true,
) {
    val web = LocalWebPalette.current
    val reduce = LocalReduceMotion.current
    val tick by androidx.compose.animation.core.animateFloatAsState(if (checked) 1f else 0f,
        motionTween(reduce, 180, PlaygroundMotion.WebEaseOut), label = "checkbox tick")
    val shape = RoundedCornerShape(5.dp)
    val border = when {
        web.isLight -> Color(15, 23, 42).copy(alpha = 0.22f)
        checked -> web.theme.rgb(0.55f)
        else -> Color.White.copy(alpha = 0.15f)
    }
    val ring = web.theme.rgb(0.15f)
    val base = modifier
        .size(size)
        // `box-shadow: 0 0 0 2px` ring outside the box; it does not take layout space.
        .drawBehind {
            if (checked && !web.isLight) {
                val spread = 2.dp.toPx()
                drawRoundRect(ring, topLeft = androidx.compose.ui.geometry.Offset(-spread, -spread),
                    size = androidx.compose.ui.geometry.Size(this.size.width + spread * 2, this.size.height + spread * 2),
                    cornerRadius = androidx.compose.ui.geometry.CornerRadius(7.dp.toPx(), 7.dp.toPx()))
            }
        }
        .clip(shape)
        .background(if (web.isLight) Color.White else Color(8, 14, 26).copy(alpha = 0.8f))
        .border(1.dp, border, shape)
    val interactive = if (onCheckedChange != null) {
        base.toggleable(checked, enabled = enabled, role = Role.Checkbox, onValueChange = onCheckedChange)
    } else base
    Box(interactive, contentAlignment = Alignment.Center) {
        if (tick > 0f) {
            Box(
                Modifier
                    .size(8.dp)
                    .graphicsLayer { scaleX = tick; scaleY = tick }
                    .clip(RoundedCornerShape(3.dp))
                    .background(Brush.verticalGradient(listOf(web.theme300, web.theme.t600))),
            )
        }
    }
}

/** An `<option>`: the stored value and the label shown by the Web select. */
internal data class WebOption(val value: String, val label: String, val group: String? = null)

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
    /** `<option disabled>` values: listed but not selectable. */
    disabledValues: Set<String> = emptySet(),
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
            options.forEachIndexed { index, option ->
                // `<optgroup label>` heading when the group changes.
                if (option.group != null && option.group != options.getOrNull(index - 1)?.group) {
                    Text(option.group, fontSize = 13.sp, fontWeight = FontWeight.Bold,
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp))
                }
                DropdownMenuItem(
                    text = {
                        Text(
                            option.label,
                            fontWeight = if (option.value == value) FontWeight.Bold else FontWeight.Normal,
                            fontSize = 14.sp,
                        )
                    },
                    onClick = { open = false; if (option.value != value) onSelect(option.value) },
                    enabled = option.value !in disabledValues,
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
 * `.composer-opt` detail chip (`body:not(.minimal-prompt-mode)`): 10px text, `min-height: 1.8rem`,
 * `padding: .28rem .58rem`, theme tint while checked (dark only; the light theme keeps the neutral pill).
 * `<label>` chips are 12px round, `<div>` groups (SysPrompt, Thinking, Effort, …) are fully round.
 */
@Composable
internal fun composerOptStyle(checked: Boolean, group: Boolean = false): WebPillStyle {
    val web = LocalWebPalette.current
    val shape = if (group) CircleShape else RoundedCornerShape(12.dp)
    val padding = PaddingValues(horizontal = 9.28.dp, vertical = 4.48.dp)
    return when {
        web.isLight -> WebPillStyle(Color(15, 23, 42).copy(alpha = 0.04f), Color(15, 23, 42).copy(alpha = 0.08f),
            Color(92, 103, 121), shape, padding, 10.sp, minHeight = 28.8.dp)
        checked -> WebPillStyle(web.theme.rgb(0.106f), web.theme.rgb(0.34f), web.theme200, shape, padding, 10.sp, minHeight = 28.8.dp)
        else -> WebPillStyle(Color(10, 16, 31).copy(alpha = 0.52f), Color(148, 163, 184).copy(alpha = 0.14f),
            Color(168, 179, 199), shape, padding, 10.sp, minHeight = 28.8.dp)
    }
}

/** The three `.composer-chip` toggles next to the model button. */
internal enum class ChipTone(val accent: Color) {
    Canvas(Tw.cyan400), Coding(Tw.emerald400), Batch(Tw.violet400),
}

/**
 * `.composer-chip` (Canvas / Coding / Batch): 11px semibold, `min-height: 1.9rem`, `padding: .32rem .62rem`,
 * 12px radius. Canvas keeps its teal tint; Coding and Batch are neutral in dark and tinted in light.
 */
@Composable
internal fun composerChipStyle(checked: Boolean, tone: ChipTone): WebPillStyle {
    val web = LocalWebPalette.current
    val neutralBg = Color(14, 22, 41).copy(alpha = 0.58f)
    val neutralBorder = Color(148, 163, 184).copy(alpha = 0.16f)
    val (bg, border, text) = when (tone) {
        ChipTone.Canvas -> when {
            web.isLight -> Triple(Color(6, 182, 212).copy(alpha = 0.12f),
                if (checked) Color(34, 211, 238).copy(alpha = 0.55f) else web.theme.rgb(0.35f), web.theme.t700)
            checked -> Triple(Color(8, 51, 68).copy(alpha = 0.45f), Color(34, 211, 238).copy(alpha = 0.55f), Color(165, 243, 252))
            else -> Triple(web.theme.rgb(0.12f), web.theme.rgb(0.35f), Color(207, 250, 254))
        }
        ChipTone.Coding -> Triple(
            if (web.isLight) Color(16, 185, 129).copy(alpha = 0.12f) else if (checked) Color(6, 95, 70).copy(alpha = 0.36f) else neutralBg,
            if (checked) Color(52, 211, 153).copy(alpha = 0.68f) else neutralBorder,
            if (web.isLight) Color(4, 120, 87) else Color(154, 166, 186),
        )
        ChipTone.Batch -> Triple(
            if (web.isLight) Color(139, 92, 246).copy(alpha = 0.12f) else neutralBg,
            neutralBorder,
            if (web.isLight) Color(109, 40, 217) else Color(154, 166, 186),
        )
    }
    return WebPillStyle(
        background = bg, border = border, content = text, shape = RoundedCornerShape(12.dp),
        padding = PaddingValues(horizontal = 9.92.dp, vertical = 5.12.dp),
        fontSize = 11.sp, fontWeight = FontWeight.SemiBold, minHeight = 30.4.dp,
    )
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
    WebPill(style, modifier.toggleable(checked, enabled = enabled, role = Role.Checkbox, onValueChange = onCheckedChange), enabled = enabled) {
        WebCheckbox(checked, onCheckedChange = null, accent = accent)
        Text(label, color = labelColor ?: style.content, maxLines = 1)
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
