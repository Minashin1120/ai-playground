package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

// Native equivalents of the Web design tokens in chat.custom.css.
private val WebDarkColors = darkColorScheme(
    primary = Color(0xFF0DD4BF), onPrimary = Color(0xFF031413),
    primaryContainer = Color(0xFF0A403D), onPrimaryContainer = Color(0xFFA5F7ED),
    secondary = Color(0xFFF59E0B), onSecondary = Color(0xFF1F1303),
    secondaryContainer = Color(0xFF1B263C), onSecondaryContainer = Color(0xFFEEF1F7),
    tertiary = Color(0xFF818CF8), onTertiary = Color(0xFF070B16),
    tertiaryContainer = Color(0xFF18203B), onTertiaryContainer = Color(0xFFC7D2FE),
    background = Color(0xFF05070F), onBackground = Color(0xFFEEF1F7),
    surface = Color(0xFF0C1224), onSurface = Color(0xFFEEF1F7),
    surfaceVariant = Color(0xFF090E1C), onSurfaceVariant = Color(0xFF8B95A8),
    surfaceContainerLowest = Color(0xFF05070F),
    surfaceContainerLow = Color(0xFF090E1C),
    surfaceContainer = Color(0xFF0C1224),
    surfaceContainerHigh = Color(0xFF10182C),
    surfaceContainerHighest = Color(0xFF162037),
    outline = Color(0xFF1A2338), outlineVariant = Color(0xFF202B43),
    error = Color(0xFFF9708D), errorContainer = Color(0xFF451827), onErrorContainer = Color(0xFFFFD9E2),
)

private val WebLightColors = lightColorScheme(
    primary = Color(0xFF087E76), onPrimary = Color.White,
    primaryContainer = Color(0xFFCCFBF1), onPrimaryContainer = Color(0xFF043A36),
    secondary = Color(0xFF9A5B00), onSecondary = Color.White,
    secondaryContainer = Color(0xFFE8F0F0), onSecondaryContainer = Color(0xFF172326),
    tertiary = Color(0xFF4F46E5), onTertiary = Color.White,
    tertiaryContainer = Color(0xFFE0E7FF), onTertiaryContainer = Color(0xFF25205F),
    background = Color(0xFFF6F8FC), onBackground = Color(0xFF131C2E),
    surface = Color(0xFFFFFFFF), onSurface = Color(0xFF131C2E),
    surfaceVariant = Color(0xFFF1F5FA), onSurfaceVariant = Color(0xFF5C6779),
    surfaceContainerLowest = Color(0xFFFFFFFF),
    surfaceContainerLow = Color(0xFFF7F9FC),
    surfaceContainer = Color(0xFFF1F5FA),
    surfaceContainerHigh = Color(0xFFEEF2F9),
    surfaceContainerHighest = Color(0xFFE7EDF6),
    outline = Color(0xFFD9E1EC), outlineVariant = Color(0xFFE2E7EF),
    error = Color(0xFFB4233E), errorContainer = Color(0xFFFFE4E8), onErrorContainer = Color(0xFF681326),
)

// Japanese and code-heavy answers need a little more line height than Material's default.
private val DefaultTypography = Typography()
private val WebTypography = Typography(
    headlineLarge = DefaultTypography.headlineLarge.copy(fontSize = 32.sp, lineHeight = 40.sp),
    headlineMedium = DefaultTypography.headlineMedium.copy(fontSize = 25.sp, lineHeight = 32.sp),
    headlineSmall = DefaultTypography.headlineSmall.copy(fontSize = 20.sp, lineHeight = 27.sp),
    titleLarge = DefaultTypography.titleLarge.copy(fontSize = 18.sp, lineHeight = 25.sp),
    titleMedium = DefaultTypography.titleMedium.copy(fontSize = 15.sp, lineHeight = 22.sp),
    bodyLarge = DefaultTypography.bodyLarge.copy(fontSize = 16.sp, lineHeight = 25.sp),
    bodyMedium = DefaultTypography.bodyMedium.copy(fontSize = 14.sp, lineHeight = 22.sp),
    bodySmall = DefaultTypography.bodySmall.copy(fontSize = 12.sp, lineHeight = 19.sp),
    labelLarge = DefaultTypography.labelLarge.copy(fontSize = 13.sp),
    labelMedium = DefaultTypography.labelMedium.copy(fontSize = 11.sp),
    labelSmall = DefaultTypography.labelSmall.copy(fontSize = 10.sp),
)

// Rounded corners mirror the Web panel, card and bubble radii.
private val WebShapes = Shapes(
    extraSmall = RoundedCornerShape(8.dp),
    small = RoundedCornerShape(12.dp),
    medium = RoundedCornerShape(16.dp),
    large = RoundedCornerShape(20.dp),
    extraLarge = RoundedCornerShape(26.dp),
)

/** Shared layout tokens for the phone/tablet navigation split and dialogs. */
enum class PlaygroundLayoutClass {
    Phone,
    LandscapePhone,
    Tablet,
}

internal fun playgroundLayoutClass(width: androidx.compose.ui.unit.Dp, height: androidx.compose.ui.unit.Dp): PlaygroundLayoutClass = when {
    width >= 768.dp -> PlaygroundLayoutClass.Tablet
    width > height -> PlaygroundLayoutClass.LandscapePhone
    else -> PlaygroundLayoutClass.Phone
}

object PlaygroundDimens {
    val breakpoint = 768.dp
    val sidePane = 264.dp
    val drawerPane = 264.dp
    val contentMax = 768.dp
    val messageMax = 720.dp
    val phoneHorizontalPadding = 12.dp
    val conversationHorizontalPadding = 16.dp
    val panelRadius = 20.dp
    val cardRadius = 16.dp
    val controlRadius = 12.dp
}

@Composable
fun PlaygroundTheme(darkTheme: Boolean = isSystemInDarkTheme(), themeColor: String? = null, liquidGlass: Boolean = false, content: @Composable () -> Unit) {
    val accent = themeColor?.trim()?.removePrefix("#")?.takeIf { it.matches(Regex("[0-9a-fA-F]{6}")) }
        ?.let { runCatching { Color(android.graphics.Color.parseColor("#$it")) }.getOrNull() }
    val scheme = if (darkTheme) WebDarkColors else WebLightColors
    val glass = if (liquidGlass) scheme.copy(
        surface = scheme.surface.copy(alpha = 0.86f),
        surfaceContainer = scheme.surfaceContainer.copy(alpha = 0.82f),
        surfaceContainerHigh = scheme.surfaceContainerHigh.copy(alpha = 0.88f),
    ) else scheme
    val themed = accent?.let { glass.copy(primary = it, primaryContainer = it.copy(alpha = if (darkTheme) .22f else .16f)) } ?: glass
    MaterialTheme(
        colorScheme = themed,
        typography = WebTypography,
        shapes = WebShapes,
        content = content,
    )
}
