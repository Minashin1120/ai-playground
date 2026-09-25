package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.remember
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * Material color roles filled from the Web `:root` tokens ([WebPalette]) so that Material components
 * that are still in use render with the Web surfaces, lines and text colors.
 */
private fun webColorScheme(web: WebPalette) = if (web.isLight) lightColorScheme(
    primary = web.theme.t600, onPrimary = Color.White,
    primaryContainer = web.theme.rgb(0.16f), onPrimaryContainer = web.theme.t700,
    secondary = Color(0xFF9A5B00), onSecondary = Color.White,
    secondaryContainer = Color(0xFFEEF2F9), onSecondaryContainer = web.text,
    tertiary = Tw.indigo600, onTertiary = Color.White,
    tertiaryContainer = Color(0xFFE0E7FF), onTertiaryContainer = Color(0xFF25205F),
    background = web.bg1, onBackground = web.text,
    surface = web.panel, onSurface = web.text,
    surfaceVariant = web.panel2, onSurfaceVariant = web.muted,
    surfaceContainerLowest = web.panel,
    surfaceContainerLow = web.panel2,
    surfaceContainer = web.bg3,
    surfaceContainerHigh = web.bg2,
    surfaceContainerHighest = Color(0xFFE7EDF6),
    outline = web.line, outlineVariant = Color(0xFFE2E8F1),
    error = Color(0xFFB91C1C), onError = Color.White, errorContainer = Color(0xFFFFE4E8), onErrorContainer = Color(0xFF681326),
    scrim = Color(15, 23, 42),
) else darkColorScheme(
    primary = web.theme.t500, onPrimary = web.textInverse,
    primaryContainer = web.theme.rgb(0.16f), onPrimaryContainer = web.theme.t200,
    secondary = web.accent2, onSecondary = Color(0xFF1F1303),
    secondaryContainer = Color(0xFF111A30), onSecondaryContainer = web.text,
    tertiary = Tw.indigo400, onTertiary = Color(0xFF070B16),
    tertiaryContainer = Color(0xFF18203B), onTertiaryContainer = Tw.indigo200,
    background = web.bg1, onBackground = web.text,
    surface = web.panel, onSurface = web.text,
    surfaceVariant = web.panel2, onSurfaceVariant = web.muted,
    surfaceContainerLowest = web.bg1,
    surfaceContainerLow = web.panel2,
    surfaceContainer = web.panel,
    surfaceContainerHigh = Color(0xFF0E1428),
    surfaceContainerHighest = Color(0xFF141C33),
    outline = web.lineStrong, outlineVariant = web.line,
    error = web.danger, onError = Color(0xFF2A0710), errorContainer = Color(0xFF451827), onErrorContainer = Color(0xFFFFD9E2),
    scrim = Color(3, 7, 16),
)

// Japanese and code-heavy answers need a little more line height than Material's default.
private val DefaultTypography = Typography()
private fun androidx.compose.ui.text.TextStyle.web(size: Int, line: Int) =
    copy(fontFamily = WebFonts.sans, fontSize = size.sp, lineHeight = line.sp)

/** Tailwind type scale (`text-xs` 12/16 … `text-3xl` 30/36) in the Web font. */
private val WebTypography = Typography(
    displayLarge = DefaultTypography.displayLarge.copy(fontFamily = WebFonts.sans),
    displayMedium = DefaultTypography.displayMedium.copy(fontFamily = WebFonts.sans),
    displaySmall = DefaultTypography.displaySmall.copy(fontFamily = WebFonts.sans),
    headlineLarge = DefaultTypography.headlineLarge.web(30, 36),
    headlineMedium = DefaultTypography.headlineMedium.web(24, 32),
    headlineSmall = DefaultTypography.headlineSmall.web(20, 28),
    titleLarge = DefaultTypography.titleLarge.web(18, 28),
    titleMedium = DefaultTypography.titleMedium.web(16, 24),
    titleSmall = DefaultTypography.titleSmall.web(14, 20),
    bodyLarge = DefaultTypography.bodyLarge.web(16, 24),
    bodyMedium = DefaultTypography.bodyMedium.web(14, 20),
    bodySmall = DefaultTypography.bodySmall.web(12, 16),
    labelLarge = DefaultTypography.labelLarge.web(14, 20),
    labelMedium = DefaultTypography.labelMedium.web(12, 16),
    labelSmall = DefaultTypography.labelSmall.web(10, 14),
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

/**
 * App theme. [darkTheme] follows Web: the light theme applies when the system prefers light
 * (`theme-light.css`) or the account enables manual light mode (`theme-light-manual.css`).
 */
@Composable
fun PlaygroundTheme(darkTheme: Boolean = isSystemInDarkTheme(), themeColor: String? = null, liquidGlass: Boolean = false, content: @Composable () -> Unit) {
    val web = remember(darkTheme, themeColor) { webPalette(light = !darkTheme, themeColor = themeColor) }
    val scheme = webColorScheme(web)
    val themed = if (liquidGlass) scheme.copy(
        surface = scheme.surface.copy(alpha = 0.86f),
        surfaceContainer = scheme.surfaceContainer.copy(alpha = 0.82f),
        surfaceContainerHigh = scheme.surfaceContainerHigh.copy(alpha = 0.88f),
    ) else scheme
    CompositionLocalProvider(LocalWebPalette provides web) {
        MaterialTheme(
            colorScheme = themed,
            typography = WebTypography,
            shapes = WebShapes,
            content = content,
        )
    }
}
