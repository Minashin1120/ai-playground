package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

// Native equivalents of the Web design tokens in chat.custom.css.
private val WebDarkColors = darkColorScheme(
    primary = Color(0xFF0DD4BF), onPrimary = Color(0xFF031413),
    primaryContainer = Color(0xFF0A403D), onPrimaryContainer = Color(0xFFA5F7ED),
    secondary = Color(0xFFF59E0B), onSecondary = Color(0xFF1F1303),
    background = Color(0xFF05070F), onBackground = Color(0xFFEEF1F7),
    surface = Color(0xFF0C1224), onSurface = Color(0xFFEEF1F7),
    surfaceVariant = Color(0xFF090E1C), onSurfaceVariant = Color(0xFF8B95A8),
    outline = Color(0xFF1A2338), error = Color(0xFFF9708D),
)

private val WebLightColors = lightColorScheme(
    primary = Color(0xFF087E76), onPrimary = Color.White,
    primaryContainer = Color(0xFFCCFBF1), onPrimaryContainer = Color(0xFF043A36),
    secondary = Color(0xFF9A5B00), onSecondary = Color.White,
    background = Color(0xFFF7FAFA), onBackground = Color(0xFF101828),
    surface = Color(0xFFFFFFFF), onSurface = Color(0xFF101828),
    surfaceVariant = Color(0xFFF0F5F5), onSurfaceVariant = Color(0xFF52606D),
    outline = Color(0xFFCBD5E1), error = Color(0xFFB4233E),
)

@Composable
fun PlaygroundTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = if (isSystemInDarkTheme()) WebDarkColors else WebLightColors, content = content)
}
