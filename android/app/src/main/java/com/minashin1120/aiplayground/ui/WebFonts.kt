package com.minashin1120.aiplayground.ui

import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.googlefonts.Font
import androidx.compose.ui.text.googlefonts.GoogleFont
import com.minashin1120.aiplayground.R

/**
 * The Web chat loads Noto Sans JP (400/700) and JetBrains Mono (400) from Google Fonts
 * (`templates/web_fonts.html`). Android downloads the same families through the Google Fonts
 * provider and falls back to the system font until they are available, like the Web `display=swap`.
 */
internal object WebFonts {
    private val provider = GoogleFont.Provider(
        providerAuthority = "com.google.android.gms.fonts",
        providerPackage = "com.google.android.gms",
        certificates = R.array.com_google_android_gms_fonts_certs,
    )

    private val notoSansJp = GoogleFont("Noto Sans JP")
    private val jetBrainsMono = GoogleFont("JetBrains Mono")

    /** `font-family: "Noto Sans JP", system-ui, ...`. Only 400 and 700 are loaded, as on Web; other weights match the nearest one. */
    val sans: FontFamily = FontFamily(
        Font(googleFont = notoSansJp, fontProvider = provider, weight = FontWeight.Normal),
        Font(googleFont = notoSansJp, fontProvider = provider, weight = FontWeight.Bold),
    )

    /** Code blocks and inline code (`font-family: "JetBrains Mono", monospace`). */
    val mono: FontFamily = FontFamily(
        Font(googleFont = jetBrainsMono, fontProvider = provider, weight = FontWeight.Normal),
    )
}
