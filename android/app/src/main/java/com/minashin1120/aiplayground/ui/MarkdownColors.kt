package com.minashin1120.aiplayground.ui

import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

/* Palette of the rendered markdown, taken from the Web `chat.custom*.css` and highlight.js themes. */

// ---------------------------------------------------------------------------------------------
// Colors of `.content-area` (chat.custom.v*.css, theme-light-manual.css)
// ---------------------------------------------------------------------------------------------

internal data class MarkdownColors(
    val text: Color,
    val quote: Color,
    val quoteBar: Color,
    val link: Color,
    val inlineCode: Color,
    val inlineCodeBackground: Color,
    val codeWrapper: Color,
    val codeWrapperBorder: Color,
    val codeHeader: Color,
    val codeHeaderText: Color,
    val codeBody: Color,
    val codeText: Color,
    val table: Color,
    val tableBorder: Color,
    val tableHead: Color,
    val tableHeadText: Color,
    val rule: Color,
    val syntax: SyntaxColors,
)

/** highlight.js theme colors (atom-one-dark on the dark theme, atom-one-light on the light theme). */
internal data class SyntaxColors(
    val keyword: Color,
    val title: Color,
    val string: Color,
    val number: Color,
    val comment: Color,
    val builtIn: Color,
    val literal: Color,
)

@Composable
internal fun markdownColors(): MarkdownColors = markdownColorsFor(LocalWebPalette.current)

internal fun markdownColorsFor(web: WebPalette): MarkdownColors {
    return if (web.isLight) MarkdownColors(
        text = web.text,
        quote = web.muted,
        quoteBar = web.theme.t500,
        link = web.theme300,
        inlineCode = web.theme300,
        inlineCodeBackground = web.theme.rgb(0.12f),
        codeWrapper = Color.White,
        codeWrapperBorder = Color(15, 23, 42).copy(alpha = 0.10f),
        codeHeader = Color(0xFFF1F4F9),
        codeHeaderText = web.muted,
        codeBody = Color(0xFFF8FAFC),
        codeText = Color(0xFF383A42),
        table = Color(0xFFF7F9FC),
        tableBorder = web.line,
        tableHead = web.theme.rgb(0.10f),
        tableHeadText = web.theme300,
        rule = web.line,
        syntax = SyntaxColors(
            keyword = Color(0xFFA626A4), title = Color(0xFF4078F2), string = Color(0xFF50A14F),
            number = Color(0xFF986801), comment = Color(0xFFA0A1A7), builtIn = Color(0xFFC18401), literal = Color(0xFF0184BB),
        ),
    ) else MarkdownColors(
        text = Color(0xFFE6E9F0),
        quote = web.muted,
        quoteBar = web.theme.t500,
        link = web.theme300,
        inlineCode = Color(0xFFCCFBF1),
        inlineCodeBackground = web.theme.rgb(0.14f),
        codeWrapper = Color(8, 12, 22).copy(alpha = 0.72f),
        codeWrapperBorder = Color.White.copy(alpha = 0.08f),
        codeHeader = Color(15, 23, 42).copy(alpha = 0.6f),
        codeHeaderText = Color(0xFF9AA3B2),
        codeBody = Color(0xFF060D1D),
        codeText = Color(0xFFABB2BF),
        table = Color(0xFF090E1C),
        tableBorder = web.line,
        tableHead = web.theme.rgb(0.12f),
        tableHeadText = web.theme200,
        rule = web.line,
        syntax = SyntaxColors(
            keyword = Color(0xFFC678DD), title = Color(0xFF61AEEE), string = Color(0xFF98C379),
            number = Color(0xFFD19A66), comment = Color(0xFF5C6370), builtIn = Color(0xFFE6C07B), literal = Color(0xFF56B6C2),
        ),
    )
}
