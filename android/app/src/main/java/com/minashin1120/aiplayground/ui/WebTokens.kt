package com.minashin1120.aiplayground.ui

import androidx.compose.runtime.Immutable
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.graphics.Color
import kotlin.math.roundToInt

/** Default accent of the Web theme (`THEME_DEFAULT` in chat_core part01). */
internal const val WEB_THEME_DEFAULT = "#0dd4bf"

/** Accent ramp derived exactly like the Web `deriveTheme()` (chat_core part01 `applyThemeColor`). */
@Immutable
internal data class WebThemeRamp(
    val t500: Color,
    val t600: Color,
    val t700: Color,
    val t300: Color,
    val t200: Color,
) {
    /** `rgba(var(--theme-rgb), alpha)` in the Web stylesheet. */
    fun rgb(alpha: Float): Color = t500.copy(alpha = alpha)
}

/** Web `normalizeHex()`: accepts `#abc`, `abc`, `#aabbcc` and returns lowercase `#aabbcc` or null. */
internal fun normalizeWebHex(value: String?): String? {
    var v = value?.trim().orEmpty()
    if (v.isEmpty()) return null
    if (!v.startsWith("#")) v = "#$v"
    if (v.length == 4) v = "#${v[1]}${v[1]}${v[2]}${v[2]}${v[3]}${v[3]}"
    if (!Regex("^#[0-9a-fA-F]{6}$").matches(v)) return null
    return v.lowercase()
}

internal fun deriveWebTheme(value: String?): WebThemeRamp {
    val hex = normalizeWebHex(value) ?: WEB_THEME_DEFAULT
    val r = hex.substring(1, 3).toInt(16)
    val g = hex.substring(3, 5).toInt(16)
    val b = hex.substring(5, 7).toInt(16)
    // JS Math.round rounds halves up; roundToInt on these non-negative values matches it.
    fun mix(a: Int, target: Int, p: Double): Int = (a + (target - a) * p).roundToInt()
    fun rgb(p: Double, target: Int) = Color(mix(r, target, p), mix(g, target, p), mix(b, target, p))
    return WebThemeRamp(
        t500 = Color(r, g, b),
        t600 = rgb(0.18, 0),
        t700 = rgb(0.32, 0),
        t300 = rgb(0.45, 255),
        t200 = rgb(0.7, 255),
    )
}

/**
 * The `:root` design tokens of `chat.custom.v*.css` (dark) and `theme-light-manual.css` (light).
 * Keep the values identical to the Web stylesheets; component-specific colors live next to the component.
 */
@Immutable
internal data class WebPalette(
    val isLight: Boolean,
    val theme: WebThemeRamp,
    val bg1: Color,
    val bg2: Color,
    val bg3: Color,
    val panel: Color,
    val panel2: Color,
    val panelElevated: Color,
    val panelChrome: Color,
    val line: Color,
    val lineSoft: Color,
    val lineStrong: Color,
    val text: Color,
    val muted: Color,
    val textInverse: Color,
    val accent2: Color,
    val danger: Color,
    val success: Color,
) {
    /** `--theme-300`: labels and active states. The light theme maps it to the darker end of the ramp. */
    val theme300: Color get() = if (isLight) theme.t700 else theme.t300

    /** `--theme-200`: the lightest accent text. The light theme maps it to `--theme-600`. */
    val theme200: Color get() = if (isLight) theme.t600 else theme.t200
}

internal fun webPalette(light: Boolean, themeColor: String?): WebPalette {
    val ramp = deriveWebTheme(themeColor)
    return if (light) WebPalette(
        isLight = true,
        theme = ramp,
        bg1 = Color(0xFFF6F8FC),
        bg2 = Color(0xFFEEF2F9),
        bg3 = Color(0xFFF1F5FB),
        panel = Color(0xFFFFFFFF),
        panel2 = Color(0xFFF7F9FC),
        panelElevated = Color(255, 255, 255).copy(alpha = 0.95f),
        panelChrome = Color(255, 255, 255).copy(alpha = 0.84f),
        line = Color(0xFFD9E1EC),
        lineSoft = Color(15, 23, 42).copy(alpha = 0.10f),
        lineStrong = Color(15, 23, 42).copy(alpha = 0.18f),
        text = Color(0xFF131C2E),
        muted = Color(0xFF5C6779),
        textInverse = Color(0xFF031413),
        accent2 = Color(0xFFF59E0B),
        danger = Color(0xFFF9708D),
        success = Color(0xFF34D399),
    ) else WebPalette(
        isLight = false,
        theme = ramp,
        bg1 = Color(0xFF05070F),
        bg2 = Color(0xFF080C18),
        bg3 = Color(0xFF070B16),
        panel = Color(0xFF0C1224),
        panel2 = Color(0xFF090E1C),
        panelElevated = Color(14, 20, 40).copy(alpha = 0.90f),
        panelChrome = Color(8, 12, 24).copy(alpha = 0.78f),
        line = Color(0xFF1A2338),
        lineSoft = Color(226, 232, 240).copy(alpha = 0.08f),
        lineStrong = Color(226, 232, 240).copy(alpha = 0.16f),
        text = Color(0xFFEEF1F7),
        muted = Color(0xFF8B95A8),
        textInverse = Color(0xFF031413),
        accent2 = Color(0xFFF59E0B),
        danger = Color(0xFFF9708D),
        success = Color(0xFF34D399),
    )
}

internal val LocalWebPalette = staticCompositionLocalOf { webPalette(light = false, themeColor = null) }

/** Tailwind CSS v3 palette used by the Web templates (`chat.tailwind.v*.css`). */
internal object Tw {
    val white = Color(0xFFFFFFFF)
    val black = Color(0xFF000000)

    val gray50 = Color(0xFFF9FAFB)
    val gray100 = Color(0xFFF3F4F6)
    val gray200 = Color(0xFFE5E7EB)
    val gray300 = Color(0xFFD1D5DB)
    val gray400 = Color(0xFF9CA3AF)
    val gray500 = Color(0xFF6B7280)
    val gray600 = Color(0xFF4B5563)
    val gray700 = Color(0xFF374151)
    val gray800 = Color(0xFF1F2937)
    val gray900 = Color(0xFF111827)
    val gray950 = Color(0xFF030712)

    val slate100 = Color(0xFFF1F5F9)
    val slate200 = Color(0xFFE2E8F0)
    val slate300 = Color(0xFFCBD5E1)
    val slate400 = Color(0xFF94A3B8)
    val slate500 = Color(0xFF64748B)
    val slate600 = Color(0xFF475569)
    val slate700 = Color(0xFF334155)
    val slate800 = Color(0xFF1E293B)
    val slate900 = Color(0xFF0F172A)

    val red100 = Color(0xFFFEE2E2)
    val red200 = Color(0xFFFECACA)
    val red300 = Color(0xFFFCA5A5)
    val red400 = Color(0xFFF87171)
    val red500 = Color(0xFFEF4444)
    val red600 = Color(0xFFDC2626)
    val red700 = Color(0xFFB91C1C)
    val red800 = Color(0xFF991B1B)
    val red900 = Color(0xFF7F1D1D)

    val orange200 = Color(0xFFFED7AA)
    val orange300 = Color(0xFFFDBA74)
    val orange400 = Color(0xFFFB923C)
    val orange500 = Color(0xFFF97316)
    val orange600 = Color(0xFFEA580C)
    val orange700 = Color(0xFFC2410C)
    val orange900 = Color(0xFF7C2D12)

    val amber100 = Color(0xFFFEF3C7)
    val amber200 = Color(0xFFFDE68A)
    val amber300 = Color(0xFFFCD34D)
    val amber400 = Color(0xFFFBBF24)
    val amber500 = Color(0xFFF59E0B)
    val amber600 = Color(0xFFD97706)
    val amber700 = Color(0xFFB45309)
    val amber800 = Color(0xFF92400E)
    val amber900 = Color(0xFF78350F)

    val yellow100 = Color(0xFFFEF9C3)
    val yellow200 = Color(0xFFFEF08A)
    val yellow300 = Color(0xFFFDE047)
    val yellow400 = Color(0xFFFACC15)
    val yellow500 = Color(0xFFEAB308)
    val yellow600 = Color(0xFFCA8A04)
    val yellow700 = Color(0xFFA16207)
    val yellow900 = Color(0xFF713F12)

    val green200 = Color(0xFFBBF7D0)
    val green300 = Color(0xFF86EFAC)
    val green400 = Color(0xFF4ADE80)
    val green500 = Color(0xFF22C55E)
    val green600 = Color(0xFF16A34A)
    val green700 = Color(0xFF15803D)
    val green900 = Color(0xFF14532D)

    val emerald100 = Color(0xFFD1FAE5)
    val emerald200 = Color(0xFFA7F3D0)
    val emerald300 = Color(0xFF6EE7B7)
    val emerald400 = Color(0xFF34D399)
    val emerald500 = Color(0xFF10B981)
    val emerald600 = Color(0xFF059669)
    val emerald700 = Color(0xFF047857)
    val emerald800 = Color(0xFF065F46)
    val emerald900 = Color(0xFF064E3B)
    val emerald950 = Color(0xFF022C22)

    val teal200 = Color(0xFF99F6E4)
    val teal300 = Color(0xFF5EEAD4)
    val teal400 = Color(0xFF2DD4BF)
    val teal500 = Color(0xFF14B8A6)
    val teal600 = Color(0xFF0D9488)
    val teal700 = Color(0xFF0F766E)
    val teal900 = Color(0xFF134E4A)

    val cyan100 = Color(0xFFCFFAFE)
    val cyan200 = Color(0xFFA5F3FC)
    val cyan300 = Color(0xFF67E8F9)
    val cyan400 = Color(0xFF22D3EE)
    val cyan500 = Color(0xFF06B6D4)
    val cyan600 = Color(0xFF0891B2)
    val cyan700 = Color(0xFF0E7490)
    val cyan900 = Color(0xFF164E63)

    val sky300 = Color(0xFF7DD3FC)
    val sky400 = Color(0xFF38BDF8)
    val sky500 = Color(0xFF0EA5E9)
    val sky600 = Color(0xFF0284C7)
    val sky700 = Color(0xFF0369A1)

    val blue100 = Color(0xFFDBEAFE)
    val blue200 = Color(0xFFBFDBFE)
    val blue300 = Color(0xFF93C5FD)
    val blue400 = Color(0xFF60A5FA)
    val blue500 = Color(0xFF3B82F6)
    val blue600 = Color(0xFF2563EB)
    val blue700 = Color(0xFF1D4ED8)
    val blue900 = Color(0xFF1E3A8A)
    val blue950 = Color(0xFF172554)

    val indigo200 = Color(0xFFC7D2FE)
    val indigo300 = Color(0xFFA5B4FC)
    val indigo400 = Color(0xFF818CF8)
    val indigo500 = Color(0xFF6366F1)
    val indigo600 = Color(0xFF4F46E5)
    val indigo900 = Color(0xFF312E81)

    val violet200 = Color(0xFFDDD6FE)
    val violet300 = Color(0xFFC4B5FD)
    val violet400 = Color(0xFFA78BFA)
    val violet500 = Color(0xFF8B5CF6)
    val violet600 = Color(0xFF7C3AED)
    val violet900 = Color(0xFF4C1D95)

    val purple100 = Color(0xFFF3E8FF)
    val purple200 = Color(0xFFE9D5FF)
    val purple300 = Color(0xFFD8B4FE)
    val purple400 = Color(0xFFC084FC)
    val purple500 = Color(0xFFA855F7)
    val purple600 = Color(0xFF9333EA)
    val purple700 = Color(0xFF7E22CE)
    val purple900 = Color(0xFF581C87)

    val pink300 = Color(0xFFF9A8D4)
    val pink400 = Color(0xFFF472B6)
    val pink500 = Color(0xFFEC4899)

    val rose200 = Color(0xFFFECDD3)
    val rose300 = Color(0xFFFDA4AF)
    val rose400 = Color(0xFFFB7185)
    val rose500 = Color(0xFFF43F5E)
    val rose600 = Color(0xFFE11D48)
    val rose700 = Color(0xFFBE123C)
}

/*
 * Tailwind utility colors under the manual light theme. `theme-light-manual.css` rewrites the
 * dark-first utilities with `!important`, so the same markup needs these mappings on Android.
 */
private fun rgb(r: Int, g: Int, b: Int, a: Float = 1f) = Color(r, g, b).copy(alpha = a)

private val lightText: Map<Color, Color> by lazy {
    buildMap {
        put(Tw.white, Color(0xFF131C2E))
        put(Tw.gray100, Color(0xFF1B2436)); put(Tw.gray200, Color(0xFF2B3546)); put(Tw.gray300, Color(0xFF3F4A5C))
        put(Tw.gray400, Color(0xFF5B6675)); put(Tw.gray500, Color(0xFF697588))
        put(Tw.slate100, Color(0xFF1B2436)); put(Tw.slate200, Color(0xFF2B3546)); put(Tw.slate300, Color(0xFF3F4A5C))
        put(Tw.slate400, Color(0xFF5B6675)); put(Tw.slate500, Color(0xFF697588))
        listOf(Tw.amber200, Tw.amber300).forEach { put(it, Color(0xFF92400E)) }
        listOf(Tw.yellow200, Tw.yellow300).forEach { put(it, Color(0xFF854D0E)) }
        listOf(Tw.orange200, Tw.orange300).forEach { put(it, Color(0xFF9A3412)) }
        listOf(Tw.red100, Tw.red200, Tw.red300).forEach { put(it, Color(0xFFB91C1C)) }
        listOf(Tw.rose200, Tw.rose300).forEach { put(it, Color(0xFFBE123C)) }
        put(Tw.pink300, Color(0xFFBE185D))
        listOf(Tw.purple200, Tw.purple300).forEach { put(it, Color(0xFF7E22CE)) }
        listOf(Tw.violet200, Tw.violet300).forEach { put(it, Color(0xFF6D28D9)) }
        put(Tw.indigo300, Color(0xFF4338CA))
        listOf(Tw.blue200, Tw.blue300).forEach { put(it, Color(0xFF1D4ED8)) }
        put(Tw.sky300, Color(0xFF0369A1))
        listOf(Tw.cyan100, Tw.cyan200, Tw.cyan300).forEach { put(it, Color(0xFF0E7490)) }
        listOf(Tw.teal200, Tw.teal300).forEach { put(it, Color(0xFF0F766E)) }
        listOf(Tw.emerald200, Tw.emerald300).forEach { put(it, Color(0xFF047857)) }
        put(Tw.green300, Color(0xFF15803D))
    }
}

private val lightSurface: Map<Color, Color> by lazy {
    mapOf(
        Tw.gray950 to Color(0xFFFFFFFF), Tw.gray900 to Color(0xFFF7F9FC), Tw.gray800 to Color(0xFFF1F5FA),
        Tw.gray700 to Color(0xFFE7EDF6), Tw.gray600 to Color(0xFFDBE3EE), Tw.gray500 to Color(0xFFCFD8E6),
        Tw.slate900 to Color(0xFFF7F9FC), Tw.slate800 to Color(0xFFF1F5FA), Tw.slate700 to Color(0xFFE7EDF6),
        Tw.slate600 to Color(0xFFDBE3EE), Tw.slate500 to Color(0xFFCFD8E6),
    )
}

/** `bg-<color>-800/…`, `-900/…` tints become a faint tint of the 500 shade. */
private val lightTint: Map<Color, Color> by lazy {
    mapOf(
        Tw.amber900 to rgb(245, 158, 11, 0.14f), Tw.yellow900 to rgb(234, 179, 8, 0.14f),
        Tw.orange900 to rgb(249, 115, 22, 0.12f), Tw.red900 to rgb(239, 68, 68, 0.12f), Tw.red800 to rgb(239, 68, 68, 0.12f),
        Tw.purple900 to rgb(168, 85, 247, 0.12f), Tw.violet900 to rgb(139, 92, 246, 0.12f),
        Tw.indigo900 to rgb(99, 102, 241, 0.12f), Tw.blue900 to rgb(59, 130, 246, 0.12f),
        Tw.cyan900 to rgb(6, 182, 212, 0.12f), Tw.teal900 to rgb(20, 184, 166, 0.12f),
        Tw.emerald900 to rgb(16, 185, 129, 0.12f), Tw.green900 to rgb(34, 197, 94, 0.12f),
    )
}

private val lightBorder: Map<Color, Color> by lazy {
    mapOf(
        Tw.gray500 to Color(0xFFC3CDDC), Tw.gray600 to Color(0xFFCDD7E4), Tw.gray700 to Color(0xFFD9E1EC),
        Tw.gray800 to Color(0xFFE2E8F1), Tw.gray900 to Color(0xFFE6EBF3),
        Tw.slate600 to Color(0xFFCDD7E4), Tw.slate700 to Color(0xFFD9E1EC), Tw.slate800 to Color(0xFFE2E8F1),
    )
}

/** `text-<color>` for the current theme. */
internal fun WebPalette.twText(color: Color): Color =
    if (isLight) lightText[color.copy(alpha = 1f)]?.copy(alpha = color.alpha) ?: color else color

/** `bg-<color>` or `bg-<color>/<alpha>` for the current theme. Light gray surfaces become opaque, as on Web. */
internal fun WebPalette.twBg(color: Color, alpha: Float = 1f): Color {
    if (!isLight) return color.copy(alpha = alpha)
    lightSurface[color]?.let { return it }
    if (alpha < 1f) lightTint[color]?.let { return it }
    if (color == Tw.white && alpha < 1f) return rgb(15, 23, 42, 0.06f)
    return color.copy(alpha = alpha)
}

/** `border-<color>` or `border-<color>/<alpha>` for the current theme. */
internal fun WebPalette.twBorder(color: Color, alpha: Float = 1f): Color {
    if (!isLight) return color.copy(alpha = alpha)
    if (color == Tw.white) return rgb(15, 23, 42, 0.14f)
    return lightBorder[color] ?: color.copy(alpha = alpha)
}
