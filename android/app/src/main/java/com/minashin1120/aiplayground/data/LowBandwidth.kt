package com.minashin1120.aiplayground.data

import java.util.Locale

/**
 * Web low-bandwidth mode (chat_core part03 `detectLowBandwidthModeAuto` and friends).
 * Android reads the same signals from the platform: Data Saver for `saveData`, and the link's
 * downstream estimate for `downlink` and Chrome's `effectiveType` buckets.
 */
data class LowBandwidthSignal(val saveData: Boolean, val effectiveType: String, val downlinkMbps: Double)

data class LowBandwidthDetection(val enabled: Boolean, val reason: String)

const val LOW_BANDWIDTH_PREF_KEY = "low_bandwidth_mode_pref_v1"
const val THREAD_INITIAL_MESSAGE_LIMIT = 50
const val THREAD_OLDER_PAGE_SIZE = 50
const val LOW_BANDWIDTH_INITIAL_MESSAGE_LIMIT = 40
const val LOW_BANDWIDTH_OLDER_PAGE_SIZE = 60

/** Chrome's effective connection type buckets from a downstream estimate in kbps. */
fun effectiveConnectionType(downstreamKbps: Int): String = when {
    downstreamKbps <= 0 -> ""
    downstreamKbps < 50 -> "slow-2g"
    downstreamKbps < 70 -> "2g"
    downstreamKbps < 700 -> "3g"
    else -> "4g"
}

/** Chrome rounds `navigator.connection.downlink` to multiples of 25 kbps, expressed in Mbps. */
fun roundedDownlinkMbps(downstreamKbps: Int): Double =
    if (downstreamKbps <= 0) 0.0 else Math.round(downstreamKbps / 25.0) * 25 / 1000.0

fun detectLowBandwidth(signal: LowBandwidthSignal?): LowBandwidthDetection {
    if (signal == null) return LowBandwidthDetection(false, "")
    val effectiveType = signal.effectiveType.lowercase(Locale.ROOT)
    val slowType = effectiveType == "slow-2g" || effectiveType == "2g" || effectiveType == "3g"
    val lowDownlink = signal.downlinkMbps > 0 && signal.downlinkMbps < 1.3
    val parts = buildList {
        if (signal.saveData) add("データ節約")
        if (effectiveType.isNotEmpty()) add("回線:$effectiveType")
        if (lowDownlink) add("下り:${formatDownlink(signal.downlinkMbps)}Mbps")
    }
    return LowBandwidthDetection(signal.saveData || slowType || lowDownlink, parts.joinToString(" / "))
}

/** Matches JavaScript number formatting (`1.25`, `0.4`, `1`). */
private fun formatDownlink(value: Double): String =
    if (value == Math.floor(value)) value.toLong().toString() else value.toString().trimEnd('0').trimEnd('.')

fun normalizeLowBandwidthPreference(raw: String?): String = when (raw?.trim()?.lowercase(Locale.ROOT)) {
    "on" -> "on"
    "off" -> "off"
    else -> "auto"
}

/** auto → on → off → auto, like the sidebar button on Web. */
fun nextLowBandwidthPreference(current: String): String = when (normalizeLowBandwidthPreference(current)) {
    "auto" -> "on"
    "on" -> "off"
    else -> "auto"
}

fun effectiveLowBandwidth(preference: String, auto: Boolean): Boolean = when (normalizeLowBandwidthPreference(preference)) {
    "on" -> true
    "off" -> false
    else -> auto
}

/** Toast of `applyLowBandwidthModeState(..., { notify: true })`. */
fun lowBandwidthToast(active: Boolean, preference: String, reason: String): String {
    val pref = if (normalizeLowBandwidthPreference(preference) == "auto") "自動" else "手動"
    val suffix = if (reason.isNotBlank()) " ($reason)" else ""
    return "低速回線モードを${if (active) "ON" else "OFF"}にしました [$pref]$suffix"
}
