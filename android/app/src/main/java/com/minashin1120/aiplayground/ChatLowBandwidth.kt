package com.minashin1120.aiplayground

import android.content.SharedPreferences
import android.net.ConnectivityManager
import com.minashin1120.aiplayground.data.LOW_BANDWIDTH_PREF_KEY
import com.minashin1120.aiplayground.data.LowBandwidthSignal
import com.minashin1120.aiplayground.data.detectLowBandwidth
import com.minashin1120.aiplayground.data.effectiveConnectionType
import com.minashin1120.aiplayground.data.effectiveLowBandwidth
import com.minashin1120.aiplayground.data.lowBandwidthToast
import com.minashin1120.aiplayground.data.nextLowBandwidthPreference
import com.minashin1120.aiplayground.data.roundedDownlinkMbps
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update

/**
 * Web `detectLowBandwidthModeAuto` / `recomputeLowBandwidthMode`: the automatic switch from the
 * device's link estimate and Data Saver, and the sidebar button (auto → on → off). State lives in
 * the view model; this class only reads it and reports through `mutable`.
 */
internal class ChatLowBandwidth(
    private val connectivity: ConnectivityManager?,
    private val state: StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val prefs: SharedPreferences,
    private val notify: (String) -> Unit,
) {
    private fun signal(): LowBandwidthSignal? {
        val manager = connectivity ?: return null
        val capabilities = manager.getNetworkCapabilities(manager.activeNetwork ?: return null) ?: return null
        val kbps = capabilities.linkDownstreamBandwidthKbps
        val saveData = manager.restrictBackgroundStatus == ConnectivityManager.RESTRICT_BACKGROUND_STATUS_ENABLED
        return LowBandwidthSignal(saveData, effectiveConnectionType(kbps), roundedDownlinkMbps(kbps))
    }

    /** Web `recomputeLowBandwidthMode`; [showToast] shows the toast only when the effective mode changes. */
    fun recompute(showToast: Boolean) {
        val detection = detectLowBandwidth(runCatching { signal() }.getOrNull())
        val current = state.value
        val active = effectiveLowBandwidth(current.lowBandwidthPreference, detection.enabled)
        val changed = active != current.lowBandwidthMode
        mutable.update { it.copy(lowBandwidthMode = active, lowBandwidthReason = detection.reason) }
        if (showToast && changed) notify(lowBandwidthToast(active, current.lowBandwidthPreference, detection.reason))
    }

    /** Sidebar button: auto → on → off, persisted like Web's localStorage preference. */
    fun cycle() {
        val next = nextLowBandwidthPreference(state.value.lowBandwidthPreference)
        prefs.edit().apply { if (next == "auto") remove(LOW_BANDWIDTH_PREF_KEY) else putString(LOW_BANDWIDTH_PREF_KEY, next) }.apply()
        mutable.update { it.copy(lowBandwidthPreference = next) }
        val detection = detectLowBandwidth(runCatching { signal() }.getOrNull())
        val active = effectiveLowBandwidth(next, detection.enabled)
        mutable.update { it.copy(lowBandwidthMode = active, lowBandwidthReason = detection.reason) }
        notify(lowBandwidthToast(active, next, detection.reason))
    }
}
