package com.minashin1120.aiplayground

import android.content.Context

private const val VIBRATION_PREFS = "vibration_setting"
private const val VIBRATION_ENABLED_KEY = "enabled"

/** ANDROID_ONLY.md: the on-device switch for the send / answer-complete vibration (on by default). */
fun isVibrationEnabled(context: Context): Boolean =
    context.getSharedPreferences(VIBRATION_PREFS, Context.MODE_PRIVATE).getBoolean(VIBRATION_ENABLED_KEY, true)

fun setVibrationEnabled(context: Context, enabled: Boolean) {
    context.getSharedPreferences(VIBRATION_PREFS, Context.MODE_PRIVATE).edit().putBoolean(VIBRATION_ENABLED_KEY, enabled).apply()
}
