package com.minashin1120.aiplayground

import android.content.Context
import com.minashin1120.aiplayground.data.COMPOSER_SELECT_DEFAULTS
import com.minashin1120.aiplayground.data.CompressionSettings
import com.minashin1120.aiplayground.data.compressionSettingsFrom
import com.minashin1120.aiplayground.data.saveCompressionSettings

private const val TRANSLATION_PREFS = "selection_translation"
private const val TRANSLATION_MODEL_KEY = "model"
private const val TRANSLATION_MODEL_NAME_KEY = "model_name"
private const val TRANSLATION_LANGUAGE_KEY = "language"
private const val TRANSLATION_THINKING_KEY = "thinking"
const val DEFAULT_TRANSLATION_LANGUAGE = "日本語"

/**
 * ANDROID_ONLY.md: the model, target language and reasoning options of 「AIで翻訳」 (kept on the device). The
 * options default to the composer's (`COMPOSER_SELECT_DEFAULTS`) with Thinking off; options the model does not
 * have are not used (`SelectionTranslator.requestOptions`).
 */
data class TranslationSetting(
    val model: String,
    val modelName: String,
    val language: String,
    val thinking: Boolean = false,
    val thinkingLevel: String = COMPOSER_SELECT_DEFAULTS.getValue("thinking_level"),
    val thinkingBudget: String = COMPOSER_SELECT_DEFAULTS.getValue("thinking_budget"),
    val reasoningEffort: String = COMPOSER_SELECT_DEFAULTS.getValue("reasoning_effort"),
    /** Compression of shared images: the prompt bar's settings and defaults, kept separately. */
    val compression: CompressionSettings = CompressionSettings(),
)

/** Composer select keys (`thinking_level`, `thinking_budget`, `reasoning_effort`) stored for the translation. */
val TRANSLATION_OPTION_KEYS = listOf("thinking_level", "thinking_budget", "reasoning_effort")

fun loadTranslationSetting(context: Context): TranslationSetting {
    val prefs = context.getSharedPreferences(TRANSLATION_PREFS, Context.MODE_PRIVATE)
    return TranslationSetting(
        prefs.getString(TRANSLATION_MODEL_KEY, null).orEmpty(),
        prefs.getString(TRANSLATION_MODEL_NAME_KEY, null).orEmpty(),
        prefs.getString(TRANSLATION_LANGUAGE_KEY, null)?.takeIf { it.isNotBlank() } ?: DEFAULT_TRANSLATION_LANGUAGE,
        thinking = prefs.getBoolean(TRANSLATION_THINKING_KEY, false),
        thinkingLevel = prefs.getString("thinking_level", null) ?: COMPOSER_SELECT_DEFAULTS.getValue("thinking_level"),
        thinkingBudget = prefs.getString("thinking_budget", null) ?: COMPOSER_SELECT_DEFAULTS.getValue("thinking_budget"),
        reasoningEffort = prefs.getString("reasoning_effort", null) ?: COMPOSER_SELECT_DEFAULTS.getValue("reasoning_effort"),
        compression = compressionSettingsFrom(prefs),
    )
}

fun setTranslationCompression(context: Context, settings: CompressionSettings) {
    saveCompressionSettings(context.getSharedPreferences(TRANSLATION_PREFS, Context.MODE_PRIVATE), settings)
}

fun setTranslationThinking(context: Context, on: Boolean) {
    context.getSharedPreferences(TRANSLATION_PREFS, Context.MODE_PRIVATE).edit().putBoolean(TRANSLATION_THINKING_KEY, on).apply()
}

/** [key] is one of [TRANSLATION_OPTION_KEYS]. */
fun setTranslationOption(context: Context, key: String, value: String) {
    require(key in TRANSLATION_OPTION_KEYS)
    context.getSharedPreferences(TRANSLATION_PREFS, Context.MODE_PRIVATE).edit().putString(key, value).apply()
}

fun setTranslationModel(context: Context, model: String, name: String) {
    context.getSharedPreferences(TRANSLATION_PREFS, Context.MODE_PRIVATE).edit()
        .putString(TRANSLATION_MODEL_KEY, model).putString(TRANSLATION_MODEL_NAME_KEY, name).apply()
}

fun setTranslationLanguage(context: Context, language: String) {
    context.getSharedPreferences(TRANSLATION_PREFS, Context.MODE_PRIVATE).edit()
        .putString(TRANSLATION_LANGUAGE_KEY, language.trim().take(40)).apply()
}
