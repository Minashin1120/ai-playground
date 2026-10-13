package com.minashin1120.aiplayground

import android.content.Context

private const val TRANSLATION_PREFS = "selection_translation"
private const val TRANSLATION_MODEL_KEY = "model"
private const val TRANSLATION_MODEL_NAME_KEY = "model_name"
private const val TRANSLATION_LANGUAGE_KEY = "language"
const val DEFAULT_TRANSLATION_LANGUAGE = "日本語"

/** ANDROID_ONLY.md: the model and target language of 「AIで翻訳」 in the text-selection menu (kept on the device). */
data class TranslationSetting(val model: String, val modelName: String, val language: String)

fun loadTranslationSetting(context: Context): TranslationSetting {
    val prefs = context.getSharedPreferences(TRANSLATION_PREFS, Context.MODE_PRIVATE)
    return TranslationSetting(
        prefs.getString(TRANSLATION_MODEL_KEY, null).orEmpty(),
        prefs.getString(TRANSLATION_MODEL_NAME_KEY, null).orEmpty(),
        prefs.getString(TRANSLATION_LANGUAGE_KEY, null)?.takeIf { it.isNotBlank() } ?: DEFAULT_TRANSLATION_LANGUAGE,
    )
}

fun setTranslationModel(context: Context, model: String, name: String) {
    context.getSharedPreferences(TRANSLATION_PREFS, Context.MODE_PRIVATE).edit()
        .putString(TRANSLATION_MODEL_KEY, model).putString(TRANSLATION_MODEL_NAME_KEY, name).apply()
}

fun setTranslationLanguage(context: Context, language: String) {
    context.getSharedPreferences(TRANSLATION_PREFS, Context.MODE_PRIVATE).edit()
        .putString(TRANSLATION_LANGUAGE_KEY, language.trim().take(40)).apply()
}
