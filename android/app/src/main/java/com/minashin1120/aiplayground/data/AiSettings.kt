package com.minashin1120.aiplayground.data

import org.json.JSONArray
import org.json.JSONObject

/** Where a `/settings` result row leads (Web `AI_SETTING_JUMP_TARGETS`): a settings tab id, or the rich-paste modal. */
data class AiSettingTarget(val label: String, val tab: String = "general", val richPaste: Boolean = false)

val AI_SETTING_JUMP_TARGETS = mapOf(
    "default_model" to AiSettingTarget("既定のモデル"),
    "default_vision_model" to AiSettingTarget("Vision Model"),
    "use_last_chat_settings" to AiSettingTarget("前回の設定を継続"),
    "default_enable_search" to AiSettingTarget("既定のSearch"),
    "default_enable_url_context" to AiSettingTarget("既定のURLs"),
    "default_enable_maps" to AiSettingTarget("既定のMaps"),
    "default_enable_python" to AiSettingTarget("既定のPython"),
    "default_enable_file_creation" to AiSettingTarget("既定のFile"),
    "default_enable_thinking" to AiSettingTarget("既定のThinking"),
    "default_thinking_level" to AiSettingTarget("Thinking Level"),
    "default_thinking_budget" to AiSettingTarget("Thinking Budget"),
    "default_reasoning_effort" to AiSettingTarget("Reasoning Effort"),
    "default_enable_system_prompt" to AiSettingTarget("既定のSysPrompt"),
    "default_enable_mcp" to AiSettingTarget("既定のMCP"),
    "default_safety_setting" to AiSettingTarget("既定のSafety"),
    "auto_search_on_links" to AiSettingTarget("Xリンクの自動検索"),
    "mic_transcribe_mode" to AiSettingTarget("マイク文字起こし方式"),
    "stt_model" to AiSettingTarget("STTモデル"),
    "llm_transcribe_prompt" to AiSettingTarget("LLM文字起こしプロンプト"),
    "enter_to_send" to AiSettingTarget("Enterで送信"),
    "compact_prompt_mode" to AiSettingTarget("プロンプトバー表示"),
    "minimal_prompt_mode" to AiSettingTarget("ミニマル表示"),
    "temp_chat_timeout_seconds" to AiSettingTarget("一時チャット保持時間"),
    "system_prompt" to AiSettingTarget("ユーザーシステムプロンプト", "prompt"),
    "system_prompt_enabled" to AiSettingTarget("システムプロンプト", "prompt"),
    "apply_global_system_prompt" to AiSettingTarget("ユーザープロンプトの適用", "prompt"),
    "apply_auto_system_prompt_notices" to AiSettingTarget("自動注入プロンプト", "prompt"),
    "auto_system_prompt_notices_config" to AiSettingTarget("自動注入プロンプト設定", "prompt"),
    "theme_color" to AiSettingTarget("テーマカラー", "display"),
    "liquid_glass_enabled" to AiSettingTarget("Liquid Glass", "display"),
    "use_sw_cache" to AiSettingTarget("高速キャッシュ", "data"),
    "enable_latency_metrics" to AiSettingTarget("レスポンス速度の計測", "data"),
    "enable_client_debug_log" to AiSettingTarget("デバッグログの拡張送信", "data"),
    "bot_detection_enabled" to AiSettingTarget("Bot Detection", "security"),
    "skip_2fa_on_google_login" to AiSettingTarget("Googleログイン時の2FA", "2fa"),
    "default_2fa_method" to AiSettingTarget("既定の2FA方式", "2fa"),
    "rich_paste_prompt_default" to AiSettingTarget("リッチ貼り付けプロンプト", richPaste = true),
    "rich_paste_prompt_use_custom_default" to AiSettingTarget("リッチ貼り付けの既定値", richPaste = true),
)

/** Web `formatAiSettingValue`. */
fun formatAiSettingValue(value: Any?): String = when {
    value == true -> "ON"
    value == false -> "OFF"
    value == "(更新)" -> "更新済み"
    value == null || value == JSONObject.NULL || value == "" -> "未設定"
    value is JSONObject || value is JSONArray -> value.toString()
    value is Double && value % 1.0 == 0.0 -> value.toLong().toString()
    else -> value.toString()
}

/** Web `summarizeAiSettingsConversationValues`: what the next `/settings` request is told about this result. */
fun summarizeAiSettings(values: JSONObject, inspect: Boolean): String {
    val prefix = if (inspect) "現在の設定を確認しました。" else "設定を更新しました。"
    val details = values.keys().asSequence().joinToString("\n") { key -> "$key: ${formatAiSettingValue(values.opt(key)).take(180)}" }
    return (prefix + if (details.isNotEmpty()) "\n$details" else "").take(1600)
}

/** One temporary `/settings` bubble in the chat (Web renders them without saving them to the thread). */
data class SettingsBubble(
    val id: String,
    val role: String,
    val text: String,
    val model: String = "",
    /** (setting key, formatted value) rows that open the matching settings screen. */
    val entries: List<Pair<String, String>> = emptyList(),
    val pending: Boolean = false,
)

/** Web `getModelProviderInfo`: which API key a model needs (for the "APIキーが必要です" dialog). */
data class ApiKeyInfo(val provider: String, val keyField: String, val label: String)

fun apiKeyInfoFor(model: String): ApiKeyInfo? {
    val id = model.lowercase().trim()
    if (id.isEmpty()) return null
    return when {
        id.startsWith("gemini") || id.startsWith("veo-") || id.startsWith("lyria-") || id.startsWith("deep-research-") ||
            id.startsWith("antigravity-") -> ApiKeyInfo("gemini", "gemini_key", "Gemini API Key")
        id.startsWith("gpt") || id.startsWith("o1") || id.startsWith("o3") -> ApiKeyInfo("openai", "openai_key", "OpenAI API Key")
        id.startsWith("deepseek") -> ApiKeyInfo("deepseek", "deepseek_key", "DeepSeek API Key")
        id.startsWith("kimi") -> ApiKeyInfo("kimi", "kimi_key", "Kimi (Moonshot) API Key")
        id.startsWith("mistral") -> ApiKeyInfo("mistral", "mistral_key", "Mistral API Key")
        id.startsWith("claude") -> ApiKeyInfo("anthropic", "anthropic_key", "Anthropic API Key")
        id.startsWith("grok") -> ApiKeyInfo("xai", "xai_key", "xAI (Grok) API Key")
        id.startsWith("google") -> ApiKeyInfo("google", "google_key", "Google API Key (TTS)")
        else -> ApiKeyInfo("openai", "openai_key", "OpenAI API Key")
    }
}

/** Web `xLinkPattern`: an X / Twitter link in the prompt or the quote. */
val X_LINK_PATTERN = Regex("(https?://)?(x\\.com|twitter\\.com)/", RegexOption.IGNORE_CASE)
