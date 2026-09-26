package com.minashin1120.aiplayground.ui

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import com.minashin1120.aiplayground.data.AutoSystemPrompt
import com.minashin1120.aiplayground.data.PROVIDER_KEY_FIELDS
import com.minashin1120.aiplayground.data.Preferences
import com.minashin1120.aiplayground.data.SECRET_MASK
import org.json.JSONObject

/** Web default for `set-temp-chat-timeout-seconds` and its bounds (`TEMP_CHAT_TIMEOUT_*_SECONDS`). */
internal const val TEMP_CHAT_TIMEOUT_MIN = 10
internal const val TEMP_CHAT_TIMEOUT_MAX = 3600
internal const val THEME_DEFAULT = "#0dd4bf"

/**
 * The editable copy of the settings modal. Like the Web form it is filled once from the server
 * payload and sent in one request by 保存.
 */
internal class SettingsForm(prefs: Preferences?) {
    var enterToSend by mutableStateOf(prefs?.enterToSend ?: false)
    var promptBarMode by mutableStateOf(prefs?.effectivePromptBarMode ?: "normal")
    var voiceStudio by mutableStateOf(prefs?.voiceStudioUi ?: true)
    var defaultModel by mutableStateOf(prefs?.defaultModel?.ifBlank { null } ?: "gemini-3.6-flash")
    var visionModel by mutableStateOf(prefs?.defaultVisionModel?.ifBlank { null } ?: "gemini-3-flash-preview")
    var useLast by mutableStateOf(prefs?.useLastChatSettings ?: false)
    var search by mutableStateOf(prefs?.defaultEnableSearch ?: false)
    var urlContext by mutableStateOf(prefs?.defaultEnableUrlContext ?: false)
    var maps by mutableStateOf(prefs?.defaultEnableMaps ?: false)
    var python by mutableStateOf(prefs?.defaultEnablePython ?: false)
    var fileCreation by mutableStateOf(prefs?.defaultEnableFileCreation ?: true)
    var thinking by mutableStateOf(prefs?.defaultEnableThinking ?: false)
    var sysPrompt by mutableStateOf(prefs?.defaultEnableSystemPrompt ?: false)
    var mcp by mutableStateOf(prefs?.defaultEnableMcp ?: true)
    var thinkingLevel by mutableStateOf(prefs?.defaultThinkingLevel ?: "high")
    var thinkingBudget by mutableStateOf((prefs?.defaultThinkingBudget ?: 4096).toString())
    var effort by mutableStateOf(prefs?.defaultReasoningEffort ?: "medium")
    var safety by mutableStateOf(prefs?.defaultSafetySetting ?: "default")
    var autoSearch by mutableStateOf(prefs?.autoSearchOnLinks ?: true)
    var micMode by mutableStateOf(prefs?.micTranscribeMode ?: "stt_api")
    var sttModel by mutableStateOf(prefs?.sttModel ?: "gpt-4o-mini-transcribe")
    var llmPrompt by mutableStateOf(prefs?.llmTranscribePrompt.orEmpty())
    val llmPromptDefault = prefs?.llmTranscribePromptDefault.orEmpty()
    var tempTimeout by mutableStateOf((prefs?.tempChatTimeoutSeconds ?: 90).toString())

    val providerKeys = mutableStateMapOf<String, String>().apply {
        PROVIDER_KEY_FIELDS.forEach { put(it, prefs?.providerKeys?.get(it).orEmpty()) }
    }
    /** Model-specific keys: existing entries hold [SECRET_MASK], new ones the typed key. */
    val modelKeys = mutableStateMapOf<String, String>().apply { prefs?.modelApiKeys?.forEach { put(it, SECRET_MASK) } }
    var geminiBackend by mutableStateOf(prefs?.geminiBackend ?: "gemini_api")
    var vertexProject by mutableStateOf(prefs?.geminiVertexProject.orEmpty())
    var vertexLocation by mutableStateOf(prefs?.geminiVertexLocation ?: "global")
    var vertexJson by mutableStateOf(if (prefs?.geminiVertexCredentialsSet == true) SECRET_MASK else "")
    var googleProject by mutableStateOf(prefs?.googleProject.orEmpty())

    var userPrompt by mutableStateOf(prefs?.systemPrompt.orEmpty())
    var userPromptEnabled by mutableStateOf(prefs?.systemPromptEnabled ?: true)
    var applyGlobal by mutableStateOf(prefs?.applyGlobalSystemPrompt ?: true)
    var applyAutoNotices by mutableStateOf(prefs?.applyAutoSystemPromptNotices ?: true)
    val autoPrompts = mutableStateListOf<AutoSystemPrompt>().apply { addAll(prefs?.autoSystemPrompts.orEmpty()) }

    var themeColor by mutableStateOf(prefs?.themeColor?.ifBlank { null } ?: THEME_DEFAULT)
    var lightMode by mutableStateOf(prefs?.lightModeEnabled ?: false)
    var liquidGlass by mutableStateOf(prefs?.liquidGlassEnabled ?: false)

    var latencyMetrics by mutableStateOf(prefs?.enableLatencyMetrics ?: false)
    var clientDebugLog by mutableStateOf(prefs?.enableClientDebugLog ?: false)

    var skip2faGoogle by mutableStateOf(prefs?.skip2faOnGoogleLogin ?: false)
    var default2fa by mutableStateOf(prefs?.default2faMethod ?: "totp")
    var passkeyOnly by mutableStateOf(prefs?.passkeyOnlyLogin ?: false)

    fun updateAutoPrompt(key: String, transform: (AutoSystemPrompt) -> AutoSystemPrompt) {
        val index = autoPrompts.indexOfFirst { it.key == key }
        if (index >= 0) autoPrompts[index] = transform(autoPrompts[index])
    }

    /** Web `resetAutoSystemPromptConfigToCodeDefaults`: 全体適用 on, every row on with its default text (MCP keeps its toggle). */
    fun resetAutoPrompts() {
        applyAutoNotices = true
        for (i in autoPrompts.indices) {
            val row = autoPrompts[i]
            autoPrompts[i] = row.copy(enabled = if (row.mcpLocked) row.enabled else true, text = row.defaultText)
        }
    }

    /** Body of the Web save request, limited to the fields the native preferences API accepts. */
    fun payload(): JSONObject = JSONObject()
        .put("enter_to_send", enterToSend)
        .put("prompt_bar_mode", promptBarMode)
        .put("voice_studio_ui", voiceStudio)
        .put("default_model", defaultModel)
        .put("default_vision_model", visionModel)
        .put("use_last_chat_settings", useLast)
        .put("default_enable_search", search)
        .put("default_enable_url_context", urlContext)
        .put("default_enable_maps", maps)
        .put("default_enable_python", python)
        .put("default_enable_file_creation", fileCreation)
        .put("default_enable_thinking", thinking)
        .put("default_enable_system_prompt", sysPrompt)
        .put("default_enable_mcp", mcp)
        .put("default_thinking_level", thinkingLevel)
        .put("default_thinking_budget", thinkingBudget.toIntOrNull()?.coerceIn(0, 32768) ?: 4096)
        .put("default_reasoning_effort", effort)
        .put("default_safety_setting", safety)
        .put("auto_search_on_links", autoSearch)
        .put("mic_transcribe_mode", micMode)
        .put("stt_model", sttModel)
        .put("llm_transcribe_prompt", llmPrompt)
        .put("temp_chat_timeout_seconds", normalizeTempChatTimeout(tempTimeout))
        .put("system_prompt", userPrompt)
        .put("system_prompt_enabled", userPromptEnabled)
        .put("apply_global_system_prompt", applyGlobal)
        .put("apply_auto_system_prompt_notices", applyAutoNotices)
        .put("auto_system_prompt_notices_config", JSONObject().apply {
            autoPrompts.forEach { row ->
                put(row.key, JSONObject().put("enabled", if (row.mcpLocked) true else row.enabled).put("text", row.text))
            }
        })
        .put("theme_color", normalizeWebHex(themeColor) ?: THEME_DEFAULT)
        .put("light_mode_enabled", lightMode)
        .put("liquid_glass_enabled", liquidGlass)
        .put("enable_latency_metrics", latencyMetrics)
        .put("enable_client_debug_log", clientDebugLog)
        .put("skip_2fa_on_google_login", skip2faGoogle)
        .put("default_2fa_method", default2fa)
        .put("passkey_only_login", passkeyOnly)
        .apply {
            providerKeys.forEach { (field, value) -> put(field, value) }
            put("model_api_keys", JSONObject().apply { modelKeys.forEach { (model, key) -> put(model, key) } })
            put("gemini_backend", geminiBackend)
            put("gemini_vertex_project", vertexProject)
            put("gemini_vertex_location", vertexLocation)
            put("gemini_vertex_credentials_json", vertexJson)
            put("google_project", googleProject)
        }
}

/** Web `normalizeTemporaryChatTimeoutSeconds`: integer seconds clamped to 10–3600 (90 when invalid). */
internal fun normalizeTempChatTimeout(raw: String): Int =
    (raw.trim().ifEmpty { "0" }.toDoubleOrNull()?.takeIf { it.isFinite() }?.toInt() ?: 90)
        .coerceIn(TEMP_CHAT_TIMEOUT_MIN, TEMP_CHAT_TIMEOUT_MAX)

/** Web `maskApiKeyPreview`. */
internal fun maskApiKeyPreview(key: String): String = when {
    key.isEmpty() -> ""
    key.length <= 8 -> SECRET_MASK
    else -> "${key.take(4)}...${key.takeLast(4)}"
}
