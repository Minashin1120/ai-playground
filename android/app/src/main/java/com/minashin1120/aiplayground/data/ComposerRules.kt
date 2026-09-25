package com.minashin1120.aiplayground.data

import java.util.Locale

/*
 * Which composer options the Web shows for a model, and in which state. This is a port of
 * chat_core part07 `toggleOptions()` with the helpers of part05/part06 (`isLlmModel`,
 * `isBatchModelKey`, `applyMcpPromptChipUi`, …). Keep the order of the checks identical: several
 * later steps undo earlier ones exactly as in the Web code (e.g. Maps ends up hidden for every
 * model that is not a DeepSeek or reasoning-effort model).
 */

/** State of one option: shown at all, dimmed (`opacity-50 pointer-events-none`), checkbox disabled, forced value. */
data class OptionRule(
    val visible: Boolean = true,
    val dimmed: Boolean = false,
    val disabled: Boolean = false,
    /** Non-null when the Web code sets the checkbox value; the user cannot change it while [disabled]. */
    val forced: Boolean? = null,
) {
    val interactive: Boolean get() = visible && !dimmed && !disabled
}

data class ComposerRules(
    val search: OptionRule,
    val urls: OptionRule,
    val maps: OptionRule,
    val python: OptionRule,
    val file: OptionRule,
    val mcp: OptionRule,
    val sysPrompt: OptionRule,
    val thinking: OptionRule,
    /** Enabled `<option>` values of the Thinking level select (min/low/medium/high). */
    val thinkingLevels: Set<String>,
    /** Value the select switches to when the current level is not allowed. */
    val thinkingFallback: String?,
    val budgetEnabled: Boolean,
    val effort: OptionRule,
    val effortOptions: List<String>,
    val effortFallback: String,
    val promptCache: OptionRule,
    val canvas: OptionRule,
    val coding: OptionRule,
    val batch: OptionRule,
    val mask: Boolean,
)

val THINKING_LEVELS = listOf("minimal", "low", "medium", "high")
val EFFORT_LEVELS = listOf("none", "low", "medium", "high", "xhigh", "max")

fun isMistralOcrModel(model: String): Boolean {
    val m = model.lowercase(Locale.ROOT)
    return m == "mistral-ocr-4-0" || m == "mistral-ocr-latest" || m.startsWith("mistral-ocr")
}

fun isGeminiImageModelKey(model: String): Boolean {
    val m = model.lowercase(Locale.ROOT)
    return m.contains("gemini") && (m.contains("image") || m.contains("nano"))
}

fun isGeminiVideoModelKey(model: String): Boolean {
    val m = model.lowercase(Locale.ROOT)
    return m.startsWith("veo-") || m.contains("omni-flash") || m.contains("omni-1.1-flash")
}

fun isGeminiMusicModelKey(model: String): Boolean = model.lowercase(Locale.ROOT).startsWith("lyria")

fun isGeminiEmbeddingModelKey(model: String): Boolean = model.lowercase(Locale.ROOT).contains("embedding")

/** Web `isLlmModel`. */
fun isLlmModel(model: String): Boolean {
    val m = model.lowercase(Locale.ROOT)
    if (isMistralOcrModel(m)) return false
    if (listOf("tts", "transcribe", "realtime", "voice-agent", "native-audio", "live", "image", "video").any { m.contains(it) } ||
        isGeminiVideoModelKey(m) || isGeminiMusicModelKey(m) || isGeminiEmbeddingModelKey(m)) return false
    if (m.contains("gemini") && (m.contains("image") || m.contains("nano"))) return false
    return m.contains("gpt") || m.contains("gemini") || m.contains("grok") || m.contains("deepseek") ||
        m.startsWith("deep-research-") || m.startsWith("antigravity-")
}

/** Web `isBatchModelKey`. */
fun isBatchModelKey(model: String): Boolean {
    val m = model.trim().lowercase(Locale.ROOT)
    if (m.startsWith("gpt-")) return !Regex("(image|audio|tts|transcribe|realtime|search)").containsMatchIn(m)
    if (m.startsWith("grok-")) return !Regex("(image|video|voice|audio|tts|realtime)").containsMatchIn(m)
    if (!m.startsWith("gemini-")) return false
    return !Regex("(embedding|video|veo|music|lyria|native-audio|tts|live|transcribe|agent|deep-research|robotics|computer-use)").containsMatchIn(m)
}

/** Web `mcpModelSupported`. */
fun mcpModelSupported(model: String): Boolean {
    val m = model.lowercase(Locale.ROOT)
    if (m.isBlank()) return false
    if (m.contains("claude") || m.startsWith("kimi")) return true
    return isLlmModel(m)
}

/** API provider used by the PromptCache model lock (Web `getModelApiProvider`). */
fun modelApiProvider(model: String): String? {
    val m = model.lowercase(Locale.ROOT).trim()
    if (m.isEmpty()) return null
    if (m.contains("claude")) return "anthropic"
    if (m.contains("deepseek")) return "deepseek"
    if (m.contains("grok") && !m.contains("gpt")) return "xai"
    if (m.contains("google-tts")) return "google"
    if (m.contains("gemini") || m.startsWith("veo-") || m.startsWith("lyria-") || m.startsWith("deep-research-") || m.startsWith("antigravity-")) return "gemini"
    return "openai"
}

val PROVIDER_LABELS = mapOf(
    "openai" to "OpenAI", "gemini" to "Gemini", "anthropic" to "Anthropic (Claude)",
    "xai" to "xAI (Grok)", "deepseek" to "DeepSeek", "google" to "Google Cloud",
)

private class MutableRule(var visible: Boolean = true, var dimmed: Boolean = false, var disabled: Boolean = false, var forced: Boolean? = null) {
    fun freeze() = OptionRule(visible, dimmed, disabled, forced)
    fun off() { forced = false }
}

/** Web `toggleOptions()` for [model]; [mcpEnabledServer] is whether any MCP server is enabled. */
fun composerRules(model: String, mcpEnabledServer: Boolean): ComposerRules {
    val ml = model.lowercase(Locale.ROOT)
    val isDeepSeek = ml.contains("deepseek")
    val isSearchModel = model == "gpt-5-search-api"
    val isTts = model.contains("tts")
    val isOcr = isMistralOcrModel(model)
    val isNb2Lite = ml.contains("gemini-3.1-flash-lite-image")
    val isNb2 = ml.contains("gemini-3.1-flash-image") && !isNb2Lite
    val isClaude = ml.contains("claude")
    val isCyber = ml == "gemini-3.8-flash-cyber"
    val llm = isLlmModel(model)
    val isGeminiImage = isGeminiImageModelKey(model)
    val isGrokImage = ml.contains("grok") && (ml.contains("imagine") || ml.contains("image")) && !ml.contains("video")
    val isGrokVideo = ml.contains("grok") && ml.contains("video")

    val search = MutableRule()
    val urls = MutableRule(visible = false)
    val maps = MutableRule(visible = false)
    val python = MutableRule()
    val sys = MutableRule()
    val thinking = MutableRule(visible = false)
    var levels = THINKING_LEVELS.toMutableSet()
    var fallback: String? = null
    var budget = false

    val promptCacheSupported = llm && !isDeepSeek && !isTts && !ml.contains("realtime") && !ml.contains("native-audio") && !ml.contains("live")
    val promptCache = if (promptCacheSupported) MutableRule() else MutableRule(dimmed = true, disabled = true, forced = false)

    when {
        isTts || isOcr -> {
            search.dimmed = true; search.off()
            urls.dimmed = true; urls.off()
            maps.dimmed = true; maps.off()
            python.dimmed = true; python.off()
            sys.off(); sys.disabled = true; sys.dimmed = true
        }
        isNb2 || isNb2Lite -> {
            maps.off(); maps.visible = false; maps.dimmed = true
            thinking.visible = true
            levels = mutableSetOf("minimal", "high")
            fallback = if (isNb2Lite) "minimal" else "high"
            thinking.disabled = false
            if (isNb2Lite) { search.off(); search.disabled = true; search.dimmed = true }
        }
        isGeminiImage -> { maps.off(); maps.visible = false; maps.dimmed = true }
        isClaude -> {
            thinking.visible = true
            budget = true
            levels = mutableSetOf()
            python.dimmed = true; python.off()
        }
        isCyber -> {
            thinking.visible = true
            thinking.forced = true; thinking.disabled = true
            levels = mutableSetOf("low", "medium", "high"); fallback = "medium"
            listOf(search, urls, maps, python).forEach { it.dimmed = true }
            listOf(search, maps, python).forEach { it.off(); it.disabled = true }
            urls.off(); urls.disabled = true
            sys.disabled = false; sys.dimmed = false
        }
        model.contains("gemini") && !isGeminiImage -> {
            thinking.visible = true
            urls.visible = true; urls.dimmed = false
            val gemini3 = model.contains("gemini-3")
            if (gemini3) { maps.visible = true; maps.dimmed = false } else { maps.off(); maps.visible = false; maps.dimmed = true }
            val flash = model.contains("flash")
            when (model) {
                "gemini-3.8-flash", "gemini-3.7-flash" -> { levels = mutableSetOf("low", "medium", "high"); fallback = "medium" }
                "gemini-3.6-flash" -> { levels = mutableSetOf("medium", "high"); fallback = "medium" }
                "gemini-3.5-flash-lite" -> { levels = mutableSetOf("minimal", "medium", "high"); fallback = "minimal" }
                else -> { levels = if (flash) THINKING_LEVELS.toMutableSet() else mutableSetOf("low", "high"); fallback = if (flash) null else "high" }
            }
            if (gemini3) { thinking.forced = true; thinking.disabled = true } else thinking.disabled = false
            budget = model.contains("gemini-2.5")
        }
    }

    val supportsEffort = llm && (listOf("gpt-5", "o1", "o3", "grok-4.3", "grok-4.5", "grok-4.6", "grok-4.20-0309-reasoning", "grok-build", "multi-agent")
        .any { ml.contains(it) } || (ml.contains("gpt") && !ml.contains("tts")))
    var effortVisible = false
    when {
        supportsEffort -> { effortVisible = true; search.dimmed = false }
        isDeepSeek -> {
            effortVisible = true
            search.off(); search.disabled = true; search.dimmed = true
            urls.off(); urls.dimmed = true
            maps.off(); maps.dimmed = true
        }
        !isOcr -> { search.dimmed = false; maps.off(); maps.visible = false; maps.dimmed = true }
    }

    if (isTts) python.dimmed = true
    else {
        python.dimmed = false
        if ((!isGeminiImage || isNb2) && !model.contains("gpt-image")) { sys.disabled = false; sys.dimmed = false }
    }
    if ((isGeminiImage && !isNb2) || model.contains("gpt-image") || isGrokImage || isGrokVideo || isOcr) {
        sys.off(); sys.disabled = true; sys.dimmed = true
    }
    if (llm) { python.visible = true; python.disabled = false } else { python.off(); python.disabled = true; python.visible = false }
    if (isSearchModel) {
        search.forced = true; search.disabled = true; search.dimmed = true
        python.off(); python.disabled = true; python.dimmed = true
    } else if (!model.contains("tts") && !isOcr && !isDeepSeek && !isNb2Lite) {
        search.disabled = false
    }
    if (isCyber) {
        listOf(search, maps, python).forEach { it.off(); it.disabled = true }
        urls.off(); urls.disabled = true
        listOf(search, urls, maps, python).forEach { it.dimmed = true }
    }

    val gpt56 = ml == "gpt-5.6" || ml.startsWith("gpt-5.6-")
    val deepSeekFlash = ml in setOf("deepseek-v4.1-flash", "deepseek-v4-flash-0731", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp")
    val deepSeekPro = ml == "deepseek-v4-pro"
    val grok45 = ml.contains("grok-4.5")
    val grok46 = ml.contains("grok-4.6")
    val effortOptions = EFFORT_LEVELS.filter { value ->
        when (value) {
            "max" -> gpt56 || deepSeekFlash || deepSeekPro
            "xhigh" -> grok46 || ml.contains("multi-agent") || gpt56
            "medium" -> ml.contains("grok-4.3") || grok45 || grok46 || ml.contains("grok-4.20-0309-reasoning") ||
                ml.contains("grok-build") || ml.contains("multi-agent") || ml.contains("gpt-5") || ml.contains("o1") || ml.contains("o3")
            "none" -> ml.contains("grok-4.3") || ml.contains("grok-build") || ml.contains("gpt-5") || deepSeekFlash || deepSeekPro
            "low" -> !deepSeekPro
            else -> true
        }
    }
    val chipDim = isOcr
    return ComposerRules(
        search = search.freeze(),
        urls = urls.freeze(),
        maps = maps.freeze(),
        python = python.freeze(),
        file = OptionRule(),
        mcp = OptionRule(visible = mcpModelSupported(model) && mcpEnabledServer),
        sysPrompt = sys.freeze(),
        thinking = thinking.freeze(),
        thinkingLevels = levels,
        thinkingFallback = fallback,
        budgetEnabled = budget,
        effort = OptionRule(visible = effortVisible),
        effortOptions = effortOptions,
        effortFallback = if (isDeepSeek) "high" else "medium",
        promptCache = promptCache.freeze(),
        canvas = OptionRule(dimmed = chipDim),
        coding = OptionRule(dimmed = chipDim),
        batch = OptionRule(visible = isBatchModelKey(model), disabled = !isBatchModelKey(model)),
        mask = model.contains("gpt-image"),
    )
}

/** Web selects shared by every model (`#thinking-level`, `#thinking-budget`, `#reasoning-effort`, `#safety-setting`). */
val COMPOSER_SELECT_DEFAULTS = mapOf(
    "thinking_level" to "high",
    "thinking_budget" to "4096",
    "reasoning_effort" to "medium",
    "safety_setting" to "default",
)

/** Text and tone of `#prompt-token-estimate` (Web `renderPromptTokenEstimate`); null hides the line. */
data class TokenEstimateLine(val text: String, val tone: String)

fun tokenEstimateLine(hasInput: Boolean, pending: Boolean, data: org.json.JSONObject?): TokenEstimateLine? {
    if (!hasInput) return null
    if (pending) return TokenEstimateLine("入力トークンを計算中...", "muted")
    if (data == null) return TokenEstimateLine("入力トークンを計算できませんでした", "error")
    if (!data.optBoolean("countable")) return TokenEstimateLine("このモデルは入力トークン表示対象外です", "muted")
    val notes = buildList {
        data.optInt("files_non_text").takeIf { it > 0 }?.let { add("非テキスト${it}件は0換算") }
        data.optInt("files_missing").takeIf { it > 0 }?.let { add("未検出${it}件") }
        data.optInt("files_error").takeIf { it > 0 }?.let { add("失敗${it}件") }
    }
    val noteText = if (notes.isEmpty()) "" else " ・ ${notes.joinToString(" / ")}"
    return TokenEstimateLine(
        "入力見積: ${data.optLong("tokens_total")} tokens (本文 ${data.optLong("tokens_prompt")} / ファイル ${data.optLong("tokens_files")})$noteText",
        "count",
    )
}
