package com.minashin1120.aiplayground

import android.content.SharedPreferences
import com.minashin1120.aiplayground.data.*
import com.minashin1120.aiplayground.data.backend.ChatBackend
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject

/**
 * The send preparation half of the composer (Web prompt bar): the draft and quote, the model and
 * option switches with each model's rules, PromptCache, Coding targets, and the attachment row's
 * names. State lives in the view model; this class only reads it and reports through `mutable`.
 */
internal class ChatComposer(
    private val state: StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val scope: CoroutineScope,
    private val prefs: SharedPreferences,
    private val backend: () -> ChatBackend,
    private val token: () -> String,
    private val notify: (String) -> Unit,
    private val report: suspend (Throwable, String?) -> Unit,
    private val notifyChatOffline: (String?, String) -> Boolean,
) {
    /** Web `modelChosenManually`: a model the user picked in this chat survives edit/regenerate/branch restores. */
    var modelChosenManually = false

    fun draft(text: String) { mutable.update { it.copy(draft = text) } }
    /**
     * Text shared from another app: appended to the prompt input like the rich-paste insert.
     * Used by the share sheet and by 「プロンプトとして添付」 in the text-selection menu.
     */
    fun insertSharedText(text: String) {
        if (state.value.starting) {
            // Launched from the share sheet: the saved session is not loaded yet.
            scope.launch {
                state.first { !it.starting }
                insertSharedText(text)
            }
            return
        }
        if (state.value.banned) return
        mutable.update {
            it.copy(draft = if (it.draft.isBlank()) text else it.draft.trimEnd() + "\n\n" + text,
                composerFocusRequest = it.composerFocusRequest + 1)
        }
    }
    fun quoteMessage(text: String, sourceMessageId: Int? = null, sourceKey: String? = sourceMessageId?.toString()) {
        val quoted = text.trim()
        if (quoted.isBlank()) return
        mutable.update { it.copy(quote = quoted, quoteMessageId = sourceMessageId, quoteSourceId = sourceKey, composerFocusRequest = it.composerFocusRequest + 1) }
    }
    fun navigateToQuote() {
        val current = state.value
        if (current.quote.isBlank() || (current.quoteSourceId == null && current.quoteMessageId == null)) return
        mutable.update { it.copy(quoteNavigationRequest = it.quoteNavigationRequest + 1) }
    }
    fun clearQuote() { mutable.update { it.copy(quote = "", quoteSourceId = null, quoteMessageId = null) } }

    /**
     * Web `applySavedUserSystemPromptSettings`: after the user system prompt is saved from Settings or
     * Chat Instructions, the composer's SysPrompt switch follows it when its text or on/off changed.
     */
    fun applySavedPreferences(current: ChatState, saved: Preferences): ChatState {
        fun active(p: Preferences?) = p != null && p.systemPrompt.isNotBlank() && p.systemPromptEnabled
        val before = current.preferences
        val next = current.copy(preferences = saved)
        val changed = before == null || before.systemPrompt != saved.systemPrompt || before.systemPromptEnabled != saved.systemPromptEnabled
        if (!changed) return next
        val on = active(saved)
        return if (next.sysPromptSuppressed) next.copy(sysPromptRestore = on) else next.copy(enableSystemPrompt = on)
    }

    /**
     * Web `toggleOptions()` writes the forced checkbox values and moves the Thinking level / Effort
     * selects to an allowed option; the new values stay after switching to another model.
     */
    fun applyModelRules(current: ChatState): ChatState = with(current) {
        val rules = composerRules(model, mcpServers.any { it.enabled }, preferences?.geminiBackend ?: "gemini_api")
        val level = chipValues["thinking_level"].orEmpty()
        val effort = chipValues["reasoning_effort"].orEmpty()
        val nextLevel = if (!rules.effortAsThinking && level !in rules.thinkingLevels && rules.thinkingFallback != null) rules.thinkingFallback else level
        val nextEffort = when {
            rules.effortAsThinking -> if (effort !in rules.thinkingLevels && rules.thinkingFallback != null) rules.thinkingFallback else effort
            rules.effort.visible && effort !in rules.effortOptions -> rules.effortFallback
            else -> effort
        }
        // Web `toggleOptions()` remembers the SysPrompt choice while a model cannot use it and brings it back afterwards.
        val sysForced = rules.sysPrompt.forced
        val sysSuppressed = sysForced != null
        val sysChecked = when {
            sysForced != null -> sysForced
            sysPromptSuppressed && sysPromptRestore -> true
            else -> enableSystemPrompt
        }
        val sysRestore = sysSuppressed && (if (sysPromptSuppressed) sysPromptRestore else enableSystemPrompt)
        copy(
            enableSearch = rules.search.forced ?: enableSearch,
            enableUrlContext = rules.urls.forced ?: enableUrlContext,
            enableMaps = rules.maps.forced ?: enableMaps,
            enablePython = rules.python.forced ?: enablePython,
            enableSystemPrompt = sysChecked,
            sysPromptSuppressed = sysSuppressed,
            sysPromptRestore = sysRestore,
            enableThinking = rules.thinking.forced ?: enableThinking,
            enablePromptCache = rules.promptCache.forced ?: enablePromptCache,
            batchMode = batchMode && rules.batch.visible,
            chipValues = chipValues + mapOf("thinking_level" to nextLevel, "reasoning_effort" to nextEffort),
        )
    }

    /**
     * Web keeps each option's checkbox when the model changes; only the per-model rules force values
     * (applied when sending). OCR turns Canvas/Coding off; GPT Image and Ideogram Object Remover retain the mask.
     */
    fun chooseModel(model: String) {
        val info = state.value.account?.models?.firstOrNull { it.id == model && it.selectable } ?: return
        if (state.value.enablePromptCache) {
            val currentProvider = modelApiProvider(state.value.model)
            val nextProvider = modelApiProvider(info.id)
            if (currentProvider != null && nextProvider != null && currentProvider != nextProvider) {
                notify("PromptCache 有効中は他API（${PROVIDER_LABELS[nextProvider] ?: nextProvider}）のモデルに変更できません。現在: ${PROVIDER_LABELS[currentProvider] ?: currentProvider}")
                return
            }
        }
        val ocr = isMistralOcrModel(info.id)
        mutable.update { it.copy(model = model,
            canvasMode = it.canvasMode && !ocr,
            codingMode = it.codingMode && !ocr,
            codingTarget = if (ocr) null else it.codingTarget,
            imageMask = if (info.id.contains("gpt-image") || isIdeogramObjectRemoval(info.id, it.generationValues)) it.imageMask else null).let { next -> applyModelRules(next) } }
        state.value.account?.let { prefs.edit().putString("model_${it.id}", model).apply() }
    }

    /** Web `selectModelByUser`: a pick from the model list or welcome screen; PromptCache may refuse it. */
    fun chooseModelByUser(model: String) {
        chooseModel(model)
        if (state.value.model == model) modelChosenManually = true
    }
    fun toggleThinking() { mutable.update { it.copy(enableThinking = !it.enableThinking) } }
    fun toggleSearch() { mutable.update { it.copy(enableSearch = !it.enableSearch) } }
    fun toggleGeminiFreeTier() { mutable.update { it.copy(geminiFreeTier = !it.geminiFreeTier) } }
    fun toggleUrlContext() { mutable.update { it.copy(enableUrlContext = !it.enableUrlContext) } }
    fun toggleMaps() { mutable.update { it.copy(enableMaps = !it.enableMaps) } }
    fun toggleFileCreation() { mutable.update { it.copy(enableFileCreation = !it.enableFileCreation) } }
    fun toggleSystemPrompt() { mutable.update { it.copy(enableSystemPrompt = !it.enableSystemPrompt) } }
    fun togglePromptCache() {
        if (state.value.enablePromptCache) {
            clearPromptCache()
            return
        }
        mutable.update { it.copy(enablePromptCache = true) }
        val provider = modelApiProvider(state.value.model)
        val label = PROVIDER_LABELS[provider] ?: provider ?: "現在のAPI"
        notify("PromptCache を有効化しました。以降は $label 以外のモデルに変更できません。")
    }

    /** Turns PromptCache off and drops the saved provider lock for the open chat. */
    fun clearPromptCache() {
        val thread = state.value.selected
        val wasOn = state.value.enablePromptCache
        mutable.update { it.copy(enablePromptCache = false) }
        scope.launch {
            if (thread != null && notifyChatOffline(thread.id, "オフライン中はチャット設定を変更できません。")) {
                mutable.update { it.copy(enablePromptCache = wasOn) }
                return@launch
            }
            if (thread != null) {
                try {
                    backend().put(
                        "/api/threads/${thread.id}/settings",
                        JSONObject().put("enable_prompt_caching", false),
                        token(),
                    )
                } catch (e: Exception) {
                    mutable.update { it.copy(enablePromptCache = wasOn) }
                    report(e, null)
                    return@launch
                }
            }
            notify("プロンプトキャッシュを解除しました。")
        }
    }
    /** Web `enable-batch-mode` change: turning Batch on releases Coding, which Batch cannot run. */
    fun toggleBatchMode() {
        val releaseCoding = !state.value.batchMode && state.value.codingMode
        mutable.update { it.copy(batchMode = !it.batchMode, codingMode = if (releaseCoding) false else it.codingMode) }
        if (releaseCoding) notify("Batch APIではCoding Modeを利用できないため解除しました")
    }
    fun togglePython() { mutable.update { it.copy(enablePython = !it.enablePython) } }
    fun toggleMcp() { mutable.update { it.copy(enableMcp = !it.enableMcp) } }
    fun toggleSkills() { mutable.update { it.copy(enableSkills = !it.enableSkills) } }
    fun toggleCanvas() { mutable.update { it.copy(canvasMode = !it.canvasMode) } }
    fun toggleCoding() { mutable.update { it.copy(codingMode = !it.codingMode) } }
    /** Web `selectCodingTargetFromButton` / `clear-coding-target-btn`; selecting does not turn Coding on. */
    fun selectCodingTarget(target: CodingTarget?) {
        mutable.update { it.copy(codingTarget = target) }
        notify(when {
            target == null -> "最新のコードブロックを自動選択します"
            state.value.codingMode -> "Coding Modeの編集対象に設定しました"
            else -> "編集対象を選択しました。プロンプトバーのCodingをオンにすると使用します"
        })
    }
    fun setImageMask(reference: String?) { mutable.update { it.copy(imageMask = reference) } }
    /** Web `selectModel` while the vision picker is active. */
    fun setVisionModel(id: String) { mutable.update { it.copy(visionModel = id) } }
    /** Web `promptRowAttachmentName`: a blank name restores the default. */
    fun renameAttachment(reference: String, input: String) {
        val next = input.trim()
        mutable.update { current -> current.copy(attachments = current.attachments.map {
            if (it.reference != reference) it else it.copy(name = next.ifEmpty { it.defaultName })
        }) }
        notify(if (next.isEmpty()) "送信名をデフォルトに戻しました" else "送信名を更新しました")
    }
    fun generationOption(key: String, value: String) {
        if (key in COMPOSER_SELECT_DEFAULTS) {
            mutable.update { it.copy(chipValues = it.chipValues + (key to value)) }
            return
        }
        if (state.value.streaming) return
        mutable.update { current -> current.copy(generationValues = current.generationValues + (key to value)) }
    }
    /** Web `schedulePromptTokenEstimate` request (`POST /api/token_estimate`). */
    suspend fun estimatePromptTokens(model: String, message: String, quote: String, imageUrls: List<String>): JSONObject =
        backend().post("/api/token_estimate", JSONObject().put("model", model).put("message", message)
            .put("quote_text", quote).put("image_urls", JSONArray(imageUrls))
            .put("gemini_free_tier", state.value.geminiFreeTier), token())

    suspend fun estimatePromptTokenDetails(model: String, message: String, quote: String, imageUrls: List<String>): JSONObject =
        backend().post("/api/token_estimate", JSONObject().put("model", model).put("message", message)
            .put("quote_text", quote).put("image_urls", JSONArray(imageUrls)).put("include_details", true)
            .put("gemini_free_tier", state.value.geminiFreeTier), token())

    fun removeAttachment(reference: String) { mutable.update { it.copy(attachments = it.attachments.filterNot { a -> a.reference == reference }) } }
}
