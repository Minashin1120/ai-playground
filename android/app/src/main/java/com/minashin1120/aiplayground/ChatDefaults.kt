package com.minashin1120.aiplayground

/** The startup composer and New Chat use the same account preferences, never the device's saved model. */
internal fun ChatState.withNewChatDefaults(preserveSelectedModel: Boolean = false): ChatState {
    val saved = preferences ?: return this
    if (selected != null || streaming) return this
    val useLast = saved.useLastChatSettings && saved.lastModel.isNotBlank()
    val preferredModel = if (useLast) saved.lastModel else saved.defaultModel
    val selectable = account?.models?.firstOrNull { it.id == preferredModel && it.selectable }?.id
    return copy(
        model = if (preserveSelectedModel) model else selectable ?: model,
        enableThinking = if (useLast) saved.lastEnableThinking else saved.defaultEnableThinking,
        enableSearch = if (useLast) saved.lastEnableSearch else saved.defaultEnableSearch,
        enableUrlContext = if (useLast) saved.lastEnableUrlContext else saved.defaultEnableUrlContext,
        enableMaps = if (useLast) saved.lastEnableMaps else saved.defaultEnableMaps,
        enablePython = if (useLast) saved.lastEnablePython else saved.defaultEnablePython,
        enableFileCreation = if (useLast) saved.lastEnableFileCreation else saved.defaultEnableFileCreation,
        enableSystemPrompt = if (useLast) saved.lastEnableSystemPrompt else saved.defaultEnableSystemPrompt,
        enableMcp = if (useLast) saved.lastEnableMcp else saved.defaultEnableMcp,
        sysPromptSuppressed = false,
        sysPromptRestore = false,
        chipValues = chipValues + mapOf(
            "thinking_level" to if (useLast) saved.lastThinkingLevel else saved.defaultThinkingLevel,
            "thinking_budget" to (if (useLast) saved.lastThinkingBudget else saved.defaultThinkingBudget).toString(),
            "reasoning_effort" to if (useLast) saved.lastReasoningEffort else saved.defaultReasoningEffort,
            "safety_setting" to if (useLast) saved.lastSafetySetting else saved.defaultSafetySetting,
        ),
    )
}
