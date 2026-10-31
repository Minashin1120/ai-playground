package com.minashin1120.aiplayground

import com.minashin1120.aiplayground.data.CodingTarget

/** Composer settings from before the first edit; never persisted or synchronized. */
internal data class PromptEditSettings(
    val modelChosenManually: Boolean,
    val model: String,
    val enableThinking: Boolean,
    val enableSearch: Boolean,
    val enableUrlContext: Boolean,
    val enableMaps: Boolean,
    val enablePython: Boolean,
    val enableFileCreation: Boolean,
    val enableMcp: Boolean,
    val enablePromptCache: Boolean,
    val enableSystemPrompt: Boolean,
    val sysPromptSuppressed: Boolean,
    val sysPromptRestore: Boolean,
    val batchMode: Boolean,
    val canvasMode: Boolean,
    val codingMode: Boolean,
    val codingTarget: CodingTarget?,
    val visionModel: String?,
    val imageMask: String?,
    val chipValues: Map<String, String>,
    val generationValues: Map<String, String>,
) {
    fun restore(current: ChatState): ChatState = current.copy(
        model = model,
        enableThinking = enableThinking,
        enableSearch = enableSearch,
        enableUrlContext = enableUrlContext,
        enableMaps = enableMaps,
        enablePython = enablePython,
        enableFileCreation = enableFileCreation,
        enableMcp = enableMcp,
        enablePromptCache = enablePromptCache,
        enableSystemPrompt = enableSystemPrompt,
        sysPromptSuppressed = sysPromptSuppressed,
        sysPromptRestore = sysPromptRestore,
        batchMode = batchMode,
        canvasMode = canvasMode,
        codingMode = codingMode,
        codingTarget = codingTarget,
        visionModel = visionModel,
        imageMask = imageMask,
        chipValues = chipValues,
        generationValues = generationValues,
    )

    companion object {
        fun capture(state: ChatState, modelChosenManually: Boolean): PromptEditSettings = PromptEditSettings(
            modelChosenManually = modelChosenManually,
            model = state.model,
            enableThinking = state.enableThinking,
            enableSearch = state.enableSearch,
            enableUrlContext = state.enableUrlContext,
            enableMaps = state.enableMaps,
            enablePython = state.enablePython,
            enableFileCreation = state.enableFileCreation,
            enableMcp = state.enableMcp,
            enablePromptCache = state.enablePromptCache,
            enableSystemPrompt = state.enableSystemPrompt,
            sysPromptSuppressed = state.sysPromptSuppressed,
            sysPromptRestore = state.sysPromptRestore,
            batchMode = state.batchMode,
            canvasMode = state.canvasMode,
            codingMode = state.codingMode,
            codingTarget = state.codingTarget,
            visionModel = state.visionModel,
            imageMask = state.imageMask,
            chipValues = state.chipValues.toMap(),
            generationValues = state.generationValues.toMap(),
        )
    }
}
