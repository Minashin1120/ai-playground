package com.minashin1120.aiplayground

import com.minashin1120.aiplayground.data.COMPOSER_SELECT_DEFAULTS
import com.minashin1120.aiplayground.data.Attachment
import com.minashin1120.aiplayground.data.ChatMessage
import com.minashin1120.aiplayground.data.activeBranchPath
import com.minashin1120.aiplayground.data.generationPanels
import com.minashin1120.aiplayground.data.latestLeafId
import com.minashin1120.aiplayground.data.numericId
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update

/**
 * プロンプトオプションの復元・編集・再生成・分岐の切替だけを持つ協作者
 * （Web `restorePromptOptionsForEdit` / `beginEditMessage` / `regenerateMessage` / 分岐の ‹ ›）。
 * 入力欄の状態は `composer`、編集前の設定の退避は `pendingSend` に置き、送信は `send` へ委ねる。
 */
internal class ChatEditing(
    private val state: StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val composer: ChatComposer,
    private val pendingSend: ChatPendingState,
    private val notify: (String) -> Unit,
    private val send: () -> Unit,
) {
    private fun ChatState.withPromptOptionsFrom(message: ChatMessage): ChatState {
        val options = message.promptOptions
        // Web `restorePromptOptionsForEdit`: a model the user picked stays; the prompt's options still come back.
        val modelId = if (composer.modelChosenManually) model else message.model.takeIf { id ->
            account?.models?.any { it.id == id && it.selectable } == true
        } ?: model
        val selects = COMPOSER_SELECT_DEFAULTS.mapValues { (key, fallback) ->
            if (options?.has(key) == true && !options.isNull(key)) options.optString(key) else chipValues[key] ?: fallback
        }
        val generation = generationValues.toMutableMap()
        generationPanels(modelId, generation).flatMap { it.fields }.forEach { field ->
            if (options?.has(field.key) == true && !options.isNull(field.key)) {
                generation[field.key] = options.opt(field.key).toString()
            }
        }
        fun savedBoolean(key: String, current: Boolean): Boolean =
            if (options?.has(key) == true && !options.isNull(key)) options.optBoolean(key) else current
        return composer.applyModelRules(copy(
            model = modelId,
            imageMask = options?.optString("image_mask")?.takeIf { it.isNotBlank() },
            enableSearch = savedBoolean("enable_search", enableSearch),
            enableUrlContext = savedBoolean("enable_url_context", enableUrlContext),
            enableMaps = savedBoolean("enable_maps", enableMaps),
            enablePython = savedBoolean("enable_python", enablePython),
            enableFileCreation = savedBoolean("enable_file_creation", enableFileCreation),
            enableMcp = savedBoolean("enable_mcp", enableMcp),
            enableSkills = savedBoolean("enable_skills", enableSkills),
            enableThinking = savedBoolean("enable_thinking", enableThinking),
            enablePromptCache = if (composer.modelChosenManually) enablePromptCache else savedBoolean("enable_prompt_caching", enablePromptCache),
            batchMode = savedBoolean("batch_mode", batchMode),
            canvasMode = savedBoolean("canvas_mode", canvasMode),
            codingMode = savedBoolean("coding_mode", codingMode),
            visionModel = options?.optString("image_vision_model")?.takeIf { it.isNotBlank() } ?: visionModel,
            chipValues = selects, generationValues = generation,
        ))
    }

    fun withLatestPromptOptions(current: ChatState): ChatState {
        if (current.editingMessageId != null) return current
        val latest = current.messages.lastOrNull { it.role == "user" }?.takeIf { it.promptOptions != null } ?: return current
        return current.withPromptOptionsFrom(latest)
    }

    /** Prepares the composer to edit a user message, branching from its parent. */
    fun beginEdit(message: ChatMessage) {
        if (message.role != "user") return
        if (state.value.editingMessageId == null || pendingSend.settingsBeforeEdit == null)
            pendingSend.settingsBeforeEdit = PromptEditSettings.capture(state.value, composer.modelChosenManually)
        pendingSend.parentId = message.parentId
        // Web `beginEditMessage`: the message's own quote comes back (or the quote bar clears) and the input gets focus.
        mutable.update { it.withPromptOptionsFrom(message).copy(
            editingMessageId = message.id,
            draft = message.content,
            attachments = message.files.map { reference -> Attachment(reference.substringAfterLast('/'), reference) },
            quote = message.quote,
            quoteSourceId = null,
            quoteMessageId = null,
            composerFocusRequest = it.composerFocusRequest + 1,
        ) }
    }

    /** Re-sends the user message that produced an assistant reply, creating a sibling branch. */
    fun regenerate(message: ChatMessage) {
        if (message.role != "assistant" || state.value.busy) return
        // Web `regenerateMessage`; while an answer streams, send() shows the same waiting notice as Web.
        val parent = message.parentId?.let { pid -> state.value.allMessages.firstOrNull { numericId(it) == pid } }
            ?: run { notify("再生成できるメッセージが見つかりません"); return }
        beginEdit(parent)
        send()
    }

    fun cancelEdit() {
        val saved = pendingSend.settingsBeforeEdit.takeIf { state.value.editingMessageId != null }
        pendingSend.clearEdit()
        if (saved != null) composer.modelChosenManually = saved.modelChosenManually
        mutable.update { current ->
            val cleared = current.copy(editingMessageId = null, draft = "", attachments = emptyList(), quote = "", quoteSourceId = null, quoteMessageId = null)
            saved?.restore(cleared) ?: cleared
        }
    }

    /** Switches the active path to the branch that contains [targetMessageId]. */
    fun switchBranch(targetMessageId: Int) {
        val all = state.value.allMessages
        val leaf = latestLeafId(all, targetMessageId)
        // Web keeps ‹ › choices for this visit only; reopening shows the pinned or the latest branch.
        mutable.update { it.copy(leafId = leaf, messages = activeBranchPath(all, leaf))
            .withLatestPromptOptions() }
    }

    fun switchBranchByIndex(siblings: List<ChatMessage>, index: Int) {
        siblings.getOrNull(index)?.let { numericId(it)?.let(::switchBranch) }
    }
}
