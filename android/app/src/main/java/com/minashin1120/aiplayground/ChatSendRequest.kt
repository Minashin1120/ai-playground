package com.minashin1120.aiplayground

import com.minashin1120.aiplayground.data.Attachment
import com.minashin1120.aiplayground.data.COMPOSER_SELECT_DEFAULTS
import com.minashin1120.aiplayground.data.CodingCandidate
import com.minashin1120.aiplayground.data.ComposerRules
import com.minashin1120.aiplayground.data.MARKER_HINT_TEXT
import com.minashin1120.aiplayground.data.OptionRule
import com.minashin1120.aiplayground.data.historyCodingTargets
import com.minashin1120.aiplayground.data.planCodingSend
import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID

/**
 * Web `sendMessage`: `/chat_stream` へ送る本体（本文・添付・生成設定・Coding Mode・Skill・分岐）を組み立てる。
 * 画面の状態は読むだけで変更せず、スレッドと親は呼び出し元が決めた値を受け取る
 * （`ChatViewModel` の送信前チェック・送信・後処理とは別に読めるようにする）。
 */
internal object ChatSendRequest {
    /** 組み立て結果。入力の検証で送れないときは、画面に出す文言だけを返す。 */
    sealed interface Result {
        data class Ready(val body: JSONObject) : Result
        data class Rejected(val message: String) : Result
    }

    fun build(
        current: ChatState,
        items: List<Attachment>,
        generation: JSONObject,
        rules: ComposerRules,
        pendingParentId: Int?,
        disableAutoSearch: Boolean,
    ): Result {
        fun effective(value: Boolean, rule: OptionRule) = rule.forced ?: value
        val effort = current.chipValues["reasoning_effort"].orEmpty()
        val deepSeekNonThinking = current.model.lowercase().contains("deepseek") && effort.lowercase() == "none"
        // Web の Thinking ボタンと同じく、Effort が none のときは思考を有効にしない。
        val effortNonThinking = rules.effortAsThinking && effort.lowercase() == "none"
        val body = JSONObject().put("model", current.model).put("message", current.draft)
            .put("client_request_id", UUID.randomUUID().toString()).put("image_urls", JSONArray(items.map { it.reference }))
            .put("image_items", JSONArray().apply {
                items.forEach { put(JSONObject().put("path", it.reference).put("source", it.source).put("name", it.name)) }
            })
            .put("uploaded_image_urls", JSONArray(items.filter { it.source == "upload" }.map { it.reference }))
            .put("marker_system_prompt", if (current.attachments.any { it.edited }) MARKER_HINT_TEXT else JSONObject.NULL)
            .put("image_vision_model", current.visionModel ?: current.preferences?.defaultVisionModel ?: JSONObject.NULL)
            .put("enable_thinking", !deepSeekNonThinking && !effortNonThinking && effective(current.enableThinking, rules.thinking))
            .put("enable_search", effective(current.enableSearch, rules.search))
            .put("gemini_free_tier", current.geminiFreeTier)
            .put("enable_url_context", effective(current.enableUrlContext, rules.urls))
            .put("enable_maps", effective(current.enableMaps, rules.maps))
            .put("enable_file_creation", effective(current.enableFileCreation, rules.file))
            .put("enable_system_prompt", effective(current.enableSystemPrompt, rules.sysPrompt))
            .put("enable_prompt_caching", effective(current.enablePromptCache, rules.promptCache))
            .put("batch_mode", current.batchMode && rules.batch.visible).put("enable_python", effective(current.enablePython, rules.python))
            .put("enable_mcp", current.enableMcp && rules.mcp.visible)
            .put("enable_skills", current.enableSkills)
            .put("coding_mode", current.codingMode)
            .put("canvas_mode", current.canvasMode)
            .put("temporary_chat", current.selected?.isTemporary ?: current.newThreadTemporary)
            .put("disable_auto_search", disableAutoSearch)
        current.imageMask?.let { body.put("image_mask", it) }
        if (current.quote.isNotBlank()) {
            body.put("quote_text", current.quote)
            current.quoteMessageId?.let { body.put("quote_message_id", it) }
        }
        if (current.codingMode) {
            val history = historyCodingTargets(current.messages, current.editingMessageId)
            val selection = current.codingTarget?.takeIf { selected -> history.any { it.messageId == selected.messageId } }
            val plan = planCodingSend(current.draft, history, selection, current.model)
            plan.error?.let { return Result.Rejected(it) }
            val target = plan.target
            if (plan.active && target != null) {
                fun JSONObject.candidateFields(candidate: CodingCandidate) = put("id", candidate.candidateId)
                    .put("code", if (candidate.promptSource) JSONObject.NULL else candidate.code)
                    .put("language", candidate.language.ifBlank { "text" })
                    .put("source", if (candidate.promptSource) "prompt" else "history")
                    .put("explicit", candidate.explicit)
                body.put("coding_target", JSONObject().candidateFields(target).put("key", target.key)
                    .put("message_id", target.messageId ?: JSONObject.NULL))
                body.put("coding_candidates", JSONArray().apply {
                    plan.candidates.forEach { put(JSONObject().candidateFields(it).put("prompt_index", it.promptIndex ?: JSONObject.NULL)) }
                })
            } else body.put("coding_mode", false)
        }
        generation.keys().forEach { key -> body.put(key, generation.get(key)) }
        COMPOSER_SELECT_DEFAULTS.forEach { (key, fallback) -> body.put(key, current.chipValues[key] ?: fallback) }
        // Web part15: the server never reads Skill.instruction, the client sends it as the system prompt.
        current.selectedSkill?.let { skill ->
            body.put("skill_uuid", skill.uuid)
            body.put("system_prompt", skill.instruction).put("enable_system_prompt", true)
        }
        if (current.editingMessageId != null) {
            // Branch from the edited message's parent; send null explicitly for the first message.
            body.put("parent_id", pendingParentId ?: JSONObject.NULL)
            body.put("parent_id_explicit", true)
        } else {
            // Keep normal sends on the currently selected branch.
            current.leafId?.let { body.put("parent_id", it) }
        }
        current.selected?.let { body.put("thread_id", it.id) }
        return Result.Ready(body)
    }
}
