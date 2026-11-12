package com.minashin1120.aiplayground

import com.minashin1120.aiplayground.data.Skill
import com.minashin1120.aiplayground.data.backend.ChatBackend
import com.minashin1120.aiplayground.data.parseSkills
import com.minashin1120.aiplayground.data.replaceSkillMention
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import org.json.JSONObject

/**
 * Skills (Web `loadSkills` / `activateSkill` / `clearActiveSkill` / `selectSkillSuggestion`): the
 * list of Skill definitions, saving and deleting, and the Skill applied to the open chat. State
 * lives in the view model; this class only reads it and reports through `mutable`.
 */
internal class ChatSkills(
    private val state: StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val scope: CoroutineScope,
    private val backend: () -> ChatBackend,
    private val token: () -> String,
    private val notify: (String) -> Unit,
    private val notifyOffline: (String) -> Boolean,
    private val notifyChatOffline: (String?, String) -> Boolean,
    private val report: suspend (Throwable, String?) -> Unit,
) {
    suspend fun fetch() {
        val skills = parseSkills(backend().getArray("/api/skills", token()))
        mutable.update { current ->
            current.copy(skills = skills, selectedSkill = current.selectedSkill?.let { selected -> skills.firstOrNull { it.uuid == selected.uuid } })
        }
    }

    fun load() {
        scope.launch {
            if (notifyOffline("オフライン中はスキルを読み込めません。")) return@launch
            mutable.update { it.copy(skillsBusy = true) }
            try { fetch() } catch (e: Exception) { report(e, null) } finally { mutable.update { it.copy(skillsBusy = false) } }
        }
    }

    fun save(uuid: String?, name: String, description: String, instruction: String, onDone: (Boolean) -> Unit) {
        scope.launch {
            if (notifyOffline("オフライン中はスキルを保存できません。")) { onDone(false); return@launch }
            mutable.update { it.copy(skillsBusy = true) }
            try {
                val payload = JSONObject().put("name", name).put("description", description)
                    .put("instruction", instruction)
                if (uuid.isNullOrBlank()) backend().post("/api/skills", payload, token())
                else backend().put("/api/skills/$uuid", payload, token())
                fetch()
                mutable.update { current -> current.copy(selectedSkill = current.selectedSkill?.let { selected -> current.skills.firstOrNull { it.uuid == selected.uuid } }) }
                onDone(true)
            } catch (e: Exception) { report(e, null); onDone(false) }
            finally { mutable.update { it.copy(skillsBusy = false) } }
        }
    }

    fun delete(skill: Skill) { scope.launch {
        if (notifyOffline("オフライン中はスキルを削除できません。")) return@launch
        try {
            backend().delete("/api/skills/${skill.uuid}", token())
            mutable.update { current -> current.copy(
                skills = current.skills.filterNot { it.uuid == skill.uuid },
                selectedSkill = current.selectedSkill?.takeIf { it.uuid != skill.uuid },
            ) }
        } catch (e: Exception) { report(e, null) }
    } }

    /**
     * Web `activateSkill` / `clearActiveSkill`: in an open chat the Skill is announced and saved as the
     * chat's `last_skill_uuid` right away; in a new chat it waits for the first message.
     */
    fun choose(skill: Skill?) {
        mutable.update { it.copy(selectedSkill = skill) }
        val thread = state.value.selected ?: return
        if (skill != null) notify("Skill \"${skill.name}\" をこのチャットに適用しました")
        if (notifyChatOffline(thread.id, "オフライン中はスキルの適用を同期できません。")) return
        scope.launch {
            runCatching {
                backend().put("/api/mobile/v1/preferences",
                    JSONObject().put("last_skill_uuid", skill?.uuid ?: JSONObject.NULL).put("thread_id", thread.id), token())
            }
        }
    }

    /** Web `selectSkillSuggestion`: applies the Skill picked from the `@` candidates and trims the mention. */
    fun applyMention(skill: Skill) {
        mutable.update { it.copy(draft = replaceSkillMention(it.draft), composerFocusRequest = it.composerFocusRequest + 1) }
        choose(skill)
    }
}
