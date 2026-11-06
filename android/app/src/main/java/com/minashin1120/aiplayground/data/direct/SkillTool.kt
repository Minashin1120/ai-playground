package com.minashin1120.aiplayground.data.direct

import org.json.JSONArray
import org.json.JSONObject

/**
 * Server `skills.py` equivalent for serverless mode: the model sees the saved skill names and short
 * descriptions and calls `use_skill` to load the full instructions.
 */
object SkillTool {
    const val NAME = "use_skill"
    private const val INSTRUCTION_LIMIT = 20_000
    private const val DESCRIPTION_LIMIT = 300

    private const val DESCRIPTION =
        "Load the full instructions for one of the user's saved skills. " +
            "Call this when the user asks to use a skill, or when a skill's short description clearly " +
            "matches the request. Use the exact skill name from the available skills list. " +
            "Never invent a skill that is not listed."

    private val parameters = JSONObject()
        .put("type", "object")
        .put("properties", JSONObject().put("name", JSONObject()
            .put("type", "string")
            .put("description", "Exact skill name from the available skills list.")))
        .put("required", JSONArray().put("name"))

    fun openAiChatDefinition(): JSONObject = JSONObject()
        .put("type", "function")
        .put("function", JSONObject().put("name", NAME).put("description", DESCRIPTION).put("parameters", parameters))

    fun openAiResponsesDefinition(): JSONObject = JSONObject()
        .put("type", "function").put("name", NAME).put("description", DESCRIPTION).put("parameters", parameters)

    fun anthropicDefinition(): JSONObject = JSONObject()
        .put("name", NAME).put("description", DESCRIPTION).put("input_schema", parameters)

    fun geminiDeclaration(): JSONObject = JSONObject()
        .put("name", NAME).put("description", DESCRIPTION).put("parameters", parameters)

    /** Returns the instruction text for the model; unknown names list the available skills. */
    fun execute(skills: JSONArray?, name: String): String {
        val query = name.trim()
        val rows = skills ?: JSONArray()
        var target: JSONObject? = null
        for (index in 0 until rows.length()) {
            val row = rows.optJSONObject(index) ?: continue
            if (row.optString("name") == query) { target = row; break }
        }
        if (target == null && query.isNotEmpty()) {
            for (index in 0 until rows.length()) {
                val row = rows.optJSONObject(index) ?: continue
                if (row.optString("name").equals(query, ignoreCase = true)) { target = row; break }
            }
        }
        if (target == null) {
            val available = (0 until rows.length()).mapNotNull { rows.optJSONObject(it)?.optString("name")?.takeIf { it.isNotBlank() } }
            val suffix = if (available.isEmpty()) "" else " Available skills: ${available.joinToString(", ")}"
            return "Error: no skill named \"$query\".$suffix"
        }
        val skillName = target.optString("name")
        val instruction = target.optString("instruction").trim().let {
            if (it.length > INSTRUCTION_LIMIT) it.take(INSTRUCTION_LIMIT).trimEnd() + "\n\n[Instructions truncated]" else it
        }
        return if (instruction.isEmpty()) "Skill: $skillName" else "Skill: $skillName\n\n$instruction"
    }

    /** Fallback for providers that cannot run tools: all instructions in the catalog. */
    fun catalogWithInstructions(skills: JSONArray?): String {
        val rows = skills ?: return ""
        return (0 until rows.length()).mapNotNull { index ->
            val row = rows.optJSONObject(index) ?: return@mapNotNull null
            val name = row.optString("name").trim()
            if (name.isEmpty()) return@mapNotNull null
            val description = row.optString("description").replace("\r\n", "\n").trim()
                .let { if (it.length > DESCRIPTION_LIMIT) it.take(DESCRIPTION_LIMIT).trimEnd() + "…" else it }
            val instruction = row.optString("instruction").trim().let {
                if (it.length > INSTRUCTION_LIMIT) it.take(INSTRUCTION_LIMIT).trimEnd() + "\n\n[Instructions truncated]" else it
            }
            "### $name\n$description\nInstructions:\n$instruction"
        }.joinToString("\n\n")
    }
}
