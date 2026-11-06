package com.minashin1120.aiplayground.data.direct

import org.json.JSONArray
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SkillToolTest {
    private val skills = JSONArray("""
        [
          {"uuid":"s1","name":"Summariser","description":"Summarise text","instruction":"Use three bullets."},
          {"uuid":"s2","name":"Translator","description":"Translate to Japanese","instruction":"Keep the original tone."}
        ]
    """.trimIndent())

    @Test
    fun executesExactAndCaseInsensitiveNames() {
        assertEquals("Skill: Summariser\n\nUse three bullets.", SkillTool.execute(skills, "Summariser"))
        assertEquals("Skill: Translator\n\nKeep the original tone.", SkillTool.execute(skills, "translator"))
    }

    @Test
    fun unknownNameListsAvailableSkills() {
        val result = SkillTool.execute(skills, "Missing")
        assertTrue(result.startsWith("Error: no skill named"))
        assertTrue(result.contains("Summariser"))
        assertTrue(result.contains("Translator"))
    }

    @Test
    fun catalogWithInstructionsContainsInstructionText() {
        val catalog = SkillTool.catalogWithInstructions(skills)
        assertTrue(catalog.contains("Use three bullets."))
        assertTrue(catalog.contains("Keep the original tone."))
    }
}
