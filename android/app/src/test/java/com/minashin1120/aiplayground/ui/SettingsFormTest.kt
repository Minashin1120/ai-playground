package com.minashin1120.aiplayground.ui

import com.minashin1120.aiplayground.data.SECRET_MASK
import com.minashin1120.aiplayground.data.parsePreferences
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test

class SettingsFormTest {
    @Test
    fun reloadKeysTakesImportedKeysIntoOpenForm() {
        val form = SettingsForm(parsePreferences(JSONObject().put("username", "user")))
        form.providerKeys["openai_key"] = "sk-typed"
        val imported = parsePreferences(JSONObject()
            .put("username", "user")
            .put("ideogram_key", SECRET_MASK)
            .put("model_api_keys", JSONObject().put("gpt-5.5", SECRET_MASK))
            .put("gemini_vertex_credentials_json", SECRET_MASK))

        form.reloadKeys(imported)

        assertEquals(SECRET_MASK, form.providerKeys["ideogram_key"])
        assertEquals("sk-typed", form.providerKeys["openai_key"])
        assertEquals(SECRET_MASK, form.modelKeys["gpt-5.5"])
        assertEquals(SECRET_MASK, form.vertexJson)
        val payload = form.payload()
        assertEquals(SECRET_MASK, payload.getString("ideogram_key"))
        assertEquals(SECRET_MASK, payload.getJSONObject("model_api_keys").getString("gpt-5.5"))
    }
}
