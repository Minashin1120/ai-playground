package com.minashin1120.aiplayground.data.local

import com.minashin1120.aiplayground.data.SECRET_MASK
import com.minashin1120.aiplayground.data.secure.EncryptedFileStore
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import javax.crypto.KeyGenerator

class LocalSettingsStoreTest {
    @get:Rule val folder = TemporaryFolder()
    private val crypto = KeyGenerator.getInstance("AES").apply { init(256) }.generateKey().let { k -> EncryptedFileStore(byteArrayOf(1)) { k } }
    private val secretKey = KeyGenerator.getInstance("AES").apply { init(256) }.generateKey()
    private val secrets = EncryptedFileStore(byteArrayOf(2)) { secretKey }
    private val settings by lazy { LocalSettingsStore(folder.root, crypto, secrets) }

    @Test fun keysAreMaskedAndTheMaskKeepsTheStoredValue() {
        settings.savePreferences(JSONObject().put("gemini_key", "AIza-real").put("default_model", "gpt-5.5")
            .put("model_api_keys", JSONObject().put("claude-opus-4-6", "sk-ant-model")))
        val payload = settings.preferencesPayload("この端末")
        assertEquals(SECRET_MASK, payload.getString("gemini_key"))
        assertEquals("", payload.getString("openai_key"))
        assertEquals(SECRET_MASK, payload.getJSONObject("model_api_keys").getString("claude-opus-4-6"))
        assertEquals("gpt-5.5", payload.getString("default_model"))
        assertFalse(payload.toString().contains("AIza-real"))
        settings.savePreferences(JSONObject().put("gemini_key", SECRET_MASK).put("model_api_keys", JSONObject().put("claude-opus-4-6", SECRET_MASK)))
        assertEquals("AIza-real", settings.apiKeyFor("gemini-3.6-flash"))
        assertEquals("sk-ant-model", settings.apiKeyFor("CLAUDE-OPUS-4-6"))
        assertNull(settings.apiKeyFor("gpt-5.5"))
        settings.savePreferences(JSONObject().put("gemini_key", ""))
        assertNull(settings.apiKeyFor("gemini-3.6-flash"))
        // Keys are not in the settings file and are encrypted with their own key.
        assertFalse(folder.root.resolve("preferences.enc").readBytes().toString(Charsets.ISO_8859_1).contains("sk-ant"))
    }

    @Test fun importKeepsExistingKeysUnlessOverwriting() {
        settings.savePreferences(JSONObject().put("openai_key", "sk-device"))
        settings.importSecrets(JSONObject().put("openai_key", "sk-server").put("xai_key", "xai-server"), overwrite = false)
        assertEquals("sk-device", settings.providerKey("openai_key"))
        assertEquals("xai-server", settings.providerKey("xai_key"))
        settings.importSecrets(JSONObject().put("openai_key", "sk-server"), overwrite = true)
        assertEquals("sk-server", settings.providerKey("openai_key"))
        assertTrue(settings.hasAnyKey())
    }

    @Test fun gemsAndCachedAccountSettings() {
        val saved = settings.saveGem(null, JSONObject().put("name", "翻訳").put("instruction", "英訳して"))
        val uuid = saved.getString("uuid")
        settings.saveGem(uuid, JSONObject().put("name", "翻訳2"))
        assertEquals("翻訳2", settings.gems().getJSONObject(0).getString("name"))
        assertEquals("英訳して", settings.gems().getJSONObject(0).getString("instruction"))
        settings.deleteGem(uuid)
        assertEquals(0, settings.gems().length())
        settings.cacheServerPreferences(JSONObject().put("system_prompt", "丁寧に").put("openai_key", SECRET_MASK))
        assertEquals("丁寧に", settings.cachedServerPreferences()!!.getString("system_prompt"))
        assertFalse(settings.cachedServerPreferences()!!.has("openai_key"))
    }
}
