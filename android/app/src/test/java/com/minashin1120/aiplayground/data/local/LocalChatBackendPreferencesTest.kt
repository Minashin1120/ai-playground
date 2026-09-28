package com.minashin1120.aiplayground.data.local

import com.minashin1120.aiplayground.data.direct.DirectHttp
import com.minashin1120.aiplayground.data.direct.DirectRouter
import com.minashin1120.aiplayground.data.direct.ServerlessDefaults
import com.minashin1120.aiplayground.data.parsePreferences
import com.minashin1120.aiplayground.data.secure.EncryptedFileStore
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import javax.crypto.KeyGenerator

class LocalChatBackendPreferencesTest {
    @get:Rule val folder = TemporaryFolder()

    @Test fun savedPromptBarModeIsReturnedImmediatelyWithoutReloading() = runBlocking {
        fun encryptedStore(id: Byte): EncryptedFileStore {
            val key = KeyGenerator.getInstance("AES").apply { init(256) }.generateKey()
            return EncryptedFileStore(byteArrayOf(id)) { key }
        }

        val crypto = encryptedStore(1)
        val settings = LocalSettingsStore(folder.root, crypto, encryptedStore(2))
        val backend = LocalChatBackend(
            LocalChatStore(folder.root, crypto), settings, ServerlessDefaults(JSONObject()),
            DirectRouter(DirectHttp()), "この端末", null, { "chat" },
        )

        for (mode in listOf("compact", "minimal", "normal")) {
            val saved = backend.put("/api/mobile/v1/preferences", JSONObject().put("prompt_bar_mode", mode), "")
            assertEquals(mode, parsePreferences(saved).effectivePromptBarMode)
            assertEquals(mode, parsePreferences(backend.get("/api/mobile/v1/preferences")).effectivePromptBarMode)
        }
    }
}
