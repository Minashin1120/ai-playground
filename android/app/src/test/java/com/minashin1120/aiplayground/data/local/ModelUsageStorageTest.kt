package com.minashin1120.aiplayground.data.local

import com.minashin1120.aiplayground.data.secure.EncryptedFileStore
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import javax.crypto.KeyGenerator

class ModelUsageStorageTest {
    @get:Rule val folder = TemporaryFolder()

    @Test fun thinkingAndRequestUsageSurviveSaveReloadAndSync() {
        val key = KeyGenerator.getInstance("AES").apply { init(256) }.generateKey()
        val crypto = EncryptedFileStore(byteArrayOf('T'.code.toByte(), 1)) { key }
        val store = LocalChatStore(folder.root.resolve("chats"), crypto)
        val thread = store.createThread(false, "test").getString("id")
        val id = store.appendMessage(thread, LocalChatStore.NewMessage("assistant", "", null, model = "gpt-6.1-sol", generating = true))
        val calls = JSONArray("""[{"model":"gpt-6.1-sol","input_tokens":100,"output_tokens":1000,"thought_tokens":900,"complete":true}]""")
        store.updateMessage(thread, id, "answer", "", tokensIn = 100, tokensOut = 1000,
            tokensThought = 900, pricingCalls = calls, done = true)
        val restarted = LocalChatStore(folder.root.resolve("chats"), crypto)
        val row = restarted.messages(thread).single()
        assertEquals(900, row.getInt("tokens_thought"))
        assertEquals(1, JSONObject(row.getString("thought_data")).getJSONArray("pricing_calls").length())
        val pending = restarted.pendingPush().single().entry.getJSONArray("messages").getJSONObject(0)
        assertEquals(900, pending.getInt("tokens_thought"))
        assertEquals(1, pending.getJSONArray("pricing_calls").length())
    }
}
