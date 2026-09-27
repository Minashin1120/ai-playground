package com.minashin1120.aiplayground.data.local

import android.content.Context
import com.minashin1120.aiplayground.data.ServerOrigin
import com.minashin1120.aiplayground.data.direct.ServerlessDefaults
import com.minashin1120.aiplayground.data.originLabel
import com.minashin1120.aiplayground.data.secure.EncryptedFileStore
import org.json.JSONObject
import java.io.File

/**
 * Device-side storage of each profile: the no-account profile ("この端末") and, for serverless mode, each
 * signed-in account on each server. Chats and settings share one Keystore key; API keys use another.
 */
class LocalProfiles(private val context: Context) {
    private val root = File(context.noBackupFilesDir, "local-profiles")
    private val crypto = EncryptedFileStore(MAGIC, EncryptedFileStore.keystoreKey("ai-playground-local-chats-v1"))
    private val secrets = EncryptedFileStore(MAGIC, EncryptedFileStore.keystoreKey("ai-playground-local-secrets-v1"))

    data class Profile(val key: String, val chats: LocalChatStore, val settings: LocalSettingsStore)

    private fun dir(key: String) = File(root, EncryptedFileStore.sha256Hex(key).take(32))

    // One store object per profile, so the store's own locking covers every caller (the sync job
    // started before an account reload keeps writing through the same object as the chat screen).
    private val opened = HashMap<String, Profile>()

    @Synchronized
    fun open(key: String): Profile = opened.getOrPut(key) {
        val base = dir(key)
        Profile(key, LocalChatStore(File(base, "chats"), crypto), LocalSettingsStore(File(base, "settings"), crypto, secrets))
    }

    fun local(): Profile = open(LOCAL_KEY)

    /** The serverless-mode store of account [accountId] on the active server. */
    fun account(accountId: Int): Profile = open(accountKey(accountId))

    fun hasLocalData(): Boolean = File(dir(LOCAL_KEY), "chats/index.enc").isFile

    @Synchronized
    fun delete(key: String) { opened.remove(key); dir(key).deleteRecursively() }

    /** `assets/serverless-defaults.json`: the server's model catalog and prompt texts for offline use. */
    fun defaults(): ServerlessDefaults = ServerlessDefaults(runCatching {
        JSONObject(context.assets.open("serverless-defaults.json").bufferedReader().use { it.readText() })
    }.getOrElse { JSONObject() })

    companion object {
        const val LOCAL_KEY = "local"
        private val MAGIC = byteArrayOf('L'.code.toByte(), 'C'.code.toByte(), 'S'.code.toByte(), 1)

        fun accountKey(accountId: Int): String = "server:${originLabel(ServerOrigin.current)}#$accountId"
    }
}
