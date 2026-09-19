package com.minashin1120.aiplayground.data

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.AtomicFile
import org.json.JSONObject
import java.io.File
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

class TokenStore(context: Context) {
    private val file = AtomicFile(File(context.noBackupFilesDir, "android-session.enc"))
    private val alias = "ai-playground-token-v1"
    private fun key(): SecretKey {
        val store = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        (store.getKey(alias, null) as? SecretKey)?.let { return it }
        return KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore").apply {
            init(KeyGenParameterSpec.Builder(alias, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM).setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256).build())
        }.generateKey()
    }
    @Synchronized fun save(session: StoredSession) {
        val plain = JSONObject().put("token", session.token).put("expiresAt", session.expiresAt).toString()
        val cipher = Cipher.getInstance("AES/GCM/NoPadding").apply { init(Cipher.ENCRYPT_MODE, key()) }
        val encrypted = cipher.doFinal(plain.toByteArray(Charsets.UTF_8))
        val stream = file.startWrite()
        try {
            stream.write(1); stream.write(cipher.iv.size); stream.write(cipher.iv); stream.write(encrypted)
            file.finishWrite(stream)
        } catch (e: Exception) { file.failWrite(stream); throw e }
    }
    @Synchronized fun load(): StoredSession? = load(requireValid = true)

    /** Returns an expired session only so the app can restore local offline data. */
    @Synchronized fun loadForOffline(): StoredSession? = load(requireValid = false)

    private fun load(requireValid: Boolean): StoredSession? {
        val result = runCatching {
            file.openRead().use { input ->
                require(input.read() == 1)
                val ivSize = input.read()
                require(ivSize == 12)
                val iv = ByteArray(ivSize)
                require(input.read(iv) == ivSize)
                val cipher = Cipher.getInstance("AES/GCM/NoPadding").apply {
                    init(Cipher.DECRYPT_MODE, key(), GCMParameterSpec(128, iv))
                }
                val plain = JSONObject(String(cipher.doFinal(input.readBytes()), Charsets.UTF_8))
                StoredSession(plain.getString("token"), plain.getLong("expiresAt"))
            }
        }.getOrElse { clear(); return null }
        return result.takeIf { !requireValid || it.expiresAt > System.currentTimeMillis() }
    }
    @Synchronized fun clear() { file.delete() }
}
