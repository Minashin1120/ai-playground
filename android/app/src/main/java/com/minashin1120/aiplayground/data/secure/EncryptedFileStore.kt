package com.minashin1120.aiplayground.data.secure

import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.security.KeyStore
import java.security.MessageDigest
import java.security.SecureRandom
import javax.crypto.Cipher
import javax.crypto.CipherInputStream
import javax.crypto.CipherOutputStream
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/**
 * AES-256-GCM encrypted files: `MAGIC + 12-byte IV + ciphertext`, written to `<name>.part` and renamed
 * into place so a crash never leaves a half-written record. The key comes from [key] (an Android
 * Keystore key in the app; a plain in-memory key in JVM unit tests, where AndroidKeyStore does not exist).
 */
class EncryptedFileStore(
    private val magic: ByteArray,
    private val key: () -> SecretKey,
) {
    private val random = SecureRandom()

    fun writeJson(file: File, payload: JSONObject) =
        writeStream(file, payload.toString().toByteArray(Charsets.UTF_8).inputStream())

    fun readJson(file: File, limit: Long = DEFAULT_JSON_LIMIT): JSONObject? = runCatching {
        if (!file.isFile) return null
        JSONObject(String(readBytes(file, limit), Charsets.UTF_8))
    }.getOrNull()

    fun writeBytes(target: File, bytes: ByteArray) = writeStream(target, bytes.inputStream())

    fun writeStream(target: File, input: InputStream) {
        target.parentFile?.mkdirs()
        val temporary = File(target.parentFile, "${target.name}.part")
        val iv = ByteArray(12).also(random::nextBytes)
        val cipher = Cipher.getInstance("AES/GCM/NoPadding").apply {
            init(Cipher.ENCRYPT_MODE, key(), GCMParameterSpec(128, iv))
        }
        try {
            FileOutputStream(temporary).buffered().use { output ->
                output.write(magic)
                output.write(iv)
                CipherOutputStream(output, cipher).use { encrypted -> input.copyTo(encrypted, 64 * 1024) }
            }
            if (target.exists() && !target.delete()) throw IOException("キャッシュを更新できません。")
            if (!temporary.renameTo(target)) throw IOException("キャッシュを保存できません。")
        } finally {
            input.close()
            temporary.delete()
        }
    }

    fun readBytes(file: File, limit: Long): ByteArray {
        FileInputStream(file).use { input ->
            openDecrypted(input).use { decrypted ->
                val output = ByteArrayOutputStream()
                val buffer = ByteArray(64 * 1024)
                var total = 0L
                while (true) {
                    val count = decrypted.read(buffer)
                    if (count < 0) break
                    total += count
                    if (total > limit) throw IOException("キャッシュされた添付が上限を超えています。")
                    output.write(buffer, 0, count)
                }
                return output.toByteArray()
            }
        }
    }

    fun decryptToFile(source: File, destination: File) {
        FileInputStream(source).use { input ->
            openDecrypted(input).use { decrypted ->
                FileOutputStream(destination).use { output -> decrypted.copyTo(output, 64 * 1024) }
            }
        }
    }

    private fun openDecrypted(input: InputStream): CipherInputStream {
        val header = ByteArray(magic.size)
        if (!readFully(input, header) || !header.contentEquals(magic)) throw IOException("キャッシュ形式が不正です。")
        val iv = ByteArray(12)
        if (!readFully(input, iv)) throw IOException("キャッシュの暗号化情報が不正です。")
        val cipher = Cipher.getInstance("AES/GCM/NoPadding").apply {
            init(Cipher.DECRYPT_MODE, key(), GCMParameterSpec(128, iv))
        }
        return CipherInputStream(input, cipher)
    }

    private fun readFully(input: InputStream, buffer: ByteArray): Boolean {
        var offset = 0
        while (offset < buffer.size) {
            val count = input.read(buffer, offset, buffer.size - offset)
            if (count < 0) return false
            offset += count
        }
        return true
    }

    companion object {
        const val DEFAULT_JSON_LIMIT = 16L * 1024 * 1024

        /** An AES-256-GCM Android Keystore key under [alias], created on first use. */
        fun keystoreKey(alias: String): () -> SecretKey = {
            val store = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
            (store.getKey(alias, null) as? SecretKey) ?: KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore").apply {
                init(KeyGenParameterSpec.Builder(alias, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
                    .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                    .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                    .setRandomizedEncryptionRequired(false)
                    .setKeySize(256)
                    .build())
            }.generateKey()
        }

        fun sha256Hex(value: String): String = MessageDigest.getInstance("SHA-256")
            .digest(value.toByteArray(Charsets.UTF_8)).joinToString("") { "%02x".format(it.toInt() and 0xff) }
    }
}
