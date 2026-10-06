package com.minashin1120.aiplayground.data.secure

import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import org.json.JSONObject
import java.io.BufferedInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.nio.ByteBuffer
import java.security.KeyStore
import java.security.MessageDigest
import java.security.SecureRandom
import java.util.concurrent.CancellationException
import javax.crypto.Cipher
import javax.crypto.CipherInputStream
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.SecretKeySpec

/**
 * AES-256-GCM encrypted files, written to `<name>.part` and renamed into place so a crash never leaves a
 * half-written record. The key comes from [key] (an Android Keystore key in the app; a plain in-memory key
 * in JVM unit tests, where AndroidKeyStore does not exist).
 *
 * Each file has its own random data key, wrapped by [key] (one small Keystore operation), and the body is
 * encrypted by the software provider in [SEGMENT_BYTES] segments: Keystore ciphers pass data through
 * binder calls and hold the whole plaintext until the end, which made a tens-of-MB video take minutes to
 * read and pile up in memory. Layout: `magic^0x80 + wrap IV(12) + wrapped key(48) + nonce prefix(7) +
 * segment size(4)`, then segments whose nonce is `prefix + counter(4) + last(1)`, so a truncated or
 * reordered file fails to decrypt. Files of the earlier format (`magic + IV(12) + ciphertext`, the
 * whole body under [key]) are still read.
 */
class EncryptedFileStore(
    private val magic: ByteArray,
    private val key: () -> SecretKey,
) {
    private val random = SecureRandom()
    private val segmentedMagic = magic.copyOf().also { it[it.lastIndex] = (it.last().toInt() xor 0x80).toByte() }
    private val headerSize = segmentedMagic.size + 12 + WRAPPED_KEY_BYTES + 7 + 4

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
        val dataKey = ByteArray(32).also(random::nextBytes)
        val wrapIv = ByteArray(12).also(random::nextBytes)
        val wrapped = Cipher.getInstance("AES/GCM/NoPadding").apply {
            init(Cipher.ENCRYPT_MODE, key(), GCMParameterSpec(128, wrapIv))
        }.doFinal(dataKey)
        val prefix = ByteArray(7).also(random::nextBytes)
        val secret = SecretKeySpec(dataKey, "AES")
        try {
            FileOutputStream(temporary).buffered(64 * 1024).use { output ->
                output.write(segmentedMagic)
                output.write(wrapIv)
                output.write(wrapped)
                output.write(prefix)
                output.write(ByteBuffer.allocate(4).putInt(SEGMENT_BYTES).array())
                // One segment of look-ahead tells whether the current one is the last.
                var current = readUpTo(input, SEGMENT_BYTES)
                var counter = 0
                while (true) {
                    val next = readUpTo(input, SEGMENT_BYTES)
                    val last = next.isEmpty()
                    output.write(segmentCipher(Cipher.ENCRYPT_MODE, secret, prefix, counter, last).doFinal(current))
                    if (last) break
                    current = next
                    counter++
                }
            }
            // rename(2) replaces the old file atomically, so a reader never finds the record missing.
            if (!temporary.renameTo(target)) {
                if (target.exists() && !target.delete()) throw IOException("キャッシュを更新できません。")
                if (!temporary.renameTo(target)) throw IOException("キャッシュを保存できません。")
            }
        } finally {
            input.close()
            temporary.delete()
        }
    }

    /** The plaintext of [file]; [cancelled] is checked between segments so a stopped answer or sync can leave. */
    fun readBytes(file: File, limit: Long, cancelled: () -> Boolean = { false }): ByteArray {
        FileInputStream(file).use { input ->
            val opened = openDecrypted(input, cancelled)
            opened.stream.use { decrypted ->
                val size = opened.plaintextSize(file.length())
                if (size != null) {
                    if (size > limit) throw IOException("キャッシュされた添付が上限を超えています。")
                    val out = ByteArray(size.toInt())
                    if (!readFully(decrypted, out) || decrypted.read() >= 0) throw IOException("キャッシュの長さが不正です。")
                    return out
                }
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
            openDecrypted(input) { false }.stream.use { decrypted ->
                FileOutputStream(destination).use { output -> decrypted.copyTo(output, 64 * 1024) }
            }
        }
    }

    private class Opened(val stream: InputStream, private val segmented: Boolean, private val headerSize: Int) {
        /** Exact plaintext length of a segmented file (its size is known from the file length), else null. */
        fun plaintextSize(fileLength: Long): Long? {
            if (!segmented) return null
            val body = fileLength - headerSize
            if (body < GCM_TAG_BYTES) throw IOException("キャッシュの長さが不正です。")
            val segments = (body + SEGMENT_BYTES + GCM_TAG_BYTES - 1) / (SEGMENT_BYTES + GCM_TAG_BYTES)
            return body - segments * GCM_TAG_BYTES
        }
    }

    private fun openDecrypted(input: InputStream, cancelled: () -> Boolean): Opened {
        val header = ByteArray(magic.size)
        if (!readFully(input, header)) throw IOException("キャッシュ形式が不正です。")
        if (header.contentEquals(segmentedMagic)) {
            val wrapIv = ByteArray(12)
            val wrapped = ByteArray(WRAPPED_KEY_BYTES)
            val prefix = ByteArray(7)
            val size = ByteArray(4)
            if (!readFully(input, wrapIv) || !readFully(input, wrapped) || !readFully(input, prefix) || !readFully(input, size)) {
                throw IOException("キャッシュの暗号化情報が不正です。")
            }
            if (ByteBuffer.wrap(size).int != SEGMENT_BYTES) throw IOException("キャッシュの暗号化情報が不正です。")
            val dataKey = Cipher.getInstance("AES/GCM/NoPadding").apply {
                init(Cipher.DECRYPT_MODE, key(), GCMParameterSpec(128, wrapIv))
            }.doFinal(wrapped)
            return Opened(SegmentedInputStream(input, SecretKeySpec(dataKey, "AES"), prefix, cancelled), true, headerSize)
        }
        if (!header.contentEquals(magic)) throw IOException("キャッシュ形式が不正です。")
        val iv = ByteArray(12)
        if (!readFully(input, iv)) throw IOException("キャッシュの暗号化情報が不正です。")
        val cipher = Cipher.getInstance("AES/GCM/NoPadding").apply {
            init(Cipher.DECRYPT_MODE, key(), GCMParameterSpec(128, iv))
        }
        return Opened(CipherInputStream(input, cipher), false, headerSize)
    }

    /** Decrypts one segment at a time (at most [SEGMENT_BYTES] of plaintext in memory). */
    private class SegmentedInputStream(
        input: InputStream,
        private val secret: SecretKey,
        private val prefix: ByteArray,
        private val cancelled: () -> Boolean,
    ) : InputStream() {
        private val source = BufferedInputStream(input, 64 * 1024)
        private val sealed = ByteArray(SEGMENT_BYTES + GCM_TAG_BYTES)
        private var plain = ByteArray(0)
        private var position = 0
        private var counter = 0
        private var finished = false

        private fun fill(): Boolean {
            if (finished) return false
            if (cancelled()) throw CancellationException("読み込みを中止しました。")
            var length = 0
            while (length < sealed.size) {
                val count = source.read(sealed, length, sealed.size - length)
                if (count < 0) break
                length += count
            }
            if (length < GCM_TAG_BYTES) throw IOException("キャッシュの長さが不正です。")
            source.mark(1)
            val last = source.read() < 0
            source.reset()
            plain = try {
                segmentCipher(Cipher.DECRYPT_MODE, secret, prefix, counter, last).doFinal(sealed, 0, length)
            } catch (e: java.security.GeneralSecurityException) {
                throw IOException("キャッシュを復号できません。", e)
            }
            position = 0
            counter++
            finished = last
            return true
        }

        override fun read(): Int {
            while (position >= plain.size) if (!fill()) return -1
            return plain[position++].toInt() and 0xff
        }

        override fun read(b: ByteArray, off: Int, len: Int): Int {
            if (len == 0) return 0
            while (position >= plain.size) if (!fill()) return -1
            val count = minOf(len, plain.size - position)
            System.arraycopy(plain, position, b, off, count)
            position += count
            return count
        }

        override fun close() = source.close()
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

    private fun readUpTo(input: InputStream, max: Int): ByteArray {
        val buffer = ByteArray(max)
        var length = 0
        while (length < max) {
            val count = input.read(buffer, length, max - length)
            if (count < 0) break
            length += count
        }
        return if (length == max) buffer else buffer.copyOf(length)
    }

    companion object {
        const val DEFAULT_JSON_LIMIT = 16L * 1024 * 1024
        /** Plaintext bytes per segment of the segmented format. */
        const val SEGMENT_BYTES = 1024 * 1024
        private const val GCM_TAG_BYTES = 16
        private const val WRAPPED_KEY_BYTES = 32 + GCM_TAG_BYTES

        private fun segmentCipher(mode: Int, secret: SecretKey, prefix: ByteArray, counter: Int, last: Boolean): Cipher {
            val nonce = ByteBuffer.allocate(12).put(prefix).putInt(counter).put(if (last) 1.toByte() else 0.toByte()).array()
            return Cipher.getInstance("AES/GCM/NoPadding").apply { init(mode, secret, GCMParameterSpec(128, nonce)) }
        }

        /** An AES-256-GCM Android Keystore key under [alias], created on first use. */
        fun keystoreKey(alias: String): () -> SecretKey = { synchronized(KEY_LOCK) { loadOrCreateKey(alias) } }

        // Two threads creating the key at once would each store one under the alias, and records
        // written with the replaced key could no longer be read.
        private val KEY_LOCK = Any()

        private fun loadOrCreateKey(alias: String): SecretKey {
            val store = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
            return (store.getKey(alias, null) as? SecretKey) ?: KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore").apply {
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
