package com.minashin1120.aiplayground.data.secure

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.spec.GCMParameterSpec

class EncryptedFileStoreTest {
    @get:Rule val folder = TemporaryFolder()
    private val key = KeyGenerator.getInstance("AES").apply { init(256) }.generateKey()
    private val store = EncryptedFileStore(byteArrayOf('T'.code.toByte(), 'S'.code.toByte(), 1)) { key }

    @Test fun roundTripsJsonAndBytesWithoutPlaintextOnDisk() {
        val file = folder.root.resolve("a/record.enc")
        store.writeJson(file, JSONObject().put("secret", "sk-test-value"))
        assertEquals("sk-test-value", store.readJson(file)?.getString("secret"))
        assertFalse(String(file.readBytes(), Charsets.ISO_8859_1).contains("sk-test-value"))
        assertFalse(folder.root.resolve("a/record.enc.part").exists())
        val bytes = folder.root.resolve("b.enc")
        store.writeBytes(bytes, byteArrayOf(1, 2, 3))
        assertArrayEquals(byteArrayOf(1, 2, 3), store.readBytes(bytes, 10))
    }

    @Test fun segmentedBodiesRoundTripAtAnySize() {
        val segment = EncryptedFileStore.SEGMENT_BYTES
        for (size in listOf(0, 1, segment - 1, segment, segment + 1, segment * 2 + 12345)) {
            val data = ByteArray(size) { (it * 31 + size).toByte() }
            val file = folder.root.resolve("seg-$size.enc")
            store.writeStream(file, data.inputStream())
            assertArrayEquals("size $size", data, store.readBytes(file, size.toLong() + 1))
            val copy = folder.root.resolve("seg-$size.out")
            store.decryptToFile(file, copy)
            assertArrayEquals("size $size", data, copy.readBytes())
        }
    }

    @Test fun readsTheEarlierWholeFileFormat() {
        val magic = byteArrayOf('T'.code.toByte(), 'S'.code.toByte(), 1)
        val iv = ByteArray(12) { it.toByte() }
        val sealed = Cipher.getInstance("AES/GCM/NoPadding").apply { init(Cipher.ENCRYPT_MODE, key, GCMParameterSpec(128, iv)) }
            .doFinal("{\"old\":true}".toByteArray())
        val file = folder.root.resolve("legacy.enc").apply { writeBytes(magic + iv + sealed) }
        assertEquals(true, store.readJson(file)?.getBoolean("old"))
    }

    @Test fun rejectsTruncatedFilesAndStopsWhenCancelled() {
        val segment = EncryptedFileStore.SEGMENT_BYTES
        val file = folder.root.resolve("long.enc")
        store.writeBytes(file, ByteArray(segment * 2 + 10))
        val truncated = folder.root.resolve("cut.enc").apply { writeBytes(file.readBytes().copyOf((file.length() - 26).toInt())) }
        assertTrue(runCatching { store.readBytes(truncated, Long.MAX_VALUE) }.isFailure)
        var reads = 0
        val stopped = runCatching { store.readBytes(file, Long.MAX_VALUE) { ++reads > 1 } }
        assertTrue(stopped.exceptionOrNull() is java.util.concurrent.CancellationException)
    }

    @Test fun rejectsOversizedAndForeignFiles() {
        val file = folder.root.resolve("big.enc")
        store.writeBytes(file, ByteArray(64))
        assertTrue(runCatching { store.readBytes(file, 10) }.isFailure)
        val foreign = folder.root.resolve("foreign.enc").apply { writeText("{\"plain\":true}") }
        assertNull(store.readJson(foreign))
        assertNull(store.readJson(folder.root.resolve("missing.enc")))
    }
}
