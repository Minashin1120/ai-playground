package com.minashin1120.aiplayground.data.secure

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import javax.crypto.KeyGenerator

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

    @Test fun rejectsOversizedAndForeignFiles() {
        val file = folder.root.resolve("big.enc")
        store.writeBytes(file, ByteArray(64))
        assertTrue(runCatching { store.readBytes(file, 10) }.isFailure)
        val foreign = folder.root.resolve("foreign.enc").apply { writeText("{\"plain\":true}") }
        assertNull(store.readJson(foreign))
        assertNull(store.readJson(folder.root.resolve("missing.enc")))
    }
}
