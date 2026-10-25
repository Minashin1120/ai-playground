package com.minashin1120.aiplayground.data.local

import android.graphics.Bitmap
import com.minashin1120.aiplayground.data.direct.DirectHttp
import com.minashin1120.aiplayground.data.direct.DirectRouter
import com.minashin1120.aiplayground.data.direct.ServerlessDefaults
import com.minashin1120.aiplayground.data.secure.EncryptedFileStore
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.ByteArrayOutputStream
import javax.crypto.KeyGenerator

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35])
class LocalChatBackendTokenTest {
    @get:Rule val folder = TemporaryFolder()

    @Test fun countsEncryptedAttachmentsAndReportsMissingFiles() = runBlocking {
        fun crypto(id: Byte): EncryptedFileStore {
            val key = KeyGenerator.getInstance("AES").apply { init(256) }.generateKey()
            return EncryptedFileStore(byteArrayOf(id)) { key }
        }
        val encrypted = crypto(1)
        val store = LocalChatStore(folder.root, encrypted)
        val backend = LocalChatBackend(store, LocalSettingsStore(folder.root, encrypted, crypto(2)),
            ServerlessDefaults(JSONObject()), DirectRouter(DirectHttp()), "test", null, { "chat" })
        val image = ByteArrayOutputStream().apply {
            val bitmap = Bitmap.createBitmap(100, 100, Bitmap.Config.ARGB_8888)
            try { bitmap.compress(Bitmap.CompressFormat.PNG, 100, this) } finally { bitmap.recycle() }
        }.toByteArray()
        val ref = store.saveFile("image.png", "image/png", image.inputStream(), image.size.toLong())
        val text = "添付のテキスト"
        val textRef = store.saveFile("test.txt", "text/plain", text.byteInputStream(), text.toByteArray().size.toLong())
        val missing = store.saveFile("missing.png", "image/png", image.inputStream(), image.size.toLong())
        folder.root.resolve("files").listFiles()!!.first { it.name.startsWith(EncryptedFileStore.sha256Hex(missing).take(40)) }.delete()
        val body = JSONObject().put("model", "gemini-3-pro-preview").put("message", "hello")
            .put("image_urls", JSONArray(listOf(ref, textRef, missing)))
        val result = backend.post("/api/token_estimate", body)
        assertEquals(1120 + LocalChatBackend.estimateTokens("[添付ファイル: test.txt]\n$text"), result.getInt("tokens_files"))
        assertEquals(result.getInt("tokens_prompt") + result.getInt("tokens_files"), result.getInt("tokens_total"))
        assertEquals(2, result.getInt("files_counted"))
        assertEquals(1, result.getInt("files_missing"))
        assertEquals(0, result.getInt("files_error"))
        assertEquals(result.toString(), backend.post("/api/token_estimate", body).toString())
    }
}
