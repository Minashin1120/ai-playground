package com.minashin1120.aiplayground.data

import com.minashin1120.aiplayground.data.secure.EncryptedFileStore
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.util.UUID

/** Durable, encrypted outbox. A record becomes visible only after all attachments were copied. */
internal class PendingFeedbackStore(
    private val root: File,
    private val cipher: EncryptedFileStore = EncryptedFileStore(
        "AIPFB1".toByteArray(), EncryptedFileStore.keystoreKey("pending_feedback_v1")),
) {
    data class Attachment(val mime: String, val name: String, val reference: String = "", val open: () -> InputStream)
    data class Record(val directory: File, val data: JSONObject)

    private fun owner(origin: String, account: Int) = File(root, EncryptedFileStore.sha256Hex("$origin\n$account"))

    fun enqueue(origin: String, account: Int, payload: JSONObject, attachments: List<Attachment>): Record {
        val directory = File(owner(origin, account), UUID.randomUUID().toString())
        try {
            val files = JSONArray()
            attachments.forEachIndexed { index, attachment ->
                attachment.open().use { cipher.writeStream(File(directory, "$index.enc"), it) }
                files.put(JSONObject().put("mime", attachment.mime).put("name", attachment.name)
                    .put("reference", attachment.reference))
            }
            val data = JSONObject().put("payload", payload).put("files", files).put("next_file", 0)
            val record = Record(directory, data)
            save(record)
            return record
        } catch (e: Exception) {
            directory.deleteRecursively()
            throw e
        }
    }

    fun records(origin: String, account: Int): List<Record> = owner(origin, account).listFiles().orEmpty()
        .filter { File(it, "record.enc").isFile }.sortedBy { it.lastModified() }.map { directory ->
            Record(directory, cipher.readJson(File(directory, "record.enc"))
                ?: throw IOException("保留中のフィードバックを読み込めません。"))
        }

    fun save(record: Record) {
        if (record.data.toString().toByteArray(Charsets.UTF_8).size > EncryptedFileStore.DEFAULT_JSON_LIMIT) {
            throw IOException("フィードバックが端末に保存できるサイズを超えています。")
        }
        cipher.writeJson(File(record.directory, "record.enc"), record.data)
    }
    fun remove(record: Record) { record.directory.deleteRecursively() }
    fun materialize(record: Record, index: Int, target: File) =
        cipher.decryptToFile(File(record.directory, "$index.enc"), target)
}
