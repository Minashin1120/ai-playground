package com.minashin1120.aiplayground

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import com.minashin1120.aiplayground.data.*
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException

/** Account export categories imported by the first-run wizard. */
private const val ACCOUNT_IMPORT_CATEGORIES =
    "settings,api_credentials,chats,skills,files,feedback,diagnostics"

/**
 * The first-run wizard's account ZIP import (Web `/setup` import): chunked, resumable upload,
 * applying the archive, and the confirmation step for the settings the archive would overwrite.
 * State stays in the view model; this class only reads it and reports through `mutable`.
 */
internal class ChatAccountImport(
    private val context: Context,
    private val api: PlaygroundApi,
    private val state: StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val scope: CoroutineScope,
    private val token: () -> String,
) {
    private var job: Job? = null

    /** The running import (cancelled at the BAN and logout boundaries). */
    val activeJob: Job? get() = job

    /** Uploads and imports an account ZIP in a resumable, cancellable chunk session. */
    fun importAccountZip(uri: Uri) {
        if (state.value.setupImportBusy) return
        job = scope.launch {
            mutable.update { it.copy(
                setupImportBusy = true, setupImportError = null, setupImportDone = false, setupImportProgress = 0,
                setupImportPendingUploadId = null, setupImportSettingsChanges = emptyList()) }
            var uploadId: String? = null
            try {
                val resolver = context.contentResolver
                val resolved = resolver.query(uri, null, null, null, null)?.use { cursor ->
                    val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    val sizeIndex = cursor.getColumnIndex(OpenableColumns.SIZE)
                    if (cursor.moveToFirst()) {
                        cursor.getString(nameIndex).orEmpty().ifBlank { "account.zip" } to cursor.getLong(sizeIndex)
                    } else null
                } ?: throw IOException("ファイルを開けません。")
                val (name, size) = resolved
                require(size > 0) { "ファイルサイズを取得できません。" }
                mutable.update { it.copy(setupImportName = name) }
                val start = api.accountImportStart(size, token())
                val id = start.getString("upload_id")
                uploadId = id
                val chunkSize = start.getInt("chunk_size")
                val total = start.getInt("total_chunks")
                mutable.update { it.copy(setupImportTotalChunks = total) }
                resolver.openInputStream(uri)?.use { input ->
                    var received = 0L
                    for (index in 0 until total) {
                        val readSize = minOf(chunkSize.toLong(), size - received).toInt()
                        val buffer = ByteArray(readSize)
                        var offset = 0
                        while (offset < readSize) {
                            val count = input.read(buffer, offset, readSize - offset)
                            if (count < 0) throw IOException("ファイルの読み込みが中断されました。")
                            offset += count
                        }
                        received += readSize
                        api.accountImportChunk(id, index, buffer, token())
                        mutable.update { it.copy(setupImportProgress = index + 1) }
                    }
                } ?: throw IOException("ファイルを開けません。")
                api.accountImportComplete(id, token())
                val result = api.accountImport(id, ACCOUNT_IMPORT_CATEGORIES, token())
                if (result.optString("status") == "settings_confirmation") {
                    val changes = parseImportSettingChanges(result)
                    if (changes.isNotEmpty()) {
                        uploadId = null
                        mutable.update { it.copy(
                            setupImportBusy = false,
                            setupImportPendingUploadId = id,
                            setupImportSettingsChanges = changes,
                        ) }
                        return@launch
                    }
                }
                uploadId = null
                mutable.update { it.copy(setupImportBusy = false, setupImportDone = true) }
            } catch (e: CancellationException) {
                uploadId?.let { id -> withContext(NonCancellable) { runCatching { api.accountImportCancel(id, token()) } } }
                throw e
            } catch (e: Exception) {
                uploadId?.let { id -> runCatching { api.accountImportCancel(id, token()) } }
                mutable.update { it.copy(setupImportBusy = false, setupImportError = e.message ?: "インポートに失敗しました。") }
            }
        }
    }

    /** Applies the uploaded archive after the user confirms the settings changes. */
    fun confirmSetupImportSettings() {
        val uploadId = state.value.setupImportPendingUploadId ?: return
        if (state.value.setupImportBusy) return
        job = scope.launch {
            mutable.update { it.copy(setupImportBusy = true, setupImportError = null) }
            try {
                api.accountImport(uploadId, ACCOUNT_IMPORT_CATEGORIES, token(), confirmSettings = true)
                mutable.update { it.copy(
                    setupImportBusy = false,
                    setupImportDone = true,
                    setupImportPendingUploadId = null,
                    setupImportSettingsChanges = emptyList(),
                ) }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                mutable.update { it.copy(
                    setupImportBusy = false,
                    setupImportError = e.message ?: "インポートに失敗しました。",
                ) }
            }
        }
    }

    /** Rejects the settings overwrite and removes the completed upload session. */
    fun cancelSetupImportConfirmation() {
        val uploadId = state.value.setupImportPendingUploadId ?: return
        mutable.update { it.copy(
            setupImportPendingUploadId = null,
            setupImportSettingsChanges = emptyList(),
            setupImportError = null,
        ) }
        scope.launch {
            runCatching { api.accountImportCancel(uploadId, token()) }
        }
    }

    fun cancelSetupImport() {
        val pendingUploadId = state.value.setupImportPendingUploadId
        job?.cancel()
        mutable.update { it.copy(
            setupImportBusy = false, setupImportProgress = 0, setupImportError = null,
            setupImportPendingUploadId = null, setupImportSettingsChanges = emptyList(),
        ) }
        if (pendingUploadId != null) {
            scope.launch { runCatching { api.accountImportCancel(pendingUploadId, token()) } }
        }
    }

    private fun parseImportSettingChanges(reply: JSONObject): List<ImportSettingChange> {
        val rows = reply.optJSONArray("settings_changes") ?: return emptyList()
        fun display(value: Any?): String = when {
            value == null || value == JSONObject.NULL -> "未設定"
            value is JSONObject || value is JSONArray -> value.toString()
            else -> value.toString()
        }.let { value -> if (value.length > 2_000) value.take(2_000) + "…" else value }
        return buildList {
            for (index in 0 until rows.length()) {
                val row = rows.optJSONObject(index) ?: continue
                val field = row.optString("field").trim()
                if (field.isBlank()) continue
                add(ImportSettingChange(
                    field = field,
                    current = display(row.opt("current")),
                    incoming = display(row.opt("incoming")),
                ))
            }
        }
    }
}
