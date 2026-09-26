package com.minashin1120.aiplayground.data

import android.content.ContentResolver
import android.net.Uri
import android.provider.OpenableColumns
import com.minashin1120.aiplayground.ImportSettingChange
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.IOException
import java.security.SecureRandom

/** `#account-transfer-progress`. */
data class TransferProgress(val progress: Int = 0, val phase: String = "", val message: String = "")

/** `#account-export-ready`. */
data class ExportReady(val jobId: String, val sizeBytes: Long, val unreadable: Int, val expiresAt: String)

/** One row of the storage-limit file picker (`#import-files-modal`). */
data class ImportFileChoice(val archivePath: String, val displayName: String, val sizeBytes: Long)

data class FileSelectionRequest(val files: List<ImportFileChoice>, val availableBytes: Long)

/** `重複データを確認して修復` preview waiting for the confirm dialog. */
data class DedupePreview(val total: Int, val parts: String, val keptReferenced: Int)

data class TransferUiState(
    val running: Boolean = false,
    val type: String? = null,
    val progress: TransferProgress? = null,
    val exportReady: ExportReady? = null,
    val downloading: Boolean = false,
    val downloadedBytes: Long = 0,
    val importResult: String? = null,
    val importResultError: Boolean = false,
    val fileSelection: FileSelectionRequest? = null,
    val settingsChanges: List<ImportSettingChange>? = null,
    val dedupeBusy: Boolean = false,
    val dedupeResult: String? = null,
    val dedupeError: Boolean = false,
    val dedupePreview: DedupePreview? = null,
)

/** Web `renderAccountTransferProgress` phase labels. */
val TRANSFER_PHASE_LABELS = mapOf(
    "queued" to "順番待ち", "preparing" to "データを準備中", "exporting_files" to "ファイルを書き出し中",
    "finalizing" to "最終処理中", "ready" to "ダウンロード準備完了", "downloading" to "ダウンロード中",
    "uploading" to "ZIPをアップロード中", "validating" to "ZIPを検証中", "validating_files" to "ファイル情報を検証中",
    "reading_files" to "ファイルを読み込み中", "importing_settings" to "設定を反映中",
    "importing_credentials" to "認証情報を反映中", "importing_gems" to "Gemを追加中",
    "saving_files" to "ファイルを保存中", "importing_chats" to "チャット履歴を追加中",
    "importing_feedback" to "フィードバックを追加中", "importing_diagnostics" to "診断データを追加中",
    "cancelling" to "キャンセル処理中", "cancelled" to "キャンセル済み", "expired" to "保存期限切れ",
    "completed" to "完了", "failed" to "失敗",
)

val DEDUPE_LABELS = listOf("chats" to "チャット", "gems" to "Gem", "files" to "ファイル", "feedback" to "フィードバック", "diagnostics" to "診断データ")

private val TERMINAL_STATES = setOf("ready", "completed", "failed", "cancelled", "expired")

/**
 * The settings Data tab's export / import / dedupe (Web part07 `account-*` handlers). Runs in the
 * ViewModel scope so a transfer continues while the settings modal is closed, like the Web tab.
 * Operations that the server gates behind a re-authentication are parked in [reauth] and replayed.
 */
class AccountTransferController(
    private val scope: CoroutineScope,
    private val account: () -> AccountApi,
    private val resolver: ContentResolver,
    private val notify: (String) -> Unit,
    private val onImported: (List<String>) -> Unit,
) {
    private val mutable = MutableStateFlow(TransferUiState())
    val state = mutable.asStateFlow()
    /** An operation waiting for the re-authentication dialog; run it after success. */
    val reauth = MutableStateFlow<(() -> Unit)?>(null)
    private var job: Job? = null
    private var activeId: String? = null
    private var uploadId: String? = null
    private var cancelRequested = false
    private var selectionAnswer: CompletableDeferred<String?>? = null
    private var settingsAnswer: CompletableDeferred<Boolean>? = null

    private fun newId(): String = ByteArray(16).also { SecureRandom().nextBytes(it) }.joinToString("") { "%02x".format(it) }

    private fun progress(progress: Int, phase: String, message: String) =
        mutable.update { it.copy(progress = TransferProgress(progress.coerceIn(0, 100), phase, message)) }

    private fun progressFrom(data: JSONObject) = progress(
        data.optDouble("progress", 0.0).toInt(), data.optString("phase"),
        data.optString("message").ifBlank { "処理状況を確認しています" },
    )

    private fun availabilityFrom(data: JSONObject, jobId: String) {
        val available = data.optBoolean("available") && data.optString("download_url").isNotBlank()
        mutable.update { it.copy(exportReady = if (!available) null else ExportReady(
            jobId = data.optString("job_id").ifBlank { jobId },
            sizeBytes = data.optLong("size_bytes"),
            unreadable = data.optInt("unreadable_count"),
            expiresAt = data.optString("expires_at"),
        )) }
    }

    private suspend fun poll(jobId: String): JSONObject? {
        while (activeId == jobId && !cancelRequested) {
            try {
                val data = account().transferStatus(jobId)
                if (data.optString("state") != "pending") progressFrom(data)
                if (data.optString("state") in TERMINAL_STATES) return data
            } catch (e: CancellationException) { throw e } catch (ignored: Exception) { }
            delay(700)
        }
        return null
    }

    private fun finish() {
        activeId = null
        mutable.update { it.copy(running = false, type = null) }
    }

    /** Web `refreshLatestAccountExport`, run when the Data tab opens. */
    fun refreshLatestExport() {
        scope.launch {
            try {
                val data = account().latestExport()
                val jobId = data.optString("job_id")
                availabilityFrom(data, jobId)
                when (val state = data.optString("state")) {
                    "ready", "failed", "cancelled", "expired" -> progressFrom(data)
                    "queued", "running", "cancelling" -> if (jobId.isNotBlank() && activeId == null) {
                        activeId = jobId
                        cancelRequested = false
                        mutable.update { it.copy(running = true, type = "export") }
                        progressFrom(data)
                        val finished = poll(jobId)
                        if (finished != null) handleFinishedExport(finished, jobId)
                        if (activeId == jobId) finish()
                    }
                    else -> if (state.isBlank()) return@launch
                }
            } catch (e: CancellationException) { throw e } catch (ignored: Exception) { }
        }
    }

    private fun handleFinishedExport(data: JSONObject, jobId: String) {
        progressFrom(data)
        availabilityFrom(data, jobId)
        when (data.optString("state")) {
            "ready" -> notify(data.optString("message").ifBlank { "エクスポートZIPの準備が完了しました" })
            "failed" -> notify(data.optString("message").ifBlank { "エクスポートに失敗しました" })
        }
    }

    /** `#account-export-btn`. */
    fun startExport() {
        if (mutable.value.running) return
        val id = newId()
        activeId = id
        cancelRequested = false
        mutable.update { it.copy(running = true, type = "export", exportReady = null) }
        progress(0, "queued", "エクスポートを受け付けています")
        job = scope.launch {
            var jobId = id
            try {
                try {
                    account().startExport(id)
                } catch (e: ApiException) {
                    when {
                        e.needsReauth() -> { finish(); mutable.update { it.copy(progress = null) }; reauth.value = ::startExport; return@launch }
                        e.status == 409 && e.code == "export_in_progress" && e.payload.optString("job_id").isNotBlank() -> {
                            jobId = e.payload.optString("job_id"); activeId = jobId
                        }
                        e.code == "rate_limit" -> throw IOException("エクスポート回数の上限に達しました")
                        else -> throw IOException(e.payload.optString("error").ifBlank { "エクスポートを開始できませんでした" })
                    }
                }
                progress(0, "queued", "バックグラウンドでエクスポートしています")
                val finished = poll(jobId)
                if (!cancelRequested && finished != null) handleFinishedExport(finished, jobId)
            } catch (e: CancellationException) { throw e } catch (e: Exception) {
                val message = e.message ?: "エクスポートを開始できませんでした"
                progress(0, "failed", message)
                notify(message)
            } finally { if (activeId == jobId || activeId == id) finish() }
        }
    }

    /** `#account-export-download-btn`: re-checks availability, then streams the ZIP to [uri]. */
    fun download(uri: Uri) {
        if (mutable.value.downloading) return
        scope.launch {
            try {
                val latest = account().latestExport()
                val jobId = latest.optString("job_id")
                if (!(latest.optBoolean("available") && latest.optString("download_url").isNotBlank())) {
                    availabilityFrom(latest, jobId)
                    progressFrom(latest)
                    notify("エクスポートZIPをダウンロードできません。最新の状態を確認してください。")
                    return@launch
                }
                mutable.update { it.copy(downloading = true, downloadedBytes = 0) }
                withContext(Dispatchers.IO) {
                    val output = resolver.openOutputStream(uri, "w") ?: throw IOException("保存先を開けません。")
                    output.use { stream -> account().downloadExport(jobId, stream) { count -> mutable.update { it.copy(downloadedBytes = count) } } }
                }
            } catch (e: CancellationException) { throw e } catch (e: Exception) {
                if (e.needsReauth()) reauth.value = { download(uri) }
                else notify("エクスポートZIPをダウンロードできません。最新の状態を確認してください。")
            } finally { mutable.update { it.copy(downloading = false) } }
        }
    }

    /** `#account-transfer-cancel-btn`. */
    fun cancel() {
        val id = activeId ?: return
        val type = mutable.value.type
        cancelRequested = true
        progress(0, "cancelling", "キャンセルしています")
        scope.launch {
            runCatching { account().cancelTransfer(id) }
            job?.cancel()
            uploadId?.let { upload -> withContext(NonCancellable) { runCatching { account().importCancel(upload) } } }
            uploadId = null
            selectionAnswer?.complete(null)
            settingsAnswer?.complete(false)
            progress(0, "cancelled", "キャンセルしました")
            if (type == "export") mutable.update { it.copy(exportReady = null) }
            finish()
            notify("処理をキャンセルしました")
        }
    }

    fun answerFileSelection(selected: String?) {
        mutable.update { it.copy(fileSelection = null) }
        selectionAnswer?.complete(selected)
    }

    fun answerSettingsChanges(confirmed: Boolean) {
        mutable.update { it.copy(settingsChanges = null) }
        settingsAnswer?.complete(confirmed)
    }

    /** `#account-import-btn` after the category checks and the confirm dialog. */
    fun startImport(uri: Uri, categories: List<String>, restoreInplace: Boolean, settingsBypass: Boolean) {
        if (mutable.value.running) return
        val id = newId()
        activeId = id
        cancelRequested = false
        mutable.update { it.copy(running = true, type = "import", importResult = null) }
        progress(0, "uploading", "アップロードを準備しています")
        job = scope.launch {
            try {
                val (size, _) = withContext(Dispatchers.IO) {
                    resolver.query(uri, null, null, null, null)?.use { cursor ->
                        val sizeIndex = cursor.getColumnIndex(OpenableColumns.SIZE)
                        val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                        if (cursor.moveToFirst()) cursor.getLong(sizeIndex) to cursor.getString(nameIndex).orEmpty() else null
                    }
                } ?: throw IOException("アップロードを開始できません")
                val start = try { account().importStart(size) } catch (e: ApiException) {
                    if (e.needsReauth()) {
                        finish(); mutable.update { it.copy(progress = null) }
                        reauth.value = { startImport(uri, categories, restoreInplace, settingsBypass) }
                        return@launch
                    }
                    throw IOException(e.payload.optString("error").ifBlank { "アップロードを開始できません" })
                }
                val upload = start.getString("upload_id").also { uploadId = it }
                val chunkSize = start.getInt("chunk_size")
                val total = start.getInt("total_chunks").coerceAtLeast(1)
                withContext(Dispatchers.IO) {
                    resolver.openInputStream(uri)?.use { input ->
                        var received = 0L
                        for (index in 0 until total) {
                            val readSize = minOf(chunkSize.toLong(), size - received).toInt()
                            val buffer = ByteArray(readSize)
                            var offset = 0
                            while (offset < readSize) {
                                val count = input.read(buffer, offset, readSize - offset)
                                if (count < 0) throw IOException("アップロードに失敗しました")
                                offset += count
                            }
                            received += readSize
                            account().importChunk(upload, index, buffer)
                            progress(minOf(35, ((index + 1) * 35) / total), "uploading", "ZIPを並列アップロードしています（${index + 1}/$total）")
                        }
                    } ?: throw IOException("アップロードを開始できません")
                }
                account().importComplete(upload)
                progress(35, "validating", "ZIPを検証しています")
                var selectedFiles = ""
                var settingsConfirmed = false
                while (true) {
                    val poller = scope.launch { poll(id) }
                    val reply = try {
                        account().importArchive(upload, categories, id, selectedFiles, restoreInplace, settingsConfirmed || settingsBypass)
                    } catch (e: ApiException) {
                        if (e.code == "storage_limit_files" && e.payload.optJSONArray("files") != null) {
                            poller.cancel()
                            val files = e.payload.getJSONArray("files")
                            val request = FileSelectionRequest(
                                (0 until files.length()).map { i ->
                                    val f = files.getJSONObject(i)
                                    ImportFileChoice(f.optString("archive_path"), f.optString("display_name"), f.optLong("size_bytes"))
                                },
                                e.payload.optLong("available_bytes"),
                            )
                            val answer = CompletableDeferred<String?>().also { selectionAnswer = it }
                            mutable.update { it.copy(fileSelection = request) }
                            val chosen = answer.await()
                            if (chosen == null) {
                                progress(0, "cancelled", "ファイル選択をキャンセルしました")
                                runCatching { account().importCancel(upload) }
                                uploadId = null
                                return@launch
                            }
                            selectedFiles = chosen
                            continue
                        }
                        throw IOException(if (e.code == "storage_limit_exceeded") "ストレージ上限を超えるためインポートできません"
                            else e.payload.optString("error").ifBlank { "インポートに失敗しました" })
                    } finally { poller.cancel() }
                    if (reply.optString("status") == "settings_confirmation" && reply.optJSONArray("settings_changes") != null) {
                        val changes = parseSettingChanges(reply)
                        val answer = CompletableDeferred<Boolean>().also { settingsAnswer = it }
                        mutable.update { it.copy(settingsChanges = changes) }
                        if (!answer.await()) {
                            progress(0, "cancelled", "設定のインポートをキャンセルしました")
                            runCatching { account().importCancel(upload) }
                            uploadId = null
                            return@launch
                        }
                        settingsConfirmed = true
                        continue
                    }
                    uploadId = null
                    val imported = reply.optJSONObject("imported") ?: JSONObject()
                    val detail = listOf(
                        "設定 ${imported.optInt("settings")}件", "API認証 ${imported.optInt("api_credentials")}件",
                        "チャット ${imported.optInt("chats")}件", "Gem ${imported.optInt("gems")}件",
                        "ファイル ${imported.optInt("files")}件", "フィードバック ${imported.optInt("feedback")}件",
                        "診断データ ${imported.optInt("diagnostics")}件",
                    ).joinToString(" / ")
                    val duplicates = reply.optJSONObject("duplicates") ?: JSONObject()
                    val dupParts = DEDUPE_LABELS.mapNotNull { (key, label) -> duplicates.optInt(key).takeIf { it > 0 }?.let { "$label ${it}件" } }
                    val dupNote = if (dupParts.isEmpty()) "" else "（重複をスキップ: ${dupParts.joinToString("、")}）"
                    mutable.update { it.copy(importResult = "完了: $detail$dupNote", importResultError = false) }
                    progress(100, "completed", "インポートが完了しました")
                    notify("選択したアカウントデータをインポートしました")
                    onImported(categories)
                    break
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                uploadId?.let { upload -> runCatching { account().importCancel(upload) } }
                uploadId = null
                if (!cancelRequested) {
                    val message = e.message ?: "インポートに失敗しました"
                    progress(0, "failed", message)
                    mutable.update { it.copy(importResult = message, importResultError = true) }
                    notify(message)
                }
            } finally { if (activeId == id) finish() }
        }
    }

    private fun parseSettingChanges(reply: JSONObject): List<ImportSettingChange> {
        val rows = reply.optJSONArray("settings_changes") ?: return emptyList()
        fun display(value: Any?): String {
            val text = when (value) {
                null, JSONObject.NULL -> return "未設定"
                true -> return "ON"
                false -> return "OFF"
                else -> value.toString()
            }
            if (text.isEmpty()) return "未設定"
            return if (text.length > 60) text.take(60) + "…" else text
        }
        return (0 until rows.length()).mapNotNull { i ->
            val row = rows.optJSONObject(i) ?: return@mapNotNull null
            ImportSettingChange(row.optString("field"), display(row.opt("current")), display(row.opt("incoming")))
        }
    }

    /** `#account-dedupe-btn`: preview, then [confirmDedupe] after the confirm dialog. */
    fun previewDedupe() {
        if (mutable.value.dedupeBusy) return
        mutable.update { it.copy(dedupeBusy = true, dedupeResult = "重複データを確認しています...", dedupeError = false) }
        scope.launch {
            try {
                val preview = account().dedupePreview()
                if (!preview.optBoolean("has_duplicates")) {
                    mutable.update { it.copy(dedupeBusy = false, dedupeResult = "重複データは見つかりませんでした") }
                    return@launch
                }
                val dups = preview.optJSONObject("duplicates") ?: JSONObject()
                val parts = DEDUPE_LABELS.mapNotNull { (key, label) -> dups.optInt(key).takeIf { it > 0 }?.let { "$label ${it}件" } }
                mutable.update { it.copy(dedupePreview = DedupePreview(preview.optInt("total"), parts.joinToString("、"), preview.optInt("kept_referenced_files"))) }
            } catch (e: CancellationException) { throw e } catch (e: Exception) {
                mutable.update { it.copy(dedupeBusy = false, dedupeError = true,
                    dedupeResult = (e as? ApiException)?.payload?.optString("error")?.ifBlank { null } ?: "重複データを確認できませんでした") }
            }
        }
    }

    fun confirmDedupe(confirmed: Boolean) {
        mutable.update { it.copy(dedupePreview = null) }
        if (!confirmed) { mutable.update { it.copy(dedupeBusy = false) }; return }
        scope.launch {
            try {
                val executed = account().dedupeExecute()
                val removed = executed.optJSONObject("removed") ?: JSONObject()
                val parts = DEDUPE_LABELS.mapNotNull { (key, label) -> removed.optInt(key).takeIf { it > 0 }?.let { "$label ${it}件" } }
                val kept = executed.optInt("kept_referenced_files").takeIf { it > 0 }?.let { "（参照のため残したファイル ${it}件）" }.orEmpty()
                mutable.update { it.copy(dedupeResult = "重複データを修復しました: ${parts.joinToString("、").ifBlank { "0件" }}$kept", dedupeError = false) }
                onImported(listOf("chats", "gems", "files"))
            } catch (e: CancellationException) { throw e } catch (e: Exception) {
                mutable.update { it.copy(dedupeError = true,
                    dedupeResult = (e as? ApiException)?.payload?.optString("error")?.ifBlank { null } ?: "重複データの修復に失敗しました") }
            } finally { mutable.update { it.copy(dedupeBusy = false) } }
        }
    }
}
