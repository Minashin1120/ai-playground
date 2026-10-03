package com.minashin1120.aiplayground

import com.minashin1120.aiplayground.data.AppUpdate
import com.minashin1120.aiplayground.data.AppUpdateDownloadProgress
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.isActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.io.File

/**
 * Process-wide owner of the APK update state and download job. The download outlives the
 * activity and the update dialog, so the user can hide the dialog or leave the app while
 * [AppUpdateDownloadService] keeps the process in the foreground.
 */
class AppUpdateDownloadManager(
    private val cacheDir: File,
    private val download: suspend (AppUpdate, File, suspend (AppUpdateDownloadProgress) -> Unit) -> File,
    private val scope: CoroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.IO),
) {
    private val mutable = MutableStateFlow(AppUpdateUiState())
    val state: StateFlow<AppUpdateUiState> = mutable.asStateFlow()
    private var downloadJob: Job? = null
    private var cancelledJob: Job? = null

    internal fun updateState(transform: (AppUpdateUiState) -> AppUpdateUiState) {
        mutable.update(transform)
    }

    /** Starts the download; returns false when there is nothing to download or it already runs. */
    fun startDownload(): Boolean {
        val update = state.value.update ?: return false
        if (downloadJob?.isActive == true) return false
        mutable.update {
            it.copy(
                phase = AppUpdatePhase.Downloading,
                downloadedBytes = 0L,
                totalBytes = update.apkSizeBytes,
                readyFile = null,
                errorMessage = null,
                dialogHidden = false,
            )
        }
        val previous = cancelledJob
        cancelledJob = null
        downloadJob = scope.launch {
            // A cancelled download deletes its partial file while unwinding; let it finish first
            // because a retry writes to the same path.
            previous?.join()
            try {
                val directory = File(cacheDir, "updates/${update.versionName}")
                val file = download(update, directory) { progress ->
                    // The stream loop never suspends, so check here to stop reading after cancel.
                    currentCoroutineContext().ensureActive()
                    mutable.update {
                        if (it.phase != AppUpdatePhase.Downloading) it
                        else it.copy(downloadedBytes = progress.downloadedBytes, totalBytes = progress.totalBytes)
                    }
                }
                // A cancelled job must not report into a retry that already replaced it.
                currentCoroutineContext().ensureActive()
                // A finished download brings the dialog back so the user can start the install.
                mutable.update {
                    if (it.phase != AppUpdatePhase.Downloading) it
                    else it.copy(phase = AppUpdatePhase.Ready, readyFile = file, errorMessage = null, dialogHidden = false)
                }
            } catch (_: CancellationException) {
                // Cancellation is represented by the Available state in cancelDownload().
            } catch (error: Throwable) {
                if (!currentCoroutineContext().isActive) return@launch
                mutable.update {
                    if (it.phase != AppUpdatePhase.Downloading) it
                    else it.copy(
                        phase = AppUpdatePhase.Error,
                        readyFile = null,
                        errorMessage = error.message ?: "更新ファイルを取得できませんでした。",
                        dialogHidden = false,
                    )
                }
            }
        }
        return true
    }

    fun cancelDownload() {
        downloadJob?.let { job ->
            job.cancel()
            cancelledJob = job
        }
        downloadJob = null
        state.value.update?.let { update ->
            mutable.update {
                it.copy(
                    update = update,
                    phase = AppUpdatePhase.Available,
                    downloadedBytes = 0L,
                    totalBytes = update.apkSizeBytes,
                    readyFile = null,
                    errorMessage = null,
                    dialogHidden = false,
                )
            }
        }
    }

    /** Hides the dialog while the download continues; a progress bar stays on top of the screen. */
    fun hideDialog() {
        mutable.update { if (it.phase == AppUpdatePhase.Downloading) it.copy(dialogHidden = true) else it }
    }

    fun showDialog() {
        mutable.update { it.copy(dialogHidden = false) }
    }

    /**
     * Closes the dialog. A verified APK is kept, so "later" never forces a new download:
     * the update stays [AppUpdatePhase.Ready] and the top bar or settings reopen the dialog.
     */
    fun dismiss() {
        mutable.update {
            when (it.phase) {
                AppUpdatePhase.Downloading, AppUpdatePhase.Installing -> it
                // Declining the install permission also returns to Ready, so resuming the app does not start the installer.
                AppUpdatePhase.Ready, AppUpdatePhase.AwaitingInstallPermission ->
                    if (it.readyFile != null) it.copy(phase = AppUpdatePhase.Ready, dialogHidden = true)
                    else it.copy(update = null, dialogHidden = false)
                else -> it.copy(update = null, readyFile = null, dialogHidden = false)
            }
        }
    }

    internal fun close() {
        scope.cancel()
    }
}
