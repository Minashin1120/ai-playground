package com.minashin1120.aiplayground

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.minashin1120.aiplayground.data.AppUpdate
import com.minashin1120.aiplayground.data.AppUpdateChecker
import com.minashin1120.aiplayground.data.AppUpdateCheckResult
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume
import java.io.File

enum class AppUpdatePhase {
    Checking,
    UpToDate,
    Available,
    Downloading,
    Ready,
    AwaitingInstallPermission,
    Installing,
    Error,
}

data class AppUpdateUiState(
    val update: AppUpdate? = null,
    val phase: AppUpdatePhase = AppUpdatePhase.Checking,
    val downloadedBytes: Long = 0L,
    val totalBytes: Long? = null,
    val readyFile: File? = null,
    val errorMessage: String? = null,
    /** True while the user keeps a download running behind a progress bar instead of the dialog. */
    val dialogHidden: Boolean = false,
)

class AppUpdateViewModel(application: Application) : AndroidViewModel(application) {
    private val checker = AppUpdateChecker()
    private val downloads = (application as PlaygroundApplication).appUpdates
    val state = downloads.state
    private var checkJob: Job? = null

    fun check(currentVersion: String) {
        if (checkJob?.isActive == true) return
        if (state.value.phase in BUSY_PHASES) return

        if (state.value.update == null) {
            downloads.updateState { it.copy(phase = AppUpdatePhase.Checking, errorMessage = null) }
        }
        checkJob = viewModelScope.launch {
            try {
                var result: AppUpdateCheckResult = AppUpdateCheckResult.Failed("更新情報を取得できませんでした。")
                for (attempt in 0 until 3) {
                    result = try {
                        checkOnce(currentVersion)
                    } catch (cancelled: CancellationException) {
                        throw cancelled
                    } catch (_: Exception) {
                        AppUpdateCheckResult.Failed("更新情報を取得できませんでした。通信状態を確認してください。")
                    }
                    if (result !is AppUpdateCheckResult.Failed || attempt == 2) break
                    delay(if (attempt == 0) 1_500L else 4_000L)
                }
                // The download is process-wide, so it may have started while this check was running.
                val checked = result
                downloads.updateState { current -> if (current.phase in BUSY_PHASES) current else when (checked) {
                    is AppUpdateCheckResult.Available -> current.copy(
                        update = checked.update,
                        phase = AppUpdatePhase.Available,
                        downloadedBytes = 0L,
                        totalBytes = checked.update.apkSizeBytes,
                        readyFile = null,
                        errorMessage = null,
                    )
                    AppUpdateCheckResult.UpToDate -> AppUpdateUiState(phase = AppUpdatePhase.UpToDate)
                    is AppUpdateCheckResult.Failed ->
                        if (current.update == null) AppUpdateUiState(phase = AppUpdatePhase.Error, errorMessage = checked.message)
                        else current
                } }
            } finally {
                checkJob = null
            }
        }
    }

    private suspend fun checkOnce(currentVersion: String): AppUpdateCheckResult =
        suspendCancellableCoroutine { continuation ->
            checker.check(currentVersion) { result ->
                if (continuation.isActive) continuation.resume(result)
            }
            continuation.invokeOnCancellation { checker.cancel() }
        }

    fun startDownload(): Boolean = downloads.startDownload()

    fun cancelDownload() = downloads.cancelDownload()

    fun hideDialog() = downloads.hideDialog()

    fun showDialog() = downloads.showDialog()

    fun awaitInstallPermission() {
        downloads.updateState { it.copy(phase = AppUpdatePhase.AwaitingInstallPermission) }
    }

    fun markInstalling() {
        downloads.updateState { it.copy(phase = AppUpdatePhase.Installing) }
    }

    fun installerClosed() {
        if (state.value.phase == AppUpdatePhase.Installing) {
            downloads.updateState { it.copy(phase = AppUpdatePhase.Ready) }
        }
    }

    fun installFailed(message: String) {
        downloads.updateState { it.copy(phase = AppUpdatePhase.Error, errorMessage = message) }
    }

    fun dismiss() = downloads.dismiss()

    private companion object {
        val BUSY_PHASES = setOf(
            AppUpdatePhase.Downloading,
            AppUpdatePhase.Ready,
            AppUpdatePhase.AwaitingInstallPermission,
            AppUpdatePhase.Installing,
        )
    }

    override fun onCleared() {
        // The download belongs to the process-wide manager and keeps running without this screen.
        checkJob?.cancel()
        checker.cancel()
        super.onCleared()
    }
}
