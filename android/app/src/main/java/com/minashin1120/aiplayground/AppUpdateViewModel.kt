package com.minashin1120.aiplayground

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.minashin1120.aiplayground.data.AppUpdate
import com.minashin1120.aiplayground.data.AppUpdateChecker
import com.minashin1120.aiplayground.data.AppUpdateCheckResult
import com.minashin1120.aiplayground.data.AppUpdateDownloadProgress
import com.minashin1120.aiplayground.data.AppUpdateDownloader
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
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
)

class AppUpdateViewModel(application: Application) : AndroidViewModel(application) {
    private val checker = AppUpdateChecker()
    private val downloader = AppUpdateDownloader()
    private val mutable = MutableStateFlow(AppUpdateUiState())
    val state = mutable.asStateFlow()
    private var downloadJob: Job? = null

    fun check(currentVersion: String) {
        mutable.update { AppUpdateUiState(phase = AppUpdatePhase.Checking) }
        checker.check(currentVersion) { result ->
            mutable.update {
                when (result) {
                    is AppUpdateCheckResult.Available -> it.copy(
                        update = result.update,
                        phase = AppUpdatePhase.Available,
                        downloadedBytes = 0L,
                        totalBytes = result.update.apkSizeBytes,
                        readyFile = null,
                        errorMessage = null,
                    )
                    AppUpdateCheckResult.UpToDate -> AppUpdateUiState(phase = AppUpdatePhase.UpToDate)
                    is AppUpdateCheckResult.Failed -> AppUpdateUiState(
                        phase = AppUpdatePhase.Error,
                        errorMessage = result.message,
                    )
                }
            }
        }
    }

    fun startDownload() {
        val update = state.value.update ?: return
        if (downloadJob?.isActive == true) return
        mutable.update {
            it.copy(
                phase = AppUpdatePhase.Downloading,
                downloadedBytes = 0L,
                totalBytes = update.apkSizeBytes,
                readyFile = null,
                errorMessage = null,
            )
        }
        // The response body is consumed after the suspending OkHttp call resumes. Keep that
        // potentially long-running stream read off the main thread so Compose can render each
        // progress update while the APK is downloading.
        downloadJob = viewModelScope.launch(Dispatchers.IO) {
            try {
                val directory = File(getApplication<Application>().cacheDir, "updates/${update.versionName}")
                val file = downloader.download(update, directory) { progress ->
                    // Publish each progress update on the UI dispatcher while the stream read
                    // remains on IO, so rapid reads do not block Compose rendering.
                    withContext(Dispatchers.Main.immediate) {
                        updateProgress(progress)
                    }
                }
                withContext(Dispatchers.Main.immediate) {
                    mutable.update { it.copy(phase = AppUpdatePhase.Ready, readyFile = file, errorMessage = null) }
                }
            } catch (_: CancellationException) {
                // Cancellation is represented by the Available state in cancelDownload().
            } catch (error: Throwable) {
                withContext(Dispatchers.Main.immediate) {
                    mutable.update {
                        it.copy(
                            phase = AppUpdatePhase.Error,
                            readyFile = null,
                            errorMessage = error.message ?: "更新ファイルを取得できませんでした。",
                        )
                    }
                }
            }
        }
    }

    fun cancelDownload() {
        downloadJob?.cancel()
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
                )
            }
        }
    }

    fun retryDownload() {
        startDownload()
    }

    fun awaitInstallPermission() {
        mutable.update { it.copy(phase = AppUpdatePhase.AwaitingInstallPermission) }
    }

    fun markInstalling() {
        mutable.update { it.copy(phase = AppUpdatePhase.Installing) }
    }

    fun installerClosed() {
        if (state.value.phase == AppUpdatePhase.Installing) {
            mutable.update { it.copy(phase = AppUpdatePhase.Ready) }
        }
    }

    fun installFailed(message: String) {
        mutable.update { it.copy(phase = AppUpdatePhase.Error, errorMessage = message) }
    }

    fun dismiss() {
        if (state.value.phase != AppUpdatePhase.Downloading) {
            mutable.update { it.copy(update = null, readyFile = null) }
        }
    }

    private fun updateProgress(progress: AppUpdateDownloadProgress) {
        mutable.update {
            it.copy(downloadedBytes = progress.downloadedBytes, totalBytes = progress.totalBytes)
        }
    }

    override fun onCleared() {
        downloadJob?.cancel()
        checker.cancel()
        super.onCleared()
    }
}
