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
)

class AppUpdateViewModel(application: Application) : AndroidViewModel(application) {
    private val checker = AppUpdateChecker()
    private val downloader = AppUpdateDownloader()
    private val mutable = MutableStateFlow(AppUpdateUiState())
    val state = mutable.asStateFlow()
    private var checkJob: Job? = null
    private var downloadJob: Job? = null

    fun check(currentVersion: String) {
        if (checkJob?.isActive == true) return
        if (state.value.phase in setOf(
                AppUpdatePhase.Downloading,
                AppUpdatePhase.Ready,
                AppUpdatePhase.AwaitingInstallPermission,
                AppUpdatePhase.Installing,
            )) return

        if (state.value.update == null) {
            mutable.update { it.copy(phase = AppUpdatePhase.Checking, errorMessage = null) }
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
                when (val checked = result) {
                    is AppUpdateCheckResult.Available -> mutable.update {
                        it.copy(
                            update = checked.update,
                            phase = AppUpdatePhase.Available,
                            downloadedBytes = 0L,
                            totalBytes = checked.update.apkSizeBytes,
                            readyFile = null,
                            errorMessage = null,
                        )
                    }
                    AppUpdateCheckResult.UpToDate -> mutable.update { AppUpdateUiState(phase = AppUpdatePhase.UpToDate) }
                    is AppUpdateCheckResult.Failed -> mutable.update {
                        if (it.update == null) AppUpdateUiState(phase = AppUpdatePhase.Error, errorMessage = checked.message)
                        else it
                    }
                }
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
        checkJob?.cancel()
        downloadJob?.cancel()
        checker.cancel()
        super.onCleared()
    }
}
