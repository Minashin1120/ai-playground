package com.minashin1120.aiplayground

import com.minashin1120.aiplayground.data.AppUpdate
import com.minashin1120.aiplayground.data.AppUpdateDownloadProgress
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.io.IOException

class AppUpdateDownloadManagerTest {
    private val update = AppUpdate(
        versionName = "9.9.9",
        tagName = "android-v9.9.9",
        apkUrl = "https://github.com/example/app-release.apk",
        checksumUrl = "https://github.com/example/app-release.apk.sha256",
        apkSizeBytes = 100L,
    )
    private val cacheDir = createTempDir(prefix = "app-update-manager-test")
    private val managers = mutableListOf<AppUpdateDownloadManager>()

    @After fun tearDown() {
        managers.forEach { it.close() }
        cacheDir.deleteRecursively()
    }

    private fun manager(
        download: suspend (AppUpdate, File, suspend (AppUpdateDownloadProgress) -> Unit) -> File,
    ) = AppUpdateDownloadManager(cacheDir, download, CoroutineScope(SupervisorJob() + Dispatchers.Default)).also {
        managers += it
        it.updateState { state -> state.copy(update = update, phase = AppUpdatePhase.Available) }
    }

    private suspend fun AppUpdateDownloadManager.awaitState(predicate: (AppUpdateUiState) -> Boolean) =
        withTimeout(5_000L) { state.first(predicate) }

    @Test fun hiddenDownloadKeepsRunningAndCanBeShownAgain() = runBlocking {
        val progressed = CompletableDeferred<Unit>()
        val manager = manager { _, _, onProgress ->
            onProgress(AppUpdateDownloadProgress(40L, 100L))
            progressed.complete(Unit)
            awaitCancellation()
        }
        assertTrue(manager.startDownload())
        progressed.await()
        manager.hideDialog()

        val hidden = manager.awaitState { it.downloadedBytes == 40L }
        assertEquals(AppUpdatePhase.Downloading, hidden.phase)
        assertTrue(hidden.dialogHidden)
        // Hiding the dialog must not dismiss the pending update.
        manager.dismiss()
        assertEquals(update, manager.state.value.update)

        manager.showDialog()
        assertFalse(manager.state.value.dialogHidden)
        assertEquals(AppUpdatePhase.Downloading, manager.state.value.phase)
    }

    @Test fun finishedDownloadReopensHiddenDialog() = runBlocking {
        val release = CompletableDeferred<Unit>()
        val apk = File(cacheDir, "app.apk")
        val manager = manager { _, _, _ -> release.await(); apk }
        manager.startDownload()
        manager.hideDialog()
        assertTrue(manager.state.value.dialogHidden)

        release.complete(Unit)
        val ready = manager.awaitState { it.phase == AppUpdatePhase.Ready }
        assertFalse(ready.dialogHidden)
        assertEquals(apk, ready.readyFile)
    }

    @Test fun failedDownloadReopensHiddenDialogWithError() = runBlocking {
        val release = CompletableDeferred<Unit>()
        val manager = manager { _, _, _ -> release.await(); throw IOException("検証に失敗しました。") }
        manager.startDownload()
        manager.hideDialog()

        release.complete(Unit)
        val failed = manager.awaitState { it.phase == AppUpdatePhase.Error }
        assertFalse(failed.dialogHidden)
        assertEquals("検証に失敗しました。", failed.errorMessage)
        assertNull(failed.readyFile)
    }

    @Test fun cancelReturnsToAvailableAndIgnoresLateProgress() = runBlocking {
        val started = CompletableDeferred<Unit>()
        val release = CompletableDeferred<Unit>()
        val manager = manager { _, _, onProgress ->
            started.complete(Unit)
            release.await()
            onProgress(AppUpdateDownloadProgress(90L, 100L))
            File(cacheDir, "late.apk")
        }
        manager.startDownload()
        started.await()
        manager.hideDialog()
        manager.cancelDownload()
        release.complete(Unit)

        val state = manager.state.value
        assertEquals(AppUpdatePhase.Available, state.phase)
        assertFalse(state.dialogHidden)
        assertEquals(0L, state.downloadedBytes)
        assertNull(state.readyFile)
    }

    @Test fun hideIsIgnoredOutsideDownloadAndStartNeedsUpdate() {
        val manager = manager { _, _, _ -> error("not started") }
        manager.hideDialog()
        assertFalse(manager.state.value.dialogHidden)

        manager.updateState { it.copy(update = null) }
        assertFalse(manager.startDownload())
        assertEquals(AppUpdatePhase.Available, manager.state.value.phase)
    }

    @Test fun dismissingReadyDialogKeepsTheDownloadedApk() = runBlocking {
        val apk = File(cacheDir, "app.apk")
        val manager = manager { _, _, _ -> apk }
        manager.startDownload()
        manager.awaitState { it.phase == AppUpdatePhase.Ready }

        manager.dismiss()
        val hidden = manager.state.value
        assertEquals(AppUpdatePhase.Ready, hidden.phase)
        assertEquals(update, hidden.update)
        assertEquals(apk, hidden.readyFile)
        assertTrue(hidden.dialogHidden)

        manager.showDialog()
        assertFalse(manager.state.value.dialogHidden)
        assertEquals(apk, manager.state.value.readyFile)
    }

    @Test fun dismissingInstallPermissionPromptReturnsToReady() = runBlocking {
        val apk = File(cacheDir, "app.apk")
        val manager = manager { _, _, _ -> apk }
        manager.startDownload()
        manager.awaitState { it.phase == AppUpdatePhase.Ready }
        manager.updateState { it.copy(phase = AppUpdatePhase.AwaitingInstallPermission) }

        manager.dismiss()
        assertEquals(AppUpdatePhase.Ready, manager.state.value.phase)
        assertEquals(apk, manager.state.value.readyFile)
    }

    @Test fun dismissingAvailableOrErrorForgetsTheUpdate() {
        val manager = manager { _, _, _ -> error("not started") }
        manager.dismiss()
        assertNull(manager.state.value.update)
    }

    @Test fun secondStartWhileRunningIsIgnored() = runBlocking {
        val manager = manager { _, _, _ -> awaitCancellation() }
        assertTrue(manager.startDownload())
        assertFalse(manager.startDownload())
        manager.cancelDownload()
    }
}
