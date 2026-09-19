package com.minashin1120.aiplayground.data

import kotlinx.coroutines.runBlocking
import mockwebserver3.MockResponse
import mockwebserver3.MockWebServer
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.security.MessageDigest

class AppUpdateDownloaderTest {
    @Test fun downloadsAndVerifiesGithubAsset() = runBlocking {
        MockWebServer().use { server ->
            server.start()
            val apk = "test apk bytes".toByteArray()
            val checksum = sha256(apk)
            server.enqueue(MockResponse.Builder().body("$checksum  app-release.apk\n").build())
            server.enqueue(MockResponse.Builder().body(apk.decodeToString()).build())
            val baseUrl = server.url("/").toString()
            val update = AppUpdate(
                versionName = "1.13.15",
                tagName = "android-v1.13.15",
                apkUrl = "${baseUrl}app-release.apk",
                checksumUrl = "${baseUrl}app-release.apk.sha256",
                apkSizeBytes = apk.size.toLong(),
            )
            val directory = createTempDir(prefix = "app-update-test")
            try {
                val progress = mutableListOf<AppUpdateDownloadProgress>()
                val file = AppUpdateDownloader(isAllowedUrl = { it.startsWith(baseUrl) })
                    .download(update, directory) { progress += it }

                assertArrayEquals(apk, file.readBytes())
                assertEquals(apk.size.toLong(), progress.last().downloadedBytes)
                assertNull(server.takeRequest().headers["Authorization"])
                assertNull(server.takeRequest().headers["Cookie"])
            } finally {
                directory.deleteRecursively()
            }
        }
    }

    @Test fun checksumMismatchDoesNotLeaveApk() = runBlocking {
        MockWebServer().use { server ->
            server.start()
            server.enqueue(MockResponse.Builder().body("0".repeat(64)).build())
            server.enqueue(MockResponse.Builder().body("tampered").build())
            val baseUrl = server.url("/").toString()
            val directory = createTempDir(prefix = "app-update-test")
            try {
                val update = AppUpdate("1.13.15", "android-v1.13.15", "${baseUrl}apk", "${baseUrl}sha", 8L)
                val error = runCatching {
                    AppUpdateDownloader(isAllowedUrl = { it.startsWith(baseUrl) }).download(update, directory) {}
                }.exceptionOrNull()
                assertTrue(error?.message?.contains("検証") == true)
                assertTrue(directory.listFiles().orEmpty().none { it.extension == "apk" })
            } finally {
                directory.deleteRecursively()
            }
        }
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes).joinToString("") { byte -> "%02x".format(byte) }
}
