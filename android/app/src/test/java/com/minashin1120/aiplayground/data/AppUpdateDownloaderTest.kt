package com.minashin1120.aiplayground.data

import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.async
import mockwebserver3.MockResponse
import mockwebserver3.MockWebServer
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.security.MessageDigest

class AppUpdateDownloaderTest {
    @Test fun cancellingDuringApkBodyReadClosesTheCallWithoutWaitingForReadTimeout() = runBlocking {
        MockWebServer().use { server ->
            server.start()
            val apk = "ab".toByteArray()
            server.enqueue(MockResponse.Builder().body(sha256(apk)).build())
            server.enqueue(MockResponse.Builder().throttleBody(1, 5, java.util.concurrent.TimeUnit.SECONDS)
                .body(apk.decodeToString()).build())
            val base = server.url("/").toString()
            val directory = createTempDir(prefix = "app-update-cancel-body")
            val receivedFirstByte = kotlinx.coroutines.CompletableDeferred<Unit>()
            val cancelled = java.util.concurrent.atomic.AtomicBoolean(false)
            val client = okhttp3.OkHttpClient.Builder().eventListener(object : okhttp3.EventListener() {
                override fun canceled(call: okhttp3.Call) { cancelled.set(true) }
            }).build()
            val job = kotlinx.coroutines.CoroutineScope(kotlinx.coroutines.Dispatchers.IO).async {
                AppUpdateDownloader(client, { it.startsWith(base) }).download(
                    AppUpdate("1.70.65", "android-v1.70.65", "${base}apk", "${base}sha", 2), directory,
                ) { receivedFirstByte.complete(Unit) }
            }
            try {
                kotlinx.coroutines.withTimeout(5_000) { receivedFirstByte.await() }
                kotlinx.coroutines.withTimeout(2_000) { job.cancel(); job.join() }
                assertTrue(cancelled.get())
                assertTrue(directory.listFiles().orEmpty().isEmpty())
                assertEquals(2, server.requestCount)
            } finally {
                job.cancel(); job.join()
                client.connectionPool.evictAll()
                directory.deleteRecursively()
            }
        }
    }

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
                assertTrue(progress.any { it.downloadedBytes > 0L })
                assertTrue(progress.all { it.totalBytes == apk.size.toLong() })
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

    @Test fun downloadsAndVerifiesApkWhenSystemDnsCannotResolveTheAssetHost() = runBlocking {
        MockWebServer().use { server ->
            server.start()
            val apk = "verified apk after DNS recovery".toByteArray()
            server.enqueue(MockResponse.Builder().body(sha256(apk)).build())
            server.enqueue(MockResponse.Builder().body(apk.decodeToString()).build())
            val base = server.url("/").newBuilder().host("assets.test").build().toString()
            var notices = 0
            val resolver = RecoveringDns(
                okhttp3.Dns { throw java.net.UnknownHostException("fake DNS failure") },
                okhttp3.Dns { listOf(java.net.InetAddress.getByAddress(byteArrayOf(127, 0, 0, 1))) },
                { it == "assets.test" }, { notices++ },
            )
            val client = okhttp3.OkHttpClient.Builder().dns(resolver).build()
            val directory = createTempDir(prefix = "app-update-dns-test")
            try {
                val update = AppUpdate("1.70.53", "android-v1.70.53", "${base}apk", "${base}sha", apk.size.toLong())
                val file = AppUpdateDownloader(client, { it.startsWith(base) }).download(update, directory) {}
                assertArrayEquals(apk, file.readBytes())
                assertTrue(notices > 0)
                assertEquals(2, server.requestCount)
                repeat(2) {
                    val request = server.takeRequest()
                    assertNull(request.headers["Authorization"])
                    assertNull(request.headers["Cookie"])
                }
            } finally {
                directory.deleteRecursively()
            }
        }
    }

    @Test fun exhaustedDnsFailureHasUsefulMessageAndLeavesNoApk() = runBlocking {
        val client = okhttp3.OkHttpClient.Builder().dns(okhttp3.Dns {
            throw java.net.UnknownHostException("fake private hostname")
        }).build()
        val directory = createTempDir(prefix = "app-update-dns-failure")
        try {
            val update = AppUpdate("1.70.53", "android-v1.70.53", "https://github.com/apk", "https://github.com/sha", 10L)
            val error = runCatching { AppUpdateDownloader(client).download(update, directory) {} }.exceptionOrNull()
            assertTrue(error?.message?.contains("名前を解決") == true)
            assertTrue(error?.message?.contains("fake private hostname") == false)
            assertTrue(directory.listFiles().orEmpty().isEmpty())
        } finally {
            directory.deleteRecursively()
        }
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes).joinToString("") { byte -> "%02x".format(byte) }
}
