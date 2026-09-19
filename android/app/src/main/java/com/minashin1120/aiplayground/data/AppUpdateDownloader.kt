package com.minashin1120.aiplayground.data

import kotlinx.coroutines.suspendCancellableCoroutine
import okhttp3.Call
import okhttp3.Callback
import okhttp3.CookieJar
import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import java.io.File
import java.io.IOException
import java.security.MessageDigest
import java.util.concurrent.TimeUnit

data class AppUpdateDownloadProgress(
    val downloadedBytes: Long,
    val totalBytes: Long?,
)

class AppUpdateDownloader internal constructor(
    private val client: OkHttpClient = OkHttpClient.Builder()
        .cookieJar(CookieJar.NO_COOKIES)
        .followRedirects(true)
        .followSslRedirects(true)
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .callTimeout(15, TimeUnit.MINUTES)
        .build(),
    private val isAllowedUrl: (String) -> Boolean = { url ->
        url.toHttpUrl().let { it.scheme == "https" && it.host == "github.com" }
    },
) {
    suspend fun download(
        update: AppUpdate,
        directory: File,
        onProgress: suspend (AppUpdateDownloadProgress) -> Unit,
    ): File {
        require(isAllowedUrl(update.apkUrl)) { "更新元がGitHubではありません。" }
        require(isAllowedUrl(update.checksumUrl)) { "検証元がGitHubではありません。" }
        if (update.apkSizeBytes != null && update.apkSizeBytes > MAX_APK_BYTES) {
            throw IOException("更新APKのサイズが上限を超えています。")
        }

        directory.mkdirs()
        val target = File(directory, "app-release-${update.versionName}.apk")
        val partial = File(directory, "${target.name}.part")
        partial.delete()
        try {
            val expectedHash = fetchExpectedHash(update.checksumUrl)
            val request = Request.Builder()
                .url(update.apkUrl)
                .header("Accept", "application/octet-stream")
                .header("User-Agent", "AIPlayground-Android/${update.versionName}")
                .get()
                .build()
            val response = execute(request)
            var downloaded = 0L
            response.use {
                if (!it.isSuccessful) throw IOException("GitHubからAPKを取得できませんでした（HTTP ${it.code}）。")
                val body = it.body ?: throw IOException("GitHubから空のAPKが返されました。")
                // The GitHub API asset size remains stable across the release-asset redirect,
                // while the CDN response may omit or alter Content-Length.
                val total = update.apkSizeBytes ?: body.contentLength().takeIf { length -> length > 0L }
                if (total != null && total > MAX_APK_BYTES) {
                    throw IOException("更新APKのサイズが上限を超えています。")
                }
                body.byteStream().use { input ->
                    partial.outputStream().buffered().use { output ->
                        val buffer = ByteArray(BUFFER_SIZE)
                        while (true) {
                            val count = input.read(buffer)
                            if (count < 0) break
                            downloaded += count
                            if (downloaded > MAX_APK_BYTES) {
                                throw IOException("更新APKのサイズが上限を超えています。")
                            }
                            output.write(buffer, 0, count)
                            onProgress(AppUpdateDownloadProgress(downloaded, total))
                        }
                    }
                }
                if (update.apkSizeBytes != null && downloaded != update.apkSizeBytes) {
                    throw IOException("更新APKのサイズが一致しません。")
                }
            }

            if (sha256(partial) != expectedHash) {
                throw IOException("更新APKの検証に失敗しました。")
            }
            target.delete()
            if (!partial.renameTo(target)) throw IOException("更新APKを保存できませんでした。")
            return target
        } catch (error: Throwable) {
            partial.delete()
            target.delete()
            throw error
        }
    }

    private suspend fun fetchExpectedHash(url: String): String {
        val response = execute(
            Request.Builder()
                .url(url)
                .header("Accept", "text/plain")
                .header("User-Agent", "AIPlayground-Android/update")
                .get()
                .build(),
        )
        return response.use {
            if (!it.isSuccessful) throw IOException("更新の検証情報を取得できませんでした（HTTP ${it.code}）。")
            val body = it.body ?: throw IOException("更新の検証情報が空です。")
            val text = readBoundedUtf8(body.byteStream(), CHECKSUM_LIMIT)
            CHECKSUM_PATTERN.find(text)?.groupValues?.get(1)?.lowercase()
                ?: throw IOException("更新の検証情報を解釈できませんでした。")
        }
    }

    private suspend fun execute(request: Request): Response = suspendCancellableCoroutine { continuation ->
        val call = client.newCall(request)
        continuation.invokeOnCancellation { call.cancel() }
        call.enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                if (continuation.isActive) continuation.resumeWith(Result.failure(e))
            }

            override fun onResponse(call: Call, response: Response) {
                if (continuation.isActive) continuation.resumeWith(Result.success(response))
                else response.close()
            }
        })
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered().use { input ->
            val buffer = ByteArray(BUFFER_SIZE)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { byte -> "%02x".format(byte) }
    }

    companion object {
        private const val BUFFER_SIZE = 64 * 1024
        private const val CHECKSUM_LIMIT = 4096
        private const val MAX_APK_BYTES = 200L * 1024L * 1024L
        private val CHECKSUM_PATTERN = Regex("(?i)\\b([0-9a-f]{64})\\b")
    }
}
