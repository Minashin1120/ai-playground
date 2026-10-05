package com.minashin1120.aiplayground.data.direct

import com.minashin1120.aiplayground.BuildConfig
import kotlinx.coroutines.suspendCancellableCoroutine
import okhttp3.Call
import okhttp3.Callback
import okhttp3.CookieJar
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.TimeUnit

/** An AI provider answered with an error (HTTP status and the provider's message). */
class DirectApiException(val status: Int, override val message: String) : IOException(message)

/**
 * HTTP to AI providers for serverless mode. Only the providers' API hosts are reachable (so a key can
 * never be sent elsewhere), no cookies, no redirects, and requests are never logged. Tests pass their
 * own [allowedHosts] and base URLs.
 */
class DirectHttp(
    private val allowedHosts: Set<String> = PROVIDER_HOSTS,
    private val allowCleartextLocalhost: Boolean = false,
) {
    private val client = OkHttpClient.Builder().cookieJar(CookieJar.NO_COOKIES)
        .followRedirects(false).followSslRedirects(false).retryOnConnectionFailure(false)
        .connectTimeout(20, TimeUnit.SECONDS).readTimeout(660, TimeUnit.SECONDS)
        .writeTimeout(180, TimeUnit.SECONDS).callTimeout(0, TimeUnit.SECONDS).build()
    private val jsonType = "application/json; charset=utf-8".toMediaType()

    fun request(url: String, headers: Map<String, String>): Request.Builder {
        val target = url.toHttpUrlOrNull() ?: throw IOException("接続先のURLが不正です。")
        val local = allowCleartextLocalhost && target.host in setOf("localhost", "127.0.0.1")
        if (!local && (target.scheme != "https" || !hostAllowed(target.host))) throw IOException("許可されていない接続先です: ${target.host}")
        return Request.Builder().url(target).header("User-Agent", "AIPlayground-Android/${BuildConfig.VERSION_NAME}")
            .apply { headers.forEach { (name, value) -> header(name, value) } }
    }

    /** Provider API hosts, plus Vertex AI's regional hosts (`<region>-aiplatform.googleapis.com`). */
    private fun hostAllowed(host: String): Boolean = host in allowedHosts ||
        (allowedHosts === PROVIDER_HOSTS && Regex("^[a-z0-9-]+-aiplatform\\.googleapis\\.com$").matches(host))

    fun jsonBody(payload: JSONObject): RequestBody = payload.toString().toRequestBody(jsonType)

    /** Runs the call; [consume] reads the response on OkHttp's thread. Cancelling the coroutine cancels the call. */
    suspend fun <T> execute(request: Request, consume: (Response) -> T): T = suspendCancellableCoroutine { continuation ->
        val call = client.newCall(request)
        continuation.invokeOnCancellation { call.cancel() }
        call.enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                if (continuation.isActive) continuation.resumeWith(Result.failure(e))
            }

            override fun onResponse(call: Call, response: Response) {
                val result = runCatching { response.use { consume(it) } }
                if (continuation.isActive) continuation.resumeWith(result)
            }
        })
    }

    suspend fun postJson(url: String, headers: Map<String, String>, payload: JSONObject): JSONObject =
        execute(request(url, headers).post(jsonBody(payload)).build()) { response ->
            val text = response.body.string()
            if (!response.isSuccessful) throw providerError(response.code, text)
            runCatching { JSONObject(text) }.getOrElse { throw IOException("事業者APIの応答を読み取れませんでした。") }
        }

    suspend fun getJson(url: String, headers: Map<String, String>): JSONObject =
        execute(request(url, headers).get().build()) { response ->
            val text = response.body.string()
            if (!response.isSuccessful) throw providerError(response.code, text)
            runCatching { JSONObject(text) }.getOrElse { throw IOException("事業者APIの応答を読み取れませんでした。") }
        }

    /** POSTs [payload] and returns the response header [name] (the Gemini Files API resumable upload URL). */
    suspend fun postJsonForHeader(url: String, headers: Map<String, String>, payload: JSONObject, name: String): String =
        execute(request(url, headers).post(jsonBody(payload)).build()) { response ->
            val text = response.body.string()
            if (!response.isSuccessful) throw providerError(response.code, text)
            response.header(name)?.takeIf { it.isNotBlank() } ?: throw IOException("アップロード先を取得できませんでした。")
        }

    /** POSTs [payload] and hands each Server-Sent Event to [onEvent] until it returns false or the stream ends. */
    suspend fun postSse(url: String, headers: Map<String, String>, payload: JSONObject, onEvent: (SseEvent) -> Boolean) {
        execute(request(url, headers).header("Accept", "text/event-stream").post(jsonBody(payload)).build()) { response ->
            if (!response.isSuccessful) throw providerError(response.code, response.body.string())
            readSse(response.body.source(), onEvent = onEvent)
        }
    }

    suspend fun postBytes(url: String, headers: Map<String, String>, body: RequestBody, limit: Long = 64L * 1024 * 1024): Pair<ByteArray, String> =
        execute(request(url, headers).post(body).build()) { response ->
            if (!response.isSuccessful) throw providerError(response.code, response.body.string())
            val mime = response.body.contentType()?.let { "${it.type}/${it.subtype}" } ?: "application/octet-stream"
            val bytes = response.body.byteStream().use { input -> input.readNBytesCompat(limit) }
            bytes to mime
        }

    /**
     * Downloads a generated file. [headers] (the API key) go only to an allowed provider host on the first
     * request; redirects (to storage hosts, at most 4, https only) are followed without them.
     */
    suspend fun download(url: String, headers: Map<String, String>, limit: Long): ByteArray {
        var target = url.toHttpUrlOrNull() ?: throw IOException("ダウンロード先のURLが不正です。")
        var credentials = headers
        repeat(5) {
            val local = allowCleartextLocalhost && target.host in setOf("localhost", "127.0.0.1")
            if (target.scheme != "https" && !local) throw IOException("httpsではないダウンロード先です。")
            val send = if (hostAllowed(target.host) || local) credentials else emptyMap()
            val req = Request.Builder().url(target).header("User-Agent", "AIPlayground-Android/${BuildConfig.VERSION_NAME}")
                .apply { send.forEach { (name, value) -> header(name, value) } }.get().build()
            val (bytes, location) = execute(req) { response ->
                if (response.isRedirect) null to response.header("Location")
                else {
                    if (!response.isSuccessful) throw providerError(response.code, response.body.string())
                    response.body.byteStream().use { input -> input.readNBytesCompat(limit) } to null
                }
            }
            if (bytes != null) return bytes
            target = location?.let { target.resolve(it) } ?: throw IOException("ダウンロードのリダイレクト先が不正です。")
            credentials = emptyMap()
        }
        throw IOException("リダイレクトが多すぎます。")
    }

    companion object {
        val PROVIDER_HOSTS = setOf(
            "generativelanguage.googleapis.com", "aiplatform.googleapis.com", "oauth2.googleapis.com",
            "texttospeech.googleapis.com", "api.openai.com", "api.anthropic.com", "api.x.ai",
            "api.deepseek.com", "api.moonshot.ai", "api.mistral.ai", "api.z.ai", "api.ideogram.ai",
        )

        /** The provider's own error message (`{"error":{"message":…}}` and similar), without echoing request data. */
        fun providerError(status: Int, body: String): DirectApiException {
            val json = runCatching { JSONObject(body) }.getOrNull()
            val error = json?.opt("error")
            val message = when (error) {
                is JSONObject -> error.optString("message")
                is String -> error
                else -> json?.optString("message").orEmpty()
            }.ifBlank { json?.optJSONArray("detail")?.optJSONObject(0)?.optString("msg").orEmpty() }
                .ifBlank { "HTTP $status" }.take(2000)
            return DirectApiException(status, message)
        }
    }
}

internal fun java.io.InputStream.readNBytesCompat(limit: Long): ByteArray {
    val output = java.io.ByteArrayOutputStream()
    val buffer = ByteArray(64 * 1024)
    var total = 0L
    while (true) {
        val count = read(buffer)
        if (count < 0) break
        total += count
        if (total > limit) throw IOException("応答が大きすぎます。")
        output.write(buffer, 0, count)
    }
    return output.toByteArray()
}
