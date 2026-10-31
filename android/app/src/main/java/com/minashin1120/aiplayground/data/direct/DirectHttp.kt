package com.minashin1120.aiplayground.data.direct

import com.minashin1120.aiplayground.data.LogPolicy

import com.minashin1120.aiplayground.BuildConfig
import com.minashin1120.aiplayground.data.ActivityLog
import kotlinx.coroutines.suspendCancellableCoroutine
import okhttp3.Call
import okhttp3.Callback
import okhttp3.CookieJar
import okhttp3.EventListener
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Protocol
import okhttp3.Request
import okhttp3.RequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import org.json.JSONObject
import java.io.EOFException
import java.io.IOException
import java.net.SocketException
import java.net.SocketTimeoutException
import java.util.concurrent.TimeUnit
import java.util.UUID
import javax.net.ssl.SSLException

/** Transport wrappers can carry the SSL failure in their cause; never match provider error text. */
internal fun isSslFailure(error: Throwable): Boolean {
    val seen = HashSet<Throwable>()
    var cause: Throwable? = error
    while (cause != null && seen.add(cause)) {
        if (cause is SSLException) return true
        cause = cause.cause
    }
    return false
}

/** An AI provider answered with an error (HTTP status and the provider's message). */
class DirectApiException(val status: Int, override val message: String) : IOException(message)

/**
 * HTTP to AI providers for serverless mode. Only the providers' API hosts are reachable (so a key can
 * never be sent elsewhere), no cookies, no redirects, and request headers and bodies are never logged
 * ([ActivityLog] keeps only the host, the path without its query, the status and the time). Tests pass
 * their own [allowedHosts] and base URLs.
 */
class DirectHttp(
    private val allowedHosts: Set<String> = PROVIDER_HOSTS,
    private val allowCleartextLocalhost: Boolean = false,
) {
    private class Trace(val started: Long = System.nanoTime(), val op: String = UUID.randomUUID().toString()) {
        fun ms(): Long = (System.nanoTime() - started) / 1_000_000
    }

    private fun record(call: Call, trace: Trace, type: String, phase: String, bytes: Long? = null) {
        ActivityLog.log("direct", "op" to trace.op, "type" to type, "phase" to phase,
            "method" to call.request().method, "provider" to LogPolicy.provider(call.request().url.host),
            "path" to call.request().url.encodedPath, "ms" to trace.ms(), "bytes" to bytes)
    }

    private fun listener(call: Call): EventListener {
        val trace = call.request().tag(Trace::class.java) ?: return EventListener.NONE
        return object : EventListener() {
            override fun callStart(call: Call) = record(call, trace, "accepted", "start")
            override fun requestBodyStart(call: Call) = record(call, trace, "start", "start")
            override fun requestBodyEnd(call: Call, byteCount: Long) = record(call, trace, "start", "done", byteCount)
        }
    }

    private val client = OkHttpClient.Builder()
        .dns(com.minashin1120.aiplayground.data.NetworkDns).protocols(listOf(Protocol.HTTP_1_1)).cookieJar(CookieJar.NO_COOKIES)
        .followRedirects(false).followSslRedirects(false).retryOnConnectionFailure(false)
        .eventListenerFactory { call -> com.minashin1120.aiplayground.data.NetworkTrace(listener(call), 2, { it.request().tag(Trace::class.java)?.op }) }
        .connectTimeout(20, TimeUnit.SECONDS).readTimeout(660, TimeUnit.SECONDS)
        .writeTimeout(180, TimeUnit.SECONDS).callTimeout(0, TimeUnit.SECONDS).build()
    // Use the same transport for binary uploads and SSE so an HTTP/2 connection cannot stall either.
    private val streamingClient = client.newBuilder().protocols(listOf(Protocol.HTTP_1_1)).build()
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

    fun jsonBody(payload: JSONObject): RequestBody = StreamingJsonBody(payload)

    /** Runs the call; [consume] reads the response on OkHttp's thread. Cancelling the coroutine cancels the call. */
    suspend fun <T> execute(request: Request, consume: (Response) -> T): T = execute(request, client, consume)

    private suspend fun <T> execute(request: Request, httpClient: OkHttpClient, consume: (Response) -> T): T =
        suspendCancellableCoroutine { continuation ->
            val trace = Trace()
            val call = httpClient.newCall(request.newBuilder().tag(Trace::class.java, trace).build())
            // ログの収集を強化: provider host, path (no query, so no key), status and time; never headers or bodies.
            val host = request.url.host
            val path = request.url.encodedPath
            continuation.invokeOnCancellation {
                call.cancel()
                ActivityLog.log("direct.cancel", "op" to trace.op, "method" to request.method, "provider" to LogPolicy.provider(host), "path" to path,
                    "ms" to trace.ms())
            }
            call.enqueue(object : Callback {
                override fun onFailure(call: Call, e: IOException) {
                    if (call.isCanceled()) return
                    discardSslConnections(e, httpClient)
                    ActivityLog.log("direct.error", "op" to trace.op, "method" to request.method, "provider" to LogPolicy.provider(host), "path" to path,
                        "ms" to trace.ms(), "error" to e.javaClass.simpleName)
                    if (continuation.isActive) continuation.resumeWith(Result.failure(e))
                }

                override fun onResponse(call: Call, response: Response) {
                    ActivityLog.log("direct", "op" to trace.op, "type" to "status", "phase" to "done", "method" to request.method, "provider" to LogPolicy.provider(host), "path" to path,
                        "status" to response.code, "ms" to trace.ms())
                    val result = runCatching { response.use { consume(it) } }
                    result.exceptionOrNull()?.let { discardSslConnections(it, httpClient) }
                    if (call.isCanceled()) return
                    if (result.isFailure) ActivityLog.log("direct.read_error", "op" to trace.op, "provider" to LogPolicy.provider(host), "path" to path,
                        "ms" to trace.ms(), "error" to result.exceptionOrNull()?.javaClass?.simpleName)
                    else record(call, trace, "done", "done")
                    if (continuation.isActive) continuation.resumeWith(result)
                }
            })
        }

    private fun discardSslConnections(error: Throwable, httpClient: OkHttpClient) {
        // Do not replay a POST whose body may already have reached the provider. Evict only idle
        // connections; concurrent active generations must keep their sockets and cancellation scope.
        if (isSslFailure(error)) httpClient.connectionPool.evictAll()
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
    suspend fun postSse(url: String, headers: Map<String, String>, payload: JSONObject, maxLine: Long = 8L * 1024 * 1024, onEvent: (SseEvent) -> Boolean) {
        execute(request(url, headers).header("Accept", "text/event-stream").post(jsonBody(payload)).build(), streamingClient) { response ->
            if (!response.isSuccessful) throw providerError(response.code, response.body.string())
            val type = response.body.contentType()
            if (type != null && type.subtype != "event-stream") throw IOException("事業者APIからストリーム形式ではない応答が返されました。")
            var first = true
            val trace = response.request.tag(Trace::class.java)
            readSse(response.body.source(), maxLine = maxLine) { event ->
                if (first) {
                    first = false
                    if (trace != null) ActivityLog.log("direct", "op" to trace.op, "type" to "message", "phase" to "done",
                        "method" to response.request.method, "provider" to LogPolicy.provider(response.request.url.host),
                        "path" to response.request.url.encodedPath, "ms" to trace.ms())
                }
                onEvent(event)
            }
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
            var result: Pair<ByteArray?, String?>? = null
            var retries = 0
            while (result == null) {
                try {
                    result = execute(req) { response ->
                        if (response.isRedirect) null to response.header("Location")
                        else {
                            if (!response.isSuccessful) throw providerError(response.code, response.body.string())
                            response.body.byteStream().use { input -> input.readNBytesCompat(limit) } to null
                        }
                    }
                } catch (error: IOException) {
                    if (retries >= 1 || !isRetryableDownloadFailure(error)) throw error
                    retries++
                }
            }
            val (bytes, location) = result ?: throw IOException("ダウンロードに失敗しました。")
            if (bytes != null) return bytes
            target = location?.let { target.resolve(it) } ?: throw IOException("ダウンロードのリダイレクト先が不正です。")
            credentials = emptyMap()
        }
        throw IOException("リダイレクトが多すぎます。")
    }

    /** Generated files use short-lived URLs; retry a transient GET failure once without regenerating. */
    private fun isRetryableDownloadFailure(error: IOException): Boolean {
        var cause: Throwable? = error
        repeat(8) {
            val current = cause ?: return false
            if (current is SocketException || current is SocketTimeoutException || current is EOFException) return true
            cause = current.cause
        }
        return false
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
