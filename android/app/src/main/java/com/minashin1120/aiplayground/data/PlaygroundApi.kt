package com.minashin1120.aiplayground.data

import com.minashin1120.aiplayground.BuildConfig
import kotlinx.coroutines.suspendCancellableCoroutine
import okhttp3.Call
import okhttp3.Callback
import okhttp3.CookieJar
import okhttp3.HttpUrl
import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.util.concurrent.TimeUnit

class PlaygroundApi internal constructor(private val origin: HttpUrl) {
    constructor() : this(BuildConfig.BASE_URL.toHttpUrl())
    private val jsonType = "application/json; charset=utf-8".toMediaType()
    private val normal = OkHttpClient.Builder().cookieJar(CookieJar.NO_COOKIES)
        .followRedirects(false).followSslRedirects(false).retryOnConnectionFailure(false)
        .connectTimeout(15, TimeUnit.SECONDS).readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(120, TimeUnit.SECONDS).callTimeout(180, TimeUnit.SECONDS).build()
    private val streaming = normal.newBuilder().readTimeout(660, TimeUnit.SECONDS)
        .callTimeout(0, TimeUnit.SECONDS).build()

    internal fun url(path: String): HttpUrl {
        require(path.startsWith('/') && !path.startsWith("//")) { "Invalid API path" }
        val result = requireNotNull(origin.resolve(path))
        require(result.scheme == origin.scheme && result.host == origin.host && result.port == origin.port)
        return result
    }
    private fun request(path: String, token: String?): Request.Builder = Request.Builder().url(url(path))
        .header("User-Agent", "AIPlayground-Android/${BuildConfig.VERSION_NAME}")
        .apply { if (token != null) header("Authorization", "Bearer $token") }

    private suspend fun <T> execute(req: Request, client: OkHttpClient = normal, consume: (Response) -> T): T =
        suspendCancellableCoroutine { continuation ->
            val call = client.newCall(req)
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

    private fun error(response: Response): ApiException {
        val payload = runCatching {
            JSONObject(readBoundedUtf8(response.body.byteStream(), 1024 * 1024))
        }.getOrElse { JSONObject().put("error", "サーバーとの通信に失敗しました（HTTP ${response.code}）。") }
        return ApiException(response.code, payload, response.header("Retry-After")?.toLongOrNull() ?: 5)
    }
    private fun jsonResponse(response: Response): JSONObject {
        if (!response.isSuccessful) throw error(response)
        if (response.body.contentType()?.subtype != "json") throw IOException("JSON以外の応答です。Webで接続状態を確認してください。")
        return JSONObject(readBoundedUtf8(response.body.byteStream(), 8 * 1024 * 1024))
    }
    suspend fun get(path: String, token: String? = null): JSONObject = execute(
        request(path, token).header("Accept", "application/json").build(), consume = ::jsonResponse)
    /** Some existing endpoints (for example `/api/gems`) return a top-level JSON array. */
    suspend fun getArray(path: String, token: String? = null): JSONArray = execute(
        request(path, token).header("Accept", "application/json").build()
    ) { response ->
        if (!response.isSuccessful) throw error(response)
        if (response.body.contentType()?.subtype != "json") throw IOException("JSON以外の応答です。Webで接続状態を確認してください。")
        JSONArray(readBoundedUtf8(response.body.byteStream(), 8 * 1024 * 1024))
    }
    suspend fun post(path: String, payload: JSONObject, token: String? = null): JSONObject = execute(
        request(path, token).header("Accept", "application/json")
            .post(payload.toString().toRequestBody(jsonType)).build(), consume = ::jsonResponse)
    suspend fun put(path: String, payload: JSONObject, token: String): JSONObject = execute(
        request(path, token).header("Accept", "application/json")
            .put(payload.toString().toRequestBody(jsonType)).build(), consume = ::jsonResponse)
    suspend fun delete(path: String, token: String): JSONObject = execute(
        request(path, token).delete().build(), consume = ::jsonResponse)

    suspend fun upload(name: String, body: RequestBody, token: String): JSONObject {
        val multipart = MultipartBody.Builder().setType(MultipartBody.FORM)
            .addFormDataPart("file", name, body).build()
        return execute(request("/upload", token).post(multipart).build(), consume = ::jsonResponse)
    }

    /** Starts a resumable chunk session; the server returns its fixed chunk size. */
    suspend fun uploadInit(filename: String, size: Long, token: String): JSONObject =
        post("/upload/init", JSONObject().put("filename", filename).put("size", size), token)

    suspend fun uploadChunk(uploadId: String, index: Int, total: Int, chunk: ByteArray, token: String): JSONObject {
        val multipart = MultipartBody.Builder().setType(MultipartBody.FORM)
            .addFormDataPart("upload_id", uploadId)
            .addFormDataPart("index", index.toString())
            .addFormDataPart("total", total.toString())
            .addFormDataPart("chunk", "chunk", chunk.toRequestBody("application/octet-stream".toMediaType()))
            .build()
        return execute(request("/upload/chunk", token).post(multipart).build(), consume = ::jsonResponse)
    }

    suspend fun uploadComplete(uploadId: String, token: String): JSONObject =
        post("/upload/complete", JSONObject().put("upload_id", uploadId), token)

    suspend fun stream(path: String, payload: JSONObject, token: String, onEvent: (JSONObject) -> Unit) {
        val req = request(path, token).header("Accept", "application/x-ndjson")
            .post(payload.toString().toRequestBody(jsonType)).build()
        execute(req, streaming) { response ->
            if (!response.isSuccessful) throw error(response)
            if (response.body.contentType()?.subtype != "x-ndjson") throw IOException("チャットの応答形式が正しくありません。")
            val source = response.body.source()
            var terminal = false
            while (!source.exhausted()) {
                val line = source.readUtf8LineStrict(8L * 1024 * 1024)
                if (line.isBlank()) continue
                val event = JSONObject(line)
                onEvent(event)
                if (event.optString("type") in setOf("done", "error")) { terminal = true; break }
            }
            if (!terminal) throw IOException("通信が中断されました。再接続して保存済みの履歴を確認してください。")
        }
    }

    private fun filePath(ref: String, thumbnail: Boolean): String =
        origin.newBuilder().addPathSegment("files").apply {
            if (thumbnail) addPathSegment("thumb")
            ref.split('/').forEach { addPathSegment(it) }
        }.build().encodedPath

    suspend fun download(reference: String, destination: File, token: String): String {
        // Never attach credentials to a provider URL or another origin.
        val ref = fileReferencePath(reference) ?: throw IOException("添付の参照が不正です。")
        val path = filePath(ref, thumbnail = false)
        return execute(request(path, token).build()) { response ->
            if (!response.isSuccessful) throw error(response)
            val mime = response.body.contentType()?.let { "${it.type}/${it.subtype}" } ?: "application/octet-stream"
            var count = 0L
            try {
                response.body.byteStream().use { input -> destination.outputStream().use { output ->
                    val buffer = ByteArray(8192)
                    while (true) {
                        val size = input.read(buffer)
                        if (size < 0) break
                        count += size
                        if (count > 100L * 1024 * 1024) throw IOException("添付が100MiBを超えています。Webから取得してください。")
                        output.write(buffer, 0, size)
                    }
                } }
            } catch (e: Exception) { destination.delete(); throw e }
            mime
        }
    }

    /** Loads a same-origin attachment or its WebP thumbnail within a byte budget. */
    suspend fun loadFileBytes(reference: String, token: String, thumbnail: Boolean, limit: Long): ByteArray {
        val ref = fileReferencePath(reference) ?: throw IOException("添付の参照が不正です。")
        return execute(request(filePath(ref, thumbnail), token).build()) { response ->
            if (!response.isSuccessful) throw error(response)
            readBoundedBytes(response.body.byteStream(), limit)
        }
    }
}
