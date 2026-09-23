package com.minashin1120.aiplayground.data

import android.content.Context
import com.google.android.play.core.integrity.IntegrityManagerFactory
import com.google.android.play.core.integrity.StandardIntegrityManager
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.suspendCancellableCoroutine
import org.json.JSONArray
import org.json.JSONObject
import java.security.MessageDigest
import java.util.Base64
import java.util.UUID
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/** Requests an on-demand Standard API token bound to one auth endpoint and attempt. */
class PlayIntegrityClient(context: Context) {
    private val manager = IntegrityManagerFactory.createStandard(context.applicationContext)
    @Volatile private var provider: StandardIntegrityManager.StandardIntegrityTokenProvider? = null
    @Volatile private var projectNumber: Long = 0L

    suspend fun requestFields(endpoint: String, cloudProjectNumber: String, body: JSONObject): JSONObject {
        val requestId = UUID.randomUUID().toString()
        val requestHash = requestHash(endpoint, requestId, body)
        var token = ""
        try {
            val number = cloudProjectNumber.toLong()
            var current = provider
            if (current == null || projectNumber != number) {
                current = awaitTask(manager.prepareIntegrityToken(
                    StandardIntegrityManager.PrepareIntegrityTokenRequest.builder()
                        .setCloudProjectNumber(number).build()
                ))
                provider = current
                projectNumber = number
            }
            token = awaitTask(current.request(StandardIntegrityManager.StandardIntegrityTokenRequest.builder()
                .setRequestHash(requestHash).build())).token()
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (_: Exception) {
            // The server treats an unavailable verdict as risk and offers Turnstile.
        }
        return JSONObject().put("integrity_request_id", requestId)
            .put("integrity_token", token)
    }

    private suspend fun <T> awaitTask(task: com.google.android.gms.tasks.Task<T>): T =
        suspendCancellableCoroutine { continuation ->
            task.addOnSuccessListener { value -> if (continuation.isActive) continuation.resume(value) }
            task.addOnFailureListener { error -> if (continuation.isActive) continuation.resumeWithException(error) }
            task.addOnCanceledListener { if (continuation.isActive) continuation.cancel() }
        }

    companion object {
        internal fun requestHash(endpoint: String, requestId: String, body: JSONObject): String {
            val bodyHash = MessageDigest.getInstance("SHA-256").digest(canonical(body).toByteArray(Charsets.UTF_8))
            val bodyHashText = Base64.getUrlEncoder().withoutPadding().encodeToString(bodyHash)
            val binding = "official-android\nPOST\n$endpoint\n$requestId\n$bodyHashText"
            val digest = MessageDigest.getInstance("SHA-256").digest(binding.toByteArray(Charsets.UTF_8))
            return Base64.getUrlEncoder().withoutPadding().encodeToString(digest)
        }

        private fun canonical(value: Any?): String = when (value) {
            is JSONObject -> value.keys().asSequence().toList().sorted().joinToString(",", "{", "}") { key ->
                JSONObject.quote(key) + ":" + canonical(value.get(key))
            }
            is JSONArray -> (0 until value.length()).joinToString(",", "[", "]") { index -> canonical(value.get(index)) }
            is String -> JSONObject.quote(value)
            JSONObject.NULL -> "null"
            else -> value.toString()
        }
    }
}
