package com.minashin1120.aiplayground.data.direct

import okhttp3.FormBody
import org.json.JSONObject
import java.security.KeyFactory
import java.security.Signature
import java.security.spec.PKCS8EncodedKeySpec
import java.util.Base64

/** Where Gemini runs on Vertex AI (server `gemini_backend = vertex_ai`). */
data class VertexTarget(val project: String, val location: String, val token: suspend () -> String) {
    val host: String get() = if (location == "global") "aiplatform.googleapis.com" else "$location-aiplatform.googleapis.com"

    fun modelUrl(model: String, method: String): String =
        "https://$host/v1/projects/$project/locations/$location/publishers/google/models/$model:$method"
}

/**
 * Access tokens for a Vertex AI service account (the JSON key stored on the device): a signed JWT
 * (RS256) exchanged at Google's OAuth token endpoint, cached until shortly before it expires.
 */
class VertexAuth(private val http: DirectHttp, credentialsJson: String) {
    private val credentials = JSONObject(credentialsJson)
    val projectId: String = credentials.optString("project_id")
    private var cached: String? = null
    private var expiresAt = 0L

    suspend fun token(): String {
        val now = System.currentTimeMillis()
        cached?.takeIf { now < expiresAt - 60_000 }?.let { return it }
        val tokenUri = credentials.optString("token_uri").ifBlank { "https://oauth2.googleapis.com/token" }
        if (!tokenUri.startsWith("https://oauth2.googleapis.com/")) throw DirectApiException(0, "Vertex AIの認証情報のtoken_uriが不正です。")
        val body = FormBody.Builder().add("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer")
            .add("assertion", jwt(tokenUri, now / 1000)).build()
        val reply = http.execute(http.request(tokenUri, emptyMap()).post(body).build()) { response ->
            val text = response.body.string()
            if (!response.isSuccessful) throw DirectHttp.providerError(response.code, text)
            JSONObject(text)
        }
        val token = reply.optString("access_token").ifBlank { throw DirectApiException(0, "Vertex AIのアクセストークンを取得できませんでした。") }
        cached = token
        expiresAt = now + reply.optLong("expires_in", 3600) * 1000
        return token
    }

    internal fun jwt(audience: String, issuedAt: Long): String {
        val encoder = Base64.getUrlEncoder().withoutPadding()
        val header = encoder.encodeToString("""{"alg":"RS256","typ":"JWT"}""".toByteArray())
        val claims = encoder.encodeToString(JSONObject().put("iss", credentials.getString("client_email"))
            .put("scope", "https://www.googleapis.com/auth/cloud-platform").put("aud", audience)
            .put("iat", issuedAt).put("exp", issuedAt + 3600).toString().toByteArray())
        val pem = credentials.getString("private_key").replace(Regex("-----[A-Z ]+-----"), "").replace(Regex("\\s"), "")
        val key = KeyFactory.getInstance("RSA").generatePrivate(PKCS8EncodedKeySpec(Base64.getDecoder().decode(pem)))
        val signature = Signature.getInstance("SHA256withRSA").apply { initSign(key); update("$header.$claims".toByteArray()) }.sign()
        return "$header.$claims.${encoder.encodeToString(signature)}"
    }
}
