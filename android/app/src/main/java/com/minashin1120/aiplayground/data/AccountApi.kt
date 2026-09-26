package com.minashin1120.aiplayground.data

import org.json.JSONArray
import org.json.JSONObject
import java.io.OutputStream

/**
 * Settings-modal operations that the Web performs through `/api/settings`, `/api/account/*`,
 * `/api/sessions/*` and `/api/easy_login` (server `routes_mobile_account.py`).
 * Calls that answer `reauth_required` are retried after [ReauthRequired] is handled by the UI.
 */
class AccountApi internal constructor(private val api: PlaygroundApi, private val token: () -> String) {
    suspend fun reauthOptions(): ReauthOptions {
        val reply = api.post("/api/mobile/v1/reauth/options", JSONObject(), token())
        val methods = reply.optJSONArray("methods")?.let { a -> (0 until a.length()).map(a::getString) }.orEmpty()
        return ReauthOptions(methods, reply.optJSONObject("public_key")?.toString(), reply.optBoolean("reauthenticated"),
            reply.optBoolean("sign_in_again"))
    }

    suspend fun reauthPassword(password: String) =
        api.post("/api/mobile/v1/reauth", JSONObject().put("method", "password").put("password", password), token())

    suspend fun reauthTotp(code: String) =
        api.post("/api/mobile/v1/reauth", JSONObject().put("method", "totp").put("code", code), token())

    suspend fun reauthPasskey(credentialJson: String) =
        api.post("/api/mobile/v1/reauth", JSONObject().put("method", "passkey").put("credential", JSONObject(credentialJson)), token())

    suspend fun changeCredentials(username: String, password: String): JSONObject {
        val body = JSONObject()
        if (username.isNotBlank()) body.put("new_username", username)
        if (password.isNotEmpty()) body.put("new_password", password)
        return api.post("/api/mobile/v1/account/credentials", body, token())
    }

    suspend fun deleteAccount() = api.post("/api/mobile/v1/account/delete", JSONObject(), token())

    suspend fun easyLogin(minutes: Int): EasyLogin {
        val reply = api.post("/api/mobile/v1/account/easy-login", JSONObject().put("minutes", minutes), token())
        return EasyLogin(reply.optString("temp_password"), reply.optString("expires_at"))
    }

    suspend fun cancelEasyLogin(): Boolean =
        api.post("/api/mobile/v1/account/easy-login", JSONObject().put("cancel", true), token()).optBoolean("cancelled")

    suspend fun sessions(): List<LoginSession> {
        val rows = api.get("/api/mobile/v1/sessions", token()).optJSONArray("sessions") ?: JSONArray()
        return (0 until rows.length()).map { i ->
            val row = rows.getJSONObject(i)
            LoginSession(
                id = row.optInt("id"),
                createdAt = row.nullableString("created_at"),
                lastSeenAt = row.nullableString("last_seen_at"),
                ipAddress = row.nullableString("ip_address"),
                userAgent = row.nullableString("user_agent"),
                current = row.optBoolean("is_current"),
                revoked = row.optBoolean("is_revoked"),
            )
        }
    }

    suspend fun revokeSession(id: Int): Boolean =
        api.post("/api/mobile/v1/sessions/revoke", JSONObject().put("id", id), token()).optBoolean("logged_out")

    suspend fun revokeOtherSessions() = api.post("/api/mobile/v1/sessions/revoke_others", JSONObject(), token())
    suspend fun revokeAllSessions() = api.post("/api/mobile/v1/sessions/revoke_all", JSONObject(), token())
    suspend fun disable2fa() = api.post("/api/mobile/v1/security/2fa/disable", JSONObject(), token())

    /** Returns the Web message (`暗号化設定の変更処理を開始しました。…`) or null when unchanged. */
    suspend fun setE2ee(enabled: Boolean): String? =
        api.post("/api/mobile/v1/security/e2ee", JSONObject().put("enabled", enabled), token()).nullableString("message").ifBlank { null }

    suspend fun encryptionScan(threadId: String?): JSONObject =
        api.get("/api/encryption_scan" + if (threadId.isNullOrBlank()) "" else "?thread_id=${java.net.URLEncoder.encode(threadId, "UTF-8")}", token())

    suspend fun startExport(jobId: String): JSONObject = api.post("/api/account/export", JSONObject().put("job_id", jobId), token())

    suspend fun importStart(size: Long): JSONObject = api.accountImportStart(size, token())
    suspend fun importChunk(uploadId: String, index: Int, chunk: ByteArray): JSONObject = api.accountImportChunk(uploadId, index, chunk, token())
    suspend fun importComplete(uploadId: String): JSONObject = api.accountImportComplete(uploadId, token())
    suspend fun importCancel(uploadId: String): JSONObject = api.accountImportCancel(uploadId, token())

    /** Web `POST /api/account/import` with every option of the settings Data tab. */
    suspend fun importArchive(
        uploadId: String, categories: List<String>, jobId: String, selectedFiles: String,
        restoreInplace: Boolean, confirmSettings: Boolean,
    ): JSONObject = api.postLong("/api/account/import", JSONObject()
        .put("upload_id", uploadId).put("categories", categories.joinToString(",")).put("job_id", jobId)
        .put("selected_files", selectedFiles).put("restore_inplace", restoreInplace).put("confirm_settings", confirmSettings), token())
    suspend fun latestExport(): JSONObject = api.get("/api/account/export/latest", token())
    suspend fun transferStatus(jobId: String): JSONObject = api.get("/api/account/transfer/$jobId", token())
    suspend fun cancelTransfer(jobId: String): JSONObject = api.post("/api/account/transfer/$jobId/cancel", JSONObject(), token())

    /** Streams the export ZIP into [output]; [onProgress] receives the bytes written so far. */
    suspend fun downloadExport(jobId: String, output: OutputStream, onProgress: (Long) -> Unit) =
        api.downloadTo("/api/account/export/$jobId/download", token(), output, onProgress)

    suspend fun unlinkGoogle() = api.post("/api/account/unlink_google", JSONObject(), token())
    suspend fun unlinkMinashin() = api.post("/api/account/unlink_minashin", JSONObject(), token())

    /** MCP tab: connection test, tool list, deletion and unauthenticated custom servers (Web `mcp_service`). */
    suspend fun mcpTest(serverId: Int): JSONObject = api.post("/api/mcp/servers/$serverId/test", JSONObject(), token())
    suspend fun mcpTools(serverId: Int): List<McpTool> {
        val rows = api.get("/api/mcp/servers/$serverId/tools", token()).optJSONArray("tools") ?: JSONArray()
        return (0 until rows.length()).map { i ->
            val row = rows.getJSONObject(i)
            McpTool(row.optString("name"), row.nullableString("description"), row.optBoolean("read_only"))
        }
    }
    suspend fun mcpDelete(serverId: Int) = api.delete("/api/mcp/servers/$serverId", token())
    suspend fun mcpAdd(name: String, url: String, description: String): JSONObject = api.post("/api/mcp/servers", JSONObject()
        .put("name", name).put("url", url).put("auth_type", "none").put("description", description), token())

    suspend fun dedupePreview(): JSONObject = api.post("/api/account/dedupe/preview", JSONObject(), token())
    suspend fun dedupeExecute(): JSONObject = api.post("/api/account/dedupe/execute", JSONObject(), token())
}

data class ReauthOptions(val methods: List<String>, val passkeyJson: String?, val reauthenticated: Boolean, val signInAgain: Boolean)
data class McpTool(val name: String, val description: String, val readOnly: Boolean)
data class EasyLogin(val password: String, val expiresAt: String)
data class LoginSession(
    val id: Int, val createdAt: String, val lastSeenAt: String, val ipAddress: String, val userAgent: String,
    val current: Boolean, val revoked: Boolean,
)

/** True when the server asks for a re-authentication before this operation. */
fun Throwable.needsReauth(): Boolean = this is ApiException && code == "reauth_required"
