package com.minashin1120.aiplayground

import android.content.SharedPreferences
import android.os.Build
import com.minashin1120.aiplayground.data.*
import com.minashin1120.aiplayground.data.backend.ChatBackend
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject

/**
 * Sign-in and account security of the Web login screen: password/Google/browser PKCE sign-in,
 * the Play Integrity and Turnstile tickets, TOTP, and passkeys/WebAuthn (login, second factor,
 * registration), plus the security settings actions. State stays in the view model; this class
 * only reads it and reports through `mutable`.
 */
internal class ChatAuth(
    private val api: PlaygroundApi,
    private val backend: () -> ChatBackend,
    private val state: StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val scope: CoroutineScope,
    private val store: TokenStore,
    private val playIntegrity: PlayIntegrityClient,
    private val prefs: SharedPreferences,
    private val token: () -> String,
    private val saveSession: (StoredSession) -> Unit,
    private val notify: (String) -> Unit,
    private val loadSetup: suspend () -> Unit,
    private val loadAccount: suspend () -> Unit,
) {
    /** Ticket for the browser Turnstile check, redeemed by the next auth request. */
    private var integrityTicket: String? = null
    /** Name typed for the passkey being registered (the server falls back to `Passkey N` when blank). */
    private var pendingPasskeyName = "Androidのパスキー"

    /** The login screen switched servers: a ticket for the old one must not be redeemed. */
    fun clearIntegrityTicket() { integrityTicket = null }

    private fun deviceName(): String = "${Build.MANUFACTURER} ${Build.MODEL}".trim().take(80)

    private suspend fun authPost(path: String, body: JSONObject): JSONObject {
        val request = JSONObject(body.toString())
        request.put("integrity_enabled", true)
        val ticket = integrityTicket
        if (ticket != null) {
            request.put("integrity_turnstile_ticket", ticket)
            integrityTicket = null
        } else if (state.value.integrityProjectNumber.isNotBlank()) {
            // Play Integrity is only verified by the official server (fixed package and service account).
            val fields = playIntegrity.requestFields(path, state.value.integrityProjectNumber, request)
            val keys = fields.keys()
            while (keys.hasNext()) {
                val key = keys.next()
                request.put(key, fields.get(key))
            }
        }
        return try {
            api.post(path, request)
        } catch (error: ApiException) {
            if (error.code == "turnstile_required") {
                val url = error.payload.optString("turnstile_url")
                if (url.startsWith("/android/integrity/turnstile?")) {
                    mutable.update { it.copy(authTurnstileUrl = withAppReturn(url)) }
                }
            }
            throw error
        }
    }

    // region: Turnstile and integrity
    /** Opens the browser Turnstile check for this signed-in account (server `mobile_security_turnstile`). */
    fun startSessionTurnstile() { scope.launch {
        try {
            val reply = backend().post("/api/mobile/v1/security/turnstile", JSONObject(), token())
            val url = reply.optString("turnstile_url")
            if (url.startsWith("/android/integrity/turnstile?")) mutable.update { it.copy(sessionTurnstileUrl = withAppReturn(url)) }
            else notify("安全性の確認を完了しました。もう一度送信してください。")
        } catch (e: CancellationException) { throw e }
        catch (e: Exception) { notify("安全性の確認を完了できませんでした。しばらく待ってから再送信してください。") }
    } }

    /** The browser check came back with a ticket that only this account can redeem. */
    fun completeSessionTurnstile(ticket: String) {
        if (ticket.length !in 20..128) return
        scope.launch {
            val ok = runCatching {
                backend().post("/api/mobile/v1/security/turnstile/complete", JSONObject().put("ticket", ticket), token())
            }.isSuccess
            mutable.update { it.copy(sessionTurnstileUrl = null) }
            notify(if (ok) "安全性の確認を完了しました。もう一度送信してください。"
                else "安全性の確認を完了できませんでした。しばらく待ってから再送信してください。")
        }
    }

    fun dismissSessionTurnstile() { mutable.update { it.copy(sessionTurnstileUrl = null) } }

    fun integrityTurnstileComplete(ticket: String) {
        if (ticket.length !in 20..128) return
        integrityTicket = ticket
        mutable.update { it.copy(authTurnstileUrl = null, authError = "安全性を確認しました。認証をもう一度実行してください。") }
    }

    private suspend fun acceptAuthResponse(reply: JSONObject) {
        val accessToken = reply.optString("access_token")
        require(accessToken.isNotBlank()) { "認証トークンを取得できませんでした。" }
        val session = StoredSession(accessToken, System.currentTimeMillis() + reply.optLong("expires_in", 2_592_000L) * 1000L,
            ServerOrigin.current.toString())
        saveSession(session)
        withContext(Dispatchers.IO) { store.save(session) }
        mutable.update { it.copy(
            authBusy = false, authError = null, authTwoFactorTransaction = null,
            googleAuthDiagnostics = null,
            auth2faMethod = "totp", credentialRequest = null,
        ) }
        if (reply.optBoolean("setup_required")) loadSetup() else loadAccount()
    }

    private suspend fun processAuthResponse(reply: JSONObject) {
        if (reply.optString("status") == "2fa_required") {
            mutable.update { it.copy(
                authBusy = false,
                authTwoFactorTransaction = reply.getString("transaction_id"),
                auth2faMethod = reply.optString("default_method", "totp").ifBlank { "totp" },
                authError = null,
            ) }
        } else {
            acceptAuthResponse(reply)
        }
    }

    // region: Authentication
    fun login(username: String, password: String) {
        authenticate("/api/mobile/v1/auth/login", username, password)
    }

    fun signup(username: String, password: String) {
        authenticate("/api/mobile/v1/auth/signup", username, password)
    }

    private fun authenticate(path: String, username: String, password: String) {
        if (state.value.authBusy) return
        scope.launch {
            mutable.update { it.copy(
                authBusy = true,
                authError = null,
                googleAuthDiagnostics = null,
                authTwoFactorTransaction = null,
            ) }
            try {
                val reply = authPost(path, JSONObject()
                    .put("username", username.trim())
                    .put("password", password)
                    .put("device_name", deviceName()))
                processAuthResponse(reply)
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                mutable.update { it.copy(authBusy = false, authError = e.message ?: "認証に失敗しました。") }
            }
        }
    }

    /** Starts Android Credential Manager's native Google account picker. */
    fun beginGoogleLogin() {
        if (state.value.authBusy) return
        if (state.value.googleServerClientId.isBlank()) {
            mutable.update { it.copy(authError = "Googleログインを設定できません。時間をおいて再試行してください。") }
            return
        }
        mutable.update { it.copy(
            authBusy = true,
            authError = null,
            googleAuthDiagnostics = null,
            authTwoFactorTransaction = null,
            googleLoginRequest = it.googleLoginRequest + 1L,
        ) }
    }

    fun completeGoogleLogin(credential: GoogleAuthClient.Result) {
        if (!state.value.authBusy) return
        scope.launch {
            try {
                processAuthResponse(authPost("/api/mobile/v1/auth/google", JSONObject()
                    .put("id_token", credential.idToken)
                    .put("nonce", credential.nonce)
                    .put("device_name", deviceName())))
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                mutable.update { it.copy(authBusy = false, authError = e.message ?: "Googleログインに失敗しました。") }
            }
        }
    }

    fun cancelGoogleLogin(
        message: String = "Googleログインをキャンセルしました。",
        diagnostics: String? = null,
    ) {
        if (state.value.authBusy) mutable.update {
            it.copy(authBusy = false, authError = message, googleAuthDiagnostics = diagnostics)
        }
    }

    fun verifyTotp(code: String) {
        val transaction = state.value.authTwoFactorTransaction ?: return
        if (state.value.authBusy) return
        scope.launch {
            mutable.update { it.copy(authBusy = true, authError = null) }
            try {
                acceptAuthResponse(authPost("/api/mobile/v1/auth/totp", JSONObject()
                    .put("transaction_id", transaction)
                    .put("code", code)
                    .put("device_name", deviceName())))
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                mutable.update { it.copy(authBusy = false, authError = e.message ?: "2段階認証に失敗しました。") }
            }
        }
    }

    /** Starts a browser login whose returned code only this app can redeem (PKCE). */
    fun browserLoginPath(provider: String): String {
        require(provider == "google" || provider == "minashin") { "Unsupported browser login" }
        val verifier = BrowserLoginPkce.newVerifier()
        prefs.edit().putString(PREF_BROWSER_LOGIN_VERIFIER, verifier)
            .putLong(PREF_BROWSER_LOGIN_STARTED_AT, System.currentTimeMillis()).apply()
        return withAppReturn("/android/auth/$provider/start?code_challenge_method=S256&code_challenge=" +
            BrowserLoginPkce.challenge(verifier))
    }

    /** Exchanges the one-time code returned to the verified HTTPS App Link. */
    fun exchangeNativeCode(code: String) {
        if (state.value.authBusy) return
        val verifier = prefs.getString(PREF_BROWSER_LOGIN_VERIFIER, null)
        val elapsed = System.currentTimeMillis() - prefs.getLong(PREF_BROWSER_LOGIN_STARTED_AT, 0L)
        prefs.edit().remove(PREF_BROWSER_LOGIN_VERIFIER).remove(PREF_BROWSER_LOGIN_STARTED_AT).apply()
        if (verifier == null || elapsed !in 0..BROWSER_LOGIN_TTL_MS) {
            // A link that this app did not start may carry someone else's account.
            notify("このアプリで開始していないログインのため、処理しませんでした。")
            return
        }
        scope.launch {
            mutable.update { it.copy(authBusy = true, authError = null) }
            try {
                val reply = authPost("/api/mobile/v1/auth/exchange", JSONObject()
                    .put("code", code)
                    .put("code_verifier", verifier))
                processAuthResponse(reply)
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                mutable.update { it.copy(authBusy = false, authError = e.message ?: "外部ログインに失敗しました。") }
            }
        }
    }

    // region: Passkeys, WebAuthn, and security
    /** Starts a passkey sign-in and hands the WebAuthn options to the UI. */
    fun beginPasskeyLogin(username: String) {
        if (state.value.authBusy) return
        scope.launch {
            mutable.update { it.copy(authBusy = true, authError = null, credentialRequest = null) }
            try {
                val reply = authPost("/api/mobile/v1/auth/passkey/options", JSONObject()
                    .put("username", username.trim())
                    .put("device_name", deviceName()))
                mutable.update { it.copy(
                    authBusy = false,
                    credentialRequest = CredentialRequest(
                        "login", reply.getString("transaction_id"), reply.getJSONObject("public_key").toString()),
                ) }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                mutable.update { it.copy(authBusy = false, authError = e.message ?: "パスキーを開始できませんでした。") }
            }
        }
    }

    /** Starts a WebAuthn second-factor ceremony for the pending 2FA transaction. */
    fun beginWebauthnTwoFactor() {
        val transaction = state.value.authTwoFactorTransaction ?: return
        if (state.value.authBusy) return
        scope.launch {
            mutable.update { it.copy(authBusy = true, authError = null, credentialRequest = null) }
            try {
                val reply = authPost("/api/mobile/v1/auth/2fa/webauthn/options",
                    JSONObject().put("transaction_id", transaction))
                mutable.update { it.copy(
                    authBusy = false,
                    credentialRequest = CredentialRequest(
                        "2fa", transaction, reply.getJSONObject("public_key").toString()),
                ) }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                mutable.update { it.copy(authBusy = false, authError = e.message ?: "パスキーを開始できませんでした。") }
            }
        }
    }

    /** Starts passkey registration from the security settings. */
    fun beginPasskeyRegistration(name: String = "") {
        if (state.value.securityBusy) return
        // Web sends the typed name; the server falls back to `Passkey N` when it is blank.
        pendingPasskeyName = name.trim().take(80)
        scope.launch {
            mutable.update { it.copy(securityBusy = true, securityError = null, credentialRequest = null) }
            try {
                val reply = backend().post("/api/mobile/v1/security/passkeys/options", JSONObject(), token())
                mutable.update { it.copy(
                    securityBusy = false,
                    credentialRequest = CredentialRequest(
                        "register", "", reply.getJSONObject("public_key").toString()),
                ) }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                mutable.update { it.copy(securityBusy = false, securityError = e.message ?: "パスキー登録を開始できませんでした。") }
            }
        }
    }

    /** Completes whichever WebAuthn ceremony the UI just ran through Credential Manager. */
    fun submitCredential(responseJson: String) {
        val request = state.value.credentialRequest ?: return
        scope.launch {
            try {
                val credential = JSONObject(responseJson)
                when (request.kind) {
                    "login" -> acceptAuthResponse(authPost("/api/mobile/v1/auth/passkey/verify", JSONObject()
                        .put("transaction_id", request.transactionId)
                        .put("credential", credential)))
                    "2fa" -> acceptAuthResponse(authPost("/api/mobile/v1/auth/2fa/webauthn/verify", JSONObject()
                        .put("transaction_id", request.transactionId)
                        .put("credential", credential)))
                    "register" -> {
                        val reply = backend().post("/api/mobile/v1/security/passkeys/verify", JSONObject()
                            .put("credential", credential)
                            .put("name", pendingPasskeyName), token())
                        mutable.update { it.copy(
                            securityBusy = false, securityError = null, credentialRequest = null,
                            security = parseSecurityInfo(reply),
                        ) }
                    }
                }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                mutable.update { it.copy(
                    authBusy = false, securityBusy = false, credentialRequest = null,
                    authError = e.message ?: "パスキー認証に失敗しました。",
                    securityError = e.message ?: "パスキー認証に失敗しました。",
                ) }
            }
        }
    }

    /** Clears a cancelled or failed Credential Manager ceremony. */
    fun cancelCredentialRequest(message: String? = null) {
        mutable.update { it.copy(
            credentialRequest = null, authBusy = false, securityBusy = false,
            authError = message ?: it.authError, securityError = message ?: it.securityError,
        ) }
    }

    fun loadSecurity() { scope.launch {
        try {
            val reply = backend().get("/api/mobile/v1/security", token())
            mutable.update { it.copy(security = parseSecurityInfo(reply), securityError = null) }
        } catch (e: CancellationException) { throw e }
        catch (e: Exception) { mutable.update { it.copy(securityError = e.message ?: "セキュリティ設定を取得できませんでした。") } }
    } }

    fun startTotpSetup() { scope.launch {
        mutable.update { it.copy(securityBusy = true, securityError = null) }
        try {
            val reply = backend().post("/api/mobile/v1/security/totp/setup", JSONObject(), token())
            mutable.update { it.copy(
                securityBusy = false,
                securityTotpSecret = reply.getString("secret"),
                securityTotpUri = reply.optString("otpauth_uri"),
                securityTotpQr = reply.optString("qr_image").ifBlank { null },
            ) }
        } catch (e: CancellationException) { throw e }
        catch (e: Exception) { mutable.update { it.copy(securityBusy = false, securityError = e.message ?: "TOTPを開始できませんでした。") } }
    } }

    fun cancelTotpSetup() { mutable.update { it.copy(securityTotpSecret = null, securityTotpUri = null) } }

    fun enableTotp(code: String) = securityAction {
        backend().post("/api/mobile/v1/security/totp/enable", JSONObject().put("code", code), token())
    }

    fun disableTotp(code: String) = securityAction {
        backend().post("/api/mobile/v1/security/totp/disable", JSONObject().put("code", code), token())
    }

    fun removePasskey(id: String) = securityAction {
        backend().post("/api/mobile/v1/security/passkeys/remove", JSONObject().put("id", id), token())
    }

    fun saveSecurityPreferences(default2fa: String, passkeyOnly: Boolean, skipGoogle: Boolean) = securityAction {
        backend().post("/api/mobile/v1/security/preferences", JSONObject()
            .put("default_2fa_method", default2fa)
            .put("passkey_only_login", passkeyOnly)
            .put("skip_2fa_on_google_login", skipGoogle), token())
    }

    private fun securityAction(block: suspend () -> JSONObject) {
        if (state.value.securityBusy) return
        scope.launch {
            mutable.update { it.copy(securityBusy = true, securityError = null) }
            try {
                val reply = block()
                mutable.update { it.copy(
                    securityBusy = false, security = parseSecurityInfo(reply),
                    securityTotpSecret = null, securityTotpUri = null,
                ) }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) { mutable.update { it.copy(securityBusy = false, securityError = e.message ?: "セキュリティ設定を保存できませんでした。") } }
        }
    }

    companion object {
        const val PREF_BROWSER_LOGIN_VERIFIER = "browser_login_pkce_verifier"
        const val PREF_BROWSER_LOGIN_STARTED_AT = "browser_login_started_at"
        const val BROWSER_LOGIN_TTL_MS = 10L * 60 * 1000
    }
}
