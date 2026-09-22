package com.minashin1120.aiplayground.data

import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import androidx.credentials.ClearCredentialStateRequest
import androidx.credentials.CredentialManager
import androidx.credentials.CustomCredential
import androidx.credentials.GetCredentialRequest
import androidx.credentials.exceptions.GetCredentialException
import com.minashin1120.aiplayground.BuildConfig
import com.google.android.libraries.identity.googleid.GetSignInWithGoogleOption
import com.google.android.libraries.identity.googleid.GoogleIdTokenCredential
import java.security.MessageDigest
import java.security.SecureRandom
import java.time.Instant
import java.util.Base64

/** Gets a Google ID token through Android Credential Manager, without a browser redirect. */
object GoogleAuthClient {
    data class Result(val idToken: String, val nonce: String)

    class AuthException(
        message: String,
        cause: Throwable,
        val diagnosticLog: String,
    ) : Exception(message, cause)

    suspend fun getIdToken(context: Context, serverClientId: String): Result {
        require(serverClientId.isNotBlank()) { "Googleログインを設定できません。" }
        val manager = CredentialManager.create(context)
        return try {
            requestIdToken(manager, context, serverClientId)
        } catch (error: Exception) {
            if (!isAccountReauthFailure(error)) {
                throw diagnosed(context, serverClientId, listOf("request" to error), error)
            }
            // Google Play services can retain a stale active-account state after
            // a failed reauthentication. Clear it and give the user one fresh
            // explicit button flow before surfacing the error.
            val clearError = runCatching {
                manager.clearCredentialState(
                    ClearCredentialStateRequest()
                )
            }.exceptionOrNull()
            try {
                requestIdToken(manager, context, serverClientId)
            } catch (retryError: Exception) {
                val failures = buildList {
                    add("first_request" to error)
                    if (clearError != null) add("clear_state" to clearError)
                    add("retry_request" to retryError)
                }
                throw diagnosed(context, serverClientId, failures, retryError)
            }
        }
    }

    suspend fun clearCredentialState(context: Context) {
        runCatching {
            CredentialManager.create(context).clearCredentialState(
                ClearCredentialStateRequest()
            )
        }
    }

    private suspend fun requestIdToken(
        manager: CredentialManager,
        context: Context,
        serverClientId: String,
    ): Result {
        val requestNonce = nonce()
        // This is an explicit "Sign in with Google" button. GetGoogleIdOption is
        // intended for the bottom-sheet/returning-user flow and can return
        // NoCredentialException before a user has authorized the app. The button
        // option also supports accounts that need re-authentication or are being
        // added through the Google sign-in flow.
        val option = GetSignInWithGoogleOption.Builder(serverClientId)
            .setNonce(requestNonce)
            .build()
        val result = manager.getCredential(
            context = context,
            request = GetCredentialRequest.Builder().addCredentialOption(option).build(),
        )
        val credential = result.credential
        if (credential !is CustomCredential ||
            credential.type != GoogleIdTokenCredential.TYPE_GOOGLE_ID_TOKEN_CREDENTIAL) {
            throw IllegalStateException("Google認証情報を取得できませんでした。")
        }
        return Result(GoogleIdTokenCredential.createFrom(credential.data).idToken, requestNonce)
    }

    private fun isAccountReauthFailure(error: Throwable): Boolean =
        generateSequence(error) { it.cause }.any { cause ->
            val message = cause.message.orEmpty()
            message.contains("Account reauth failed", ignoreCase = true) || message.contains("[16]")
        }

    private fun diagnosed(
        context: Context,
        serverClientId: String,
        failures: List<Pair<String, Throwable>>,
        finalError: Throwable,
    ): AuthException = AuthException(
        message = finalError.message ?: "Googleログインに失敗しました。",
        cause = finalError,
        diagnosticLog = diagnosticLog(context, serverClientId, failures),
    )

    @Suppress("DEPRECATION")
    private fun diagnosticLog(
        context: Context,
        serverClientId: String,
        failures: List<Pair<String, Throwable>>,
    ): String {
        val packageManager = context.packageManager
        val playServices = runCatching {
            packageManager.getPackageInfo("com.google.android.gms", 0)
        }.getOrNull()
        val signatures = runCatching {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                packageManager.getPackageInfo(
                    context.packageName,
                    PackageManager.GET_SIGNING_CERTIFICATES,
                ).signingInfo?.apkContentsSigners?.toList().orEmpty()
            } else {
                packageManager.getPackageInfo(
                    context.packageName,
                    PackageManager.GET_SIGNATURES,
                ).signatures?.toList().orEmpty()
            }
        }.getOrDefault(emptyList())
        fun fingerprint(algorithm: String): String = signatures.joinToString(",") { signature ->
            MessageDigest.getInstance(algorithm).digest(signature.toByteArray())
                .joinToString(":") { byte -> "%02X".format(byte.toInt() and 0xff) }
        }.ifBlank { "unavailable" }
        fun packageVersion(): String = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            playServices?.longVersionCode?.toString()
        } else {
            playServices?.versionCode?.toString()
        } ?: "not-installed"
        val exceptionLines = failures.flatMap { (phase, error) ->
            generateSequence(error) { it.cause }.mapIndexed { index, cause ->
                val credentialType = (cause as? GetCredentialException)?.type
                buildString {
                    append("error.").append(phase).append('.').append(index).append("=")
                    append(cause::class.java.name)
                    if (!credentialType.isNullOrBlank()) append(" type=").append(credentialType)
                    if (!cause.message.isNullOrBlank()) {
                        append(" message=").append(cause.message.orEmpty().replace('\n', ' ').take(500))
                    }
                }
            }.toList()
        }
        return buildList {
            add("AI Playground Google sign-in diagnostics")
            add("timestamp=${Instant.now()}")
            add("app=${BuildConfig.VERSION_NAME} (${BuildConfig.VERSION_CODE})")
            add("package=${context.packageName}")
            add("android=${Build.VERSION.RELEASE} sdk=${Build.VERSION.SDK_INT} securityPatch=${Build.VERSION.SECURITY_PATCH}")
            add("device=${Build.MANUFACTURER} ${Build.MODEL}")
            add("googlePlayServices=${packageVersion()}")
            add("credentialManager=1.6.0 googleId=1.2.1")
            add("serverClientId=$serverClientId")
            add("signingSha1=${fingerprint("SHA-1")}")
            add("signingSha256=${fingerprint("SHA-256")}")
            addAll(exceptionLines)
            add("containsToken=false containsNonce=false")
        }.joinToString("\n")
    }

    private fun nonce(): String {
        val bytes = ByteArray(32)
        SecureRandom().nextBytes(bytes)
        return Base64.getUrlEncoder().withoutPadding().encodeToString(bytes)
    }
}
