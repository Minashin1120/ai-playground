package com.minashin1120.aiplayground.data

import android.content.Context
import androidx.credentials.CredentialManager
import androidx.credentials.CustomCredential
import androidx.credentials.GetCredentialRequest
import com.google.android.libraries.identity.googleid.GetGoogleIdOption
import com.google.android.libraries.identity.googleid.GoogleIdTokenCredential
import java.security.SecureRandom
import java.util.Base64

/** Gets a Google ID token through Android Credential Manager, without a browser redirect. */
object GoogleAuthClient {
    data class Result(val idToken: String, val nonce: String)

    suspend fun getIdToken(context: Context, serverClientId: String): Result {
        require(serverClientId.isNotBlank()) { "Googleログインを設定できません。" }
        val nonce = nonce()
        val option = GetGoogleIdOption.Builder()
            .setFilterByAuthorizedAccounts(false)
            .setServerClientId(serverClientId)
            .setAutoSelectEnabled(false)
            .setNonce(nonce)
            .build()
        val result = CredentialManager.create(context).getCredential(
            context = context,
            request = GetCredentialRequest.Builder().addCredentialOption(option).build(),
        )
        val credential = result.credential
        if (credential !is CustomCredential ||
            credential.type != GoogleIdTokenCredential.TYPE_GOOGLE_ID_TOKEN_CREDENTIAL) {
            throw IllegalStateException("Google認証情報を取得できませんでした。")
        }
        return Result(GoogleIdTokenCredential.createFrom(credential.data).idToken, nonce)
    }

    private fun nonce(): String {
        val bytes = ByteArray(32)
        SecureRandom().nextBytes(bytes)
        return Base64.getUrlEncoder().withoutPadding().encodeToString(bytes)
    }
}
