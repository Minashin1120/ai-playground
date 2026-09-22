package com.minashin1120.aiplayground.data

import android.content.Context
import androidx.credentials.CredentialManager
import androidx.credentials.CustomCredential
import androidx.credentials.GetCredentialRequest
import com.google.android.libraries.identity.googleid.GetSignInWithGoogleOption
import com.google.android.libraries.identity.googleid.GoogleIdTokenCredential
import java.security.SecureRandom
import java.util.Base64

/** Gets a Google ID token through Android Credential Manager, without a browser redirect. */
object GoogleAuthClient {
    data class Result(val idToken: String, val nonce: String)

    suspend fun getIdToken(context: Context, serverClientId: String): Result {
        require(serverClientId.isNotBlank()) { "Googleログインを設定できません。" }
        val nonce = nonce()
        // This is an explicit "Sign in with Google" button. GetGoogleIdOption is
        // intended for the bottom-sheet/returning-user flow and can return
        // NoCredentialException before a user has authorized the app. The button
        // option also supports accounts that need re-authentication or are being
        // added through the Google sign-in flow.
        val option = GetSignInWithGoogleOption.Builder(serverClientId)
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
