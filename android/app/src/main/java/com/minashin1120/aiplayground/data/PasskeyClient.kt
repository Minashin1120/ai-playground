package com.minashin1120.aiplayground.data

import android.content.Context
import androidx.credentials.CreatePublicKeyCredentialRequest
import androidx.credentials.CreatePublicKeyCredentialResponse
import androidx.credentials.CredentialManager
import androidx.credentials.GetCredentialRequest
import androidx.credentials.GetPublicKeyCredentialOption
import androidx.credentials.PublicKeyCredential

/**
 * Thin wrapper over Android Credential Manager for passkey create/get ceremonies.
 *
 * The server returns standard WebAuthn JSON from the same options generator used
 * by the Web client, so the request and response payloads pass through unchanged.
 * The [Context] must be an Activity so Credential Manager can show its sheet.
 */
object PasskeyClient {
    suspend fun create(context: Context, publicKeyJson: String): String {
        val manager = CredentialManager.create(context)
        val response = manager.createCredential(context, CreatePublicKeyCredentialRequest(publicKeyJson))
        return (response as CreatePublicKeyCredentialResponse).registrationResponseJson
    }

    suspend fun get(context: Context, publicKeyJson: String): String {
        val manager = CredentialManager.create(context)
        val request = GetCredentialRequest.Builder()
            .addCredentialOption(GetPublicKeyCredentialOption(publicKeyJson))
            .build()
        val response = manager.getCredential(context, request)
        return (response.credential as PublicKeyCredential).authenticationResponseJson
    }
}
