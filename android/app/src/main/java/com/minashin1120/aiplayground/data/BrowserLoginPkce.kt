package com.minashin1120.aiplayground.data

import java.security.MessageDigest
import java.security.SecureRandom
import java.util.Base64

/**
 * RFC 7636 (S256) binding for browser logins that return through the App Link.
 * Only the app that started the login holds the verifier needed to redeem the code.
 */
object BrowserLoginPkce {
    private val encoder = Base64.getUrlEncoder().withoutPadding()

    fun newVerifier(random: SecureRandom = SecureRandom()): String =
        encoder.encodeToString(ByteArray(32).also { random.nextBytes(it) })

    fun challenge(verifier: String): String =
        encoder.encodeToString(MessageDigest.getInstance("SHA-256").digest(verifier.toByteArray(Charsets.US_ASCII)))
}
