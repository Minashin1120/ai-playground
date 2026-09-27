package com.minashin1120.aiplayground.data

import com.minashin1120.aiplayground.BuildConfig
import okhttp3.HttpUrl
import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import org.json.JSONObject

/**
 * The server the app talks to. The official host is the build default; the login screen can point the
 * app at another (self-hosted) server. One process-wide value so every same-origin check (Bearer token,
 * `/files/` links, relative Markdown links, the browser-dialog title) follows the active server.
 */
object ServerOrigin {
    val DEFAULT: HttpUrl = BuildConfig.BASE_URL.toHttpUrl()

    @Volatile
    var current: HttpUrl = DEFAULT
        private set

    fun set(origin: HttpUrl) { current = origin }

    fun reset() { current = DEFAULT }

    /** True while the app talks to the official server (App Links, Play Integrity, native Google login). */
    val isOfficial: Boolean get() = isOfficial(current)

    fun isOfficial(origin: HttpUrl): Boolean =
        origin.scheme == DEFAULT.scheme && origin.host == DEFAULT.host && origin.port == DEFAULT.port

    /** `https://host[:port]` without the trailing slash, for building absolute links. */
    val base: String get() = current.toString().trimEnd('/')

    val host: String get() = current.host
}

/**
 * Parses what the user typed into the "接続先" field: `example.com`, `example.com:8443` or
 * `https://example.com/`. Only https, no user info, path, query or fragment. Returns the origin
 * with a trailing slash, or null when the input is not an acceptable server address.
 */
fun parseServerOrigin(input: String): HttpUrl? {
    val raw = input.trim()
    if (raw.isEmpty() || raw.length > 253 + 12 || raw.any { it.isWhitespace() }) return null
    val lower = raw.lowercase()
    if (lower.startsWith("http://")) return null
    val withScheme = if (lower.startsWith("https://")) raw else if (raw.contains("://")) return null else "https://$raw"
    val url = withScheme.toHttpUrlOrNull() ?: return null
    if (url.scheme != "https" || url.username.isNotEmpty() || url.password.isNotEmpty()) return null
    if (url.encodedPath != "/" || url.query != null || url.fragment != null) return null
    val host = url.host
    // Hostnames need a dot (no bare intranet names that cannot carry a public certificate);
    // IPv6 literals are rejected for the same reason.
    if (!host.contains('.') || host.contains(':') || host.startsWith('.') || host.endsWith('.')) return null
    return "https://$host${if (url.port != 443) ":${url.port}" else ""}/".toHttpUrl()
}

/** Short label of an origin for the UI: the host, plus the port when it is not 443. */
fun originLabel(origin: HttpUrl): String = origin.host + if (origin.port != 443) ":${origin.port}" else ""

/** `url` plus `return=app` on servers other than the official host, whose browser flows come back through the app scheme. */
fun withAppReturn(url: String): String =
    if (ServerOrigin.isOfficial) url else url + (if (url.contains('?')) "&" else "?") + "return=app"

/** What `/api/mobile/v1/config` says about a server, as the login screen needs it. */
data class ServerInfo(
    val origin: String,
    val name: String,
    val password: Boolean = true,
    val passkey: Boolean = true,
    val googleNative: Boolean = false,
    val googleBrowser: Boolean = true,
    val minashin: Boolean = true,
    val playIntegrity: Boolean = false,
    /** Browser flows can come back to the app: App Link on the official host, the app scheme elsewhere. */
    val browserReturn: Boolean = true,
    val syncApiVersion: Int = 0,
    val secretsExport: Boolean = false,
    val googleServerClientId: String = "",
    val integrityProjectNumber: String = "",
)

/**
 * Parses the config of [origin]; null when it is not an AI Playground server this app can use. Older
 * servers without `auth_methods` keep today's behaviour on the official host; elsewhere only the
 * methods that do not depend on the official host's App Link, OAuth client or Play Integrity are offered.
 */
fun parseServerInfo(origin: HttpUrl, config: JSONObject): ServerInfo? {
    if (config.optInt("api_version") != 1 || config.optString("client_id") != "official-android") return null
    val official = ServerOrigin.isOfficial(origin)
    val methods = config.optJSONObject("auth_methods")
    fun method(key: String, fallback: Boolean) = methods?.optBoolean(key, fallback) ?: fallback
    val modes = config.optJSONArray("auth_callback_modes")
    val appScheme = modes != null && (0 until modes.length()).any { modes.optString(it) == "app_scheme" }
    val clientId = config.optString("google_server_client_id")
    val projectNumber = config.optString("play_integrity_cloud_project_number")
    return ServerInfo(
        origin = origin.toString(),
        name = config.optString("server_name").ifBlank { originLabel(origin) }.take(80),
        password = method("password", true),
        passkey = method("passkey", official),
        googleNative = official && clientId.isNotBlank() && method("google_native", true),
        googleBrowser = method("google_browser", official) && (official || appScheme),
        minashin = method("minashin", true) && (official || appScheme),
        playIntegrity = official && projectNumber.isNotBlank() && config.optBoolean("play_integrity_enabled", true),
        browserReturn = official || appScheme,
        syncApiVersion = config.optInt("sync_api_version", 0),
        secretsExport = config.optBoolean("secrets_export", false),
        googleServerClientId = if (official) clientId else "",
        integrityProjectNumber = if (official) projectNumber else "",
    )
}
