package com.minashin1120.aiplayground.data

import com.minashin1120.aiplayground.BuildConfig
import com.minashin1120.aiplayground.data.direct.DirectHttp
import okhttp3.CookieJar
import okhttp3.Dns
import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.OkHttpClient
import okhttp3.dnsoverhttps.DnsOverHttps
import java.net.InetAddress
import java.net.UnknownHostException
import java.util.Locale
import java.util.UUID
import java.util.concurrent.TimeUnit

/** Resolve before connecting, without replaying a model request or weakening TLS verification. */
internal class RecoveringDns(
    private val primary: Dns,
    private val fallback: Dns,
    private val allowed: (String) -> Boolean,
    private val onFallback: () -> Unit = {},
) : Dns {
    override fun lookup(hostname: String): List<InetAddress> {
        val host = hostname.lowercase(Locale.ROOT).removeSuffix(".")
        val original = try {
            val addresses = primary.lookup(hostname)
            if (addresses.isNotEmpty()) return addresses
            UnknownHostException("Empty DNS response")
        } catch (error: UnknownHostException) {
            error
        }
        if (!allowed(host)) throw original
        val op = UUID.randomUUID().toString()
        val started = System.nanoTime()
        ActivityLog.log("dns.fallback", "op" to op, "phase" to "start")
        runCatching { onFallback() }
        try {
            val addresses = fallback.lookup(host)
            if (addresses.isEmpty()) throw UnknownHostException("Empty DNS response")
            ActivityLog.log("dns.fallback", "op" to op, "phase" to "done", "count" to addresses.size,
                "ms" to (System.nanoTime() - started) / 1_000_000)
            return addresses
        } catch (error: UnknownHostException) {
            ActivityLog.log("dns.fallback", "op" to op, "phase" to "error",
                "ms" to (System.nanoTime() - started) / 1_000_000)
            if (error !== original) original.addSuppressed(error)
            throw original
        }
    }
}

/** Coalesces parallel lookups and repeated failures into one visible notice per 30 seconds. */
internal class DnsFallbackNotice(
    private val emit: () -> Unit,
    private val now: () -> Long = { System.nanoTime() / 1_000_000 },
) {
    private var last: Long? = null
    @Synchronized fun show() {
        val time = now()
        if (last?.let { time - it < 30_000 } == true) return
        last = time
        emit()
    }
}

object NetworkDns : Dns {
    const val NOTICE = "端末のDNSに失敗したため、DNS補助（Cloudflare）を使用しています。"
    @Volatile var onFallbackNotice: () -> Unit = {}
    private val notice = DnsFallbackNotice({ onFallbackNotice() })
    private val resolver by lazy {
        // A dedicated client has no credentials or cookies and cannot recursively use this resolver.
        // Bootstrap IPs keep the resolver reachable even when Android cannot resolve its hostname.
        val transport = OkHttpClient.Builder().cookieJar(CookieJar.NO_COOKIES)
            .followRedirects(false).followSslRedirects(false)
            .connectTimeout(3, TimeUnit.SECONDS).readTimeout(3, TimeUnit.SECONDS)
            .callTimeout(5, TimeUnit.SECONDS).build()
        val encrypted = DnsOverHttps.Builder().client(transport)
            .url("https://cloudflare-dns.com/dns-query".toHttpUrl()).post(true)
            .bootstrapDnsHosts(
                InetAddress.getByAddress(byteArrayOf(1, 1, 1, 1)),
                InetAddress.getByAddress(byteArrayOf(1, 0, 0, 1)),
                InetAddress.getByName("2606:4700:4700::1111"),
                InetAddress.getByName("2606:4700:4700::1001"),
            ).build()
        RecoveringDns(Dns.SYSTEM, encrypted, ::allowsFallback, notice::show)
    }
    override fun lookup(hostname: String): List<InetAddress> = resolver.lookup(hostname)

    internal fun allowsFallback(host: String): Boolean =
        host == BuildConfig.BASE_URL.toHttpUrl().host || host in DirectHttp.PROVIDER_HOSTS ||
            Regex("^[a-z0-9-]+-aiplatform\\.googleapis\\.com$").matches(host) ||
            host in setOf("github.com", "api.github.com", "githubusercontent.com") ||
            host.endsWith(".githubusercontent.com")
}

internal fun dnsFailureMessage(error: Throwable): String? {
    val seen = HashSet<Throwable>()
    var cause: Throwable? = error
    while (cause != null && seen.add(cause)) {
        if (cause is UnknownHostException)
            return "接続先の名前を解決できませんでした。通信環境やVPN・プライベートDNSの設定を確認して再試行してください。"
        cause = cause.cause
    }
    return null
}
