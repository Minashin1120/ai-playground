package com.minashin1120.aiplayground.data

import okhttp3.Dns
import org.junit.Assert.*
import org.junit.Test
import java.net.InetAddress
import java.net.UnknownHostException
import javax.net.ssl.SSLHandshakeException

class NetworkDnsTest {
    private val address = InetAddress.getByAddress(byteArrayOf(127, 0, 0, 1))
    private val failed = Dns { throw UnknownHostException("fake private hostname") }

    @Test fun workingSystemDnsNeverUsesSupplementOrNotifies() {
        val dns = RecoveringDns(Dns { listOf(address) }, Dns { error("Unexpected fallback") }, { true }, { error("Unexpected notice") })
        assertEquals(listOf(address), dns.lookup("api.openai.com"))
    }

    @Test fun failedDnsUsesSupplementBeforeRequestAndNotifies() {
        var notices = 0
        var queried = ""
        val dns = RecoveringDns(failed, Dns { queried = it; listOf(address) }, { true }, { notices++ })
        assertEquals(listOf(address), dns.lookup("API.OPENAI.COM"))
        assertEquals("api.openai.com", queried)
        assertEquals(1, notices)
    }

    @Test fun notificationFailureCannotBreakDnsRecovery() {
        val dns = RecoveringDns(failed, Dns { listOf(address) }, { true }, { error("Notice unavailable") })
        assertEquals(listOf(address), dns.lookup("github.com"))
    }

    @Test fun emptySystemAnswerAlsoUsesSupplement() {
        assertEquals(listOf(address), RecoveringDns(Dns { emptyList() }, Dns { listOf(address) }, { true }).lookup("github.com"))
    }

    @Test fun privateAndCustomNamesAreNotSentToPublicResolver() {
        val dns = RecoveringDns(failed, Dns { error("Must not leak private host") }, NetworkDns::allowsFallback)
        for (host in listOf("localhost", "192.168.1.1", "private.example.com", "github.com.attacker.example", "api.openai.com.attacker.example")) {
            assertNotNull(runCatching { dns.lookup(host) }.exceptionOrNull() as? UnknownHostException)
        }
        for (host in listOf("github.com", "api.github.com", "release-assets.githubusercontent.com", "api.openai.com", "us-central1-aiplatform.googleapis.com")) {
            assertTrue(NetworkDns.allowsFallback(host))
        }
    }

    @Test fun bothResolversFailWithoutReturningAnAddressOrLooping() {
        var calls = 0
        val dns = RecoveringDns(failed, Dns { calls++; throw UnknownHostException("fake fallback error") }, { true })
        val error = runCatching { dns.lookup("github.com") }.exceptionOrNull()!!
        assertTrue(error is UnknownHostException)
        assertEquals(1, calls)
        assertEquals(1, error.suppressed.size)
        assertFalse(dnsFailureMessage(error)!!.contains("fake private hostname"))
    }

    @Test fun certificateFailureIsNeverTreatedAsDnsFailure() {
        val certificate = SSLHandshakeException("fake certificate error")
        val dns = RecoveringDns(Dns { throw certificate }, Dns { error("Must not bypass TLS failure") }, { true })
        assertSame(certificate, runCatching { dns.lookup("github.com") }.exceptionOrNull())
        assertNull(dnsFailureMessage(certificate))
    }

    @Test fun repeatedOrParallelLookupsDoNotFloodNotices() {
        var time = 0L
        var notices = 0
        val notice = DnsFallbackNotice({ notices++ }, { time })
        notice.show()
        repeat(20) { notice.show() }
        assertEquals(1, notices)
        time = 29_999
        notice.show()
        assertEquals(1, notices)
        time = 30_000
        notice.show()
        assertEquals(2, notices)
    }
}
