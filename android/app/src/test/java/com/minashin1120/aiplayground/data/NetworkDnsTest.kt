package com.minashin1120.aiplayground.data

import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.yield
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

    @Test fun failedCloudflareUsesGoogleForApiAndApkHosts() {
        val queried = mutableListOf<String>()
        val google = Dns { queried.add(it); listOf(address) }
        val dns = RecoveringDns(failed, FailoverDns(listOf(failed, google)), NetworkDns::allowsFallback)
        for (host in listOf("api.openai.com", "generativelanguage.googleapis.com", "github.com", "release-assets.githubusercontent.com")) {
            assertEquals(listOf(address), dns.lookup(host))
        }
        assertEquals(4, queried.size)
    }

    @Test fun successfulFirstSupplementDoesNotContactSecond() {
        val dns = FailoverDns(listOf(Dns { listOf(address) }, Dns { error("Unexpected second query") }))
        assertEquals(listOf(address), dns.lookup("github.com"))
    }

    @Test fun emptyFirstSupplementContinuesToSecond() {
        val dns = FailoverDns(listOf(Dns { emptyList() }, Dns { listOf(address) }))
        assertEquals(listOf(address), dns.lookup("github.com"))
    }

    @Test fun allSupplementsFailWithTheirCausesPreserved() {
        val first = UnknownHostException("fake first error")
        val second = UnknownHostException("fake second error")
        val dns = FailoverDns(listOf(Dns { throw first }, Dns { throw second }))
        assertSame(first, runCatching { dns.lookup("github.com") }.exceptionOrNull())
        assertEquals(listOf(second), first.suppressed.toList())
    }

    @Test fun certificateFailureDoesNotAdvanceToAnotherSupplement() {
        val certificate = SSLHandshakeException("fake certificate error")
        val dns = FailoverDns(listOf(Dns { throw certificate }, Dns { error("Must not bypass TLS failure") }))
        assertSame(certificate, runCatching { dns.lookup("github.com") }.exceptionOrNull())
    }

    @Test fun certificateFailureIsNeverTreatedAsDnsFailure() {
        val certificate = SSLHandshakeException("fake certificate error")
        val dns = RecoveringDns(Dns { throw certificate }, Dns { error("Must not bypass TLS failure") }, { true })
        assertSame(certificate, runCatching { dns.lookup("github.com") }.exceptionOrNull())
        assertNull(dnsFailureMessage(certificate))
    }

    @Test fun backgroundNoticesAreDroppedWithoutDelayingTheNextForegroundNotice() = runBlocking {
        var time = 0L
        val notifications = DnsFallbackNotifications { time }
        var received = 0
        notifications.show() // No visible app: do not emit or consume the cooldown.
        val visible = launch(start = CoroutineStart.UNDISPATCHED) { notifications.notices.collect { received++ } }
        notifications.show()
        yield()
        assertEquals(1, received)
        visible.cancelAndJoin()
        time = 30_000
        notifications.show() // Background polling may still resolve a host; it must stay silent.
        val resumed = launch(start = CoroutineStart.UNDISPATCHED) { notifications.notices.collect { received++ } }
        yield()
        assertEquals(1, received) // Returning never replays background notifications.
        notifications.show()
        yield()
        assertEquals(2, received)
        resumed.cancelAndJoin()
    }

    @Test fun leavingBeforeDeliveryDoesNotQueueAnOldNoticeForTheNextWindow() = runBlocking {
        var time = 0L
        val notifications = DnsFallbackNotifications { time }
        var received = 0
        val leaving = launch(start = CoroutineStart.UNDISPATCHED) { notifications.notices.collect { received++ } }
        notifications.show()
        leaving.cancelAndJoin() // Cancel before the suspended UI collector runs.
        val returned = launch(start = CoroutineStart.UNDISPATCHED) { notifications.notices.collect { received++ } }
        yield()
        assertEquals(0, received)
        time = 30_000
        notifications.show()
        yield()
        assertEquals(1, received)
        returned.cancelAndJoin()
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
