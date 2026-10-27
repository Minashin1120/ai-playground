package com.minashin1120.aiplayground.data

import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.EOFException
import java.io.IOException
import java.net.SocketException
import java.net.SocketTimeoutException
import javax.net.ssl.SSLHandshakeException

class AttachmentReadRetryTest {
    @Test fun transientConnectionAndBodyFailuresRecoverWithoutReopeningViewer() = runBlocking {
        var attempts = 0
        val bytes = retryAttachmentRead {
            when (++attempts) {
                1 -> throw SocketException("connection reset")
                2 -> throw EOFException("truncated body")
                else -> byteArrayOf(1, 2, 3)
            }
        }
        assertArrayEquals(byteArrayOf(1, 2, 3), bytes)
        assertEquals(3, attempts)
    }

    @Test fun persistentTimeoutStopsAfterThreeAttempts() = runBlocking {
        var attempts = 0
        val failure = SocketTimeoutException()
        val result = runCatching { retryAttachmentRead<ByteArray> { attempts++; throw failure } }
        assertSame(failure, result.exceptionOrNull())
        assertEquals(3, attempts)
    }

    @Test fun httpTlsAndByteLimitFailuresAreNotRetried() = runBlocking {
        for (failure in listOf(
            ApiException(401, JSONObject()), ApiException(404, JSONObject()), ApiException(503, JSONObject()),
            SSLHandshakeException("certificate"), IOException("byte budget exceeded"),
        )) {
            var attempts = 0
            val result = runCatching { retryAttachmentRead<ByteArray> { attempts++; throw failure } }
            assertSame(failure, result.exceptionOrNull())
            assertEquals(1, attempts)
        }
    }

    @Test fun closingViewerCancelsPendingRetry() = runBlocking {
        var attempts = 0
        val job = launch(start = CoroutineStart.UNDISPATCHED) {
            retryAttachmentRead<ByteArray> { attempts++; throw SocketException() }
        }
        job.cancelAndJoin()
        assertTrue(job.isCancelled)
        assertEquals(1, attempts)
    }
}
