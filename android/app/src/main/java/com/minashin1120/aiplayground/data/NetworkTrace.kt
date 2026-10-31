package com.minashin1120.aiplayground.data

import okhttp3.*
import java.io.IOException
import java.io.InterruptedIOException
import java.net.*
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import javax.net.ssl.SSLException

internal data class NetworkTraceParent(val op: String)

/** Content-free transport stages. Numeric codes are documented in DIAGNOSTICS.md. */
internal class NetworkTrace(
    private val delegate: EventListener = EventListener.NONE,
    private val context: Int,
    private val parent: (Call) -> String? = { it.request().tag(NetworkTraceParent::class.java)?.op },
    private val emit: (Map<String, Any?>) -> Unit = { fields -> ActivityLog.log("http", *fields.toList().toTypedArray()) },
) : EventListener() {
    private val op = UUID.randomUUID().toString()
    private val started = System.nanoTime()
    @Volatile private var stage = 100
    private val attempts = AtomicInteger()
    private val counted = AtomicBoolean(false)
    private fun record(call: Call, code: Int, phase: String, vararg extra: Pair<String, Any?>) {
        runCatching {
            emit(mapOf("op" to op, "parent_op" to parent(call), "from" to context,
                "reason_code" to code, "phase" to phase, "ms" to (System.nanoTime() - started) / 1_000_000,
                "index" to attempts.get(), "pending_count" to active.get()) + extra.toMap())
        }
    }
    private fun start(call: Call, code: Int) { stage = code; record(call, code, "start") }
    private fun finish() { if (counted.compareAndSet(true, false)) active.decrementAndGet() }
    override fun callStart(call: Call) {
        if (counted.compareAndSet(false, true)) active.incrementAndGet()
        start(call, 100); delegate.callStart(call)
    }
    override fun dnsStart(call: Call, domainName: String) { start(call, 101); delegate.dnsStart(call, domainName) }
    override fun dnsEnd(call: Call, domainName: String, inetAddressList: List<InetAddress>) {
        record(call, 101, "done", "count" to inetAddressList.size); delegate.dnsEnd(call, domainName, inetAddressList)
    }
    override fun connectStart(call: Call, inetSocketAddress: InetSocketAddress, proxy: Proxy) {
        attempts.incrementAndGet(); stage = 102
        record(call, 102, "start", "to" to if (inetSocketAddress.address is Inet6Address) 6 else 4)
        delegate.connectStart(call, inetSocketAddress, proxy)
    }
    override fun secureConnectStart(call: Call) { start(call, 103); delegate.secureConnectStart(call) }
    override fun secureConnectEnd(call: Call, handshake: Handshake?) {
        record(call, 103, "done"); delegate.secureConnectEnd(call, handshake)
    }
    override fun connectEnd(call: Call, inetSocketAddress: InetSocketAddress, proxy: Proxy, protocol: Protocol?) {
        record(call, 102, "done", "to" to protocolCode(protocol)); delegate.connectEnd(call, inetSocketAddress, proxy, protocol)
    }
    override fun connectFailed(call: Call, inetSocketAddress: InetSocketAddress, proxy: Proxy, protocol: Protocol?, ioe: IOException) {
        record(call, stage, "error", "error_code" to failureCode(ioe)); delegate.connectFailed(call, inetSocketAddress, proxy, protocol, ioe)
    }
    override fun connectionAcquired(call: Call, connection: Connection) {
        stage = 104; record(call, 104, "done", "to" to protocolCode(connection.protocol()))
        delegate.connectionAcquired(call, connection)
    }
    override fun requestHeadersStart(call: Call) { start(call, 105); delegate.requestHeadersStart(call) }
    override fun requestHeadersEnd(call: Call, request: Request) { record(call, 105, "done"); delegate.requestHeadersEnd(call, request) }
    override fun requestBodyStart(call: Call) { start(call, 106); delegate.requestBodyStart(call) }
    override fun requestBodyEnd(call: Call, byteCount: Long) {
        record(call, 106, "done", "bytes" to byteCount); stage = 107; delegate.requestBodyEnd(call, byteCount)
    }
    override fun requestFailed(call: Call, ioe: IOException) {
        record(call, stage, "error", "error_code" to failureCode(ioe)); delegate.requestFailed(call, ioe)
    }
    override fun responseHeadersStart(call: Call) { start(call, 107); delegate.responseHeadersStart(call) }
    override fun responseHeadersEnd(call: Call, response: Response) {
        record(call, 107, "done", "status" to response.code); delegate.responseHeadersEnd(call, response)
    }
    override fun responseBodyStart(call: Call) { start(call, 108); delegate.responseBodyStart(call) }
    override fun responseBodyEnd(call: Call, byteCount: Long) {
        record(call, 108, "done", "bytes" to byteCount); delegate.responseBodyEnd(call, byteCount)
    }
    override fun responseFailed(call: Call, ioe: IOException) {
        record(call, stage, "error", "error_code" to failureCode(ioe)); delegate.responseFailed(call, ioe)
    }
    override fun connectionReleased(call: Call, connection: Connection) { record(call, 109, "done"); delegate.connectionReleased(call, connection) }
    override fun canceled(call: Call) { record(call, stage, "cancelled"); delegate.canceled(call) }
    override fun callEnd(call: Call) { finish(); record(call, 100, "done"); delegate.callEnd(call) }
    override fun callFailed(call: Call, ioe: IOException) {
        finish(); record(call, 100, if (call.isCanceled()) "cancelled" else "error",
            "error_code" to failureCode(ioe), "to" to stage); delegate.callFailed(call, ioe)
    }
    companion object {
        private val active = AtomicInteger()
        private fun protocolCode(protocol: Protocol?): Int = when (protocol) { Protocol.HTTP_1_1 -> 1; Protocol.HTTP_2 -> 2; else -> 0 }
        fun failureCode(error: Throwable): Int {
            val seen = HashSet<Throwable>()
            var cause: Throwable? = error
            var result = 0
            while (cause != null && seen.add(cause) && seen.size <= 16) {
                when (cause) {
                    is UnknownHostException -> return 1
                    is ConnectException -> return 2
                    is SSLException -> return 3
                    is SocketTimeoutException -> return 4
                    is SocketException -> result = 5
                    is InterruptedIOException -> if (result == 0) result = 6
                    is IOException -> if (result == 0) result = 7
                }
                cause = cause.cause
            }
            return result
        }
    }
}
