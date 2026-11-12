package com.minashin1120.aiplayground

import android.app.Application
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.os.SystemClock
import com.minashin1120.aiplayground.data.ActivityLog
import com.minashin1120.aiplayground.data.ApiException
import com.minashin1120.aiplayground.data.ConnectionProbeHealth
import com.minashin1120.aiplayground.data.ConnectionStatus
import com.minashin1120.aiplayground.data.PlaygroundApi
import com.minashin1120.aiplayground.data.ServerOrigin
import com.minashin1120.aiplayground.data.backend.ChatBackend
import com.minashin1120.aiplayground.data.connectionStatusForHttp
import com.minashin1120.aiplayground.data.defaultMessage
import com.minashin1120.aiplayground.data.isDisconnected
import com.minashin1120.aiplayground.data.probeIntervalMillis
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withTimeout

/**
 * Server reachability (Web `setUnavailable` / `markReachable` / the heartbeat probe): the network
 * callback, the foreground probe loop, the unstable-after-three-slow-probes rule, the recovered
 * banner, and the activity-log transport record. State lives in the view model; this class only
 * reads it and reports through `mutable`.
 */
internal class ChatConnectionMonitor(
    application: Application,
    private val state: StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val scope: CoroutineScope,
    private val backend: () -> ChatBackend,
    private val foreground: () -> Boolean,
    private val retryFeedback: () -> Unit,
    private val scheduleSync: () -> Unit,
    private val client: () -> PlaygroundApi,
    private val syncClient: () -> PlaygroundApi,
    private val discardIdle: suspend () -> Unit,
    private val lowBandwidthChanged: () -> Unit,
) {
    private val connectivity = application.getSystemService(ConnectivityManager::class.java)
    private val probeMutex = Mutex()
    private val probeHealth = ConnectionProbeHealth()
    private var lastNetworkLog: List<Any?>? = null
    private var monitorJob: Job? = null
    private var recoveredHideJob: Job? = null
    private var slowCount = 0
    private var callbackRegistered = false

    fun hasUsableNetwork(): Boolean {
        val manager = connectivity ?: return false
        val active = manager.activeNetwork ?: return false
        val capabilities = manager.getNetworkCapabilities(active) ?: return false
        // VPNs and filtered networks can reach this server even when Android's validation probe cannot.
        // Use our HTTP probe to establish reachability instead of treating VALIDATED as a prerequisite.
        return capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
    }

    private fun isConnectionOperationActive(): Boolean {
        val current = state.value
        return current.pairing || current.busy || current.streaming || current.uploading ||
            current.realtime.active || current.lyria.active
    }

    private fun recordNetworkCapabilities(capabilities: NetworkCapabilities?) {
        if (!ActivityLog.enabled) return
        val transport = when {
            capabilities == null -> 0
            capabilities.hasTransport(NetworkCapabilities.TRANSPORT_VPN) -> 4
            capabilities.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) -> 2
            capabilities.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) -> 1
            capabilities.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET) -> 3
            else -> 5
        }
        val online = capabilities?.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) == true
        val validated = capabilities?.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED) == true
        val metered = capabilities != null && !capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_NOT_METERED)
        val restriction = connectivity?.restrictBackgroundStatus ?: 0
        val values = listOf(transport, online, validated, metered, restriction)
        synchronized(this) {
            if (values == lastNetworkLog) return
            lastNetworkLog = values
        }
        ActivityLog.log("state", "reason_code" to 110, "from" to transport, "online" to online,
            "accepted" to validated, "to" to if (metered) 1 else 0, "status" to restriction)
    }

    private val networkCallback = object : ConnectivityManager.NetworkCallback() {
        override fun onAvailable(network: Network) {
            recordNetworkCapabilities(connectivity?.getNetworkCapabilities(network))
            discardIdleServerConnections()
            if (foreground()) scope.launch { retryFeedback(); probe() }
        }

        override fun onLost(network: Network) {
            recordNetworkCapabilities(connectivity?.activeNetwork?.let { connectivity?.getNetworkCapabilities(it) })
            if (!hasUsableNetwork()) setUnavailable(ConnectionStatus.OFFLINE)
        }

        override fun onCapabilitiesChanged(network: Network, networkCapabilities: NetworkCapabilities) {
            recordNetworkCapabilities(networkCapabilities)
            // Web recomputes on `navigator.connection` change and toasts when the mode flips in auto.
            scope.launch { lowBandwidthChanged() }
        }
    }

    fun start() {
        if (!callbackRegistered) {
            runCatching {
                connectivity?.registerDefaultNetworkCallback(networkCallback)
                callbackRegistered = connectivity != null
            }
        }
        if (monitorJob?.isActive == true) return
        monitorJob = scope.launch {
            retryFeedback()
            probe()
            while (isActive && foreground()) {
                delay(state.value.connectionStatus.probeIntervalMillis())
                retryFeedback()
                probe()
            }
        }
    }

    fun stop() {
        monitorJob?.cancel()
        monitorJob = null
        if (callbackRegistered) {
            runCatching { connectivity?.unregisterNetworkCallback(networkCallback) }
            callbackRegistered = false
        }
    }

    suspend fun probe() = probeMutex.withLock {
        if (!foreground() || isConnectionOperationActive()) return@withLock
        if (!hasUsableNetwork()) {
            setUnavailable(ConnectionStatus.OFFLINE)
            return@withLock
        }
        val startedAt = SystemClock.elapsedRealtime()
        val probeStartedNanos = System.nanoTime()
        try {
            val reply = withTimeout(15_000L) {
                backend().get("/api/version?heartbeat=${System.currentTimeMillis()}")
            }
            probeHealth.succeeded()
            val latencyMs = SystemClock.elapsedRealtime() - startedAt
            val wasDisconnected = state.value.connectionStatus.isDisconnected()
            slowCount = if (latencyMs >= 2_000L) slowCount + 1 else 0
            if (slowCount >= 3) {
                setUnavailable(ConnectionStatus.UNSTABLE, "サーバーとの通信が不安定です（遅延 ${latencyMs}ms）")
            } else {
                markReachable(wasDisconnected, resetSlowCount = false)
            }
            // Parsing also ensures a proxy's non-JSON response is not accepted as a heartbeat.
            reply.optString("version")
        } catch (e: TimeoutCancellationException) {
            probeHealth.failed(probeStartedNanos, hasUsableNetwork())?.let { setUnavailable(it) }
        } catch (e: CancellationException) {
            throw e
        } catch (e: ApiException) {
            val mode = connectionStatusForHttp(e.status)
            if (mode != null) setUnavailable(mode)
            else setUnavailable(ConnectionStatus.UNSTABLE, "サーバーでエラーが発生しています（HTTP ${e.status}）")
        } catch (ignored: Exception) {
            probeHealth.failed(probeStartedNanos, hasUsableNetwork())?.let { setUnavailable(it) }
        }
    }

    /** Any server reply shows the server is reachable again (`apiFetch` in Web). */
    fun observe(client: PlaygroundApi) {
        client.onSuccessfulServerResponse = { source ->
            val active = ServerOrigin.current
            if (source.encodedPath != "/api/version" && source.scheme == active.scheme && source.host == active.host && source.port == active.port) {
                probeHealth.succeeded()
                scope.launch {
                    if (foreground() && !state.value.localProfile && state.value.connectionStatus != ConnectionStatus.ONLINE) {
                        markReachable()
                    }
                }
            }
        }
    }

    fun setUnavailable(requested: ConnectionStatus, message: String = requested.defaultMessage()) {
        if (state.value.localProfile) return
        // Web `setUnavailable`: a failed request only shows this server is unreachable; the Internet-offline
        // wording is kept for a device without a network.
        val siteOnly = requested == ConnectionStatus.OFFLINE && hasUsableNetwork()
        val status = if (siteOnly) ConnectionStatus.UNSTABLE else requested
        val text = if (siteOnly) "このサイトへの通信が完了しませんでした。接続を再確認しています" else message
        recoveredHideJob?.cancel()
        recoveredHideJob = null
        mutable.update { it.copy(
            connectionStatus = status,
            connectionMessage = text,
            connectionBannerVisible = true,
            offline = status.isDisconnected(),
        ) }
    }

    fun markReachable(wasDisconnected: Boolean = state.value.connectionStatus.isDisconnected(), resetSlowCount: Boolean = true) {
        if (resetSlowCount) slowCount = 0
        recoveredHideJob?.cancel()
        mutable.update { it.copy(
            connectionStatus = ConnectionStatus.ONLINE,
            connectionMessage = if (wasDisconnected) ConnectionStatus.ONLINE.defaultMessage() else "",
            connectionBannerVisible = wasDisconnected,
            offline = false,
        ) }
        retryFeedback()
        // Serverless mode: answers finished while the server was out of reach go up now.
        if (wasDisconnected && state.value.serverless) scheduleSync()
        if (wasDisconnected) {
            recoveredHideJob = scope.launch {
                delay(5_000L)
                mutable.update { current ->
                    if (current.connectionStatus == ConnectionStatus.ONLINE) current.copy(connectionBannerVisible = false)
                    else current
                }
            }
        }
    }

    fun cancelRecoveredHide() {
        recoveredHideJob?.cancel()
        recoveredHideJob = null
    }

    /** Web `discardIdleServerConnections`: drops the idle sockets of the clients the screen keeps around. */
    fun discardIdleServerConnections() = scope.launch {
        client().discardIdleConnections()
        syncClient().discardIdleConnections()
        discardIdle()
    }

    /** View-model teardown (Web `onCleared`): stops the probe loop and the recovered banner. */
    fun release() {
        stop()
        cancelRecoveredHide()
    }
}
