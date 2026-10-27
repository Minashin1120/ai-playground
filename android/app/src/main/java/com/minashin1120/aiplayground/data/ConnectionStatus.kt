package com.minashin1120.aiplayground.data

/** Connection states shared by the heartbeat monitor and the Compose banner. */
enum class ConnectionStatus {
    UNKNOWN,
    ONLINE,
    OFFLINE,
    UNSTABLE,
    MAINTENANCE,
    SERVER_DOWN,
}

fun ConnectionStatus.isDisconnected(): Boolean = when (this) {
    ConnectionStatus.OFFLINE,
    ConnectionStatus.MAINTENANCE,
    ConnectionStatus.SERVER_DOWN -> true
    ConnectionStatus.UNKNOWN,
    ConnectionStatus.UNSTABLE,
    ConnectionStatus.ONLINE -> false
}

fun ConnectionStatus.defaultMessage(): String = when (this) {
    ConnectionStatus.OFFLINE -> "インターネット接続が切断されています"
    ConnectionStatus.UNSTABLE -> "サーバーとの通信が不安定です"
    ConnectionStatus.MAINTENANCE -> "サーバーはメンテナンス中です（自動再接続します）"
    ConnectionStatus.SERVER_DOWN -> "サーバーが停止しているか応答していません（自動再接続します）"
    ConnectionStatus.ONLINE -> "サーバーとの通信が復帰しました"
    ConnectionStatus.UNKNOWN -> ""
}

fun ConnectionStatus.probeIntervalMillis(): Long =
    if (isDisconnected() || this == ConnectionStatus.UNSTABLE) 2_000L else 5_000L

fun connectionStatusForHttp(code: Int): ConnectionStatus? = when (code) {
    503 -> ConnectionStatus.MAINTENANCE
    502, 504, 520, 521, 522, 523, 524 -> ConnectionStatus.SERVER_DOWN
    else -> null
}

/** Isolated probe failures must not override successful application traffic. */
internal class ConnectionProbeHealth {
    private var lastSuccessNanos = Long.MIN_VALUE
    private var failures = 0

    @Synchronized fun succeeded(atNanos: Long = System.nanoTime()) {
        lastSuccessNanos = atNanos
        failures = 0
    }

    @Synchronized fun failed(probeStartedNanos: Long, usableNetwork: Boolean): ConnectionStatus? {
        if (!usableNetwork) return ConnectionStatus.OFFLINE
        if (lastSuccessNanos >= probeStartedNanos) return null
        return if (++failures >= 2) ConnectionStatus.UNSTABLE else null
    }
}
