/* connection_monitor.js
 *
 * 接続状態の監視・表示・復旧まわりを一元管理するモジュール。
 *
 * - サーバーとの通信状態を定期ハートビート（/api/version）で監視し、
 *   オフライン／不安定／メンテナンス／サーバーダウン／復帰をバナー表示する。
 * - ファイルアップロードや回答生成のような長時間処理（operation）の実行中は
 *   ネットワークハートビートを送らず、その処理自体の進捗（reportActivity）を
 *   接続が生きている証拠として扱う。処理の失敗は各処理側のエラーハンドラーが
 *   通知するため、誤検知による「切断」表示を防ぐ。
 *
 * 外部からは window.ConnectionMonitor として利用する。
 * chat_core より先に読み込むこと。
 */
(function () {
    'use strict';

    const get = (id) => document.getElementById(id);

    const CONNECTION_CHECK_INTERVAL_MS = 5000;
    const CONNECTION_CHECK_FAST_INTERVAL_MS = 2000;
    const CONNECTION_CHECK_TIMEOUT_MS = 3000;
    const CONNECTION_UNSTABLE_LATENCY_MS = 2000;
    const CONNECTION_SLOW_TO_UNSTABLE = 3;
    const CONNECTION_RECOVERED_BANNER_MS = 5000;
    const CONNECTION_RETRY_DELAY_MS = 2000;

    let connectionCheckTimer = null;
    let connectionCheckIntervalMs = 0;
    let connectionCheckInFlight = false;
    let connectionCheckAbortController = null;
    let connectionProbeSequence = 0;
    let connectionConsecutiveSlow = 0;
    let connectionStatus = 'unknown';
    let connectionRecoveredHideTimer = null;
    let activeOperationCount = 0;
    let versionChangeHandler = null;

    function setVersionChangeHandler(fn) {
        versionChangeHandler = typeof fn === 'function' ? fn : null;
    }

    function setConnectionBanner(mode, message = '') {
        const b = get('offline-banner');
        const icon = get('offline-banner-icon');
        const text = get('offline-banner-text');
        if (!b || !icon || !text) return;
        if (mode !== 'online' && connectionRecoveredHideTimer) {
            window.clearTimeout(connectionRecoveredHideTimer);
            connectionRecoveredHideTimer = null;
        }
        if (mode === 'hidden') {
            b.classList.remove('visible', 'offline', 'unstable', 'maintenance', 'server-down', 'online');
            document.body.classList.remove('network-banner-visible');
            return;
        }
        b.classList.add('visible');
        document.body.classList.add('network-banner-visible');
        if (mode === 'offline') {
            b.classList.add('offline');
            b.classList.remove('unstable', 'maintenance', 'server-down', 'online');
            icon.className = 'fas fa-unlink';
            text.textContent = message || 'インターネット接続が切断されています';
            return;
        }
        if (mode === 'maintenance') {
            b.classList.add('maintenance');
            b.classList.remove('offline', 'unstable', 'server-down', 'online');
            icon.className = 'fas fa-screwdriver-wrench';
            text.textContent = message || 'サーバーはメンテナンス中です（自動再接続します）';
            return;
        }
        if (mode === 'server-down') {
            b.classList.add('server-down');
            b.classList.remove('offline', 'unstable', 'maintenance', 'online');
            icon.className = 'fas fa-server';
            text.textContent = message || 'サーバーが停止しているか応答していません（自動再接続します）';
            return;
        }
        if (mode === 'online') {
            b.classList.add('online');
            b.classList.remove('offline', 'unstable', 'maintenance', 'server-down');
            icon.className = 'fas fa-check-circle';
            text.textContent = message || 'サーバーとの通信が復帰しました';
            return;
        }
        b.classList.add('unstable');
        b.classList.remove('offline', 'maintenance', 'server-down', 'online');
        icon.className = 'fas fa-exclamation-triangle';
        text.textContent = message || 'サーバーとの通信が不安定です';
    }

    function isDisconnectedConnectionStatus(status = connectionStatus) {
        return ['offline', 'unstable', 'maintenance', 'server-down'].includes(status);
    }

    function setUnavailable(mode, message = '') {
        connectionConsecutiveSlow = 0;
        connectionStatus = mode;
        setConnectionBanner(mode, message);
        refreshConnectionMonitorTimer();
    }

    function markReachable() {
        const wasDisconnected = isDisconnectedConnectionStatus();
        connectionConsecutiveSlow = 0;
        connectionStatus = 'online';
        if (wasDisconnected) showConnectionRecoveredBanner();
        else setConnectionBanner('hidden');
        refreshConnectionMonitorTimer();
    }

    function showConnectionRecoveredBanner(message = 'サーバーとの通信が復帰しました') {
        setConnectionBanner('online', message);
        connectionRecoveredHideTimer = window.setTimeout(() => {
            connectionRecoveredHideTimer = null;
            if (connectionStatus === 'online') {
                setConnectionBanner('hidden');
            }
        }, CONNECTION_RECOVERED_BANNER_MS);
    }

    function getConnectionCheckIntervalMs() {
        if (isDisconnectedConnectionStatus()) return CONNECTION_CHECK_FAST_INTERVAL_MS;
        return CONNECTION_CHECK_INTERVAL_MS;
    }

    function refreshConnectionMonitorTimer(force = false) {
        const nextIntervalMs = getConnectionCheckIntervalMs();
        if (!force && connectionCheckTimer && connectionCheckIntervalMs === nextIntervalMs) return;
        if (connectionCheckTimer) {
            window.clearInterval(connectionCheckTimer);
            connectionCheckTimer = null;
        }
        connectionCheckIntervalMs = nextIntervalMs;
        connectionCheckTimer = window.setInterval(probeServerConnection, nextIntervalMs);
    }

    function cancelProbe() {
        connectionProbeSequence += 1;
        if (connectionCheckAbortController) {
            connectionCheckAbortController.abort();
            connectionCheckAbortController = null;
        }
        connectionCheckInFlight = false;
    }

    function isOperationActive() {
        return activeOperationCount > 0;
    }

    function operationStarted() {
        if (activeOperationCount === 0) {
            cancelProbe();
            if (connectionCheckTimer) {
                window.clearInterval(connectionCheckTimer);
                connectionCheckTimer = null;
                connectionCheckIntervalMs = 0;
            }
            connectionConsecutiveSlow = 0;
            connectionStatus = 'online';
            setConnectionBanner('hidden');
        }
        activeOperationCount += 1;
    }

    function operationEnded() {
        if (activeOperationCount > 0) activeOperationCount -= 1;
        if (activeOperationCount === 0) {
            refreshConnectionMonitorTimer(true);
            probeServerConnection();
        }
    }

    function reportActivity() {
        if (activeOperationCount === 0) return;
        connectionConsecutiveSlow = 0;
        connectionStatus = 'online';
        setConnectionBanner('hidden');
    }

    async function probeServerConnection() {
        if (isOperationActive()) return;
        if (!navigator.onLine) {
            if (connectionCheckInFlight) cancelProbe();
            setUnavailable('offline');
            return;
        }
        if (connectionCheckInFlight) return;
        connectionCheckInFlight = true;
        const probeSequence = ++connectionProbeSequence;
        const startedAt = performance.now();
        const ctrl = new AbortController();
        connectionCheckAbortController = ctrl;
        const timeoutId = window.setTimeout(() => ctrl.abort(), CONNECTION_CHECK_TIMEOUT_MS);
        try {
            const heartbeatRes = await fetch(`/api/version?heartbeat=${Date.now()}`, {
                method: 'GET',
                cache: 'no-store',
                credentials: 'include',
                signal: ctrl.signal,
                headers: { 'Accept': 'application/json', 'Cache-Control': 'no-cache' }
            });
            if (probeSequence !== connectionProbeSequence) return;
            if (heartbeatRes.status === 503) {
                setUnavailable('maintenance');
                return;
            }
            if ([502, 504, 520, 521, 522, 523, 524].includes(heartbeatRes.status)) {
                setUnavailable('server-down');
                return;
            }
            if (!heartbeatRes.ok) {
                connectionStatus = 'unstable';
                setConnectionBanner('unstable', `サーバーでエラーが発生しています（HTTP ${heartbeatRes.status}）`);
                refreshConnectionMonitorTimer();
                return;
            }
            const hbData = await heartbeatRes.json();
            if (probeSequence !== connectionProbeSequence) return;
            const latencyMs = Math.round(performance.now() - startedAt);
            const wasDisconnected = isDisconnectedConnectionStatus();
            if (latencyMs >= CONNECTION_UNSTABLE_LATENCY_MS) {
                connectionConsecutiveSlow += 1;
            } else {
                connectionConsecutiveSlow = 0;
            }
            if (connectionConsecutiveSlow >= CONNECTION_SLOW_TO_UNSTABLE) {
                connectionStatus = 'unstable';
                setConnectionBanner('unstable', `サーバーとの通信が不安定です（遅延 ${latencyMs}ms）`);
            } else {
                connectionStatus = 'online';
                if (wasDisconnected) {
                    showConnectionRecoveredBanner();
                } else {
                    setConnectionBanner('hidden');
                }
            }
            refreshConnectionMonitorTimer();
            const hbVersion = hbData.version || "";
            if (hbVersion && versionChangeHandler) versionChangeHandler(hbVersion);
        } catch (e) {
            if (probeSequence !== connectionProbeSequence) return;
            if (isOperationActive()) return;
            setUnavailable('offline');
        } finally {
            window.clearTimeout(timeoutId);
            if (probeSequence === connectionProbeSequence) {
                connectionCheckAbortController = null;
                connectionCheckInFlight = false;
            }
        }
    }

    function start() {
        refreshConnectionMonitorTimer(true);
        probeServerConnection();
    }

    function stop() {
        if (connectionCheckTimer) {
            window.clearInterval(connectionCheckTimer);
            connectionCheckTimer = null;
            connectionCheckIntervalMs = 0;
        }
        cancelProbe();
        if (connectionRecoveredHideTimer) {
            window.clearTimeout(connectionRecoveredHideTimer);
            connectionRecoveredHideTimer = null;
        }
    }

    function probeNow() {
        cancelProbe();
        probeServerConnection();
    }

    function waitForRetry(signal, delayMs = CONNECTION_RETRY_DELAY_MS) {
        return new Promise((resolve, reject) => {
            if (signal && signal.aborted) {
                reject(new DOMException('Aborted', 'AbortError'));
                return;
            }
            const timer = window.setTimeout(() => {
                if (signal) signal.removeEventListener('abort', onAbort);
                resolve();
            }, delayMs);
            const onAbort = () => {
                window.clearTimeout(timer);
                if (signal) signal.removeEventListener('abort', onAbort);
                reject(new DOMException('Aborted', 'AbortError'));
            };
            if (signal) signal.addEventListener('abort', onAbort, { once: true });
        });
    }

    function retryModeForResponse(response) {
        if (response.status === 503) return 'maintenance';
        if ([502, 504, 520, 521, 522, 523, 524].includes(response.status)) return 'server-down';
        return null;
    }

    window.ConnectionMonitor = {
        start,
        stop,
        probeNow,
        cancelProbe,
        markReachable,
        setUnavailable,
        isDisconnected: isDisconnectedConnectionStatus,
        isOperationActive,
        operationStarted,
        operationEnded,
        reportActivity,
        waitForRetry,
        retryModeForResponse,
        setVersionChangeHandler
    };
})();
