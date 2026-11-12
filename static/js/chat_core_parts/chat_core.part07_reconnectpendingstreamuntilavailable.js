        async function reconnectPendingStreamUntilAvailable(pending, threadId) {
            const reconnectThreadId = threadId !== null && threadId !== undefined ? String(threadId) : '';
            const reconnectJobId = normalizeJobIdForUi(pending && pending.job_id);
            const reconnectKey = reconnectJobId || `thread:${reconnectThreadId}`;
            if (!reconnectThreadId || pendingStreamReconnectJobs.has(reconnectKey)) return;
            pendingStreamReconnectJobs.add(reconnectKey);
            const reconnectController = new AbortController();
            let handedOffToResume = false;
            abortController = reconnectController;
            currentJobId = reconnectJobId;
            setSendBtnToStopMode();
            try {
                while (!reconnectController.signal.aborted) {
                    if (String(currentThreadId || '') !== reconnectThreadId) return;
                    if (reconnectJobId && isPendingJobSuppressed(reconnectJobId)) return;
                    const bubble = getActiveStreamingBubbleElement();
                    updatePendingSkeletonStatus(
                        bubble,
                        'サーバーへの再接続を待っています...',
                        '回答処理はバックグラウンドで継続しています'
                    );
                    await window.ConnectionMonitor.waitForRetry(reconnectController.signal);
                    const loaded = await loadMessages(reconnectThreadId, {
                        preserveDraft: true,
                        silent: true,
                        skipHistory: true
                    });
                    if (!loaded) {
                        window.ConnectionMonitor.probeNow();
                        continue;
                    }
                    const latestPending = currentThreadPending;
                    if (latestPending && latestPending.job_id && !isPendingJobSuppressed(latestPending.job_id)) {
                        if (abortController === reconnectController) abortController = null;
                        handedOffToResume = true;
                        latestPending.pendingStartedAt = pending.pendingStartedAt;
                        resumePendingStream(latestPending);
                    } else {
                        window.ConnectionMonitor.markReachable();
                    }
                    return;
                }
            } catch (error) {
                if (error.name !== 'AbortError') {
                    sendClientDebugLog('error', `Stream reconnect failed: ${error.message}`);
                }
            } finally {
                pendingStreamReconnectJobs.delete(reconnectKey);
                if (abortController === reconnectController) abortController = null;
                if (!handedOffToResume) {
                    currentJobId = null;
                    setSendBtnToSendMode();
                    updateFilePreview();
                }
            }
        }
        window.initTurnstileWidget = () => {
            if (!botConfig || !botConfig.turnstileSiteKey || !window.turnstile) return;
            if (turnstileWidgetId !== null) return;
            const container = document.getElementById('turnstile-container');
            if (!container) return;
            // Turnstile must not be rendered inside a display:none container or
            // it may never initialize (this was why the box never appeared).
            // The widget uses appearance:'interaction-only', so it stays visually
            // invisible for normal users while still being ready to execute.
            container.classList.remove('hidden');
            turnstileWidgetId = window.turnstile.render(container, {
                sitekey: botConfig.turnstileSiteKey,
                size: 'compact',
                appearance: 'interaction-only',
                callback: (token) => { turnstileToken = token; turnstilePending = false; verifyTurnstileOnServer(token); },
                'expired-callback': () => { turnstileToken = null; turnstilePending = false; },
                'error-callback': () => { turnstileToken = null; turnstilePending = false; }
            });
            if (isBotDetectionActive()) runBotDetectionGate();
        };
        async function getTurnstileToken(timeoutMs = 1500) {
            if (!botConfig || !botConfig.turnstileSiteKey) return null;
            if (turnstileToken) return turnstileToken;
            if (!window.turnstile) return null;
            // When the blocking dialog is visible, wait for its dedicated
            // (visible) widget to be solved instead of executing the hidden
            // background widget (which would pop a box in the corner).
            if (botDetectionOverlayShown && botDetectionDialogWidgetId !== null) {
                turnstilePending = true;
                return await new Promise((resolve) => {
                    const prevToken = turnstileToken;
                    const timeout = setTimeout(() => resolve(null), Math.max(500, Number(timeoutMs) || 1500));
                    const interval = setInterval(() => {
                        if (turnstileToken && turnstileToken !== prevToken) {
                            clearTimeout(timeout);
                            clearInterval(interval);
                            resolve(turnstileToken);
                        }
                    }, 50);
                });
            }
            if (turnstileWidgetId === null) return null;
            const container = document.getElementById('turnstile-container');
            if (container) container.classList.remove('hidden');
            turnstilePending = true;
            return await new Promise((resolve) => {
                const prevToken = turnstileToken;
                const timeout = setTimeout(() => resolve(null), Math.max(500, Number(timeoutMs) || 1500));
                try {
                    window.turnstile.execute(turnstileWidgetId);
                } catch (e) {
                    clearTimeout(timeout);
                    resolve(null);
                    return;
                }
                const interval = setInterval(() => {
                    if (turnstileToken && turnstileToken !== prevToken) {
                        clearTimeout(timeout);
                        clearInterval(interval);
                        verifyTurnstileOnServer(turnstileToken);
                        resolve(turnstileToken);
                    }
                }, 50);
            });
        }
        function resetTurnstileToken() {
            turnstileToken = null;
            turnstilePending = false;
            if (window.turnstile && turnstileWidgetId !== null) {
                try { window.turnstile.reset(turnstileWidgetId); } catch (e) {}
            }
            if (window.turnstile && botDetectionDialogWidgetId !== null) {
                try { window.turnstile.reset(botDetectionDialogWidgetId); } catch (e) {}
            }
        }
        function isBotDetectionActive() {
            return !!(botConfig && botConfig.globalEnabled && botConfig.accountEnabled && !isAdminUser && botConfig.turnstileSiteKey);
        }
        function renderBotDetectionDialogWidget() {
            if (botDetectionDialogWidgetId !== null) return;
            if (!botConfig || !botConfig.turnstileSiteKey) return;
            const box = document.getElementById('bot-detection-widget-box');
            if (!box) return;
            if (!window.turnstile) {
                // Turnstile API not loaded yet: retry shortly so the dialog box
                // reliably appears once the script finishes loading.
                setTimeout(renderBotDetectionDialogWidget, 250);
                return;
            }
            try {
                botDetectionDialogWidgetId = window.turnstile.render(box, {
                    sitekey: botConfig.turnstileSiteKey,
                    theme: 'dark',
                    size: 'flexible',
                    callback: (token) => {
                        turnstileToken = token;
                        turnstilePending = false;
                        verifyTurnstileOnServer(token, true, true);
                    },
                    'expired-callback': () => {
                        turnstileToken = null;
                        turnstilePending = false;
                        if (botDetectionDialogWidgetId !== null) {
                            try { window.turnstile.reset(botDetectionDialogWidgetId); } catch (e) {}
                        }
                    },
                    'error-callback': () => {
                        turnstileToken = null;
                        turnstilePending = false;
                        if (botDetectionDialogWidgetId !== null) {
                            try { window.turnstile.reset(botDetectionDialogWidgetId); } catch (e) {}
                        }
                    }
                });
            } catch (e) { console.error('bot-detection dialog widget error', e); }
        }
        function showBotDetectionOverlay(message = '') {
            let overlay = document.getElementById('bot-detection-overlay');
            if (!overlay) {
                overlay = document.createElement('div');
                overlay.id = 'bot-detection-overlay';
                overlay.style.cssText = 'position:fixed;inset:0;z-index:2147483000;background:rgba(3,7,18,0.92);display:flex;flex-direction:column;align-items:center;justify-content:center;padding:24px;';
                const card = document.createElement('div');
                card.style.cssText = 'max-width:420px;width:100%;background:#0f172a;border:1px solid #334155;border-radius:12px;padding:24px;text-align:center;box-shadow:0 10px 40px rgba(0,0,0,.5);display:flex;flex-direction:column;align-items:stretch;gap:12px;';
                const title = document.createElement('div');
                title.id = 'bot-detection-overlay-title';
                title.style.cssText = 'font-weight:700;font-size:15px;color:#f1f5f9;';
                title.textContent = message || '安全性の確認中...';
                const desc = document.createElement('div');
                desc.style.cssText = 'font-size:12px;color:#94a3b8;line-height:1.6;';
                desc.textContent = '自動アクセス防止のため、確認を完了してください。';
                const box = document.createElement('div');
                box.id = 'bot-detection-widget-box';
                box.style.cssText = 'margin-top:8px;min-height:65px;display:flex;justify-content:center;';
                card.appendChild(title);
                card.appendChild(desc);
                card.appendChild(box);
                overlay.appendChild(card);
                document.body.appendChild(overlay);
            } else {
                overlay.style.display = 'flex';
            }
            const titleEl = document.getElementById('bot-detection-overlay-title');
            if (message && titleEl) titleEl.textContent = message;
            botDetectionOverlayShown = true;
            renderBotDetectionDialogWidget();
        }
        function hideBotDetectionOverlay() {
            botDetectionOverlayShown = false;
            if (botDetectionDialogWidgetId !== null) {
                try { window.turnstile.remove(botDetectionDialogWidgetId); } catch (e) {}
                botDetectionDialogWidgetId = null;
            }
            const box = document.getElementById('bot-detection-widget-box');
            if (box) box.replaceChildren();
            const overlay = document.getElementById('bot-detection-overlay');
            if (overlay) overlay.remove();
        }
        let botLockOverlay = null;
        let botLockTimer = null;
        function showBotLockOverlay(message = '送信操作が速すぎるため、一時的にロックしています。', remainingSeconds = 600) {
            hideBotDetectionOverlay();
            let overlay = document.getElementById('bot-lock-overlay');
            if (!overlay) {
                overlay = document.createElement('div');
                overlay.id = 'bot-lock-overlay';
                overlay.style.cssText = 'position:fixed;inset:0;z-index:2147483000;background:rgba(3,7,18,0.94);display:flex;flex-direction:column;align-items:center;justify-content:center;padding:24px;';
                const card = document.createElement('div');
                card.style.cssText = 'max-width:440px;width:100%;background:#0f172a;border:1px solid #f59e0b;border-radius:12px;padding:24px;text-align:center;box-shadow:0 10px 40px rgba(0,0,0,.5);display:flex;flex-direction:column;align-items:center;gap:12px;';
                const icon = document.createElement('div');
                icon.style.cssText = 'font-size:26px;color:#fbbf24;';
                icon.innerHTML = '<i class="fas fa-lock"></i>';
                const title = document.createElement('div');
                title.id = 'bot-lock-overlay-title';
                title.style.cssText = 'font-weight:700;font-size:16px;color:#fbbf24;';
                title.textContent = 'アカウントが一時的にロックされました';
                const desc = document.createElement('div');
                desc.id = 'bot-lock-overlay-message';
                desc.style.cssText = 'font-size:13px;color:#f1f5f9;line-height:1.7;';
                desc.textContent = message;
                const timer = document.createElement('div');
                timer.id = 'bot-lock-overlay-timer';
                timer.style.cssText = 'font-size:12px;color:#94a3b8;margin-top:2px;';
                const note = document.createElement('div');
                note.style.cssText = 'font-size:11px;color:#94a3b8;line-height:1.6;';
                note.textContent = 'ロック解除までしばらくお待ちください。同じ操作を繰り返すとBANされる場合があります。';
                card.appendChild(icon);
                card.appendChild(title);
                card.appendChild(desc);
                card.appendChild(timer);
                card.appendChild(note);
                overlay.appendChild(card);
                document.body.appendChild(overlay);
            } else {
                overlay.style.display = 'flex';
                const msgEl = document.getElementById('bot-lock-overlay-message');
                if (msgEl && message) msgEl.textContent = message;
            }
            botLockOverlay = overlay;
            updateBotLockTimer(remainingSeconds);
            return overlay;
        }
        function updateBotLockTimer(remainingSeconds) {
            if (botLockTimer) { clearInterval(botLockTimer); botLockTimer = null; }
            const timerEl = document.getElementById('bot-lock-overlay-timer');
            if (!timerEl) return;
            const render = () => {
                const s = Math.max(0, Math.round(Number(remainingSeconds) || 0));
                const m = Math.floor(s / 60);
                const sec = String(s % 60).padStart(2, '0');
                timerEl.textContent = `ロック解除まで: ${m}:${sec}`;
            };
            render();
            botLockTimer = setInterval(() => {
                remainingSeconds -= 1;
                render();
                if (remainingSeconds <= 0) {
                    if (botLockTimer) { clearInterval(botLockTimer); botLockTimer = null; }
                    location.reload();
                }
            }, 1000);
        }
        function hideBotLockOverlay() {
            if (botLockTimer) { clearInterval(botLockTimer); botLockTimer = null; }
            const overlay = document.getElementById('bot-lock-overlay');
            if (overlay) overlay.remove();
            botLockOverlay = null;
        }
        async function applyBotLockFromServer(reason) {
            // Report rapid operation to the server, which locks the account and
            // returns the remaining lock time. Repeated locks escalate to a ban.
            // Admins are never locked (server returns skipped; no overlay).
            if (isAdminUser) return true;
            let remaining = 600;
            try {
                const res = await apiFetch('/api/bot/lock', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ reason: reason || '' })
                });
                if (res.status === 403) {
                    let data = null;
                    try { data = await res.json(); } catch (e) {}
                    if (data && data.error === 'banned') {
                        showToast('ロックが繰り返されたためBANされました。', 'error', true);
                        setTimeout(() => { location.href = '/banned'; }, 800);
                        return false;
                    }
                }
                const data = await res.json().catch(() => ({}));
                if (data && (data.status === 'skipped' || data.skipped)) return true;
                if (data && typeof data.remaining_seconds === 'number') remaining = data.remaining_seconds;
            } catch (e) {}
            showBotLockOverlay(reason || '送信操作が速すぎるため、一時的にロックしています。', remaining);
            return false;
        }
        const runBotDetectionGate = () => {
            if (botDetectionVerified || !isBotDetectionActive()) return Promise.resolve(true);
            if (botDetectionGatePromise) return botDetectionGatePromise;
            botDetectionGatePromise = (async () => {
                let silentAttempts = 0;
                while (!botDetectionVerified) {
                    // Silent phase: don't show the dialog. Normal users are
                    // verified invisibly (interaction-only widget) and never see
                    // the overlay. Only escalate to the visible dialog once the
                    // silent challenge keeps failing or behavior looks suspicious.
                    if (!botDetectionOverlayShown) {
                        if (!window.__turnstileApiLoaded || turnstileWidgetId === null) {
                            await new Promise((resolve) => setTimeout(resolve, 1000));
                            continue;
                        }
                        const token = await getTurnstileToken(8000);
                        if (token) {
                            const ok = await verifyTurnstileOnServer(token, true, false);
                            if (ok) break;
                        }
                        silentAttempts += 1;
                        let suspicious = false;
                        try {
                            suspicious = !!(botTelemetry && botTelemetry.looksSuspicious && botTelemetry.looksSuspicious());
                        } catch (e) {}
                        if (silentAttempts >= 2 || suspicious) {
                            showBotDetectionOverlay();
                        }
                        continue;
                    }
                    // Dialog is shown: block until the user completes the visible box.
                    const token = await getTurnstileToken(25000);
                    if (token) {
                        const ok = await verifyTurnstileOnServer(token, true, true);
                        if (ok) break;
                    }
                    try { botTelemetry.send(true, { forceReport: true }); } catch (e) {}
                    await new Promise((resolve) => setTimeout(resolve, 5000));
                }
                hideBotDetectionOverlay();
                return true;
            })().finally(() => { botDetectionGatePromise = null; });
            return botDetectionGatePromise;
        };
        function registerSendButtonSpam() {
            const now = performance.now();
            sendButtonSpamTimestamps.push(now);
            // 3s window: slightly more tolerant of accidental double-taps / retries
            sendButtonSpamTimestamps = sendButtonSpamTimestamps.filter(t => now - t <= 3000);
            return sendButtonSpamTimestamps.length;
        }
        function resetSendButtonSpam() {
            sendButtonSpamTimestamps = [];
        }
        async function runSendSpamVerification() {
            // Rapid send-button clicking is treated as suspicious: report it to
            // the server, which temporarily locks the account (10 min) with a
            // visible reason. Repeated locks escalate to a ban.
            if (!isBotDetectionActive()) return true;
            return await applyBotLockFromServer('送信操作が速すぎるため、一時的にロックしています。');
        }
        let turnstileServerVerifiedAt = 0;
        // Single-flight + per-token dedup for server verification. Widget
        // callback, getTurnstileToken, and runBotDetectionGate can all fire for
        // the same single-use Turnstile token; only one POST must go out.
        let turnstileVerifyInFlight = null;
        let turnstileVerifyInFlightToken = null;
        let turnstileLastSubmittedToken = null;
        async function verifyTurnstileOnServer(token, force = false, challenged = null) {
            if (!token || !isBotDetectionActive()) return true;
            if (botDetectionVerified) return true;
            if (challenged === null) challenged = botDetectionOverlayShown;
            const now = Date.now();
            if (!force && now - turnstileServerVerifiedAt < 60 * 1000) return true;
            // Join an in-flight verify for the same token, or skip a token that
            // was already submitted (Turnstile tokens are single-use).
            if (turnstileVerifyInFlight && turnstileVerifyInFlightToken === token) {
                return turnstileVerifyInFlight;
            }
            if (turnstileLastSubmittedToken === token && !force) {
                return !!botDetectionVerified;
            }
            // force=true still must not re-POST a token that already went out.
            if (turnstileLastSubmittedToken === token) {
                if (turnstileVerifyInFlight && turnstileVerifyInFlightToken === token) {
                    return turnstileVerifyInFlight;
                }
                return !!botDetectionVerified;
            }
            turnstileLastSubmittedToken = token;
            turnstileVerifyInFlightToken = token;
            const challengedFlag = !!challenged;
            turnstileVerifyInFlight = (async () => {
                try {
                    const res = await apiFetch('/api/bot/turnstile-verify', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ turnstile_token: token, challenged: challengedFlag })
                    });
                    if (res.ok) {
                        turnstileServerVerifiedAt = Date.now();
                        botDetectionVerified = true;
                        hideBotDetectionOverlay();
                        return true;
                    }
                    return false;
                } catch (e) {
                    return false;
                } finally {
                    if (turnstileVerifyInFlightToken === token) {
                        turnstileVerifyInFlight = null;
                        turnstileVerifyInFlightToken = null;
                    }
                }
            })();
            return turnstileVerifyInFlight;
        }
        function botTurnstileTokenForRequest() {
            return isBotDetectionActive() ? turnstileToken : null;
        }
        const botTelemetry = (() => {
            const state = {
                enabled: false,
                windowStart: performance.now(),
                lastSend: 0,
                clicks: 0,
                keys: 0,
                moves: 0,
                fastClicks: 0,
                fastKeys: 0,
                untrustedInput: false,
                clickTimes: [],
                keyTimes: [],
                clickIntervals: [],
                lastClickTs: 0,
                lastKeyTs: 0,
                lastMove: null,
                speedMax: 0,
                speedSum: 0,
                speedSamples: 0,
                lastMoveSample: 0
            };
            const refreshEnabled = () => {
                state.enabled = !!(botConfig && botConfig.globalEnabled && botConfig.accountEnabled && !isAdminUser);
            };
            const resetWindow = () => {
                state.windowStart = performance.now();
                state.clicks = 0;
                state.keys = 0;
                state.moves = 0;
                state.fastClicks = 0;
                state.fastKeys = 0;
                state.untrustedInput = false;
                state.clickTimes = [];
                state.keyTimes = [];
                state.clickIntervals = [];
                state.speedMax = 0;
                state.speedSum = 0;
                state.speedSamples = 0;
            };
            const isControlClick = (e) => {
                const el = e && e.target;
                if (!el || typeof el.closest !== 'function') return false;
                return !!el.closest('[data-bot-ignore-click], #new-chat-btn, #mobile-new-chat-btn, #bot-detection-overlay');
            };
            const recordClick = (e) => {
                if (isControlClick(e)) return;
                // Script-injected synthetic events (console / automation) have
                // isTrusted === false. A normal user cannot produce these, so
                // treat them as definitive bot evidence and report immediately.
                if (e && e.isTrusted === false) {
                    state.untrustedInput = true;
                    send(true);
                    return;
                }
                const now = performance.now();
                state.clicks += 1;
                if (state.lastClickTs) {
                    const delta = now - state.lastClickTs;
                    state.clickIntervals.push(delta);
                    if (state.clickIntervals.length > 10) state.clickIntervals.shift();
                    if (delta < 120) state.fastClicks += 1;
                }
                state.lastClickTs = now;
                state.clickTimes.push(now);
                state.clickTimes = state.clickTimes.filter(t => now - t <= 2000);
                if (state.fastClicks >= 4) send(true);
            };
            const recordKey = (e) => {
                // Script-injected synthetic key events are also bot evidence.
                if (e && e.isTrusted === false) {
                    state.untrustedInput = true;
                    send(true);
                    return;
                }
                const now = performance.now();
                state.keys += 1;
                if (state.lastKeyTs) {
                    const delta = now - state.lastKeyTs;
                    if (delta < 50) state.fastKeys += 1;
                }
                state.lastKeyTs = now;
                state.keyTimes.push(now);
                state.keyTimes = state.keyTimes.filter(t => now - t <= 2000);
            };
            const recordMove = (e) => {
                const now = performance.now();
                if (now - state.lastMoveSample < 80) return;
                state.lastMoveSample = now;
                state.moves += 1;
                if (state.lastMove) {
                    const dx = e.clientX - state.lastMove.x;
                    const dy = e.clientY - state.lastMove.y;
                    const dt = now - state.lastMove.t;
                    if (dt > 0) {
                        const speed = Math.sqrt(dx * dx + dy * dy) / (dt / 1000);
                        state.speedMax = Math.max(state.speedMax, speed);
                        state.speedSum += speed;
                        state.speedSamples += 1;
                    }
                }
                state.lastMove = { x: e.clientX, y: e.clientY, t: now };
            };
            const computeStats = () => {
                const windowMs = Math.max(1, performance.now() - state.windowStart);
                const clickBurst = state.clickTimes.length;
                const keyBurst = state.keyTimes.length;
                const avgSpeed = state.speedSamples ? (state.speedSum / state.speedSamples) : 0;
                let avgClick = 0;
                let clickCv = 1.0;
                if (state.clickIntervals.length >= 3) {
                    const mean = state.clickIntervals.reduce((a, b) => a + b, 0) / state.clickIntervals.length;
                    const variance = state.clickIntervals.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / state.clickIntervals.length;
                    avgClick = mean;
                    clickCv = mean > 0 ? Math.sqrt(variance) / mean : 1.0;
                }
                return {
                    window_ms: Math.round(windowMs),
                    clicks: state.clicks,
                    keys: state.keys,
                    moves: state.moves,
                    fast_clicks: state.fastClicks,
                    fast_keys: state.fastKeys,
                    untrusted_input: !!state.untrustedInput,
                    click_burst: clickBurst,
                    key_burst: keyBurst,
                    avg_click_ms: avgClick,
                    click_cv: clickCv,
                    event_rate: (state.clicks + state.keys + state.moves) / (windowMs / 1000),
                    pointer_speed_max: state.speedMax,
                    pointer_speed_avg: avgSpeed
                };
            };
            const isSuspicious = (payload) => {
                if (payload.fast_clicks >= 4) return true;
                if (payload.fast_keys >= 8) return true;
                if (payload.click_burst >= 8) return true;
                if (payload.key_burst >= 14) return true;
                if (payload.event_rate >= 20) return true;
                if (payload.avg_click_ms > 0 && payload.avg_click_ms < 160 && payload.click_cv < 0.08) return true;
                return false;
            };
            const send = async (force = false, opts = {}) => {
                if (!state.enabled) return;
                const now = performance.now();
                if (!force && now - state.lastSend < 3000) return;
                state.lastSend = now;
                const payload = computeStats();
                if (!opts.forceReport && (payload.clicks + payload.keys + payload.moves) === 0 && !payload.untrusted_input) return;
                if (!force && !payload.untrusted_input && !isSuspicious(payload)) return;
                payload.turnstile_token = await getTurnstileToken();
                // Only report a Turnstile failure when the user is NOT verified AND
                // the verification dialog is actually on screen. Verified users can
                // momentarily have no client token (it is reset after each report);
                // counting that as a failure was banning legit users. Reporting a
                // failure while the dialog is NOT shown (silent phase) was also
                // banning users who were never shown the dialog, so we only flag it
                // when the user has actually been challenged.
                if (botConfig && botConfig.turnstileSiteKey && !payload.turnstile_token && !botDetectionVerified && botDetectionOverlayShown) {
                    payload.turnstile_failed = true;
                    payload.challenged = true;
                }
                try {
                    const res = await apiFetch('/api/bot-telemetry', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    if (res.status === 403) {
                        let data = null;
                        try { data = await res.json(); } catch (e) {}
                        if (data && data.error === 'banned') {
                            showToast('ボット判定によりBANされました。', 'error', true);
                            setTimeout(() => { location.href = '/banned'; }, 800);
                            return;
                        }
                    }
                } catch (e) {}
                resetTurnstileToken();
                resetWindow();
            };
            const looksSuspicious = () => {
                if (!state.enabled) return false;
                const payload = computeStats();
                return isSuspicious(payload);
            };
            const start = () => {
                refreshEnabled();
                if (!state.enabled) return;
                if (typeof window.PointerEvent !== 'undefined') {
                    document.addEventListener('pointerdown', recordClick, true);
                } else {
                    document.addEventListener('click', recordClick, true);
                }
                document.addEventListener('keydown', recordKey, true);
                document.addEventListener('wheel', () => { state.moves += 1; }, { passive: true });
                document.addEventListener('mousemove', recordMove, true);
                setInterval(() => send(false), 4000);
            };
            return { start, refreshEnabled, send, looksSuspicious };
        })();
        function openImageViewer(url) {
            const viewer = get('image-viewer');
            const img = get('image-viewer-img');
            if (!viewer || !img) return;
            img.src = url;
            viewer.classList.add('visible');
        }
        function closeImageViewer() {
            const viewer = get('image-viewer');
            const img = get('image-viewer-img');
            if (!viewer || !img) return;
            img.src = '';
            viewer.classList.remove('visible');
        }
        function openFileViewer(url, filename = '') {
            if (!url) return;
            const ext = (filename || url).split('.').pop().toLowerCase();
            const imageExt = ['png','jpg','jpeg','webp','gif'];
            const videoExt = ['mp4','mov','mkv','avi','m4v','webm'];
            const audioExt = ['mp3','wav','m4a','ogg','flac'];
            const docExt = ['pdf','txt','md','csv','log','json','docx'];
            if (imageExt.includes(ext)) {
                openImageViewer(url);
                return;
            }
            const viewer = get('file-viewer');
            const body = get('file-viewer-body');
            const title = get('file-viewer-title');
            if (!viewer || !body || !title) return;
            title.textContent = filename || 'File Preview';
            body.replaceChildren();
            if (videoExt.includes(ext)) {
                const video = document.createElement('video');
                video.src = String(url);
                video.controls = true;
                video.playsInline = true;
                video.preload = 'metadata';
                body.appendChild(video);
            } else if (audioExt.includes(ext)) {
                const audio = document.createElement('audio');
                audio.src = String(url);
                audio.controls = true;
                body.appendChild(audio);
            } else if (docExt.includes(ext)) {
                const frame = document.createElement('iframe');
                frame.src = String(url);
                frame.setAttribute('sandbox', '');
                frame.referrerPolicy = 'no-referrer';
                body.appendChild(frame);
            } else {
                const fallback = document.createElement('div');
                fallback.className = 'fallback';
                fallback.appendChild(document.createTextNode('この形式はプレビューできません。'));
                const actions = document.createElement('div');
                actions.className = 'mt-3 flex justify-center gap-2';
                const download = document.createElement('a');
                download.href = String(url);
                download.download = '';
                download.className = 'px-3 py-1 bg-gray-800 text-white rounded text-xs border border-gray-700';
                download.textContent = 'ダウンロード';
                const open = document.createElement('a');
                open.href = String(url);
                open.target = '_blank';
                open.rel = 'noopener noreferrer';
                open.className = download.className;
                open.textContent = '新しいタブで開く';
                actions.append(download, open);
                fallback.appendChild(actions);
                body.appendChild(fallback);
            }
            viewer.classList.add('visible');
        }
        function closeFileViewer() {
            const viewer = get('file-viewer');
            const body = get('file-viewer-body');
            if (!viewer || !body) return;
            body.innerHTML = '';
            viewer.classList.remove('visible');
        }
        // Hoisted: lets static/js/activity_log.js show and record toasts.
        window.showToast = showToast;
        function showToast(msg, type = "error", sticky = false, onClick = null) {
            const stack = get('toast-stack');
            if (!stack) return;
            while (stack.children.length >= 3) {
                stack.removeChild(stack.firstChild);
            }
            const el = document.createElement('div');
            el.className = `toast ${type}${onClick ? ' toast-clickable' : ''}`;
            el.innerHTML = `<i class="fas ${type==='error' ? 'fa-triangle-exclamation' : 'fa-circle-info'}"></i><span class="flex-1">${escapeHtml(msg)}</span><button aria-label="close"><i class="fas fa-times"></i></button>`;
            el.querySelector('button').onclick = (event) => {
                event.stopPropagation();
                el.remove();
            };
            if (onClick) el.addEventListener('click', onClick);
            stack.appendChild(el);
            if (!sticky) setTimeout(() => { if (el.parentNode) el.remove(); }, 7000);
            return el;
        }
        function showProgressToast(msg, type = "info") {
            const stack = get('toast-stack');
            if (!stack) return null;
            while (stack.children.length >= 3) {
                stack.removeChild(stack.firstChild);
            }
            const el = document.createElement('div');
            el.className = `toast ${type} flex-col !items-start min-w-[240px]`;
            el.innerHTML = `
                <div class="flex items-center gap-2 w-full">
                    <i class="fas ${type==='error' ? 'fa-triangle-exclamation' : 'fa-circle-info'}"></i>
                    <span class="flex-1 font-bold">${escapeHtml(msg)}</span>
                    <button aria-label="close" class="ml-auto opacity-50 hover:opacity-100"><i class="fas fa-times"></i></button>
                </div>
                <div class="w-full bg-white/10 h-1.5 rounded-full mt-2.5 overflow-hidden">
                    <div class="progress-bar h-full bg-blue-500 transition-all duration-300 shadow-[0_0_8px_rgba(59,130,246,0.5)]" style="width: 0%"></div>
                </div>
                <div class="w-full text-[10px] text-right mt-1.5 opacity-70 font-mono progress-text">0%</div>
            `;
            el.querySelector('button').onclick = () => el.remove();
            stack.appendChild(el);
            return {
                update: (pct) => {
                    const bar = el.querySelector('.progress-bar');
                    const text = el.querySelector('.progress-text');
                    if (bar) bar.style.width = `${Math.min(100, Math.max(0, pct))}%`;
                    if (text) text.innerText = `${Math.round(pct)}%`;
                },
                remove: () => { if (el.parentNode) el.remove(); }
            };
        }
        let activeSettingsTab = 'general';
        const TAB_LABELS = {
            general: '一般',
            api: 'APIキー',
            prompt: 'プロンプト',
            display: '表示',
            data: 'データ',
            account: 'アカウント',
            security: 'セキュリティ',
            '2fa': '2要素認証',
            feedback: 'フィードバック',
            mcp: 'MCP'
        };
        const ALL_TABS = ['general', 'api', 'prompt', 'display', 'data', 'account', 'security', '2fa', 'feedback', 'mcp'];
        function getSectionHeading(el) {
            const h3 = el.querySelector('h3');
            if (h3) return h3.textContent.trim();
            const bold = el.querySelector('.font-bold');
            if (bold && !bold.querySelector('input') && !bold.querySelector('select')) return bold.textContent.trim();
            const firstLabel = el.querySelector('label');
            if (firstLabel) {
                const t = firstLabel.textContent.trim().replace(/[：:].*$/, '').trim();
                if (t) return t;
            }
            return '';
        }
        function getSectionSnippet(el, query) {
            const text = el.textContent;
            const lower = text.toLowerCase();
            const idx = lower.indexOf(query.toLowerCase());
            if (idx === -1) return '';
            const start = Math.max(0, idx - 25);
            const end = Math.min(text.length, idx + query.length + 35);
            let snippet = text.substring(start, end).replace(/\s+/g, ' ').trim();
            if (start > 0) snippet = '…' + snippet;
            if (end < text.length) snippet = snippet + '…';
            return snippet;
        }
        function removeSearchOverlays() {
            ALL_TABS.forEach(tabId => {
                const tab = get('tab-' + tabId);
                if (!tab) return;
                const ov = tab.querySelector('.settings-search-overlay');
                if (ov) ov.remove();
                Array.from(tab.children).forEach(child => {
                    if (child.classList.contains('settings-no-results')) return;
                    child.style.display = '';
                });
            });
        }
        function filterSettings() {
            const input = get('settings-search');
            if (!input) return;
            const q = input.value.trim().toLowerCase();
            const clearBtn = get('settings-search-clear');
            if (clearBtn) clearBtn.classList.toggle('hidden', !q);
            removeSearchOverlays();
            if (!q) {
                ALL_TABS.forEach(tabId => {
                    const btn = get('btn-tab-' + tabId);
                    if (btn) { const b = btn.querySelector('.settings-search-badge'); if (b) b.remove(); }
                    const tab = get('tab-' + tabId);
                    if (!tab) return;
                    tab.classList.toggle('hidden', tabId !== activeSettingsTab);
                });
                return;
            }
            let results = [];
            ALL_TABS.forEach(tabId => {
                const tab = get('tab-' + tabId);
                if (!tab) return;
                tab.classList.add('hidden');
                Array.from(tab.children).forEach(child => {
                    if (child.classList.contains('settings-no-results') || child.classList.contains('settings-search-overlay')) return;
                    if (child.textContent.toLowerCase().includes(q)) {
                        const title = getSectionHeading(child) || tabId;
                        const snippet = getSectionSnippet(child, q);
                        results.push({ tabId, title, snippet, element: child });
                    }
                });
            });
            let targetTabId = activeSettingsTab;
            if (!results.some(r => r.tabId === targetTabId)) {
                const fr = results.find(r => r.tabId);
                if (fr) targetTabId = fr.tabId;
            }
            const targetTab = get('tab-' + targetTabId);
            if (!targetTab) return;
            targetTab.classList.remove('hidden');
            Array.from(targetTab.children).forEach(child => {
                if (child.classList.contains('settings-no-results') || child.classList.contains('settings-search-overlay')) return;
                child.style.display = 'none';
            });
            const overlay = document.createElement('div');
            overlay.className = 'settings-search-overlay';
            if (results.length === 0) {
                const empty = document.createElement('div');
                empty.className = 'settings-empty-state';
                empty.innerHTML = '<div class="settings-empty-icon"><i class="fas fa-search"></i></div><div class="settings-empty-title">一致する設定はありません</div>';
                const sub = document.createElement('div');
                sub.className = 'settings-empty-sub';
                sub.textContent = '「' + q + '」に一致する設定項目はありません。';
                empty.appendChild(sub);
                overlay.appendChild(empty);
            } else {
                const hdr = document.createElement('div');
                hdr.className = 'settings-search-count';
                hdr.textContent = results.length + '件の一致';
                overlay.appendChild(hdr);
                let prevTabId = null;
                results.forEach((r, i) => {
                    if (r.tabId !== prevTabId) {
                        if (prevTabId !== null) {
                            const sep = document.createElement('div');
                            sep.className = 'border-t border-gray-700/50 my-1.5';
                            overlay.appendChild(sep);
                        }
                        if (r.tabId !== targetTabId) {
                            const lbl = document.createElement('div');
                            lbl.className = 'text-[10px] text-gray-500 px-1 pb-1 font-bold';
                            lbl.textContent = '▼ ' + (TAB_LABELS[r.tabId] || r.tabId);
                            overlay.appendChild(lbl);
                        }
                        prevTabId = r.tabId;
                    }
                    const item = document.createElement('div');
                    item.className = 'settings-search-result-item flex items-start gap-2.5 px-3 py-2.5 rounded-lg cursor-pointer transition-all duration-150';
                    item.style.animation = 'fadeIn 0.28s cubic-bezier(0.22, 1, 0.36, 1) both';
                    item.style.animationDelay = (i * 30) + 'ms';
                    const badge = document.createElement('span');
                    badge.className = 'settings-result-tab-badge shrink-0 mt-0.5';
                    badge.textContent = TAB_LABELS[r.tabId] || r.tabId;
                    const inner = document.createElement('div');
                    inner.className = 'min-w-0 flex-1';
                    const heading = document.createElement('div');
                    heading.className = 'text-sm font-bold text-white truncate';
                    heading.textContent = r.title;
                    const snippet = document.createElement('div');
                    snippet.className = 'text-[11px] text-gray-400 truncate mt-0.5';
                    snippet.textContent = r.snippet;
                    inner.appendChild(heading);
                    inner.appendChild(snippet);
                    item.appendChild(badge);
                    item.appendChild(inner);
                    item.addEventListener('click', () => jumpToSetting(r.tabId, r.element));
                    overlay.appendChild(item);
                });
            }
            targetTab.insertBefore(overlay, targetTab.firstChild);
        }
        function jumpToSetting(tabId, element) {
            const ss = get('settings-search');
            if (ss) ss.value = '';
            removeSearchOverlays();
            filterSettings();
            if (tabId !== activeSettingsTab) switchTab(tabId);
            setTimeout(() => {
                element.scrollIntoView({ behavior: 'smooth', block: 'center' });
                element.classList.add('settings-jump-highlight');
                setTimeout(() => element.classList.remove('settings-jump-highlight'), 2000);
            }, 260);
        }
        function clickTab(t) {
            const ss = get('settings-search');
            if (ss) ss.value = '';
            switchTab(t);
        }
        function switchTab(t) {
            if (t === activeSettingsTab) return;
            if (!ALL_TABS.includes(t)) return;
            const prev = get('tab-' + activeSettingsTab);
            if (prev) {
                prev.classList.remove('tab-enter');
                prev.classList.add('tab-exit');
                setTimeout(() => {
                    prev.classList.add('hidden');
                    prev.classList.remove('tab-exit');
                }, 170);
            }
            ALL_TABS.forEach(x => {
                const btn = get('btn-tab-'+x);
                const panel = get('tab-'+x);
                if(x === t) {
                    if (panel) {
                        panel.classList.remove('hidden');
                        panel.classList.remove('tab-exit');
                        panel.classList.remove('tab-enter');
                        void panel.offsetWidth;
                        panel.classList.add('tab-enter');
                    }
                    if (btn) {
                        btn.classList.add('is-active');
                        try { btn.scrollIntoView({ inline: 'nearest', block: 'nearest', behavior: 'smooth' }); } catch (_) {}
                    }
                } else if (btn) {
                    btn.classList.remove('is-active');
                }
            });
            activeSettingsTab = t;
            filterSettings();
            refreshSettingsTabsScroll();
        }
        function initSettingsSwipeNav() {
            try {
                const area = document.querySelector('#settings-modal .settings-main');
                if (!area || area.dataset.swipeBound === '1') return;
                area.dataset.swipeBound = '1';
                let sx = 0, sy = 0, st = 0, ok = false;
                const blocked = (el) => {
                    for (let n = el; n && n !== area; n = n.parentElement) {
                        if (n.id === 'settings-tabs-wrap') return true;
                        const tag = n.tagName;
                        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || tag === 'PRE' || tag === 'CODE') return true;
                        if (n.scrollWidth > n.clientWidth + 2) {
                            const ox = getComputedStyle(n).overflowX;
                            if (ox === 'auto' || ox === 'scroll') return true;
                        }
                    }
                    return false;
                };
                area.addEventListener('touchstart', (e) => {
                    ok = false;
                    if (e.touches.length !== 1) return;
                    const ss = get('settings-search');
                    if (ss && ss.value.trim()) return;
                    if (blocked(e.target)) return;
                    sx = e.touches[0].clientX; sy = e.touches[0].clientY; st = Date.now(); ok = true;
                }, { passive: true });
                area.addEventListener('touchend', (e) => {
                    if (!ok) return;
                    ok = false;
                    const t = e.changedTouches[0];
                    if (!t) return;
                    const dx = t.clientX - sx, dy = t.clientY - sy;
                    if (Date.now() - st > 600 || Math.abs(dx) < 60 || Math.abs(dx) < Math.abs(dy) * 1.5) return;
                    const next = ALL_TABS[ALL_TABS.indexOf(activeSettingsTab) + (dx < 0 ? 1 : -1)];
                    if (next) switchTab(next);
                }, { passive: true });
            } catch (_) {}
        }
