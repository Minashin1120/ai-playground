(function () {
    if (window.__progressSpinnerInstalled) {
        return;
    }
    window.__progressSpinnerInstalled = true;

    const DISPLAY_DELAY_MS = 500;
    const USER_ACTION_WINDOW_MS = 1200;
    const FORM_FALLBACK_MS = 15000;

    let pendingCount = 0;
    let showTimer = null;
    let spinnerEl = null;
    let lastUserActionAt = 0;

    function ensureSpinnerElement() {
        if (spinnerEl) {
            return spinnerEl;
        }

        const style = document.createElement('style');
        style.id = 'global-progress-spinner-style';
        style.textContent = [
            '#global-progress-spinner{',
            'position:fixed;',
            'right:16px;',
            'bottom:16px;',
            'z-index:99999;',
            'display:flex;',
            'align-items:center;',
            'gap:10px;',
            'padding:10px 12px;',
            'border-radius:999px;',
            'background:rgba(2,6,23,0.92);',
            'border:1px solid rgba(148,163,184,0.35);',
            'box-shadow:0 14px 32px rgba(0,0,0,0.35);',
            'color:#e2e8f0;',
            'font-size:12px;',
            'font-weight:600;',
            'letter-spacing:0.02em;',
            'opacity:0;',
            'transform:translateY(6px);',
            'pointer-events:none;',
            'visibility:hidden;',
            'transition:opacity 0.18s ease, transform 0.18s ease, visibility 0.18s ease;',
            '}',
            '#global-progress-spinner.active{',
            'opacity:1;',
            'transform:translateY(0);',
            'visibility:visible;',
            '}',
            '#global-progress-spinner .spinner{',
            'width:16px;',
            'height:16px;',
            'border-radius:50%;',
            'border:2px solid rgba(148,163,184,0.35);',
            'border-top-color:#5eead4;',
            'animation:global-progress-spinner-rotate 0.7s linear infinite;',
            '}',
            '@keyframes global-progress-spinner-rotate{',
            'to{transform:rotate(360deg);}',
            '}'
        ].join('');

        if (!document.getElementById(style.id)) {
            document.head.appendChild(style);
        }

        spinnerEl = document.createElement('div');
        spinnerEl.id = 'global-progress-spinner';
        spinnerEl.setAttribute('aria-live', 'polite');
        spinnerEl.setAttribute('aria-label', '処理中');
        spinnerEl.innerHTML = '<span class="spinner" aria-hidden="true"></span><span>処理中...</span>';
        document.body.appendChild(spinnerEl);
        return spinnerEl;
    }

    function showSpinner() {
        const el = ensureSpinnerElement();
        el.classList.add('active');
    }

    function hideSpinner() {
        if (spinnerEl) {
            spinnerEl.classList.remove('active');
        }
    }

    function scheduleShowIfNeeded() {
        if (showTimer || pendingCount <= 0) {
            return;
        }
        showTimer = window.setTimeout(function () {
            showTimer = null;
            if (pendingCount > 0) {
                showSpinner();
            }
        }, DISPLAY_DELAY_MS);
    }

    function clearShowTimer() {
        if (!showTimer) {
            return;
        }
        window.clearTimeout(showTimer);
        showTimer = null;
    }

    function beginPending() {
        pendingCount += 1;
        scheduleShowIfNeeded();
    }

    function endPending() {
        if (pendingCount > 0) {
            pendingCount -= 1;
        }
        if (pendingCount <= 0) {
            pendingCount = 0;
            clearShowTimer();
            hideSpinner();
        }
    }

    function markUserAction() {
        lastUserActionAt = Date.now();
    }

    function isLikelyUserInitiated() {
        if (!lastUserActionAt) {
            return false;
        }
        return (Date.now() - lastUserActionAt) <= USER_ACTION_WINDOW_MS;
    }

    function installInteractionTracking() {
        document.addEventListener('click', function (event) {
            const target = event.target;
            if (!(target instanceof Element)) {
                return;
            }
            const buttonLike = target.closest('button, input[type="submit"], input[type="button"], [role="button"]');
            if (buttonLike) {
                markUserAction();
            }
        }, true);

        document.addEventListener('submit', function (event) {
            if (event.defaultPrevented) {
                return;
            }
            markUserAction();
            beginPending();
            window.setTimeout(endPending, FORM_FALLBACK_MS);
        }, true);

        document.addEventListener('keydown', function (event) {
            if (event.key !== 'Enter' && event.key !== ' ') {
                return;
            }
            const target = event.target;
            if (!(target instanceof Element)) {
                return;
            }
            if (target.closest('button, [role="button"], input, textarea, select')) {
                markUserAction();
            }
        }, true);

        window.addEventListener('pagehide', function () {
            pendingCount = 0;
            clearShowTimer();
            hideSpinner();
        });
    }

    function installFetchTracking() {
        if (typeof window.fetch !== 'function') {
            return;
        }
        const originalFetch = window.fetch.bind(window);
        window.fetch = function () {
            const tracked = isLikelyUserInitiated();
            if (tracked) {
                beginPending();
            }

            let result;
            try {
                result = originalFetch.apply(window, arguments);
            } catch (err) {
                if (tracked) {
                    endPending();
                }
                throw err;
            }

            if (tracked && result && typeof result.finally === 'function') {
                return result.finally(endPending);
            }
            return result;
        };
    }

    function installXhrTracking() {
        if (typeof window.XMLHttpRequest !== 'function') {
            return;
        }

        const originalSend = window.XMLHttpRequest.prototype.send;
        window.XMLHttpRequest.prototype.send = function () {
            const tracked = isLikelyUserInitiated();
            if (tracked) {
                beginPending();
                this.addEventListener('loadend', endPending, { once: true });
            }
            return originalSend.apply(this, arguments);
        };
    }

    function boot() {
        installInteractionTracking();
        installFetchTracking();
        installXhrTracking();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', boot, { once: true });
    } else {
        boot();
    }
})();
