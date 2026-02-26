(function () {
    if (window.__progressSpinnerInstalled) {
        return;
    }
    window.__progressSpinnerInstalled = true;

    const DISPLAY_DELAY_MS = 500;
    const USER_ACTION_WINDOW_MS = 1200;
    const FORM_FALLBACK_MS = 15000;
    const EXPECTED_SLOW_FALLBACK_MS = 4000;
    const DEFAULT_SPINNER_TEXT = '処理中...';

    const ENGLISH_SLOW_HINT_RE = /(send|submit|save|upload|generate|login|signup|verify|revoke|refresh|delete|remove|regenerate|create|setup|appeal|ban|sync|migrate|export|import)/i;
    const JAPANESE_SLOW_HINT_RE = /(送信|保存|アップロード|生成|ログイン|認証|削除|更新|作成|再生成|同期|設定|申し立て)/;

    let pendingCount = 0;
    let showTimer = null;
    let spinnerEl = null;
    let spinnerTextEl = null;
    let lastUserActionAt = 0;
    let lastUserActionLabel = DEFAULT_SPINNER_TEXT;
    let expectedSlowRelease = null;
    let expectedSlowTimer = null;

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
        spinnerEl.innerHTML = '<span class="spinner" aria-hidden="true"></span><span class="progress-text"></span>';
        document.body.appendChild(spinnerEl);
        spinnerTextEl = spinnerEl.querySelector('.progress-text');
        setSpinnerText(lastUserActionLabel || DEFAULT_SPINNER_TEXT);
        return spinnerEl;
    }

    function setSpinnerText(text) {
        const nextText = (typeof text === 'string' && text.trim()) ? text.trim() : DEFAULT_SPINNER_TEXT;
        if (!spinnerEl) {
            spinnerEl = document.getElementById('global-progress-spinner');
        }
        if (!spinnerTextEl && spinnerEl) {
            spinnerTextEl = spinnerEl.querySelector('.progress-text');
        }
        if (spinnerTextEl) {
            spinnerTextEl.textContent = nextText;
        }
        if (spinnerEl) {
            spinnerEl.setAttribute('aria-label', nextText.replace(/\.\.\.$/, ''));
        }
    }

    function normalizeSpinnerLabel(raw) {
        if (typeof raw !== 'string') {
            return '';
        }
        return raw.replace(/\s+/g, ' ').trim();
    }

    function setPendingLabel(label) {
        const normalized = normalizeSpinnerLabel(label);
        lastUserActionLabel = normalized || DEFAULT_SPINNER_TEXT;
        setSpinnerText(lastUserActionLabel);
    }

    function inferSpinnerTextFromText(rawText) {
        const text = normalizeSpinnerLabel(rawText);
        if (!text) {
            return '';
        }

        if (/(送信|送る|submit|send|paper-plane)/i.test(text)) {
            return '送信中...';
        }
        if (/(保存|save)/i.test(text)) {
            return '保存中...';
        }
        if (/(アップロード|upload|添付|ファイルを選択|写真|カメラ)/i.test(text)) {
            return 'アップロード中...';
        }
        if (/(生成|generate|imagine)/i.test(text)) {
            return '生成中...';
        }
        if (/(ログイン|login|sign in|signin)/i.test(text)) {
            return 'ログイン中...';
        }
        if (/(認証|verify|2fa|二要素)/i.test(text)) {
            return '認証中...';
        }
        if (/(削除|delete|remove)/i.test(text)) {
            return '削除中...';
        }
        if (/(作成|create|new)/i.test(text)) {
            return '作成中...';
        }
        if (/(設定|setting|config|preference)/i.test(text)) {
            return '設定を反映中...';
        }
        if (/(更新|refresh|reload|読み込み|load)/i.test(text)) {
            return '読み込み中...';
        }
        return '';
    }

    function inferSpinnerTextFromButton(buttonLike) {
        if (!(buttonLike instanceof Element)) {
            return '';
        }

        const explicitLabel = inferSpinnerTextFromText(buttonLike.getAttribute('data-progress-label'));
        if (explicitLabel) {
            return explicitLabel;
        }

        const candidates = [
            buttonLike.getAttribute('aria-label') || '',
            buttonLike.getAttribute('title') || '',
            buttonLike.value || '',
            buttonLike.textContent || '',
            buttonLike.id || '',
            buttonLike.getAttribute('name') || '',
            typeof buttonLike.className === 'string' ? buttonLike.className : ''
        ];

        for (let i = 0; i < candidates.length; i += 1) {
            const inferred = inferSpinnerTextFromText(candidates[i]);
            if (inferred) {
                return inferred;
            }
        }

        if (buttonLike.closest('form')) {
            return '送信中...';
        }
        return '';
    }

    function inferSpinnerTextFromUrl(url, method) {
        const urlText = normalizeSpinnerLabel(url).toLowerCase();
        const methodText = normalizeSpinnerLabel(method).toUpperCase();
        const combined = (methodText + ' ' + urlText).trim();

        if (!combined) {
            return '';
        }
        if (/upload|attachment|file|\/photo/.test(combined)) {
            return 'アップロード中...';
        }
        if (/message|chat|prompt|stream|reply/.test(combined)) {
            return '送信中...';
        }
        if (/setting|config|preference/.test(combined)) {
            return '保存中...';
        }
        if (/login|signin/.test(combined)) {
            return 'ログイン中...';
        }
        if (/verify|2fa|totp/.test(combined)) {
            return '認証中...';
        }
        if (/generate|image|imagine/.test(combined)) {
            return '生成中...';
        }
        if (/delete|remove/.test(combined) || methodText === 'DELETE') {
            return '削除中...';
        }
        if (/save|update/.test(combined) || methodText === 'PUT' || methodText === 'PATCH') {
            return '保存中...';
        }
        if (methodText === 'GET') {
            return '読み込み中...';
        }
        if (methodText === 'POST') {
            return '送信中...';
        }
        return '';
    }

    function inferSpinnerTextFromFetchArgs(args) {
        const input = args && args[0];
        const init = args && args[1];
        let url = '';
        let method = '';

        if (typeof input === 'string') {
            url = input;
        } else if (input && typeof input.url === 'string') {
            url = input.url;
        }

        if (init && typeof init.method === 'string') {
            method = init.method;
        } else if (input && typeof input.method === 'string') {
            method = input.method;
        }

        return inferSpinnerTextFromUrl(url, method);
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

    function acquirePending(options) {
        const immediate = !!(options && options.immediate);
        const label = options && options.label;
        if (label) {
            setSpinnerText(label);
        } else if (pendingCount <= 0) {
            setSpinnerText(lastUserActionLabel);
        }
        pendingCount += 1;
        if (immediate) {
            clearShowTimer();
            showSpinner();
        } else {
            scheduleShowIfNeeded();
        }

        let released = false;
        return function releasePending() {
            if (released) {
                return;
            }
            released = true;
            if (pendingCount > 0) {
                pendingCount -= 1;
            }
            if (pendingCount <= 0) {
                pendingCount = 0;
                clearShowTimer();
                hideSpinner();
                setSpinnerText(DEFAULT_SPINNER_TEXT);
            }
        };
    }

    function clearExpectedSlowPending() {
        if (expectedSlowTimer) {
            window.clearTimeout(expectedSlowTimer);
            expectedSlowTimer = null;
        }
        if (expectedSlowRelease) {
            expectedSlowRelease();
            expectedSlowRelease = null;
        }
    }

    function startExpectedSlowPending(options) {
        clearExpectedSlowPending();
        expectedSlowRelease = acquirePending({
            immediate: true,
            label: options && options.label
        });
        expectedSlowTimer = window.setTimeout(function () {
            clearExpectedSlowPending();
        }, EXPECTED_SLOW_FALLBACK_MS);
    }

    function startTrackedPending(options) {
        if (!expectedSlowRelease) {
            return acquirePending({ label: options && options.label });
        }

        const releaseExpected = expectedSlowRelease;
        expectedSlowRelease = null;
        if (expectedSlowTimer) {
            window.clearTimeout(expectedSlowTimer);
            expectedSlowTimer = null;
        }

        const trackedRelease = acquirePending({
            immediate: true,
            label: options && options.label
        });
        releaseExpected();
        return trackedRelease;
    }

    function markUserAction(label) {
        lastUserActionAt = Date.now();
        if (label) {
            setPendingLabel(label);
        }
    }

    function isLikelyUserInitiated() {
        if (!lastUserActionAt) {
            return false;
        }
        return (Date.now() - lastUserActionAt) <= USER_ACTION_WINDOW_MS;
    }

    function isExpectedSlowButton(buttonLike) {
        if (!(buttonLike instanceof Element)) {
            return false;
        }

        const noSpinner = (buttonLike.getAttribute('data-progress-no-spinner') || '').toLowerCase();
        if (noSpinner === '1' || noSpinner === 'true' || noSpinner === 'yes') {
            return false;
        }

        const expectedAttr = (buttonLike.getAttribute('data-progress-expected-slow') || '').toLowerCase();
        if (expectedAttr === '1' || expectedAttr === 'true' || expectedAttr === 'yes') {
            return true;
        }
        if (expectedAttr === '0' || expectedAttr === 'false' || expectedAttr === 'no') {
            return false;
        }

        if (buttonLike.closest('[data-progress-no-spinner="true"]')) {
            return false;
        }

        if (buttonLike.matches('button[type="submit"], input[type="submit"]')) {
            return true;
        }
        if (buttonLike.closest('form')) {
            return true;
        }

        const identity = [
            buttonLike.id || '',
            buttonLike.getAttribute('name') || '',
            typeof buttonLike.className === 'string' ? buttonLike.className : ''
        ].join(' ');

        if (ENGLISH_SLOW_HINT_RE.test(identity)) {
            return true;
        }

        const label = (buttonLike.textContent || '').trim();
        return JAPANESE_SLOW_HINT_RE.test(label);
    }

    function installInteractionTracking() {
        document.addEventListener('click', function (event) {
            const target = event.target;
            if (!(target instanceof Element)) {
                return;
            }
            const buttonLike = target.closest('button, input[type="submit"], input[type="button"], [role="button"]');
            if (buttonLike) {
                const actionLabel = inferSpinnerTextFromButton(buttonLike);
                markUserAction(actionLabel);
                if (isExpectedSlowButton(buttonLike)) {
                    startExpectedSlowPending({ label: actionLabel });
                }
            }
        }, true);

        document.addEventListener('submit', function (event) {
            if (event.defaultPrevented) {
                return;
            }
            const submitLabel = inferSpinnerTextFromButton(event.submitter) || inferSpinnerTextFromUrl(event.target && event.target.action, 'POST');
            markUserAction(submitLabel);
            const release = startTrackedPending({ label: submitLabel });
            window.setTimeout(release, FORM_FALLBACK_MS);
        });

        document.addEventListener('keydown', function (event) {
            if (event.key !== 'Enter' && event.key !== ' ') {
                return;
            }
            const target = event.target;
            if (!(target instanceof Element)) {
                return;
            }
            const buttonLike = target.closest('button, input[type="submit"], [role="button"]');
            if (buttonLike) {
                const actionLabel = inferSpinnerTextFromButton(buttonLike);
                markUserAction(actionLabel);
                if (isExpectedSlowButton(buttonLike)) {
                    startExpectedSlowPending({ label: actionLabel });
                }
                return;
            }
            if (target.closest('input, textarea, select')) {
                markUserAction();
            }
        }, true);

        window.addEventListener('pagehide', function () {
            clearExpectedSlowPending();
            pendingCount = 0;
            clearShowTimer();
            hideSpinner();
            setSpinnerText(DEFAULT_SPINNER_TEXT);
        });
    }

    function installFetchTracking() {
        if (typeof window.fetch !== 'function') {
            return;
        }
        const originalFetch = window.fetch.bind(window);
        window.fetch = function () {
            const tracked = isLikelyUserInitiated();
            const requestLabel = tracked ? inferSpinnerTextFromFetchArgs(arguments) : '';
            const release = tracked ? startTrackedPending({ label: requestLabel || lastUserActionLabel }) : null;

            let result;
            try {
                result = originalFetch.apply(window, arguments);
            } catch (err) {
                if (release) {
                    release();
                }
                throw err;
            }

            if (release && result && typeof result.finally === 'function') {
                return result.finally(release);
            }
            return result;
        };
    }

    function installXhrTracking() {
        if (typeof window.XMLHttpRequest !== 'function') {
            return;
        }

        const originalOpen = window.XMLHttpRequest.prototype.open;
        const originalSend = window.XMLHttpRequest.prototype.send;
        window.XMLHttpRequest.prototype.open = function (method, url) {
            this.__progressSpinnerMethod = method;
            this.__progressSpinnerUrl = url;
            return originalOpen.apply(this, arguments);
        };
        window.XMLHttpRequest.prototype.send = function () {
            if (!isLikelyUserInitiated()) {
                return originalSend.apply(this, arguments);
            }

            const xhrLabel = inferSpinnerTextFromUrl(this.__progressSpinnerUrl, this.__progressSpinnerMethod);
            const release = startTrackedPending({ label: xhrLabel || lastUserActionLabel });
            this.addEventListener('loadend', release, { once: true });
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
