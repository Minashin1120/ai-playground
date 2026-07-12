(function () {
    if (window.__progressSpinnerInstalled) {
        return;
    }
    window.__progressSpinnerInstalled = true;

    const DISPLAY_DELAY_MS = 400;
    const FORM_FALLBACK_MS = 15000;
    const DEFAULT_SPINNER_TEXT = '処理中...';
    const PASSIVE_REQUEST_RE = /(?:\/api\/version(?:[/?]|$)|\/api\/(?:debug|metrics)(?:[/?]|$)|\/api\/bot-telemetry(?:[/?]|$)|\/api\/temporary_chat\/heartbeat(?:[/?]|$))/i;

    let nextOperationId = 1;
    let showTimer = null;
    let spinnerEl = null;
    let spinnerTextEl = null;
    let activeInteractionLabel = '';
    let suppressCurrentInteraction = false;
    const operations = new Map();

    function ensureSpinnerElement() {
        if (spinnerEl && spinnerEl.isConnected !== false) {
            return spinnerEl;
        }

        let style = document.getElementById('global-progress-spinner-style');
        if (!style) {
            style = document.createElement('style');
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
            document.head.appendChild(style);
        }

        spinnerEl = document.getElementById('global-progress-spinner');
        if (!spinnerEl) {
            spinnerEl = document.createElement('div');
            spinnerEl.id = 'global-progress-spinner';
            spinnerEl.setAttribute('role', 'status');
            spinnerEl.setAttribute('aria-live', 'polite');
            spinnerEl.setAttribute('aria-label', '処理中');
            spinnerEl.innerHTML = '<span class="spinner" aria-hidden="true"></span><span class="progress-text"></span>';
            document.body.appendChild(spinnerEl);
        }
        spinnerTextEl = spinnerEl.querySelector('.progress-text');
        updateSpinnerText();
        return spinnerEl;
    }

    function normalizeSpinnerLabel(raw) {
        return typeof raw === 'string' ? raw.replace(/\s+/g, ' ').trim() : '';
    }

    function inferSpinnerTextFromText(rawText) {
        const text = normalizeSpinnerLabel(rawText);
        if (!text) return '';
        if (/(送信|送る|submit|send|paper-plane)/i.test(text)) return '送信中...';
        if (/(保存|save)/i.test(text)) return '保存中...';
        if (/(アップロード|upload|添付|ファイルを選択|写真|カメラ)/i.test(text)) return 'アップロード中...';
        if (/(生成|generate|imagine)/i.test(text)) return '生成中...';
        if (/(ログイン|login|sign in|signin)/i.test(text)) return 'ログイン中...';
        if (/(認証|verify|2fa|二要素)/i.test(text)) return '認証中...';
        if (/(削除|delete|remove)/i.test(text)) return '削除中...';
        if (/(作成|create|new)/i.test(text)) return '作成中...';
        if (/(設定|setting|config|preference)/i.test(text)) return '設定を反映中...';
        if (/(更新|refresh|reload|読み込み|load)/i.test(text)) return '読み込み中...';
        return '';
    }

    function inferSpinnerTextFromButton(buttonLike) {
        if (!(buttonLike instanceof Element)) return '';
        const candidates = [
            buttonLike.getAttribute('data-progress-label') || '',
            buttonLike.getAttribute('aria-label') || '',
            buttonLike.getAttribute('title') || '',
            buttonLike.value || '',
            buttonLike.textContent || '',
            buttonLike.id || '',
            buttonLike.getAttribute('name') || ''
        ];
        for (let i = 0; i < candidates.length; i += 1) {
            const inferred = inferSpinnerTextFromText(candidates[i]);
            if (inferred) return inferred;
        }
        return buttonLike.closest('form') ? '送信中...' : '';
    }

    function inferSpinnerTextFromUrl(url, method) {
        const urlText = normalizeSpinnerLabel(url).toLowerCase();
        const methodText = normalizeSpinnerLabel(method || 'GET').toUpperCase();
        const combined = `${methodText} ${urlText}`;
        if (/delete|remove/.test(combined) || methodText === 'DELETE') return '削除中...';
        if (/message|chat|prompt|stream|reply/.test(combined)) return '送信中...';
        if (/setting|config|preference/.test(combined)) return '保存中...';
        if (/login|signin/.test(combined)) return 'ログイン中...';
        if (/verify|2fa|totp|webauthn|passkey/.test(combined)) return '認証中...';
        if (/upload|attachment|\/photo/.test(combined)) return 'アップロード中...';
        if (/generate|image|imagine/.test(combined)) return '生成中...';
        if (/save|update/.test(combined) || methodText === 'PUT' || methodText === 'PATCH') return '保存中...';
        if (methodText === 'GET' || methodText === 'HEAD') return '読み込み中...';
        if (methodText === 'POST') return '送信中...';
        return DEFAULT_SPINNER_TEXT;
    }

    function getFetchDetails(args) {
        const input = args && args[0];
        const init = (args && args[1]) || {};
        const url = typeof input === 'string' || input instanceof URL
            ? String(input)
            : (input && typeof input.url === 'string' ? input.url : '');
        const method = init.method || (input && input.method) || 'GET';
        return { url, method, disabled: init.progressSpinner === false };
    }

    function isPassiveRequest(url) {
        if (!url) return false;
        try {
            const parsed = new URL(url, window.location.href);
            return PASSIVE_REQUEST_RE.test(parsed.pathname);
        } catch (_) {
            return PASSIVE_REQUEST_RE.test(String(url));
        }
    }

    function latestOperation() {
        let latest = null;
        operations.forEach(function (operation) {
            if (!latest || operation.id > latest.id) latest = operation;
        });
        return latest;
    }

    function updateSpinnerText() {
        const latest = latestOperation();
        const text = (latest && latest.label) || DEFAULT_SPINNER_TEXT;
        if (spinnerTextEl) spinnerTextEl.textContent = text;
        if (spinnerEl) spinnerEl.setAttribute('aria-label', text.replace(/\.\.\.$/, ''));
    }

    function hideSpinner() {
        if (spinnerEl) spinnerEl.classList.remove('active');
    }

    function syncSpinner() {
        if (operations.size === 0) {
            if (showTimer) window.clearTimeout(showTimer);
            showTimer = null;
            hideSpinner();
            updateSpinnerText();
            return;
        }

        updateSpinnerText();
        if (spinnerEl && spinnerEl.classList.contains('active')) return;
        if (showTimer) return;
        showTimer = window.setTimeout(function () {
            showTimer = null;
            if (operations.size > 0) ensureSpinnerElement().classList.add('active');
        }, DISPLAY_DELAY_MS);
    }

    function startOperation(options) {
        const operation = {
            id: nextOperationId++,
            label: normalizeSpinnerLabel(options && options.label) || DEFAULT_SPINNER_TEXT
        };
        operations.set(operation.id, operation);
        syncSpinner();

        let finished = false;
        return function finishOperation() {
            if (finished) return;
            finished = true;
            operations.delete(operation.id);
            syncSpinner();
        };
    }

    function shouldTrackRequest(url, explicitlyDisabled) {
        return !explicitlyDisabled && !suppressCurrentInteraction && !isPassiveRequest(url);
    }

    function setInteractionContext(buttonLike) {
        const noSpinner = buttonLike && buttonLike.closest('[data-progress-no-spinner="true"], [data-progress-no-spinner="1"]');
        suppressCurrentInteraction = !!noSpinner;
        activeInteractionLabel = noSpinner ? '' : inferSpinnerTextFromButton(buttonLike);
        window.setTimeout(function () {
            suppressCurrentInteraction = false;
            activeInteractionLabel = '';
        }, 0);
    }

    function installInteractionTracking() {
        document.addEventListener('click', function (event) {
            const target = event.target;
            if (!(target instanceof Element)) return;
            setInteractionContext(target.closest('button, input[type="submit"], input[type="button"], [role="button"]'));
        }, true);

        document.addEventListener('keydown', function (event) {
            if (event.key !== 'Enter' && event.key !== ' ') return;
            const target = event.target;
            if (!(target instanceof Element)) return;
            setInteractionContext(target.closest('button, input[type="submit"], [role="button"]'));
        }, true);

        document.addEventListener('submit', function (event) {
            const form = event.target;
            if (form && form.closest && form.closest('[data-progress-no-spinner="true"]')) return;
            const label = inferSpinnerTextFromButton(event.submitter)
                || inferSpinnerTextFromUrl(form && form.action, form && form.method);
            window.queueMicrotask(function () {
                // A later submit listener may cancel navigation and perform a tracked fetch instead.
                if (event.defaultPrevented) return;
                const finish = startOperation({ label });
                window.setTimeout(finish, FORM_FALLBACK_MS);
            });
        });

        window.addEventListener('pagehide', function () {
            operations.clear();
            syncSpinner();
        });
    }

    function installFetchTracking() {
        if (typeof window.fetch !== 'function') return;
        const originalFetch = window.fetch.bind(window);

        function keepOperationThroughResponseBody(response, finish) {
            if (!response || typeof response !== 'object') {
                finish();
                return response;
            }

            let bodyStarted = false;
            const unusedResponseTimer = window.setTimeout(finish, 0);
            const beginBody = function () {
                if (bodyStarted) return;
                bodyStarted = true;
                window.clearTimeout(unusedResponseTimer);
            };

            ['arrayBuffer', 'blob', 'formData', 'json', 'text'].forEach(function (methodName) {
                if (typeof response[methodName] !== 'function') return;
                const originalMethod = response[methodName].bind(response);
                try {
                    response[methodName] = function () {
                        beginBody();
                        let bodyResult;
                        try {
                            bodyResult = originalMethod.apply(response, arguments);
                        } catch (error) {
                            finish();
                            throw error;
                        }
                        return Promise.resolve(bodyResult).finally(finish);
                    };
                } catch (_) {}
            });

            const stream = response.body;
            if (stream && typeof stream.getReader === 'function') {
                const originalGetReader = stream.getReader.bind(stream);
                try {
                    stream.getReader = function () {
                        beginBody();
                        const reader = originalGetReader.apply(stream, arguments);
                        if (!reader || typeof reader.read !== 'function') return reader;
                        const originalRead = reader.read.bind(reader);
                        reader.read = function () {
                            let readResult;
                            try {
                                readResult = originalRead.apply(reader, arguments);
                            } catch (error) {
                                finish();
                                throw error;
                            }
                            return Promise.resolve(readResult).then(function (chunk) {
                                if (chunk && chunk.done) finish();
                                return chunk;
                            }, function (error) {
                                finish();
                                throw error;
                            });
                        };
                        if (typeof reader.cancel === 'function') {
                            const originalCancel = reader.cancel.bind(reader);
                            reader.cancel = function () {
                                return Promise.resolve(originalCancel.apply(reader, arguments)).finally(finish);
                            };
                        }
                        return reader;
                    };
                } catch (_) {}
            }
            return response;
        }

        window.fetch = function () {
            const details = getFetchDetails(arguments);
            const tracked = shouldTrackRequest(details.url, details.disabled);
            const label = activeInteractionLabel || inferSpinnerTextFromUrl(details.url, details.method);
            const finish = tracked ? startOperation({ label }) : null;
            let result;
            try {
                result = originalFetch.apply(window, arguments);
            } catch (error) {
                if (finish) finish();
                throw error;
            }
            if (!finish) return result;
            if (result && typeof result.then === 'function') {
                return result.then(function (response) {
                    return keepOperationThroughResponseBody(response, finish);
                }, function (error) {
                    finish();
                    throw error;
                });
            }
            finish();
            return result;
        };
    }

    function installXhrTracking() {
        if (typeof window.XMLHttpRequest !== 'function') return;
        const originalOpen = window.XMLHttpRequest.prototype.open;
        const originalSend = window.XMLHttpRequest.prototype.send;
        window.XMLHttpRequest.prototype.open = function (method, url) {
            this.__progressSpinnerMethod = method;
            this.__progressSpinnerUrl = url;
            return originalOpen.apply(this, arguments);
        };
        window.XMLHttpRequest.prototype.send = function () {
            const tracked = shouldTrackRequest(this.__progressSpinnerUrl, false);
            const label = activeInteractionLabel || inferSpinnerTextFromUrl(this.__progressSpinnerUrl, this.__progressSpinnerMethod);
            const finish = tracked ? startOperation({ label }) : null;
            if (finish) this.addEventListener('loadend', finish, { once: true });
            try {
                return originalSend.apply(this, arguments);
            } catch (error) {
                if (finish) finish();
                throw error;
            }
        };
    }

    function installPublicApi() {
        window.ProgressSpinner = Object.freeze({
            start: function (label) {
                return startOperation({ label: inferSpinnerTextFromText(label) || label });
            },
            track: function (promise, label) {
                const finish = startOperation({ label: inferSpinnerTextFromText(label) || label });
                return Promise.resolve(promise).finally(finish);
            },
            getActiveCount: function () {
                return operations.size;
            }
        });
    }

    function boot() {
        installPublicApi();
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
