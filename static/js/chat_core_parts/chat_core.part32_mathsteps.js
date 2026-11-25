        // ------------------------------------------------------------------
        // 数学ステップ解説（math steps）
        // 出力は「## ステップ N 見出し」で区切られた Markdown。1ステップずつ表示し、
        // 進捗はサーバー（/api/math_steps/progress）へ保存する。
        // ```mathsvg フェンスは画像だけを描き、コードUIは出さない。
        // ------------------------------------------------------------------
        const MATH_STEPS_HEADING_RE = /^\s{0,3}#{1,6}\s*(?:ステップ|Step)\s*(\d+)\s*(.*)$/i;
        const MATH_STEPS_DEFAULT_CONFIG = Object.freeze({
            basics: false,
            level: 'easy',
            step_size: 'normal',
            diagrams: 'auto',
            animation: true
        });
        const MATH_STEPS_PROGRESS_ROUTE = '/api/math_steps/progress';
        const MATH_STEPS_SIMPLIFY_ROUTE = '/api/math_steps/simplify';
        const MATH_STEPS_SVG_MAX_CHARS = 300000;
        const MATH_STEPS_MAX_EXPANDED_PER_STEP = 8;
        const MATH_STEPS_MAX_DISPLAY_STEPS = 50;
        const mathStepsUiState = {
            master: false,
            enabled: false,
            config: Object.assign({}, MATH_STEPS_DEFAULT_CONFIG),
            loaded: false,
            modalBusy: false
        };
        // A rendered step root keeps its parsed step text in memory; the DOM only
        // holds the current step, so a reload/re-render rebuilds from the source.
        const mathStepsRootState = new WeakMap();
        const mathStepsTextByRoot = new WeakMap();
        const mathStepsProgressTimers = new WeakMap();
        const mathStepsProgressInFlight = new WeakSet();

        function clampMathStepsIndex(value, max) {
            const n = Math.floor(Number(value));
            if (!Number.isFinite(n) || n < 0) return 0;
            const upper = Math.max(0, Number(max) || 0);
            return Math.min(n, upper);
        }

        function normalizeMathStepsConfig(value) {
            let raw = value;
            if (typeof raw === 'string') {
                try { raw = JSON.parse(raw); } catch (e) { raw = null; }
            }
            const src = (raw && typeof raw === 'object') ? raw : {};
            const level = ['easy', 'normal', 'detailed'].includes(String(src.level || '')) ? String(src.level) : MATH_STEPS_DEFAULT_CONFIG.level;
            const stepSize = ['small', 'normal', 'large'].includes(String(src.step_size || '')) ? String(src.step_size) : MATH_STEPS_DEFAULT_CONFIG.step_size;
            const diagrams = ['auto', 'off'].includes(String(src.diagrams || '')) ? String(src.diagrams) : MATH_STEPS_DEFAULT_CONFIG.diagrams;
            return {
                basics: src.basics === true,
                level,
                step_size: stepSize,
                diagrams,
                animation: src.animation !== false
            };
        }

        // 「もっとやさしく」 replaces an original step with smaller easier steps.
        // The map is keyed by the original step index; a sub-step can be split again.
        function normalizeMathStepsExpanded(value) {
            const expanded = {};
            if (!value || typeof value !== 'object' || Array.isArray(value)) return expanded;
            Object.keys(value).forEach((key) => {
                if (!/^\d{1,4}$/.test(String(key))) return;
                const list = value[key];
                if (!Array.isArray(list)) return;
                const parts = [];
                list.slice(0, MATH_STEPS_MAX_EXPANDED_PER_STEP).forEach((item) => {
                    if (!item || typeof item !== 'object') return;
                    const body = typeof item.body === 'string' ? item.body : '';
                    if (!body.trim()) return;
                    const title = typeof item.title === 'string' ? item.title : '';
                    parts.push({ title: title.trim(), body });
                });
                if (parts.length) expanded[String(key)] = parts;
            });
            return expanded;
        }

        function normalizeMathStepsMessageState(value) {
            let raw = value;
            if (typeof raw === 'string') {
                try { raw = JSON.parse(raw); } catch (e) { raw = null; }
            }
            const src = (raw && typeof raw === 'object' && !Array.isArray(raw)) ? raw : {};
            const simplified = {};
            if (src.simplified && typeof src.simplified === 'object' && !Array.isArray(src.simplified)) {
                Object.keys(src.simplified).forEach((key) => {
                    const text = src.simplified[key];
                    if (typeof text === 'string' && text.trim()) simplified[String(key)] = text;
                });
            }
            return {
                current: clampMathStepsIndex(src.current, 50),
                max_reached: clampMathStepsIndex(src.max_reached, 50),
                simplified,
                expanded: normalizeMathStepsExpanded(src.expanded)
            };
        }

        // Flattens the original steps plus any saved replacements into the list the
        // reader navigates. Display index and source index differ after a split.
        function buildMathStepsDisplaySteps(originalSteps, expanded) {
            const out = [];
            const source = Array.isArray(originalSteps) ? originalSteps : [];
            const map = (expanded && typeof expanded === 'object') ? expanded : {};
            source.forEach((step) => {
                const parts = map[String(step.index)];
                if (Array.isArray(parts) && parts.length) {
                    parts.forEach((part, partIndex) => {
                        out.push({
                            index: out.length,
                            sourceIndex: step.index,
                            partIndex,
                            expanded: true,
                            title: String(part.title || step.title || ''),
                            body: String(part.body || '')
                        });
                    });
                    return;
                }
                out.push({
                    index: out.length,
                    sourceIndex: step.index,
                    partIndex: null,
                    expanded: false,
                    title: step.title || '',
                    body: step.body || ''
                });
            });
            if (out.length > MATH_STEPS_MAX_DISPLAY_STEPS) out.length = MATH_STEPS_MAX_DISPLAY_STEPS;
            return out;
        }

        // Splits on step headings only outside fenced code blocks.
        function parseMathSteps(text) {
            const source = String(text || '').replace(/\r\n?/g, '\n');
            const steps = [];
            const preamble = [];
            let current = null;
            let fence = null;
            const lines = source.split('\n');
            for (const line of lines) {
                if (!fence) {
                    const open = line.match(/^\s{0,3}(`{3,}|~{3,})(.*)$/);
                    if (open && !(open[1][0] === '`' && open[2].includes('`'))) {
                        fence = open[1];
                        (current ? current.body : preamble).push(line);
                        continue;
                    }
                    const heading = line.match(MATH_STEPS_HEADING_RE);
                    if (heading) {
                        if (current) steps.push(current);
                        current = {
                            index: steps.length,
                            number: Number(heading[1]) || (steps.length + 1),
                            title: String(heading[2] || '').trim(),
                            body: []
                        };
                        continue;
                    }
                } else {
                    const trimmed = line.trim();
                    if (trimmed.length >= fence.length && trimmed[0] === fence[0] && /^(`+|~+)$/.test(trimmed)) {
                        fence = null;
                    }
                }
                (current ? current.body : preamble).push(line);
            }
            if (current) steps.push(current);
            const normalized = steps.map((step, index) => ({
                index,
                number: step.number,
                title: step.title,
                body: step.body.join('\n').replace(/^\s*\n+/, '').replace(/\n+\s*$/, '')
            }));
            const preambleText = preamble.join('\n').trim();
            if (normalized.length && preambleText) {
                // Do not lose a title/intro the model wrote before the first heading.
                normalized[0].body = `${preambleText}\n\n${normalized[0].body}`.trim();
            }
            return {
                steps: normalized,
                preamble: preambleText,
                hasSteps: normalized.length > 0
            };
        }
        window.parseMathSteps = parseMathSteps;

        function mathStepsReducedMotion() {            try {
                return !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
            } catch (e) {
                return false;
            }
        }

        // Only the current step is drawn; keep the parsed text on the root.
        function renderMathStepsBodyHtml(markdown) {
            const text = String(markdown || '').trim();
            if (!text) return '<div class="math-steps-empty">このステップの本文を準備しています…</div>';
            return sanitizeMarkdownHtml(text);
        }

        // A mathsvg fence becomes an image only: no code header, preview or download UI.
        // Scripts/foreignObject/external references are dropped, and animation is
        // stripped when the reader asked for reduced motion (or turned it off).
        function sanitizeMathSvgMarkup(code) {
            const raw = String(code || '').trim();
            if (!raw || raw.length > MATH_STEPS_SVG_MAX_CHARS) return '';
            if (!/^<svg[\s>]/i.test(raw.replace(/^(?:<\?xml[\s\S]*?\?>\s*|<!--[\s\S]*?-->\s*|<!DOCTYPE[^>]*>\s*)*/i, ''))) return '';
            let doc = null;
            try {
                doc = new DOMParser().parseFromString(raw, 'image/svg+xml');
            } catch (e) {
                return '';
            }
            if (!doc || !doc.documentElement) return '';
            if (doc.querySelector('parsererror')) return '';
            const root = doc.documentElement;
            if (String(root.nodeName || '').toLowerCase() !== 'svg') return '';
            const stripMotion = mathStepsReducedMotion() || mathStepsUiState.config.animation === false;
            const blockedTags = new Set(['script', 'foreignobject', 'iframe', 'object', 'embed', 'image', 'video', 'audio']);
            const motionTags = new Set(['animate', 'animatetransform', 'animatemotion', 'set', 'discard']);
            const walk = (node) => {
                if (!node || !node.childNodes) return;
                Array.from(node.childNodes).forEach((child) => {
                    if (child.nodeType === 1) {
                        const tag = String(child.nodeName || '').toLowerCase();
                        if (blockedTags.has(tag) || (stripMotion && motionTags.has(tag))) {
                            child.remove();
                            return;
                        }
                        if (tag === 'style') {
                            const css = child.textContent || '';
                            if (/@import|url\(\s*["']?\s*(?:https?:)?\/\//i.test(css)) {
                                child.remove();
                                return;
                            }
                            if (stripMotion && /@keyframes|animation/i.test(css)) {
                                child.remove();
                                return;
                            }
                        }
                        Array.from(child.attributes || []).forEach((attr) => {
                            const name = String(attr.name || '').toLowerCase();
                            const value = String(attr.value || '');
                            if (name.startsWith('on')) {
                                child.removeAttribute(attr.name);
                                return;
                            }
                            if ((name === 'href' || name === 'xlink:href') && !/^#/.test(value.trim())) {
                                child.removeAttribute(attr.name);
                            }
                            if (stripMotion && /animation(-name|-duration|-iteration-count)?$/.test(name)) {
                                child.removeAttribute(attr.name);
                            }
                            if (name === 'style' && /url\(\s*["']?\s*(?:https?:)?\/\//i.test(value)) {
                                child.removeAttribute(attr.name);
                            }
                        });
                        walk(child);
                    }
                });
            };
            walk(root);
            let serialized = '';
            try {
                serialized = new XMLSerializer().serializeToString(root);
            } catch (e) {
                return '';
            }
            if (!/^<svg[\s>]/i.test(serialized)) return '';
            return serialized;
        }

        function buildMathSvgRenderHtml(code) {
            const markup = sanitizeMathSvgMarkup(code);
            if (!markup) return '';
            let next = markup.replace(/<svg\b([^>]*)>/i, (tag, attrs) => {
                let a = attrs;
                if (!/\sxmlns\s*=/i.test(a)) a = ` xmlns="http://www.w3.org/2000/svg"${a}`;
                if (/\bxlink:/i.test(markup) && !/\sxmlns:xlink\s*=/i.test(a)) a = ` xmlns:xlink="http://www.w3.org/1999/xlink"${a}`;
                return `<svg${a}>`;
            });
            const src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(next)}`;
            const key = hashString(markup);
            return `<span class="mathsvg-render" data-svg-key="${escapeHtml(key)}"><img src="${src}" alt="図" loading="lazy" decoding="async"></span>`;
        }
        window.buildMathSvgRenderHtml = buildMathSvgRenderHtml;

        function mathStepsStateAttr(state) {
            try {
                return escapeHtml(JSON.stringify(state || {}));
            } catch (e) {
                return '';
            }
        }

        function mathStepsStepMarkdown(state, index) {
            const step = state.steps[index];
            if (!step) return '';
            if (step.expanded) return step.body;
            const simplified = state.simplified ? state.simplified[String(step.sourceIndex)] : '';
            const useOriginal = state.showOriginal && state.showOriginal.has(index);
            return (simplified && !useOriginal) ? simplified : step.body;
        }

        // An expanded step or a legacy simplified step can both be restored.
        function mathStepsCanRestore(state, index) {
            const step = state.steps[index];
            if (!step) return false;
            if (step.expanded) return true;
            const simplified = state.simplified ? state.simplified[String(step.sourceIndex)] : '';
            return !!(simplified && !state.showOriginal.has(index));
        }

        function mathStepsTotalLabel(state) {
            const total = state.steps.length;
            return `${state.current + 1} / ${total}${state.streaming ? '+' : ''}`;
        }

        function mathStepsPendingNext(state) {
            return !!state.streaming && state.current >= state.steps.length - 1;
        }

        function buildMathStepsHtml(text, opts = {}) {
            const parsed = parseMathSteps(text);
            if (!parsed.steps.length) return '';
            const incoming = normalizeMathStepsMessageState(opts.state || {});
            const originalSteps = parsed.steps;
            const steps = buildMathStepsDisplaySteps(originalSteps, incoming.expanded);
            const state = {
                originalSteps,
                expanded: Object.assign({}, incoming.expanded),
                steps,
                current: clampMathStepsIndex(opts.current !== undefined ? opts.current : incoming.current, steps.length - 1),
                maxReached: Math.max(incoming.max_reached, incoming.current),
                simplified: Object.assign({}, incoming.simplified),
                showOriginal: new Set(),
                streaming: !!opts.streaming
            };
            const step = state.steps[state.current];
            const markdown = mathStepsStepMarkdown(state, state.current);
            const simplifiedActive = mathStepsCanRestore(state, state.current);
            const pendingNext = mathStepsPendingNext(state);
            const isLast = state.current >= state.steps.length - 1;
            const nextLabel = (isLast && !pendingNext) ? '完了' : (pendingNext ? '次のステップを準備中…' : 'わかった、次へ');
            const nextDisabled = pendingNext || (isLast && !pendingNext);
            const messageId = opts.messageId !== undefined && opts.messageId !== null ? String(opts.messageId) : '';
            const jobId = opts.jobId !== undefined && opts.jobId !== null ? String(opts.jobId) : '';
            const attrs = [
                'data-math-steps-root="1"',
                `data-current="${state.current}"`,
                `data-max-reached="${state.maxReached}"`,
                `data-total="${state.steps.length}"`,
                `data-streaming="${state.streaming ? '1' : '0'}"`,
                messageId ? `data-message-id="${escapeHtml(messageId)}"` : '',
                jobId ? `data-job-id="${escapeHtml(jobId)}"` : '',
                `data-math-steps-state="${mathStepsStateAttr({ current: incoming.current, max_reached: incoming.max_reached, simplified: incoming.simplified, expanded: incoming.expanded })}"`
            ].filter(Boolean).join(' ');
            return `<div class="math-steps content-area" role="group" aria-label="数学ステップ解説" ${attrs}>`
                + `<div class="math-steps-header"><span class="math-steps-counter">ステップ ${mathStepsTotalLabel(state)}</span>`
                + `<span class="math-steps-title">${escapeHtml(step.title || '')}</span></div>`
                + `<div class="math-steps-body" data-step-index="${state.current}" aria-live="polite">`
                + `<div class="prose prose-invert text-sm break-words math-steps-prose">${renderMathStepsBodyHtml(markdown)}</div>`
                + (simplifiedActive ? '<button type="button" class="math-steps-restore" data-math-steps-action="restore">元の説明に戻す</button>' : '')
                + '</div>'
                + `<div class="math-steps-actions">`
                + `<button type="button" class="math-steps-next" data-math-steps-action="next"${nextDisabled ? ' disabled aria-disabled="true"' : ''}>${escapeHtml(nextLabel)}</button>`
                + `<div class="math-steps-sub-actions">`
                + `<button type="button" class="math-steps-prev" data-math-steps-action="prev"${state.current <= 0 ? ' disabled aria-disabled="true"' : ''}>前へ</button>`
                + `<button type="button" class="math-steps-restart" data-math-steps-action="restart"${state.current <= 0 ? ' disabled aria-disabled="true"' : ''}>最初から</button>`
                + `<button type="button" class="math-steps-simplify" data-math-steps-action="simplify"${state.streaming ? ' disabled aria-disabled="true"' : ''} title="このステップを、やさしい複数のステップに分けます"><i class="fas fa-wand-magic-sparkles" aria-hidden="true"></i>もっとやさしく</button>`
                + `</div></div>`
                + `<div class="math-steps-status" role="status" aria-live="polite"></div>`
                + `</div>`;
        }
        window.buildMathStepsHtml = buildMathStepsHtml;

        // Entry point used by buildAiMarkdownHtml (part21). Returns '' when the
        // answer has no step headings so the normal Markdown path is unchanged.
        function buildMathStepsBubbleHtml(text, opts = {}) {
            const parsed = parseMathSteps(text);
            if (!parsed.steps.length) return '';
            const mathSteps = opts.mathSteps || {};
            return buildMathStepsHtml(text, {
                state: mathSteps.state,
                messageId: mathSteps.messageId,
                jobId: mathSteps.jobId,
                streaming: !!mathSteps.streaming
            });
        }
        window.buildMathStepsBubbleHtml = buildMathStepsBubbleHtml;

        function mathStepsRootCounterEl(root) {
            return root ? root.querySelector('.math-steps-counter') : null;
        }

        function mathStepsStatus(root, message) {
            const el = root ? root.querySelector('.math-steps-status') : null;
            if (el) el.textContent = String(message || '');
        }

        function getMathStepsRootState(root) {
            if (!root) return null;
            let state = mathStepsRootState.get(root);
            if (state) return state;
            const messageId = root.getAttribute('data-message-id') || '';
            let text = mathStepsTextByRoot.get(root) || '';
            if (!text && messageId && typeof messageStore === 'object' && messageStore) {
                text = messageStore[messageId] || '';
            }
            const parsed = parseMathSteps(text);
            if (!parsed.steps.length) return null;
            const incoming = normalizeMathStepsMessageState(root.getAttribute('data-math-steps-state') || '');
            const originalSteps = parsed.steps;
            const steps = buildMathStepsDisplaySteps(originalSteps, incoming.expanded);
            state = {
                originalSteps,
                expanded: Object.assign({}, incoming.expanded),
                steps,
                current: clampMathStepsIndex(root.getAttribute('data-current'), steps.length - 1),
                maxReached: Math.max(incoming.max_reached, clampMathStepsIndex(root.getAttribute('data-max-reached'), steps.length - 1)),
                simplified: Object.assign({}, incoming.simplified),
                showOriginal: new Set(),
                streaming: root.getAttribute('data-streaming') === '1'
            };
            mathStepsRootState.set(root, state);
            return state;
        }

        // Re-renders only the current step body (plus header/buttons).
        function paintMathStepsRoot(root) {
            const state = getMathStepsRootState(root);
            if (!root || !state || !state.steps.length) return;
            state.current = clampMathStepsIndex(state.current, state.steps.length - 1);
            const step = state.steps[state.current];
            const counter = mathStepsRootCounterEl(root);
            if (counter) counter.textContent = `ステップ ${mathStepsTotalLabel(state)}`;
            const title = root.querySelector('.math-steps-title');
            if (title) title.textContent = step.title || '';
            const body = root.querySelector('.math-steps-body');
            const markdown = mathStepsStepMarkdown(state, state.current);
            const simplifiedActive = mathStepsCanRestore(state, state.current);
            if (body) {
                const oldSvg = new Map();
                body.querySelectorAll('.mathsvg-render[data-svg-key]').forEach((el) => {
                    const key = el.getAttribute('data-svg-key');
                    if (key && !oldSvg.has(key)) oldSvg.set(key, el);
                });
                body.setAttribute('data-step-index', String(state.current));
                body.innerHTML = `<div class="prose prose-invert text-sm break-words math-steps-prose">${renderMathStepsBodyHtml(markdown)}</div>`
                    + (simplifiedActive ? '<button type="button" class="math-steps-restore" data-math-steps-action="restore">元の説明に戻す</button>' : '');
                body.querySelectorAll('.mathsvg-render[data-svg-key]').forEach((fresh) => {
                    const old = oldSvg.get(fresh.getAttribute('data-svg-key'));
                    if (old) fresh.replaceWith(old);
                });
                try { queueMessageDecorations(body, markdown); } catch (e) {}
            }
            const pendingNext = mathStepsPendingNext(state);
            const isLast = state.current >= state.steps.length - 1;
            const next = root.querySelector('.math-steps-next');
            if (next) {
                next.textContent = (isLast && !pendingNext) ? '完了' : (pendingNext ? '次のステップを準備中…' : 'わかった、次へ');
                const disabled = pendingNext || (isLast && !pendingNext);
                next.disabled = disabled;
                next.setAttribute('aria-disabled', disabled ? 'true' : 'false');
            }
            const prev = root.querySelector('.math-steps-prev');
            if (prev) {
                const disabled = state.current <= 0;
                prev.disabled = disabled;
                prev.setAttribute('aria-disabled', disabled ? 'true' : 'false');
            }
            const restart = root.querySelector('.math-steps-restart');
            if (restart) {
                const disabled = state.current <= 0;
                restart.disabled = disabled;
                restart.setAttribute('aria-disabled', disabled ? 'true' : 'false');
            }
            const simplify = root.querySelector('.math-steps-simplify');
            if (simplify && !simplify.dataset.busy) {
                simplify.disabled = !!state.streaming;
                simplify.setAttribute('aria-disabled', state.streaming ? 'true' : 'false');
            }
            root.setAttribute('data-current', String(state.current));
            root.setAttribute('data-max-reached', String(state.maxReached));
            root.setAttribute('data-total', String(state.steps.length));
            root.setAttribute('data-streaming', state.streaming ? '1' : '0');
            state.renderedStep = state.current;
            state.renderedMarkdown = markdown;
        }

        function renderMathStepsInto(container, text, opts = {}) {
            if (!container) return false;
            const parsed = parseMathSteps(text);
            if (!parsed.steps.length) return false;
            const previousRoot = container.querySelector('.math-steps');
            const previousState = previousRoot ? getMathStepsRootState(previousRoot) : null;
            const incoming = normalizeMathStepsMessageState(opts.state || {});
            const state = previousState || {
                originalSteps: parsed.steps,
                expanded: Object.assign({}, incoming.expanded),
                steps: buildMathStepsDisplaySteps(parsed.steps, incoming.expanded),
                current: incoming.current,
                maxReached: incoming.max_reached,
                simplified: Object.assign({}, incoming.simplified),
                showOriginal: new Set(),
                streaming: !!opts.streaming
            };
            state.originalSteps = parsed.steps;
            state.expanded = state.expanded || {};
            state.steps = buildMathStepsDisplaySteps(state.originalSteps, state.expanded);
            state.streaming = opts.streaming === true;
            state.current = clampMathStepsIndex(state.current, state.steps.length - 1);
            state.maxReached = Math.max(state.maxReached, state.current);
            const previousMarkdown = previousRoot && previousState && previousState.renderedStep === state.current
                ? previousState.renderedMarkdown : null;
            const markdown = mathStepsStepMarkdown(state, state.current);
            const previousBody = previousRoot ? previousRoot.querySelector('.math-steps-body') : null;
            const keepBody = !!previousBody && previousMarkdown !== null && previousMarkdown === markdown;
            const html = buildMathStepsHtml(text, {
                state: { current: state.current, max_reached: state.maxReached, simplified: state.simplified, expanded: state.expanded },
                messageId: opts.mathSteps && opts.mathSteps.messageId !== undefined ? opts.mathSteps.messageId : undefined,
                jobId: opts.mathSteps && opts.mathSteps.jobId !== undefined ? opts.mathSteps.jobId : undefined,
                streaming: state.streaming
            });
            if (!html) return false;
            container.innerHTML = html;
            const root = container.querySelector('.math-steps');
            if (!root) return false;
            mathStepsRootState.set(root, state);
            mathStepsTextByRoot.set(root, String(text || ''));
            state.renderedStep = state.current;
            state.renderedMarkdown = markdown;
            const body = root.querySelector('.math-steps-body');
            if (keepBody && previousBody && body) {
                const oldSvg = new Map();
                previousBody.querySelectorAll('.mathsvg-render[data-svg-key]').forEach((el) => {
                    const key = el.getAttribute('data-svg-key');
                    if (key && !oldSvg.has(key)) oldSvg.set(key, el);
                });
                body.querySelectorAll('.mathsvg-render[data-svg-key]').forEach((fresh) => {
                    const old = oldSvg.get(fresh.getAttribute('data-svg-key'));
                    if (old) fresh.replaceWith(old);
                });
                body.replaceWith(previousBody);
            } else if (body) {
                try { queueMessageDecorations(body, markdown); } catch (e) {}
            }
            return true;
        }
        window.renderMathStepsInto = renderMathStepsInto;

        // ------------------------------------------------------------------
        // 進捗の保存（楽観更新 + デバウンス + 失敗時は静かに1回だけ再試行）
        // ------------------------------------------------------------------
        function mathStepsRootTargets(root) {
            if (!root) return null;
            const messageId = root.getAttribute('data-message-id');
            const jobId = root.getAttribute('data-job-id');
            if (messageId && /^\d+$/.test(String(messageId))) return { message_id: Number(messageId) };
            if (jobId) return { job_id: String(jobId) };
            return null;
        }

        function postMathStepsProgress(root, attempt = 0) {
            const state = getMathStepsRootState(root);
            const target = mathStepsRootTargets(root);
            if (!state || !target) return;
            if (mathStepsProgressInFlight.has(root)) return;
            const payload = Object.assign({}, target, {
                current: state.current,
                max_reached: state.maxReached
            });
            mathStepsProgressInFlight.add(root);
            return apiFetch(MATH_STEPS_PROGRESS_ROUTE, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            }).then((res) => {
                if (res && res.ok) return;
                throw new Error(`progress ${res ? res.status : 'failed'}`);
            }).catch(() => {
                // Progress is best-effort; a lost update must not disturb the reader.
                // A streaming answer may not have a server message id yet, so retry
                // with a short backoff until the save completes.
                if (attempt < 4) {
                    window.setTimeout(() => postMathStepsProgress(root, attempt + 1), 1200 * (attempt + 1));
                }
            }).finally(() => {
                mathStepsProgressInFlight.delete(root);
            });
        }

        // Browser-fast has no job id; once the save returns its assistant message id,
        // bind it so the current reading position can be persisted.
        function bindMathStepsMessageId(root, messageId) {
            if (!root || messageId === undefined || messageId === null) return;
            root.setAttribute('data-message-id', String(messageId));
            return postMathStepsProgress(root);
        }
        window.bindMathStepsMessageId = bindMathStepsMessageId;

        function scheduleMathStepsProgress(root) {
            if (!root) return;
            const timer = mathStepsProgressTimers.get(root);
            if (timer) window.clearTimeout(timer);
            mathStepsProgressTimers.set(root, window.setTimeout(() => {
                mathStepsProgressTimers.delete(root);
                postMathStepsProgress(root);
            }, 450));
        }

        // The reader preferred the original explanation: forget the saved replacement
        // (legacy simplified text or an expanded step) on the server as well.
        function forgetMathStepsVariants(root, stepIndex) {
            const target = mathStepsRootTargets(root);
            const state = getMathStepsRootState(root);
            if (!target || !state) return;
            apiFetch(MATH_STEPS_PROGRESS_ROUTE, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(Object.assign({}, target, {
                    current: state.current,
                    max_reached: state.maxReached,
                    remove_simplified: [stepIndex],
                    remove_expanded: [stepIndex]
                }))
            }).catch(() => {});
        }

        // ------------------------------------------------------------------
        // 「もっとやさしく」（今のステップを、やさしい複数ステップに分ける）
        // ------------------------------------------------------------------
        function requestMathStepsSimplify(root, button) {
            const state = getMathStepsRootState(root);
            const target = mathStepsRootTargets(root);
            if (!state || !target) return;
            if (!target.message_id) {
                showToast('回答が保存された後に「もっとやさしく」を使えます', 'info', false);
                return;
            }
            if (state.streaming) {
                showToast('回答の生成が終わってから使えます', 'info', false);
                return;
            }
            const current = state.steps[state.current];
            if (!current) return;
            const payload = { message_id: target.message_id, step_index: current.sourceIndex };
            if (current.partIndex !== null && current.partIndex !== undefined) {
                payload.part_index = current.partIndex;
            }
            if (button) {
                button.dataset.busy = '1';
                button.disabled = true;
                button.innerHTML = '<i class="fas fa-circle-notch fa-spin" aria-hidden="true"></i>やさしくしています…';
            }
            mathStepsStatus(root, 'このステップを、やさしい複数のステップに分けています…');
            apiFetch(MATH_STEPS_SIMPLIFY_ROUTE, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            }).then(async (res) => {
                const data = await res.json().catch(() => ({}));
                let rawParts = (data && Array.isArray(data.steps)) ? data.steps : [];
                // A cached older server only returns `text`; keep the button usable
                // during the deploy transition by treating it as one easier step.
                if (!rawParts.length && data && typeof data.text === 'string' && data.text.trim()) {
                    rawParts = [{ title: current.title, body: data.text }];
                }
                if (!res.ok || !data || data.status !== 'ok' || !rawParts.length) {
                    const reason = res.status === 409
                        ? 'これ以上ステップを増やせません'
                        : (data && data.error ? data.error : 'やさしい説明を作れませんでした');
                    throw new Error(reason);
                }
                const parts = normalizeMathStepsExpanded({ [String(current.sourceIndex)]: rawParts })[String(current.sourceIndex)];
                if (!parts || !parts.length) throw new Error('やさしい説明を作れませんでした');
                const nextState = normalizeMathStepsMessageState(data.math_steps_state || {});
                state.simplified = Object.assign({}, nextState.simplified);
                state.expanded = Object.assign({}, state.expanded, nextState.expanded, { [String(current.sourceIndex)]: parts });
                state.steps = buildMathStepsDisplaySteps(state.originalSteps, state.expanded);
                state.current = clampMathStepsIndex(state.current, state.steps.length - 1);
                state.showOriginal.delete(state.current);
                if (button) {
                    delete button.dataset.busy;
                    button.disabled = false;
                    button.innerHTML = '<i class="fas fa-wand-magic-sparkles" aria-hidden="true"></i>もっとやさしく';
                }
                mathStepsStatus(root, `やさしい説明に分けました（${parts.length}ステップ）`);
                paintMathStepsRoot(root);
            }).catch((error) => {
                mathStepsStatus(root, '');
                showToast(error && error.message ? error.message : 'やさしい説明を作れませんでした', 'error', true);
                if (button) {
                    delete button.dataset.busy;
                    button.disabled = false;
                    button.innerHTML = '<i class="fas fa-wand-magic-sparkles" aria-hidden="true"></i>もっとやさしく';
                }
            });
        }

        // ------------------------------------------------------------------
        // ステップ操作（イベント委譲）
        // ------------------------------------------------------------------
        function handleMathStepsAction(event) {
            const button = event.target && event.target.closest ? event.target.closest('[data-math-steps-action]') : null;
            if (!button) return;
            const root = button.closest('.math-steps');
            if (!root) return;
            const state = getMathStepsRootState(root);
            if (!state) return;
            const action = button.getAttribute('data-math-steps-action');
            if (action === 'simplify') {
                event.preventDefault();
                requestMathStepsSimplify(root, button);
                return;
            }
            event.preventDefault();
            if (action === 'restore') {
                const current = state.steps[state.current];
                if (!current) return;
                if (current.expanded) {
                    // Collapse the whole split back to the original step and stay on it.
                    const first = state.steps.findIndex((step) => step.sourceIndex === current.sourceIndex);
                    delete state.expanded[String(current.sourceIndex)];
                    delete state.simplified[String(current.sourceIndex)];
                    state.steps = buildMathStepsDisplaySteps(state.originalSteps, state.expanded);
                    state.current = clampMathStepsIndex(first >= 0 ? first : 0, state.steps.length - 1);
                    state.maxReached = clampMathStepsIndex(state.maxReached, state.steps.length - 1);
                    state.showOriginal.delete(state.current);
                } else {
                    state.showOriginal.add(state.current);
                    delete state.simplified[String(current.sourceIndex)];
                }
                forgetMathStepsVariants(root, current.sourceIndex);
                mathStepsStatus(root, '元の説明に戻しました');
                scheduleMathStepsProgress(root);
            } else if (action === 'next') {
                if (state.current >= state.steps.length - 1) return;
                state.current = clampMathStepsIndex(state.current + 1, state.steps.length - 1);
                state.maxReached = Math.max(state.maxReached, state.current);
                state.showOriginal.delete(state.current);
                mathStepsStatus(root, `ステップ ${state.current + 1} を表示しています`);
                scheduleMathStepsProgress(root);
            } else if (action === 'prev') {
                if (state.current <= 0) return;
                state.current = clampMathStepsIndex(state.current - 1, state.steps.length - 1);
                state.showOriginal.delete(state.current);
                mathStepsStatus(root, `ステップ ${state.current + 1} に戻りました`);
                scheduleMathStepsProgress(root);
            } else if (action === 'restart') {
                state.current = 0;
                state.showOriginal.clear();
                mathStepsStatus(root, '最初のステップに戻りました');
                scheduleMathStepsProgress(root);
            } else {
                return;
            }
            paintMathStepsRoot(root);
        }

        // ------------------------------------------------------------------
        // 設定（マスター / チャット既定値 / 歯車モーダル）
        // ------------------------------------------------------------------
        function mathStepsChip() {
            return get('enable-math-steps');
        }

        function setMathStepsEnabled(enabled, options = {}) {
            const next = !!enabled && mathStepsUiState.master;
            mathStepsUiState.enabled = next;
            const chip = mathStepsChip();
            if (chip && chip.checked !== next) chip.checked = next;
            return next;
        }
        window.setMathStepsEnabled = setMathStepsEnabled;

        function applyMathStepsMaster(enabled) {
            mathStepsUiState.master = !!enabled;
            const container = get('math-steps-container');
            if (container) container.classList.toggle('hidden', !mathStepsUiState.master);
            if (!mathStepsUiState.master) setMathStepsEnabled(false, { persist: false });
        }

        function applyMathStepsSettings(data) {
            if (!data || typeof data !== 'object') return;
            mathStepsUiState.config = normalizeMathStepsConfig(
                Object.prototype.hasOwnProperty.call(data, 'math_steps_config')
                    ? data.math_steps_config
                    : mathStepsUiState.config
            );
            if (Object.prototype.hasOwnProperty.call(data, 'math_steps_enabled')) {
                applyMathStepsMaster(data.math_steps_enabled === true);
            }
            mathStepsUiState.loaded = true;
        }
        window.applyMathStepsSettings = applyMathStepsSettings;

        function applyMathStepsChatDefaults(data, useLast) {
            if (!data || typeof data !== 'object') return;
            applyMathStepsSettings(data);
            if (!mathStepsUiState.master) {
                setMathStepsEnabled(false, { persist: false });
                return;
            }
            const lastVal = Object.prototype.hasOwnProperty.call(data, 'last_math_steps') ? data.last_math_steps : undefined;
            const defaultValue = Object.prototype.hasOwnProperty.call(data, 'default_math_steps') ? data.default_math_steps : false;
            if (useLast && lastVal !== undefined && lastVal !== null) {
                setMathStepsEnabled(lastVal === true, { persist: false });
                return;
            }
            setMathStepsEnabled(defaultValue === true, { persist: false });
        }
        window.applyMathStepsChatDefaults = applyMathStepsChatDefaults;

        function collectMathStepsSendOption() {
            return !!(mathStepsUiState.master && mathStepsUiState.enabled);
        }
        window.collectMathStepsSendOption = collectMathStepsSendOption;

        function openMathStepsModal() {
            const modal = get('math-steps-config-modal');
            if (!modal) return;
            const config = normalizeMathStepsConfig(mathStepsUiState.config);
            if (get('math-steps-cfg-basics')) get('math-steps-cfg-basics').checked = config.basics;
            if (get('math-steps-cfg-level')) get('math-steps-cfg-level').value = config.level;
            if (get('math-steps-cfg-step-size')) get('math-steps-cfg-step-size').value = config.step_size;
            if (get('math-steps-cfg-diagrams')) get('math-steps-cfg-diagrams').value = config.diagrams;
            if (get('math-steps-cfg-animation')) get('math-steps-cfg-animation').checked = config.animation;
            const status = get('math-steps-config-status');
            if (status) status.textContent = '';
            if (typeof showModal === 'function') showModal('math-steps-config-modal');
            else modal.classList.remove('hidden');
            if (location.pathname !== '/math-steps') {
                try { history.pushState({ modal: 'math-steps' }, '', '/math-steps'); } catch (e) {}
            }
        }
        window.openMathStepsModal = openMathStepsModal;

        function closeMathStepsModal(skipHistory = false) {
            const modal = get('math-steps-config-modal');
            if (modal) {
                if (typeof hideModal === 'function') hideModal('math-steps-config-modal');
                else modal.classList.add('hidden');
            }
            if (!skipHistory && location.pathname === '/math-steps') {
                try { history.back(); } catch (e) {}
            }
        }
        window.closeMathStepsModal = closeMathStepsModal;

        async function saveMathStepsConfig() {
            if (mathStepsUiState.modalBusy) return;
            const config = normalizeMathStepsConfig({
                basics: !!(get('math-steps-cfg-basics') && get('math-steps-cfg-basics').checked),
                level: get('math-steps-cfg-level') ? get('math-steps-cfg-level').value : undefined,
                step_size: get('math-steps-cfg-step-size') ? get('math-steps-cfg-step-size').value : undefined,
                diagrams: get('math-steps-cfg-diagrams') ? get('math-steps-cfg-diagrams').value : undefined,
                animation: !!(get('math-steps-cfg-animation') && get('math-steps-cfg-animation').checked)
            });
            mathStepsUiState.modalBusy = true;
            const saveBtn = get('math-steps-config-save');
            if (saveBtn) saveBtn.disabled = true;
            const status = get('math-steps-config-status');
            if (status) status.textContent = '保存しています…';
            try {
                const res = await apiFetch(CHAT_CONFIG.urls.handleSettings, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ math_steps_config: config })
                });
                if (!res.ok) throw new Error('設定を保存できませんでした');
                mathStepsUiState.config = config;
                if (typeof userSettingsSnapshot === 'object' && userSettingsSnapshot) {
                    userSettingsSnapshot.math_steps_config = config;
                }
                // Browser-fast gets its system prompt from the bootstrap response.
                try { browserFastBootstrap = null; } catch (e) {}
                if (status) status.textContent = '保存しました';
                showToast('数学ステップの設定を保存しました', 'success', false);
                closeMathStepsModal();
            } catch (error) {
                if (status) status.textContent = '';
                showToast(error && error.message ? error.message : '設定を保存できませんでした', 'error', true);
            } finally {
                mathStepsUiState.modalBusy = false;
                if (saveBtn) saveBtn.disabled = false;
            }
        }
        window.saveMathStepsConfig = saveMathStepsConfig;

        // ------------------------------------------------------------------
        // PDF / コピー
        // ------------------------------------------------------------------
        function buildMathStepsPdfHtml(text, stateValue) {
            const parsed = parseMathSteps(text);
            if (!parsed.steps.length) return '';
            const state = normalizeMathStepsMessageState(stateValue);
            const steps = buildMathStepsDisplaySteps(parsed.steps, state.expanded);
            return steps.map((step) => {
                const simplified = step.expanded ? '' : state.simplified[String(step.sourceIndex)];
                const body = step.expanded ? step.body : ((simplified && simplified.trim()) ? simplified : step.body);
                return `<div class="math-steps-pdf-step">`
                    + `<h3 class="math-steps-pdf-heading">ステップ ${step.index + 1}${step.title ? ` ${pdfEscapeAttr(step.title)}` : ''}</h3>`
                    + `<div class="math-steps-pdf-body">${sanitizeMarkdownHtml(body)}</div>`
                    + ((step.expanded || (simplified && simplified.trim())) ? '<div class="math-steps-pdf-note">（やさしい説明）</div>' : '')
                    + `</div>`;
            }).join('');
        }
        window.buildMathStepsPdfHtml = buildMathStepsPdfHtml;

        function prepareMathStepsCopyText(text) {
            const source = String(text || '');
            if (!source) return source;
            const withoutMathSvg = (value) => String(value || '').replace(
                /(`{3,}|~{3,})mathsvg[^\n]*\n[\s\S]*?\n\1[ \t]*(?=\n|$)/gi,
                '［図］\n'
            );
            const parsed = parseMathSteps(source);
            if (!parsed.steps.length) return withoutMathSvg(source);
            const lines = [];
            parsed.steps.forEach((step) => {
                const title = step.title ? ` ${step.title}` : '';
                lines.push(`## ステップ ${step.index + 1}${title}`);
                lines.push('');
                lines.push(withoutMathSvg(step.body).trim());
                lines.push('');
            });
            return lines.join('\n').replace(/\n{3,}/g, '\n\n').trim();
        }
        window.prepareMathStepsCopyText = prepareMathStepsCopyText;

        // ------------------------------------------------------------------
        // 初期化
        // ------------------------------------------------------------------
        function initMathSteps() {
            const chip = mathStepsChip();
            if (chip) {
                chip.addEventListener('change', () => {
                    const turningOn = !!chip.checked;
                    if (turningOn) {
                        const turnedOff = [];
                        if (typeof canvasModeEnabled !== 'undefined' && canvasModeEnabled) {
                            try { syncCanvasModeUi(false); } catch (e) {}
                            turnedOff.push('Canvas');
                        }
                        if (typeof codingModeEnabled !== 'undefined' && codingModeEnabled) {
                            try { syncCodingModeUi(false); } catch (e) {}
                            turnedOff.push('Coding');
                        }
                        if (turnedOff.length) showToast(`${turnedOff.join('と')}をオフにしました`, 'info', false);
                    }
                    setMathStepsEnabled(turningOn, { persist: true });
                });
            }
            ['enable-canvas-mode', 'enable-coding-mode'].forEach((id) => {
                const el = get(id);
                if (!el) return;
                el.addEventListener('change', () => {
                    if (!el.checked || !mathStepsUiState.enabled) return;
                    setMathStepsEnabled(false, { persist: false });
                    showToast('数学ステップをオフにしました', 'info', false);
                });
            });
            document.addEventListener('click', (event) => {
                try { handleMathStepsAction(event); } catch (e) { console.error(e); }
            });
            ensureUserSettingsSnapshot().then((data) => {
                if (!data) return;
                applyMathStepsSettings(data);
                if (!mathStepsUiState.master) return;
                const useLast = !!data.use_last_chat_settings;
                if (useLast) {
                    const lastVal = Object.prototype.hasOwnProperty.call(data, 'last_math_steps') ? data.last_math_steps : null;
                    if (lastVal !== null && lastVal !== undefined) {
                        setMathStepsEnabled(lastVal === true, { persist: false });
                        return;
                    }
                }
                setMathStepsEnabled(data.default_math_steps === true, { persist: false });
            }).catch(() => {});
            document.addEventListener('keydown', (event) => {
                if (event.key === 'Escape' && get('math-steps-config-modal')?.classList.contains('modal-open')) {
                    closeMathStepsModal();
                }
            });
        }
        window.initMathSteps = initMathSteps;
        document.addEventListener('DOMContentLoaded', () => {
            try { initMathSteps(); } catch (e) { console.error('math steps init failed', e); }
        }, { once: true });
