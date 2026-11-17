
        function estimateConversationCost(messages, pricing = costPricing, context = messages) {
            const result = { low: 0, high: 0, priced: 0, unpriced: 0, inferred: 0, long: 0, requests: 0 };
            const rows = new Map((context || []).filter(Boolean).map(m => [String(m.id), m]));
            const seen = new Set();
            for (const m of messages || []) {
                if (!m || m.role !== 'assistant' || seen.has(String(m.id))) continue;
                seen.add(String(m.id));
                let metadata = null;
                try { metadata = typeof m.thought_data === 'string' ? JSON.parse(m.thought_data) : m.thought_data; } catch (_) { /* Legacy reasoning. */ }
                const date = /^\d{4}-\d{2}-\d{2}/.exec(m.created_at || '')?.[0] || pricing?.checked_at;
                const batch = m.batch_job?.state === 'JOB_STATE_SUCCEEDED';
                const promptOptions = (rows.get(String(m.parent_id)) || m).prompt_options || m.prompt_options || {};
                const flex = promptOptions && promptOptions.gemini_flex === true;
                if (Array.isArray(metadata?.pricing_calls)) {
                    if (!metadata.pricing_calls.length) result.priced++; // All requests explicitly rejected before generation.
                    for (const call of metadata.pricing_calls) {
                        if (!call || typeof call !== 'object') { result.unpriced++; continue; }
                        const callDate = /^\d{4}-\d{2}-\d{2}/.exec(call.created_at || '')?.[0] || date;
                        // Older image call records predate speed capture; recover only from their saved prompt.
                        const rate = costRate(pricing, call.model, callDate);
                        const request = rate?.image_selector === 'speed' && !call.speed && call.model === m.model
                            ? { ...call, speed: legacyImageRequest(m, rows).speed } : call;
                        addRequestCost(result, request, pricing, callDate, batch, flex);
                    }
                    continue;
                }
                if (costRate(pricing, m.model, date)?.unit === 'image') {
                    addRequestCost(result, legacyImageRequest(m, rows), pricing, date, false, flex);
                    continue;
                }
                const out = costTokenNumber(m.tokens_out);
                if (out === null && !m.content && !m.tokens) continue;
                if (out === null) { result.unpriced++; continue; }
                let input = costTokenNumber(metadata?.pricing_input_tokens) ?? costTokenNumber(m.tokens_in);
                let incomplete = false;
                if (input === null) {
                    result.inferred++;
                    input = 0;
                    let parent = m.parent_id;
                    const ancestors = new Set([String(m.id)]);
                    if (parent == null) incomplete = true;
                    while (parent != null) {
                        const id = String(parent);
                        if (ancestors.has(id) || !rows.has(id)) { incomplete = true; break; }
                        ancestors.add(id);
                        const ancestor = rows.get(id);
                        const count = ancestor.role === 'user' ? costTokenNumber(ancestor.tokens_in)
                            : costTokenNumber(ancestor.tokens_content) ?? (costTokenNumber(ancestor.tokens_out) === null ? null
                                : Math.max(0, Number(ancestor.tokens_out) - (costTokenNumber(ancestor.tokens_thought) || 0)));
                        if (count === null) incomplete = true;
                        else input += count;
                        parent = ancestor.parent_id;
                    }
                }
                addRequestCost(result, { model: m.model, input_tokens: input, output_tokens: out, complete: !incomplete }, pricing, date, batch, flex);
            }
            return result;
        }

        function costEstimateLabel(cost, prefix = '費用予想') {
            if (!cost || !cost.priced) return `${prefix}: 算出不可`;
            const money = n => n > 0 && n < 0.000001 ? '<$0.000001' : `$${n.toFixed(6)}`;
            const amount = cost.lowerBound ? `${money(cost.low)}以上`
                : cost.high - cost.low > 1e-12 ? `${money(cost.low)}〜${money(cost.high)}` : money(cost.low);
            return `${prefix}: ${amount}${cost.unpriced ? '（一部未算入）' : ''}`;
        }

        // USD→JPY conversion for the token detail. Free, keyless public reference data
        // (ECB rates via Frankfurter, then a public CDN mirror as a fallback), cached on the
        // device. Only the rate is fetched; no chat, usage or account data is sent.
        const COST_JPY_RATE_STORAGE_KEY = 'cost_jpy_rate_v1';
        const COST_JPY_RATE_TTL_MS = 12 * 60 * 60 * 1000;
        const COST_JPY_RATE_NOTE = '無料の公開為替レート（欧州中央銀行の参考レートなど）でドルから円へ換算した概算です。実際の請求時のレートとは異なります。';
        const COST_JPY_RATE_SOURCES = [
            {
                name: 'ECB',
                url: 'https://api.frankfurter.dev/v2/rates?base=USD&quotes=JPY',
                read: data => {
                    const row = Array.isArray(data) ? data.find(item => item && item.quote === 'JPY') : null;
                    return { rate: Number(row?.rate), date: row?.date };
                }
            },
            {
                name: 'currency-api',
                url: 'https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json',
                read: data => ({ rate: Number(data?.usd?.jpy), date: data?.date })
            }
        ];
        let costJpyRate = null;
        let costJpyRateFailed = false;
        let costJpyRateFailedAt = 0;
        let costJpyRatePromise = null;

        function normalizeJpyRate(value) {
            const rate = Number(value?.rate);
            if (!Number.isFinite(rate) || rate < 50 || rate > 500) return null;
            return {
                rate,
                date: typeof value.date === 'string' ? value.date.slice(0, 10) : '',
                source: value.source || ''
            };
        }

        function readCachedJpyRate() {
            try {
                const saved = JSON.parse(localStorage.getItem(COST_JPY_RATE_STORAGE_KEY) || 'null');
                if (!saved || !(Date.now() - Number(saved.saved) < COST_JPY_RATE_TTL_MS)) return null;
                return normalizeJpyRate(saved);
            } catch (_) { return null; }
        }

        function loadJpyRate() {
            if (costJpyRate) return Promise.resolve(costJpyRate);
            if (costJpyRateFailed && Date.now() - costJpyRateFailedAt < 60000) return Promise.resolve(null);
            if (costJpyRatePromise) return costJpyRatePromise;
            costJpyRateFailed = false;
            const cached = readCachedJpyRate();
            if (cached) {
                costJpyRate = cached;
                return Promise.resolve(cached);
            }
            costJpyRatePromise = (async () => {
                for (const source of COST_JPY_RATE_SOURCES) {
                    let timer = null;
                    try {
                        const controller = new AbortController();
                        timer = setTimeout(() => controller.abort(), 6000);
                        const response = await fetch(source.url, { signal: controller.signal, credentials: 'omit' });
                        if (!response.ok) continue;
                        const rate = normalizeJpyRate({ ...source.read(await response.json()), source: source.name });
                        if (!rate) continue;
                        costJpyRate = rate;
                        try {
                            localStorage.setItem(COST_JPY_RATE_STORAGE_KEY, JSON.stringify({ ...rate, saved: Date.now() }));
                        } catch (_) { /* Storage can be unavailable. */ }
                        return rate;
                    } catch (_) {
                        /* Try the next free source. */
                    } finally {
                        clearTimeout(timer);
                    }
                }
                costJpyRateFailed = true;
                costJpyRateFailedAt = Date.now();
                return null;
            })().finally(() => { costJpyRatePromise = null; });
            return costJpyRatePromise;
        }

        function jpyMoneyText(amount) {
            if (!(amount > 0)) return '0円';
            if (amount < 0.01) return '0.01円未満';
            if (amount < 1000) return `${amount.toFixed(2)}円`;
            return `${Math.round(amount).toLocaleString('ja-JP')}円`;
        }

        function costEstimateJpyLabel(cost, rate) {
            if (!cost || !cost.priced || !rate) return '日本円換算: 算出不可';
            const low = jpyMoneyText(cost.low * rate.rate);
            const amount = cost.lowerBound ? (low.endsWith('未満') ? low : `${low}以上`)
                : cost.high - cost.low > 1e-12 ? `${low}〜${jpyMoneyText(cost.high * rate.rate)}` : low;
            const unpriced = cost.unpriced ? '（一部未算入）' : '';
            const date = rate.date ? `・${rate.date}` : '';
            return `日本円換算: 約${amount}${unpriced}（参考レート 1ドル=${rate.rate.toFixed(2)}円${date}）`;
        }

        // Fills the token-detail line under the USD estimate, fetching the rate on first use.
        function fillCostEstimateJpy(el, cost) {
            if (!el) return;
            if (!cost || !cost.priced) {
                el.textContent = `日本円換算: ${!costPricing && !costPricingFailed ? '換算中…' : '算出不可'}`;
                return;
            }
            el.title = COST_JPY_RATE_NOTE;
            el.textContent = '日本円換算: 換算中…';
            if (costJpyRate) {
                el.textContent = costEstimateJpyLabel(cost, costJpyRate);
                return;
            }
            loadJpyRate().then(rate => {
                el.textContent = rate ? costEstimateJpyLabel(cost, rate) : '日本円換算: 為替レートを取得できませんでした';
            });
        }

        function refreshCostBar(path = costBarPath) {
            costBarReady = true;
            costBarPath = path;
            const el = get('total-cost-estimate');
            if (!el) return;
            const answers = path.some(m => m && m.role === 'assistant' && (m.content || Number(m.tokens_out) > 0));
            el.classList.toggle('hidden', !answers);
            const cost = estimateConversationCost(path, costPricing, path);
            el.textContent = costPricing ? costEstimateLabel(cost, '費用予想')
                : `費用予想: ${costPricingFailed ? '算出不可' : '料金を読み込み中…'}`;
            el.title = COST_ESTIMATE_NOTE;
            if (messageMeta.__total__) messageMeta.__total__.cost_estimate = costPricing ? cost : null;
            el.onclick = () => openTokenDetail('__total__');
        }

        // Static official-rate data only; no chat data is sent by this request.
        fetch('/static/model_pricing.json?checked=2026-11-06', { credentials: 'same-origin' })
            .then(response => { if (!response.ok) throw new Error('pricing_unavailable'); return response.json(); })
            .then(data => { costPricing = data; if (costBarReady) refreshCostBar(); })
            .catch(() => { costPricingFailed = true; if (costBarReady) refreshCostBar(); });
        // Cache portions replace ordinary input, while hosted tool fees are added once.
        function requestTokenCost(request, rate, long, batch, flex = false) {
            const input = costTokenNumber(request.input_tokens) || 0;
            const output = costTokenNumber(request.output_tokens) || 0;
            const read = costTokenNumber(request.cache_read_tokens) || 0;
            const writes = costTokenNumber(request.cache_write_tokens) || 0;
            const ttl = ['5m', '1h', '30m', '12h'];
            const parts = ttl.map(t => costTokenNumber(request[`cache_write_${t}_tokens`]) || 0);
            const written = Math.max(writes, parts.reduce((a, b) => a + b, 0));
            const ordinary = Math.max(0, input - read - written);
            const inRate = long ? rate.long_input : rate.input;
            const outRate = long ? rate.long_output : rate.output;
            const cached = long ? rate.long_cached_input ?? rate.cached_input : rate.cached_input;
            const writeRates = rate.cache_write_multipliers || {};
            const multiplier = batch ? rate.batch_multiplier ?? 1 : 1;
            // Some providers keep cache-read pricing at the standard rate in Batch.
            const cachedMultiplier = batch ? rate.batch_cached_multiplier ?? multiplier : 1;
            const cachedFactor = cached != null && multiplier > 0 ? cachedMultiplier / multiplier : 1;
            const unknownWrites = Math.max(0, written - parts.reduce((a, b) => a + b, 0));
            const possible = Object.values(writeRates);
            let incomplete = !!request.pricing_incomplete || read + written > input || !!request.explicit_cache;
            if (read && cached == null) incomplete = true;
            let lowInput = ordinary * inRate + read * (cached ?? inRate) * cachedFactor;
            let highInput = ordinary * (long ? inRate : rate.max_input ?? inRate) + read * (long ? cached ?? inRate : rate.max_cached_input ?? cached ?? inRate) * cachedFactor;
            for (let i = 0; i < ttl.length; i++) {
                const factor = writeRates[ttl[i]];
                if (parts[i] && factor == null) incomplete = true;
                lowInput += parts[i] * inRate * (factor ?? 1);
                highInput += parts[i] * (long ? inRate : rate.max_input ?? inRate) * (factor ?? 1);
            }
            if (unknownWrites) {
                incomplete = true;
                lowInput += unknownWrites * inRate * (possible.length ? Math.min(...possible) : 1);
                highInput += unknownWrites * (long ? inRate : rate.max_input ?? inRate) * (possible.length ? Math.max(...possible) : 1);
            }
            let low = (lowInput + output * outRate) * multiplier * (flex ? 0.5 : 1) / 1e6;
            let high = (highInput + output * (rate.max_output ?? outRate)) * multiplier * (flex ? 0.5 : 1) / 1e6;
            let lowerBound = false;
            const tools = rate.tool_rates || {};
            const add = (count, name) => {
                if (!count) return;
                if (tools[name] == null) { incomplete = true; return; }
                low += count * tools[name]; high += count * tools[name];
            };
            const searches = costTokenNumber(request.web_search_calls) || 0;
            add(searches, request.search_preview ? 'web_search_preview' : 'web_search');
            if (searches && !request.search_preview && rate.search_content_tokens) {
                low += searches * rate.search_content_tokens * inRate / 1e6;
                high += searches * rate.search_content_tokens * inRate / 1e6;
            }
            add(costTokenNumber(request.file_search_calls), 'file_search');
            const grounding = costTokenNumber(request.grounding_queries) || 0;
            add(tools.grounding_prompt != null ? Math.min(1, grounding) : grounding, tools.grounding_prompt != null ? 'grounding_prompt' : 'grounding_query');
            const code = costTokenNumber(request.code_execution_calls) || 0;
            if (code && !request.code_execution_free) {
                if (rate.provider === 'openai') {
                    add((costTokenNumber(request.container_sessions) ?? 1) * (costTokenNumber(request.container_memory_gb) || 1), 'container_session');
                    incomplete = true; // Runtime/eligible prorating is not returned in generation usage.
                } else if (rate.provider === 'anthropic') {
                    add(1, 'code_execution_minimum'); incomplete = true; lowerBound = true;
                } else add(code, 'code_execution');
            }
            return { low, high, incomplete, lowerBound };
        }
