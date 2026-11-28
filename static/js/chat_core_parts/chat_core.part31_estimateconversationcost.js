
        function estimateConversationCost(messages, pricing = costPricing, context = messages) {
            const result = { low: 0, high: 0, priced: 0, unpriced: 0, inferred: 0, long: 0, requests: 0, items: [] };
            // Snapshot the running totals so each saved API request can report its own share.
            const capture = (request, date, inferred, batch, flex) => {
                const before = { low: result.low, high: result.high, priced: result.priced, unpriced: result.unpriced, long: result.long };
                addRequestCost(result, request, pricing, date, batch, flex);
                result.items.push(describeRequestCost(result, before, request, date, pricing, !!batch, !!flex, !!inferred));
            };
            const rows = new Map((context || []).filter(Boolean).map(m => [String(m.id), m]));
            const seen = new Set();
            for (const m of messages || []) {
                if (!m || m.role !== 'assistant' || seen.has(String(m.id))) continue;
                seen.add(String(m.id));
                let metadata = null;
                try { metadata = typeof m.thought_data === 'string' ? JSON.parse(m.thought_data) : m.thought_data; } catch (_) { /* Legacy reasoning. */ }
                // The account-wide usage tab sends already-resolved per-request rows.
                if (!metadata && Array.isArray(m.pricing_calls)) metadata = { pricing_calls: m.pricing_calls };
                const date = /^\d{4}-\d{2}-\d{2}/.exec(m.created_at || '')?.[0] || pricing?.checked_at;
                const batch = m.batch_job?.state === 'JOB_STATE_SUCCEEDED' || m.batch === true;
                const promptOptions = (rows.get(String(m.parent_id)) || m).prompt_options || m.prompt_options || {};
                const flex = (promptOptions && promptOptions.gemini_flex === true) || m.flex === true;
                if (Array.isArray(metadata?.pricing_calls)) {
                    if (!metadata.pricing_calls.length) result.priced++; // All requests explicitly rejected before generation.
                    for (const call of metadata.pricing_calls) {
                        if (!call || typeof call !== 'object') { result.unpriced++; continue; }
                        const callDate = /^\d{4}-\d{2}-\d{2}/.exec(call.created_at || '')?.[0] || date;
                        // Older image call records predate speed capture; recover only from their saved prompt.
                        const rate = costRate(pricing, call.model, callDate);
                        const request = rate?.image_selector === 'speed' && !call.speed && call.model === m.model
                            ? { ...call, speed: legacyImageRequest(m, rows).speed } : call;
                        capture(request, callDate, false, batch, flex);
                    }
                    continue;
                }
                const mRate = costRate(pricing, m.model, date);
                if (mRate?.unit === 'image') {
                    capture(legacyImageRequest(m, rows), date, false, false, flex);
                    continue;
                }
                if (mRate?.unit && mRate.unit !== 'token') {
                    capture(legacyUnitRequest(m, rows, mRate), date, true, false, flex);
                    continue;
                }
                const out = costTokenNumber(m.tokens_out);
                if (out === null && !m.content && !m.tokens) continue;
                if (out === null) { result.unpriced++; continue; }
                let input = costTokenNumber(metadata?.pricing_input_tokens) ?? costTokenNumber(m.tokens_in);
                let incomplete = false;
                let inferred = false;
                if (input === null) {
                    result.inferred++;
                    inferred = true;
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
                const legReq = { model: m.model, input_tokens: input, output_tokens: out, complete: !incomplete };
                if (mRate?.image_per_image) {
                    let files = m.image_url;
                    try { if (typeof files === 'string') files = JSON.parse(files); } catch (_) { files = files ? [files] : []; }
                    const nImg = Array.isArray(files) && files.length ? files.length : (/!\[Image\]\(/i.test(String(m.content || '')) ? 1 : 0);
                    if (nImg) legReq.images = nImg;
                }
                capture(legReq, date, inferred, batch, flex);
            }
            return result;
        }

        function costMoneyText(value) {
            const n = Number(value || 0);
            return n > 0 && n < 0.000001 ? '<$0.000001' : `$${n.toFixed(6)}`;
        }

        function costAmountText(cost) {
            if (!cost || !cost.priced) return '算出不可';
            return cost.lowerBound ? `${costMoneyText(cost.low)}以上`
                : cost.high - cost.low > 1e-12 ? `${costMoneyText(cost.low)}〜${costMoneyText(cost.high)}`
                    : costMoneyText(cost.low);
        }

        // One rows entry for the per-request breakdown; the running totals give its own share.
        function describeRequestCost(result, before, request, date, pricing, batch, flex, inferred) {
            const count = value => costTokenNumber(value);
            let writes = count(request.cache_write_tokens) || 0;
            for (const ttl of ['5m', '1h', '30m', '12h']) writes = Math.max(writes, count(request[`cache_write_${ttl}_tokens`]) || 0);
            const ttlWrites = ['5m', '1h', '30m', '12h'].reduce((sum, ttl) => sum + (count(request[`cache_write_${ttl}_tokens`]) || 0), 0);
            const tools = ['web_search_calls', 'file_search_calls', 'grounding_queries', 'code_execution_calls']
                .reduce((sum, key) => sum + (count(request[key]) || 0), 0);
            const priced = result.priced > before.priced;
            const reason = !priced ? (costRate(pricing, request.model, date) ? '使用量未取得' : '単価未登録')
                : result.unpriced > before.unpriced ? '一部未算入' : '';
            return {
                model: String(request.model || ''),
                date,
                input: count(request.input_tokens),
                output: count(request.output_tokens),
                cache_read: count(request.cache_read_tokens),
                cache_write: Math.max(writes, ttlWrites) || null,
                images: count(request.images),
                pages: count(request.pages),
                characters: count(request.characters),
                songs: count(request.songs),
                duration_ms: count(request.duration_ms),
                tools,
                low: result.low - before.low,
                high: result.high - before.high,
                priced,
                reason,
                inferred: !!inferred,
                long: result.long > before.long,
                batch: !!batch,
                flex: !!flex,
                api_key_id: String(request.api_key_id || ''),
                api_key_label: String(request.api_key_label || ''),
                api_key_inferred: !!request.api_key_inferred
            };
        }

        function costItemTokenText(item) {
            const parts = [];
            if (item.images !== null && item.images !== undefined) parts.push(`${item.images}枚`);
            if (item.songs !== null && item.songs !== undefined) parts.push(`${item.songs}曲`);
            if (item.pages !== null && item.pages !== undefined) parts.push(`${item.pages}ページ`);
            if (item.characters !== null && item.characters !== undefined) parts.push(`${item.characters}文字`);
            if (item.duration_ms !== null && item.duration_ms !== undefined) parts.push(`${(item.duration_ms / 1000).toFixed(1)}秒`);
            if (item.input !== null || item.output !== null) {
                parts.push(`入 ${item.input ?? '-'} / 出 ${item.output ?? '-'}`);
            }
            if (item.cache_read || item.cache_write) {
                parts.push(`キャッシュ 読込 ${item.cache_read || 0} / 書込 ${item.cache_write || 0}`);
            }
            if (item.tools) parts.push(`ツール ${item.tools}回`);
            return parts.join(' · ');
        }

        function costItemJpyText(item, rate) {
            if (!rate || !item.priced) return '';
            const low = jpyMoneyText(item.low * rate.rate);
            return item.high - item.low > 1e-12
                ? `約${low}〜${jpyMoneyText(item.high * rate.rate)}` : `約${low}`;
        }

        function appendCostItemList(container, items) {
            const list = document.createElement('div');
            list.className = 'text-[11px]';
            list.style.cssText = 'max-height:16rem;overflow-y:auto;margin-top:0.5rem;';
            items.forEach((item, index) => {
                const row = document.createElement('div');
                row.style.cssText = 'border-top:1px solid rgba(107,114,128,0.4);padding:0.25rem 0;';
                const head = document.createElement('div');
                head.className = 'flex justify-between';
                const name = document.createElement('span');
                name.className = 'font-mono';
                name.textContent = `${index + 1}. ${item.model || 'モデル不明'}`;
                const amount = document.createElement('span');
                amount.className = 'font-mono';
                amount.textContent = costAmountText(item);
                head.append(name, amount);
                row.appendChild(head);
                const line = document.createElement('div');
                line.className = 'text-gray-400';
                const flags = [];
                if (item.inferred) flags.push('履歴から推定');
                if (item.long) flags.push('長文料金');
                if (item.batch) flags.push('Batch');
                if (item.flex) flags.push('Flex');
                if (item.reason) flags.push(item.reason);
                const jpy = costItemJpyText(item, costJpyRate);
                const parts = [costItemTokenText(item), item.date, flags.join(' / '), jpy].filter(Boolean);
                if (parts.length) line.textContent = parts.join(' · ');
                if (parts.length) row.appendChild(line);
                list.appendChild(row);
            });
            container.appendChild(list);
        }

        function costEstimateLabel(cost, prefix = '費用予想') {
            if (!cost || !cost.priced) return `${prefix}: 算出不可`;
            return `${prefix}: ${costAmountText(cost)}${cost.unpriced ? '（一部未算入）' : ''}`;
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
            if (typeof document !== 'undefined' && typeof document.querySelectorAll === 'function') {
                document.querySelectorAll('.msg-cost-estimate-btn[data-msg-id]').forEach(btn => {
                    const msgId = btn.dataset.msgId;
                    const msg = path.find(m => m && String(m.id) === String(msgId));
                    if (!msg) return;
                    const msgCost = costPricing ? estimateConversationCost([msg], costPricing, path) : null;
                    if (messageMeta[msgId]) messageMeta[msgId].cost_estimate = msgCost;
                    btn.textContent = costPricing ? costEstimateLabel(msgCost, '費用予想')
                        : `費用予想: ${costPricingFailed ? '算出不可' : '料金を読み込み中…'}`;
                    btn.title = COST_ESTIMATE_NOTE;
                });
            }
        }

        // Static official-rate data only; no chat data is sent by this request.
        fetch('/static/model_pricing.json?checked=2026-11-28', { credentials: 'same-origin' })
            .then(response => { if (!response.ok) throw new Error('pricing_unavailable'); return response.json(); })
            .then(data => {
                costPricing = data;
                if (costBarReady) refreshCostBar();
                if (usageCostPendingData) renderUsageCostSummary(usageCostPendingData);
            })
            .catch(() => {
                costPricingFailed = true;
                if (costBarReady) refreshCostBar();
                if (usageCostPendingData) setUsageCostStatus('費用予想: 算出不可');
            });
        function legacyUnitRequest(message, rows, rate) {
            const parent = rows.get(String(message.parent_id));
            if (rate.unit === 'song') return { model: message.model, songs: 1, complete: true };
            if (rate.unit === 'page') {
                const match = /合計処理ページ数:\s*(\d+)/.exec(String(message.content || ''));
                return { model: message.model, pages: match ? Number(match[1]) : 1, complete: !!match };
            }
            if (rate.unit === 'character') {
                const chars = parent?.content ? String(parent.content).length : null;
                return { model: message.model, characters: chars, complete: chars !== null };
            }
            if (rate.unit === 'video') {
                const opts = parent?.prompt_options || {};
                const rawDur = String(message.model || '').startsWith('grok-') ? opts.grok_video_duration : opts.gemini_video_duration;
                const dur = costTokenNumber(rawDur);
                const res = String((String(message.model || '').startsWith('grok-') ? opts.grok_video_resolution : opts.gemini_video_resolution) || '').toLowerCase();
                return { model: message.model, duration_ms: dur ? dur * 1000 : null, resolution: res || null, complete: false };
            }
            return { model: message.model, complete: false };
        }

        function requestImageUnitCost(result, request, rate, batch, flex) {
            const flexMul = flex ? 0.5 : 1;
            const batchMul = batch ? rate.batch_multiplier ?? 1 : 1;
            const inTok = costTokenNumber(request.input_tokens);
            const outTok = costTokenNumber(request.output_tokens);
            if ((inTok !== null || outTok !== null) && (rate.image_output != null || rate.output != null)) {
                const cost = requestTokenCost(request, rate, false, batch, flex);
                result.low += cost.low; result.high += cost.high;
                if (cost.lowerBound) result.lowerBound = true;
                result.priced++;
                if (request.complete === false || cost.incomplete) result.unpriced++;
                return;
            }
            const count = costTokenNumber(request.images);
            if (count === null) { result.unpriced++; return; }
            let low = null, high = null, lowerBound = false, incomplete = false;
            if (rate.per_image_rates) {
                const qMap = rate.per_image_rates;
                const qVal = request.quality && request.quality !== 'auto' ? qMap[request.quality] : null;
                const sVal = request.size && request.size !== 'auto' ? request.size : null;
                const vals = [];
                if (qVal) {
                    if (sVal && qVal[sVal] != null) vals.push(qVal[sVal]);
                    else Object.values(qVal).forEach(v => { if (typeof v === 'number') vals.push(v); });
                } else {
                    Object.values(qMap).forEach(sub => Object.entries(sub).forEach(([k, v]) => {
                        if ((!sVal || k === sVal) && typeof v === 'number') vals.push(v);
                    }));
                }
                if (vals.length) {
                    low = count * Math.min(...vals) * batchMul * flexMul;
                    high = count * Math.max(...vals) * batchMul * flexMul;
                    if (vals.length > 1) incomplete = true;
                }
            }
            if (low === null && rate.resolution_rates && request.resolution && rate.resolution_rates[request.resolution] != null) {
                const sub = rate.resolution_rates[request.resolution];
                const u = typeof sub === 'number' ? sub : (sub[request.quality] ?? sub.low ?? Object.values(sub)[0]);
                if (Number.isFinite(u)) { low = count * u; high = count * u; }
            }
            if (low === null) {
                const prices = rate.image_rates?.[request.operation] ?? rate.image_rates?.generate;
                const selected = prices?.[rate.image_selector === 'speed' ? request.speed : request.quality];
                const resKey = String(request.resolution || '').toLowerCase();
                const resPrice = (selected && typeof selected === 'object' && resKey && typeof selected[resKey] === 'number')
                    ? selected[resKey] : null;
                if (typeof prices === 'number' || typeof selected === 'number' || typeof resPrice === 'number') {
                    const unit = typeof prices === 'number' ? prices : typeof selected === 'number' ? selected : resPrice;
                    low = count * unit * flexMul;
                    high = count * unit * flexMul;
                } else {
                    const vals = (selected && typeof selected === 'object')
                        ? Object.entries(selected).filter(([k, v]) => k !== 'minimum' && typeof v === 'number').map(([, v]) => v)
                        : [];
                    if (vals.length) {
                        low = count * Math.min(...vals) * flexMul;
                        high = count * Math.max(...vals) * flexMul;
                        if (Math.min(...vals) !== Math.max(...vals)) incomplete = true;
                    } else {
                        const minimum = selected?.minimum ?? prices?.minimum;
                        lowerBound = Number.isFinite(minimum);
                        if (!Number.isFinite(minimum)) { result.unpriced++; return; }
                        low = count * minimum * flexMul;
                        high = count * minimum * flexMul;
                    }
                }
            }
            const extraIn = (costTokenNumber(request.input_images) || 0) * (rate.input_image_rate || 0);
            result.low += low + extraIn;
            result.high += high + extraIn;
            result.priced++;
            if (lowerBound) result.lowerBound = true;
            if (request.complete === false || lowerBound || incomplete) result.unpriced++;
        }

        function requestUnitCost(result, request, rate, batch = false, flex = false) {
            if (rate.unit === 'image') { requestImageUnitCost(result, request, rate, batch, flex); return; }
            const batchMul = batch ? rate.batch_multiplier ?? 1 : 1;
            const flexMul = flex ? 0.5 : 1;
            let low = 0, high = 0, incomplete = request.complete === false, lowerBound = false;
            if (rate.unit === 'song') {
                const n = costTokenNumber(request.songs) ?? 1;
                low = high = n * (rate.song_rate || 0);
            } else if (rate.unit === 'page') {
                const pages = costTokenNumber(request.pages);
                if (pages === null) { result.unpriced++; return; }
                low = high = pages * (rate.page_rate || 0) * batchMul;
            } else if (rate.unit === 'character') {
                const chars = costTokenNumber(request.characters);
                if (chars === null) { result.unpriced++; return; }
                low = high = chars * (rate.character_rate || 0) / 1e6;
            } else if (rate.unit === 'duration') {
                const ms = costTokenNumber(request.duration_ms);
                const audioMs = (costTokenNumber(request.audio_in_ms) || 0) + (costTokenNumber(request.audio_out_ms) || 0);
                const mins = ms !== null ? ms / 60000 : (request.audio_in_ms != null || request.audio_out_ms != null) ? audioMs / 60000 : null;
                if (mins === null && rate.duration_rate_per_minute > 0) {
                    low = rate.duration_rate_per_minute / 60; high = rate.duration_rate_per_minute; incomplete = true;
                } else {
                    const base = (mins || 0) * (rate.duration_rate_per_minute || 0);
                    const standby = ((costTokenNumber(request.standby_ms) || 0) / 60000) * (rate.standby_rate_per_minute || 0);
                    const turns = (costTokenNumber(request.text_turns) || 0) * (rate.text_turn_rate || 0);
                    low = high = base + standby + turns;
                }
            } else if (rate.unit === 'video') {
                const inTok = costTokenNumber(request.input_tokens);
                const vidTok = costTokenNumber(request.video_output_tokens);
                if (rate.video_output != null && inTok !== null && vidTok !== null) {
                    const outTok = costTokenNumber(request.output_tokens) || vidTok;
                    const textOut = Math.max(0, outTok - vidTok);
                    low = high = (inTok * (rate.input || 0) + textOut * (rate.output || 0) + vidTok * rate.video_output) / 1e6;
                } else {
                    const sec = request.duration_ms != null ? costTokenNumber(request.duration_ms) / 1000 : costTokenNumber(request.duration_seconds);
                    const tbl = (request.without_audio && rate.video_only_rates) ? rate.video_only_rates : rate.video_rates || {};
                    const resKey = String(request.resolution || '').toLowerCase();
                    const rVal = resKey && tbl[resKey] != null ? tbl[resKey] : null;
                    const vals = Object.values(tbl).filter(v => typeof v === 'number');
                    if (sec !== null) {
                        const loRate = rVal ?? tbl.default ?? (vals.length ? Math.min(...vals) : 0);
                        const hiRate = rVal ?? tbl.default ?? (vals.length ? Math.max(...vals) : 0);
                        const extra = (costTokenNumber(request.input_images) || 0) * (rate.input_image_rate || 0);
                        low = sec * loRate + extra; high = sec * hiRate + extra;
                        if (rVal == null && vals.length > 1) incomplete = true;
                    } else if (vals.length) {
                        const dSec = String(request.model || '').startsWith('grok-') ? 5 : 8;
                        low = dSec * Math.min(...vals); high = dSec * Math.max(...vals);
                        incomplete = true;
                    } else { result.unpriced++; return; }
                }
            } else { result.unpriced++; return; }
            result.low += low * flexMul; result.high += high * flexMul;
            result.priced++;
            if (lowerBound) result.lowerBound = true;
            if (incomplete) result.unpriced++;
        }

        // Cache portions replace ordinary input, while hosted tool fees are added once.
        function requestTokenCost(request, rate, long, batch, flex = false) {
            const rawIn = costTokenNumber(request.input_tokens);
            const rawOut = costTokenNumber(request.output_tokens);
            let input = rawIn || 0;
            let output = rawOut || 0;
            let derived = false;
            const audioInTok = costTokenNumber(request.audio_input_tokens);
            const textInTok = costTokenNumber(request.text_input_tokens);
            const imgInTok = costTokenNumber(request.image_input_tokens);
            const audioOutTok = costTokenNumber(request.audio_output_tokens);
            const imgOutTok = costTokenNumber(request.image_output_tokens);
            let estAudioIn = null, estAudioOut = null;
            if (rate.audio_tokens_per_second) {
                if (rawIn === null && (request.audio_in_ms != null || request.duration_ms != null)) {
                    estAudioIn = ((costTokenNumber(request.audio_in_ms ?? request.duration_ms) || 0) / 1000) * rate.audio_tokens_per_second;
                    derived = true;
                }
                if (rawOut === null && request.audio_out_ms != null && rate.audio_output != null) {
                    estAudioOut = ((costTokenNumber(request.audio_out_ms) || 0) / 1000) * rate.audio_tokens_per_second;
                    derived = true;
                }
            }
            const read = costTokenNumber(request.cache_read_tokens) || 0;
            const writes = costTokenNumber(request.cache_write_tokens) || 0;
            const ttl = ['5m', '1h', '30m', '12h'];
            const parts = ttl.map(t => costTokenNumber(request[`cache_write_${t}_tokens`]) || 0);
            const written = Math.max(writes, parts.reduce((a, b) => a + b, 0));
            const ordinary = Math.max(0, input - read - written);
            const inRate = long ? rate.long_input : (rate.text_input ?? rate.input);
            const outRate = long ? rate.long_output : rate.output;
            const cached = long ? rate.long_cached_input ?? rate.cached_input : rate.cached_input;
            const writeRates = rate.cache_write_multipliers || {};
            const multiplier = batch ? rate.batch_multiplier ?? 1 : 1;
            // Some providers keep cache-read pricing at the standard rate in Batch.
            const cachedMultiplier = batch ? rate.batch_cached_multiplier ?? multiplier : 1;
            const cachedFactor = cached != null && multiplier > 0 ? cachedMultiplier / multiplier : 1;
            const unknownWrites = Math.max(0, written - parts.reduce((a, b) => a + b, 0));
            const possible = Object.values(writeRates);
            let incomplete = !!request.pricing_incomplete || !!rate.pricing_unlisted || read + written > input || !!request.explicit_cache;
            if (read && cached == null) incomplete = true;
            let lowInput = 0, highInput = 0;
            const hasModalityIn = audioInTok !== null || textInTok !== null || imgInTok !== null;
            if (hasModalityIn) {
                const cText = costTokenNumber(request.cached_text_tokens);
                const cImg = costTokenNumber(request.cached_image_tokens);
                const cAudio = costTokenNumber(request.cached_audio_tokens);
                const hasModCache = cText !== null || cImg !== null || cAudio !== null;
                const ordText = Math.max(0, (textInTok || 0) - (cText || 0));
                const ordImg = Math.max(0, (imgInTok || 0) - (cImg || 0));
                const ordAudio = Math.max(0, (audioInTok || 0) - (cAudio || 0));
                const remOrd = Math.max(0, ordinary - ordText - ordImg - ordAudio);
                lowInput = ordText * inRate + ordImg * (rate.image_input ?? inRate) + ordAudio * (rate.audio_input ?? inRate) + remOrd * inRate;
                highInput = ordText * (rate.max_input ?? inRate) + ordImg * (rate.image_input ?? inRate) + ordAudio * (rate.audio_input ?? inRate) + remOrd * (rate.max_input ?? inRate);
                if (hasModCache) {
                    lowInput += ((cText || 0) * (cached ?? inRate) + (cImg || 0) * (rate.cached_image_input ?? cached ?? rate.image_input ?? inRate) + (cAudio || 0) * (rate.cached_audio_input ?? cached ?? rate.audio_input ?? inRate)) * cachedFactor;
                    highInput += ((cText || 0) * (cached ?? inRate) + (cImg || 0) * (rate.cached_image_input ?? cached ?? rate.image_input ?? inRate) + (cAudio || 0) * (rate.cached_audio_input ?? cached ?? rate.audio_input ?? inRate)) * cachedFactor;
                } else {
                    lowInput += read * (cached ?? inRate) * cachedFactor;
                    highInput += read * (rate.max_cached_input ?? cached ?? inRate) * cachedFactor;
                }
            } else if (estAudioIn !== null) {
                lowInput = estAudioIn * (rate.audio_input ?? inRate);
                highInput = lowInput;
            } else {
                const modInMax = Math.max(rate.max_input ?? inRate, ...(rate.unit === 'image' && rate.image_input != null ? [rate.image_input] : []));
                lowInput = ordinary * inRate + read * (cached ?? inRate) * cachedFactor;
                highInput = ordinary * (long ? inRate : modInMax) + read * (long ? cached ?? inRate : rate.max_cached_input ?? cached ?? inRate) * cachedFactor;
            }
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
            let lowOut = 0, highOut = 0, perImageExtra = 0;
            if (imgOutTok !== null && rate.image_output != null) {
                const txtOut = Math.max(0, output - imgOutTok);
                lowOut = txtOut * outRate + imgOutTok * rate.image_output;
                highOut = txtOut * (rate.max_output ?? outRate) + imgOutTok * rate.image_output;
            } else if (audioOutTok !== null && rate.audio_output != null) {
                const txtOut = Math.max(0, output - audioOutTok);
                lowOut = txtOut * outRate + audioOutTok * rate.audio_output;
                highOut = txtOut * (rate.max_output ?? outRate) + audioOutTok * rate.audio_output;
            } else if (estAudioOut !== null && rate.audio_output != null) {
                lowOut = estAudioOut * rate.audio_output;
                highOut = lowOut;
            } else if (rate.image_per_image && request.images != null) {
                const cnt = costTokenNumber(request.images) || 0;
                const res = String(request.resolution || '').toLowerCase();
                const tbl = rate.image_per_image;
                const pVal = tbl[res] ?? tbl.default;
                const vals = Object.values(tbl).filter(v => typeof v === 'number');
                const loP = pVal ?? (vals.length ? Math.min(...vals) : 0);
                const hiP = pVal ?? (vals.length ? Math.max(...vals) : 0);
                if (cnt > 0) {
                    derived = true;
                    perImageExtra = cnt * loP;
                    const txtOut = imgOutTok !== null ? Math.max(0, output - imgOutTok) : (rawOut !== null && rate.image_output != null ? 0 : output);
                    lowOut = txtOut * outRate;
                    highOut = txtOut * (rate.max_output ?? outRate) + cnt * Math.max(0, hiP - loP) * 1e6;
                }
            } else if (rate.image_output != null && output > 0) {
                // Legacy Gemini image call without separate image token count or image count: bracket text..image output
                lowOut = output * outRate;
                highOut = output * rate.image_output;
                incomplete = true;
            } else {
                lowOut = output * outRate;
                highOut = output * (rate.max_output ?? outRate);
            }
            const mul = multiplier * (flex ? 0.5 : 1);
            let low = ((lowInput + lowOut) / 1e6 + perImageExtra) * mul;
            let high = ((highInput + highOut) / 1e6 + perImageExtra) * mul;
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
            return { low, high, incomplete, lowerBound, derived };
        }

        // ---- 全チャットの費用予想（設定 > 費用タブ） ----
        const USAGE_COST_PERIODS = [
            { id: '1d', label: '1日以内' },
            { id: '7d', label: '7日以内' },
            { id: '14d', label: '2週間以内' },
            { id: '1m', label: '1か月以内' },
            { id: '3m', label: '3か月以内' },
            { id: '1y', label: '1年間' }
        ];
        let usageCostPeriod = '1m';
        let usageCostLoadedPeriod = null;
        let usageCostRequest = 0;
        let usageCostPendingData = null;

        function usageCostElements() {
            return {
                panel: get('tab-usage'),
                status: get('usage-cost-status'),
                total: get('usage-cost-total'),
                jpy: get('usage-cost-jpy'),
                meta: get('usage-cost-meta'),
                breakdown: get('usage-cost-breakdown'),
                note: get('usage-cost-note')
            };
        }

        function markUsageCostPeriod(period) {
            document.querySelectorAll('#usage-period-buttons [data-period]').forEach(btn => {
                const active = btn.dataset.period === period;
                btn.classList.toggle('is-active', active);
                btn.setAttribute('aria-pressed', active ? 'true' : 'false');
            });
        }

        function setUsageCostStatus(text) {
            const el = usageCostElements().status;
            if (el) el.textContent = text;
        }

        function usageCostModelRows(items) {
            const rows = new Map();
            for (const item of items || []) {
                if (!item) continue;
                const key = item.model || 'モデル不明';
                let row = rows.get(key);
                if (!row) {
                    row = { model: key, low: 0, high: 0, priced: 0, unpriced: 0, requests: 0 };
                    rows.set(key, row);
                }
                row.low += Number(item.low) || 0;
                row.high += Number(item.high) || 0;
                row.requests++;
                if (item.priced) row.priced++; else row.unpriced++;
            }
            return [...rows.values()].sort((a, b) => (b.high - a.high) || (b.low - a.low));
        }

        function usageCostKeyRows(items) {
            const rows = new Map();
            for (const item of items || []) {
                if (!item) continue;
                const key = item.api_key_id || 'unknown';
                const label = item.api_key_label || '未記録（キー未設定）';
                let row = rows.get(key);
                if (!row) {
                    row = { key, label, low: 0, high: 0, priced: 0, unpriced: 0, requests: 0, inferred: 0 };
                    rows.set(key, row);
                }
                row.low += Number(item.low) || 0;
                row.high += Number(item.high) || 0;
                row.requests++;
                if (item.priced) row.priced++; else row.unpriced++;
                if (item.api_key_inferred) row.inferred++;
            }
            return [...rows.values()].sort((a, b) => (b.high - a.high) || (b.low - a.low));
        }

        function appendUsageCostBreakdownSection(container, heading, rows, labelFor, detailFor) {
            if (!rows.length) return;
            const title = document.createElement('div');
            title.className = 'text-[11px] font-bold text-gray-400 mt-3';
            title.textContent = heading;
            container.appendChild(title);
            rows.forEach(row => {
                const line = document.createElement('div');
                line.className = 'border-t border-gray-700/60 py-1.5';
                const head = document.createElement('div');
                head.className = 'flex justify-between gap-3';
                const name = document.createElement('span');
                name.className = 'font-mono text-[11px] break-all';
                name.textContent = labelFor(row);
                const amount = document.createElement('span');
                amount.className = 'font-mono text-[11px] shrink-0';
                amount.textContent = costAmountText(row) + (row.unpriced ? '（一部未算入）' : '');
                head.append(name, amount);
                line.appendChild(head);
                const detail = document.createElement('div');
                detail.className = 'text-[10px] text-gray-400';
                detail.textContent = detailFor(row);
                line.appendChild(detail);
                container.appendChild(line);
            });
        }

        function renderUsageCostBreakdown(modelRows, keyRows = []) {
            const breakdown = usageCostElements().breakdown;
            if (!breakdown) return;
            breakdown.replaceChildren();
            appendUsageCostBreakdownSection(
                breakdown,
                'APIキー別',
                keyRows,
                row => row.label,
                row => `API ${row.requests}回 · 算定 ${row.priced}回` +
                    (row.unpriced ? ` · 未算入 ${row.unpriced}回` : '') +
                    (row.inferred ? ' · 一部現在の設定から推定' : '')
            );
            appendUsageCostBreakdownSection(
                breakdown,
                'モデル別',
                modelRows,
                row => row.model,
                row => `API ${row.requests}回 · 算定 ${row.priced}回` + (row.unpriced ? ` · 未算入 ${row.unpriced}回` : '')
            );
        }

        function renderUsageCostSummary(data) {
            const el = usageCostElements();
            if (!el.panel || !data) return;
            if (!costPricing) {
                usageCostPendingData = costPricingFailed ? null : data;
                setUsageCostStatus(`費用予想: ${costPricingFailed ? '算出不可' : '料金を読み込み中…'}`);
                return;
            }
            usageCostPendingData = null;
            const messages = Array.isArray(data.messages) ? data.messages : [];
            const cost = estimateConversationCost(messages, costPricing, messages);
            if (el.total) {
                el.textContent = costAmountText(cost) + (cost.unpriced ? '（一部未算入）' : '');
                el.title = COST_ESTIMATE_NOTE;
            }
            if (el.jpy) fillCostEstimateJpy(el.jpy, cost);
            if (el.meta) {
                const parts = [`回答 ${data.answers}件`, `API呼び出し ${cost.requests}回`];
                if (cost.inferred) parts.push(`履歴から推定 ${cost.inferred}件`);
                if (cost.unpriced) parts.push(`一部未算入 ${cost.unpriced}件`);
                if (data.unreadable) parts.push(`使用量未取得 ${data.unreadable}件`);
                if (data.truncated) parts.push('件数が多いため直近分のみ');
                el.meta.textContent = parts.join(' · ');
            }
            renderUsageCostBreakdown(usageCostModelRows(cost.items), usageCostKeyRows(cost.items));
            setUsageCostStatus('');
            if (el.note) el.note.classList.remove('hidden');
        }

        function loadUsageCostSummary(period = usageCostPeriod) {
            const el = usageCostElements();
            if (!el.panel) return;
            usageCostPeriod = USAGE_COST_PERIODS.some(p => p.id === period) ? period : '1m';
            markUsageCostPeriod(usageCostPeriod);
            if (usageCostLoadedPeriod === usageCostPeriod && el.total && el.total.textContent && !costPricingFailed) return;
            const requestId = ++usageCostRequest;
            setUsageCostStatus('集計中…');
            fetch(`/api/usage/costs?period=${encodeURIComponent(usageCostPeriod)}`, {
                credentials: 'same-origin',
                headers: { 'Accept': 'application/json' }
            })
                .then(response => {
                    if (!response.ok) throw new Error('usage_unavailable');
                    return response.json();
                })
                .then(data => {
                    if (requestId !== usageCostRequest) return;
                    usageCostLoadedPeriod = usageCostPeriod;
                    renderUsageCostSummary(data);
                })
                .catch(() => {
                    if (requestId !== usageCostRequest) return;
                    setUsageCostStatus('集計できませんでした。時間をおいて再度お試しください。');
                });
        }

        function refreshUsageCostSummary() {
            usageCostLoadedPeriod = null;
            loadUsageCostSummary(usageCostPeriod);
        }

        function selectUsageCostPeriod(period) {
            if (usageCostPeriod === period && usageCostLoadedPeriod === period) {
                markUsageCostPeriod(period);
                return;
            }
            usageCostLoadedPeriod = null;
            loadUsageCostSummary(period);
        }
