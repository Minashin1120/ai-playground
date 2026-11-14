
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
