        // Costs use each saved answer's model, never the composer selection.
        let costPricing = null;
        let costPricingFailed = false;
        let costBarPath = [];
        let costBarReady = false;
        const COST_ESTIMATE_NOTE = '有料APIの通常料金による概算（USD）。回答ごとの保存済みモデルで計算します。入力使用量がない回答は会話履歴から推定します。モデルAPIの呼び出しごとに入力・出力（Thinkingを含む）を合算します。Ideogramはモデル・品質・速度・処理別の単価と画像枚数で計算します。単価未公表の設定は公表最安値を下限として「以上・一部未算入」で表示します。使用量未取得の呼び出しは一部未算入です。API報告のキャッシュ読込・書込と検索などの追加料金を含みます。Batch APIは事業者ごとの割引率（割引のない事業者を含む）で計算し、割引対象外のキャッシュ読込は通常単価のままです。保存時間や実行時間など取得できない課金情報は一部未算入です。その他の画像・音声・動画生成、税、無料枠・個別契約は含みません。時間帯や入力種別で単価が異なる場合は範囲で表示します。過去の請求額を再現するものではありません。';

        function costTokenNumber(value) {
            if (value === null || value === undefined || value === '') return null;
            const n = Number(value);
            return Number.isFinite(n) && n >= 0 ? n : null;
        }

        // Token-detail summary from saved per-request cache usage; null when no request reported cache fields.
        function costCacheUsage(thoughtData) {
            let metadata = thoughtData;
            if (typeof thoughtData === 'string') {
                if (!thoughtData.includes('cache_')) return null;
                try { metadata = JSON.parse(thoughtData); } catch (_) { return null; }
            }
            if (!metadata || typeof metadata !== 'object' || !Array.isArray(metadata.pricing_calls)) return null;
            let read = 0, write = 0, known = false;
            const ttls = {};
            for (const call of metadata.pricing_calls) {
                if (!call || typeof call !== 'object') continue;
                const readTokens = costTokenNumber(call.cache_read_tokens);
                const writeTokens = costTokenNumber(call.cache_write_tokens);
                let ttlTotal = 0;
                for (const ttl of ['5m', '1h', '30m', '12h']) {
                    const value = costTokenNumber(call[`cache_write_${ttl}_tokens`]);
                    if (value === null) continue;
                    ttls[ttl] = (ttls[ttl] || 0) + value;
                    ttlTotal += value;
                    known = true;
                }
                if (readTokens !== null || writeTokens !== null) known = true;
                read += readTokens || 0;
                write += Math.max(writeTokens || 0, ttlTotal);
            }
            return known ? { read, write, ttls } : null;
        }

        function costRate(pricing, model, date) {
            const key = pricing && (pricing.aliases[model] || model);
            const base = pricing && pricing.models[key];
            if (!base) return null;
            const rate = { ...base };
            for (const change of rate.changes || []) if (date >= change.from) Object.assign(rate, change);
            return rate;
        }

        function addRequestCost(result, request, pricing, date, batch = false) {
            result.requests++;
            if (String(request.model).startsWith('grok-') && Number.isSafeInteger(request.cost_in_usd_ticks) && request.cost_in_usd_ticks >= 0) {
                result.low += request.cost_in_usd_ticks / 1e10; result.high += request.cost_in_usd_ticks / 1e10;
                result.priced++;
                if (request.complete === false) result.unpriced++;
                return;
            }
            const rate = costRate(pricing, request.model, date);
            if (!rate) { result.unpriced++; return; }
            if (rate.unit === 'image') {
                const count = costTokenNumber(request.images);
                const prices = rate.image_rates?.[request.operation];
                const selected = prices?.[rate.image_selector === 'speed' ? request.speed : request.quality];
                const minimum = selected?.minimum ?? prices?.minimum;
                const unit = typeof prices === 'number' ? prices : typeof selected === 'number' ? selected : minimum;
                const lowerBound = Number.isFinite(unit) && typeof prices !== 'number' && typeof selected !== 'number';
                if (count === null || !Number.isFinite(unit)) { result.unpriced++; return; }
                result.low += count * unit;
                result.high += count * unit;
                result.priced++;
                if (lowerBound) result.lowerBound = true;
                if (request.complete === false || lowerBound) result.unpriced++;
                return;
            }
            const input = costTokenNumber(request.input_tokens);
            const output = costTokenNumber(request.output_tokens);
            if (input === null && output === null) { result.unpriced++; return; }
            const long = rate.threshold != null && (rate.threshold_inclusive ? (input || 0) >= rate.threshold : (input || 0) > rate.threshold);
            if (long) result.long++;
            const cost = requestTokenCost(request, rate, long, batch);
            result.low += cost.low; result.high += cost.high;
            if (cost.lowerBound) result.lowerBound = true;
            result.priced++;
            if (input === null || output === null || request.complete === false || cost.incomplete) result.unpriced++;

        }

        function legacyImageRequest(message, rows) {
            let files = message.image_url;
            try { if (typeof files === 'string') files = JSON.parse(files); } catch (_) { files = files ? [files] : []; }
            const count = Array.isArray(files) ? files.length : 0;
            const parent = rows.get(String(message.parent_id));
            const options = parent?.prompt_options || {};
            let source = parent?.image_url;
            let hasSource = Array.isArray(source) ? source.length > 0 : !!source && source !== '[]' && source !== 'null';
            // Image edits can reuse the latest generated image without a new attachment.
            let node = parent;
            const seen = new Set();
            while (!hasSource && node?.parent_id != null && !seen.has(String(node.id))) {
                seen.add(String(node.id)); node = rows.get(String(node.parent_id));
                if (node?.role === 'assistant' && node.image_url && node.image_url !== '[]' && node.image_url !== 'null') hasSource = true;
            }
            const operation = ['remove_object', 'remove_background'].includes(options.ideogram_operation)
                ? options.ideogram_operation : hasSource && message.model === 'ideogram-4.5' ? 'edit' : 'generate';
            let quality = options.ideogram_quality;
            if (!['very_low', 'low', 'medium', 'high'].includes(quality) || (quality === 'very_low' && operation === 'generate')) {
                quality = operation === 'edit' ? 'medium' : 'high';
            }
            const speed = parent ? (['turbo', 'default', 'quality'].includes(options.ideogram_speed) ? options.ideogram_speed : 'default') : null;
            return { model: message.model, images: count || null, operation, quality, speed, complete: !!parent };
        }

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
                if (Array.isArray(metadata?.pricing_calls)) {
                    if (!metadata.pricing_calls.length) result.priced++; // All requests explicitly rejected before generation.
                    for (const call of metadata.pricing_calls) {
                        if (!call || typeof call !== 'object') { result.unpriced++; continue; }
                        const callDate = /^\d{4}-\d{2}-\d{2}/.exec(call.created_at || '')?.[0] || date;
                        // Older image call records predate speed capture; recover only from their saved prompt.
                        const rate = costRate(pricing, call.model, callDate);
                        const request = rate?.image_selector === 'speed' && !call.speed && call.model === m.model
                            ? { ...call, speed: legacyImageRequest(m, rows).speed } : call;
                        addRequestCost(result, request, pricing, callDate, batch);
                    }
                    continue;
                }
                if (costRate(pricing, m.model, date)?.unit === 'image') {
                    addRequestCost(result, legacyImageRequest(m, rows), pricing, date);
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
                addRequestCost(result, { model: m.model, input_tokens: input, output_tokens: out, complete: !incomplete }, pricing, date, batch);
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
