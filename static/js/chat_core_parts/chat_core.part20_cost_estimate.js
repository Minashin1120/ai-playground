        // Costs use each saved answer's model, never the composer selection.
        let costPricing = null;
        let costPricingFailed = false;
        let costBarPath = [];
        let costBarAll = [];
        let costBarReady = false;
        const COST_ESTIMATE_NOTE = '有料APIの通常料金による概算（USD）。回答ごとの保存済みモデルで計算します。入力使用量がない回答は会話履歴から推定します。モデルAPIの呼び出しごとに入力・出力（Thinkingを含む）を合算します。Ideogram 4.5は生成・編集の画像枚数と品質で計算します。使用量未取得の呼び出しは一部未算入です。キャッシュ割引・保存料、検索・ツールの追加料金、その他の画像・音声・動画生成、税、無料枠・個別契約は含みません。時間帯や入力種別で単価が異なる場合は範囲で表示します。過去の請求額を再現するものではありません。';

        function costTokenNumber(value) {
            if (value === null || value === undefined || value === '') return null;
            const n = Number(value);
            return Number.isFinite(n) && n >= 0 ? n : null;
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
            const rate = costRate(pricing, request.model, date);
            if (!rate) { result.unpriced++; return; }
            if (rate.unit === 'image') {
                const count = costTokenNumber(request.images);
                const unit = rate.image_rates?.[request.operation]?.[request.quality];
                if (count === null || !Number.isFinite(unit)) { result.unpriced++; return; }
                result.low += count * unit;
                result.high += count * unit;
                result.priced++;
                if (request.complete === false) result.unpriced++;
                return;
            }
            const input = costTokenNumber(request.input_tokens);
            const output = costTokenNumber(request.output_tokens);
            if (input === null && output === null) { result.unpriced++; return; }
            const long = rate.threshold != null && (rate.threshold_inclusive ? (input || 0) >= rate.threshold : (input || 0) > rate.threshold);
            if (long) result.long++;
            const inRate = long ? rate.long_input : rate.input;
            const outRate = long ? rate.long_output : rate.output;
            const multiplier = batch ? (rate.batch_multiplier ?? 1) : 1;
            result.low += ((input || 0) * inRate + (output || 0) * outRate) * multiplier / 1e6;
            result.high += ((input || 0) * (long ? inRate : (rate.max_input ?? inRate))
                + (output || 0) * (rate.max_output ?? outRate)) * multiplier / 1e6;
            result.priced++;
            if (input === null || output === null || request.complete === false) result.unpriced++;
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
                ? options.ideogram_operation : hasSource ? 'edit' : 'generate';
            let quality = options.ideogram_quality;
            if (!['very_low', 'low', 'medium', 'high'].includes(quality) || (quality === 'very_low' && operation === 'generate')) {
                quality = operation === 'edit' ? 'medium' : 'high';
            }
            return { model: message.model, images: count || null, operation, quality, complete: !!parent };
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
                        addRequestCost(result, call, pricing, callDate, batch);
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
            const amount = cost.high - cost.low > 1e-12 ? `${money(cost.low)}〜${money(cost.high)}` : money(cost.low);
            return `${prefix}: ${amount}${cost.unpriced ? '（一部未算入）' : ''}`;
        }

        function refreshCostBar(path = costBarPath, all = costBarAll) {
            costBarReady = true;
            costBarPath = path;
            costBarAll = all;
            for (const [id, rows, metaId, prefix] of [
                ['total-cost-estimate', path, '__total__', '費用予想'],
                ['total-cost-estimate-all', all, '__total_all_branches__', '全分岐の費用予想']
            ]) {
                const el = get(id);
                if (!el) continue;
                const answers = rows.some(m => m && m.role === 'assistant' && (m.content || Number(m.tokens_out) > 0));
                el.classList.toggle('hidden', !answers);
                const cost = estimateConversationCost(rows, costPricing, all.length ? all : path);
                el.textContent = costPricing ? costEstimateLabel(cost, prefix)
                    : `${prefix}: ${costPricingFailed ? '算出不可' : '料金を読み込み中…'}`;
                el.title = COST_ESTIMATE_NOTE;
                if (messageMeta[metaId]) messageMeta[metaId].cost_estimate = costPricing ? cost : null;
                el.onclick = () => openTokenDetail(metaId);
            }
        }

        // Static official-rate data only; no chat data is sent by this request.
        fetch('/static/model_pricing.json?checked=2026-11-06', { credentials: 'same-origin' })
            .then(response => { if (!response.ok) throw new Error('pricing_unavailable'); return response.json(); })
            .then(data => { costPricing = data; if (costBarReady) refreshCostBar(); })
            .catch(() => { costPricingFailed = true; if (costBarReady) refreshCostBar(); });
