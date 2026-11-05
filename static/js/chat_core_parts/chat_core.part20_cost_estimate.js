        // Costs use each saved answer's model, never the composer selection.
        let costPricing = null;
        let costPricingFailed = false;
        let costBarPath = [];
        let costBarAll = [];
        let costBarReady = false;
        const COST_ESTIMATE_NOTE = '有料APIの通常料金による概算（USD）。回答ごとの保存済みモデルで計算します。入力使用量がない回答は会話履歴から推定します。キャッシュ割引・保存料、検索・ツール・画像・音声・動画生成、税、無料枠・個別契約は含みません。時間帯や入力種別で単価が異なる場合は範囲で表示します。過去の請求額を再現するものではありません。';

        function costTokenNumber(value) {
            if (value === null || value === undefined || value === '') return null;
            const n = Number(value);
            return Number.isFinite(n) && n >= 0 ? n : null;
        }

        function estimateConversationCost(messages, pricing = costPricing, context = messages) {
            const result = { low: 0, high: 0, priced: 0, unpriced: 0, inferred: 0, long: 0 };
            const rows = new Map((context || []).filter(Boolean).map(m => [String(m.id), m]));
            const seen = new Set();
            for (const m of messages || []) {
                if (!m || m.role !== 'assistant' || seen.has(String(m.id))) continue;
                seen.add(String(m.id));
                const out = costTokenNumber(m.tokens_out);
                if (out === null && !m.content && !m.tokens) continue; // Pending answers are not expenses.
                const key = pricing && (pricing.aliases[m.model] || m.model);
                let rate = pricing && pricing.models[key];
                if (!rate || out === null) { result.unpriced++; continue; }
                rate = { ...rate };
                // Fixed reference date is intentional: opening an old chat tomorrow must not reprice it.
                const date = /^\d{4}-\d{2}-\d{2}/.exec(m.created_at || '')?.[0] || pricing.checked_at;
                for (const change of rate.changes || []) if (date >= change.from) Object.assign(rate, change);
                let usage = null;
                try { usage = typeof m.thought_data === 'string' ? JSON.parse(m.thought_data) : m.thought_data; } catch (_) { /* Legacy plain reasoning. */ }
                let input = costTokenNumber(usage?.pricing_input_tokens) ?? costTokenNumber(m.tokens_in);
                let incomplete = false;
                if (input === null) {
                    result.inferred++;
                    input = 0;
                    let parent = m.parent_id;
                    const ancestors = new Set([String(m.id)]);
                    if (parent === null || parent === undefined) incomplete = true;
                    while (parent !== null && parent !== undefined) {
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
                // A threshold applies to the whole request, not just the excess tokens.
                const long = rate.threshold != null && (rate.threshold_inclusive ? input >= rate.threshold : input > rate.threshold);
                if (long) result.long++;
                const inRate = long ? rate.long_input : rate.input;
                const outRate = long ? rate.long_output : rate.output;
                const batch = m.batch_job && m.batch_job.state === 'JOB_STATE_SUCCEEDED';
                // Unverified discounts are not guessed.
                const multiplier = batch ? (rate.batch_multiplier ?? 1) : 1;
                result.low += (input * inRate + out * outRate) * multiplier / 1e6;
                result.high += (input * (long ? inRate : (rate.max_input ?? inRate))
                    + out * (rate.max_output ?? outRate)) * multiplier / 1e6;
                result.priced++;
                if (incomplete) result.unpriced++;
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
        fetch('/static/model_pricing.json?checked=2026-11-05', { credentials: 'same-origin' })
            .then(response => { if (!response.ok) throw new Error('pricing_unavailable'); return response.json(); })
            .then(data => { costPricing = data; if (costBarReady) refreshCostBar(); })
            .catch(() => { costPricingFailed = true; if (costBarReady) refreshCostBar(); });
