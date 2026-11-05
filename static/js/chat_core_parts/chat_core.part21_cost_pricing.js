        // Cache portions replace ordinary input, while hosted tool fees are added once.
        function requestTokenCost(request, rate, long, batch) {
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
            let low = (lowInput + output * outRate) * multiplier / 1e6;
            let high = (highInput + output * (rate.max_output ?? outRate)) * multiplier / 1e6;
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
