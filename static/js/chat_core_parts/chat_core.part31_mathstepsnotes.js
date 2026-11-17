        // ------------------------------------------------------------------
        // 数学ステップ解説: 「わかりやすくする」補足の保存・削除
        // 元のステップ（"3"）と分割した小ステップ（"3-1"）ごとに補足を持ち、
        // 補足の生成・後始末だけをここへ分けている（part30 から抽出）。
        // ------------------------------------------------------------------
        // The reader preferred the original explanation: forget the saved replacement
        // (legacy simplified text or an expanded step) on the server as well.
        function forgetMathStepsVariants(root, stepIndex, noteKeys = null) {
            const target = mathStepsRootTargets(root);
            const state = getMathStepsRootState(root);
            if (!target || !state) return;
            const payload = Object.assign({}, target, {
                current: state.current,
                max_reached: state.maxReached,
                remove_simplified: [stepIndex],
                remove_expanded: [stepIndex]
            });
            if (Array.isArray(noteKeys) && noteKeys.length) payload.remove_notes = noteKeys;
            apiFetch(MATH_STEPS_PROGRESS_ROUTE, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            }).catch(() => {});
        }

        // All note keys that belong to one source step (the step itself and its split parts).
        function mathStepsNoteKeysForSource(state, sourceIndex) {
            const prefix = `${sourceIndex}-`;
            const keys = [];
            Object.keys(state.notes || {}).forEach((key) => {
                if (key === String(sourceIndex) || key.indexOf(prefix) === 0) keys.push(key);
            });
            return keys;
        }

        // ------------------------------------------------------------------
        // 「わかりやすくする」（今のステップに「なぜそうなるのか」の補足を足す）
        // ------------------------------------------------------------------
        function requestMathStepsExplain(root, button) {
            const state = getMathStepsRootState(root);
            const target = mathStepsRootTargets(root);
            if (!state || !target) return;
            if (!target.message_id) {
                showToast('回答が保存された後に「わかりやすくする」を使えます', 'info', false);
                return;
            }
            if (state.streaming) {
                showToast('回答の生成が終わってから使えます', 'info', false);
                return;
            }
            const current = state.steps[state.current];
            if (!current) return;
            const payload = { mode: 'explain', message_id: target.message_id, step_index: current.sourceIndex };
            if (current.partIndex !== null && current.partIndex !== undefined) {
                payload.part_index = current.partIndex;
            }
            if (button) {
                button.dataset.busy = '1';
                button.disabled = true;
                button.innerHTML = '<i class="fas fa-circle-notch fa-spin" aria-hidden="true"></i>説明しています…';
            }
            mathStepsStatus(root, 'なぜそうなるのかを説明しています…');
            apiFetch(MATH_STEPS_SIMPLIFY_ROUTE, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            }).then(async (res) => {
                const data = await res.json().catch(() => ({}));
                const note = (data && typeof data.note === 'string') ? data.note.trim() : '';
                if (!res.ok || !data || data.status !== 'ok' || !note) {
                    const reason = res.status === 409
                        ? 'このステップの補足をこれ以上保存できません'
                        : (data && data.error ? data.error : '補足を作れませんでした');
                    throw new Error(reason);
                }
                const nextState = normalizeMathStepsMessageState(data.math_steps_state || {});
                state.simplified = Object.assign({}, nextState.simplified);
                state.expanded = Object.assign({}, state.expanded, nextState.expanded);
                state.notes = Object.assign({}, state.notes, nextState.notes);
                state.notes[mathStepsNoteKey(current)] = note;
                if (button) {
                    delete button.dataset.busy;
                    button.disabled = false;
                    button.innerHTML = '<i class="fas fa-lightbulb" aria-hidden="true"></i>わかりやすくする';
                }
                mathStepsStatus(root, '「なぜそうなるの？」の補足を追加しました');
                paintMathStepsRoot(root);
            }).catch((error) => {
                mathStepsStatus(root, '');
                showToast(error && error.message ? error.message : '補足を作れませんでした', 'error', true);
                if (button) {
                    delete button.dataset.busy;
                    button.disabled = false;
                    button.innerHTML = '<i class="fas fa-lightbulb" aria-hidden="true"></i>わかりやすくする';
                }
            });
        }
