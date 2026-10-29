        // Presentation-only state: never persisted, sent, or replayed by history loading.
        function takePromptForSend(animate = true) {
            const input = get('prompt-input');
            const preview = get('upload-list');
            const dock = document.querySelector('.composer-dock');
            const oldHeight = dock?.getBoundingClientRect().height;
            const bounds = [input, get('file-preview'), get('quote-bar')]
                .filter((el) => el && el.getClientRects().length && getComputedStyle(el).display !== 'none')
                .map((el) => el.getBoundingClientRect());
            const left = Math.min(...bounds.map((r) => r.left)), top = Math.min(...bounds.map((r) => r.top));
            const source = { left, top, width: Math.max(...bounds.map((r) => r.right)) - left,
                height: Math.max(...bounds.map((r) => r.bottom)) - top };
            const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
            const saved = {
                navigation: threadLoadSequence, text: input.value, images: currentImageUrls, mask: currentMaskImage,
                quote: currentQuote, quoteId: currentQuoteMessageId,
                rows: preview ? Array.from(preview.childNodes) : [],
                files: new Map(browserFastLocalFiles), markers: new Set(markerAppliedUploads),
                progress: uploadProgressState,
            };
            let edited = false, settled = false, restored = false;
            const onInput = () => { edited = true; };
            input.addEventListener('input', onInput);
            input.value = '';
            ['file-input', 'photo-input', 'mask-input'].forEach((id) => { const el = get(id); if (el) el.value = ''; });
            input.style.height = 'auto';
            currentImageUrls = [];
            currentMaskImage = null;
            browserFastLocalFiles.clear();
            markerAppliedUploads.clear();
            uploadProgressState = { total: 0, completed: 0, active: 0, perFilePct: {} };
            if (preview) preview.replaceChildren();
            clearQuote();
            updateFilePreview();
            schedulePromptTokenEstimate(true);
            updateMaskPreview();
            let shrink;
            if (dock && animate && !reduced && dock.animate) {
                shrink = dock.animate([{ height: oldHeight + 'px' }, { height: dock.getBoundingClientRect().height + 'px' }],
                    { duration: 350, easing: 'cubic-bezier(0.22, 1, 0.36, 1)' });
            }
            const releaseFiles = () => {
                if (!settled || restored) return;
                saved.files.forEach((entry) => {
                    const url = entry?.rowObj?.row?.getAttribute('data-local-url');
                    if (url) URL.revokeObjectURL(url);
                });
            };
            const finish = (restore, release = true) => {
                if (settled) return;
                settled = true;
                restore = restore && saved.navigation === threadLoadSequence;
                restored = restore;
                input.removeEventListener('input', onInput);
                if (restore) {
                    if (!edited && !input.value) {
                        input.value = saved.text;
                        input.style.height = 'auto';
                        input.style.height = Math.min(input.scrollHeight, 150) + 'px';
                    }
                    currentImageUrls = Array.from(new Set([...saved.images, ...currentImageUrls]));
                    if (!currentMaskImage) currentMaskImage = saved.mask;
                    saved.files.forEach((value, key) => browserFastLocalFiles.set(key, value));
                    saved.markers.forEach((value) => markerAppliedUploads.add(value));
                    if (preview) preview.prepend(...saved.rows);
                    if (!currentQuote && saved.quote) {
                        currentQuote = saved.quote;
                        currentQuoteMessageId = saved.quoteId;
                        get('quote-text-display').innerText = saved.quote;
                        get('quote-bar').classList.add('visible');
                    }
                    if (!uploadProgressState.total) uploadProgressState = saved.progress;
                    updateFilePreview();
                    updateMaskPreview();
                    schedulePromptTokenEstimate(true);
                } else {
                    if (release) releaseFiles();
                }
            };
            return { source, canRetry: () => !edited && saved.navigation === threadLoadSequence, stopDock: () => shrink?.cancel(), files: saved.files, releaseFiles, accept: (release = true) => finish(false, release), restore: () => finish(true) };
        }

        function animatePromptToMessage(row, transaction, follow) {
            let overlay, frame = 0, closed = false;
            const animations = [];
            const container = get('chat-container');
            const input = get('prompt-input');
            const reduce = window.matchMedia('(prefers-reduced-motion: reduce)');
            const stop = () => {
                if (closed) return;
                closed = true;
                window.ActivityLog?.log('ui.hide');
                transaction.stopDock();
                cancelAnimationFrame(frame);
                animations.forEach((animation) => animation.cancel());
                overlay?.remove();
                row?.classList.remove('prompt-send-hidden');
                window.removeEventListener('resize', stop);
                window.removeEventListener('popstate', stop);
                window.removeEventListener('pagehide', stop);
                document.removeEventListener('visibilitychange', stop);
                container?.removeEventListener('wheel', stop);
                container?.removeEventListener('touchstart', stop);
                input?.removeEventListener('input', stop);
                window.visualViewport?.removeEventListener('resize', stop);
                reduce.removeEventListener('change', stop);
            };
            if (!follow || reduce.matches || !row || !Element.prototype.animate) return stop;
            try {
                row.classList.remove('fade-in');
                row.classList.add('prompt-send-hidden');
                frame = requestAnimationFrame(() => {
                    try {
                        if (!row.isConnected) return stop();
                        const bubble = row.querySelector('.message-bubble');
                        const target = bubble.getBoundingClientRect();
                        const viewport = container.getBoundingClientRect();
                        if (!target.width || !target.height || target.bottom <= viewport.top || target.top >= viewport.bottom) return stop();
                        const source = transaction.source;
                        overlay = bubble.cloneNode(true);
                        overlay.removeAttribute('id');
                        overlay.querySelectorAll('[id]').forEach((el) => el.removeAttribute('id'));
                        overlay.querySelectorAll('.msg-controls, video, audio').forEach((el) => el.remove());
                        overlay.classList.add('prompt-send-overlay');
                        overlay.setAttribute('aria-hidden', 'true');
                        overlay.inert = true;
                        Object.assign(overlay.style, { width: target.width + 'px', height: target.height + 'px', left: target.left + 'px', top: target.top + 'px' });
                        document.body.appendChild(overlay);
                        window.ActivityLog?.log('ui.show');
                        const scale = Math.min(source.width / target.width, Math.max(source.height, 42) / target.height, 1);
                        const dx = source.left + (source.width - target.width * scale) / 2 - target.left;
                        const dy = source.top + (source.height - target.height * scale) / 2 - target.top;
                        const motion = overlay.animate([
                            { transform: `translate(${dx}px, ${dy}px) scale(${scale})`, borderRadius: '12px', backgroundColor: getComputedStyle(get('input-row')).backgroundColor },
                            { transform: 'translate(0, 0) scale(1)', borderRadius: getComputedStyle(bubble).borderRadius, backgroundColor: getComputedStyle(bubble).backgroundColor },
                        ], { duration: 350, easing: 'cubic-bezier(0.22, 1, 0.36, 1)', fill: 'both' });
                        animations.push(motion);
                        animations.push(overlay.animate([{ opacity: 1, offset: 0 }, { opacity: 1, offset: 0.8 }, { opacity: 0 }], { duration: 350, fill: 'both' }));
                        row.classList.remove('prompt-send-hidden');
                        animations.push(bubble.animate([{ opacity: 0, offset: 0 }, { opacity: 0, offset: 0.8 }, { opacity: 1 }], { duration: 350, fill: 'both' }));
                        const watch = () => {
                            if (!row.isConnected || !userAutoScroll || document.hidden) return stop();
                            const now = bubble.getBoundingClientRect();
                            overlay.style.left = now.left + 'px';
                            overlay.style.top = now.top + 'px';
                            frame = requestAnimationFrame(watch);
                        };
                        frame = requestAnimationFrame(watch);
                        motion.finished.then(stop, stop);
                    } catch (_) { console.warn('Prompt send animation failed'); stop(); }
                });
                window.addEventListener('resize', stop);
                window.addEventListener('popstate', stop);
                window.addEventListener('pagehide', stop);
                document.addEventListener('visibilitychange', stop);
                container.addEventListener('wheel', stop, { passive: true });
                container.addEventListener('touchstart', stop, { passive: true });
                input.addEventListener('input', stop);
                window.visualViewport?.addEventListener('resize', stop);
                reduce.addEventListener('change', stop);
            } catch (_) { console.warn('Prompt send animation failed'); stop(); }
            return stop;
        }
