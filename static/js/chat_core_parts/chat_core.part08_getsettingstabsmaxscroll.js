        function getSettingsTabsMaxScroll(tabs) {
            if (!tabs) return 0;
            return Math.max(0, tabs.scrollWidth - tabs.clientWidth);
        }
        function syncSettingsTabsOverflow() {
            const wrap = get('settings-tabs-wrap');
            const tabs = get('settings-tabs');
            const leftBtn = get('settings-tabs-arrow-left');
            const rightBtn = get('settings-tabs-arrow-right');
            if (!wrap || !tabs) return;
            const max = getSettingsTabsMaxScroll(tabs);
            const left = tabs.scrollLeft;
            const canLeft = max > 2 && left > 2;
            const canRight = max > 2 && left < max - 2;
            wrap.classList.toggle('can-scroll', max > 2);
            wrap.classList.toggle('can-scroll-left', canLeft);
            wrap.classList.toggle('can-scroll-right', canRight);
            if (leftBtn) {
                leftBtn.disabled = !canLeft;
                leftBtn.setAttribute('aria-hidden', canLeft ? 'false' : 'true');
            }
            if (rightBtn) {
                rightBtn.disabled = !canRight;
                rightBtn.setAttribute('aria-hidden', canRight ? 'false' : 'true');
            }
        }
        function refreshSettingsTabsScroll() {
            initSettingsTabsScroll();
            initSettingsSwipeNav();
            syncSettingsTabsOverflow();
        }
        function initSettingsTabsScroll() {
            const wrap = get('settings-tabs-wrap');
            const tabs = get('settings-tabs');
            const leftBtn = get('settings-tabs-arrow-left');
            const rightBtn = get('settings-tabs-arrow-right');
            if (!wrap || !tabs || !leftBtn || !rightBtn) return;
            if (wrap.dataset.scrollBound === '1') {
                syncSettingsTabsOverflow();
                return;
            }
            wrap.dataset.scrollBound = '1';
            const EDGE_PX = 56;
            let holdTimer = 0;
            let holdRaf = 0;
            let holdDir = 0;

            const updateEdgeHover = (clientX) => {
                const rect = wrap.getBoundingClientRect();
                if (!rect.width) return;
                const x = clientX - rect.left;
                wrap.classList.toggle('is-edge-left', x >= 0 && x <= EDGE_PX);
                wrap.classList.toggle('is-edge-right', x >= rect.width - EDGE_PX && x <= rect.width);
            };
            const clearEdgeHover = () => {
                if (holdDir) return;
                wrap.classList.remove('is-edge-left', 'is-edge-right');
            };
            const scrollTabsBy = (delta, smooth) => {
                const max = getSettingsTabsMaxScroll(tabs);
                if (max <= 0 || !delta) return;
                const next = Math.max(0, Math.min(max, tabs.scrollLeft + delta));
                if (smooth && typeof tabs.scrollTo === 'function') {
                    tabs.scrollTo({ left: next, behavior: 'smooth' });
                } else {
                    tabs.scrollLeft = next;
                }
                syncSettingsTabsOverflow();
            };
            const stopHold = () => {
                holdDir = 0;
                if (holdTimer) { clearTimeout(holdTimer); holdTimer = 0; }
                if (holdRaf) { cancelAnimationFrame(holdRaf); holdRaf = 0; }
            };
            const startHold = (dir) => {
                stopHold();
                holdDir = dir;
                wrap.classList.toggle('is-edge-left', dir < 0);
                wrap.classList.toggle('is-edge-right', dir > 0);
                scrollTabsBy(dir * Math.max(120, tabs.clientWidth * 0.55), true);
                holdTimer = setTimeout(() => {
                    const step = () => {
                        if (!holdDir) return;
                        scrollTabsBy(holdDir * 14, false);
                        holdRaf = requestAnimationFrame(step);
                    };
                    holdRaf = requestAnimationFrame(step);
                }, 280);
            };

            wrap.addEventListener('pointermove', (e) => {
                if (e.pointerType === 'touch') return;
                updateEdgeHover(e.clientX);
            });
            wrap.addEventListener('pointerenter', (e) => {
                if (e.pointerType === 'touch') return;
                updateEdgeHover(e.clientX);
            });
            wrap.addEventListener('pointerleave', (e) => {
                if (e.pointerType === 'touch') return;
                stopHold();
                clearEdgeHover();
            });
            wrap.addEventListener('wheel', (e) => {
                const max = getSettingsTabsMaxScroll(tabs);
                if (max <= 2) return;
                const primarilyVertical = Math.abs(e.deltaY) >= Math.abs(e.deltaX);
                const delta = primarilyVertical ? e.deltaY : e.deltaX;
                if (!delta) return;
                const next = Math.max(0, Math.min(max, tabs.scrollLeft + delta));
                if (next === tabs.scrollLeft) return;
                e.preventDefault();
                tabs.scrollLeft = next;
                syncSettingsTabsOverflow();
            }, { passive: false });
            leftBtn.addEventListener('pointerdown', (e) => {
                if (e.button != null && e.button !== 0) return;
                e.preventDefault();
                startHold(-1);
            });
            rightBtn.addEventListener('pointerdown', (e) => {
                if (e.button != null && e.button !== 0) return;
                e.preventDefault();
                startHold(1);
            });
            leftBtn.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); });
            rightBtn.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); });
            window.addEventListener('pointerup', stopHold);
            window.addEventListener('pointercancel', stopHold);
            window.addEventListener('blur', stopHold);
            tabs.addEventListener('scroll', syncSettingsTabsOverflow, { passive: true });
            window.addEventListener('resize', syncSettingsTabsOverflow);
            if (typeof ResizeObserver !== 'undefined') {
                try {
                    const ro = new ResizeObserver(() => syncSettingsTabsOverflow());
                    ro.observe(tabs);
                    ro.observe(wrap);
                } catch (_) {}
            }
            syncSettingsTabsOverflow();
        }
        // region: Chat auto-scroll and scroll-to-bottom
        initSettingsTabsScroll();
        const chatContainer = get('chat-container');
        const scrollToBottomBtn = get('scroll-to-bottom-btn');
        const CHAT_BOTTOM_THRESHOLD = 64;
        let chatAutoScrollFrame = 0;
        let chatTouchY = null;
        let chatScrollbarDragging = false;
        let chatManualScrollPaused = false;
        let chatManualResumeArmed = false;
        let chatManualPauseIntent = false;
        let chatPauseIntentTimer = 0;
        let chatLastScrollTop = chatContainer ? chatContainer.scrollTop : 0;

        function isChatNearBottom() {
            if (!chatContainer) return true;
            return (chatContainer.scrollHeight - chatContainer.scrollTop - chatContainer.clientHeight) <= CHAT_BOTTOM_THRESHOLD;
        }

        function syncScrollToBottomButton() {
            if (!scrollToBottomBtn) return;
            const shouldShow = !userAutoScroll && !isChatNearBottom();
            scrollToBottomBtn.classList.toggle('hidden', !shouldShow);
        }

        function clearChatAutoScrollPauseIntent() {
            chatManualPauseIntent = false;
            if (chatPauseIntentTimer) {
                clearTimeout(chatPauseIntentTimer);
                chatPauseIntentTimer = 0;
            }
        }

        function armChatAutoScrollPause() {
            if (!chatContainer || chatManualScrollPaused) return;
            chatManualPauseIntent = true;
            if (chatPauseIntentTimer) clearTimeout(chatPauseIntentTimer);
            chatPauseIntentTimer = setTimeout(() => {
                chatManualPauseIntent = false;
                chatPauseIntentTimer = 0;
            }, 500);
        }

        function pauseChatAutoScroll() {
            if (!chatContainer) return;
            if (chatAutoScrollFrame) {
                cancelAnimationFrame(chatAutoScrollFrame);
                chatAutoScrollFrame = 0;
            }
            clearChatAutoScrollPauseIntent();
            chatManualScrollPaused = true;
            chatManualResumeArmed = false;
            userAutoScroll = false;
            syncScrollToBottomButton();
        }

        function resumeChatAutoScroll(options = {}) {
            clearChatAutoScrollPauseIntent();
            chatManualScrollPaused = false;
            chatManualResumeArmed = false;
            userAutoScroll = true;
            if (chatContainer) {
                if (options.scroll !== false) chatContainer.scrollTop = chatContainer.scrollHeight;
                chatLastScrollTop = chatContainer.scrollTop;
            }
            if (options.scroll === false) syncScrollToBottomButton();
            else scrollToBottom();
        }

        function performChatAutoScroll() {
            chatAutoScrollFrame = 0;
            if (!chatContainer || !userAutoScroll) return;
            chatContainer.scrollTop = chatContainer.scrollHeight;
            syncScrollToBottomButton();
        }

        function scrollToBottom(force = false) {
            if (!chatContainer) return;
            if (force) {
                clearChatAutoScrollPauseIntent();
                chatManualScrollPaused = false;
                chatManualResumeArmed = false;
                userAutoScroll = true;
            }
            if (!userAutoScroll) {
                syncScrollToBottomButton();
                return;
            }
            if (chatAutoScrollFrame) return;
            chatAutoScrollFrame = requestAnimationFrame(performChatAutoScroll);
        }

        if (chatContainer) {
            chatContainer.addEventListener('scroll', () => {
                const currentScrollTop = chatContainer.scrollTop;
                if (chatManualPauseIntent && currentScrollTop < chatLastScrollTop - 0.5) {
                    pauseChatAutoScroll();
                } else if (chatScrollbarDragging && currentScrollTop < chatLastScrollTop - 0.5) {
                    pauseChatAutoScroll();
                } else if (chatScrollbarDragging && chatManualScrollPaused && currentScrollTop > chatLastScrollTop + 0.5) {
                    chatManualResumeArmed = true;
                }
                if (chatManualScrollPaused) {
                    if (chatManualResumeArmed && isChatNearBottom()) {
                        chatManualScrollPaused = false;
                        chatManualResumeArmed = false;
                        userAutoScroll = true;
                    } else {
                        userAutoScroll = false;
                    }
                } else if (isChatNearBottom()) {
                    userAutoScroll = true;
                }
                chatLastScrollTop = currentScrollTop;
                syncScrollToBottomButton();
            }, { passive: true });
            chatContainer.addEventListener('wheel', (event) => {
                if (event.deltaY < 0) armChatAutoScrollPause();
                else if (event.deltaY > 0 && chatManualScrollPaused) chatManualResumeArmed = true;
            }, { passive: true });
            chatContainer.addEventListener('touchstart', (event) => {
                chatTouchY = event.touches.length ? event.touches[0].clientY : null;
            }, { passive: true });
            chatContainer.addEventListener('touchmove', (event) => {
                if (!event.touches.length) return;
                const nextY = event.touches[0].clientY;
                if (chatTouchY !== null && nextY > chatTouchY + 2) armChatAutoScrollPause();
                else if (chatTouchY !== null && nextY < chatTouchY - 2 && chatManualScrollPaused) chatManualResumeArmed = true;
                chatTouchY = nextY;
            }, { passive: true });
            chatContainer.addEventListener('touchend', () => { chatTouchY = null; }, { passive: true });
            chatContainer.addEventListener('pointerdown', (event) => {
                const scrollbarEdge = chatContainer.getBoundingClientRect().right - 20;
                if (event.button === 0 && event.clientX >= scrollbarEdge) chatScrollbarDragging = true;
            }, { passive: true });
            document.addEventListener('pointerup', () => { chatScrollbarDragging = false; }, { passive: true });

            const resizedMessages = new ResizeObserver(() => scrollToBottom());
            const observeMessageSizes = () => {
                Array.from(chatContainer.children).forEach((child) => resizedMessages.observe(child));
            };
            observeMessageSizes();
            new MutationObserver((mutations) => {
                mutations.forEach((mutation) => {
                    mutation.addedNodes.forEach((node) => {
                        if (node.nodeType === Node.ELEMENT_NODE && node.parentElement === chatContainer) {
                            resizedMessages.observe(node);
                        }
                    });
                });
                scrollToBottom();
            }).observe(chatContainer, { childList: true, subtree: true, characterData: true });
        }
        if (scrollToBottomBtn) {
            scrollToBottomBtn.addEventListener('click', () => scrollToBottom(true));
        }
        document.addEventListener('keydown', (event) => {
            const target = event.target;
            const isTyping = target && (target.matches('input, textarea, select') || target.isContentEditable);
            if (!isTyping && ['ArrowUp', 'PageUp', 'Home'].includes(event.key)) armChatAutoScrollPause();
            else if (!isTyping && chatManualScrollPaused && ['ArrowDown', 'PageDown', 'End'].includes(event.key)) chatManualResumeArmed = true;
        });

        // endregion

        // region: Image viewer
        // Image Viewer Logic
        let viewerImages = [];
        let viewerIndex = 0;
        let s = null;
        let p = null;
        let z = 1;
        let vX = 0;
        let vY = 0;
        let suppressViewerCloseClick = false;
        let viewerClosing = false, viewerAnimToken = 0, viewerHiddenSourceEl = null;

        function isViewerReducedMotion() {
            return Boolean(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
        }

        function setViewerHiddenSource(el) {
            if (viewerHiddenSourceEl && viewerHiddenSourceEl !== el) {
                try { viewerHiddenSourceEl.style.visibility = ''; } catch (_) {}
                viewerHiddenSourceEl = null;
            }
            if (el && el.isConnected) {
                try { el.style.visibility = 'hidden'; viewerHiddenSourceEl = el; } catch (_) {}
            }
        }

        function getViewerSourceRect(el) {
            if (!el || !el.isConnected) return null;
            const rect = el.getBoundingClientRect();
            if (!rect || rect.width < 4 || rect.height < 4 || rect.bottom <= 0 || rect.top >= window.innerHeight || rect.right <= 0 || rect.left >= window.innerWidth) return null;
            return rect;
        }

        function getViewerTargetRect(img, sourceEl) {
            const content = document.querySelector('.viewer-content');
            const stage = content ? content.getBoundingClientRect() : { left: 0, top: 0, width: window.innerWidth, height: window.innerHeight };
            let width = img ? img.offsetWidth : 0, height = img ? img.offsetHeight : 0;
            if (!(width > 0 && height > 0)) {
                const natW = (img && img.naturalWidth) || (sourceEl && sourceEl.naturalWidth) || 0;
                const natH = (img && img.naturalHeight) || (sourceEl && sourceEl.naturalHeight) || 0;
                if (natW > 0 && natH > 0) {
                    const maxW = Math.min(stage.width || window.innerWidth, window.innerWidth * 0.96);
                    const maxH = Math.min(stage.height || window.innerHeight, window.innerHeight * 0.85);
                    const fit = Math.min(maxW / natW, maxH / natH, 1);
                    width = natW * fit; height = natH * fit;
                }
            }
            if (!(width > 0 && height > 0)) return null;
            return { left: (stage.left + stage.width / 2) - width / 2, top: (stage.top + stage.height / 2) - height / 2, width, height };
        }

        function computeViewerFlipTransform(sourceRect, targetRect) {
            if (!sourceRect || !targetRect || targetRect.width <= 0 || targetRect.height <= 0) return null;
            const srcCx = sourceRect.left + sourceRect.width / 2, srcCy = sourceRect.top + sourceRect.height / 2;
            const dstCx = targetRect.left + targetRect.width / 2, dstCy = targetRect.top + targetRect.height / 2;
            const scale = Math.max(0.05, Math.min(2, Math.min(sourceRect.width / targetRect.width, sourceRect.height / targetRect.height)));
            return { dx: srcCx - dstCx, dy: srcCy - dstCy, scale };
        }

        function applyViewerTransform() {
            const img = get('image-viewer-img');
            if (!img) return;
            img.style.transform = `translate(${vX}px, ${vY}px) scale(${z})`;
        }

        // Keep the established swipe-regression markers in the readable bundle.
        // let viewerSwipe = null
        // viewerSwipe.dir = dx > 0 ? -1 : 1
        // viewerSwipe.resist = true
        // const effDx = viewerSwipe.resist ? dx * 0.3 : dx

        function resetViewerTransform() {
            p = null;
            z = 1;
            vX = vY = 0;
        }

        function openImageViewer(startUrl, groupSelector = '.chat-image', sourceElement = null) {
            const allImgs = Array.from(document.querySelectorAll(groupSelector));
            const items = allImgs.map(img => ({
                url: img.dataset.viewerSrc || img.currentSrc || img.src,
                filename: img.dataset.viewerFilename || img.title || (img.dataset.viewerSrc || img.currentSrc || img.src).split('/').pop(),
                element: img
            }));
            let idx = sourceElement ? items.findIndex(item => item.element === sourceElement) : -1;
            if (idx === -1) idx = items.findIndex(item => item.url === startUrl);
            if (idx === -1) return openViewerWithItems([{ url: startUrl, filename: startUrl.split('/').pop(), element: sourceElement || null }], 0);
            openViewerWithItems(items, idx);
        }

        function openViewerWithItems(items, index) {
            const viewer = get('image-viewer');
            if (!viewer) return;
            viewerAnimToken += 1;
            viewerClosing = false;
            viewer.classList.remove('closing', 'viewer-shared-open');
            setViewerHiddenSource(null);
            viewerImages = items;
            viewerIndex = (index >= 0 && index < items.length) ? index : 0;
            resetViewerTransform();
            clearViewerAdjacent();
            const item = viewerImages[viewerIndex], sourceEl = item && item.element ? item.element : null;
            const sourceRect = isViewerReducedMotion() ? null : getViewerSourceRect(sourceEl);
            if (sourceRect) viewer.classList.add('viewer-shared-open');
            viewer.classList.add('visible');
            sourceRect ? updateViewerState({ fade: false, sourceEl, sourceRect }) : updateViewerState();
            document.addEventListener('keydown', handleViewerKeydown);
        }

        function finalizeCloseViewer(token) {
            if (token !== viewerAnimToken) return;
            const viewer = get('image-viewer'), img = get('image-viewer-img');
            viewerClosing = false;
            if (viewer) viewer.classList.remove('visible', 'closing', 'viewer-shared-open');
            document.removeEventListener('keydown', handleViewerKeydown);
            clearViewerAdjacent();
            setViewerHiddenSource(null);
            if (img) {
                img.onload = img.onerror = null;
                img.style.transition = img.style.transform = img.style.opacity = '';
            }
            viewerImages = []; viewerIndex = 0; s = null; resetViewerTransform();
        }

        function closeImageViewer(opts) {
            const viewer = get('image-viewer');
            if (!viewer || !viewer.classList.contains('visible')) return;
            const immediate = Boolean(opts && opts.immediate) || isViewerReducedMotion();
            if (viewerClosing && !immediate) return;

            const item = viewerImages[viewerIndex], sourceEl = item && item.element ? item.element : null;
            const sourceRect = immediate ? null : getViewerSourceRect(sourceEl);
            const img = get('image-viewer-img');
            const targetRect = (sourceRect && img) ? getViewerTargetRect(img, sourceEl) : null;
            const flip = (sourceRect && targetRect) ? computeViewerFlipTransform(sourceRect, targetRect) : null;

            const token = ++viewerAnimToken;
            if (!immediate && flip && img) {
                viewerClosing = true;
                clearViewerAdjacent();
                document.removeEventListener('keydown', handleViewerKeydown);
                viewer.classList.remove('viewer-shared-open');
                viewer.classList.add('closing');
                img.style.transition = 'transform 0.26s var(--ease-out), opacity 0.24s var(--ease-out)';
                img.style.transform = `translate(${flip.dx}px, ${flip.dy}px) scale(${flip.scale})`;
                img.style.opacity = '1';
                let doneCalled = false;
                const done = () => { if (!doneCalled) { doneCalled = true; finalizeCloseViewer(token); } };
                const onEnd = (e) => { if (!e || e.target === img) { img.removeEventListener('transitionend', onEnd); done(); } };
                img.addEventListener('transitionend', onEnd);
                setTimeout(done, 290);
                return;
            }
            finalizeCloseViewer(token);
        }

        function clearViewerAdjacent() {
            const adj = document.querySelector('.viewer-adjacent');
            if (adj) adj.remove();
        }

        function renderViewerChrome() {
            if (!viewerImages.length) return;
            const meta = get('image-viewer-meta');
            const prevBtn = document.querySelector('.viewer-nav.prev');
            const nextBtn = document.querySelector('.viewer-nav.next');
            const item = viewerImages[viewerIndex];
            const editButton = get('viewer-edit-image');
            if (editButton) editButton.style.display = getViewerAttachmentRow(item) ? 'flex' : 'none';

            meta.innerText = `${viewerIndex + 1} / ${viewerImages.length} • ${item.filename}`;

            // Preload next image
            if (viewerIndex < viewerImages.length - 1) {
                const preload = new Image();
                preload.src = viewerImages[viewerIndex + 1].url;
            }

            prevBtn.style.display = viewerImages.length > 1 ? 'flex' : 'none';
            nextBtn.style.display = viewerImages.length > 1 ? 'flex' : 'none';
            prevBtn.style.opacity = viewerIndex > 0 ? '1' : '0.3';
            nextBtn.style.opacity = viewerIndex < viewerImages.length - 1 ? '1' : '0.3';
            prevBtn.style.pointerEvents = viewerIndex > 0 ? 'auto' : 'none';
            nextBtn.style.pointerEvents = viewerIndex < viewerImages.length - 1 ? 'auto' : 'none';
        }

        function getViewerAttachmentRow(item) {
            const row = item && item.element && item.element.uploadRow;
            return row && row.isConnected && row.querySelector('.upload-marker') ? row : null;
        }

        function editCurrentAttachmentImage() {
            const row = getViewerAttachmentRow(viewerImages[viewerIndex]);
            if (!row) return;
            closeImageViewer({ immediate: true });
            openMarkerModalForRow(row);
        }

        function updateViewerState(opts) {
            if (!viewerImages.length) return;
            const img = get('image-viewer-img');
            if (!img) return;
            const item = viewerImages[viewerIndex];
            const fade = !opts || opts.fade !== false;
            const sourceRect = opts && opts.sourceRect ? opts.sourceRect : null;
            const sourceEl = (opts && opts.sourceEl) || (item && item.element) || null;
            const token = viewerAnimToken;

            resetViewerTransform();

            renderViewerChrome();

            if (sourceRect) {
                const hasLoadedSource = Boolean(sourceEl && sourceEl.tagName === 'IMG' && sourceEl.complete && sourceEl.naturalWidth > 0);
                const thumbSrc = hasLoadedSource ? (sourceEl.currentSrc || sourceEl.src) : '';
                const useThumbFirst = Boolean(thumbSrc && thumbSrc !== item.url);
                img.onload = img.onerror = null;
                img.src = useThumbFirst ? thumbSrc : item.url;
                const startOpenFlip = () => {
                    if (token !== viewerAnimToken || viewerClosing) return;
                    const targetRect = getViewerTargetRect(img, sourceEl);
                    const flip = computeViewerFlipTransform(sourceRect, targetRect);
                    if (!flip) {
                        img.style.transition = 'transform 0.28s var(--ease-out), opacity 0.28s var(--ease-out)';
                        img.style.opacity = '1'; img.style.transform = 'scale(1)';
                        return;
                    }
                    setViewerHiddenSource(sourceEl);
                    img.style.transition = 'none'; img.style.opacity = '1';
                    img.style.transform = `translate(${flip.dx}px, ${flip.dy}px) scale(${flip.scale})`;
                    void img.offsetWidth;
                    requestAnimationFrame(() => {
                        if (token !== viewerAnimToken || viewerClosing) return;
                        img.style.transition = 'transform 0.3s var(--ease-out), opacity 0.24s var(--ease-out)';
                        img.style.transform = 'translate(0px, 0px) scale(1)';
                        setTimeout(() => {
                            if (token !== viewerAnimToken || viewerClosing) return;
                            const v = get('image-viewer'); if (v) v.classList.remove('viewer-shared-open');
                            img.style.transition = '';
                        }, 320);
                    });
                };
                if ((img.complete && img.naturalWidth) || hasLoadedSource) {
                    startOpenFlip();
                } else {
                    img.style.transition = 'none'; img.style.opacity = '0';
                    img.onload = img.onerror = startOpenFlip;
                }
                if (useThumbFirst) {
                    const full = new Image();
                    full.onload = () => {
                        if (token === viewerAnimToken && !viewerClosing && viewerImages[viewerIndex] === item) img.src = item.url;
                    };
                    full.src = item.url;
                }
                return;
            }

            if (viewerHiddenSourceEl) setViewerHiddenSource(sourceEl);

            img.style.transition = 'none';
            img.style.transform = fade ? 'scale(0.96)' : 'translate(0, 0) scale(1)';
            img.style.opacity = fade ? '0.35' : '0';

            const reveal = () => {
                if (token !== viewerAnimToken || viewerClosing) return;
                img.style.transition = fade ? 'transform 0.28s var(--ease-out), opacity 0.28s var(--ease-out)' : 'none';
                img.style.opacity = '1';
                img.style.transform = 'scale(1)';
                if (!fade) clearViewerAdjacent();
            };

            if (fade) {
                setTimeout(() => {
                    if (token !== viewerAnimToken || viewerClosing) return;
                    if (s && s.active) return;
                    img.src = item.url;
                    img.onload = reveal;
                    img.onerror = reveal;
                    if (img.complete && img.naturalWidth) reveal();
                }, 140);
            } else {
                img.src = item.url;
                img.onload = reveal;
                img.onerror = reveal;
                if (img.complete && img.naturalWidth) reveal();
            }
        }

        function navImage(dir) {
            const newIndex = viewerIndex + dir;
            if (newIndex >= 0 && newIndex < viewerImages.length) {
                clearViewerAdjacent();
                viewerIndex = newIndex;
                updateViewerState();
            }
        }

        function getViewerAdjacent(dir) {
            const content = document.querySelector('.viewer-content');
            if (!content) return null;
            const targetIndex = viewerIndex + dir;
            if (targetIndex < 0 || targetIndex >= viewerImages.length) return null;
            let adj = content.querySelector('.viewer-adjacent');
            if (!adj) {
                adj = document.createElement('img');
                adj.className = 'viewer-adjacent';
                adj.alt = '';
                content.appendChild(adj);
            }
            adj.src = viewerImages[targetIndex].url;
            adj.dataset.dir = String(dir);
            return adj;
        }

        function onViewerTouchStart(e) {
            if (!viewerImages.length) return;
            if (e.touches.length >= 2) {
                const first = e.touches[0];
                const second = e.touches[1];
                p = {
                    d: Math.hypot(first.clientX - second.clientX, first.clientY - second.clientY) || 1,
                    s: z,
                    mx: (first.clientX + second.clientX) / 2,
                    my: (first.clientY + second.clientY) / 2,
                    x: vX,
                    y: vY
                };
                s = null;
                e.preventDefault();
                return;
            }
            if (e.touches.length !== 1) return;
            const touch = e.touches[0];
            if (z > 1) {
                s = {
                    startX: touch.clientX,
                    startY: touch.clientY,
                    dx: vX,
                    dy: vY
                };
                return;
            }
            s = {
                startX: touch.clientX,
                startY: touch.clientY,
                lastX: touch.clientX,
                dx: 0,
                dy: 0,
                vx: 0,
                dir: 0,
                active: false,
                resist: false,
                adjacent: null,
                lastTime: Date.now()
            };
        }

        function onViewerTouchMove(e) {
            if (e.touches.length >= 2) {
                if (!p) {
                    const first = e.touches[0];
                    const second = e.touches[1];
                    p = {
                        d: Math.hypot(first.clientX - second.clientX, first.clientY - second.clientY) || 1,
                        s: z,
                        mx: (first.clientX + second.clientX) / 2,
                        my: (first.clientY + second.clientY) / 2,
                        x: vX,
                        y: vY
                    };
                    s = null;
                }
                const first = e.touches[0];
                const second = e.touches[1];
                const distance = Math.hypot(first.clientX - second.clientX, first.clientY - second.clientY) || 1;
                const midpointX = (first.clientX + second.clientX) / 2;
                const midpointY = (first.clientY + second.clientY) / 2;
                z = Math.min(6, Math.max(1, p.s * distance / p.d));
                if (z <= 1.0001) {
                    z = 1;
                    vX = 0;
                    vY = 0;
                } else {
                    vX = p.x + midpointX - p.mx;
                    vY = p.y + midpointY - p.my;
                }
                const img = get('image-viewer-img');
                if (!img) return;
                e.preventDefault();
                img.style.transition = 'none';
                applyViewerTransform();
                return;
            }
            if (p) return;
            if (z > 1 && s && !s.active) {
                const touch = e.touches[0];
                if (!touch) return;
                e.preventDefault();
                vX = s.dx + touch.clientX - s.startX;
                vY = s.dy + touch.clientY - s.startY;
                applyViewerTransform();
                return;
            }
            if (!s) return;
            const touch = e.touches[0];
            if (!touch) return;
            const dx = touch.clientX - s.startX;
            const dy = touch.clientY - s.startY;
            const now = Date.now();
            const dt = Math.max(now - s.lastTime, 1);
            const instantVx = (touch.clientX - s.lastX) / dt;
            s.vx = instantVx * 0.6 + s.vx * 0.4;
            s.lastX = touch.clientX;
            s.lastTime = now;
            s.dx = dx;

            if (!s.active) {
                if (Math.abs(dx) < 10 && Math.abs(dy) < 10) return;
                if (Math.abs(dx) < Math.abs(dy) * 1.15) {
                    s = null;
                    return;
                }
                s.active = true;
                s.dir = dx > 0 ? -1 : 1;
                s.adjacent = getViewerAdjacent(s.dir);
                if (!s.adjacent) s.resist = true;
            }

            e.preventDefault();

            const img = get('image-viewer-img');
            if (!img) return;

            const content = document.querySelector('.viewer-content');
            const stageWidth = content ? content.clientWidth : window.innerWidth;
            const effDx = s.resist ? dx * 0.3 : dx;

            img.style.transition = 'none';
            img.style.transform = `translateX(${effDx}px) scale(${1 - Math.min(Math.abs(effDx) / (stageWidth * 4), 0.04)})`;
            img.style.opacity = String(Math.max(1 - Math.min(Math.abs(effDx) / (stageWidth * 0.45), 0.55), 0.4));

            const adj = s.adjacent;
            if (adj) {
                const adjDir = Number(adj.dataset.dir) || 0;
                adj.style.transition = 'none';
                adj.style.transform = `translate(-50%, -50%) translateX(${adjDir * stageWidth + dx}px) scale(0.97)`;
                adj.style.opacity = String(Math.min(Math.abs(dx) / (stageWidth * 0.3), 1));
            }
        }

        function onViewerTouchEnd() {
            if (p) {
                p = null;
                s = null;
                return;
            }
            if (!s) return;
            const swipe = s;
            s = null;

            if (!swipe.active) return;

            suppressViewerCloseClick = true;
            setTimeout(() => { suppressViewerCloseClick = false; }, 120);

            const img = get('image-viewer-img');
            if (!img) return;

            const content = document.querySelector('.viewer-content');
            const stageWidth = content ? content.clientWidth : window.innerWidth;
            const threshold = stageWidth * 0.22;
            const dir = swipe.dir || (swipe.dx > 0 ? -1 : 1);
            const reducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
            const willNav = !swipe.resist && (Math.abs(swipe.dx) > threshold || (Math.abs(swipe.vx) > 0.45 && Math.sign(swipe.dx) === dir));
            const adj = swipe.adjacent;

            if (!willNav) {
                img.style.transition = 'transform 0.32s var(--ease-out), opacity 0.32s var(--ease-out)';
                img.style.transform = 'translateX(0) scale(1)';
                img.style.opacity = '1';
                if (adj) {
                    const adjRef = adj;
                    adj.style.transition = 'transform 0.32s var(--ease-out), opacity 0.32s var(--ease-out)';
                    adj.style.transform = `translate(-50%, -50%) translateX(${dir * stageWidth}px) scale(0.97)`;
                    adj.style.opacity = '0';
                    setTimeout(() => { if (adjRef.isConnected) adjRef.remove(); }, 340);
                }
                return;
            }

            if (reducedMotion) {
                finishSwipeNav(dir);
                return;
            }

            const exitX = dir * stageWidth;
            img.style.transition = 'transform 0.3s var(--ease-out), opacity 0.3s var(--ease-out)';
            img.style.transform = `translateX(${exitX}px) scale(0.96)`;
            img.style.opacity = '0.2';
            if (adj) {
                adj.style.transition = 'transform 0.3s var(--ease-out), opacity 0.3s var(--ease-out)';
                adj.style.transform = 'translate(-50%, -50%) translateX(0) scale(1)';
                adj.style.opacity = '1';
            }
            setTimeout(() => finishSwipeNav(dir), 300);
        }

        function finishSwipeNav(dir) {
            if (!viewerImages.length) return;
            if (s && s.active) return;
            const viewer = get('image-viewer');
            if (!viewer || !viewer.classList.contains('visible')) {
                clearViewerAdjacent();
                return;
            }
            const newIndex = viewerIndex + dir;
            if (newIndex < 0 || newIndex >= viewerImages.length) return;
            viewerIndex = newIndex;
            updateViewerState({ fade: false });
        }

        function handleViewerKeydown(e) {
            if (e.key === 'ArrowLeft') navImage(-1);
            if (e.key === 'ArrowRight') navImage(1);
            if (e.key === 'Escape') closeImageViewer();
        }

        function downloadCurrentImage() {
            if (!viewerImages.length) return;
            const item = viewerImages[viewerIndex];
            const a = document.createElement('a');
            a.href = item.url;
            a.download = item.filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }

        function copyCurrentImageUrl() {
            if (!viewerImages.length) return;
            const url = viewerImages[viewerIndex].url;
            // Try to resolve full URL if relative
            const fullUrl = new URL(url, window.location.origin).href;
            copyToClipboard(fullUrl, () => showToast("画像URLをコピーしました", "success"), () => showToast("コピーに失敗しました"));
        }

        function reuseCurrentImage() {
            if (!viewerImages.length) return;
            const item = viewerImages[viewerIndex];
            let path = item.url;
            try {
                // Extract relative path from URL: /files/user/filename -> user/filename
                const urlObj = new URL(path, window.location.origin);
                if (urlObj.pathname.startsWith('/files/')) {
                    path = decodeURIComponent(urlObj.pathname.replace('/files/', ''));
                }
            } catch(e) {}

            if (path) {
                if (!currentImageUrls.includes(path)) {
                    currentImageUrls.push(path);
                    setAttachmentNameForPath(path, item.filename || '');
                    updateFilePreview();
                    showToast("画像を添付ファイルに追加しました", "success");
                    closeImageViewer();
                } else {
                    showToast("この画像は既に添付されています", "info");
                }
            }
        }

        // endregion

        // region: Copy fallback
        // Robust Copy Function with Fallback
        async function copyToClipboard(text, successCallback, errorCallback) {
            try {
                if (navigator.clipboard && navigator.clipboard.writeText) {
                    await navigator.clipboard.writeText(text);
                    if(successCallback) successCallback();
                } else {
                    throw new Error("Clipboard API unavailable");
                }
            } catch (err) {
                try {
                    const textArea = document.createElement("textarea");
                    textArea.value = text;
                    textArea.style.position = "fixed";
                    textArea.style.left = "-9999px";
                    document.body.appendChild(textArea);
                    textArea.focus();
                    textArea.select();
                    const successful = document.execCommand('copy');
                    document.body.removeChild(textArea);
                    if (successful) {
                        if(successCallback) successCallback();
                    } else {
                        if(errorCallback) errorCallback(err);
                    }
                } catch (e) {
                    if(errorCallback) errorCallback(e);
                }
            }
        }

        // endregion

        // region: Quote reply
        // Quote Reply Logic
        const isQuoteMobileLayout = () => window.matchMedia('(max-width: 768px)').matches;
        let quotePreviewText = "";
        let quotePreviewMessageId = null;

        // The id of the message the current text selection sits in (`msg-<id>`), or
        // null when it is not a saved message. Sent so the server can say whom and
        // where the quote came from.
        function getSelectionMessageId() {
            try {
                const sel = window.getSelection();
                let node = sel && sel.anchorNode;
                if (node && node.nodeType !== 1) node = node.parentElement;
                while (node) {
                    const m = /^msg-(\d+)$/.exec(node.id || '');
                    if (m) return Number(m[1]);
                    node = node.parentElement;
                }
            } catch (e) {}
            return null;
        }

        // Render the quote bar as the formatting the selected text had: emphasis,
        // inline code, links and math are drawn instead of their Markdown/TeX source.
        // Block markers stay literal because the bar is a single-line preview.
        function renderQuoteBarContent(text) {
            const display = get('quote-text-display');
            if (!display) return;
            const source = String(text || '');
            display.innerHTML = sanitizeMarkdownHtml(source, { inline: true });
            // The preview is not a navigation target; keep the text on the bar.
            display.querySelectorAll('a').forEach((link) => link.removeAttribute('href'));
            // Attachments, code-block controls and media belong to the message body,
            // not to a one-line quote preview.
            display.querySelectorAll('img').forEach((img) => {
                img.replaceWith(document.createTextNode(img.getAttribute('alt') || ''));
            });
            display.querySelectorAll('.code-wrapper').forEach((wrapper) => {
                wrapper.replaceWith(document.createTextNode(wrapper.textContent || ''));
            });
            display.querySelectorAll('button, svg, video, audio, pre').forEach((el) => el.remove());
            if (maybeNeedsMathJax(source)) queueMathTypeset(display, source);
        }

        function showQuotePreview(text) {
            const bar = get('quote-bar');
            const changed = quotePreviewText !== text;
            quotePreviewText = text;
            quotePreviewMessageId = getSelectionMessageId();
            // A new pending quote supersedes any already-applied quote so the
            // bar always reflects what the next message would quote.
            if (!bar.classList.contains('preview')) {
                currentQuote = "";
                currentQuoteMessageId = null;
                bar.classList.add('preview');
            }
            if (changed || !bar.classList.contains('visible')) renderQuoteBarContent(text);
            bar.classList.add('visible');
            schedulePromptTokenEstimate();
        }

        function handleQuotePopover() {
            const sel = window.getSelection();
            const btn = get('quote-popover');
            if (!btn) return;
            const mobile = isQuoteMobileLayout();
            if (!sel || sel.rangeCount === 0 || document.querySelector('.modal-open, .modal-prep')) {
                btn.style.display = 'none';
                btn.classList.remove('show');
                return;
            }
            const text = sel.toString().trim();
            if (text.length > 0 && get('chat-container').contains(sel.anchorNode)) {
                if (mobile) {
                    // On mobile the floating button is hidden behind the native
                    // selection UI, so show a one-line quote preview in the
                    // composer bar instead. It is applied by #quote-confirm-btn.
                    showQuotePreview(text);
                    return;
                }
                const range = sel.getRangeAt(0);
                const rect = range.getBoundingClientRect();
                const wasHidden = btn.style.display === 'none' || !btn.style.display || getComputedStyle(btn).display === 'none';
                btn.style.display = 'block';
                btn.style.top = (rect.top - 40) + 'px';
                btn.style.left = rect.left + 'px';
                if (wasHidden) {
                    btn.classList.remove('show');
                    void btn.offsetWidth;
                    btn.classList.add('show');
                }
            } else {
                btn.style.display = 'none';
                btn.classList.remove('show');
            }
        }
        document.addEventListener('mouseup', handleQuotePopover);
        document.addEventListener('touchend', () => setTimeout(handleQuotePopover, 0), { passive: true });
        document.addEventListener('selectionchange', () => {
            // Avoid excessive updates during selection drag
            if (window.getSelection && window.getSelection().type === 'Range') {
                handleQuotePopover();
            }
        });

        get('quote-popover').onclick = () => {
            currentQuote = window.getSelection().toString().trim();
            currentQuoteMessageId = currentQuote ? getSelectionMessageId() : null;
            if(currentQuote) {
                renderQuoteBarContent(currentQuote);
                get('quote-bar').classList.add('visible');
                get('prompt-input').focus();
            }
            schedulePromptTokenEstimate();
            const qBtn = get('quote-popover');
            if (qBtn) {
                qBtn.style.display = 'none';
                qBtn.classList.remove('show');
            }
        };

        // Mobile: confirm the pending quote shown in the composer bar.
        get('quote-confirm-btn').onclick = () => {
            if (!quotePreviewText) return;
            currentQuote = quotePreviewText;
            currentQuoteMessageId = quotePreviewMessageId;
            quotePreviewText = "";
            quotePreviewMessageId = null;
            const bar = get('quote-bar');
            bar.classList.remove('preview');
            get('prompt-input').focus();
            schedulePromptTokenEstimate();
        };

        window.clearQuote = () => {
            currentQuote = "";
            currentQuoteMessageId = null;
            quotePreviewText = "";
            quotePreviewMessageId = null;
            const bar = get('quote-bar');
            bar.classList.remove('preview');
            bar.classList.remove('visible');
            get('quote-text-display').innerText = "";
            schedulePromptTokenEstimate();
        };

