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

        // Image Viewer Logic
        let viewerImages = [];
        let viewerIndex = 0;
        let viewerSwipe = null;
        let suppressViewerCloseClick = false;

        function openImageViewer(startUrl, groupSelector = '.chat-image') {
            const allImgs = Array.from(document.querySelectorAll(groupSelector));
            // Filter out duplicate sources if needed, but keep DOM order
            const items = allImgs.map(img => ({
                url: img.dataset.viewerSrc || img.currentSrc || img.src,
                filename: img.dataset.viewerFilename || img.title || (img.dataset.viewerSrc || img.currentSrc || img.src).split('/').pop(),
                element: img
            }));

            // Find index of startUrl
            const foundIndex = items.findIndex(item => item.url === startUrl);
            if (foundIndex === -1) {
                // Fallback if not found in DOM list (single view)
                openViewerWithItems([{ url: startUrl, filename: startUrl.split('/').pop(), element: null }], 0);
                return;
            }
            openViewerWithItems(items, foundIndex);
        }

        function openViewerWithItems(items, index) {
            viewerImages = items;
            viewerIndex = (index >= 0 && index < items.length) ? index : 0;
            clearViewerAdjacent();
            updateViewerState();
            get('image-viewer').classList.add('visible');
            document.addEventListener('keydown', handleViewerKeydown);
        }

        function closeImageViewer() {
            get('image-viewer').classList.remove('visible');
            document.removeEventListener('keydown', handleViewerKeydown);
            clearViewerAdjacent();
            viewerImages = [];
            viewerIndex = 0;
            viewerSwipe = null;
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

        function updateViewerState(opts) {
            if (!viewerImages.length) return;
            const img = get('image-viewer-img');
            if (!img) return;
            const item = viewerImages[viewerIndex];
            const fade = !opts || opts.fade !== false;

            renderViewerChrome();

            img.style.transition = 'none';
            img.style.transform = fade ? 'scale(0.96)' : 'translateX(0) scale(1)';
            img.style.opacity = fade ? '0.35' : '0';

            const reveal = () => {
                img.style.transition = fade ? 'transform 0.28s var(--ease-out), opacity 0.28s var(--ease-out)' : 'none';
                img.style.opacity = '1';
                img.style.transform = 'scale(1)';
                if (!fade) clearViewerAdjacent();
            };

            if (fade) {
                setTimeout(() => {
                    if (viewerSwipe && viewerSwipe.active) return;
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
            if (e.touches.length !== 1) return;
            const t = e.touches[0];
            viewerSwipe = {
                startX: t.clientX,
                startY: t.clientY,
                lastX: t.clientX,
                lastY: t.clientY,
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
            if (!viewerSwipe) return;
            const t = e.touches[0];
            const dx = t.clientX - viewerSwipe.startX;
            const dy = t.clientY - viewerSwipe.startY;
            const now = Date.now();
            const dt = Math.max(now - viewerSwipe.lastTime, 1);
            const instantVx = (t.clientX - viewerSwipe.lastX) / dt;
            viewerSwipe.vx = instantVx * 0.6 + viewerSwipe.vx * 0.4;
            viewerSwipe.lastX = t.clientX;
            viewerSwipe.lastY = t.clientY;
            viewerSwipe.lastTime = now;
            viewerSwipe.dx = dx;
            viewerSwipe.dy = dy;

            if (!viewerSwipe.active) {
                if (Math.abs(dx) < 10 && Math.abs(dy) < 10) return;
                if (Math.abs(dx) < Math.abs(dy) * 1.15) {
                    viewerSwipe = null;
                    return;
                }
                viewerSwipe.active = true;
                viewerSwipe.dir = dx > 0 ? -1 : 1;
                viewerSwipe.adjacent = getViewerAdjacent(viewerSwipe.dir);
                if (!viewerSwipe.adjacent) viewerSwipe.resist = true;
            }

            e.preventDefault();

            const img = get('image-viewer-img');
            if (!img) return;
            const content = document.querySelector('.viewer-content');
            const stageWidth = content ? content.clientWidth : window.innerWidth;
            const effDx = viewerSwipe.resist ? dx * 0.3 : dx;

            img.style.transition = 'none';
            img.style.transform = `translateX(${effDx}px) scale(${1 - Math.min(Math.abs(effDx) / (stageWidth * 4), 0.04)})`;
            img.style.opacity = String(Math.max(1 - Math.min(Math.abs(effDx) / (stageWidth * 0.45), 0.55), 0.4));

            const adj = viewerSwipe.adjacent;
            if (adj) {
                const adjDir = Number(adj.dataset.dir) || 0;
                adj.style.transition = 'none';
                adj.style.transform = `translate(-50%, -50%) translateX(${adjDir * stageWidth + dx}px) scale(0.97)`;
                adj.style.opacity = String(Math.min(Math.abs(dx) / (stageWidth * 0.3), 1));
            }
        }

        function onViewerTouchEnd() {
            if (!viewerSwipe) return;
            const swipe = viewerSwipe;
            viewerSwipe = null;
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
            if (viewerSwipe && viewerSwipe.active) return;
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

        // Quote Reply Logic
        const isQuoteMobileLayout = () => window.matchMedia('(max-width: 768px)').matches;
        let quotePreviewText = "";

        function showQuotePreview(text) {
            const bar = get('quote-bar');
            quotePreviewText = text;
            // A new pending quote supersedes any already-applied quote so the
            // bar always reflects what the next message would quote.
            if (!bar.classList.contains('preview')) {
                currentQuote = "";
                bar.classList.add('preview');
            }
            get('quote-text-display').innerText = text;
            bar.classList.add('visible');
            schedulePromptTokenEstimate();
        }

        function handleQuotePopover() {
            const sel = window.getSelection();
            const btn = get('quote-popover');
            if (!btn) return;
            const mobile = isQuoteMobileLayout();
            if (!sel || sel.rangeCount === 0) {
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
            if(currentQuote) {
                get('quote-text-display').innerText = currentQuote;
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
            quotePreviewText = "";
            const bar = get('quote-bar');
            bar.classList.remove('preview');
            get('prompt-input').focus();
            schedulePromptTokenEstimate();
        };

        window.clearQuote = () => {
            currentQuote = "";
            quotePreviewText = "";
            const bar = get('quote-bar');
            bar.classList.remove('preview');
            bar.classList.remove('visible');
            get('quote-text-display').innerText = "";
            schedulePromptTokenEstimate();
        };

        const MODELS = [
            {
                category: "Gemini 3.8 / 3.7 / 3.6 / 3.5",
                icon: "fas fa-star text-yellow-400",
                description: "Google's latest multimodal models",
                items: [
                    { id: "gemini-3.8-flash", implementedAt: "2026-09-05", implementedRank: 9160, quickEmoji: "⚡", name: "Gemini 3.8 Flash", desc: "Most intelligent Flash model for long-horizon software engineering, autonomous agents, and complex enterprise workflows.", price: "In $0.75/1M, Out $3.75/1M (through 2026-12-31)", agenticView: true },
                    { id: "gemini-3.7-flash", implementedAt: "2026-08-14", implementedRank: 8000, quickEmoji: "⚡", name: "Gemini 3.7 Flash", desc: "Most capable Flash model for complex coding, agentic workflows, and multimodal tasks.", price: "In $0.75/1M, Out $3.75/1M (introductory)", agenticView: true },
                    { id: "gemini-3.6-flash", implementedAt: "2026-07-30", implementedRank: 6411, quickEmoji: "⚡", name: "Gemini 3.6 Flash", desc: "Latest Flash model for agentic, coding, and multimodal tasks.", price: "In $1.50/1M, Out $7.50/1M", agenticView: true },
                    { id: "gemini-3.5-flash", implementedAt: "2026-06-13", implementedRank: 5900, quickEmoji: "✨", name: "Gemini 3.5 Flash", desc: "Most intelligent Gemini 3.5 model built for speed.", price: "In $1.50/1M, Out $9.00/1M", agenticView: true },
                    { id: "gemini-3.5-flash-lite", implementedAt: "2026-07-30", implementedRank: 6410, quickEmoji: "🚀", name: "Gemini 3.5 Flash-Lite", desc: "Fastest, lowest-cost Gemini 3.5 model for high-throughput execution.", price: "In $0.30/1M, Out $2.50/1M", agenticView: true },
                ]
            },
            {
                category: "Gemini 3.1 / Previous",
                icon: "fas fa-star text-yellow-400",
                description: "Previous Gemini 3.x generation models",
                items: [
                    { id: "gemini-3.1-flash-lite", implementedAt: "2026-07-30", implementedRank: 6440, quickEmoji: "💨", name: "Gemini 3.1 Flash-Lite", desc: "Stable, cost-efficient model for high-volume lightweight tasks.", price: "In $0.25/1M, Out $1.50/1M", agenticView: true },
                    { id: "gemini-3.1-pro-preview", implementedAt: "2026-02-20", implementedRank: 2430, name: "Gemini 3.1 Pro", desc: "Next-gen native multimodal model.", price: "In $2.00/1M, Out $12.00/1M (≤200k)" },
                    { id: "gemini-3.1-flash-lite-preview", implementedAt: "2026-03-04", implementedRank: 3000, name: "Gemini 3.1 Flash-Lite Preview", desc: "Retired preview model retained for chat history compatibility.", price: "In $0.25/1M, Out $1.50/1M", deprecated: true },
                    { id: "gemini-3-flash-preview", implementedAt: "2026-06-13", implementedRank: 5930, name: "Gemini 3.0 Flash", desc: "Fastest and most cost-efficient.", price: "In $0.50/1M, Out $3.00/1M" },
                    { id: "gemini-3-pro-preview", implementedAt: "2026-01-15", implementedRank: 100, name: "Gemini 3.0 Pro", desc: "Shut down by Google (March 2026). Retained for chat history compatibility.", price: "In $2.00/1M, Out $12.00/1M (≤200k)", deprecated: true }
                ]
            },
            {
                category: "Gemini 2.5",
                icon: "fas fa-history text-gray-400",
                description: "Gemini 2.5 generation models",
                items: [
                    { id: "gemini-2.5-pro", implementedAt: "2026-08-25", implementedRank: 8524, quickEmoji: "🧠", name: "Gemini 2.5 Pro", desc: "Most advanced Gemini 2.5 model for complex reasoning, coding, and long-context analysis.", price: "In $1.25/1M (≤200k), Out $10.00/1M (≤200k)" },
                    { id: "gemini-2.5-flash-lite", implementedAt: "2026-02-07", implementedRank: 1530, name: "Gemini 2.5 Flash-Lite", desc: "Fastest and most cost-efficient Gemini 2.5 model.", price: "In $0.10/1M, Out $0.40/1M" },
                    { id: "gemini-2.5-flash", implementedAt: "2026-02-07", implementedRank: 1531, name: "Gemini 2.5 Flash", desc: "Balanced performance.", price: "In $0.30/1M, Out $2.50/1M" }
                ]
            },
            {
                category: "Gemini Image (Banana)",
                icon: "fas fa-image text-pink-400",
                description: "Gemini image generation models",
                items: [
                    { id: "gemini-2.5-flash-image", implementedAt: "2026-01-20", implementedRank: 120, quickEmoji: "🍌", name: "Nano Banana", desc: "Fast image generation.", price: "In $0.30/1M, Out $0.039/image" },
                    { id: "gemini-3.1-flash-image", implementedAt: "2026-08-25", implementedRank: 8526, quickEmoji: "🍌", name: "Nano Banana 2", desc: "High-efficiency image generation and editing (stable).", price: "In $0.50/1M; Text/Thinking Out $3.00/1M; Image Out $60.00/1M ($0.067/1K image)" },
                    { id: "gemini-3.1-flash-image-preview", implementedAt: "2026-02-26", implementedRank: 2860, name: "Nano Banana 2 (Preview)", desc: "Retired preview retained for chat history compatibility. Use gemini-3.1-flash-image.", price: "In $0.50/1M, Out $0.067/1K image ($60/1M img tokens)", deprecated: true },
                    { id: "gemini-3.1-flash-lite-image", implementedAt: "2026-07-01", implementedRank: 6020, quickEmoji: "🍌", name: "Nano Banana 2 Lite", desc: "Low-latency Gemini image generation and editing with 1K output.", price: "In $0.25/1M; Text/Thinking Out $1.50/1M; Image Out $30/1M ($0.0336/1K image)" },
                    { id: "gemini-3-pro-image", implementedAt: "2026-08-25", implementedRank: 8525, quickEmoji: "🍌", name: "Nano Banana Pro", desc: "Professional image generation and editing with 4K output (stable).", price: "In $2.00/1M; Text/Thinking Out $12.00/1M; Image Out $120.00/1M ($0.134/1K-2K, $0.24/4K)" },
                    { id: "gemini-3-pro-image-preview", implementedAt: "2026-01-25", implementedRank: 130, name: "Nano Banana Pro (Preview)", desc: "Retired preview retained for chat history compatibility. Use gemini-3-pro-image.", price: "In $2.00/1M, Out $0.134 (1K/2K) or $0.24 (4K)", deprecated: true }
                ]
            },
            {
                category: "Gemini Video Generation",
                icon: "fas fa-clapperboard text-cyan-400",
                description: "Gemini video generation models (Veo 3.1 / Omni Flash)",
                items: [
                    { id: "gemini-omni-1.1-flash", implementedAt: "2026-09-02", implementedRank: 9010, quickEmoji: "🎬", name: "Gemini Omni 1.1 Flash", desc: "Fastest multimodal video generation and conversational editing from text, images, video, and audio (native audio in output).", price: "In $1.50/1M (text/image/video/audio); Text Out $9.00/1M; Video $17.50/1M (≈$0.10/sec)" },
                    { id: "gemini-omni-flash", implementedAt: "2026-08-25", implementedRank: 8522, quickEmoji: "🎬", name: "Gemini Omni Flash", desc: "Fast conversational video generation and editing from text and images.", price: "In $1.50/1M; Text Out $9.00/1M; Video ≈$0.10/sec" },
                    { id: "veo-3.1-generate-preview", implementedAt: "2026-08-25", implementedRank: 8521, quickEmoji: "🎥", name: "Veo 3.1", desc: "Cinematic video generation with native audio and 4K output.", price: "$0.40/sec (720p/1080p), $0.60/sec (4K)" },
                    { id: "veo-3.1-fast-generate-preview", implementedAt: "2026-08-25", implementedRank: 8520, name: "Veo 3.1 Fast", desc: "Low-cost, fast video generation from the Veo 3.1 family.", price: "$0.10/sec (720p), $0.12/sec (1080p)" },
                    { id: "veo-3.1-lite-generate-preview", implementedAt: "2026-08-25", implementedRank: 8519, name: "Veo 3.1 Lite", desc: "High-efficiency, developer-first video generation (no 4K).", price: "$0.05/sec (720p), $0.08/sec (1080p)" }
                ]
            },
            {
                category: "Gemini Music Generation",
                icon: "fas fa-music text-fuchsia-400",
                description: "Lyria music generation models",
                items: [
                    { id: "lyria-3.5", implementedAt: "2026-09-05", implementedRank: 9050, quickEmoji: "🎼", name: "Lyria 3.5", desc: "Full-length song generation from text or images with vocals, lyrics, and structured arrangements.", price: "See Google AI pricing" },
                    { id: "lyria-3-pro-preview", implementedAt: "2026-08-25", implementedRank: 8518, quickEmoji: "🎵", name: "Lyria 3 Pro", desc: "Flagship music generation for full-length songs with structural coherence.", price: "$0.08 / song" },
                    { id: "lyria-3-clip-preview", implementedAt: "2026-08-25", implementedRank: 8517, quickEmoji: "🎶", name: "Lyria 3 Clip", desc: "Short musical clips, loops, and previews (30 seconds).", price: "$0.04 / song" },
                    { id: "lyria-realtime-exp", implementedAt: "2026-08-25", implementedRank: 8516, name: "Lyria RealTime", desc: "Experimental realtime music generation with deep melodic control.", price: "Experimental (no vocals)" }
                ]
            },
            {
                category: "Gemini Transcription",
                icon: "fas fa-microphone text-teal-400",
                description: "Gemini speech-to-text transcription models",
                items: [
                    { id: "gemini-3.5-transcribe", implementedAt: "2026-08-27", implementedRank: 8621, quickEmoji: "🎙️", name: "Gemini 3.5 Transcribe", desc: "Audio-file speech-to-text with language detection, speaker diarization, word timestamps, and smart formatting (audio file up to 1 hour).", price: "In $2.00/1M (audio), Out $12.00/1M (text)" },
                    { id: "gemini-3.5-transcribe-live", implementedAt: "2026-08-27", implementedRank: 8622, quickEmoji: "🔴", name: "Gemini 3.5 Transcribe Live", desc: "Real-time low-latency streaming speech-to-text over the Live API (microphone input, sessions up to 10 minutes).", price: "In $3.50/1M (audio), Out $21.00/1M (text)" }
                ]
            },
            {
                category: "OpenAI Image Gen",
                icon: "fas fa-paint-brush text-purple-400",
                description: "GPT Image models",
                items: [
                    { id: "gpt-image-2", implementedAt: "2026-04-30", implementedRank: 4680, name: "GPT Image 2", desc: "State-of-the-art image generation and editing.", price: "Text In $5/1M; Image In $8/1M; Image Out $30/1M" },
                    { id: "gpt-image-1.5", implementedAt: "2026-03-13", implementedRank: 3410, name: "GPT Image 1.5", desc: "Previous-generation flagship image model.", price: "Text In $5/1M, Text Out $10/1M; Image Out $32/1M" },
                    { id: "gpt-image-1", implementedAt: "2026-03-13", implementedRank: 3411, name: "GPT Image 1", desc: "Standard quality.", price: "Text In $5/1M; Image Out $40/1M" },
                    { id: "gpt-image-1-mini", implementedAt: "2026-03-13", implementedRank: 3412, name: "GPT Image 1 Mini", desc: "Faster, lower resolution.", price: "Text In $2/1M; Image In $2.50/1M; Image Out $8/1M" }
                ]
            },
            {
                category: "OpenAI GPT",
                icon: "fas fa-brain text-green-400",
                description: "OpenAI's flagship models",
                items: [
                    { id: "gpt-5.6-sol", implementedAt: "2026-07-31", implementedRank: 6550, quickEmoji: "☀️", name: "GPT-5.6 Sol", desc: "Frontier reasoning model for complex professional work with 1.05M context.", price: "In $5.00/1M, Cached $0.50/1M, Out $30.00/1M (over 272K: In $10.00, Out $45.00)" },
                    { id: "gpt-5.6-terra", implementedAt: "2026-07-31", implementedRank: 6560, quickEmoji: "🌍", name: "GPT-5.6 Terra", desc: "Balanced intelligence and cost for everyday work with 1.05M context.", price: "In $2.00/1M, Cached $0.20/1M, Out $12.00/1M (over 272K: In $4.00, Out $18.00)" },
                    { id: "gpt-5.6-luna", implementedAt: "2026-07-31", implementedRank: 6561, quickEmoji: "🌙", name: "GPT-5.6 Luna", desc: "Cost-efficient model for high-volume workloads with 1.05M context.", price: "In $0.20/1M, Cached $0.02/1M, Out $1.20/1M (over 272K: In $0.40, Out $1.80)" },
                    { id: "gpt-4o", implementedAt: "2026-06-04", implementedRank: 5820, name: "GPT-4o", desc: "Multimodal flagship model.", price: "In $2.50/1M, Out $10.00/1M" },
                    { id: "gpt-4o-mini", implementedAt: "2026-06-04", implementedRank: 5821, name: "GPT-4o mini", desc: "Fast, low-cost model.", price: "In $0.15/1M, Out $0.60/1M" },
                    { id: "gpt-5.5", implementedAt: "2026-04-26", implementedRank: 4500, name: "GPT-5.5", desc: "Experimental OpenAI model ID for accounts with access.", price: "In $5.00/1M, Out $30.00/1M" },
                    { id: "gpt-5.5-mini", implementedAt: "2026-04-26", implementedRank: 4501, name: "GPT-5.5 mini", desc: "Smaller and more cost-efficient GPT-5.5 tier.", price: "Pricing not publicly listed" },
                    { id: "gpt-5.5-nano", implementedAt: "2026-04-26", implementedRank: 4502, name: "GPT-5.5 nano", desc: "Smallest and fastest GPT-5.5 tier.", price: "Pricing not publicly listed" },
                    { id: "gpt-5.5-pro", implementedAt: "2026-04-26", implementedRank: 4503, name: "GPT-5.5 Pro", desc: "Higher-capacity GPT-5.5 tier for accounts with access.", price: "In $30.00/1M, Out $180.00/1M" },
                    { id: "gpt-5.4", implementedAt: "2026-03-08", implementedRank: 3150, name: "GPT-5.4", desc: "Experimental OpenAI model ID for accounts with access.", price: "In $2.50/1M, Out $15.00/1M" },
                    { id: "gpt-5.4-mini", implementedAt: "2026-03-08", implementedRank: 3151, name: "GPT-5.4 mini", desc: "Smaller and more cost-efficient GPT-5.4 tier.", price: "In $0.75/1M, Out $4.50/1M" },
                    { id: "gpt-5.4-nano", implementedAt: "2026-03-08", implementedRank: 3152, name: "GPT-5.4 nano", desc: "Smallest and fastest GPT-5.4 tier.", price: "In $0.20/1M, Out $1.25/1M" },
                    { id: "gpt-5.4-pro", implementedAt: "2026-03-08", implementedRank: 3153, name: "GPT-5.4 Pro", desc: "Higher-capacity GPT-5.4 tier for accounts with access.", price: "In $30.00/1M, Out $180.00/1M" },
                    { id: "gpt-5.2", implementedAt: "2026-02-15", implementedRank: 200, name: "GPT-5.2 (Responses API)", desc: "Most capable reasoning model.", price: "In $1.75/1M, Out $14.00/1M" },
                    { id: "gpt-5-search-api", implementedAt: "2026-02-02", implementedRank: 740, name: "GPT-5 Search (API)", desc: "Search-optimized model (Chat Completions).", price: "Model rates + Web search $10/1k calls" },
                    { id: "gpt-5.1", implementedAt: "2026-02-05", implementedRank: 200, name: "GPT-5.1", desc: "High intelligence.", price: "In $1.25/1M, Out $10.00/1M" },
                    { id: "gpt-5-mini", implementedAt: "2026-02-02", implementedRank: 770, name: "GPT-5 mini", desc: "Small and efficient.", price: "In $0.25/1M, Out $2.00/1M" }
                ]
            },
            {
                category: "DeepSeek V4",
                icon: "fas fa-bolt text-cyan-400",
                description: "DeepSeek's OpenAI-compatible text models",
                items: [
                    { id: "deepseek-v4-flash-vision-exp", implementedAt: "2026-08-23", implementedRank: 8260, quickEmoji: "👁️", name: "DeepSeek V4 Flash Vision Exp", desc: "Experimental V4 Flash with native image input (JPEG/PNG/GIF/WebP), 1M context, up to 384K output, thinking, tools, and JSON output.", price: "In $0.007/1M (hit), $0.22/1M (miss), Out $0.66/1M (off-peak)" },
                    { id: "deepseek-v4-flash-0731", implementedAt: "2026-07-31", implementedRank: 6610, quickEmoji: "⚡", apiId: "deepseek-v4-flash", name: "DeepSeek V4 Flash", desc: "Official V4 Flash release with 1M context, up to 384K output, thinking, tools, and JSON output.", price: "In $0.0028/1M (hit), $0.14/1M (miss), Out $0.28/1M" },
                    { id: "deepseek-v4-flash", implementedAt: "2026-04-26", implementedRank: 4510, name: "DeepSeek V4 Flash Preview", desc: "Retired preview key retained for chat history compatibility.", price: "Legacy preview", deprecated: true },
                    { id: "deepseek-v4-pro", implementedAt: "2026-04-26", implementedRank: 4511, name: "DeepSeek V4 Pro", desc: "Higher-capacity DeepSeek V4 model with 1M context and up to 384K output.", price: "In $0.003625/1M (hit), $0.435/1M (miss), Out $0.87/1M" }
                ]
            },
            {
                category: "Kimi K3",
                icon: "fas fa-brain text-violet-400",
                description: "Moonshot AI's flagship 2.8T-parameter model with 1M context and always-on thinking",
                items: [
                    { id: "kimi-k3", implementedAt: "2026-07-30", implementedRank: 6340, quickEmoji: "🧠", name: "Kimi K3", desc: "Always-reasoning flagship model with 1M context, vision, tool calling.", price: "In $3.00/1M (miss), $0.30/1M (hit), Out $15.00/1M" }
                ]
            },
            {
                category: "Mistral Document OCR",
                icon: "fas fa-file text-orange-300",
                description: "Document OCR (PDF / image / DOCX / PPTX). Not a chat completion model.",
                items: [
                    { id: "mistral-ocr-4-0", implementedAt: "2026-08-15", implementedRank: 8130, quickEmoji: "📄", name: "Mistral OCR 4", desc: "Document AI OCR with markdown, tables, headers/footers, and paragraph bounding boxes. Chat history is not sent.", price: "$4 / 1,000 pages ($5 / 1,000 annotated pages)" }
                ]
            },
            {
                category: "Anthropic Claude",
                icon: "fas fa-brain text-orange-400",
                description: "Anthropic's latest deep reasoning models",
                items: [
                    { id: "claude-opus-4-6", implementedAt: "2026-05-01", implementedRank: 480, name: "Claude Opus 4.6", desc: "Most capable model for deep reasoning and complex tasks.", price: "In $5.00/1M, Out $25.00/1M" },
                    { id: "claude-sonnet-4-6", implementedAt: "2026-05-01", implementedRank: 481, name: "Claude Sonnet 4.6", desc: "Excellent balance of speed and intelligence with adaptive thinking.", price: "In $3.00/1M, Out $15.00/1M" }
                ]
            },
            {
                category: "Audio (TTS)",
                icon: "fas fa-microphone text-red-400",
                description: "Text-to-Speech models",
                items: [
                    { id: "gemini-3.1-flash-tts-preview", implementedAt: "2026-04-17", implementedRank: 4250, name: "Gemini 3.1 Flash TTS", desc: "Google TTS (Preview).", price: "Text In $1.00/1M, Audio Out $20.00/1M" },
                    { id: "gpt-4o-mini-tts", implementedAt: "2026-03-01", implementedRank: 250, name: "GPT-4o Mini TTS", desc: "OpenAI TTS.", price: "Text In $0.60/1M, Audio Out $12.00/1M" },
                    { id: "gemini-2.5-flash-preview-tts", implementedAt: "2026-02-10", implementedRank: 160, name: "Gemini 2.5 Flash TTS", desc: "Google TTS (Preview).", price: "Text In $0.50/1M, Audio Out $10.00/1M" },
                    { id: "gemini-2.5-pro-preview-tts", implementedAt: "2026-02-10", implementedRank: 161, name: "Gemini 2.5 Pro TTS", desc: "Google TTS Pro (Preview).", price: "Text In $1.00/1M, Audio Out $20.00/1M" },
                    { id: "google-tts-studio", implementedAt: "2026-01-20", implementedRank: 110, name: "Google TTS (Studio)", desc: "High fidelity studio voices.", price: "$160 / 1M chars" },
                    { id: "google-tts-neural", implementedAt: "2026-01-20", implementedRank: 111, name: "Google TTS (Neural2)", desc: "Standard neural voices.", price: "$16 / 1M chars" },
                    { id: "grok-tts", implementedAt: "2026-05-27", implementedRank: 5560, quickEmoji: "🔊", name: "Grok TTS", desc: "xAI Text-to-Speech with expressive voices.", price: "$15.00 / 1M chars" }
                ]
            },
            {
                category: "OpenAI Transcription",
                icon: "fas fa-closed-captioning text-emerald-400",
                description: "Speech-to-text models (audio in / text out)",
                items: [
                    { id: "gpt-transcribe", implementedAt: "2026-07-29", implementedRank: 6330, name: "GPT Transcribe", desc: "High-accuracy file and committed-turn transcription.", price: "$0.0045 / minute" },
                    { id: "gpt-live-transcribe", implementedAt: "2026-07-29", implementedRank: 6331, name: "GPT Live Transcribe", desc: "Low-latency realtime transcription.", price: "$0.017 / minute" }
                ]
            },
            {
                category: "Realtime Audio (STS)",
                icon: "fas fa-headset text-cyan-400",
                description: "Realtime voice models (audio in / audio out)",
                items: [
                    { id: "gpt-realtime-2", implementedAt: "2026-05-11", implementedRank: 5080, name: "OpenAI Realtime 2", desc: "Most capable speech-to-speech reasoning model.", price: "Audio In $32/1M, Audio Out $64/1M" },
                    { id: "gpt-realtime-translate", implementedAt: "2026-05-11", implementedRank: 5081, name: "OpenAI Realtime Translate", desc: "Streaming speech-to-speech translation.", price: "$0.034 / minute" },
                    { id: "gpt-realtime-whisper", implementedAt: "2026-05-11", implementedRank: 5082, name: "OpenAI Realtime Whisper", desc: "Streaming speech-to-text (transcription).", price: "$0.017 / minute" },
                    { id: "gpt-realtime-1.5", implementedAt: "2026-02-24", implementedRank: 2530, name: "OpenAI Realtime 1.5", desc: "Latest OpenAI speech-to-speech flagship model.", price: "Audio In $32/1M, Audio Out $64/1M" },
                    { id: "gpt-realtime", implementedAt: "2026-02-24", implementedRank: 2531, name: "OpenAI Realtime", desc: "OpenAI realtime speech-to-speech model.", price: "Audio In $32/1M, Audio Out $64/1M" },
                    { id: "gpt-realtime-mini", implementedAt: "2026-02-24", implementedRank: 2532, name: "OpenAI Realtime Mini", desc: "Lower-latency, smaller realtime model.", price: "Audio In $10/1M, Audio Out $20/1M" },
                    { id: "gemini-2.5-flash-native-audio-preview-12-2025", implementedAt: "2026-01-15", implementedRank: 90, name: "Gemini 2.5 Flash Native Audio (Live)", desc: "Google Live native audio model.", price: "Audio In $3.00/1M, Audio Out $12.00/1M" },
                    { id: "gemini-3.1-flash-live-preview", implementedAt: "2026-03-29", implementedRank: 3870, name: "Gemini 3.1 Flash Live", desc: "Google Live native audio model.", price: "Audio In $3.00/1M (~$0.005/min), Out $12.00/1M" },
                    { id: "gemini-3.5-live-translate-preview", implementedAt: "2026-08-25", implementedRank: 8523, quickEmoji: "🌐", name: "Gemini 3.5 Live Translate", desc: "Low-latency real-time speech-to-speech translation supporting 70+ languages.", price: "Audio In $3.50/1M, Audio Out $21.00/1M" },
                    { id: "grok-voice-think-fast-2.0", implementedAt: "2026-08-25", implementedRank: 8502, quickEmoji: "🎤", name: "Grok Voice Think Fast 2.0", desc: "Current xAI speech-to-speech model.", price: "$0.08 / min ($4.80 / hr) audio + $0.004 / text input" },
                    { id: "grok-voice-latest", implementedAt: "2026-05-27", implementedRank: 5550, name: "Grok Voice Latest", desc: "Alias for the current flagship voice model.", price: "$0.08 / min ($4.80 / hr) audio + $0.004 / text input" },
                    { id: "grok-voice-think-fast-1.0", implementedAt: "2026-05-11", implementedRank: 5140, name: "Grok Voice Think Fast 1.0", desc: "Deprecated xAI realtime voice model retained for history compatibility.", price: "$0.05 / min ($3.00 / hr)", deprecated: true },
                    { id: "grok-voice-fast-1.0", implementedAt: "2026-05-01", implementedRank: 500, name: "Grok Voice Fast 1.0", desc: "Legacy xAI realtime voice model retained for history compatibility.", price: "$0.05 / min ($3.00 / hr)", deprecated: true },
                    { id: "grok-voice-agent", implementedAt: "2026-04-01", implementedRank: 380, name: "Grok Voice Agent", desc: "xAI realtime voice agent API.", price: "$0.05 / min (Realtime)", deprecated: true }
                ]
            },
            {
                category: "Gemini Agent / Specialized",
                icon: "fas fa-robot text-indigo-400",
                description: "Gemini agent and specialized models",
                items: [
                    { id: "gemini-robotics-er-2-preview", implementedAt: "2026-08-25", implementedRank: 8515, name: "Gemini Robotics ER 2", desc: "Embodied reasoning model for robots with advanced video understanding.", price: "In $2.00/1M, Out $8.00/1M" },
                    { id: "deep-research-preview-04-2026", implementedAt: "2026-08-25", implementedRank: 8514, quickEmoji: "🔎", name: "Gemini Deep Research", desc: "Agentic multi-step research producing comprehensive cited reports.", price: "Standard Gemini rates + tool usage fees" },
                    { id: "deep-research-max-preview-04-2026", implementedAt: "2026-08-25", implementedRank: 8513, name: "Gemini Deep Research Max", desc: "Maximum-comprehension research agent over hundreds of sources.", price: "Standard Gemini rates + tool usage fees" },
                    { id: "antigravity-preview-05-2026", implementedAt: "2026-08-25", implementedRank: 8512, name: "Antigravity Agent", desc: "Managed agent that plans, runs code, manages files, and browses the web in a sandbox.", price: "Standard Gemini rates (sandbox compute free during preview)" },
                    { id: "gemini-2.5-computer-use-preview-10-2025", implementedAt: "2026-08-25", implementedRank: 8511, name: "Gemini 2.5 Computer Use", desc: "Browser / desktop control agent model for UI automation.", price: "In $1.25/1M (≤200k), Out $10.00/1M (≤200k)" },
                    { id: "gemini-embedding-2", implementedAt: "2026-08-25", implementedRank: 8510, name: "Gemini Embedding 2", desc: "Multimodal embedding model (text / image / audio / video / PDF).", price: "Text In $0.20/1M, Image $0.45/1M" }
                ]
            },
            {
                category: "Grok Imagine",
                icon: "fas fa-magic text-blue-400",
                description: "Grok generation models",
                items: [
                    { id: "grok-imagine-image-2.0", implementedAt: "2026-08-22", implementedRank: 8250, quickEmoji: "🎨", name: "Grok Imagine Image 2.0", desc: "Precise image generation and editing with 1K/2K output and low/medium quality control.", price: "from $0.04 / image" },
                    { id: "grok-imagine-image-quality", implementedAt: "2026-05-09", implementedRank: 5020, name: "Grok Imagine Image Quality", desc: "Next-gen Grok image generation with 1K/2K support.", price: "$0.05 / image" },
                    { id: "grok-imagine-image", implementedAt: "2026-01-30", implementedRank: 520, name: "Grok Imagine Image", desc: "Latest Grok image generation.", price: "$0.02 / image" },
                    { id: "grok-imagine-image-pro", implementedAt: "2026-02-01", implementedRank: 530, name: "Grok Imagine Image Pro", desc: "Discontinued by xAI. Retained for chat history compatibility.", price: "$0.07 / image", deprecated: true },
                    { id: "grok-imagine-video-1.5", implementedAt: "2026-08-25", implementedRank: 8501, quickEmoji: "🎬", name: "Grok Imagine Video 1.5", desc: "Current xAI video generation model with 1080p text/image-to-video support.", price: "$0.080 / second" },
                    { id: "grok-imagine-video", implementedAt: "2026-01-30", implementedRank: 530, name: "Grok Imagine Video", desc: "Legacy Grok video generation.", price: "$0.05 / second" }
                ]
            },
            {
                category: "xAI Grok",
                icon: "fas fa-rocket text-white",
                description: "Models by xAI",
                items: [
                    { id: "grok-4.6", implementedAt: "2026-08-19", implementedRank: 8161, name: "Grok 4.6", desc: "Frontier model for coding, agentic tasks, and knowledge work.", price: "In $2.00/1M, Out $6.00/1M" },
                    { id: "grok-4.5", implementedAt: "2026-08-19", implementedRank: 8160, name: "Grok 4.5", desc: "Intelligent coding model for agentic software and engineering tasks.", price: "In $2.00/1M, Out $6.00/1M" },
                    { id: "grok-4.3", implementedAt: "2026-05-27", implementedRank: 5530, name: "Grok 4.3", desc: "Most intelligent and fastest flagship model.", price: "In $1.25/1M, Out $2.50/1M" },
                    { id: "grok-build-0.1", implementedAt: "2026-05-27", implementedRank: 5520, quickEmoji: "🛠️", name: "Grok Build 0.1 (Coding)", desc: "Fast agentic coding model with vision and reasoning support.", price: "In $1.00/1M, Out $2.00/1M" },
                    { id: "grok-4.20-0309-reasoning", implementedAt: "2026-08-25", implementedRank: 8503, name: "Grok 4.20 (Reasoning, 0309)", desc: "Dated Grok 4.20 reasoning release.", price: "In $1.25/1M, Out $2.50/1M" },
                    { id: "grok-4.20-0309-non-reasoning", implementedAt: "2026-08-25", implementedRank: 8504, name: "Grok 4.20 (Non-Reasoning, 0309)", desc: "Dated Grok 4.20 standard release.", price: "In $1.25/1M, Out $2.50/1M" },
                    { id: "grok-4.20-multi-agent-0309", implementedAt: "2026-08-25", implementedRank: 8505, name: "Grok 4.20 Multi-Agent (0309)", desc: "Dated Grok 4.20 multi-agent release.", price: "In $1.25/1M, Out $2.50/1M" },
                    { id: "grok-4.20-reasoning", implementedAt: "2026-04-09", implementedRank: 4000, name: "Grok 4.20 (Reasoning)", desc: "Flagship reasoning model.", price: "In $1.25/1M, Out $2.50/1M" },
                    { id: "grok-4.20-non-reasoning", implementedAt: "2026-04-09", implementedRank: 4001, name: "Grok 4.20 (Non-Reasoning)", desc: "Flagship standard model.", price: "In $1.25/1M, Out $2.50/1M" },
                    { id: "grok-4.20-multi-agent", implementedAt: "2026-04-09", implementedRank: 4002, name: "Grok 4.20 Multi-Agent", desc: "Agentic flagship model.", price: "In $1.25/1M, Out $2.50/1M" },
                    { id: "grok-4-1-fast-reasoning", implementedAt: "2026-03-01", implementedRank: 280, name: "Grok 4.1 Fast (Reasoning)", desc: "Fast with reasoning capabilities.", price: "In $0.20/1M, Out $0.50/1M", deprecated: true },
                    { id: "grok-4-1-fast-non-reasoning", implementedAt: "2026-03-01", implementedRank: 281, name: "Grok 4.1 Fast (Non-Reasoning)", desc: "Fast standard model.", price: "In $0.20/1M, Out $0.50/1M", deprecated: true },
                    { id: "grok-4-fast-reasoning", implementedAt: "2026-02-01", implementedRank: 150, name: "Grok 4 Fast (Reasoning)", desc: "Previous gen reasoning.", price: "In $0.20/1M, Out $0.50/1M", deprecated: true },
                    { id: "grok-4-fast-non-reasoning", implementedAt: "2026-02-01", implementedRank: 151, name: "Grok 4 Fast (Non-Reasoning)", desc: "Previous gen standard.", price: "In $0.20/1M, Out $0.50/1M", deprecated: true }
                ]
            }
        ];


        const WELCOME_QUICK_START_LIMIT = 5;
        const listModelsFlat = () => {
            const out = [];
            MODELS.forEach((group) => {
                (group.items || []).forEach((item) => {
                    if (item && item.id) out.push(item);
                });
            });
            return out;
        };
        const compareModelsByImplementedAt = (a, b) => {
            const da = String(a && a.implementedAt || '');
            const db = String(b && b.implementedAt || '');
            if (da !== db) return db.localeCompare(da);
            const ra = Number(a && a.implementedRank || 0);
            const rb = Number(b && b.implementedRank || 0);
            if (ra !== rb) return rb - ra;
            return String(a && a.id || '').localeCompare(String(b && b.id || ''));
        };
        const getRecentModelsForQuickStart = (limit = WELCOME_QUICK_START_LIMIT) => {
            return listModelsFlat()
                .filter((m) => m && m.id && !m.deprecated && m.implementedAt)
                .sort(compareModelsByImplementedAt)
                .slice(0, Math.max(0, Number(limit) || 0));
        };
        const renderWelcomeQuickStart = () => {
            const grid = get('welcome-quick-start');
            if (!grid) return;
            const models = getRecentModelsForQuickStart(WELCOME_QUICK_START_LIMIT);
            if (!models.length) {
                grid.innerHTML = '';
                return;
            }
            grid.innerHTML = models.map((m, idx) => {
                const delay = (0.1 + idx * 0.02).toFixed(2);
                const emoji = m.quickEmoji ? `${escapeHtml(String(m.quickEmoji))} ` : '';
                const name = escapeHtml(String(m.name || m.id));
                const id = String(m.id).replace(/\\/g, '\\\\').replace(/'/g, "\\'");
                return `<button type="button" class="welcome-btn p-3 rounded text-sm text-left transition btn-hover slide-in-animate" style="animation-delay: ${delay}s" onclick="quickStart('${id}')">${emoji}${name}</button>`;
            }).join('');
        };

        const normalizeModelApiKeyMap = (raw) => {
            if (!raw || typeof raw !== 'object') return {};
            const out = {};
            Object.entries(raw).forEach(([modelId, apiKey]) => {
                const mk = String(modelId || '').trim();
                const kv = String(apiKey || '').trim();
                if (!mk || !kv) return;
                out[mk] = kv;
            });
            return out;
        };
        const MODEL_NAME_BY_ID = (() => {
            const map = new Map();
            MODELS.forEach((group) => {
                (group.items || []).forEach((item) => {
                    const id = String(item.id || '').trim();
                    if (!id || map.has(id)) return;
                    map.set(id, String(item.name || id));
                });
            });
            return map;
        })();
        const getModelNameById = (modelId) => {
            const mk = String(modelId || '').trim();
            if (!mk) return '';
            return MODEL_NAME_BY_ID.get(mk) || mk;
        };
        const maskApiKeyPreview = (key) => {
            const txt = String(key || '');
            if (!txt) return '';
            if (txt.length <= 8) return '********';
            return `${txt.slice(0, 4)}...${txt.slice(-4)}`;
        };
        const getModelProviderInfo = (modelId) => {
            const id = String(modelId || '').toLowerCase().trim();
            if (!id) return null;
            if (id.startsWith('gemini') || id.startsWith('veo-') || id.startsWith('lyria-') || id.startsWith('deep-research-') || id.startsWith('antigravity-')) return { provider: 'gemini', keyField: 'gemini_key', inputId: 'set-gemini', label: 'Gemini API Key' };
            if (id.startsWith('gpt') || id.startsWith('o1') || id.startsWith('o3')) return { provider: 'openai', keyField: 'openai_key', inputId: 'set-openai', label: 'OpenAI API Key' };
            if (id.startsWith('deepseek')) return { provider: 'deepseek', keyField: 'deepseek_key', inputId: 'set-deepseek', label: 'DeepSeek API Key' };
            if (id.startsWith('kimi')) return { provider: 'kimi', keyField: 'kimi_key', inputId: 'set-kimi', label: 'Kimi (Moonshot) API Key' };
            if (id.startsWith('mistral')) return { provider: 'mistral', keyField: 'mistral_key', inputId: 'set-mistral', label: 'Mistral API Key' };
            if (id.startsWith('claude')) return { provider: 'anthropic', keyField: 'anthropic_key', inputId: 'set-anthropic', label: 'Anthropic API Key' };
            if (id.startsWith('grok')) return { provider: 'xai', keyField: 'xai_key', inputId: 'set-xai', label: 'xAI (Grok) API Key' };
            if (id.startsWith('google')) return { provider: 'google', keyField: 'google_key', inputId: 'set-google-key', label: 'Google API Key (TTS)' };
            return { provider: 'openai', keyField: 'openai_key', inputId: 'set-openai', label: 'OpenAI API Key' };
        };
        const setModelApiKeyPanelOpen = (open) => {
            const panel = get('model-api-keys-panel');
            const btn = get('toggle-model-api-keys-btn');
            if (!panel || !btn) return;
            const show = !!open;
            panel.classList.toggle('hidden', !show);
            btn.innerText = show ? 'モデル別APIキー設定を閉じる' : 'モデル別のAPIキーを設定する';
        };
        const syncModelApiKeyModelOptions = () => {
            const select = get('model-api-key-model');
            if (!select) return;
            const prev = select.value || '';
            select.innerHTML = '';
            const first = document.createElement('option');
            first.value = '';
            first.textContent = 'モデルを選択';
            select.appendChild(first);
            MODELS.forEach((group) => {
                const items = Array.isArray(group.items) ? group.items.filter(m => !m.deprecated) : [];
                if (!items.length) return;
                const optgroup = document.createElement('optgroup');
                optgroup.label = String(group.category || 'Models');
                items.forEach((item) => {
                    const id = String(item.id || '').trim();
                    if (!id) return;
                    const op = document.createElement('option');
                    op.value = id;
                    op.textContent = `${String(item.name || id)} (${id})`;
                    optgroup.appendChild(op);
                });
                if (optgroup.children.length > 0) select.appendChild(optgroup);
            });
            if (prev) {
                const hasPrev = Array.from(select.options).some((op) => op.value === prev);
                if (hasPrev) select.value = prev;
            }
        };
        const renderModelApiKeyList = () => {
            const list = get('model-api-key-list');
            if (!list) return;
            modelApiKeyMap = normalizeModelApiKeyMap(modelApiKeyMap);
            const entries = Object.entries(modelApiKeyMap).sort((a, b) => a[0].localeCompare(b[0]));
            list.innerHTML = '';
            if (!entries.length) {
                const empty = document.createElement('div');
                empty.className = 'text-[11px] text-gray-500';
                empty.textContent = 'モデル別キーは未設定です。';
                list.appendChild(empty);
                return;
            }
            entries.forEach(([modelId, key]) => {
                const row = document.createElement('div');
                row.className = 'flex items-center justify-between gap-3 rounded border border-gray-700 bg-gray-900/70 px-3 py-2';
                const left = document.createElement('div');
                left.className = 'min-w-0';
                const title = document.createElement('div');
                title.className = 'text-[11px] text-gray-200 truncate';
                title.textContent = `${getModelNameById(modelId)} (${modelId})`;
                const keyView = document.createElement('div');
                keyView.className = 'text-[10px] text-cyan-300 font-mono';
                keyView.textContent = maskApiKeyPreview(key);
                left.appendChild(title);
                left.appendChild(keyView);
                const delBtn = document.createElement('button');
                delBtn.type = 'button';
                delBtn.className = 'text-[10px] bg-red-700/80 hover:bg-red-600 text-white px-2 py-1 rounded font-bold btn-hover shrink-0';
                delBtn.textContent = '削除';
                delBtn.onclick = () => {
                    delete modelApiKeyMap[modelId];
                    renderModelApiKeyList();
                    showToast(`モデル別APIキーを削除: ${modelId}`, 'success');
                };
                row.appendChild(left);
                row.appendChild(delBtn);
                list.appendChild(row);
            });
        };
        const bindModelApiKeySettingsControls = () => {
            const toggleBtn = get('toggle-model-api-keys-btn');
            if (toggleBtn && !toggleBtn.dataset.bound) {
                toggleBtn.dataset.bound = '1';
                toggleBtn.addEventListener('click', () => {
                    const panel = get('model-api-keys-panel');
                    setModelApiKeyPanelOpen(panel ? panel.classList.contains('hidden') : true);
                });
            }
            const addBtn = get('model-api-key-apply-btn');
            if (addBtn && !addBtn.dataset.bound) {
                addBtn.dataset.bound = '1';
                addBtn.addEventListener('click', () => {
                    const modelSel = get('model-api-key-model');
                    const keyInput = get('model-api-key-input');
                    const modelId = modelSel ? String(modelSel.value || '').trim() : '';
                    const keyVal = keyInput ? String(keyInput.value || '').trim() : '';
                    if (!modelId) {
                        showToast('モデルを選択してください', 'error', true);
                        return;
                    }
                    if (!keyVal) {
                        showToast('APIキーを入力してください', 'error', true);
                        return;
                    }
                    modelApiKeyMap = normalizeModelApiKeyMap(modelApiKeyMap);
                    modelApiKeyMap[modelId] = keyVal;
                    if (keyInput) keyInput.value = '';
                    renderModelApiKeyList();
                    showToast(`モデル別APIキーを設定: ${modelId}`, 'success');
                });
            }
            const keyInput = get('model-api-key-input');
            if (keyInput && !keyInput.dataset.bound) {
                keyInput.dataset.bound = '1';
                keyInput.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter') {
                        e.preventDefault();
                        const addBtnEl = get('model-api-key-apply-btn');
                        if (addBtnEl) addBtnEl.click();
                    }
                });
            }
            syncModelApiKeyModelOptions();
            renderModelApiKeyList();
            setModelApiKeyPanelOpen(false);
        };

        let activeModelTag = 'all';
        const MODEL_TAGS = ['all','openai','gemini','deepseek','xai','image','audio','reasoning','fast','agentic view'];

        // Slash command system (extensible command palette triggered by / in prompt bar)
        const SLASH_COMMANDS = [
            {
                id: 'settings',
                label: '/settings',
                description: 'AIで自然言語を使って設定を変更（現在選択中のモデルを使用）',
                icon: 'fa-cog',
                example: 'デフォルトモデルを gemini-2.5-flash に変更して thinking をオンに'
            }
            // 将来コマンドをここに追加予定
        ];
        let slashSuggestionsVisible = false;
        let slashSelectedIndex = 0;
        let lastSlashFilter = null; // last filter the palette was rendered with
        let pendingSlashCommand = null; // 'settings' など。コマンド選択後に残る引数テキストで発動
        const AI_SETTINGS_CONVERSATION_KEY = `ai-settings-conversation:${(typeof CHAT_CONFIG !== 'undefined' && CHAT_CONFIG.currentUsername) || 'anonymous'}`;
        let aiSettingsConversation = [];

        function loadAiSettingsConversation() {
            try {
                const raw = sessionStorage.getItem(AI_SETTINGS_CONVERSATION_KEY);
                const parsed = raw ? JSON.parse(raw) : [];
                if (!Array.isArray(parsed)) return [];
                return parsed.filter((item) => item && (item.role === 'user' || item.role === 'assistant') && typeof item.content === 'string')
                    .slice(-10)
                    .map((item) => ({ role: item.role, content: item.content.slice(0, 1600) }));
            } catch (e) {
                return [];
            }
        }

        function persistAiSettingsConversation() {
            try {
                sessionStorage.setItem(AI_SETTINGS_CONVERSATION_KEY, JSON.stringify(aiSettingsConversation.slice(-10)));
            } catch (e) {
                // Private browsing or storage limits must not block settings use.
            }
        }

        function clearAiSettingsConversation() {
            aiSettingsConversation = [];
            try { sessionStorage.removeItem(AI_SETTINGS_CONVERSATION_KEY); } catch (e) {}
        }

        function appendAiSettingsConversation(role, content) {
            const text = String(content || '').trim();
            if (!text) return;
            aiSettingsConversation.push({ role, content: text.slice(0, 1600) });
            aiSettingsConversation = aiSettingsConversation.slice(-10);
            persistAiSettingsConversation();
        }

        aiSettingsConversation = loadAiSettingsConversation();

        function summarizeAiSettingsConversationValues(values, mode) {
            const entries = Object.entries(values || {});
            const prefix = mode === 'inspect' ? '現在の設定を確認しました。' : '設定を更新しました。';
            const details = entries.map(([key, value]) => `${key}: ${formatAiSettingValue(value).slice(0, 180)}`).join('\n');
            return `${prefix}${details ? `\n${details}` : ''}`.slice(0, 1600);
        }

        // Gem suggestion system (triggered by @ in prompt bar)
        let gemSuggestionsVisible = false;
        let gemSelectedIndex = 0;

        const STS_MODELS = new Set([
            'gpt-transcribe',
            'gpt-live-transcribe',
            'gpt-realtime-2',
            'gpt-realtime-translate',
            'gpt-realtime-whisper',
            'gpt-realtime-1.5',
            'gpt-realtime',
            'gpt-realtime-mini',
            'gemini-2.5-flash-native-audio-preview-12-2025',
            'gemini-3.1-flash-live-preview',
            'gemini-3.5-live-translate-preview',
            'gemini-3.5-transcribe-live',
            'grok-voice-think-fast-2.0',
            'grok-voice-latest',
            'grok-voice-think-fast-1.0',
            'grok-voice-fast-1.0',
            'grok-voice-agent'
        ]);
        const FILE_BASE_URL = CHAT_CONFIG.urls.serveFileBase;
        const FILE_THUMB_BASE_URL = CHAT_CONFIG.urls.serveFileThumbBase;
        const RICH_PASTE_PDF_SERVER_ROUTE = CHAT_CONFIG.urls.richPastePdfServer;
        const IMAGE_EXTS = ['png','jpg','jpeg','webp','gif','bmp','avif','heic','heif'];
        const AUDIO_EXTS = ['mp3','wav','aac','ogg','flac','aiff','aif','m4a','opus','oga','weba','webm'];
        const VIDEO_EXTS = ['mp4','mov','avi','mkv','m4v','webm','mpg','mpeg','wmv','3gp','3gpp','flv'];
        const getFileExt = (name) => {
            const text = typeof name === 'string' ? name : (name == null ? '' : String(name));
            if (!text) return '';
            const idx = text.lastIndexOf('.');
            if (idx === -1) return '';
            return text.slice(idx + 1).toLowerCase();
        };
        const normalizeAttachmentPath = (value) => {
            if (!value) return '';
            let v = '';
            if (typeof value === 'string') {
                v = value;
            } else if (typeof value === 'object') {
                v = String(value.path || value.url || value.name || value.filename || value.filepath || '');
            }
            if (!v) return '';

            try {
                if (v.includes('://')) {
                    v = new URL(v, window.location.origin).pathname || '';
                }
            } catch (e) {}
            if (v.includes('?')) v = v.split('?', 1)[0];
            if (v.includes('#')) v = v.split('#', 1)[0];
            v = v.replace(/^\/+/, '');
            if (v.startsWith('files/')) v = v.slice(6);
            try { v = decodeURIComponent(v); } catch (e) {}
            return v;
        };
        const isGeminiImageModelKey = (model) => {
            const m = (model || '').toLowerCase();
            return m.includes('gemini') && (m.includes('image') || m.includes('nano'));
        };
        const isClaudeModelKey = (model) => {
            const m = (model || '').toLowerCase();
            return m.includes('claude');
        };
        /** API provider for a model (prompt-cache lock key). */
        const getModelApiProvider = (modelId) => {
            const m = String(modelId || '').toLowerCase().trim();
            if (!m) return null;
            if (m.includes('claude')) return 'anthropic';
            if (m.includes('deepseek')) return 'deepseek';
            if (m.includes('grok') && !m.includes('gpt')) return 'xai';
            if (m.includes('google-tts')) return 'google';
            if (m.includes('gemini') || m.startsWith('veo-') || m.startsWith('lyria-') || m.startsWith('deep-research-') || m.startsWith('antigravity-')) return 'gemini';
            return 'openai';
        };
        const PROVIDER_LABELS = {
            openai: 'OpenAI',
            gemini: 'Gemini',
            anthropic: 'Anthropic (Claude)',
            xai: 'xAI (Grok)',
            deepseek: 'DeepSeek',
            google: 'Google Cloud'
        };
        const isPromptCacheEnabled = () => {
            const el = get('enable-prompt-cache');
            return !!(el && el.checked);
        };
        const getPromptCacheLockedProvider = () => {
            if (!isPromptCacheEnabled()) return null;
            const modelEl = get('model-select');
            return getModelApiProvider(modelEl ? modelEl.value : '');
        };
        const updatePromptCacheUi = () => {
            const cacheCont = get('prompt-cache-container');
            const cacheChk = get('enable-prompt-cache');
            const modelBtn = get('model-selector-btn');
            if (!cacheChk) return;
            const enabled = !!cacheChk.checked;
            if (cacheCont) {
                cacheCont.classList.toggle('ring-1', enabled);
                cacheCont.classList.toggle('ring-teal-500/50', enabled);
                cacheCont.classList.toggle('rounded', enabled);
                cacheCont.classList.toggle('px-1', enabled);
            }
            if (modelBtn) {
                if (enabled) {
                    modelBtn.title = 'PromptCache有効: 同一APIプロバイダのモデルのみ選択可能';
                    modelBtn.classList.add('border-teal-500/60');
                } else {
                    modelBtn.title = '';
                    modelBtn.classList.remove('border-teal-500/60');
                }
            }
        };
        const bindPromptCacheControls = () => {
            const cacheChk = get('enable-prompt-cache');
            if (!cacheChk || cacheChk.dataset.bound === '1') return;
            cacheChk.dataset.bound = '1';
            cacheChk.addEventListener('change', () => {
                updatePromptCacheUi();
                if (cacheChk.checked) {
                    const prov = getModelApiProvider(get('model-select') ? get('model-select').value : '');
                    const label = PROVIDER_LABELS[prov] || prov || '現在のAPI';
                    showToast(`PromptCache を有効化しました。以降は ${label} 以外のモデルに変更できません。`, 'info', true);
                }
            });
        };
        const getModelMediaSupport = (model) => {
            const m = (model || '').toLowerCase();
            if (!m.includes('gemini')) return { audio: false, video: false };
            if (m.includes('image') || m.includes('nano') || m.includes('tts') || m.includes('native-audio') || m.includes('live')) {
                return { audio: false, video: false };
            }
            if (m.includes('embedding') || m.startsWith('veo-') || m.includes('omni-flash') || m.includes('omni-1.1-flash') || m.startsWith('lyria-')) {
                return { audio: false, video: false };
            }
            return { audio: true, video: true };
        };
        const supportsAudioInputModel = () => getModelMediaSupport(get('model-select').value).audio;
        const supportsVideoInputModel = () => getModelMediaSupport(get('model-select').value).video;
        const isImagePath = (path) => IMAGE_EXTS.includes(getFileExt(path || ''));
        const isAudioPath = (path) => AUDIO_EXTS.includes(getFileExt(path || ''));
        const isVideoPath = (path) => VIDEO_EXTS.includes(getFileExt(path || ''));
        const OPENAI_TTS_VOICES = ['alloy','ash','ballad','coral','echo','fable','nova','onyx','sage','shimmer','verse','marin','cedar'];
        const GEMINI_TTS_VOICES = [
            'Zephyr','Puck','Charon','Kore','Fenrir','Leda','Orus','Aoede','Callirrhoe','Autonoe',
            'Enceladus','Iapetus','Umbriel','Algieba','Despina','Erinome','Algenib','Rasalgethi','Laomedeia','Achernar',
            'Alnilam','Schedar','Gacrux','Pulcherrima','Achird','Zubenelgenubi','Vindemiatrix','Sadachbia','Sadaltager','Sulafat'
        ];
        const OPENAI_STS_VOICES = ['alloy','ash','ballad','coral','echo','sage','shimmer','verse','marin','cedar'];
        const GROK_STS_VOICES = ['Ara','Rex','Sal','Eve','Leo'];
        const GROK_TTS_VOICES = ['Eve','Ara','Rex','Sal','Leo'];
        const GEMINI_STS_VOICES = [
            'Zephyr','Puck','Charon','Kore','Fenrir','Leda','Orus','Aoede','Callirrhoe','Autonoe',
            'Enceladus','Iapetus','Umbriel','Algieba','Despina','Erinome','Algenib','Rasalgethi','Laomedeia','Achernar',
            'Alnilam','Schedar','Gacrux','Pulcherrima','Achird','Zubenelgenubi','Vindemiatrix','Sadachbia','Sadaltager','Sulafat'
        ];
        const GROK_PCM_RATES = [8000,16000,21050,24000,32000,44100,48000];
        const isTtsModel = () => get('model-select').value.includes('tts');
        const isGptImageModel = () => (get('model-select').value || '').includes('gpt-image');
        const isGeminiImageModel = () => isGeminiImageModelKey(get('model-select').value);
        const isMistralOcrModel = (model) => {
            const m = String(model != null ? model : ((get('model-select') && get('model-select').value) || '')).toLowerCase();
            return m === 'mistral-ocr-4-0' || m === 'mistral-ocr-latest' || m.startsWith('mistral-ocr');
        };
        const isLlmModel = () => {
            const m = (get('model-select').value || '').toLowerCase();
            if (isMistralOcrModel(m)) return false;
            if (
                m.includes('tts') ||
                m.includes('transcribe') ||
                m.includes('realtime') ||
                m.includes('voice-agent') ||
                m.includes('native-audio') ||
                m.includes('live') ||
                m.includes('image') ||
                m.includes('video') ||
                isGeminiVideoModelKey(m) ||
                isGeminiMusicModelKey(m) ||
                isGeminiEmbeddingModelKey(m)
            ) return false;
            if (m.includes('gemini') && (m.includes('image') || m.includes('nano'))) return false;
            return m.includes('gpt') || m.includes('gemini') || m.includes('grok') || m.includes('deepseek') || m.startsWith('deep-research-') || m.startsWith('antigravity-');
        };
        const isGrokImageModel = () => {
            const m = (get('model-select').value || '').toLowerCase();
            return m.includes('grok') && (m.includes('imagine') || m.includes('image')) && !m.includes('video');
        };
        const isGrokVideoModel = () => {
            const m = (get('model-select').value || '').toLowerCase();
            return m.includes('grok') && m.includes('video');
        };
        const isGeminiVideoModelKey = (model) => {
            const m = (model || '').toLowerCase();
            return m.startsWith('veo-') || m.includes('omni-flash') || m.includes('omni-1.1-flash');
        };
        const isGeminiVideoModel = () => isGeminiVideoModelKey(get('model-select').value);
        const isGeminiMusicModelKey = (model) => {
            const m = (model || '').toLowerCase();
            return m.startsWith('lyria-');
        };
        const isGeminiMusicModel = () => isGeminiMusicModelKey(get('model-select').value);
        const isGeminiEmbeddingModelKey = (model) => {
            const m = (model || '').toLowerCase();
            return m.includes('gemini-embedding');
        };
        const isGeminiEmbeddingModel = () => isGeminiEmbeddingModelKey(get('model-select').value);

        const isStsModel = () => STS_MODELS.has(get('model-select').value);
        const isTranscriptionModel = () => {
            const model = get('model-select') ? get('model-select').value : '';
            return model === 'gpt-transcribe' || model === 'gpt-live-transcribe';
        };
        const isGeminiLiveModel = () => {
            const m = get('model-select').value;
            return m === 'gemini-3.1-flash-live-preview' || m === 'gemini-3.5-live-translate-preview' || m === 'gemini-3.5-transcribe-live';
        };
        const isGeminiLiveTranslateModel = () => get('model-select').value === 'gemini-3.5-live-translate-preview';
        const isGeminiLiveTranscribeModel = () => get('model-select').value === 'gemini-3.5-transcribe-live';
        const isGeminiRealtimeMusicModel = () => (get('model-select').value || '') === 'lyria-realtime-exp';
        const isLyriaRealtimeModel = () => isGeminiRealtimeMusicModel();
        // True real-time server-session STS models: OpenAI Realtime conversation
        // models, Grok Voice, and Gemini native-audio. Gemini Live models stream
        // browser-direct, transcription models stay one-shot.
        const isRealtimeSessionModel = () => {
            if (!isStsModel()) return false;
            if (isGeminiLiveModel()) return false;
            if (isTranscriptionModel()) return false;
            if (get('model-select') && get('model-select').value === 'gpt-realtime-whisper') return false;
            return true;
        };
        const getStsProvider = (model) => {
            const m = (model || '').toLowerCase();
            if (m.includes('gpt-realtime') || m === 'gpt-transcribe' || m === 'gpt-live-transcribe') return 'openai';
            if (m.includes('grok-voice')) return 'xai';
            if (m.includes('gemini') && (m.includes('native-audio') || m.includes('live'))) return 'gemini';
            return null;
        };
        function setStsStatus(text, recording = false) {
            const s = get('sts-status');
            const b = get('sts-mic-btn');
            if (s && text) s.innerText = text;
            if (b) {
                if (recording) {
                    b.classList.add('bg-red-600', 'animate-pulse');
                    b.classList.remove('bg-cyan-600');
                } else {
                    b.classList.remove('bg-red-600', 'animate-pulse');
                    b.classList.add('bg-cyan-600');
                }
            }
        }
        function updateStsUi() {
            const sts = isStsModel();
            const studioMode = sts && voiceStudioUiEnabled !== false;
            const inputRow = get('input-row');
            const stsPanel = get('sts-panel');
            const studioBar = get('voice-studio-bar');
            const filePreview = get('file-preview');
            if (sts) {
                if (inputRow) inputRow.classList.add('hidden');
                if (filePreview) filePreview.classList.add('hidden');
                if (studioMode) {
                    // The STS panel lives inside the studio modal while it is open.
                    if (stsPanel) {
                        if (window.VoiceStudioOpen) stsPanel.classList.remove('hidden');
                        else stsPanel.classList.add('hidden');
                    }
                    if (studioBar) studioBar.classList.remove('hidden');
                } else {
                    if (stsPanel) stsPanel.classList.remove('hidden');
                    if (studioBar) studioBar.classList.add('hidden');
                    if (window.VoiceStudio) window.VoiceStudio.closeIfOpen();
                }
                setStsStatus('Tap to speak', false);
            } else {
                if (inputRow) inputRow.classList.remove('hidden');
                if (stsPanel) stsPanel.classList.add('hidden');
                if (studioBar) studioBar.classList.add('hidden');
                if (window.VoiceStudio) window.VoiceStudio.closeIfOpen();
            }
        }
        function updateStsOptions() {
            if (!isStsModel()) return;
            const model = get('model-select').value || '';
            const provider = getStsProvider(model);
            const voiceSel = get('sts-voice');
            const speedWrap = get('sts-speed-wrap');
            const speedInput = get('sts-speed');
            const speedLabel = get('sts-speed-label');
            const rateWrap = get('sts-rate-wrap');
            const rateIn = get('sts-rate-in');
            const rateOut = get('sts-rate-out');
            const thinkingWrap = get('sts-thinking-wrap');
            const note = get('sts-note');
            const voiceWrap = get('sts-voice-wrap');
            const autoPlayWrap = get('sts-auto-play-wrap');
            const modeLabel = get('sts-mode-label');
            const transcription = isTranscriptionModel() || isGeminiLiveTranscribeModel();
            const langWrap = get('sts-lang-wrap');

            if (transcription) {
                if (modeLabel) modeLabel.textContent = 'Realtime Speech-to-Text';
                if (voiceWrap) voiceWrap.classList.add('hidden');
                if (autoPlayWrap) autoPlayWrap.classList.add('hidden');
                if (speedWrap) speedWrap.classList.add('hidden');
                if (rateWrap) rateWrap.classList.add('hidden');
                if (thinkingWrap) thinkingWrap.classList.add('hidden');
                if (langWrap) langWrap.classList.add('hidden');
                const transcribeWrap = get('sts-transcribe-wrap');
                const customVocabWrap = get('sts-custom-vocab-wrap');
                if (transcribeWrap) transcribeWrap.classList.toggle('hidden', !isGeminiLiveTranscribeModel());
                if (customVocabWrap) customVocabWrap.classList.toggle('hidden', !isGeminiLiveTranscribeModel());
                if (note) {
                    note.textContent = isGeminiLiveTranscribeModel()
                        ? 'リアルタイム低遅延文字起こし（16kHz PCM / 最大10分）'
                        : model === 'gpt-live-transcribe'
                            ? '低遅延ライブ文字起こし（24kHz PCM）'
                            : '高精度なコミット単位の文字起こし（24kHz PCM）';
                }
            } else if (provider === 'openai') {
                if (modeLabel) modeLabel.textContent = 'Speech-to-Speech Live';
                if (voiceWrap) voiceWrap.classList.remove('hidden');
                if (autoPlayWrap) autoPlayWrap.classList.remove('hidden');
                setSelectOptions(voiceSel, OPENAI_STS_VOICES, voiceSel.value || 'alloy');
                if (speedWrap) speedWrap.classList.remove('hidden');
                if (speedInput) {
                    speedInput.min = 0.25; speedInput.max = 1.5; speedInput.step = 0.05;
                    if (!speedInput.value) speedInput.value = 1;
                    if (Number(speedInput.value) < 0.25) speedInput.value = 0.25;
                    if (Number(speedInput.value) > 1.5) speedInput.value = 1.5;
                }
                if (rateWrap) rateWrap.classList.add('hidden');
                if (thinkingWrap) thinkingWrap.classList.add('hidden');
                if (langWrap) langWrap.classList.add('hidden');
                if (note) note.textContent = 'OpenAI Realtimeは24kHz PCM固定';
            } else if (provider === 'xai') {
                if (modeLabel) modeLabel.textContent = 'Speech-to-Speech Live';
                if (voiceWrap) voiceWrap.classList.remove('hidden');
                if (autoPlayWrap) autoPlayWrap.classList.remove('hidden');
                setSelectOptions(voiceSel, GROK_STS_VOICES, voiceSel.value || 'Ara');
                if (speedWrap) speedWrap.classList.add('hidden');
                if (rateWrap) rateWrap.classList.remove('hidden');
                if (thinkingWrap) thinkingWrap.classList.add('hidden');
                if (langWrap) langWrap.classList.add('hidden');
                setSelectOptions(rateIn, GROK_PCM_RATES, Number(rateIn.value || 24000));
                setSelectOptions(rateOut, GROK_PCM_RATES, Number(rateOut.value || 24000));
                if (note) note.textContent = 'xAIはPCMサンプルレート変更可';
            } else if (provider === 'gemini') {
                if (modeLabel) modeLabel.textContent = 'Speech-to-Speech Live';
                if (voiceWrap) voiceWrap.classList.remove('hidden');
                if (autoPlayWrap) autoPlayWrap.classList.remove('hidden');
                setSelectOptions(voiceSel, GEMINI_STS_VOICES, voiceSel.value || 'Kore');
                if (speedWrap) speedWrap.classList.add('hidden');
                if (rateWrap) rateWrap.classList.add('hidden');
                if (thinkingWrap) thinkingWrap.classList.remove('hidden');
                if (langWrap) langWrap.classList.add('hidden');
                if (note) note.textContent = 'Gemini Liveは音声速度変更非対応';
                if (model === 'gemini-3.5-live-translate-preview') {
                    if (modeLabel) modeLabel.textContent = 'Realtime Translation';
                    if (thinkingWrap) thinkingWrap.classList.add('hidden');
                    if (voiceWrap) voiceWrap.classList.add('hidden');
                    if (langWrap) langWrap.classList.remove('hidden');
                    if (note) note.textContent = '70以上の言語に対応するリアルタイム音声翻訳（Think非対応・音声選択不可）';
                }
            }
            if (speedWrap && speedLabel && speedInput && !speedWrap.classList.contains('hidden')) {
                speedLabel.textContent = `${Number(speedInput.value || 1).toFixed(2)}x`;
            }
        }
        function stsOpt(id) {
            const el = get(id);
            if (id === 'sts-auto-play' || id === 'sts-auto-restart') {
                return el ? !!el.checked : true;
            }
            return el ? !!el.checked : false;
        }
        function getStsSilenceMs() {
            const el = get('sts-silence-sec');
            let v = el ? parseFloat(el.value) : 1.5;
            if (isNaN(v) || v < 0.5) v = 0.5;
            if (v > 10) v = 10;
            return Math.round(v * 1000);
        }
        function getTtsProvider(model) {
            if (!model) return null;
            const m = model.toLowerCase();
            if (m.includes('google-tts')) return 'google';
            if (m.includes('gemini') && m.includes('tts')) return 'gemini';
            if (m.includes('grok-tts') || m.includes('xai-tts')) return 'xai';
            if (m.includes('tts')) return 'openai';
            return null;
        }
        function setSelectOptions(el, list, selected) {
            if (!el) return;
            el.innerHTML = '';
            list.forEach(v => {
                const opt = document.createElement('option');
                opt.value = v.value || v;
                opt.textContent = v.label || v;
                if ((v.value || v) === selected) opt.selected = true;
                el.appendChild(opt);
            });
        }
        function updateTtsUi() {
            const model = get('model-select').value || '';
            const provider = getTtsProvider(model);
            const wrap = get('audio-gen-options');
            if (!wrap) return;
            if (!provider) {
                wrap.classList.add('hidden');
                return;
            }
            wrap.classList.remove('hidden');
            const voiceSel = get('tts-voice');
            const voiceCustomWrap = get('tts-voice-custom-wrap');
            const voiceCustom = get('tts-voice-custom');
            const langWrap = get('tts-language-wrap');
            const langInput = get('tts-language');
            const speedWrap = get('tts-speed-wrap');
            const speedInput = get('tts-speed');
            const speedLabel = get('tts-speed-label');
            const speedNote = get('tts-speed-note');

            if (provider === 'openai') {
                setSelectOptions(voiceSel, OPENAI_TTS_VOICES, voiceSel.value || 'alloy');
                voiceCustomWrap.classList.add('hidden');
                langWrap.classList.add('hidden');
                if (speedInput) {
                    speedInput.min = 0.25; speedInput.max = 4; speedInput.step = 0.05;
                    if (!speedInput.value) speedInput.value = 1;
                    if (Number(speedInput.value) < 0.25) speedInput.value = 0.25;
                    if (Number(speedInput.value) > 4) speedInput.value = 4;
                    speedInput.disabled = false;
                }
                if (speedNote) speedNote.textContent = '';
            } else if (provider === 'gemini') {
                setSelectOptions(voiceSel, GEMINI_TTS_VOICES, voiceSel.value || 'Kore');
                voiceCustomWrap.classList.add('hidden');
                langWrap.classList.add('hidden');
                if (speedInput) { speedInput.disabled = true; }
                if (speedNote) speedNote.textContent = '(Gemini TTSは速度変更非対応)';
            } else if (provider === 'google') {
                setSelectOptions(voiceSel, [
                    { value: 'auto', label: 'Auto (Studio/Neural2)' },
                    { value: 'custom', label: 'Custom Voice Name' }
                ], voiceSel.value || 'auto');
                if (voiceSel.value === 'custom') {
                    voiceCustomWrap.classList.remove('hidden');
                } else {
                    voiceCustomWrap.classList.add('hidden');
                    if (voiceCustom) voiceCustom.value = '';
                }
                langWrap.classList.remove('hidden');
                if (langInput && !langInput.value) langInput.value = 'ja-JP';
                if (speedInput) {
                    speedInput.min = 0.25; speedInput.max = 2; speedInput.step = 0.05;
                    if (!speedInput.value) speedInput.value = 1;
                    if (Number(speedInput.value) < 0.25) speedInput.value = 0.25;
                    if (Number(speedInput.value) > 2) speedInput.value = 2;
                    speedInput.disabled = false;
                }
                if (speedNote) speedNote.textContent = '';
            } else if (provider === 'xai') {
                setSelectOptions(voiceSel, GROK_TTS_VOICES, voiceSel.value || 'Eve');
                voiceCustomWrap.classList.remove('hidden');
                langWrap.classList.remove('hidden');
                if (langInput && !langInput.value) langInput.value = 'ja';
                if (speedInput) {
                    speedInput.min = 0.7; speedInput.max = 1.5; speedInput.step = 0.05;
                    if (!speedInput.value) speedInput.value = 1;
                    if (Number(speedInput.value) < 0.7) speedInput.value = 0.7;
                    if (Number(speedInput.value) > 1.5) speedInput.value = 1.5;
                    speedInput.disabled = false;
                }
                if (speedNote) speedNote.textContent = 'xAI TTS supports speed 0.7–1.5 and speech tags';
            }
            if (speedInput && speedLabel) {
                speedLabel.textContent = `${Number(speedInput.value || 1).toFixed(2)}x`;
            }
        }
