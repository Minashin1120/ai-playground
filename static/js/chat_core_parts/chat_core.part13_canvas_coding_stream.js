        function previewCanvasCodeFromButton(btn) {
            if (!btn) return false;
            const payload = collectCanvasBlocksFromButton(btn);
            if (!payload || !payload.blocks || !payload.blocks.length) return false;
            const currentBlocks = Array.isArray(canvasPreviewState.blocks) ? canvasPreviewState.blocks : [];
            const matchedIndex = currentBlocks.findIndex((block) => block && block.key === payload.selectedKey);
            if (matchedIndex !== -1 && currentBlocks.length > 1) {
                return applyCanvasSelection(matchedIndex);
            }
            const selectedBlock = payload.blocks[payload.selectedIndex] || payload.blocks[0] || null;
            canvasPreviewState.blocks = payload.blocks;
            canvasPreviewState.rawText = selectedBlock && selectedBlock.code !== undefined && selectedBlock.code !== null ? String(selectedBlock.code) : '';
            canvasPreviewState.renderText = canvasPreviewState.rawText;
            canvasPreviewState.selectedIndex = Number.isInteger(payload.selectedIndex) ? payload.selectedIndex : 0;
            canvasPreviewState.selectedKey = payload.selectedKey || (selectedBlock && selectedBlock.key) || '';
            canvasPreviewState.selectionMode = 'manual';
            resetCanvasScrollState();
            canvasPreviewState.lastCanvasData = {
                renderText: canvasPreviewState.renderText,
                blocks: payload.blocks,
                primaryBlock: selectedBlock,
                primaryIndex: canvasPreviewState.selectedIndex,
                rawText: canvasPreviewState.rawText
            };
            canvasPreviewState.mobileView = 'preview';
            syncCanvasPanelViewUi('preview', { focus: false });
            refreshCanvasPreviewPanel();
            return true;
        }
        function buildCanvasPreviewDocument(block) {
            const code = String(block && block.code !== undefined && block.code !== null ? block.code : '');
            const lang = String(block && block.lang ? block.lang : '').trim().toLowerCase();
            const isHtml = isCanvasHtmlPreviewCandidate(lang, code);
            if (isHtml) {
                return sanitizeHtmlForPreview(code);
            }
            const title = lang ? `Canvas Preview: ${lang}` : 'Canvas Preview';
            const safeCode = escapeHtml(code || '');
            return `<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>${escapeHtml(title)}</title><style>
                :root { color-scheme: dark; }
                html, body { margin: 0; min-height: 100%; background: #0b1220; color: #e5e7eb; font-family: "Noto Sans JP", system-ui, -apple-system, "Segoe UI", sans-serif; }
                body { box-sizing: border-box; padding: 16px; }
                .frame {
                    background: linear-gradient(180deg, rgba(15, 23, 42, 0.92), rgba(2, 6, 23, 0.94));
                    border: 1px solid rgba(148, 163, 184, 0.18);
                    border-radius: 14px;
                    padding: 14px;
                    box-shadow: 0 20px 48px rgba(0, 0, 0, 0.34);
                }
                .label {
                    font-size: 11px;
                    text-transform: uppercase;
                    letter-spacing: 0.14em;
                    color: #67e8f9;
                    margin-bottom: 10px;
                }
                pre {
                    margin: 0;
                    white-space: pre-wrap;
                    word-break: break-word;
                    overflow-wrap: anywhere;
                    font-family: "JetBrains Mono", "Noto Sans Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
                    font-size: 13px;
                    line-height: 1.6;
                    color: #e2e8f0;
                }
                .muted { color: #94a3b8; }
            </style></head><body><div class="frame"><div class="label">${escapeHtml(title)}</div><pre>${safeCode || '<span class="muted">Canvasで表示中</span>'}</pre></div></body></html>`;
        }
        function syncCanvasModeUi(enabled = canvasModeEnabled, options = {}) {
            const persist = options.persist !== false;
            canvasModeEnabled = !!enabled;
            if (persist) {
                try {
                    localStorage.setItem(CANVAS_MODE_STORAGE_KEY, canvasModeEnabled ? 'true' : 'false');
                } catch (e) {}
            }
            const checkbox = get('enable-canvas-mode');
            if (checkbox && checkbox.checked !== canvasModeEnabled) checkbox.checked = canvasModeEnabled;
            if (!canvasModeEnabled) {
                hideCanvasPreviewPanel(options.animate !== false);
                if (!activeStreamingBubbleId && currentThreadId) {
                    try {
                        renderThreadTree({ silent: true, keepScroll: true });
                    } catch (e) {}
                }
                return;
            }
            showCanvasPreviewPanel();
            if (isCanvasMobileLayout()) {
                syncCanvasPanelViewUi('preview', { focus: false });
            }
            syncCanvasPanelViewUi(canvasPreviewState.mobileView || 'preview', { focus: false });
            if (options.skipReset) return;
            if (!activeStreamingBubbleId) {
                resetCanvasPreviewPanel();
                if (currentThreadId) {
                    try {
                        renderThreadTree({ silent: true, keepScroll: true });
                    } catch (e) {}
                }
            } else {
                refreshCanvasPreviewPanel();
            }
        }

        function normalizeMarkdownNewlines(text) {
            return String(text || '').replace(/\r\n/g, '\n').replace(/\r/g, '\n');
        }

        function stripExactFencedBlock(text, language, body) {
            let result = normalizeMarkdownNewlines(text);
            const bodyStr = normalizeMarkdownNewlines(body);
            if (!bodyStr && bodyStr !== '') return result;
            const langs = language ? [String(language), ''] : [''];
            for (const fenceChar of ['`', '~']) {
                for (let n = 3; n <= 10; n++) {
                    const fence = fenceChar.repeat(n);
                    for (const lang of langs) {
                        const open = `${fence}${lang}\n`;
                        const close = `\n${fence}`;
                        const candidate = open + bodyStr + close;
                        if (!result.includes(candidate)) continue;
                        result = result.split(candidate).join('');
                    }
                }
            }
            return result;
        }

        function stripVisiblePythonOutputBlock(text, output) {
            let result = normalizeMarkdownNewlines(text);
            const outStr = normalizeMarkdownNewlines(output == null ? '' : String(output));
            const prefixes = ['**Output:**\n', '**Output:** \n', '**Output:**'];
            for (const prefix of prefixes) {
                for (const fenceChar of ['`', '~']) {
                    for (let n = 3; n <= 10; n++) {
                        const fence = fenceChar.repeat(n);
                        const candidates = [
                            `${prefix}${fence}\n${outStr}\n${fence}`,
                            `${prefix}\n${fence}\n${outStr}\n${fence}`,
                            `\n${prefix}${fence}\n${outStr}\n${fence}`,
                            `\n${prefix}\n${fence}\n${outStr}\n${fence}`,
                        ];
                        candidates.forEach((candidate) => {
                            if (result.includes(candidate)) {
                                result = result.split(candidate).join('\n');
                            }
                        });
                    }
                }
            }
            return result;
        }

        function buildChatErrorBubbleHtml(errorText) {
            const msg = String(errorText == null ? '' : errorText).trim() || 'Unknown error';
            return `<div class="text-red-400 text-xs mt-2 border border-red-500 p-2 rounded chat-error-box" role="alert"><i class="fas fa-triangle-exclamation mr-1"></i>Error: ${escapeHtml(msg)}</div>`;
        }

        function buildChatErrorMarkdown(errorText, partialContent = '') {
            let err = String(errorText == null ? '' : errorText).trim() || 'Unknown error';
            // Keep the fence well-formed even if the message contains backticks.
            err = err.replace(/```/g, "'''");
            if (err.length > 50000) err = err.slice(0, 50000) + '…';
            const fence = '```chat_error\n' + err + '\n```';
            const partial = String(partialContent == null ? '' : partialContent).replace(/\s+$/, '');
            return partial ? (partial + '\n\n' + fence) : fence;
        }

        function extractPythonExecutionsFromContent(rawText) {
            const source = normalizeMarkdownNewlines(rawText);
            const executions = [];
            if (!source) return { text: '', executions };

            // ```pyexec ... ``` (fences of length >= 3)
            const pyexecRe = /(?:^|\n)(`{3,}|~{3,})pyexec[ \t]*\n([\s\S]*?)\n\1[ \t]*(?=\n|$)/g;
            let cleaned = source.replace(pyexecRe, (match, fence, body) => {
                const raw = String(body || '').trim();
                try {
                    const obj = JSON.parse(raw);
                    executions.push({
                        code: obj && obj.code != null ? String(obj.code) : '',
                        output: obj && obj.output != null ? String(obj.output) : ''
                    });
                } catch (e) {
                    executions.push({ code: raw, output: '' });
                }
                return '\n';
            });

            executions.forEach((ex) => {
                if (ex.code) {
                    cleaned = stripExactFencedBlock(cleaned, 'python', ex.code);
                    cleaned = stripExactFencedBlock(cleaned, 'py', ex.code);
                }
                cleaned = stripVisiblePythonOutputBlock(cleaned, ex.output);
            });

            cleaned = cleaned
                .replace(/[ \t]+\n/g, '\n')
                .replace(/\n{3,}/g, '\n\n')
                .replace(/^\n+/, '')
                .replace(/\n+$/, '');

            return { text: cleaned, executions };
        }

        function extractMcpExecutionNotesFromContent(rawText) {
            const source = normalizeMarkdownNewlines(rawText);
            const notes = [];
            if (!source) return { text: '', notes };

            const keptLines = [];
            source.split('\n').forEach((line) => {
                // MCP execution notices are emitted as one Markdown blockquote
                // line. Keep them out of the streamed prose so they can be
                // rendered together after the answer text.
                if (/^>\s*(?:🔧|🚫)\s*\*\*MCPツール実行(?:[:：]|は|（)/.test(line)) {
                    notes.push(line.trim());
                } else {
                    keptLines.push(line);
                }
            });

            const cleaned = keptLines.join('\n')
                .replace(/[ \t]+\n/g, '\n')
                .replace(/\n{3,}/g, '\n\n')
                .replace(/^\n+/, '')
                .replace(/\n+$/, '');
            return { text: cleaned, notes };
        }

        function appendMcpExecutionNotes(text, notes) {
            const body = String(text || '').trim();
            const items = Array.isArray(notes) ? notes.filter(Boolean) : [];
            if (!items.length) return body;
            return body ? `${body}\n\n${items.join('\n')}` : items.join('\n');
        }

        function buildPythonExecDetailBoxHtml(ex, index, total) {
            const codeRaw = ex && ex.code != null ? String(ex.code) : '';
            const outputRaw = ex && ex.output != null ? String(ex.output) : '';
            let codeHtml = '';
            try {
                if (window.hljs && typeof window.hljs.highlight === 'function') {
                    codeHtml = window.hljs.highlight(codeRaw, { language: 'python' }).value;
                } else {
                    codeHtml = escapeHtml(codeRaw);
                }
            } catch (e) {
                codeHtml = escapeHtml(codeRaw);
            }
            const outputHtml = escapeHtml(outputRaw);
            const encCode = encodeURIComponent(codeRaw).replace(/'/g, '%27');
            const encOut = encodeURIComponent(outputRaw).replace(/'/g, '%27');
            const codeKey = hashString(`pyexec-detail\n${codeRaw}\n${outputRaw}\n${index}`);
            const label = total > 1 ? `Python Execution ${index + 1}/${total}` : 'Python Execution';
            const downloadBtn = `<button class="download-btn" data-code="${encCode}" data-lang="python" title="コードをダウンロード" aria-label="コードをダウンロード"><i class="fas fa-download"></i></button>`;
            const codingBtn = `<button class="coding-target-btn" data-code="${encCode}" data-code-key="${codeKey}" data-coding-lang="python" aria-pressed="false" title="Coding Modeの編集対象に指定" aria-label="編集対象に指定"><i class="fas fa-quote-right"></i></button>`;
            return `<div class="code-wrapper python-box" data-collapsed="false" data-code-key="${codeKey}"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> ${escapeHtml(label)}</span><div class="code-actions">${codingBtn}${downloadBtn}<button class="copy-btn" data-copy="code" data-code="${encCode}" title="コードをコピー" aria-label="コードをコピー"><i class="fas fa-copy"></i></button><button class="copy-btn" data-copy="output" data-code="${encOut}" title="出力をコピー" aria-label="出力をコピー"><i class="fas fa-align-left"></i></button></div></div><div class="code-body"><div class="python-section"><div class="python-label">Code</div><pre><code class="hljs language-python python-code">${codeHtml}</code></pre></div><div class="python-section"><div class="python-label">Output</div><pre><code class="hljs language-plaintext python-output">${outputHtml}</code></pre></div></div></div>`;
        }

        function showPythonExecDetailModal(messageId = null) {
            if (location.pathname !== '/python-execution') {
                const state = { modal: 'python-execution' };
                if (messageId !== null) state.messageId = messageId;
                history.pushState(state, '', '/python-execution');
            }
            showModal('python-exec-modal');
        }

        function openPythonExecDetail(id) {
            const meta = messageMeta[id];
            const modal = get('python-exec-modal');
            const body = get('python-exec-modal-body');
            const title = get('python-exec-modal-title');
            if (!modal || !body) return;
            const executions = (meta && Array.isArray(meta.python_executions)) ? meta.python_executions : [];
            if (!executions.length) {
                showToast('Python実行結果がありません', 'info', false);
                return;
            }
            if (title) {
                const countLabel = executions.length > 1 ? `（${executions.length}件）` : '';
                title.textContent = `Python 実行結果${countLabel}`;
            }
            body.innerHTML = executions.map((ex, i) => buildPythonExecDetailBoxHtml(ex, i, executions.length)).join('');
            if (codingModeEnabled) {
                syncCodingTargetButtons(body);
                syncCodingModeUi(true, { persist: false });
            }
            showPythonExecDetailModal(id);
        }
        window.openPythonExecDetail = openPythonExecDetail;

        function closePythonExecDetail(skipHistory = false) {
            const modal = get('python-exec-modal');
            if (!modal) return;
            hideModal('python-exec-modal');
            if (!skipHistory && location.pathname === '/python-execution') {
                history.back();
            }
        }
        window.closePythonExecDetail = closePythonExecDetail;

        function buildAiMarkdownHtml(text) {
            const mcpExtract = extractMcpExecutionNotesFromContent(text);
            const displayText = appendMcpExecutionNotes(mcpExtract.text, mcpExtract.notes);
            const canvasData = canvasModeEnabled ? parseCanvasMarkdown(displayText) : { renderText: displayText, blocks: [], primaryBlock: null, rawText: displayText };
            if (canvasModeEnabled) {
                updateCanvasPreviewState(canvasData);
                refreshCanvasPreviewPanel();
            }
            const wrap = document.createElement('div');
            wrap.className = 'prose prose-invert text-sm break-words';
            wrap.innerHTML = sanitizeMarkdownHtml(canvasData.renderText);
            wrapRenderedSvgBoxes(wrap);
            if (!lowBandwidthMode) {
                if (maybeNeedsHighlight(canvasData.renderText, wrap)) ensureHighlightLoaded().catch(() => {});
                if (maybeNeedsMathJax(canvasData.renderText)) ensureMathJaxLoaded().catch(() => {});
            }
            return wrap.outerHTML;
        }
        function renderAiMarkdownInto(container, text, opts = {}) {
            if (!container) return;
            const mcpExtract = extractMcpExecutionNotesFromContent(text);
            const displayText = appendMcpExecutionNotes(mcpExtract.text, mcpExtract.notes);
            const canvasData = canvasModeEnabled ? parseCanvasMarkdown(displayText) : { renderText: displayText, blocks: [], primaryBlock: null, rawText: displayText };
            if (canvasModeEnabled) {
                updateCanvasPreviewState(canvasData);
                refreshCanvasPreviewPanel();
            }
            if (opts.incrementalMath) {
                const template = document.createElement('template');
                template.innerHTML = sanitizeMarkdownHtml(canvasData.renderText, { streamMathSegments: true });
                const preserved = new Map();
                container.querySelectorAll('.stream-math-segment[data-stream-math-key]').forEach((el) => {
                    const key = el.getAttribute('data-stream-math-key');
                    if (key) preserved.set(key, el);
                });
                const newMathSegments = [];
                template.content.querySelectorAll('.stream-math-segment[data-stream-math-key]').forEach((fresh) => {
                    const old = preserved.get(fresh.getAttribute('data-stream-math-key'));
                    if (old) fresh.replaceWith(old);
                    else newMathSegments.push(fresh);
                });
                container.replaceChildren(template.content);
                wrapRenderedSvgBoxes(container);
                queueHighlight(container, canvasData.renderText);
                queueIncrementalMathTypeset(newMathSegments);
                return;
            }
            container.innerHTML = sanitizeMarkdownHtml(canvasData.renderText);
            wrapRenderedSvgBoxes(container);
            queueMessageDecorations(container, canvasData.renderText);
        }
        function wrapRenderedSvgBoxes(root) {
            if (!root || typeof root.querySelectorAll !== 'function') return;
            root.querySelectorAll('svg').forEach((svg) => {
                if (!svg || !svg.parentNode) return;
                if (svg.closest('.svg-render-box')) return;
                if (svg.closest('pre, code, .code-wrapper, .thought-container')) return;
                const frame = document.createElement('span');
                frame.className = 'svg-render-box';
                svg.parentNode.insertBefore(frame, svg);
                frame.appendChild(svg);
            });
        }
        function renderMessage(id, role, text, imgUrl, thoughtData, modelName, versionInfo = null, animate = true, quoteText = null, tokenCount = null, tokenIn = null, tokenOut = null, isEncrypted = null, tokensContent = null, tokensThought = null, target = null, doScroll = true, parentId = null, gemName = null) {
            const isUser = role === 'user';
            const bg = isUser ? 'bg-blue-600' : 'bg-gray-700';
            const align = isUser ? 'justify-end' : 'justify-start';
            messageStore[id] = text;
            const pythonExtract = (!isUser && text) ? extractPythonExecutionsFromContent(text) : { text: text || '', executions: [] };
            const displayText = isUser ? text : pythonExtract.text;
            let totalTokens = tokenCount;
            if (totalTokens === null || totalTokens === undefined) {
                const inVal = (tokenIn !== null && tokenIn !== undefined) ? Number(tokenIn) : 0;
                const outVal = (tokenOut !== null && tokenOut !== undefined) ? Number(tokenOut) : 0;
                if ((tokenIn !== null && tokenIn !== undefined) || (tokenOut !== null && tokenOut !== undefined)) {
                    totalTokens = inVal + outVal;
                }
            }
            messageMeta[id] = {
                tokens_in: tokenIn,
                tokens_out: tokenOut,
                tokens_total: totalTokens,
                tokens_content: tokensContent,
                tokens_thought: tokensThought,
                is_encrypted: isEncrypted,
                role: role,
                model: modelName,
                parent_id: parentId,
                quote_text: quoteText,
                image_url: imgUrl,
                gem_name: gemName,
                python_executions: isUser ? [] : (pythonExtract.executions || [])
            };

            let qh = '';
            if (quoteText) {
                qh = `<div class="mb-2 p-2 bg-black/20 rounded border-l-4 border-blue-400 text-xs text-gray-300 italic truncate max-w-full"><i class="fas fa-quote-left mr-1 opacity-50"></i>${escapeHtml(quoteText)}</div>`;
            }

            let th = '';
            if (thoughtData && !isUser) {
                let tx = "";
                try { const o = JSON.parse(thoughtData); tx = o.text || ""; } catch(e) { tx = thoughtData; }
                if (tx) th = `<div class="thought-container"><div class="thought-header" onclick="toggleThinking(this)"><i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content collapsed">${escapeHtml(tx)}</div></div>`;
            }

            let at = '';
            if(imgUrl) {
                try {
                    const imgs = JSON.parse(imgUrl);
                    if(imgs.length) {
                        const items = [];
                        imgs.forEach(u => {
                            let path = u;
                            let source = 'unknown';
                            if (path && typeof path === 'object') {
                                source = normalizeAttachmentSource(path.source);
                                path = path.filepath || path.path || path.url || path.file || '';
                            }
                            path = normalizeAttachmentPath(path) || path;
                            if (!path) return;
                            setAttachmentSourceForPath(path, source);
                            const displayPath = path.replace(/^\d+\//, '');
                            const url = buildFileUrl(displayPath);
                            const previewUrl = buildAttachmentPreviewUrl(displayPath);
                            const fn = path.split('/').pop();
                            const ext = fn.split('.').pop().toLowerCase();
                            if(['jpg','jpeg','png','webp','gif'].includes(ext)) {
                                items.push(`<img src="${previewUrl}" data-viewer-src="${url}" data-viewer-filename="${escapeHtml(fn)}" class="chat-image" loading="lazy" onclick="openImageViewer('${url}')" title="${fn}">`);
                            } else {
                                items.push(`<div class="file-thumb bg-gray-800 border border-gray-600 rounded flex flex-col items-center justify-center cursor-pointer hover:bg-gray-700" onclick="window.open('${url}')" title="${fn}"><i class="fas fa-file text-2xl text-gray-400 mb-1"></i><span class="text-[9px] truncate w-20 text-center">${fn}</span></div>`);
                            }
                        });

                        if (items.length > 0) {
                            let gridClass = 'grid-multi';
                            if (items.length === 1) gridClass = 'grid-1';
                            else if (items.length === 2) gridClass = 'grid-2';
                            else if (items.length === 3) gridClass = 'grid-3';
                            else if (items.length === 4) gridClass = 'grid-4';

                            at = `<div class="image-grid ${gridClass}">${items.join('')}</div>`;
                        }
                    }
                } catch(e){}
            }

            const regenBtn = !isUser ? `<button class="ctrl-btn" onclick="regenerateMessage('${id}')"><i class="fas fa-rotate-right"></i></button>` : '';
            const ctrl = `<div class="msg-controls absolute -top-5 right-0 hidden group-hover:flex gap-1 z-10"><button class="ctrl-btn" onclick="window.copyMessage('${id}', this)"><i class="fas fa-copy"></i></button>${isUser?`<button class="ctrl-btn edit-btn" data-id="${id}"><i class="fas fa-pen"></i></button>`:''}${regenBtn}<button class="ctrl-btn" onclick="deleteMessage('${id}')"><i class="fas fa-trash"></i></button></div>`;
            const footerParts = [];
            if (!isUser && modelName) footerParts.push(escapeHtml(modelName));
            if (gemName) {
                if (isUser) {
                    footerParts.push(`<span class="text-purple-300/90"><i class="fas fa-gem mr-0.5"></i>${escapeHtml(gemName)}</span>`);
                } else {
                    footerParts.push(`<span class="text-purple-300/90"><i class="fas fa-gem mr-0.5"></i>${escapeHtml(gemName)}</span>`);
                }
            }
            const tokenParts = [];
            if (tokenIn !== null && tokenIn !== undefined) tokenParts.push(`In ${tokenIn}`);
            if (tokenOut !== null && tokenOut !== undefined) {
                let outLabel = `Out ${tokenOut}`;
                if (tokensThought !== null && tokensThought !== undefined && Number(tokensThought) > 0) {
                    outLabel += ` (Thought ${tokensThought})`;
                }
                tokenParts.push(outLabel);
            }
            if (tokenParts.length || (tokenCount !== null && tokenCount !== undefined)) {
                const tLabel = tokenParts.length ? tokenParts.join(' / ') : `${tokenCount} tokens`;
                footerParts.push(`<button class="underline decoration-dotted hover:text-white token-detail-btn" onclick="openTokenDetail('${id}')">${tLabel}</button>`);
            }
            if (isEncrypted !== null && isEncrypted !== undefined) {
                const lockIcon = isEncrypted ? 'fa-lock' : 'fa-lock-open';
                const lockTitle = isAdminUser
                    ? (isEncrypted ? '暗号化状態（タップで復号化）' : '平文状態（タップで再暗号化）')
                    : (isEncrypted ? 'Encrypted' : 'Plain');
                const lockColor = isAdminUser
                    ? (isEncrypted ? 'text-amber-300/90 hover:text-amber-200' : 'text-cyan-300/90 hover:text-cyan-200')
                    : 'text-slate-300/80 hover:text-white';
                footerParts.push(`<button class="${lockColor}" title="${lockTitle}" onclick="openEncryptionSettings('${id}')"><i class="fas ${lockIcon}"></i></button>`);
            }
            if (!isUser && pythonExtract.executions && pythonExtract.executions.length) {
                const pyCount = pythonExtract.executions.length;
                const pyLabel = pyCount > 1 ? `Python ×${pyCount}` : 'Python';
                footerParts.push(`<button type="button" class="python-exec-btn" onclick="openPythonExecDetail('${id}')" title="Python実行結果を表示" aria-label="Python実行結果を表示"><i class="fas fa-terminal"></i><span>${pyLabel}</span></button>`);
            }
            const mHtml = footerParts.length ? `<div class="text-[10px] text-slate-300/90 mt-2 text-right font-mono message-footer-meta">${footerParts.join(' • ')}</div>` : '';

            let contentHtml;
            if (isUser) {
                // User message: RAW TEXT DISPLAY (Preserve whitespace, no markdown)
                contentHtml = `<div class="content-area whitespace-pre-wrap font-sans text-sm break-words">${escapeHtml(text||'')}</div>`;
            } else {
                // AI message: Markdown rendered with tool notices grouped after the prose.
                contentHtml = buildAiMarkdownHtml(displayText);
                // Ensure content-area class is present if not already in buildAiMarkdownHtml
                if (!contentHtml.includes('content-area')) {
                    contentHtml = contentHtml.replace('prose ', 'content-area prose ');
                }
            }

            // Version Switcher UI
            let versionSwitcher = '';
            if (versionInfo) {
                const prevId = versionInfo.siblings[versionInfo.current - 2];
                const nextId = versionInfo.siblings[versionInfo.current];
                versionSwitcher = `
                    <div class="flex items-center gap-2 text-[10px] text-gray-400 mt-1 select-none">
                        <button class="hover:text-white disabled:opacity-30" onclick="switchVersion(${prevId})" ${!prevId ? 'disabled' : ''}><i class="fas fa-chevron-left"></i></button>
                        <span>${versionInfo.current} / ${versionInfo.total}</span>
                        <button class="hover:text-white disabled:opacity-30" onclick="switchVersion(${nextId})" ${!nextId ? 'disabled' : ''}><i class="fas fa-chevron-right"></i></button>
                    </div>
                `;
            }

            const fadeClass = animate ? 'fade-in' : '';
            const msgEl = document.createElement('div');
            msgEl.className = `flex ${align} mb-4 ${fadeClass} relative message-group group`;
            msgEl.id = `msg-${id}`;
            msgEl.innerHTML = `<div class="message-bubble ${bg} text-white p-4 rounded-2xl shadow-md relative">${ctrl}${qh}${th}${contentHtml}${at}${versionSwitcher}${mHtml}</div>`;

            const container = target || get('chat-container');
            if (container) {
                container.appendChild(msgEl);
                if (doScroll) scrollToBottom();
                if (!isUser) {
                    queueMessageDecorations(msgEl, displayText);
                    syncCodingTargetButtons(msgEl);
                    syncCodingModeUi(codingModeEnabled, { persist: false });
                }
            }
            return msgEl;
        }

        function showTokenDetailModal(messageId = null) {
            if (location.pathname !== '/token-details') {
                const state = { modal: 'token-details' };
                if (messageId !== null) state.messageId = messageId;
                history.pushState(state, '', '/token-details');
            }
            showModal('token-detail-modal');
        }

        function openTokenDetail(id) {
            const meta = messageMeta[id];
            if (!meta) return;
            const modal = get('token-detail-modal');
            if (!modal) return;
            const total = meta.tokens_total !== null && meta.tokens_total !== undefined ? meta.tokens_total : '-';
            const tIn = meta.tokens_in !== null && meta.tokens_in !== undefined ? meta.tokens_in : '-';
            const tOut = meta.tokens_out !== null && meta.tokens_out !== undefined ? meta.tokens_out : '-';
            const tContent = meta.tokens_content !== null && meta.tokens_content !== undefined ? meta.tokens_content : '-';
            const tThought = meta.tokens_thought !== null && meta.tokens_thought !== undefined ? meta.tokens_thought : '-';
            const enc = meta.is_encrypted === null || meta.is_encrypted === undefined ? '-' : (meta.is_encrypted ? 'Encrypted' : 'Plain');
            get('token-detail-total').innerText = total;
            get('token-detail-in').innerText = tIn;
            get('token-detail-out').innerText = tOut;
            get('token-detail-content').innerText = tContent;
            get('token-detail-thought').innerText = tThought;
            get('token-detail-encrypted').innerText = enc;
            const title = meta.model ? `${meta.model} (${meta.role})` : `${meta.role}`;
            get('token-detail-title').innerText = title;
            showTokenDetailModal(id);
        }

        function closeTokenDetail(skipHistory = false) {
            const modal = get('token-detail-modal');
            if (!modal) return;
            hideModal('token-detail-modal');
            if (!skipHistory && location.pathname === '/token-details') {
                history.back();
            }
        }

        function openEncryptionSettings(id) {
            const meta = messageMeta[id];
            if (!meta) return;
            openEncryptionModal(meta.is_encrypted);
        }

        function openEncryptionModal(isEncrypted) {
            const modal = get('encryption-status-modal');
            if (!modal) return;
            const title = get('encryption-status-title');
            const body = get('encryption-status-body');
            const adminBox = get('encryption-status-admin-actions');
            const adminBtn = get('encryption-status-admin-toggle');
            const enc = !!isEncrypted;
            if (enc) {
                if (title) title.innerText = '暗号化されています';
                if (body) {
                    body.innerText = isAdminUser
                        ? 'このメッセージはE2EEで暗号化されています。管理者は下のボタンでこのチャット全体を復号化できます。'
                        : 'このメッセージはE2EEで暗号化されています。';
                }
            } else {
                if (title) title.innerText = '暗号化されていません';
                if (body) {
                    body.innerText = isAdminUser
                        ? 'このメッセージは暗号化されていません。管理者は下のボタンでこのチャット全体を再暗号化できます。'
                        : 'このメッセージは暗号化されていません。';
                }
            }
            if (adminBox && adminBtn) {
                const canToggle = !!(isAdminUser && currentThreadId);
                if (canToggle) {
                    adminBox.classList.remove('hidden');
                    // Encrypted message → offer decrypt (enable=false). Plain → re-encrypt (enable=true).
                    adminBtn.dataset.enable = enc ? '0' : '1';
                    adminBtn.disabled = false;
                    adminBtn.textContent = enc ? 'このチャットを復号化' : 'このチャットを再暗号化';
                    adminBtn.className = enc
                        ? 'w-full px-3 py-2 text-xs font-bold rounded text-white bg-amber-600 hover:bg-amber-500 btn-hover'
                        : 'w-full px-3 py-2 text-xs font-bold rounded text-white bg-cyan-700 hover:bg-cyan-600 btn-hover';
                } else {
                    adminBox.classList.add('hidden');
                }
            }
            showEncryptionStatusModal();
        }

        function showEncryptionStatusModal() {
            if (location.pathname !== '/encryption-status') {
                history.pushState({ modal: 'encryption-status' }, '', '/encryption-status');
            }
            showModal('encryption-status-modal');
        }

        async function toggleThreadEncryptionFromModal() {
            const adminBtn = get('encryption-status-admin-toggle');
            if (!adminBtn || !isAdminUser || !currentThreadId) return;
            if (adminBtn.disabled) return;
            const enable = adminBtn.getAttribute('data-enable') === '1';
            const action = enable ? '再暗号化' : '復号化';
            if (!confirm(`このチャットを${action}しますか？`)) return;
            adminBtn.disabled = true;
            const original = adminBtn.textContent;
            adminBtn.textContent = '処理中...';
            try {
                if (typeof window.__setAdminThreadEncryption !== 'function') {
                    showToast('暗号化操作を利用できません', 'error', true);
                    return;
                }
                const ok = await window.__setAdminThreadEncryption(currentThreadId, enable, {
                    confirmPrompt: false,
                    reloadCurrent: true
                });
                if (ok) closeEncryptionModal();
            } finally {
                adminBtn.disabled = false;
                adminBtn.textContent = original;
            }
        }

        function closeEncryptionModal(skipHistory = false) {
            hideModal('encryption-status-modal');
            if (!skipHistory && location.pathname === '/encryption-status') {
                history.back();
            }
        }

        function goToEncryptionSettings() {
            hideModal('encryption-status-modal');
            if (location.pathname === '/encryption-status') {
                history.replaceState({ modal: 'settings', from: '/encryption-status' }, '', '/settings');
            }
            if (typeof openSettingsModal === 'function') {
                openSettingsModal();
                switchTab('security');
                setTimeout(() => {
                    const card = (isAdminUser && get('admin-enc-card')) || get('e2ee-card');
                    if (card) card.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }, 150);
            }
        }
        function openTemporaryChatSettings() {
            if (typeof openSettingsModal !== 'function') return;
            openSettingsModal();
            switchTab('general');
            setTimeout(() => {
                const card = get('temp-chat-settings-card');
                if (!card) return;
                card.scrollIntoView({ behavior: 'smooth', block: 'center' });
                card.classList.add('ring-1', 'ring-amber-400/70');
                setTimeout(() => card.classList.remove('ring-1', 'ring-amber-400/70'), 1400);
            }, 150);
        }

        const isGeminiLocalPythonMode = (modelId, hasAudio, hasVideo, pyEnabled) => {
            const m = (modelId || '').toLowerCase();
            if (!m.includes('gemini')) return false;
            if (m.includes('image') || m.includes('nano') || m.includes('tts') || m.includes('native-audio')) return false;
            return !!pyEnabled && (hasAudio || hasVideo);
        };
        const confirmGeminiLocalPythonSwitch = async () => {
            if (!isGeminiLocalPyDialogEnabled()) return true;
            const modal = get('gemini-local-python-modal');
            if (!modal) return true;
            const remember = get('gemini-local-python-dont-show');
            const okBtn = get('gemini-local-python-continue');
            const cancelBtn = get('gemini-local-python-cancel');
            const closeBtn = get('gemini-local-python-close');
            if (remember) remember.checked = false;
            showModal('gemini-local-python-modal');
            return await new Promise((resolve) => {
                let done = false;
                function cleanup() {
                    if (okBtn) okBtn.removeEventListener('click', onOk);
                    if (cancelBtn) cancelBtn.removeEventListener('click', onCancel);
                    if (closeBtn) closeBtn.removeEventListener('click', onCancel);
                    modal.removeEventListener('click', onOverlay, true);
                }
                function finalize(proceed) {
                    if (done) return;
                    done = true;
                    const skip = remember && remember.checked;
                    if (skip) {
                        setGeminiLocalPyDialogEnabled(false);
                        syncGeminiLocalPyDialogSetting();
                    }
                    cleanup();
                    hideModal('gemini-local-python-modal');
                    resolve(proceed);
                }
                function onOk() { finalize(true); }
                function onCancel() { finalize(false); }
                function onOverlay(e) {
                    if (e.target === modal) {
                        e.preventDefault();
                        e.stopImmediatePropagation();
                        onCancel();
                    }
                }
                if (okBtn) okBtn.addEventListener('click', onOk);
                if (cancelBtn) cancelBtn.addEventListener('click', onCancel);
                if (closeBtn) closeBtn.addEventListener('click', onCancel);
                modal.addEventListener('click', onOverlay, true);
            });
        };

        function renderPendingMessage(target = null, animate = true, doScroll = true, pendingId = null, modelId = null) {
            const fadeClass = animate ? 'fade-in' : '';
            const idAttr = pendingId ? ` id="${pendingId}"` : '';
            const skeletonHtml = buildPendingSkeletonHtml(modelId, '回答を生成中...');
            const html = `<div class="flex justify-start mb-4 ${fadeClass}"><div${idAttr} class="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-md relative">${skeletonHtml}</div></div>`;
            const container = target || get('chat-container');
            if (!container) return;
            if (typeof container.insertAdjacentHTML === 'function') {
                container.insertAdjacentHTML('beforeend', html);
            } else {
                // renderThreadTree builds into a DocumentFragment, which has no
                // insertAdjacentHTML. Build the node and append it so a pending
                // stream bubble renders on reload during generation.
                const wrap = document.createElement('div');
                wrap.innerHTML = html;
                const node = wrap.firstElementChild;
                if (node) container.appendChild(node);
            }
            if (doScroll) scrollToBottom();
        }

        function beginPendingToStreamTransition(bubbleEl) {
            if (!bubbleEl) return;
            if (bubbleEl.getAttribute('data-stream-transition') === '1') return;
            const contentArea = bubbleEl.querySelector('.content-area');
            if (contentArea) {
                contentArea.classList.remove('pending-shimmer', 'skeleton-pending');
                contentArea.removeAttribute('data-skeleton-kind');
            }
            bubbleEl.setAttribute('data-stream-transition', '1');
            bubbleEl.classList.remove('ai-pending-bubble');
            bubbleEl.classList.add('ai-stream-transition');
            if (contentArea) {
                contentArea.classList.add('ai-stream-content-transition');
                setTimeout(() => {
                    if (contentArea) contentArea.classList.remove('ai-stream-content-transition');
                }, 300);
            }
            setTimeout(() => {
                if (bubbleEl) bubbleEl.classList.remove('ai-stream-transition');
            }, 320);
        }

        function normalizeJobIdForUi(jobId) {
            if (jobId === null || jobId === undefined || jobId === '') return null;
            return String(jobId);
        }

        function getActiveStreamingBubbleElement() {
            if (!activeStreamingBubbleId) return null;
            return get(activeStreamingBubbleId);
        }

        function captureStoppedPartialBubbleSnapshot(bubbleEl) {
            if (!bubbleEl) return null;
            const hasRenderedContent = Array.from(bubbleEl.querySelectorAll('.prose')).some((el) => String(el.textContent || '').trim());
            const hasPythonBox = !!bubbleEl.querySelector('.python-box');
            const hasThoughtContent = Array.from(bubbleEl.querySelectorAll('.thought-content')).some((el) => {
                const txt = String(el.textContent || '').trim();
                return !!txt && el.getAttribute('data-placeholder') !== '1';
            });
            if (!hasRenderedContent && !hasPythonBox && !hasThoughtContent) return null;
            const wrapper = bubbleEl.parentElement;
            if (!wrapper) return null;
            const clone = wrapper.cloneNode(true);
            clone.setAttribute('data-local-stopped-partial', '1');
            clone.classList.remove('fade-in');
            const cloneBubble = clone.querySelector('.message-bubble');
            if (cloneBubble) {
                cloneBubble.classList.remove('ai-pending-bubble', 'ai-stream-transition');
                cloneBubble.removeAttribute('data-stream-transition');
                cloneBubble.removeAttribute('id');
                if (!clone.querySelector('[data-stopped-partial-note="1"]')) {
                    const note = document.createElement('div');
                    note.setAttribute('data-stopped-partial-note', '1');
                    note.className = 'text-[10px] text-amber-200/90 mt-2 text-right';
                    note.textContent = '停止済み（途中まで）';
                    cloneBubble.appendChild(note);
                }
            }
            return {
                html: clone.outerHTML,
                threadId: (currentThreadId !== null && currentThreadId !== undefined && currentThreadId !== '')
                    ? String(currentThreadId)
                    : null
            };
        }

        function appendStoppedPartialBubbleSnapshot(snapshot, expectedThreadId = null) {
            if (!snapshot || !snapshot.html) return false;
            const current = (currentThreadId !== null && currentThreadId !== undefined && currentThreadId !== '')
                ? String(currentThreadId)
                : null;
            const expected = (expectedThreadId !== null && expectedThreadId !== undefined && expectedThreadId !== '')
                ? String(expectedThreadId)
                : (snapshot.threadId ? String(snapshot.threadId) : null);
            if (expected && current && expected !== current) return false;
            const container = get('chat-container');
            if (!container) return false;
            container.querySelectorAll('[data-local-stopped-partial="1"]').forEach((el) => el.remove());
            container.insertAdjacentHTML('beforeend', snapshot.html);
            scrollToBottom();
            return true;
        }

        function suppressPendingJob(jobId) {
            const id = normalizeJobIdForUi(jobId);
            if (!id) return;
            suppressedPendingJobIds.add(id);
        }

        function isPendingJobSuppressed(jobId) {
            const id = normalizeJobIdForUi(jobId);
            return !!(id && suppressedPendingJobIds.has(id));
        }

        function isManualStopAbortForThread(startedThreadId = null) {
            if (!manualStopContext) return false;
            const stopThreadId = manualStopContext.threadId ? String(manualStopContext.threadId) : null;
            const started = (startedThreadId !== null && startedThreadId !== undefined && startedThreadId !== '')
                ? String(startedThreadId)
                : null;
            const current = (currentThreadId !== null && currentThreadId !== undefined && currentThreadId !== '')
                ? String(currentThreadId)
                : null;
            if (stopThreadId && started && stopThreadId !== started) return false;
            if (stopThreadId && current && stopThreadId !== current) return false;
            return true;
        }

        async function syncThreadAfterAbortedStream(startedThreadId = null, opts = {}) {
            const retries = Math.max(0, Number(opts.retries ?? 1) || 0);
            const retryDelayMs = Math.max(0, Number(opts.retryDelayMs ?? 180) || 0);
            const notifyOnFailure = !!opts.notifyOnFailure;
            const started = (startedThreadId !== null && startedThreadId !== undefined && startedThreadId !== '')
                ? String(startedThreadId)
                : null;
            const current = (currentThreadId !== null && currentThreadId !== undefined && currentThreadId !== '')
                ? String(currentThreadId)
                : null;
            if (!current) return false;
            if (started && current !== started) return false;
            for (let attempt = 0; attempt <= retries; attempt++) {
                try {
                    if ((currentThreadId !== null && currentThreadId !== undefined && currentThreadId !== '') && String(currentThreadId) !== current) {
                        return false;
                    }
                    await loadMessages(current, { preserveDraft: true, silent: true });
                    return true;
                } catch (e) {
                    if (attempt < retries && retryDelayMs > 0) {
                        await new Promise((resolve) => setTimeout(resolve, retryDelayMs));
                    }
                }
            }
            if (notifyOnFailure) {
                const nowThread = (currentThreadId !== null && currentThreadId !== undefined && currentThreadId !== '')
                    ? String(currentThreadId)
                    : null;
                if (nowThread === current) {
                    showToast("停止後の履歴同期に失敗しました。画面を再読み込みすると確実です。", "warning", true);
                }
            }
            return false;
        }

        function vibrateHelper(pattern) {
            try {
                if (typeof navigator !== 'undefined' && navigator.vibrate) {
                    navigator.vibrate(pattern);
                }
            } catch (e) {
                console.warn('Vibration failed:', e);
            }
        }

        // === Slash command palette helpers ===
        function visibleSlashCommands(filter = '') {
            const normalized = String(filter || '').toLowerCase();
            return SLASH_COMMANDS.filter((command) => {
                if (command.kind === 'minimal' && !minimalPromptMode) return false;
                return command.label.toLowerCase().includes(normalized)
                    || command.description.toLowerCase().includes(normalized);
            });
        }

        function slashCommandSuggestionFilter(token, value) {
            if (String(token || '').toLowerCase() !== 'thinking') return token;
            const trimmed = String(value || '').trimStart();
            const match = trimmed.match(/^\/thinking(\s+.*)$/i);
            return match ? `thinking${match[1]}`.toLowerCase() : token;
        }

        function parseSlashToggleArgument(argument) {
            const value = String(argument || '').trim().toLowerCase();
            if (!value || value === 'toggle' || value === '切替' || value === '切り替え') return null;
            if (['on', 'true', '1', 'オン', '有効'].includes(value)) return true;
            if (['off', 'false', '0', 'オフ', '無効'].includes(value)) return false;
            return undefined;
        }

        function executeMinimalSlashCommand(commandId, argument = '') {
            const command = MINIMAL_SLASH_COMMANDS.find((item) => item.id === commandId);
            if (!command || !minimalPromptMode) return false;
            if (command.action === 'options') {
                openMinimalOptions();
                return true;
            }
            const item = MINIMAL_POPUP_ITEMS.find((candidate) => candidate.key === command.itemKey);
            if (!item || !minimalOptionVisible(item)) {
                showToast(`/${commandId} は現在のモデルでは利用できません`, 'warning');
                return true;
            }
            if (minimalOptionDisabled(item) && item.special !== 'thinking') {
                showToast(`/${commandId} は現在変更できません`, 'warning');
                return true;
            }

            const rawArgument = String(command.presetArgument || argument || '').trim();
            if (item.selectId) {
                if (!rawArgument) {
                    showToast(`使い方: ${command.label} ${command.id === 'effort' ? 'none / low / medium / high / xhigh / max' : 'default / none'}`, 'info');
                    return false;
                }
                const select = get(item.selectId);
                const normalized = rawArgument.toLowerCase();
                const option = select ? Array.from(select.options).find((candidate) =>
                    candidate.value.toLowerCase() === normalized || candidate.textContent.trim().toLowerCase() === normalized
                ) : null;
                if (!select || !option) {
                    showToast(`${command.label}: 指定値「${rawArgument}」は利用できません`, 'warning');
                    return false;
                }
                select.value = option.value;
                select.dispatchEvent(new Event('change', { bubbles: true }));
                refreshMinimalOptionItems();
                showToast(`${item.label}: ${option.textContent.trim()}`, 'success');
                return true;
            }

            if (item.special === 'thinking' && rawArgument) {
                const normalized = rawArgument.toLowerCase();
                const levelAliases = { min: 'minimal', minimal: 'minimal', low: 'low', mid: 'medium', medium: 'medium', high: 'high' };
                const desired = parseSlashToggleArgument(rawArgument);
                const checkbox = get(item.checkboxId);
                if (Object.prototype.hasOwnProperty.call(levelAliases, normalized)) {
                    if (checkbox && !checkbox.checked && !checkbox.disabled) {
                        checkbox.checked = true;
                        checkbox.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                    const level = get('thinking-level');
                    if (level) {
                        level.value = levelAliases[normalized];
                        level.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                    refreshMinimalOptionItems();
                    showToast(`Thinking: ${normalized}`, 'success');
                    return true;
                }
                if (desired === undefined) {
                    showToast('使い方: /thinking on / off / min / low / mid / high', 'info');
                    return false;
                }
            }

            if (item.checkboxId && rawArgument) {
                const desired = parseSlashToggleArgument(rawArgument);
                if (desired === undefined) {
                    showToast(`使い方: ${command.label} on / off`, 'info');
                    return false;
                }
                const checkbox = get(item.checkboxId);
                if (desired !== null && checkbox && checkbox.checked === desired) {
                    showToast(`${item.label}: ${desired ? 'ON' : 'OFF'}`, 'info');
                    return true;
                }
            }

            handleMinimalOptionClick(item);
            return true;
        }

        // Extract the leading slash command token (Latin word chars) from the
        // input, so text typed right after the command name without a space
        // (e.g. "/settingsデフォルトモデルを...") does not prevent the palette
        // from recognizing the command. Returns null when the input does not
        // begin with '/'.
        function extractSlashCommandToken(val) {
            const trimmed = String(val || '').trimStart();
            if (!trimmed.startsWith('/')) return null;
            const token = trimmed.substring(1).split(/\s+/)[0] || '';
            const m = token.match(/^[a-z][\w-]*/i);
            return m ? m[0] : token;
        }

        function hideSlashCommandSuggestions() {
            const box = get('slash-command-suggestions');
            if (box) box.classList.add('hidden');
            slashSuggestionsVisible = false;
            slashSelectedIndex = 0;
        }

        function showPendingSlashCommandIndicator(cmdId) {
            const indicator = get('slash-command-indicator');
            const nameEl = get('slash-command-name');
            if (!indicator || !nameEl) return;

            const cmd = SLASH_COMMANDS.find(c => c.id === cmdId);
            nameEl.textContent = cmd ? cmd.label : `/${cmdId}`;

            indicator.classList.remove('hidden');
            indicator.classList.add('flex');

            // Give contextual guidance via placeholder
            const input = get('prompt-input');
            if (input) {
                if (cmd) {
                    input.dataset.originalPlaceholder = input.placeholder;
                    input.placeholder = cmd.argumentHint || '設定変更の指示を入力（例: デフォルトモデルをgemini-2.5-flashに変更）...';
                }
            }
        }

        function hidePendingSlashCommandIndicator() {
            const indicator = get('slash-command-indicator');
            if (indicator) {
                indicator.classList.remove('flex');
                indicator.classList.add('hidden');
            }

            // Restore placeholder
            const input = get('prompt-input');
            if (input && input.dataset.originalPlaceholder) {
                input.placeholder = input.dataset.originalPlaceholder;
                delete input.dataset.originalPlaceholder;
            }

            const wasSettingsCommand = pendingSlashCommand === 'settings';
            pendingSlashCommand = null;
            if (wasSettingsCommand) clearAiSettingsConversation();
        }

        function showSlashCommandSuggestions(filter = '') {
            const box = get('slash-command-suggestions');
            const listEl = get('slash-command-list');
            const inputRow = get('input-row');
            if (!box || !listEl || !inputRow) return;

            const filtered = visibleSlashCommands(filter);

            if (filtered.length === 0) {
                hideSlashCommandSuggestions();
                return;
            }
            slashSelectedIndex = Math.min(slashSelectedIndex, filtered.length - 1);

            listEl.innerHTML = '';
            filtered.forEach((cmd, idx) => {
                const item = document.createElement('div');
                item.className = `px-3 py-2 flex items-center gap-3 cursor-pointer text-sm hover:bg-gray-700 ${idx === slashSelectedIndex ? 'bg-gray-700' : ''}`;
                item.innerHTML = `
                    <i class="fas ${cmd.icon || 'fa-terminal'} w-4 text-blue-400"></i>
                    <div class="flex-1 min-w-0">
                        <div class="font-mono text-blue-300">${cmd.label}</div>
                        <div class="text-[11px] text-gray-400 truncate">${cmd.description}</div>
                    </div>
                `;
                let selectedByPointer = false;
                item.addEventListener('pointerdown', (event) => {
                    if (typeof event.button === 'number' && event.button !== 0) return;
                    // On touch devices, textarea blur can hide and remove this
                    // item before the delayed click event is dispatched.
                    event.preventDefault();
                    selectedByPointer = true;
                    selectSlashCommand(cmd.id);
                });
                item.addEventListener('click', (event) => {
                    event.preventDefault();
                    if (!selectedByPointer) selectSlashCommand(cmd.id);
                });
                item.onmouseenter = () => {
                    slashSelectedIndex = idx;
                    showSlashCommandSuggestions(filter); // re-render highlight
                };
                listEl.appendChild(item);
            });

            // Smart vertical positioning to avoid going off-screen
            const rect = inputRow.getBoundingClientRect();
            const viewportHeight = window.innerHeight;
            const spaceBelow = viewportHeight - rect.bottom;
            const spaceAbove = rect.top;

            const desiredMaxHeight = 260; // px
            const padding = 8;

            box.style.position = 'fixed';
            box.style.left = `${Math.max(8, rect.left)}px`;
            box.style.zIndex = '80';
            box.style.maxHeight = 'none'; // reset

            // Decide whether to show below or above
            const showAbove = spaceBelow < 180 && spaceAbove > spaceBelow;

            if (showAbove) {
                // Place above the input row
                const popupHeight = Math.min(desiredMaxHeight, spaceAbove - padding);
                box.style.top = 'auto';
                box.style.bottom = `${viewportHeight - rect.top + 4}px`;
                listEl.style.maxHeight = `${popupHeight}px`;
            } else {
                // Place below (preferred)
                const popupHeight = Math.min(desiredMaxHeight, spaceBelow - padding);
                box.style.top = `${rect.bottom + 4}px`;
                box.style.bottom = 'auto';
                listEl.style.maxHeight = `${popupHeight}px`;
            }

            box.classList.remove('hidden');
            slashSuggestionsVisible = true;
        }

        function selectSlashCommand(cmdId) {
            const input = get('prompt-input');
            if (!input) return;

            const val = input.value;
            const token = extractSlashCommandToken(val);
            if (token !== null) {
                // Input begins with a slash command token, possibly followed
                // by argument text typed without a space. Strip only the
                // command part and keep the rest as the instruction.
                const trimmed = String(val || '').trimStart();
                input.value = trimmed.substring(1 + token.length).trimStart();
            } else {
                // Find the last '/' before cursor or end, and remove from there to the command
                const lastSlash = val.lastIndexOf('/');
                if (lastSlash !== -1) {
                    // Keep everything before the command start
                    input.value = val.substring(0, lastSlash).trimEnd();
                } else {
                    input.value = '';
                }
            }

            hideSlashCommandSuggestions();
            const command = SLASH_COMMANDS.find((candidate) => candidate.id === cmdId);
            const argument = input.value.trim();
            if (command && command.kind === 'minimal' && (!command.requiresArgument || argument)) {
                input.value = '';
                executeMinimalSlashCommand(cmdId, argument);
                input.dispatchEvent(new Event('input', { bubbles: true }));
                input.focus();
                return;
            }
            pendingSlashCommand = cmdId;

            // Show visual feedback that we are now in command argument mode
            showPendingSlashCommandIndicator(cmdId);

            // Optional: give a small hint in placeholder or just let user start typing the instruction
            input.focus();
            // Dispatch input to update any token estimate etc.
            input.dispatchEvent(new Event('input', { bubbles: true }));
        }

        const AI_SETTING_JUMP_TARGETS = {
            default_model: { label: '既定のモデル', tab: 'general', control: 'set-default-model' },
            default_vision_model: { label: 'Vision Model', tab: 'general', control: 'set-default-vision-model' },
            use_last_chat_settings: { label: '前回の設定を継続', tab: 'general', control: 'set-use-last-settings' },
            default_enable_search: { label: '既定のSearch', tab: 'general', control: 'set-default-search' },
            default_enable_url_context: { label: '既定のURLs', tab: 'general', control: 'set-default-url-context' },
            default_enable_maps: { label: '既定のMaps', tab: 'general', control: 'set-default-maps' },
            default_enable_python: { label: '既定のPython', tab: 'general', control: 'set-default-python' },
            default_enable_file_creation: { label: '既定のFile', tab: 'general', control: 'set-default-file-creation' },
            default_enable_thinking: { label: '既定のThinking', tab: 'general', control: 'set-default-thinking' },
            default_thinking_level: { label: 'Thinking Level', tab: 'general', control: 'set-default-thinking-level' },
            default_thinking_budget: { label: 'Thinking Budget', tab: 'general', control: 'set-default-thinking-budget' },
            default_reasoning_effort: { label: 'Reasoning Effort', tab: 'general', control: 'set-default-reasoning-effort' },
            default_enable_system_prompt: { label: '既定のSysPrompt', tab: 'general', control: 'set-default-sys-prompt' },
            default_enable_mcp: { label: '既定のMCP', tab: 'general', control: 'set-default-mcp' },
            default_safety_setting: { label: '既定のSafety', tab: 'general', control: 'set-default-safety' },
            auto_search_on_links: { label: 'Xリンクの自動検索', tab: 'general', control: 'set-auto-search-links' },
            mic_transcribe_mode: { label: 'マイク文字起こし方式', tab: 'general', control: 'set-mic-transcribe-mode' },
            stt_model: { label: 'STTモデル', tab: 'general', control: 'set-stt-model' },
            llm_transcribe_prompt: { label: 'LLM文字起こしプロンプト', tab: 'general', control: 'set-llm-transcribe-prompt' },
            enter_to_send: { label: 'Enterで送信', tab: 'general', control: 'set-enter-to-send' },
            compact_prompt_mode: { label: 'プロンプトバー表示', tab: 'general', control: 'set-compact-prompt-mode' },
            minimal_prompt_mode: { label: 'ミニマル表示', tab: 'general', control: 'set-minimal-prompt-mode' },
            temp_chat_timeout_seconds: { label: '一時チャット保持時間', tab: 'general', control: 'set-temp-chat-timeout-seconds' },
            system_prompt: { label: 'ユーザーシステムプロンプト', tab: 'prompt', control: 'sys-prompt-text' },
            system_prompt_enabled: { label: 'システムプロンプト', tab: 'prompt', control: 'set-global-sys-prompt-enabled' },
            apply_global_system_prompt: { label: 'ユーザープロンプトの適用', tab: 'prompt', control: 'set-apply-global-sys-prompt' },
            apply_auto_system_prompt_notices: { label: '自動注入プロンプト', tab: 'prompt', control: 'set-apply-auto-sys-prompt-notices' },
            auto_system_prompt_notices_config: { label: '自動注入プロンプト設定', tab: 'prompt', control: 'auto-sys-prompt-settings' },
            theme_color: { label: 'テーマカラー', tab: 'display', control: 'set-theme-color' },
            liquid_glass_enabled: { label: 'Liquid Glass', tab: 'display', control: 'set-liquid-glass' },
            use_sw_cache: { label: '高速キャッシュ', tab: 'data', control: 'set-use-sw-cache' },
            enable_latency_metrics: { label: 'レスポンス速度の計測', tab: 'data', control: 'set-latency-metrics' },
            enable_client_debug_log: { label: 'デバッグログの拡張送信', tab: 'data', control: 'set-client-debug-log' },
            bot_detection_enabled: { label: 'Bot Detection', tab: 'security', control: 'set-bot-detect' },
            skip_2fa_on_google_login: { label: 'Googleログイン時の2FA', tab: '2fa', control: 'set-skip-2fa-google' },
            default_2fa_method: { label: '既定の2FA方式', tab: '2fa', control: 'set-default-2fa-method' },
            rich_paste_prompt_default: { label: 'リッチ貼り付けプロンプト', modal: 'rich-paste', control: 'rich-paste-prompt' },
            rich_paste_prompt_use_custom_default: { label: 'リッチ貼り付けの既定値', modal: 'rich-paste', control: 'rich-paste-use-default' }
        };

        function formatAiSettingValue(value) {
            if (value === true) return 'ON';
            if (value === false) return 'OFF';
            if (value === '(更新)') return '更新済み';
            if (value === null || value === undefined || value === '') return '未設定';
            if (typeof value === 'object') {
                try { return JSON.stringify(value); } catch (e) { return '更新済み'; }
            }
            return String(value);
        }

        function findSettingsJumpElement(tabId, controlId) {
            const tab = get(`tab-${tabId}`);
            let element = get(controlId);
            if (!tab || !element) return null;
            while (element.parentElement && element.parentElement !== tab) element = element.parentElement;
            return element.parentElement === tab ? element : get(controlId);
        }

        function openAiSettingJumpTarget(key) {
            const target = AI_SETTING_JUMP_TARGETS[key];
            if (!target) {
                if (typeof window.openSettingsModal === 'function') window.openSettingsModal();
                return;
            }
            if (target.modal === 'rich-paste') {
                openRichPasteModal();
                setTimeout(() => {
                    const control = get(target.control);
                    if (control) {
                        control.scrollIntoView({ behavior: 'smooth', block: 'center' });
                        control.focus({ preventScroll: true });
                    }
                }, 260);
                return;
            }
            if (typeof window.openSettingsModal === 'function') window.openSettingsModal();
            setTimeout(() => {
                const element = findSettingsJumpElement(target.tab, target.control);
                if (element) jumpToSetting(target.tab, element);
                else switchTab(target.tab || 'general');
            }, 320);
        }

        function removeEphemeralMessageControls(messageEl) {
            if (!messageEl) return;
            const controls = messageEl.querySelector('.msg-controls');
            if (controls) controls.remove();
        }

        function renderAiSettingsResultBubble(values, modelId, mode = 'update') {
            const entries = Object.entries(values || {});
            const id = `settings-result-${Date.now()}`;
            const inspecting = mode === 'inspect';
            const message = entries.length
                ? (inspecting
                    ? '現在の設定を確認しました。\n\n確認した項目をタップすると、設定画面の該当箇所へ移動できます。'
                    : '設定を更新しました。\n\n変更した項目をタップすると、設定画面の該当箇所へ移動できます。')
                : (inspecting ? '確認できる設定項目がありませんでした。' : '変更された設定項目はありませんでした。');
            const messageEl = renderMessage(id, 'assistant', message, null, null, modelId, null, true, null, null, null, null, null, null, null, null, true);
            if (!messageEl) return;
            removeEphemeralMessageControls(messageEl);
            const bubble = messageEl.querySelector('.message-bubble');
            if (!bubble || !entries.length) return;

            const list = document.createElement('div');
            list.className = 'mt-3 space-y-2 ai-settings-result-list';
            entries.forEach(([key, value]) => {
                const target = AI_SETTING_JUMP_TARGETS[key] || { label: key };
                const button = document.createElement('button');
                button.type = 'button';
                button.className = 'w-full flex items-center gap-3 rounded-xl border border-white/10 bg-black/20 px-3 py-2.5 text-left hover:bg-black/30 hover:border-blue-400/40 transition ai-settings-result-item';
                const text = document.createElement('span');
                text.className = 'min-w-0 flex-1';
                const label = document.createElement('span');
                label.className = 'block text-xs font-bold text-blue-200';
                label.textContent = target.label;
                const valueEl = document.createElement('span');
                valueEl.className = 'block mt-0.5 text-[11px] text-gray-300 break-words';
                valueEl.textContent = formatAiSettingValue(value);
                const icon = document.createElement('i');
                icon.className = 'fas fa-arrow-up-right-from-square text-[10px] text-blue-300 shrink-0';
                text.appendChild(label);
                text.appendChild(valueEl);
                button.appendChild(text);
                button.appendChild(icon);
                button.addEventListener('click', () => openAiSettingJumpTarget(key));
                list.appendChild(button);
            });
            const footer = bubble.querySelector('.message-footer-meta');
            if (footer) bubble.insertBefore(list, footer);
            else bubble.appendChild(list);
            scrollToBottom();
        }

        async function runAiSettingsCommand(instruction, modelId) {
            // Keep the command context active so the next prompt is a follow-up
            // instruction instead of falling back to a normal chat request.
            if (pendingSlashCommand !== 'settings') {
                pendingSlashCommand = 'settings';
                showPendingSlashCommandIndicator('settings');
            }
            appendAiSettingsConversation('user', instruction);
            const timestamp = Date.now();
            const userEl = renderMessage(`settings-user-${timestamp}`, 'user', `/settings ${instruction}`, null, null, null, null, true, null, null, null, null, null, null, null, null, true);
            removeEphemeralMessageControls(userEl);
            const welcome = get('welcome-screen');
            if (welcome) welcome.classList.add('hidden');
            const pendingId = `settings-pending-${timestamp}`;
            const chat = get('chat-container');
            if (chat) {
                chat.insertAdjacentHTML('beforeend', `<div id="${pendingId}" class="flex justify-start mb-4 fade-in"><div class="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-md relative">${buildPendingSkeletonHtml(modelId, '設定リクエストを確認しています...')}</div></div>`);
                scrollToBottom();
            }
            try {
                const res = await apiFetch('/api/settings/apply-ai-prompt', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        prompt: instruction,
                        model: modelId,
                        conversation: aiSettingsConversation,
                    })
                });
                const data = await res.json().catch(() => ({}));
                const pending = get(pendingId);
                if (pending) pending.remove();
                if (data && data.status === 'ok' && data.mode === 'inspect' && data.current) {
                    appendAiSettingsConversation('assistant', summarizeAiSettingsConversationValues(data.current, 'inspect'));
                    showToast(`現在の設定を確認しました（${Object.keys(data.current).length}項目）`, 'success');
                    renderAiSettingsResultBubble(data.current, modelId, 'inspect');
                    return;
                }
                if (data && data.status === 'ok' && data.applied) {
                    appendAiSettingsConversation('assistant', summarizeAiSettingsConversationValues(data.applied, 'update'));
                    showToast(`設定を更新しました（${Object.keys(data.applied).length}項目）`, 'success');
                    try {
                        const fresh = await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery).then((response) => response.json());
                        populateAiSafeFormFields(fresh);
                        cacheUserSettings(fresh);
                    } catch (e) {}
                    renderAiSettingsResultBubble(data.applied, modelId);
                    return;
                }
                const msg = data.message || data.error || '設定変更に失敗しました';
                appendAiSettingsConversation('assistant', `設定操作に失敗しました: ${msg}`);
                const errorEl = renderMessage(`settings-error-${Date.now()}`, 'assistant', `設定変更に失敗しました。\n\n${msg}`, null, null, modelId, null, true, null, null, null, null, null, null, null, null, true);
                removeEphemeralMessageControls(errorEl);
                showToast(msg, 'error', true);
            } catch (error) {
                appendAiSettingsConversation('assistant', '設定操作の通信に失敗しました。');
                const pending = get(pendingId);
                if (pending) pending.remove();
                const errorEl = renderMessage(`settings-error-${Date.now()}`, 'assistant', '設定変更の通信に失敗しました。時間をおいて再度お試しください。', null, null, modelId, null, true, null, null, null, null, null, null, null, null, true);
                removeEphemeralMessageControls(errorEl);
                showToast('設定変更の通信に失敗しました', 'error', true);
            }
        }

        // === Gem suggestion helpers (triggered by @ in prompt bar) ===
        function hideGemSuggestions() {
            const box = get('gem-suggestions');
            if (box) box.classList.add('hidden');
            gemSuggestionsVisible = false;
            gemSelectedIndex = 0;
        }

        function showGemSuggestions(filter = '') {
            const box = get('gem-suggestions');
            const listEl = get('gem-suggestions-list');
            const inputRow = get('input-row');
            if (!box || !listEl || !inputRow) return;

            if (!loadedGems || loadedGems.length === 0) {
                hideGemSuggestions();
                return;
            }

            const f = filter.toLowerCase();
            const filtered = loadedGems.filter(g =>
                g.name.toLowerCase().includes(f) || (g.description && g.description.toLowerCase().includes(f))
            );

            if (filtered.length === 0) {
                hideGemSuggestions();
                return;
            }

            // Clamp selected index
            if (gemSelectedIndex >= filtered.length) gemSelectedIndex = 0;

            listEl.innerHTML = '';
            filtered.forEach((gem, idx) => {
                const item = document.createElement('div');
                item.className = `px-3 py-2 flex items-center gap-3 cursor-pointer text-sm hover:bg-gray-700 ${idx === gemSelectedIndex ? 'bg-gray-700' : ''}`;
                item.innerHTML = `
                    <i class="fas fa-gem w-4 text-blue-400"></i>
                    <div class="flex-1 min-w-0">
                        <div class="text-blue-300 truncate font-medium">${escapeHtml(gem.name)}</div>
                        ${gem.description ? `<div class="text-[11px] text-gray-400 truncate">${escapeHtml(gem.description)}</div>` : ''}
                    </div>
                `;
                item.onclick = () => selectGemSuggestion(gem);
                item.onmouseenter = () => {
                    gemSelectedIndex = idx;
                    showGemSuggestions(filter);
                };
                listEl.appendChild(item);
            });

            // Smart vertical positioning (same logic as slash commands)
            const rect = inputRow.getBoundingClientRect();
            const viewportHeight = window.innerHeight;
            const spaceBelow = viewportHeight - rect.bottom;
            const spaceAbove = rect.top;

            const desiredMaxHeight = 260;
            const padding = 8;

            box.style.position = 'fixed';
            box.style.left = `${Math.max(8, rect.left)}px`;
            box.style.zIndex = '80';
            box.style.maxHeight = 'none';

            const showAbove = spaceBelow < 180 && spaceAbove > spaceBelow;

            if (showAbove) {
                const popupHeight = Math.min(desiredMaxHeight, spaceAbove - padding);
                box.style.top = 'auto';
                box.style.bottom = `${viewportHeight - rect.top + 4}px`;
                listEl.style.maxHeight = `${popupHeight}px`;
            } else {
                const popupHeight = Math.min(desiredMaxHeight, spaceBelow - padding);
                box.style.top = `${rect.bottom + 4}px`;
                box.style.bottom = 'auto';
                listEl.style.maxHeight = `${popupHeight}px`;
            }

            box.classList.remove('hidden');
            gemSuggestionsVisible = true;
        }

        function selectGemSuggestion(gem) {
            const input = get('prompt-input');
            if (!input) return;

            const val = input.value;
            const lastAt = val.lastIndexOf('@');
            if (lastAt !== -1) {
                input.value = val.substring(0, lastAt).trimEnd();
            } else {
                input.value = '';
            }

            hideGemSuggestions();
            activateGem(gem);
            input.focus();
            input.dispatchEvent(new Event('input', { bubbles: true }));
        }

        function browserFastModeIneligibility(rawText) {
            const model = String(get('model-select') ? get('model-select').value : '').toLowerCase();
            if (!rawText || !rawText.trim()) return 'プロンプトを入力してください';
            if (!model.startsWith('gemini-') || /(image|native-audio|tts|live)/.test(model)) return 'Geminiテキストモデル専用です';
            if (currentImageUrls.length) return 'サーバー保存済み添付があるため通常モードが必要です';
            if (activeGem) return 'Gems利用時は通常モードが必要です';
            if (currentQuote || editingMessageId) return '引用・編集時は通常モードが必要です';
            if (codingModeEnabled) return 'Coding Mode利用時は通常モードが必要です';
            const enabledIds = ['enable-search', 'enable-url-context', 'enable-maps', 'enable-sys-prompt', 'enable-prompt-cache', 'enable-mcp'];
            if (enabledIds.some((id) => { const el = get(id); return !!(el && el.checked); })) return '検索・URL参照・システム機能利用時は通常モードが必要です';
            const custom = get('thread-custom-instruction');
            if (custom && String(custom.value || '').trim()) return 'チャット固有指示利用時は通常モードが必要です';
            const entries = Array.from(browserFastLocalFiles.values());
            if (entries.length > BROWSER_FAST_MAX_IMAGES) return '画像は4枚までです';
            const total = entries.reduce((sum, entry) => sum + Number(entry.file && entry.file.size || 0), 0);
            if (total > BROWSER_FAST_MAX_BYTES) return '画像合計は12MBまでです';
            if (entries.some((entry) => !entry.file || !String(entry.file.type || '').startsWith('image/'))) return '画像以外は利用できません';
            return '';
        }

        function fileToBase64Payload(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => {
                    const raw = String(reader.result || '');
                    const comma = raw.indexOf(',');
                    if (comma < 0) return reject(new Error('画像の読み込みに失敗しました'));
                    resolve(raw.slice(comma + 1));
                };
                reader.onerror = () => reject(reader.error || new Error('画像の読み込みに失敗しました'));
                reader.readAsDataURL(file);
            });
        }

        async function buildBrowserFastHistoryContents(history) {
            const contents = [];
            let omittedImages = 0;
            for (const item of (Array.isArray(history) ? history : [])) {
                if (!item || !['user', 'model'].includes(item.role)) continue;
                const parts = [];
                if (item.role === 'model' && Array.isArray(item.thought_signatures)) {
                    item.thought_signatures.forEach((signature) => {
                        if (signature) parts.push({ thoughtSignature: String(signature) });
                    });
                }
                if (item.text) parts.push({ text: String(item.text) });
                for (const image of (Array.isArray(item.images) ? item.images : [])) {
                    try {
                        const imageResponse = await fetch(buildFileUrl(image.path), {
                            credentials: 'same-origin',
                            cache: 'no-store',
                        });
                        if (!imageResponse.ok) throw new Error(`HTTP ${imageResponse.status}`);
                        const blob = await imageResponse.blob();
                        parts.push({
                            inlineData: {
                                mimeType: image.mime_type || blob.type || 'application/octet-stream',
                                data: await fileToBase64Payload(blob),
                            },
                        });
                    } catch (error) {
                        omittedImages++;
                    }
                }
                if (parts.length) contents.push({ role: item.role, parts });
            }
            if (omittedImages) {
                showToast(`履歴画像${omittedImages}件を再取得できなかったため、テキスト履歴だけで続行します`, 'warning', true);
            }
            return contents;
        }

        async function uploadBrowserFastLocalFiles() {
            const entries = Array.from(browserFastLocalFiles.entries());
            for (const [uploadId, entry] of entries) {
                if (!entry || !entry.file || !entry.rowObj) throw new Error('ローカル画像の状態が失われました');
                if (entry.rowObj.status) entry.rowObj.status.textContent = '回答完了・サーバー保存中...';
                const ok = await uploadFileWithProgress(entry.file, entry.rowObj);
                if (!ok) throw new Error(`${entry.file.name || '画像'}をサーバーへ保存できませんでした`);
                browserFastLocalFiles.delete(uploadId);
            }
        }

        function browserFastThinkingConfig(model) {
            const thinking = get('enable-thinking');
            if (!thinking || !thinking.checked) return null;
            const rawLevel = String(get('thinking-level') ? get('thinking-level').value : 'high').toLowerCase();
            if (model.includes('2.5')) {
                const manual = Number(get('thinking-budget') ? get('thinking-budget').value : 4096);
                const budget = Number.isFinite(manual) ? Math.max(0, Math.min(32768, Math.trunc(manual))) : 4096;
                return { includeThoughts: true, thinkingBudget: budget };
            }
            let level = rawLevel.toUpperCase();
            if (model.includes('3.6') && !['MEDIUM', 'HIGH'].includes(level)) level = 'MEDIUM';
            if (model.includes('3.5') && !['MINIMAL', 'MEDIUM', 'HIGH'].includes(level)) level = 'MINIMAL';
            return { includeThoughts: true, thinkingLevel: level };
        }

        function browserFastPythonBoxHtml(pyId) {
            return `<div class="code-wrapper python-box collapsed" data-py-id="${pyId}" data-collapsed="true" data-code-key="${pyId}"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Execution</span><div class="code-actions"><button class="code-toggle" aria-expanded="false" title="展開" aria-label="展開"><i class="fas fa-chevron-down"></i></button><button class="copy-btn" data-copy="code" data-code="" title="コードをコピー" aria-label="コードをコピー"><i class="fas fa-copy"></i></button><button class="copy-btn" data-copy="output" data-code="" title="出力をコピー" aria-label="出力をコピー"><i class="fas fa-align-left"></i></button></div></div><div class="code-body"><div class="python-section"><div class="python-label">Code</div><pre><code class="hljs language-python python-code"></code></pre></div><div class="python-section"><div class="python-label">Output</div><pre><code class="hljs language-plaintext python-output"></code></pre></div></div></div>`;
        }

        function updateBrowserFastPythonBox(box, field, value) {
            if (!box) return;
            if (field === 'code') {
                const codeText = value == null ? '' : String(value);
                const codeEl = box.querySelector('.python-code');
                if (codeEl) {
                    codeEl.textContent = codeText;
                    codeEl.removeAttribute('data-highlighted');
                    queueHighlight(box, codeText);
                }
                const codeBtn = box.querySelector('.copy-btn[data-copy="code"]');
                if (codeBtn) codeBtn.setAttribute('data-code', encodeURIComponent(codeText).replace(/'/g, "%27"));
            } else if (field === 'output') {
                const outText = value == null ? '' : String(value);
                const outEl = box.querySelector('.python-output');
                if (outEl) outEl.textContent = outText;
                const outBtn = box.querySelector('.copy-btn[data-copy="output"]');
                if (outBtn) outBtn.setAttribute('data-code', encodeURIComponent(outText).replace(/'/g, "%27"));
            }
        }
