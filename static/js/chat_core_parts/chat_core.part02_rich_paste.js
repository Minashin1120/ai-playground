        const normalizeRichPastePrintHtml = (contentHtml) => {
            const template = document.createElement('template');
            template.innerHTML = String(contentHtml || '');
            const nodes = Array.from(template.content.querySelectorAll('*'));
            const complexLayoutCount = nodes.reduce((total, node) => {
                const display = String(node.style && node.style.display || '').trim().toLowerCase();
                return total + (['flex', 'inline-flex', 'grid', 'inline-grid'].includes(display) ? 1 : 0);
            }, 0);
            const screenLayoutCount = nodes.reduce((total, node) => {
                if (!node || !node.style || !['article', 'div', 'main', 'section'].includes(String(node.tagName || '').toLowerCase())) {
                    return total;
                }
                const styleText = String(node.getAttribute('style') || '');
                const largeSidePadding = Array.from(
                    styleText.matchAll(/(?:^|;)\s*padding(?:-left|-right|-inline|-inline-start|-inline-end)?\s*:\s*([^;]+)/gi)
                ).some((match) => Array.from(match[1].matchAll(/(-?\d+(?:\.\d+)?)px/gi))
                    .some((pixelMatch) => Math.abs(Number(pixelMatch[1]) || 0) >= 96));
                const oversizedWidth = Array.from(
                    styleText.matchAll(/(?:^|;)\s*(?:width|min-width)\s*:\s*(-?\d+(?:\.\d+)?)px/gi)
                ).some((match) => Math.abs(Number(match[1]) || 0) > 720);
                return total + (largeSidePadding || oversizedWidth ? 1 : 0);
            }, 0);
            if (nodes.length <= 500 && complexLayoutCount <= 24 && screenLayoutCount === 0) {
                return template.innerHTML;
            }

            const layoutProps = new Set([
                'align-items', 'align-self', 'column-gap', 'flex', 'flex-basis', 'flex-direction',
                'flex-grow', 'flex-shrink', 'flex-wrap', 'gap', 'grid', 'grid-auto-columns',
                'grid-auto-flow', 'grid-auto-rows', 'grid-column', 'grid-column-end',
                'grid-column-start', 'grid-row', 'grid-row-end', 'grid-row-start', 'grid-template',
                'grid-template-areas', 'grid-template-columns', 'grid-template-rows',
                'justify-content', 'justify-items', 'justify-self', 'order', 'row-gap'
            ]);
            const blockWidthTags = new Set(['article', 'div', 'main', 'section']);
            const sidePaddingProps = new Set([
                'padding', 'padding-left', 'padding-right', 'padding-inline',
                'padding-inline-start', 'padding-inline-end'
            ]);
            nodes.forEach((node) => {
                if (!node || !node.style) return;
                const tag = String(node.tagName || '').toLowerCase();
                const declarations = [];
                String(node.getAttribute('style') || '').split(';').forEach((declaration) => {
                    if (!declaration || declaration.indexOf(':') < 0) return;
                    const separator = declaration.indexOf(':');
                    const prop = declaration.slice(0, separator).trim().toLowerCase();
                    let value = declaration.slice(separator + 1).trim();
                    if (!prop || !value || layoutProps.has(prop)) return;
                    if (['height', 'max-height', 'min-height', 'overflow', 'overflow-x', 'overflow-y'].includes(prop)) {
                        return;
                    }
                    if (['width', 'min-width'].includes(prop) && blockWidthTags.has(tag)) {
                        return;
                    }
                    if (sidePaddingProps.has(prop) && blockWidthTags.has(tag)) {
                        const pixelValues = Array.from(value.matchAll(/(-?\d+(?:\.\d+)?)px/gi))
                            .map((match) => Math.abs(Number(match[1]) || 0));
                        if (pixelValues.some((pixelValue) => pixelValue >= 96)) {
                            value = '0px';
                        }
                    }
                    if (prop === 'display') {
                        const displayValue = value.toLowerCase();
                        if (['flex', 'grid'].includes(displayValue)) value = 'block';
                        else if (['inline-flex', 'inline-grid'].includes(displayValue)) value = 'inline-block';
                    }
                    declarations.push(`${prop}: ${value}`);
                });
                if (declarations.length) node.setAttribute('style', declarations.join('; '));
                else node.removeAttribute('style');
            });
            return template.innerHTML;
        };
        const getRichPasteSelectionRange = (editor) => {
            const selection = window.getSelection && window.getSelection();
            if (!selection || !selection.rangeCount) return null;
            const range = selection.getRangeAt(0);
            if (editor && editor.contains(range.commonAncestorContainer)) return range;
            const fallback = document.createRange();
            fallback.selectNodeContents(editor);
            fallback.collapse(false);
            return fallback;
        };
        const insertNodeIntoRichPasteEditor = (node) => {
            const editor = getRichPasteEditor();
            if (!editor || !node) return;
            editor.appendChild(node);
            updateRichPasteStatus();
        };
        const insertHtmlIntoRichPasteEditor = (html) => {
            const safeHtml = sanitizeRichPasteHtml(html);
            if (!safeHtml || safeHtml.trim() === '') return false;
            const template = document.createElement('template');
            template.innerHTML = safeHtml;
            const frag = template.content.cloneNode(true);
            insertNodeIntoRichPasteEditor(frag);
            return true;
        };
        const insertTextIntoRichPasteEditor = (text) => {
            if (text === null || text === undefined) return;
            const node = document.createTextNode(String(text));
            insertNodeIntoRichPasteEditor(node);
        };
        const blobToDataUrl = (blob) => new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(String(reader.result || ''));
            reader.onerror = () => reject(reader.error || new Error('clipboard_image_read_failed'));
            reader.readAsDataURL(blob);
        });
        const insertClipboardImageBlob = async (blob, alt = 'clipboard-image') => {
            if (!blob) return false;
            const dataUrl = await blobToDataUrl(blob);
            if (!dataUrl) return false;
            insertHtmlIntoRichPasteEditor(`<p><img src="${escapeHtml(dataUrl)}" alt="${escapeHtml(alt)}"></p>`);
            return true;
        };
        const readClipboardRichContent = async () => {
            if (!navigator.clipboard || !navigator.clipboard.read) {
                throw new Error('このブラウザはリッチクリップボード読み取りに対応していません');
            }
            const capture = getRichPasteCapture();
            if (capture) capture.value = '';
            const items = await navigator.clipboard.read();
            if (!items || !items.length) return false;
            let inserted = false;
            for (const item of items) {
                if (!item) continue;
                const itemTypes = Array.from(item.types || []);
                let htmlInserted = false;
                if (itemTypes.includes('text/html')) {
                    const htmlBlob = await item.getType('text/html');
                    const html = await htmlBlob.text();
                    if (html && insertHtmlIntoRichPasteEditor(html)) {
                        inserted = true;
                        htmlInserted = true;
                    }
                }
                if (!htmlInserted && itemTypes.includes('text/plain')) {
                    const textBlob = await item.getType('text/plain');
                    const text = await textBlob.text();
                    if (text) {
                        insertTextIntoRichPasteEditor(text);
                        inserted = true;
                    }
                }
                const imageType = itemTypes.find((type) => type && type.startsWith('image/'));
                if (!htmlInserted && imageType) {
                    const imageBlob = await item.getType(imageType);
                    if (await insertClipboardImageBlob(imageBlob, 'clipboard-image')) {
                        inserted = true;
                    }
                }
            }
            return inserted;
        };
        const ingestRichPasteClipboardData = async (clipboardData) => {
            if (!clipboardData) return false;
            let inserted = false;
            const html = clipboardData.getData && clipboardData.getData('text/html');
            const text = clipboardData.getData && clipboardData.getData('text/plain');
            let htmlInserted = false;
            if (html) {
                if (insertHtmlIntoRichPasteEditor(html)) {
                    inserted = true;
                    htmlInserted = true;
                }
            }
            if (!htmlInserted && text) {
                insertTextIntoRichPasteEditor(text);
                inserted = true;
            }
            const items = Array.from(clipboardData.items || []);
            const imageFiles = items
                .filter((item) => item && item.kind === 'file')
                .map((item) => item.getAsFile())
                .filter((file) => file && file.type && file.type.startsWith('image/'));
            if (!htmlInserted && imageFiles.length) {
                for (const imageFile of imageFiles) {
                    try {
                        if (await insertClipboardImageBlob(imageFile, imageFile.name || 'clipboard-image')) {
                            inserted = true;
                        }
                    } catch (e) {}
                }
            }
            return inserted;
        };
        const buildRichPastePdfFilename = () => {
            const now = new Date();
            const pad = (n) => String(n).padStart(2, '0');
            return `clipboard_rich_${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}.pdf`;
        };
        const getRichPasteProgressElements = () => {
            return {
                container: get('rich-paste-progress-container'),
                bar: get('rich-paste-progress-bar'),
                text: get('rich-paste-progress-text')
            };
        };
        const setRichPasteProgress = (pct, text = null) => {
            const { container, bar, text: textEl } = getRichPasteProgressElements();
            const safePct = Math.max(0, Math.min(100, Number(pct) || 0));
            if (container) {
                container.classList.remove('hidden');
                container.style.setProperty('display', 'block', 'important');
            }
            if (bar) {
                bar.style.width = `${safePct}%`;
                bar.style.transform = 'none';
            }
            if (textEl) {
                textEl.textContent = `${Math.round(safePct)}%`;
            }
            if (text && container) {
                const label = container.querySelector('.text-amber-400');
                if (label) label.innerHTML = `<i class="fas fa-spinner fa-spin"></i> ${escapeHtml(text)}`;
            }
        };
        const hideRichPasteProgress = () => {
            const { container, bar } = getRichPasteProgressElements();
            if (bar) {
                bar.style.transform = 'scaleX(0)';
            }
            if (container) {
                container.classList.add('hidden');
                container.style.display = 'none';
            }
        };
        const inferRichPasteTitle = () => {
            const editor = getRichPasteEditor();
            if (!editor) return 'Clipboard Export';
            const heading = editor.querySelector('h1, h2, h3, h4, h5, h6');
            if (heading && heading.textContent && heading.textContent.trim()) {
                return heading.textContent.trim().slice(0, 48);
            }
            const text = (editor.innerText || '').trim().replace(/\s+/g, ' ');
            if (!text) return 'Clipboard Export';
            return text.slice(0, 48);
        };
        const waitForRichPasteMedia = async (root, timeoutMs = 2500) => {
            if (!root) return;
            const settle = new Promise((resolve) => setTimeout(resolve, Math.max(0, timeoutMs)));
            const loadPromise = Promise.all(Array.from(root.querySelectorAll('img') || []).map((img) => {
                if (!img) return Promise.resolve();
                if (img.complete) return Promise.resolve();
                return new Promise((resolve) => {
                    let done = false;
                    const finish = () => {
                        if (done) return;
                        done = true;
                        resolve();
                    };
                    img.addEventListener('load', finish, { once: true });
                    img.addEventListener('error', finish, { once: true });
                    setTimeout(finish, Math.max(250, Math.min(timeoutMs, 2000)));
                });
            }));
            await Promise.race([loadPromise, settle]);
            if (document.fonts && document.fonts.ready) {
                try {
                    await Promise.race([
                        document.fonts.ready,
                        settle
                    ]);
                } catch (e) {}
            }
        };
        const normalizeRichPastePdfText = (value) => {
            return String(value || '')
                .replace(/\u00a0/g, ' ')
                .replace(/\r\n?/g, '\n')
                .replace(/[ \t\f\v]+/g, ' ')
                .replace(/\n[ \t]+/g, '\n')
                .replace(/[ \t]+\n/g, '\n')
                .replace(/\n{3,}/g, '\n\n')
                .trim();
        };
        const normalizeRichPastePdfCodeText = (value) => {
            return String(value || '').replace(/\u00a0/g, ' ').replace(/\r\n?/g, '\n');
        };
        const collectRichPasteInlineSegments = (node, options = {}) => {
            if (!node) return [];
            const allowLinks = options.allowLinks !== false;
            const segments = [];
            const walk = (current, state) => {
                if (!current) return;
                if (current.nodeType === Node.TEXT_NODE) {
                    const text = current.textContent || '';
                    if (text) {
                        segments.push(Object.assign({}, state, { text }));
                    }
                    return;
                }
                if (current.nodeType !== Node.ELEMENT_NODE) return;
                const tag = String(current.tagName || '').toLowerCase();
                if (RICH_PASTE_NOISE_TAGS.has(tag)) return;
                if (tag === 'br') {
                    segments.push({ text: '\n' });
                    return;
                }
                const newState = Object.assign({}, state);
                if (['b', 'strong'].includes(tag)) newState.bold = true;
                if (['i', 'em'].includes(tag)) newState.italic = true;
                if (tag === 'a' && allowLinks) {
                    newState.link = String(current.getAttribute('href') || '').trim();
                }
                if (tag === 'code') newState.monospace = true;

                Array.from(current.childNodes || []).forEach((child) => walk(child, newState));
            };
            walk(node, { bold: !!options.bold, italic: !!options.italic });
            return segments;
        };
        const collectRichPasteInlineText = (node, options = {}) => {
            const segments = collectRichPasteInlineSegments(node, options);
            return segments.map(s => s.text).join('');
        };
        const collectRichPasteTableRows = (tableEl) => {
            const rows = [];
            Array.from(tableEl.querySelectorAll('tr') || []).forEach((row) => {
                if (row && row.closest && row.closest('table') === tableEl) {
                    rows.push(row);
                }
            });
            return rows;
        };
        const makeRichPasteTableMarkdown = (tableEl) => {
            const caption = tableEl && tableEl.querySelector ? tableEl.querySelector('caption') : null;
            const captionText = caption ? normalizeRichPastePdfText(collectRichPasteInlineText(caption)) : '';
            const rows = collectRichPasteTableRows(tableEl).map((row) => {
                const cells = Array.from(row.children || []).filter((cell) => {
                    const tag = String(cell.tagName || '').toLowerCase();
                    return tag === 'th' || tag === 'td';
                }).map((cell) => normalizeRichPastePdfText(collectRichPasteInlineText(cell)) || ' ');
                return cells;
            }).filter((row) => row.length);
            if (!rows.length) {
                return captionText || '[table]';
            }
            const colCount = rows.reduce((max, row) => Math.max(max, row.length), 0);
            const paddedRows = rows.map((row) => {
                const copy = row.slice(0, colCount);
                while (copy.length < colCount) copy.push(' ');
                return copy;
            });
            const separator = `| ${Array(colCount).fill('---').join(' | ')} |`;
            const lines = [];
            if (captionText) {
                lines.push(`Table: ${captionText}`);
                lines.push('');
            }
            lines.push(`| ${paddedRows[0].join(' | ')} |`);
            lines.push(separator);
            for (let i = 1; i < paddedRows.length; i += 1) {
                lines.push(`| ${paddedRows[i].join(' | ')} |`);
            }
            return lines.join('\n');
        };
        const collectRichPasteListBlocks = (listEl, ordered = false, depth = 0) => {
            const blocks = [];
            const items = Array.from(listEl.children || []).filter((child) => {
                const tag = String(child.tagName || '').toLowerCase();
                return tag === 'li';
            });
            let itemIndex = 1;
            items.forEach((li) => {
                const clone = li.cloneNode(true);
                Array.from(clone.querySelectorAll('ul,ol') || []).forEach((nested) => {
                    try { nested.remove(); } catch (e) {}
                });
                const segments = collectRichPasteInlineSegments(clone);
                if (segments.length > 0) {
                    blocks.push({
                        type: 'list_item',
                        ordered,
                        depth,
                        index: itemIndex,
                        segments: segments
                    });
                }
                Array.from(li.children || []).forEach((child) => {
                    const tag = String(child.tagName || '').toLowerCase();
                    if (tag === 'ul' || tag === 'ol') {
                        blocks.push(...collectRichPasteListBlocks(child, tag === 'ol', depth + 1));
                    }
                });
                itemIndex += 1;
            });
            return blocks;
        };
        const collectRichPastePdfBlocks = (root, depth = 0) => {
            const blocks = [];
            if (!root) return blocks;
            let buffer = [];
            const flushBuffer = () => {
                if (buffer.length === 0) return;
                blocks.push({ type: 'paragraph', segments: [...buffer] });
                buffer = [];
            };
            Array.from(root.childNodes || []).forEach((node) => {
                if (!node) return;
                if (node.nodeType === Node.TEXT_NODE) {
                    const text = (node.textContent || '').replace(/\u00a0/g, ' ');
                    if (text) buffer.push({ text });
                    return;
                }
                if (node.nodeType !== Node.ELEMENT_NODE) return;
                const tag = String(node.tagName || '').toLowerCase();
                if (RICH_PASTE_NOISE_TAGS.has(tag)) return;
                if (tag === 'br') {
                    buffer.push({ text: '\n' });
                    return;
                }
                if (/^h[1-6]$/.test(tag)) {
                    flushBuffer();
                    const segments = collectRichPasteInlineSegments(node);
                    if (segments.length > 0) {
                        blocks.push({ type: 'heading', level: Number(tag.slice(1)) || 1, segments });
                    }
                    return;
                }
                if (tag === 'p') {
                    flushBuffer();
                    const segments = collectRichPasteInlineSegments(node);
                    if (segments.length > 0) {
                        blocks.push({ type: 'paragraph', segments });
                    }
                    return;
                }
                if (tag === 'blockquote') {
                    flushBuffer();
                    const segments = collectRichPasteInlineSegments(node, { italic: true });
                    if (segments.length > 0) {
                        blocks.push({ type: 'blockquote', segments });
                    }
                    return;
                }
                if (tag === 'pre') {
                    flushBuffer();
                    const text = normalizeRichPastePdfCodeText(node.innerText || node.textContent || '');
                    if (text.trim()) {
                        blocks.push({ type: 'code', text });
                    }
                    return;
                }
                if (tag === 'table') {
                    flushBuffer();
                    const text = makeRichPasteTableMarkdown(node);
                    if (text) {
                        blocks.push({ type: 'table', text });
                    }
                    return;
                }
                if (tag === 'ul' || tag === 'ol') {
                    flushBuffer();
                    blocks.push(...collectRichPasteListBlocks(node, tag === 'ol', depth));
                    return;
                }
                if (tag === 'hr') {
                    flushBuffer();
                    blocks.push({ type: 'hr' });
                    return;
                }
                if (tag === 'figure') {
                    flushBuffer();
                    const img = node.querySelector('img');
                    if (img) {
                        blocks.push({
                            type: 'image',
                            src: String(img.getAttribute('src') || '').trim(),
                            alt: String(img.getAttribute('alt') || img.getAttribute('title') || '').trim(),
                            title: String(img.getAttribute('title') || '').trim()
                        });
                    }
                    const figcaption = node.querySelector('figcaption');
                    if (figcaption) {
                        const segments = collectRichPasteInlineSegments(figcaption);
                        if (segments.length > 0) {
                            blocks.push({ type: 'paragraph', segments });
                        }
                    }
                    return;
                }
                if (tag === 'img') {
                    flushBuffer();
                    blocks.push({
                        type: 'image',
                        src: String(node.getAttribute('src') || '').trim(),
                        alt: String(node.getAttribute('alt') || node.getAttribute('title') || '').trim(),
                        title: String(node.getAttribute('title') || '').trim()
                    });
                    return;
                }
                if (tag === 'li') {
                    flushBuffer();
                    blocks.push(...collectRichPasteListBlocks(node, false, depth));
                    return;
                }
                const hasBlockChild = Array.from(node.children || []).some((child) => {
                    const childTag = String(child.tagName || '').toLowerCase();
                    return /^h[1-6]$/.test(childTag) || ['p', 'div', 'section', 'article', 'main', 'blockquote', 'pre', 'table', 'ul', 'ol', 'hr', 'figure', 'img', 'li'].includes(childTag);
                });
                if (hasBlockChild && ['div', 'section', 'article', 'main', 'figure'].includes(tag)) {
                    flushBuffer();
                    blocks.push(...collectRichPastePdfBlocks(node, depth + 1));
                    return;
                }
                const inlineSegments = collectRichPasteInlineSegments(node);
                if (inlineSegments.length > 0) {
                    buffer.push(...inlineSegments);
                }
            });
            flushBuffer();
            return blocks;
        };
        const detectImageMimeType = (dataUrl) => {
            const match = String(dataUrl || '').match(/^data:(image\/[a-z0-9.+-]+);/i);
            return match ? match[1].toLowerCase() : 'image/png';
        };
        const loadRichPasteImageData = async (src, timeoutMs = 3000) => {
            const rawSrc = String(src || '').trim();
            if (!rawSrc) return null;
            if (rawSrc.startsWith('data:image/')) {
                return { dataUrl: rawSrc, mimeType: detectImageMimeType(rawSrc) };
            }
            let resolvedUrl = null;
            try {
                resolvedUrl = new URL(rawSrc, window.location.href);
            } catch (e) {
                return null;
            }
            const allowedSameOrigin = resolvedUrl.origin === window.location.origin;
            if (!allowedSameOrigin) {
                return null;
            }
            const fetchPromise = (async () => {
                try {
                    const response = await fetch(resolvedUrl.toString(), { credentials: 'same-origin', cache: 'force-cache' });
                    if (!response.ok) return null;
                    const blob = await response.blob();
                    const dataUrl = await blobToDataUrl(blob);
                    return {
                        dataUrl,
                        mimeType: blob.type || detectImageMimeType(dataUrl)
                    };
                } catch (e) {
                    return null;
                }
            })();
            return await Promise.race([
                fetchPromise,
                new Promise((resolve) => setTimeout(() => resolve(null), Math.max(250, timeoutMs)))
            ]);
        };
        const buildRichPastePreviewHtml = (mode = 'preview') => {
            const editor = getRichPasteEditor();
            if (!editor) return '';
            const title = inferRichPasteTitle();
            const createdAt = new Date().toLocaleString('ja-JP');
            const sanitizedHtml = sanitizeRichPasteHtml(editor.innerHTML || '');
            const theme = detectRichPasteTheme(sanitizedHtml);
            const contentHtml = normalizeRichPastePrintHtml(sanitizedHtml);
            const isPdf = mode === 'pdf';
            return `<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${escapeHtml(title)} - Preview</title>
  <style>
        :root {
          color-scheme: ${theme.mode};
          --rp-background: ${theme.background};
          --rp-foreground: ${theme.foreground};
          --rp-muted: ${theme.muted};
          --rp-border: ${theme.border};
          --rp-surface: ${theme.surface};
          --rp-quote: ${theme.quote};
          --rp-link: ${theme.link};
        }
	    body { margin: 0; background: ${isPdf ? 'var(--rp-background)' : '#eef2f7'}; color: var(--rp-foreground); font-family: "Noto Sans JP", system-ui, sans-serif; }
	    .page { max-width: ${isPdf ? '794px' : '920px'}; margin: 0 auto; padding: ${isPdf ? '28px 30px 36px' : '24px'}; }
	    .card { background: var(--rp-background); color: var(--rp-foreground); border: 1px solid var(--rp-border); border-radius: 18px; padding: 20px; box-shadow: ${isPdf ? 'none' : '0 18px 45px rgba(15,23,42,0.14)'}; }
	    .title { margin: 0; font-size: ${isPdf ? '22px' : '24px'}; line-height: 1.35; color: var(--rp-foreground); }
	    .meta { margin-top: 8px; color: var(--rp-muted); font-size: 12px; }
	    .content { margin-top: 18px; color: var(--rp-foreground); font-size: 15px; line-height: 1.7; word-break: break-word; overflow-wrap: anywhere; }
	    .content img, .content video, .content iframe, .content table, .content pre, .content blockquote { max-width: 100%; }
	    .content table { display: block; overflow-x: auto; border-collapse: collapse; }
	    .content th, .content td { border: 1px solid var(--rp-border); padding: 8px 10px; }
	    .content th { background: var(--rp-surface); }
	    .content pre { padding: 14px 16px; border: 1px solid var(--rp-border); border-radius: 14px; background: var(--rp-surface); color: var(--rp-foreground); overflow: auto; }
	    .content code { background: var(--rp-surface); color: var(--rp-foreground); }
	    .content pre code { background: transparent; }
	    .content blockquote { margin: 1em 0; padding: 12px 16px; border-left: 4px solid #f59e0b; background: var(--rp-quote); color: var(--rp-foreground); border-radius: 12px; }
	    .content a { color: var(--rp-link); }
    .toolbar { display:${isPdf ? 'none' : 'flex'}; gap:10px; margin-top: 16px; flex-wrap: wrap; }
    .toolbar button { border: 1px solid var(--rp-border); background: var(--rp-surface); color: var(--rp-foreground); border-radius: 999px; padding: 8px 12px; cursor: pointer; }
    ${isPdf ? '.card { border-radius: 0; } .page { max-width: none; padding: 0; }' : ''}
  </style>
</head>
<body>
  <div class="page">
    <div class="card">
      <h1 class="title">${escapeHtml(title)}</h1>
      <div class="meta">Clipboard import | ${escapeHtml(createdAt)} | 本文確認用プレビュー</div>
      <div class="toolbar">
        <button onclick="window.close()">閉じる</button>
      </div>
      <div class="content">${contentHtml || '<p>内容がありません。</p>'}</div>
    </div>
  </div>
</body>
</html>`;
        };
        const openSandboxedHtmlTab = (html) => {
            const frameHtml = JSON.stringify(String(html || ''))
                .replace(/</g, '\\u003c')
                .replace(/\u2028/g, '\\u2028')
                .replace(/\u2029/g, '\\u2029');
            const shell = `<!doctype html><html><head><meta charset="utf-8"><meta name="referrer" content="no-referrer"><style>html,body,iframe{width:100%;height:100%;margin:0;border:0;background:#fff}body{overflow:hidden}</style></head><body><iframe id="preview" sandbox="allow-scripts allow-forms allow-modals allow-popups" referrerpolicy="no-referrer"></iframe><script>document.getElementById('preview').srcdoc=${frameHtml};<\/script></body></html>`;
            const blob = new Blob([shell], { type: 'text/html;charset=utf-8' });
            const url = URL.createObjectURL(blob);
            const tab = window.open(url, '_blank', 'noopener,noreferrer');
            if (!tab) {
                URL.revokeObjectURL(url);
                return false;
            }
            setTimeout(() => URL.revokeObjectURL(url), 60000);
            return true;
        };
        const openRichPastePreviewTab = () => {
            const html = buildRichPastePreviewHtml('preview');
            if (!html) {
                showToast('確認する内容がありません', 'warning', true);
                return;
            }
            if (!openSandboxedHtmlTab(html)) {
                showToast('別タブの表示に失敗しました', 'error', true);
            }
        };
        const renderRichPastePdfBlob = async () => {
            const progressContainer = get('rich-paste-progress-container');
            const progressBar = get('rich-paste-progress-bar');
            const progressText = get('rich-paste-progress-text');
            const updateProgress = (pct) => {
                const safePct = Math.max(0, Math.min(100, Number(pct) || 0));
                if (progressBar) {
                    progressBar.style.width = '100%';
                    progressBar.style.transformOrigin = 'left center';
                    // Ensure smooth transition
                    if (!progressBar.style.transition || progressBar.style.transition.indexOf('transform') === -1) {
                        progressBar.style.transition = 'transform 0.45s cubic-bezier(0.22, 1, 0.36, 1)';
                    }
                    progressBar.style.transform = `scaleX(${safePct / 100})`;
                    progressBar.style.willChange = 'transform';
                }
                if (progressText) progressText.innerText = `${Math.round(safePct)}%`;
            };

            // Show progress bar immediately
            if (progressContainer) {
                progressContainer.classList.remove('hidden');
                progressContainer.style.setProperty('display', 'block', 'important');
            }

            if (progressBar) {
                // Initialize without transition to prevent initial jump
                progressBar.style.transition = 'none';
                progressBar.style.width = '100%';
                progressBar.style.transformOrigin = 'left center';
                progressBar.style.transform = 'scaleX(0)';
                // Force reflow
                void progressBar.offsetHeight;
                // Set the smooth transition
                progressBar.style.transition = 'transform 0.45s cubic-bezier(0.22, 1, 0.36, 1)';
            }

            updateProgress(0);

            // Give browser a moment to paint the progress bar
            await new Promise(res => requestAnimationFrame(() => setTimeout(res, 150)));

            const editor = getRichPasteEditor();
            if (!editor) throw new Error('PDF化する内容がありません');
            const title = inferRichPasteTitle();
            const sanitizedHtml = sanitizeRichPasteHtml(editor.innerHTML || '');
            const theme = detectRichPasteTheme(sanitizedHtml);
            const contentHtml = normalizeRichPastePrintHtml(sanitizedHtml);
            await ensurePdfLibraries();
            const JsPdfCtor = window.jspdf && window.jspdf.jsPDF ? window.jspdf.jsPDF : null;
            if (!JsPdfCtor) throw new Error('jsPDF ライブラリが読み込まれていません');
            const html2canvasFn = window.html2canvas;
            if (typeof html2canvasFn !== 'function') throw new Error('html2canvas ライブラリが読み込まれていません');

            updateProgress(5);

	            const wrapper = document.createElement('div');
	            wrapper.style.position = 'absolute';
	            wrapper.style.left = '-10000px';
	            wrapper.style.top = '0';
	            wrapper.style.width = '794px';
            wrapper.style.background = theme.background;
            wrapper.style.color = theme.foreground;
            wrapper.style.boxSizing = 'border-box';
            wrapper.style.fontFamily = '"Noto Sans JP", "Segoe UI", "Helvetica Neue", Arial, sans-serif';
            wrapper.innerHTML = `
                <style>
                        :root {
                            color-scheme: ${theme.mode};
                            --rp-background: ${theme.background};
                            --rp-foreground: ${theme.foreground};
                            --rp-muted: ${theme.muted};
                            --rp-border: ${theme.border};
                            --rp-surface: ${theme.surface};
                            --rp-quote: ${theme.quote};
                            --rp-link: ${theme.link};
                        }
	                    .pdf-root-wrapper {
	                        background-color: var(--rp-background);
	                        color: var(--rp-foreground);
	                        padding: 40px;
	                        width: 794px;
	                        min-height: 1123px;
	                        box-sizing: border-box;
	                        color-scheme: ${theme.mode};
	                        line-height: 1.6;
	                        font-size: 15px;
	                    }
	                    .pdf-root-wrapper * {
	                        box-sizing: border-box;
	                    }
                    .pdf-root-wrapper h1,
                    .pdf-root-wrapper h2,
                    .pdf-root-wrapper h3,
                    .pdf-root-wrapper h4,
                    .pdf-root-wrapper h5,
                    .pdf-root-wrapper h6 {
	                        line-height: 1.25;
	                        margin: 1.1em 0 0.55em 0;
	                    }
	                    .pdf-title {
	                        font-size: 26px;
	                        font-weight: bold;
	                        margin: 0 0 15px 0;
	                        border-bottom: 2px solid var(--rp-border);
	                        padding-bottom: 10px;
	                        line-height: 1.2;
                            color: var(--rp-foreground);
	                    }
	                    .pdf-meta {
	                        font-size: 12px;
	                        color: var(--rp-muted);
	                        margin-bottom: 30px;
	                    }
	                    .pdf-content {
	                        font-size: 15px;
	                        line-height: 1.6;
	                        color: inherit;
	                        overflow-wrap: anywhere;
	                    }
	                    .pdf-content p { margin: 0 0 1em 0; }
	                    .pdf-content img { max-width: 100%; height: auto; }
	                    .pdf-content video,
	                    .pdf-content iframe {
	                        max-width: 100%;
	                    }
	                    .pdf-content table { max-width: 100%; border-collapse: collapse; margin: 20px 0; border: 1px solid var(--rp-border); }
	                    .pdf-content th, .pdf-content td { border: 1px solid var(--rp-border); padding: 10px; text-align: left; word-break: break-word; vertical-align: top; }
	                    .pdf-content th { background-color: var(--rp-surface); color: var(--rp-foreground); font-weight: bold; }
	                    .pdf-content pre {
	                        background-color: var(--rp-surface);
	                        color: var(--rp-foreground);
	                        border: 1px solid var(--rp-border);
	                        padding: 15px;
	                        border-radius: 5px;
	                        white-space: pre-wrap;
	                        word-break: break-word;
	                        font-family: "Noto Sans Mono", monospace;
	                        font-size: 13px;
	                        margin: 1.2em 0;
	                        line-height: 1.4;
	                        display: block;
	                        width: 100%;
	                        overflow-wrap: anywhere;
	                    }
	                    .pdf-content code {
	                        font-family: "Noto Sans Mono", monospace;
	                        background-color: var(--rp-surface);
	                        color: var(--rp-foreground);
	                        padding: 1px 4px;
	                        border-radius: 3px;
	                        font-size: 0.9em;
	                    }
	                    .pdf-content pre code {
	                        display: block;
	                        padding: 0;
	                        margin: 0;
	                        border-radius: 0;
	                        background: transparent;
	                        color: inherit;
	                        font-size: inherit;
	                        line-height: inherit;
	                        white-space: pre-wrap;
	                    }
	                    .pdf-content pre code * {
	                        background: transparent;
	                        color: inherit;
	                    }
	                    .pdf-content blockquote {
	                        border-left: 5px solid #f59e0b;
                            background: var(--rp-quote);
                            color: var(--rp-foreground);
	                        padding: 5px 0 5px 20px;
	                        margin: 1em 0;
	                        font-style: italic;
	                    }
	                    .pdf-content a {
	                        color: var(--rp-link);
	                        text-decoration: underline;
	                    }
	                    .pdf-content ul,
	                    .pdf-content ol {
	                        margin: 0 0 1em 0;
	                        padding-left: 1.5em;
	                    }
	                    .pdf-content li { margin-bottom: 0.4em; }
                </style>
                <div class="pdf-root-wrapper">
                    <div class="pdf-title">${escapeHtml(title)}</div>
                    <div class="pdf-meta">Created at: ${new Date().toLocaleString('ja-JP')}</div>
                    <div class="pdf-content">${contentHtml}</div>
                </div>
            `;
            document.body.appendChild(wrapper);
            await waitForRichPasteMedia(wrapper, 4000);
            updateProgress(15);

            try {
                const pdf = new JsPdfCtor({ unit: 'mm', format: 'a4', orientation: 'portrait', compress: true });
                const pdfWidth = pdf.internal.pageSize.getWidth();
	                const pageHeightMm = pdf.internal.pageSize.getHeight();
	                const captureWidthPx = 794;
	                const pageHeightPx = Math.floor((pageHeightMm / pdfWidth) * captureWidthPx);

	                // Get total height of the wrapper
	                const totalHeight = wrapper.scrollHeight || wrapper.offsetHeight;
	                let currentY = 0;
	                let isFirstPage = true;

	                const totalChunks = Math.ceil(totalHeight / pageHeightPx);
	                let chunkIndex = 0;

                while (currentY < totalHeight) {
                    if (richPasteAbortController && richPasteAbortController.signal.aborted) {
                        throw new DOMException('Aborted', 'AbortError');
                    }
	                    const h = Math.min(pageHeightPx, totalHeight - currentY);
	                    const canvas = await new Promise((resolve, reject) => {
                        const timer = setTimeout(() => reject(new Error('PDF chunk rendering timed out')), 120000);
                        html2canvasFn(wrapper, {
	                            scale: 1, // Use scale 1 for large documents to avoid canvas limits
	                            useCORS: true,
	                            allowTaint: false,
	                            backgroundColor: theme.background,
	                            logging: false,
	                            imageTimeout: 5000,
	                            x: 0,
	                            y: currentY,
	                            width: captureWidthPx,
	                            height: h,
	                            windowWidth: captureWidthPx,
	                            scrollX: 0,
	                            scrollY: 0,
	                            signal: richPasteAbortController ? richPasteAbortController.signal : undefined,
	                            onclone: (clonedDoc) => {
	                                prepareRichPastePdfClone(clonedDoc, theme);
	                                const root = clonedDoc.querySelector('.pdf-root-wrapper');
                                if (root) {
                                    root.style.position = 'relative';
                                    root.style.left = '0';
                                    root.style.top = '0';
                                }
                            }
                        }).then(c => { clearTimeout(timer); resolve(c); }).catch(e => { clearTimeout(timer); reject(e); });
                    });

	                    const imgData = canvas.toDataURL('image/jpeg', 0.95);
	                    const imgProps = pdf.getImageProperties(imgData);
	                    const chunkHeightMm = Math.min(pageHeightMm, (imgProps.height * pdfWidth) / imgProps.width);

                    if (!isFirstPage) {
                        pdf.addPage();
                    }

                    pdf.addImage(imgData, 'JPEG', 0, 0, pdfWidth, chunkHeightMm);
                    isFirstPage = false;
	                    currentY += h;
                    chunkIndex++;

                    // Update progress after each chunk
                    const progress = Math.min(100, 15 + Math.round((chunkIndex / totalChunks) * 85));
                    updateProgress(progress);

                    // Small delay to prevent UI freezing and allow progress update
                    await new Promise(res => setTimeout(res, 100));
                }

                updateProgress(100);

                const blob = pdf.output('blob');
                return { blob, fileName: buildRichPastePdfFilename() };
            } finally {
                if (progressContainer) {
                    progressContainer.classList.add('hidden');
                    progressContainer.style.display = 'none';
                }
                if (wrapper && wrapper.parentNode) document.body.removeChild(wrapper);
            }
        };
        const createRichPastePdfBlob = async () => {
            return await renderRichPastePdfBlob();
        };
        const buildRichPasteServerPayload = () => {
            const editor = getRichPasteEditor();
            if (!editor) throw new Error('PDF化する内容がありません');
            const rawHtml = String(editor.innerHTML || '').trim();
            const rawText = String(editor.textContent || '').trim();
            const contentHtml = rawHtml || (rawText ? `<p>${escapeHtml(rawText).replace(/\n/g, '<br/>')}</p>` : '');
            return {
                title: inferRichPasteTitle(),
                html: contentHtml,
                created_at: new Date().toLocaleString('ja-JP'),
                theme: detectRichPasteTheme(sanitizeRichPasteHtml(contentHtml))
            };
        };
        const attachRichPastePdfAndSend = async (pdfBlob, fileName, promptText, previousPrompt) => {
            const beforePaths = new Set(collectAttachmentItemsForSend().map((it) => it.path));
            const pdfFile = new File([pdfBlob], fileName, { type: 'application/pdf', lastModified: Date.now() });
            const promptInput = get('prompt-input');
            if (promptInput) promptInput.value = promptText;
            await handleFiles([pdfFile], { openModal: false });
            const afterPaths = collectAttachmentItemsForSend().map((it) => it.path);
            const attached = afterPaths.some((path) => !beforePaths.has(path));
            if (!attached) {
                if (promptInput) promptInput.value = previousPrompt;
                throw new Error('PDFの添付に失敗しました');
            }
            const sendPromise = sendMessage();
            clearRichPasteEditor(true);
            window.closeRichPasteModal();
            showToast('PDFを添付して送信を開始しました', 'success');
            if (sendPromise && typeof sendPromise.catch === 'function') {
                sendPromise.catch(() => {});
            }
        };
        const openRichPasteModal = async () => {
            await ensureUserSettingsSnapshot();
            showModal('rich-paste-modal');
            if (location.pathname !== '/paste') {
                history.pushState({ modal: 'paste' }, '', '/paste');
            }
            const prompt = getRichPastePrompt();
            if (prompt) {
                richPastePromptPreferenceSyncing = true;
                prompt.value = getRichPasteEffectivePrompt(userSettingsSnapshot);
                richPastePromptPreferenceSyncing = false;
            }
            updateRichPasteStatus();
            setTimeout(() => focusRichPasteEditor(), 80);
        };
        window.closeRichPasteModal = (skipHistory = false) => {
            hideModal('rich-paste-modal');
            if (!skipHistory && location.pathname === '/paste') {
                history.back();
            }
        };
        const sendRichPasteToModel = async (options = {}) => {
            const serverSide = !!(options && options.serverSide);
            if (abortController || richPasteAbortController) {
                showToast('回答生成中またはPDF変換中です。完了までお待ちいただくか、停止してください。', 'warning', true);
                return;
            }
            const editor = getRichPasteEditor();
            const promptEl = getRichPastePrompt();
            const sendBtn = serverSide ? get('rich-paste-send-server-btn') : get('rich-paste-send-btn');
            const cancelBtn = get('rich-paste-cancel-btn');

            if (!editor || !editor.innerText || !editor.innerText.trim()) {
                showToast('貼り付ける内容を入力してください', 'warning', true);
                return;
            }

            richPasteAbortController = new AbortController();
            if (cancelBtn) {
                cancelBtn.onclick = () => {
                    if (richPasteAbortController) {
                        richPasteAbortController.abort();
                        showToast('PDF変換をキャンセルしました', 'info');
                    }
                };
            }

            const promptText = (promptEl && promptEl.value && promptEl.value.trim()) ? promptEl.value.trim() : RICH_PASTE_DEFAULT_PROMPT;
            const previousPrompt = get('prompt-input') ? get('prompt-input').value : '';
            if (sendBtn) sendBtn.disabled = true;
            try {
                // Clear any existing generation toast
                const stack = get('toast-stack');
                if (stack) {
                    stack.querySelectorAll('.toast').forEach(t => {
                        if (t.innerText.includes('PDFを生成しています') || t.innerText.includes('サーバー側でPDFを生成しています')) t.remove();
                    });
                }
                if (serverSide) {
                    showToast('サーバー側でPDFを生成しています...', 'info', true);
                    setRichPasteProgress(2, 'サーバー側でPDFを生成しています...');
                } else {
                    showToast('PDFを生成しています...', 'info', true);
                }
                if (serverSide) {
                    if (!RICH_PASTE_PDF_SERVER_ROUTE) {
                        throw new Error('サーバー側PDF生成のURLが見つかりません');
                    }
                    const payload = buildRichPasteServerPayload();
                    setRichPasteProgress(10, 'サーバーへ送信中...');
                    const response = await apiFetch(RICH_PASTE_PDF_SERVER_ROUTE, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload),
                        signal: richPasteAbortController.signal
                    });
                    setRichPasteProgress(60, 'PDFを受信中...');
                    if (!response.ok) {
                        let message = '';
                        try {
                            const data = await response.json();
                            message = data && (data.message || data.error) ? String(data.message || data.error) : '';
                        } catch (e) {
                            try {
                                message = await response.text();
                            } catch (err) {
                                message = '';
                            }
                        }
                        if (message === 'missing_html') {
                            throw new Error('サーバーへ送るHTMLが空です。クリップボード内容の取り込みを先に行ってください');
                        }
                        throw new Error(message ? `サーバーPDF生成に失敗しました: ${message}` : 'サーバーPDF生成に失敗しました');
                    }
                    setRichPasteProgress(75, 'PDFを添付中...');
                    const blob = await response.blob();
                    const fileName = response.headers.get('X-Rich-Paste-Filename') || buildRichPastePdfFilename();
                    const downloadOnly = !!(get('rich-paste-download-only') && get('rich-paste-download-only').checked);

                    if (downloadOnly) {
                        setRichPasteProgress(90, 'ダウンロード中...');
                        downloadBlob(blob, fileName);
                        showToast('PDFをダウンロードしました', 'success');
                        hideModal('rich-paste-modal', { skipConfirm: true });
                    } else {
                        await attachRichPastePdfAndSend(blob, fileName, promptText, previousPrompt);
                    }
                    setRichPasteProgress(100, '完了');
                    setTimeout(() => hideRichPasteProgress(), 400);
                } else {
                    const generated = await createRichPastePdfBlob();
                    const downloadOnly = !!(get('rich-paste-download-only') && get('rich-paste-download-only').checked);
                    if (downloadOnly) {
                        downloadBlob(generated.blob, generated.fileName);
                        showToast('PDFをダウンロードしました', 'success');
                        hideModal('rich-paste-modal', { skipConfirm: true });
                    } else {
                        await attachRichPastePdfAndSend(generated.blob, generated.fileName, promptText, previousPrompt);
                    }
                }
            } catch (e) {
                if (e.name === 'AbortError') {
                    console.log('PDF generation aborted by user');
                    if (serverSide) {
                        setRichPasteProgress(0, 'キャンセルされました');
                        setTimeout(() => hideRichPasteProgress(), 800);
                    }
                    return; // Early return for manual cancel
                }
                if (get('prompt-input')) get('prompt-input').value = previousPrompt;
                const msg = (e && e.message) ? e.message : 'PDF化して送信できませんでした';
                showToast(msg, 'error', true);
                if (serverSide) {
                    setRichPasteProgress(0, '失敗しました');
                    setTimeout(() => hideRichPasteProgress(), 1200);
                }
            } finally {
                if (sendBtn) sendBtn.disabled = false;
                richPasteAbortController = null;
            }
        };
        let csrfToken = document.querySelector('meta[name="csrf-token"]').content;
        let csrfRefreshPromise = null;
        const refreshCsrfToken = async () => {
            if (csrfRefreshPromise) return csrfRefreshPromise;
            csrfRefreshPromise = (async () => {
                const response = await fetch('/api/csrf_token', {
                    method: 'GET',
                    credentials: 'include',
                    cache: 'no-store',
                    headers: {'Accept': 'application/json'}
                });
                if (!response.ok) return false;
                const data = await response.json().catch(() => ({}));
                const refreshedToken = data && typeof data.csrf_token === 'string'
                    ? data.csrf_token
                    : '';
                if (!refreshedToken) return false;
                csrfToken = refreshedToken;
                const meta = document.querySelector('meta[name="csrf-token"]');
                if (meta) meta.setAttribute('content', refreshedToken);
                return true;
            })().catch(() => false).finally(() => {
                csrfRefreshPromise = null;
            });
            return csrfRefreshPromise;
        };
        const apiFetch = async (url, opts = {}) => {
            const method = (opts.method || 'GET').toUpperCase();
            const headers = Object.assign({}, opts.headers || {});
            const requiresCsrf = !['GET', 'HEAD', 'OPTIONS'].includes(method);
            if (requiresCsrf) {
                headers['X-CSRF-Token'] = csrfToken;
            }
            const credentials = opts.credentials || 'include';
            let response = await fetch(url, Object.assign({}, opts, { headers, credentials }));
            // Apache's error override can surface an application CSRF 403 as a 404 page.
            // Refresh from the current signed session and replay an unsafe request at most once.
            if (requiresCsrf && (response.status === 403 || response.status === 404)) {
                let errBody = null;
                try { errBody = await response.clone().json(); } catch (e) {}
                // Bot / lock / ban responses are intentional application errors —
                // never treat them as CSRF failures and never retry the same
                // body (Turnstile tokens are single-use; retrying a spent token
                // used to stack verify_fail counts and ban new accounts).
                const botErr = errBody && errBody.error;
                if (botErr === 'account_locked') {
                    // Admins are never subject to temporary locks (server-side
                    // also skips them); never show the lock overlay for admins.
                    if (!isAdminUser && !document.getElementById('bot-lock-overlay')) {
                        showBotLockOverlay(errBody.message || 'アカウントが一時的にロックされています。', errBody.remaining_seconds);
                    }
                    return response;
                }
                if (botErr === 'banned' || botErr === 'turnstile_failed' || botErr === 'rate_limit') {
                    return response;
                }
                if (botErr === 'turnstile_required' && isBotDetectionActive()) {
                    // Marker expired mid-session: force a fresh verification, then retry once.
                    botDetectionVerified = false;
                    const verified = await Promise.race([
                        runBotDetectionGate(),
                        new Promise((resolve) => setTimeout(() => resolve(false), 30000))
                    ]);
                    if (verified) {
                        headers['X-CSRF-Token'] = csrfToken;
                        response = await fetch(url, Object.assign({}, opts, { headers, credentials }));
                    }
                    return response;
                }
                const refreshed = await refreshCsrfToken();
                if (refreshed) {
                    headers['X-CSRF-Token'] = csrfToken;
                    response = await fetch(url, Object.assign({}, opts, { headers, credentials }));
                }
            }
            return response;
        };
        const manualSpinnerRequestOptions = (options) => window.ProgressSpinner
            ? window.ProgressSpinner.manualRequestOptions(options)
            : options;

        window.updateGoogleLinkUI = (d) => {
            const linkText = get('google-link-text');
            const emailText = get('google-email-text');
            const actionArea = get('google-action-area');
            const icon = get('google-link-icon');
            if (!linkText || !actionArea) return;

            if (d.google_id) {
                linkText.innerText = '連携済み';
                linkText.classList.replace('text-gray-200', 'text-green-400');
                emailText.innerText = d.google_email || '連携中の Google アカウント';
                icon.classList.replace('bg-gray-800', 'bg-green-900/30');
                icon.classList.add('text-green-400');
                actionArea.innerHTML = `<button onclick="unlinkGoogleAccount()" class="px-4 py-2 bg-red-900/20 hover:bg-red-900/40 text-red-400 border border-red-800 rounded text-xs font-bold transition btn-hover">連携を解除</button>`;
            } else {
                linkText.innerText = '未連携';
                linkText.classList.replace('text-green-400', 'text-gray-200');
                emailText.innerText = 'Google アカウントでログインできるようになります。';
                icon.classList.replace('bg-green-900/30', 'bg-gray-800');
                icon.classList.remove('text-green-400');
                actionArea.innerHTML = `<a href="/login/google" class="inline-block px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded text-xs font-bold transition btn-hover">Google と連携する</a>`;
            }
        };

        window.unlinkGoogleAccount = async () => {
            if (!confirm('Google 連携を解除しますか？\n解除後は Google ログインが利用できなくなります（パスワードが設定されていない場合はログインできなくなる可能性があります）。')) return;
            try {
                const res = await apiFetch(CHAT_CONFIG.urls.unlinkGoogleAccount, {method: 'POST'});
                if (res.ok) {
                    showToast('Google 連携を解除しました');
                    // Refresh UI
                    apiFetch(CHAT_CONFIG.urls.handleSettingsQuery).then(r=>r.json()).then(d=>updateGoogleLinkUI(d));
                } else {
                    const err = await res.json();
                    showToast(err.error || '解除に失敗しました', 'error', true);
                }
            } catch (e) {
                showToast('ネットワークエラーが発生しました', 'error', true);
            }
        };

        window.updateMinashinLinkUI = (d) => {
            const linkText = get('minashin-link-text');
            const emailText = get('minashin-email-text');
            const actionArea = get('minashin-action-area');
            const icon = get('minashin-link-icon');
            if (!linkText || !actionArea) return;

            if (d.minashin_sub) {
                linkText.innerText = '連携済み';
                linkText.classList.replace('text-gray-200', 'text-green-400');
                emailText.innerText = d.minashin_email || '連携中の Minashin アカウント';
                icon.classList.replace('bg-gray-800', 'bg-green-900/30');
                actionArea.innerHTML = `<button onclick="unlinkMinashinAccount()" class="px-4 py-2 bg-red-900/20 hover:bg-red-900/40 text-red-400 border border-red-800 rounded text-xs font-bold transition btn-hover">連携を解除</button>`;
            } else {
                linkText.innerText = '未連携';
                linkText.classList.replace('text-green-400', 'text-gray-200');
                emailText.innerText = 'Minashin アカウントでログインできるようになります。';
                icon.classList.replace('bg-green-900/30', 'bg-gray-800');
                actionArea.innerHTML = `<a href="/login/minashin" class="inline-block px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded text-xs font-bold transition btn-hover">Minashin と連携する</a>`;
            }
        };

        window.unlinkMinashinAccount = async () => {
            if (!confirm('Minashin 連携を解除しますか？\n解除後は Minashin ログインが利用できなくなります（パスワードが設定されていない場合はログインできなくなる可能性があります）。')) return;
            try {
                const res = await apiFetch(CHAT_CONFIG.urls.unlinkMinashinAccount, {method: 'POST'});
                if (res.ok) {
                    showToast('Minashin 連携を解除しました');
                    // Refresh UI
                    apiFetch(CHAT_CONFIG.urls.handleSettingsQuery).then(r=>r.json()).then(d=>updateMinashinLinkUI(d));
                } else {
                    const err = await res.json();
                    showToast(err.error || '解除に失敗しました', 'error', true);
                }
            } catch (e) {
                showToast('ネットワークエラーが発生しました', 'error', true);
            }
        };
        let lastClientDebugEnabled = null;
        const isClientDebugLogEnabled = () => {
            const settingEl = get('set-client-debug-log');
            return !!(settingEl && settingEl.checked);
        };
        const sendClientDebugLog = (level, message) => {
            if (!isClientDebugLogEnabled()) return;
            const payload = {
                level: String(level || 'info'),
                message: String(message || '')
            };
            apiFetch('/api/debug/client_log', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            }).catch(() => {});
        };
        const syncClientDebugLogToggle = (enabled, source) => {
            const settingEl = get('set-client-debug-log');
            if (settingEl) {
                settingEl.checked = !!enabled;
            }
            const normalized = !!enabled;
            if (normalized && lastClientDebugEnabled !== true) {
                sendClientDebugLog('info', `Client debug logging enabled (${source}).`);
            }
            lastClientDebugEnabled = normalized;
        };
        const nowPerfMs = () => {
            if (window.performance && typeof window.performance.now === 'function') {
                return window.performance.now();
            }
            return Date.now();
        };
        const reportFirstTokenLatency = (payload) => {
            if (!enableLatencyMetrics) return;
            try {
                if (!payload || typeof payload !== 'object') return;
                const secRaw = Number(payload.latency_seconds);
                if (!Number.isFinite(secRaw) || secRaw < 0 || secRaw > 600) return;
                const msRaw = Number(payload.latency_ms);
                const body = {
                    latency_seconds: Number(secRaw.toFixed(6)),
                    latency_ms: Number.isFinite(msRaw) ? Math.max(0, Math.round(msRaw)) : Math.round(secRaw * 1000),
                    thread_id: payload.thread_id ? String(payload.thread_id) : null,
                    job_id: payload.job_id ? String(payload.job_id) : null,
                    model: payload.model ? String(payload.model) : null,
                    first_event_type: payload.first_event_type ? String(payload.first_event_type) : "content",
                    client_sent_at_ms: Number.isFinite(Number(payload.client_sent_at_ms))
                        ? Math.round(Number(payload.client_sent_at_ms))
                        : Date.now(),
                    is_total: !!payload.is_total,
                    client_done_at_ms: Number.isFinite(Number(payload.client_done_at_ms))
                        ? Math.round(Number(payload.client_done_at_ms))
                        : null
                };
                apiFetch('/api/metrics/first_token', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(body)
                }).catch(() => {});
            } catch (_e) {}
        };
        let currentThreadId = CHAT_CONFIG.initialThreadId;
        if (currentThreadId !== null && currentThreadId !== undefined) {
            currentThreadId = String(currentThreadId);
        }
        const ATTACHMENT_MAX_FILES = Number(CHAT_CONFIG.attachmentMaxFiles) || 30;
        const UPLOAD_CONCURRENCY = Math.max(1, Number(CHAT_CONFIG.uploadConcurrency) || 3);
        const TEMP_CHAT_TIMEOUT_MIN_SECONDS = 10;
        const TEMP_CHAT_TIMEOUT_MAX_SECONDS = 3600;
        const TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS = 90;
        const TEMP_CHAT_HEARTBEAT_MIN_MS = 4000;
        const TEMP_CHAT_HEARTBEAT_MAX_MS = 15000;
        let activeGem = null, editingGemUuid = null, currentImageUrls = [], currentMaskImage = null, abortController = null, richPasteAbortController = null, userAutoScroll = true, searchTimeout;

        // Prompt History
        let promptHistory = [];
        let historyIndex = -1;
        let tempPrompt = "";

        const markerAppliedUploads = new Set();
        const attachmentSourceByPath = new Map();
        const attachmentNameByPath = new Map();
        const BROWSER_FAST_IGNORE_WARNING_STORAGE = 'browser_fast_mode_ignore_warning';
        const BROWSER_FAST_MAX_IMAGES = 4;
        const BROWSER_FAST_MAX_BYTES = 12 * 1024 * 1024;
        const browserFastLocalFiles = new Map();
        let browserFastModeEnabled = false;
        let browserFastApiKey = '';
        let browserFastApiKeyModel = '';
        let browserFastBootstrap = null;
        let browserFastPreviousOptions = null;
        let cameraCaptureStream = null;
        let cameraCaptureFacingMode = 'environment';
        let cameraCaptureBusy = false;
        let cameraCaptureSequence = 0;
        const cameraCapturePendingFiles = [];
        const cameraCapturePendingPreviewUrls = [];
        let modalThreadId = null;
        const MARKER_HINT_TEXT = "編集済みの画像を見てください。";
        const MARKER_OPACITY_MIN_PCT = 0.1;
        const MARKER_OPACITY_MAX_PCT = 100;
        const MARKER_OPACITY_MIN_ALPHA = MARKER_OPACITY_MIN_PCT / 100;
        const markerState = {
            row: null,
            filename: '',
            hasStroke: false,
            naturalWidth: 0,
            naturalHeight: 0,
            colorHex: '#facc15',
            opacity: 0.6,
            history: [],
            mode: 'draw',
            cropRect: null,
            mosaicRects: [],
            mosaicPreviewRect: null,
            baseCanvas: null,
            baseImageData: null
        };
        const markerView = { scale: 1, offsetX: 0, offsetY: 0, minScale: 1, maxScale: 4 };
        const threadGemMap = {};
        let pendingGemForNewThread = null;
        let loadedGems = [];
        let currentJobId = null;
        let currentThreadPending = null;
        let currentVisionModel = null;
        let activeStreamingBubbleId = null;
        let manualStopContext = null;
        let manualStopSeq = 0;
        let isStopMode = false;
        const suppressedPendingJobIds = new Set();
        const pendingStreamReconnectJobs = new Set();
        let editingMessageId = null; // Track message being edited
        const messageStore = {}, lib = { modal: get('lib-modal'), grid: get('lib-grid'), files: [], selected: new Set(), attachMode: false, searchQuery: '' };
        const LIB_SORT_KEY = 'lib_sort_order';
        let threadPage = 1, threadLoading = false, hasMoreThreads = true;
        let threadObserver = null;
        let currentQuote = "";
        let currentThreadTitle = null;
        let temporaryChatEnabled = false;
        let temporaryChatTimeoutSeconds = TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS;
        let tempChatExpiresAtMs = null;
        let tempChatHeartbeatTimer = null;
        let tempChatHeartbeatIntervalMs = 0;
        let tempChatHeartbeatInFlight = false;
        let tempChatHeaderTicker = null;
        let enterToSend = CHAT_CONFIG.enterToSend;
        let autoSearchOnLinks = CHAT_CONFIG.autoSearchOnLinks;
        let useSwCache = CHAT_CONFIG.useSwCache;
        let compactPromptMode = CHAT_CONFIG.compactPromptMode;
        let minimalPromptMode = !!CHAT_CONFIG.minimalPromptMode;
        let voiceStudioUiEnabled = true;
        const CANVAS_MODE_STORAGE_KEY = 'canvas_mode_enabled_v1';
        const CODING_MODE_STORAGE_KEY = 'coding_mode_enabled_v1';
        let canvasModeEnabled = false;
        let codingModeEnabled = false;
        // UI preference and model-facing behavior are intentionally separate.
        // This becomes true only after a complete code block exists.
        let codingModeEffective = false;
        let codingTargetSelection = null;
        const canvasPreviewState = {
            blocks: [],
            rawText: '',
            renderText: '',
            selectedIndex: -1,
            selectedKey: '',
            selectionMode: 'auto',
            mobileView: 'preview',
            sourceScrollTop: 0,
            sourceScrollLeft: 0,
            frameScrollX: 0,
            frameScrollY: 0,
            frameRenderToken: 0,
            panelAnimationToken: 0,
            panelHideTimer: null,
            viewAnimationToken: 0,
            viewAnimationTimer: null,
            lastCanvasData: null
        };
        try {
            canvasModeEnabled = localStorage.getItem(CANVAS_MODE_STORAGE_KEY) === 'true';
        } catch (e) {
            canvasModeEnabled = false;
        }
        try {
            codingModeEnabled = localStorage.getItem(CODING_MODE_STORAGE_KEY) === 'true';
        } catch (e) {
            codingModeEnabled = false;
        }
        let enableLatencyMetrics = CHAT_CONFIG.enableLatencyMetrics;
        let promptControlsExpanded = false;
        const appVersion = CHAT_CONFIG.appVersion;
        const botConfig = CHAT_CONFIG.botConfig;
        const isAdminUser = botConfig && botConfig.isAdmin;
        const currentUsername = CHAT_CONFIG.currentUsername;
        let turnstileWidgetId = null;
        let turnstileToken = null;
        let turnstilePending = false;
        let botDetectionVerified = false;
        let botDetectionGatePromise = null;
        let botDetectionOverlayShown = false;
        let botDetectionDialogWidgetId = null;
        let sendButtonSpamTimestamps = [];
        let chatDefaultsLoaded = false;
        let modelApiKeyMap = {};
        const THREAD_INITIAL_MESSAGE_LIMIT = 50;
        const THREAD_OLDER_PAGE_SIZE = 50;
        const LOW_BANDWIDTH_INITIAL_MESSAGE_LIMIT = 40;
        const LOW_BANDWIDTH_OLDER_PAGE_SIZE = 60;
        const LOW_BANDWIDTH_MODE_STORAGE_KEY = 'low_bandwidth_mode_pref_v1';
        const LOW_BANDWIDTH_DECORATION_VISIBILITY_THRESHOLD = 0.02;
        const MATHJAX_SRC = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js';
        const HLJS_JS_SRC = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js';
        const HLJS_CSS_SRC = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css';
        let mathJaxLoadPromise = null;
        let incrementalMathTypesetChain = Promise.resolve();
        let highlightLoadPromise = null;
        let lowBandwidthModePreference = 'auto';
        let lowBandwidthModeAuto = false;
        let lowBandwidthMode = false;
        let lowBandwidthModeReason = '';
        let lowBandwidthConnectionListenerAttached = false;
        let deferredDecorationObserver = null;
        const deferredDecorationTextMap = new WeakMap();
        let threadHasOlderMessages = false;
        let oldestLoadedMessageId = null;
        let loadingOlderMessages = false;
        let threadLoadSequence = 0;

        // Tree State
        let allMessages = [];
        let currentLeafId = null;
        let currentParentId = null;

        function loadScriptOnce(src, id) {
            const existingById = id ? document.getElementById(id) : null;
            if (existingById) {
                if (existingById.dataset.loaded === '1') return Promise.resolve(existingById);
                return new Promise((resolve, reject) => {
                    existingById.addEventListener('load', () => resolve(existingById), { once: true });
                    existingById.addEventListener('error', reject, { once: true });
                });
            }
            return new Promise((resolve, reject) => {
                const s = document.createElement('script');
                if (id) s.id = id;
                s.src = src;
                s.async = true;
                s.onload = () => {
                    s.dataset.loaded = '1';
                    resolve(s);
                };
                s.onerror = reject;
                document.head.appendChild(s);
            });
        }
        function loadStylesheetOnce(href, id) {
            const existingById = id ? document.getElementById(id) : null;
            if (existingById) return Promise.resolve(existingById);
            const existing = Array.from(document.querySelectorAll('link[rel="stylesheet"]')).find((el) => el.href === href);
            if (existing) return Promise.resolve(existing);
            return new Promise((resolve, reject) => {
                const link = document.createElement('link');
                if (id) link.id = id;
                link.rel = 'stylesheet';
                link.href = href;
                link.onload = () => resolve(link);
                link.onerror = reject;
                document.head.appendChild(link);
            });
        }
        async function ensureMathJaxLoaded() {
            if (window.MathJax && typeof window.MathJax.typesetPromise === 'function') return window.MathJax;
            if (!mathJaxLoadPromise) {
                window.MathJax = window.MathJax || {
                    tex: {
                        inlineMath: [['\\(', '\\)'], ['$', '$']],
                        displayMath: [['$$', '$$'], ['\\[', '\\]']],
                        processEscapes: true
                    },
                    options: {
                        // skipHtmlTags の pre/code を再処理する用途のみ。全要素は通常どおり走査する
                        ignoreHtmlClass: 'tex2jax_ignore|mathjax_ignore',
                        processHtmlClass: 'tex2jax_process|mathjax_process'
                    },
                    startup: {
                        typeset: false
                    }
                };
                mathJaxLoadPromise = loadScriptOnce(MATHJAX_SRC, 'MathJax-script').catch((err) => {
                    mathJaxLoadPromise = null;
                    throw err;
                });
            }
            await mathJaxLoadPromise;
            return window.MathJax || null;
        }
        async function ensureHighlightLoaded() {
            if (window.hljs) return window.hljs;
            if (!highlightLoadPromise) {
                highlightLoadPromise = Promise.all([
                    loadStylesheetOnce(HLJS_CSS_SRC, 'hljs-theme-chat'),
                    loadScriptOnce(HLJS_JS_SRC, 'hljs-script')
                ]).then(() => window.hljs || null).catch((err) => {
                    highlightLoadPromise = null;
                    throw err;
                });
            }
            return await highlightLoadPromise;
        }
        function maybeNeedsMathJax(text) {
            const t = String(text || '');
            if (t.includes('$$') || t.includes('\\(') || t.includes('\\[') || t.includes('\\begin{')) return true;
            // 単独 $...$ （通貨 $12 は除外し、数式らしい中身のみ）
            return /(?<!\$)\$(?!\$)(?=[\s\S]*?[A-Za-z\\^_{}])(?:[^$\n\\]|\\.)+?\$(?!\$)/.test(t);
        }
        /**
         * marked は \( \[ などのバックスラッシュを Markdown エスケープとして除去するため、
         * 数式セグメントをプレースホルダへ退避してから parse する。
         */
        function protectMathSegments(src) {
            const source = String(src || '');
            const blocks = [];
            const stash = (match) => {
                const key = `@@MATHJAX_BLOCK_${blocks.length}@@`;
                blocks.push(match);
                return key;
            };
            // コードフェンス内は触らない（表示用の LaTeX ソースを壊さない）
            const parts = [];
            const fenceRe = /(^|\n)([ \t]*)(`{3,}|~{3,})[^\n]*\n[\s\S]*?(?:\n\2\3[ \t]*(?:\n|$)|$)/g;
            let last = 0;
            let m;
            while ((m = fenceRe.exec(source)) !== null) {
                const start = m.index;
                if (start > last) {
                    parts.push({ type: 'text', value: source.slice(last, start) });
                }
                parts.push({ type: 'code', value: m[0] });
                last = start + m[0].length;
            }
            if (last < source.length) {
                parts.push({ type: 'text', value: source.slice(last) });
            }
            if (!parts.length) {
                parts.push({ type: 'text', value: source });
            }
            const protectedText = parts.map((part) => {
                if (part.type === 'code') return part.value;
                let t = part.value;
                // 長い／優先度の高いデリミタから順に退避
                t = t.replace(/\$\$([\s\S]+?)\$\$/g, stash);
                t = t.replace(/\\\(([\s\S]+?)\\\)/g, stash);
                t = t.replace(/\\\[([\s\S]+?)\\\]/g, stash);
                t = t.replace(/\\begin\{([a-zA-Z*]+)\}([\s\S]+?)\\end\{\1\}/g, stash);
                // 単独 $...$ （空や空白のみ、および $$ は除外）
                t = t.replace(/(?<!\$)\$(?!\$)([^\s$](?:(?:[^$\n\\]|\\.)*?[^\s$])?)\$(?!\$)/g, stash);
                return t;
            }).join('');
            return { text: protectedText, blocks };
        }
