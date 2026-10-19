        // Image splitting uses the Android grid rules. Cropping itself is lossless;
        // the shared compression pipeline is applied once, with its own saved settings.
        const SPLIT_COMPRESSION_KEY = 'image_split_compression';
        const splitState = { files: [], index: 0, image: null, release: null, revision: 0, busy: false, urls: [] };
        function splitClamp(value, min, max, fallback) {
            const n = Number.parseInt(value, 10);
            return Number.isFinite(n) ? Math.max(min, Math.min(max, n)) : fallback;
        }
        function chooseSplitGrid(pieces, width, height) {
            const count = splitClamp(pieces, 2, 30, 4);
            let best = { columns: count, rows: 1 }, bestScore = Infinity;
            for (let columns = 1; columns <= count; columns++) {
                if (count % columns) continue;
                const rows = count / columns;
                const score = Math.abs(Math.log((Math.max(1, width) / columns) / (Math.max(1, height) / rows)));
                if (score < bestScore - 1e-9 || (Math.abs(score - bestScore) <= 1e-9 &&
                    (width >= height ? columns > best.columns : rows > best.rows))) {
                    best = { columns, rows }; bestScore = score;
                }
            }
            return best;
        }
        function splitTiles(width, height, grid, overlapPct) {
            const overlap = splitClamp(overlapPct, 0, 50, 10) / 100;
            const padX = Math.round(width / grid.columns * overlap / 2);
            const padY = Math.round(height / grid.rows * overlap / 2);
            const tiles = [];
            for (let row = 0; row < grid.rows; row++) for (let column = 0; column < grid.columns; column++) {
                const cellLeft = Math.floor(column * width / grid.columns), cellRight = Math.floor((column + 1) * width / grid.columns);
                const cellTop = Math.floor(row * height / grid.rows), cellBottom = Math.floor((row + 1) * height / grid.rows);
                tiles.push({ index: tiles.length, row, column, cellLeft, cellTop, cellRight, cellBottom,
                    left: Math.max(0, cellLeft - padX), top: Math.max(0, cellTop - padY),
                    right: Math.min(width, cellRight + padX), bottom: Math.min(height, cellBottom + padY) });
            }
            return tiles;
        }
        function splitFits(grid, width, height) {
            return Math.floor(width / grid.columns) >= 16 && Math.floor(height / grid.rows) >= 16;
        }
        function splitTileCaption(tile, grid, width, height) {
            return `Tile ${tile.index + 1}/${grid.columns * grid.rows}  row ${tile.row + 1}/${grid.rows}, col ${tile.column + 1}/${grid.columns}  x ${tile.left}-${tile.right}, y ${tile.top}-${tile.bottom} of ${width}x${height}`;
        }
        function splitBaseName(name) {
            return (String(name).split('/').pop().replace(/\.[^.]*$/, '').replace(/[\\/:*?"<>|\s]+/g, '_')
                .replace(/^[_.]+|[_.]+$/g, '').slice(0, 80)) || 'image';
        }
        function splitTileName(base, tile, grid) {
            const count = grid.columns * grid.rows;
            return `${base}_part${String(tile.index + 1).padStart(String(count).length, '0')}of${count}_r${tile.row + 1}c${tile.column + 1}.png`;
        }
        function readSplitCompression() {
            let saved;
            try { saved = JSON.parse(localStorage.getItem(SPLIT_COMPRESSION_KEY) || '{}'); } catch (e) {}
            saved = saved && typeof saved === 'object' ? saved : {};
            return { enabled: saved.enabled !== false, maxSizeMB: Math.max(0.1, Math.min(10, Number(saved.maxSizeMB) || 1)),
                maxDimension: splitClamp(saved.maxDimension, 256, 8192, 1920),
                outputType: ['original', 'image/jpeg', 'image/png', 'image/webp'].includes(saved.outputType) ? saved.outputType : 'original',
                formatOnly: saved.formatOnly === true };
        }
        function splitOptions() {
            return { pieces: splitClamp(get('image-split-pieces').value, 2, 30, 4),
                overlap: splitClamp(get('image-split-overlap').value, 0, 50, 10),
                includeOriginal: get('image-split-original').checked,
                markOriginal: get('image-split-mark-original').checked,
                markPieces: get('image-split-mark-pieces').checked };
        }
        function splitCanvas(width, height) {
            const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
            if (canvas.width !== width || canvas.height !== height || !canvas.getContext('2d')) throw new Error('画像が大きすぎるため処理できません。');
            return canvas;
        }
        async function loadSplitImage(file) {
            // HTMLImageElement applies EXIF orientation. Never request a reduced output size.
            const url = URL.createObjectURL(file), img = new Image();
            try {
                await new Promise((resolve, reject) => { img.onload = resolve; img.onerror = () => reject(new Error(`画像を読み込めません: ${file.name}`)); img.src = url; });
                return { image: img, release: () => { img.src = ''; URL.revokeObjectURL(url); } };
            } catch (e) { URL.revokeObjectURL(url); throw e; }
        }
        function drawSplitOverview(canvas, image, width, height, grid, tiles) {
            const ctx = canvas.getContext('2d'), sx = canvas.width / width, sy = canvas.height / height;
            ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
            const first = tiles[0], padX = first.right - first.cellRight, padY = first.bottom - first.cellBottom;
            ctx.fillStyle = 'rgba(250,204,21,0.18)';
            const lines = [];
            for (let col = 1; col < grid.columns; col++) {
                const x = Math.floor(col * width / grid.columns);
                ctx.fillRect((x - padX) * sx, 0, 2 * padX * sx, canvas.height); lines.push([x * sx, 0, x * sx, canvas.height]);
            }
            for (let row = 1; row < grid.rows; row++) {
                const y = Math.floor(row * height / grid.rows);
                ctx.fillRect(0, (y - padY) * sy, canvas.width, 2 * padY * sy); lines.push([0, y * sy, canvas.width, y * sy]);
            }
            drawSplitLines(ctx, lines, Math.max(2, Math.min(5, Math.max(canvas.width, canvas.height) / 400)), false);
            const size = Math.max(14, Math.min(64, Math.min(canvas.width / grid.columns, canvas.height / grid.rows) * 0.18));
            ctx.font = `bold ${size}px monospace`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            for (const tile of tiles) {
                const x = (tile.cellLeft + tile.cellRight) / 2 * sx, y = (tile.cellTop + tile.cellBottom) / 2 * sy;
                ctx.lineWidth = Math.max(2, size / 10); ctx.strokeStyle = 'rgba(0,0,0,0.7)'; ctx.strokeText(String(tile.index + 1), x, y);
                ctx.fillStyle = '#facc15'; ctx.fillText(String(tile.index + 1), x, y);
            }
        }
        function drawSplitLines(ctx, lines, stroke, dashed) {
            ctx.save(); ctx.lineWidth = stroke; ctx.strokeStyle = 'rgba(250,204,21,0.6)';
            ctx.shadowColor = 'rgba(0,0,0,0.7)'; ctx.shadowBlur = 2;
            if (dashed) ctx.setLineDash([stroke * 4, stroke * 3]);
            ctx.beginPath(); lines.forEach(([x1,y1,x2,y2]) => { ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); }); ctx.stroke(); ctx.restore();
        }
        async function splitCanvasFile(canvas, name) {
            try {
                const blob = await new Promise((resolve, reject) => canvas.toBlob(b => b ? resolve(b) : reject(new Error('画像を書き出せません。')), 'image/png'));
                return new File([blob], name, { type: 'image/png' });
            } finally { canvas.width = 0; canvas.height = 0; }
        }
        async function renderSplitFiles(file, options, compression) {
            const loaded = await loadSplitImage(file), image = loaded.image;
            try {
                const width = image.naturalWidth, height = image.naturalHeight;
                const grid = chooseSplitGrid(options.pieces, width, height);
                if (!splitFits(grid, width, height)) throw new Error(`画像が小さすぎるため${options.pieces}枚に分割できません: ${file.name}`);
                const tiles = splitTiles(width, height, grid, options.overlap), base = splitBaseName(file.name), outputs = [];
                const append = async output => { outputs.push(await applyImageCompression(output, compression)); };
                if (options.includeOriginal) {
                    if (options.markOriginal) {
                        const canvas = splitCanvas(width, height); drawSplitOverview(canvas, image, width, height, grid, tiles);
                        await append(await splitCanvasFile(canvas, `${base}_split_grid.png`));
                    } else await append(file);
                }
                for (const tile of tiles) {
                    const w = tile.right - tile.left, h = tile.bottom - tile.top;
                    let fontSize = Math.max(12, Math.min(40, Math.max(w, h) / 45)), header = 0;
                    const caption = splitTileCaption(tile, grid, width, height);
                    if (options.markPieces) {
                        const measure = splitCanvas(1,1), ctx = measure.getContext('2d'); ctx.font = `bold ${fontSize}px monospace`;
                        fontSize = Math.max(6, fontSize * Math.min(1, w * 0.96 / ctx.measureText(caption).width));
                        header = Math.max(12, Math.floor(fontSize * 1.8)); measure.width = 0;
                    }
                    const canvas = splitCanvas(w, h + header), ctx = canvas.getContext('2d');
                    if (header) { ctx.fillStyle = '#111827'; ctx.fillRect(0,0,w,header); }
                    ctx.drawImage(image, tile.left, tile.top, w, h, 0, header, w, h);
                    if (options.markPieces) {
                        ctx.fillStyle = '#fff'; ctx.font = `bold ${fontSize}px monospace`; ctx.textBaseline = 'middle'; ctx.fillText(caption, w * 0.02, header / 2);
                        const lines = [];
                        for (const x of [tile.cellLeft, tile.cellRight]) if (x > tile.left && x < tile.right) lines.push([x-tile.left,header,x-tile.left,header+h]);
                        for (const y of [tile.cellTop, tile.cellBottom]) if (y > tile.top && y < tile.bottom) lines.push([0,header+y-tile.top,w,header+y-tile.top]);
                        drawSplitLines(ctx, lines, Math.max(1.5, Math.min(4, Math.max(w,h)/500)), true);
                    }
                    await append(await splitCanvasFile(canvas, splitTileName(base, tile, grid)));
                }
                return outputs;
            } finally { loaded.release(); }
        }
        function clearSplitDownloads() {
            splitState.urls.forEach(url => URL.revokeObjectURL(url)); splitState.urls = [];
            get('image-split-downloads').replaceChildren();
        }
        function closeImageSplit() {
            if (!get('image-split-modal') || splitState.busy) return;
            splitState.revision++; if (splitState.release) splitState.release();
            splitState.release = null; splitState.image = null; splitState.files = [];
            clearSplitDownloads(); get('image-split-preview').width = 0; hideModal('image-split-compression-modal'); hideModal('image-split-modal');
        }
        async function openImageSplit(files) {
            const images = Array.from(files).filter(Boolean);
            if (!images.length) return;
            if (images.length > 10) showToast('画像の分割は一度に10枚までです。先頭10枚のみ使います。', 'warning', true);
            if (splitState.busy) return;
            closeImageSplit(); splitState.files = images.slice(0,10); splitState.index = 0;
            showModal('image-split-modal'); await loadSplitPreview();
        }
        async function loadSplitPreview() {
            const revision = ++splitState.revision;
            if (splitState.release) splitState.release(); splitState.release = null; splitState.image = null;
            get('image-split-status').textContent = '読み込み中...';
            try {
                const loaded = await loadSplitImage(splitState.files[splitState.index]);
                if (revision !== splitState.revision) { loaded.release(); return; }
                splitState.image = loaded.image; splitState.release = loaded.release;
                get('image-split-status').textContent = ''; updateSplitPreview();
            } catch (e) { if (revision === splitState.revision) { get('image-split-status').textContent = e.message; updateSplitPreview(); } }
        }
        function updateSplitPreview() {
            const options = splitOptions(), image = splitState.image, canvas = get('image-split-preview');
            get('image-split-mark-original').disabled = !options.includeOriginal || splitState.busy;
            get('image-split-page').textContent = `${splitState.index + 1} / ${splitState.files.length}（すべて同じ設定で分割します）`;
            get('image-split-prev').disabled = splitState.busy || splitState.index === 0;
            get('image-split-next').disabled = splitState.busy || splitState.index >= splitState.files.length - 1;
            get('image-split-count').textContent = `作成される画像: ${(options.pieces + Number(options.includeOriginal)) * splitState.files.length}枚`;
            let fits = false;
            if (image) {
                const w = image.naturalWidth, h = image.naturalHeight, grid = chooseSplitGrid(options.pieces,w,h); fits = splitFits(grid,w,h);
                const scale = Math.min(1, 800/w, 520/h); canvas.width = Math.round(w*scale); canvas.height = Math.round(h*scale);
                if (fits) drawSplitOverview(canvas,image,w,h,grid,splitTiles(w,h,grid,options.overlap));
                else canvas.getContext('2d').drawImage(image,0,0,canvas.width,canvas.height);
                get('image-split-dimensions').textContent = fits ? `${grid.columns}列 × ${grid.rows}行・1枚あたり約 ${Math.floor(w/grid.columns)}×${Math.floor(h/grid.rows)} px（元画像 ${w}×${h} px）` : `この画像は小さすぎるため${options.pieces}枚に分割できません。分割数を減らしてください。`;
            } else { canvas.width = 0; get('image-split-dimensions').textContent = ''; }
            get('image-split-save').disabled = get('image-split-attach').disabled = splitState.busy || !fits;
        }
        function setSplitBusy(busy) {
            splitState.busy = busy;
            get('image-split-modal').querySelectorAll('input,button').forEach(el => { el.disabled = busy; }); updateSplitPreview();
        }
        async function runImageSplit(attach) {
            if (splitState.busy) return;
            const options = splitOptions(), compression = readSplitCompression();
            if (attach && browserFastModeEnabled && browserFastLocalFiles.size + splitState.files.length * (options.pieces + Number(options.includeOriginal)) > BROWSER_FAST_MAX_IMAGES) {
                get('image-split-status').textContent = '高速モードの画像は4枚・合計12MBまでです'; return;
            }
            setSplitBusy(true); clearSplitDownloads();
            try {
                const outputs = [];
                // Release the preview before decoding full-size sources, one at a time.
                if (splitState.release) splitState.release(); splitState.release = null; splitState.image = null;
                for (let index = 0; index < splitState.files.length; index++) {
                    get('image-split-status').textContent = `処理中... ${index+1} / ${splitState.files.length}`;
                    outputs.push(...await renderSplitFiles(splitState.files[index], options, compression));
                }
                if (attach) {
                    const existingBytes = Array.from(browserFastLocalFiles.values()).reduce((sum, entry) => sum + entry.file.size, 0);
                    if (browserFastModeEnabled && existingBytes + outputs.reduce((sum,f)=>sum+f.size,0) > BROWSER_FAST_MAX_BYTES) throw new Error('高速モードの画像は4枚・合計12MBまでです');
                    await handleFiles(outputs, { skipCompression: true });
                    setSplitBusy(false); closeImageSplit();
                } else {
                    for (const file of outputs) {
                        const url = URL.createObjectURL(file); splitState.urls.push(url);
                        const link = document.createElement('a'); link.href = url; link.download = file.name; link.textContent = file.name; link.className = 'block underline';
                        get('image-split-downloads').append(link); link.click();
                    }
                    get('image-split-status').textContent = `${outputs.length}枚のダウンロードを開始しました。保存されない画像は下のリンクから保存できます。`;
                }
            } catch (e) { get('image-split-status').textContent = e.message || '画像を分割できませんでした。'; }
            finally {
                setSplitBusy(false);
                if (splitState.files.length) {
                    const status = get('image-split-status').textContent; await loadSplitPreview(); get('image-split-status').textContent = status;
                }
            }
        }
        function openSplitCompression() {
            const s = readSplitCompression(), prefix = 'image-split-compression';
            get(`${prefix}-enabled`).checked = s.enabled; get(`${prefix}-output-type`).value = s.outputType;
            get(`${prefix}-format-only`).checked = s.formatOnly; get(`${prefix}-max-size`).value = s.maxSizeMB; get(`${prefix}-max-dim`).value = s.maxDimension;
            syncSplitCompressionUi(); showModal(`${prefix}-modal`);
        }
        function syncSplitCompressionUi() {
            const prefix = 'image-split-compression';
            for (const field of ['size','dim']) {
                const off = get(`${prefix}-format-only`).checked;
                get(`${prefix}-max-${field}`).disabled = off; get(`${prefix}-${field}-wrap`).style.opacity = off ? '0.4' : '1';
            }
        }
        window.addEventListener('popstate', () => { if (!splitState.busy) closeImageSplit(); });
        document.addEventListener('keydown', event => {
            if (event.key !== 'Escape' || !get('image-split-modal')) return;
            if (!get('image-split-compression-modal').classList.contains('hidden')) hideModal('image-split-compression-modal');
            else if (!get('image-split-modal').classList.contains('hidden')) closeImageSplit();
        });
        document.addEventListener('DOMContentLoaded', () => {
            if (!get('image-split-modal')) return;
            get('upload-split-btn').onclick = () => get('image-split-input').click();
            get('image-split-input').onchange = e => { openImageSplit(e.target.files); e.target.value = ''; };
            get('image-split-prev').onclick = () => { splitState.index--; loadSplitPreview(); };
            get('image-split-next').onclick = () => { splitState.index++; loadSplitPreview(); };
            for (const field of ['pieces','overlap','original','mark-original','mark-pieces']) get(`image-split-${field}`).oninput = updateSplitPreview;
            get('image-split-cancel').onclick = closeImageSplit;
            get('image-split-save').onclick = () => runImageSplit(false); get('image-split-attach').onclick = () => runImageSplit(true);
            get('image-split-settings').onclick = openSplitCompression;
            get('image-split-compression-enabled').onchange = get('image-split-compression-format-only').onchange = syncSplitCompressionUi;
            get('image-split-compression-close').onclick = () => hideModal('image-split-compression-modal');
            get('image-split-compression-save').onclick = () => {
                const p = 'image-split-compression';
                const size = get(`${p}-max-size`), dim = get(`${p}-max-dim`);
                if (!size.reportValidity() || !dim.reportValidity()) return;
                const settings = { enabled: get(`${p}-enabled`).checked, outputType: get(`${p}-output-type`).value,
                    formatOnly: get(`${p}-format-only`).checked, maxSizeMB: Number(size.value), maxDimension: Number(dim.value) };
                localStorage.setItem(SPLIT_COMPRESSION_KEY, JSON.stringify(settings)); hideModal(`${p}-modal`); showToast('設定を保存しました', 'success');
            };
        });
