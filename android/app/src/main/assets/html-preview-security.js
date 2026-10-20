// HTML preview security helpers synchronized from Web chat_core.part03_security_token_promptbar.js.
const BLOCKED_SCRIPT_HOSTS = ['polyfill.io', 'cdn.polyfill.io'];
function isBlockedScriptSrc(src) {
    if (!src) return false;
    const s = String(src).trim();
    if (!s) return false;
    let testUrl = s;
    if (s.startsWith('//')) testUrl = 'https:' + s;
    else if (!/^https?:\/\//i.test(s) && !s.startsWith('data:') && !s.startsWith('blob:')) {
        testUrl = 'https://' + s;
    }
    try {
        const u = new URL(testUrl, 'https://example.com');
        const host = (u.hostname || '').toLowerCase();
        return BLOCKED_SCRIPT_HOSTS.some(bad => host === bad || host.endsWith('.' + bad));
    } catch (e) {
        return /polyfill\.io/i.test(s);
    }
}
function isPasswordPromptingScript(code) {
    if (!code) return false;
    const c = String(code);
    const lower = c.toLowerCase();
    // Patterns commonly seen in malicious credential-harvesting "demo" scripts
    if (/prompt\s*\(\s*(['"`]).{0,40}(pass|pwd|password|secret|credential|認証|パスワード|login|pin|暗証)/i.test(c)) return true;
    if (/confirm\s*\(\s*(['"`]).{0,40}(pass|password|削除|重要|delete all|全削除)/i.test(c)) return true;
    if (/(type\s*=\s*['"]?password|name\s*=\s*['"]?password|password.*input|input.*password|getPassword|promptForPass)/i.test(lower)) return true;
    // Prompt + data exfil heuristic (common in bad generated phishing pages)
    if (/prompt\s*\(/.test(c) && /(fetch\(|XMLHttpRequest|\.send\(|navigator\.sendBeacon|location\s*\.\s*(href|replace)|document\.cookie\s*=)/i.test(c)) return true;
    return false;
}
function detectBlockedScriptsInCode(code) {
    if (!code) return false;
    const html = String(code);
    // 1) External script srcs
    const srcRe = /<script\b[^>]*\bsrc\s*=\s*["']?([^"'\s>]+)/gi;
    let m;
    while ((m = srcRe.exec(html)) !== null) {
        if (isBlockedScriptSrc(m[1])) return true;
    }
    // 2) Inline <script> content that asks for passwords / credentials
    const inlineRe = /<script\b(?![^>]*\bsrc\s*=)[^>]*>([\s\S]*?)<\/script>/gi;
    while ((m = inlineRe.exec(html)) !== null) {
        if (isPasswordPromptingScript(m[1])) return true;
    }
    // 3) Direct references in strings or attributes (e.g. const u = 'https://polyfill.io/...')
    if (/["'`]https?:\/\/[^"'`\s]*polyfill\.io/i.test(html) || /src\s*=\s*["'`][^"'`]*polyfill\.io/i.test(html)) return true;
    return false;
}
function sanitizeHtmlForPreview(rawHtml) {
    if (!rawHtml) return '';
    const hadBlocked = detectBlockedScriptsInCode(rawHtml);
    let output = String(rawHtml);
    try {
        const parser = new DOMParser();
        const doc = parser.parseFromString(output, 'text/html');
        let modified = false;
        doc.querySelectorAll('script').forEach((scriptEl) => {
            const src = scriptEl.getAttribute('src') || '';
            let replaced = false;
            if (src && isBlockedScriptSrc(src)) {
                const warn = doc.createElement('div');
                warn.setAttribute('data-blocked-script', 'true');
                warn.style.cssText = 'background:#fee2e2;border:1px solid #ef4444;color:#991b1b;padding:6px 10px;border-radius:6px;font-size:12px;margin:6px 0;font-family:system-ui;';
                const short = src.length > 70 ? src.slice(0, 67) + '...' : src;
                warn.textContent = '⚠ ブロック済み: ' + short + ' （polyfill.io などの危険ドメインはプレビューで無効化されます）';
                if (scriptEl.parentNode) scriptEl.parentNode.replaceChild(warn, scriptEl);
                modified = true;
                replaced = true;
            } else if (!src) {
                const inner = scriptEl.textContent || '';
                if (isPasswordPromptingScript(inner)) {
                    const warn = doc.createElement('div');
                    warn.setAttribute('data-blocked-script', 'true');
                    warn.style.cssText = 'background:#fef3c7;border:1px solid #f59e0b;color:#92400e;padding:6px 10px;border-radius:6px;font-size:12px;margin:6px 0;font-family:system-ui;';
                    warn.textContent = '⚠ ブロック済み: パスワード入力要求などの疑わしいインラインスクリプトを無効化しました';
                    if (scriptEl.parentNode) scriptEl.parentNode.replaceChild(warn, scriptEl);
                    modified = true;
                    replaced = true;
                }
            }
            // non-blocked scripts are intentionally left intact so that legitimate HTML/JS demos continue to work
        });
        // Neutralize javascript: links (defense in depth)
        doc.querySelectorAll('a[href^="javascript:" i], area[href^="javascript:" i]').forEach(a => {
            a.setAttribute('href', '#');
            a.setAttribute('title', (a.getAttribute('title') || '') + ' [javascript: disabled in preview]');
        });
        const head = doc.head || doc.querySelector('head');
        if (head && !head.querySelector('base')) {
            const base = doc.createElement('base');
            base.setAttribute('href', previewBase);
            head.insertBefore(base, head.firstChild);
        }
        if (hadBlocked || modified) {
            const body = doc.body || doc.documentElement;
            if (body) {
                const banner = doc.createElement('div');
                banner.style.cssText = 'position:sticky;top:0;left:0;right:0;z-index:2147483647;background:#7f1d1d;color:#fff;padding:8px 12px;text-align:center;font-size:12px;font-family:system-ui;border-bottom:1px solid #b91c1c;';
                banner.innerHTML = '⚠ <strong>安全プレビュー</strong>: polyfill.io などの危険なスクリプトをブロックしています。実行は自己責任で。';
                if (body.firstChild) {
                    body.insertBefore(banner, body.firstChild);
                } else {
                    body.appendChild(banner);
                }
            }
        }
        output = '<!DOCTYPE html>\n' + (doc.documentElement ? doc.documentElement.outerHTML : output);
    } catch (err) {
        // Regex fallback: at least strip the worst external polyfill scripts
        output = output.replace(/<script\b([^>]*\bsrc\s*=\s*["']?[^"'\s>]*polyfill\.io[^"'\s>]*)["']?[^>]*>[\s\S]*?<\/script>/gi,
            '<!-- blocked polyfill.io script for safety -->');
    }
    return output;
}
