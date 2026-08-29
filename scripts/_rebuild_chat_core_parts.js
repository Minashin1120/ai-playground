#!/usr/bin/env node
/* Re-split the combined chat_core source into ordered parts under
 * static/js/chat_core_parts/, then regenerate the base README table.
 *
 * Usage:
 *   node scripts/_rebuild_chat_core_parts.js <combined-source.js> <parts-dir> <esbuild-node_modules>
 *
 * A boundary after line L is "clean" when the prefix ends at a complete
 * statement (esbuild parses it) or at a complete statement inside the big
 * DOMContentLoaded callback (parses when closed with `});`).  Concatenating the
 * produced parts is always byte-identical to the source.
 */
const esbuild = require(process.argv[4] + '/esbuild');
const fs = require('fs');
const path = require('path');

const SRC = process.argv[2];
const PARTS_DIR = process.argv[3];
const TARGET = 1700; // desired part size
const MIN = 900;      // smallest acceptable part
const SLACK = 300;    // allowed overshoot above TARGET when picking a boundary
const MAX = 2000;     // hard cap (must match _release_common.CHAT_CORE_PART_MAX_LINES)

const src = fs.readFileSync(SRC, 'utf8');
const lines = src.split('\n');
const n = lines.length;

function parseOk(prefix) {
  try {
    esbuild.transformSync(prefix, { loader: 'js', logLevel: 'silent' });
    return true;
  } catch (e) {
    return false;
  }
}

// Candidate lines to test: every 60 lines plus lines that end a statement.
const candidates = new Set([n]);
for (let i = 0; i < n; i++) {
  if (/^ {8,12}\};?$/.test(lines[i]) || /^ {8,12}\}$/.test(lines[i]) || (i + 1) % 60 === 0) {
    candidates.add(i + 1);
  }
}

// clean boundary value b (0-based) => lines[0..b-1] form a complete prefix.
const bounds = [];
for (const L of candidates) {
  if (L <= 0 || L >= n) continue;
  const prefix = lines.slice(0, L).join('\n');
  if (parseOk(prefix) || parseOk(prefix + '\n});')) {
    bounds.push(L);
  }
}
bounds.push(n); // whole file is always a valid prefix
bounds.sort((a, b) => a - b);

function pick(start) {
  const goal = start + TARGET;
  const hi = Math.min(goal + SLACK, n);
  const inRange = bounds.filter((b) => b >= start + MIN && b <= hi);
  if (inRange.length) {
    return inRange.reduce((a, b) => Math.abs(b - goal) < Math.abs(a - goal) ? b : a);
  }
  const later = bounds.filter((b) => b >= start + MIN);
  return later.length ? later[0] : n;
}

const parts = [];
let start = 0;
while (start < n) {
  const end = pick(start);
  parts.push([start, end - 1]); // 0-based inclusive line indexes
  if (end <= start) throw new Error('re-split made no progress');
  start = end;
}

// byte-identity check
const partBody = (a, b) => {
  const isLast = b === n - 1;
  return lines.slice(a, b + 1).join('\n') + (isLast ? '' : '\n');
};
const rebuilt = parts.map(([a, b]) => partBody(a, b)).join('');
if (rebuilt !== src) {
  throw new Error('rebuilt source does not match the original');
}

// remove old parts, write new ones
if (fs.existsSync(PARTS_DIR)) {
  for (const f of fs.readdirSync(PARTS_DIR)) {
    if (/^chat_core\.part\d+_.*\.js$/.test(f)) fs.unlinkSync(path.join(PARTS_DIR, f));
  }
} else {
  fs.mkdirSync(PARTS_DIR, { recursive: true });
}

const JS_KEYWORDS = new Set([
  'if', 'for', 'while', 'return', 'else', 'switch', 'case', 'try', 'catch',
  'finally', 'do', 'in', 'of', 'function', 'const', 'let', 'var', 'class',
  'async', 'await', 'new', 'this', 'typeof', 'delete', 'void', 'throw',
  'continue', 'break', 'export', 'import', 'yield', 'get', 'set',
]);
const cleanSlug = (s) =>
  s.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 40);

function slugFor(a) {
  const end = Math.min(lines.length, a + 40);
  for (let i = a; i < end; i++) {
    const t = lines[i].trim();
    let m = t.match(/^window\.([A-Za-z_$][\w$]*)/);
    if (m) return cleanSlug(m[1]);
    m = t.match(/^(?:const|let|var|function|async function|class)\s+([A-Za-z_$][\w$]*)/);
    if (m && !JS_KEYWORDS.has(m[1]) && m[1].length >= 4) return cleanSlug(m[1]);
    m = t.match(/^([A-Za-z_$][\w$]*)\s*=/);
    if (m && !JS_KEYWORDS.has(m[1]) && m[1].length >= 4) return cleanSlug(m[1]);
    m = t.match(/^\/\/\s*[-=#]*\s*([A-Za-z_$][\w$-]*)/);
    if (m && m[1].length >= 4) return cleanSlug(m[1]);
  }
  return 'section';
}

const written = [];
parts.forEach(([a, b], idx) => {
  const name = `chat_core.part${String(idx + 1).padStart(2, '0')}_${slugFor(a)}.js`;
  const body = partBody(a, b);
  fs.writeFileSync(path.join(PARTS_DIR, name), body, 'utf8');
  written.push({ name, first: a + 1, last: b + 1, count: b - a + 1 });
});

console.log(`re-split ${SRC} into ${written.length} parts (${src.length} bytes, byte-identical)`);
for (const w of written) {
  console.log(`  ${w.name}  lines ${w.first}-${w.last} (${w.count})`);
}

// regenerate the base README table (overviews are filled in by the developer)
const rows = written
  .map((w) => `| \`${w.name}\` | ~${String(w.count).padStart(4)} | ${w.first}〜${w.last}行 | （分割後に内容を確認して概要を記入） |`)
  .join('\n');

const readme = `# chat_core_parts — チャットコアの部品ファイル

\`static/js/chat_core.v4.8.*.js\`（約2万行）は、編集しやすくするためにこのディレクトリの**順序付き部品**（\`chat_core.partNN_名前.js\`）に分割されています。\`scripts/build_frontend.sh\` が部品を番号順に連結して結合ソース \`chat_core.v4.8.*.js\` を再生成し、それを圧縮して \`chat_core.min.v4.8.*.js\` を作ります。

- 部品は**連結順に番号が振られており、順番を変えてはいけません**（変えると動作が壊れます）。
- 結合ソースは部品の連結と**バイト単位で一致**することが検証で保証されています。
- **編集対象は部品ファイル**です。結合ソースと圧縮ファイルは手で編集せず、編集後は必ず \`scripts/build_frontend.sh\` を実行してください。

> このREADMEは \`scripts/rebuild_chat_core_parts.sh\` による再分割時に自動生成されました。行数・行範囲は正確ですが、「概要」欄は各ファイルの内容を確認して記入・更新してください。

## 各部品の概要

| 部品 | 行数 | 行範囲 | 主な内容 |
|---|---|---|---|
${rows}

## 再分割が必要になったら

部品が \`CHAT_CORE_PART_MAX_LINES\`（2000行）を超えると \`scripts/verify_changes.sh\` が警告します。その場合は \`scripts/rebuild_chat_core_parts.sh\` を実行して再分割し、このREADMEの「概要」欄を更新してください。
`;
fs.writeFileSync(path.join(PARTS_DIR, 'README.md'), readme, 'utf8');
console.log(`regenerated ${path.join(PARTS_DIR, 'README.md')}`);
