// Generate Android display metadata from the Web source of truth. No Android SDK required.
import { readFileSync, writeFileSync, mkdirSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { resolve, dirname } from 'node:path';
import { runInNewContext } from 'node:vm';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const partsDir = resolve(root, 'static/js/chat_core_parts');
const parts = readdirSync(partsDir).filter(name => /^chat_core\.part.*\.js$/.test(name)).sort();
const combinedSource = parts.map(name => readFileSync(resolve(partsDir, name), 'utf8')).join('\n');
const match = combinedSource.match(/const MODELS\s*=\s*(\[[\s\S]*?\n\s*\]);/);
const tagFunction = combinedSource.match(/function getModelTags\(m, group\)\s*\{[\s\S]*?return tags;\s*\}/);
// The model picker also matches capability words (thinking levels, effort values, …).
const termsFunction = combinedSource.match(/function getModelCapabilitySearchTerms\(m\)\s*\{[\s\S]*?return \[\.\.\.new Set\(terms\)\];\s*\}/);
if (!match || !tagFunction || !termsFunction) throw new Error('Web catalog boundaries changed; update the extractor.');
const data = runInNewContext(`${tagFunction[0]}\n${termsFunction[0]}\n(${match[1]}).flatMap(group => group.items.map(m => ({
    id: m.id, name: m.name, description: m.desc || '', price: m.price || '', category: group.category,
    categoryIcon: group.icon || '', categoryDescription: group.description || '',
    apiId: String(m.apiId || m.id || '').trim(), agenticView: !!m.agenticView,
    implementedAt: m.implementedAt || '', implementedRank: m.implementedRank || 0,
    emoji: m.quickEmoji || '', deprecated: !!m.deprecated, tags: getModelTags(m, group),
    searchTerms: getModelCapabilitySearchTerms(m)
})))`, {}, { timeout: 1000 });
if (data.length < 50 || new Set(data.map(m => m.id)).size !== data.length) throw new Error('Invalid model catalog');
const output = `${JSON.stringify(data, null, 2)}\n`;
const destination = resolve(root, 'android/app/src/main/assets/web-model-catalog.json');
if (process.argv.includes('--check')) {
    if (readFileSync(destination, 'utf8') !== output) throw new Error('Android model catalog is stale. Run node android/ci/sync-web-catalog.mjs');
} else {
    mkdirSync(dirname(destination), { recursive: true });
    writeFileSync(destination, output);
}
console.log(`Android display metadata: ${data.length} Web models${process.argv.includes('--check') ? ' verified' : ' generated'}.`);
