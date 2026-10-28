"""Execute the shared picker and history handlers with a small DOM/history fixture."""
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "static/js/chat_core_parts"


class SettingsModelPickerRegressionTests(unittest.TestCase):
    def test_draft_selection_filtering_and_same_path_history(self):
        source = (PARTS / "chat_core.part06_model_media_prompt_cache.js").read_text()
        history_source = (PARTS / "chat_core.part09_domcontent_popstate_modals.js").read_text()
        chunks = [
            source[source.index("let settingsModelPickerTarget = null;"):source.index("const modelListGroups = [];")],
            source[source.index('function renderModelList(filter = "", options = {})'):source.index("function scheduleModelListRender")],
            source[source.index("function openModelModal()"):source.index("function populateAiSafeFormFields")],
            history_source[:history_source.index("const initialPath = location.pathname;")],
        ]
        fixture = r"""
const assert = require('node:assert/strict');
const listeners = {};
global.window = global;
window.addEventListener = (name, fn) => { listeners[name] = fn; };
global.document = { addEventListener: (name, fn) => { listeners[name] = fn; } };
global.location = { pathname: '/settings' };
const entries = [{ url: '/', state: {} }, { url: '/settings', state: { modal: 'settings' } }];
let cursor = 1, events = [];
global.history = {
    get state() { return entries[cursor].state; },
    pushState(state, _, url) { entries.splice(++cursor); entries.push({ state, url }); location.pathname = url; },
    back() { if (cursor) { --cursor; location.pathname = entries[cursor].url; events.push(() => listeners.popstate({ state: history.state })); } },
    forward() { if (cursor < entries.length - 1) { ++cursor; location.pathname = entries[cursor].url; events.push(() => listeners.popstate({ state: history.state })); } },
};
function flush() { while (events.length) events.shift()(); }
function classes(initial = []) {
    const values = new Set(initial);
    return { contains: x => values.has(x), add: x => values.add(x), remove: x => values.delete(x),
        toggle(x, on) { if (on) values.add(x); else values.delete(x); } };
}
function element() { return { classList: classes(), focus() {}, value: '', dispatchEvent() {} }; }
const dom = {};
const get = id => dom[id];
['settings-modal', 'model-modal', 'model-search', 'model-select', 'model-selector-text', 'model-list-container'].forEach(id => dom[id] = element());
dom['settings-modal'].classList.add('modal-open');
let changes = 0;
function select(id, options, value) {
    dom[id] = { ...element(), options: options.map(value => ({ value, textContent: value + ' name' })), value,
        get selectedOptions() { return this.options.filter(o => o.value === this.value); },
        dispatchEvent() { changes++; } };
    dom[id + '-label'] = element(); dom[id + '-picker'] = element();
}
select('set-default-model', ['gpt-5.5', 'claude-test', 'gemini-test'], 'claude-test');
select('set-default-vision-model', ['gemini-test', 'claude-test'], 'gemini-test');
dom['model-select'].value = 'gpt-5.5';
let activeModelTag = 'all', settingsModalLoaded = true, modelChosenManually = false;
let currentVisionModel = 'chat-vision', opens = 0, modelListAnimated = false, scrolled = '';
const requestAnimationFrame = fn => { fn(); return 1; };
const updateModelTagUi = () => {}, syncModelSearchClear = () => {};
const scrollSelectedModelIntoView = () => { scrolled = modelPickerSelectedId(); };
const showModal = id => dom[id].classList.add('modal-open');
const hideModal = id => dom[id].classList.remove('modal-open');
const isPromptCacheEnabled = () => true, getPromptCacheLockedProvider = () => 'openai';
const getModelApiProvider = id => id === 'gpt-5.5' ? 'openai' : 'other';
const PROVIDER_LABELS = { openai: 'OpenAI' };
const modelListBanner = element(), modelListEmpty = element();
const entriesForModels = ['gpt-5.5', 'claude-test', 'gemini-test'].map(id => ({ model: { id }, button: element(),
    provider: getModelApiProvider(id), searchText: id, tags: new Set(['all']) }));
const modelListGroups = [{ element: element(), entries: entriesForModels }];
const buildModelList = () => {};
const updateModelButtonSelection = (entry, selected) => entry.selected = entry.model.id === selected;
const MODELS = [{ items: entriesForModels.map(e => ({ id: e.model.id, name: e.model.id })) }];
window.innerWidth = 1000;
window.openSettingsModal = async () => { opens++; showModal('settings-modal'); };
const MODAL_CONFIG = {
    '/settings': { id: 'settings-modal', open: window.openSettingsModal },
    '/model': { id: 'model-modal', open: () => openModelModal() },
};
const closeModalById = (id, skip) => id === 'model-modal' ? window.closeModelModal(skip) : hideModal(id);
const startNewChat = () => {}, loadMessages = () => {};
let currentThreadId = null;
"""
        scenarios = r"""
syncSettingsModelPickerLabels();
assert.equal(dom['set-default-model-label'].textContent, 'claude-test name');
openSettingsModelPicker('set-default-model');
assert.equal(location.pathname, '/settings');
assert.equal(history.state.settingsModelPicker, 'set-default-model');
assert.equal(dom['settings-modal'].inert, true);
assert.equal(scrolled, 'claude-test');
assert.equal(entriesForModels.find(e => e.model.id === 'claude-test').selected, true);
assert.equal(entriesForModels.every(e => !e.button.classList.contains('hidden')), true);
history.back(); flush();
assert.equal(dom['model-modal'].classList.contains('modal-open'), false);
assert.equal(dom['settings-modal'].classList.contains('modal-open'), true);
assert.equal(dom['settings-modal'].inert, false);
assert.equal(opens, 0);
history.forward(); flush();
assert.equal(settingsModelPickerTarget, 'set-default-model');
selectModelByUser('gemini-test', 'Gemini'); flush();
assert.equal(dom['set-default-model'].value, 'gemini-test');
assert.equal(dom['set-default-model-label'].textContent, 'gemini-test name');
assert.equal(dom['model-select'].value, 'gpt-5.5');
assert.equal(currentVisionModel, 'chat-vision');
assert.equal(modelChosenManually, false);
assert.equal(changes, 1);
assert.equal(settingsModelPickerTarget, null);
assert.equal(history.state.settingsModelPicker, undefined);
assert.equal(opens, 0);
openSettingsModelPicker('set-default-vision-model');
assert.equal(entriesForModels[0].button.classList.contains('hidden'), true);
selectModelByUser('gpt-5.5', 'GPT');
assert.equal(dom['set-default-vision-model'].value, 'gemini-test');
assert.equal(settingsModelPickerTarget, 'set-default-vision-model');
selectModelByUser('claude-test', 'Claude'); flush();
assert.equal(dom['set-default-vision-model'].value, 'claude-test');
assert.equal(dom['set-default-model'].value, 'gemini-test');
assert.equal(changes, 2);
openSettingsModelPicker('set-default-model');
listeners.keydown({ key: 'Escape', preventDefault() {} }); flush();
assert.equal(settingsModelPickerTarget, null);
assert.equal(dom['set-default-model'].value, 'gemini-test');
openSettingsModelPicker('set-default-model');
history.back(); flush(); history.back(); flush();
assert.equal(location.pathname, '/');
assert.equal(dom['settings-modal'].classList.contains('modal-open'), false);
// Return to settings normally: the parent opens once, then its child reuses it.
history.forward(); flush(); history.forward(); flush();
assert.equal(opens, 1);
assert.equal(settingsModelPickerTarget, 'set-default-model');
window.closeModelModal(); flush();
openModelModal();
assert.equal(location.pathname, '/model');
assert.equal(modelListBanner.classList.contains('hidden'), false);
assert.equal(entriesForModels[1].button.classList.contains('hidden'), true);
window.closeModelModal(); flush();
settingsModalLoaded = false;
openSettingsModelPicker('set-default-model');
assert.equal(settingsModelPickerTarget, null);
"""
        result = subprocess.run(
            ["node", "-e", fixture + "\n" + "\n".join(chunks) + "\n" + scenarios],
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
