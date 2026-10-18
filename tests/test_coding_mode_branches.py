"""Coding targets must follow the ancestry of an edited prompt's new branch."""
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which("node"), "Node.js is required")
class CodingModeBranchTests(unittest.TestCase):
    def test_editing_uses_only_ancestor_code_and_new_prompt_code(self):
        source = (Path(__file__).resolve().parents[1] / "static/js/chat_core_parts/"
                  "chat_core.part03_security_token_promptbar.js").read_text()
        functions = source[source.index("        function codingBranchAllowsTarget("):
                           source.index("        function syncCodingTargetButtons(")]
        setup = r'''
const assert = require('node:assert/strict');
let currentThreadId = 'thread', currentParentId = null, editingMessageId = '10505';
let codingTargetSelection = null;
const allMessages = [
    {id: 10505, parent_id: null}, {id: 10506, parent_id: 10505},
    {id: 10509, parent_id: 10506}, {id: 10510, parent_id: 10509},
    {id: 10511, parent_id: null}, {id: 10512, parent_id: 10511}
];
const messageMeta = {};
const buttons = [
    {code: '<body>ancestor</body>', language: 'html', message_id: '10506'},
    {code: '<body>old descendant</body>', language: 'html', message_id: '10510'},
    {code: '<body>other branch</body>', language: 'html', message_id: '10512'}
];
const get = () => ({querySelectorAll: () => buttons});
const getCodingTargetFromButton = button => button;
const hashString = text => String(text.length);
'''
        checks = r'''
// Feedback 05521f5142ba: editing the first prompt starts at parent=null.
assert.deepEqual(collectCodingCandidates('Create a new animation'), []);
assert.equal(resolveCodingTarget('Create a new animation'), null);
codingTargetSelection = {...buttons[1], thread_id: currentThreadId};
assert.deepEqual(collectCodingCandidates('Create a new animation'), []);
assert.equal(codingTargetSelection, null);

// Editing a later prompt may edit ancestor code, never its old answer or siblings.
editingMessageId = '10509'; currentParentId = 10506;
assert.deepEqual(collectCodingCandidates('Change it').map(x => x.message_id), ['10506']);
assert.equal(resolveCodingTarget('Change it').message_id, '10506');
codingTargetSelection = {...buttons[0], thread_id: currentThreadId};
assert.equal(collectCodingCandidates('Change it')[0].explicit, true);
codingTargetSelection = {...buttons[2], thread_id: currentThreadId};
assert.equal(resolveCodingTarget('Change it').message_id, '10506');
assert.equal(codingTargetSelection, null);

// Code supplied in the edited prompt itself remains a legitimate target.
editingMessageId = '10505'; currentParentId = null;
const prompt = 'Change this\n```js\nconst fresh = 1;\n```';
assert.equal(collectCodingCandidates(prompt).length, 1);
assert.equal(collectCodingCandidates(prompt)[0].prompt_source, true);
assert.equal(resolveCodingTarget(prompt).code, 'const fresh = 1;');

// Cancelling editing restores the existing normal-send candidate behavior.
editingMessageId = null;
assert.equal(collectCodingCandidates('Continue').length, 3);
'''
        result = subprocess.run(["node", "-e", setup + functions + checks],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
