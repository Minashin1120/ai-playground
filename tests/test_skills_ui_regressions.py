import os
import unittest
from pathlib import Path

from tests.chat_template import read_chat_markup

os.environ.setdefault("FLASK_SECRET_KEY", "skills-ui-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-skills-ui-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target

ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "static/js/chat_core_parts"


def part(name):
    return (PARTS / name).read_text(encoding="utf-8")


class SkillsUiRegressionTests(unittest.TestCase):
    def test_sidebar_and_template_use_skill_names(self):
        html = read_chat_markup()
        self.assertIn('id="add-skill-btn"', html)
        self.assertIn('id="skill-list"', html)
        self.assertIn('id="skill-pull-indicator"', html)
        self.assertIn('id="skill-active-indicator"', html)
        self.assertIn('id="active-skill-name"', html)
        self.assertIn('id="skill-suggestions"', html)
        self.assertIn('id="skill-suggestions-list"', html)
        for old in ('id="gem-list"', 'id="add-gem-btn"', 'id="gem-modal"',
                    'id="fixed-prompts-bar"', 'id="gem-default-model"'):
            self.assertNotIn(old, html)

    def test_skill_modal_has_only_name_description_and_instructions(self):
        modal = (ROOT / "templates/chat/overlay_model_skill.html").read_text(encoding="utf-8")
        self.assertIn('id="skill-name"', modal)
        self.assertIn('id="skill-desc"', modal)
        self.assertIn('id="skill-inst"', modal)
        self.assertIn("Short Description", modal)
        self.assertIn("Instructions", modal)
        self.assertNotIn("Default Model", modal)
        self.assertNotIn("Fixed Prompts", modal)
        self.assertNotIn("gem-", modal)

    def test_js_functions_and_urls_use_skill(self):
        joined = "\n".join(path.read_text(encoding="utf-8") for path in PARTS.glob("chat_core.part*.js"))
        for symbol in ("async function loadSkills", "async function openEditSkillModal",
                       "function applyActiveSkill", "function activateSkill",
                       "function clearActiveSkill", "function selectSkillSuggestion",
                       "function showSkillSuggestions", "function hideSkillSuggestions",
                       "function deleteSkill"):
            self.assertIn(symbol, joined)
        self.assertIn("CHAT_CONFIG.urls.handleSkills", joined)
        self.assertIn("/api/skills/", joined)
        for old in ("async function loadGems", "async function openEditGemModal",
                    "function applyActiveGem", "function activateGem"):
            self.assertNotIn(old, joined)

    def test_legacy_gem_url_is_kept_as_alias(self):
        source = part("chat_core.part08_domcontent_account_transfer.js")
        self.assertIn("MODAL_CONFIG['/gem'] = MODAL_CONFIG['/skill']", source)

    def test_account_transfer_uses_skills_category(self):
        settings = (ROOT / "templates/chat/overlay_settings.html").read_text(encoding="utf-8")
        setup = (ROOT / "templates/setup.html").read_text(encoding="utf-8")
        self.assertIn('value="skills"', settings)
        self.assertIn('value="skills"', setup)
        self.assertNotIn('value="gems"', settings)
        self.assertNotIn('value="gems"', setup)
        self.assertIn("imported.skills", part("chat_core.part07_domcontent_initial.js"))

    def test_css_uses_skill_selectors(self):
        css = (ROOT / "static/css/chat.custom.v4.8.1156.css").read_text(encoding="utf-8")
        self.assertIn("#skill-list .skill-item", css)
        self.assertIn("#skill-suggestions", css)
        self.assertNotIn("#gem-list", css)
        self.assertNotIn(".gem-item", css)
        self.assertNotIn("#fixed-prompts-bar", css)


if __name__ == "__main__":
    unittest.main()
