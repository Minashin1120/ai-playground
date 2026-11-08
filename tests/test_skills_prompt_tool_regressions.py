import os
import unittest
from types import SimpleNamespace

from tests.app_source import read_app_source

os.environ.setdefault("FLASK_SECRET_KEY", "skills-tool-test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/ai-chat-skills-tool-tests.db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6399/15")
os.environ.setdefault("RUN_SCHEMA_MIGRATIONS", "0")
os.environ.setdefault("VERBOSE_DEBUG_LOGS", "0")

import app as target


def make_skill(name, description, instruction, uuid=None):
    return SimpleNamespace(
        uuid=uuid or f"uuid-{name}",
        name=name,
        description=description,
        instruction=instruction,
    )


class SkillsPromptToolTests(unittest.TestCase):
    def test_skills_notice_is_registered(self):
        self.assertIn("skills", target.AUTO_SYSTEM_PROMPT_NOTICE_KEYS)
        self.assertIn("skills", target.AUTO_SYSTEM_PROMPT_NOTICE_LABELS)
        notice = target.AUTO_SYSTEM_PROMPT_NOTICE_DEFAULTS["skills"]
        self.assertIn("{{skills}}", notice)
        self.assertIn("use_skill", notice)
        # The list is provided up front; the model is no longer told to output it.
        self.assertIn("already listed below", notice)
        self.assertNotIn("If the user asks which skills are available", notice)

    def test_saved_legacy_skills_notice_is_migrated(self):
        import json
        legacy = (
            "You can use the user's saved skills. A skill is a reusable instruction set.\n"
            "Call use_skill(name) when the user asks to use a skill, or when a skill's "
            "short description clearly matches the request. Load only the skills you need.\n"
            "If the user asks which skills are available, list the names and descriptions below.\n"
            "Never invent a skill that is not listed. When the full instructions are already "
            "included below, follow them without calling a tool.\n\n"
            "Available skills:\n{{skills}}"
        )
        user = SimpleNamespace(
            auto_system_prompt_notices_config=json.dumps({"skills": {"enabled": False, "text": legacy}}),
            apply_auto_system_prompt_notices=True,
        )
        config = target.get_user_auto_system_prompt_notices_config(user)
        self.assertEqual(config["skills"]["text"], target.AUTO_SYSTEM_PROMPT_NOTICE_DEFAULTS["skills"])
        self.assertTrue(config["skills"]["enabled"])
        # A custom text is kept as-is.
        user.auto_system_prompt_notices_config = json.dumps({"skills": {"text": "My own skill note"}})
        config = target.get_user_auto_system_prompt_notices_config(user)
        self.assertEqual(config["skills"]["text"], "My own skill note")

    def test_prompt_bar_switch_gates_catalog_and_tool(self):
        app_source = read_app_source()
        self.assertIn("_skills_enabled = options.get('enable_skills', True)", app_source)
        self.assertIn(
            'if is_llm_model and _skills_enabled and _auto_notice_enabled("skills")',
            app_source,
        )
        self.assertIn("Skill catalog skipped: disabled by prompt bar", app_source)
        self.assertIn("'enable_skills': enable_skills_requested", app_source)
        self.assertIn("'enable_skills'", app_source)
        self.assertIn("'enable_file_creation', 'enable_mcp', 'enable_skills'", app_source)

    def test_catalog_hides_instructions_unless_requested(self):
        skills = [
            make_skill("Summariser", "Summarise text", "Always use three bullets.", "u1"),
            make_skill("Translator", "Translate to Japanese", "Keep the original tone.", "u2"),
        ]
        tool_catalog = target.build_skill_catalog(skills)
        self.assertIn("Summariser: Summarise text", tool_catalog)
        self.assertIn("Translator: Translate to Japanese", tool_catalog)
        self.assertNotIn("Always use three bullets.", tool_catalog)

        embedded = target.build_skill_catalog(
            skills, include_instructions=True, active_uuid="u1"
        )
        self.assertIn("Translator", embedded)
        self.assertNotIn("Always use three bullets.", embedded)
        self.assertIn("Keep the original tone.", embedded)

    def test_render_notice_replaces_tokens(self):
        skills = [make_skill("A", "desc A", "instr A"), make_skill("B", "desc B", "instr B")]
        rendered = target.render_skill_notice(
            "Skills ({{skill_count}}):\n{{skills}}", skills
        )
        self.assertIn("Skills (2):", rendered)
        self.assertIn("A: desc A", rendered)
        self.assertIn("B: desc B", rendered)

    def test_execute_skill_tool_matches_exact_then_case_insensitive(self):
        skills = [
            make_skill("Summariser", "desc", "Use three bullets.", "u1"),
            make_skill("Translator", "desc", "Keep tone.", "u2"),
        ]
        text, meta = target.execute_skill_tool(skills, "Summariser")
        self.assertTrue(meta["ok"])
        self.assertEqual(meta["uuid"], "u1")
        self.assertIn("Use three bullets.", text)

        text, meta = target.execute_skill_tool(skills, "translator")
        self.assertTrue(meta["ok"])
        self.assertEqual(meta["uuid"], "u2")

        text, meta = target.execute_skill_tool(skills, "missing")
        self.assertFalse(meta["ok"])
        self.assertIn("Summariser", text)
        self.assertIn("Translator", text)

    def test_long_instructions_are_truncated(self):
        skills = [make_skill("Long", "desc", "x" * (target.SKILL_INSTRUCTION_LIMIT + 50), "u1")]
        text, meta = target.execute_skill_tool(skills, "Long")
        self.assertTrue(meta["ok"])
        self.assertIn("[Instructions truncated]", text)
        self.assertLessEqual(len(text), target.SKILL_INSTRUCTION_LIMIT + 100)

    def test_tool_support_matches_step_routes(self):
        self.assertTrue(target.skill_tool_supported("gemini-3.6-flash"))
        self.assertTrue(target.skill_tool_supported("claude-sonnet-4-6"))
        self.assertTrue(target.skill_tool_supported("deepseek-v4-flash"))
        self.assertTrue(target.skill_tool_supported("gpt-5.6-sol"))
        self.assertTrue(target.skill_tool_supported("grok-4.6"))
        self.assertTrue(target.skill_tool_supported("kimi-k3"))
        self.assertTrue(target.skill_tool_supported("glm-5.3"))
        self.assertFalse(target.skill_tool_supported("gpt-5-search-api"))
        self.assertFalse(target.skill_tool_supported("gemini-3.6-flash", batch_mode=True))
        # Local Python only replaces the code-execution tool; function tools still work.
        self.assertTrue(target.skill_tool_supported(
            "gemini-3.6-flash", gemini_local_python=True))

    def test_server_parts_and_tool_branches_are_wired(self):
        app_source = read_app_source()
        self.assertIn('"skills.py"', app_source)
        self.assertIn("render_skill_notice(", app_source)
        self.assertIn('if is_llm_model and _skills_enabled and _auto_notice_enabled("skills")', app_source)
        self.assertIn("build_gemini_skill_tool(", app_source)
        self.assertIn("build_skill_tool_anthropic()", app_source)
        self.assertIn("build_skill_tool_openai_chat()", app_source)
        self.assertIn("build_skill_tool_openai_responses()", app_source)
        self.assertIn("kimi_tools.append(build_skill_tool_openai_chat())", app_source)
        self.assertIn("if kimi_tools:", app_source)
        self.assertIn("if call_name == SKILL_TOOL_NAME", app_source)
        self.assertIn("if tu_name == SKILL_TOOL_NAME", app_source)
        self.assertIn("elif name == SKILL_TOOL_NAME", app_source)
        self.assertIn("thought_payload['used_skills']", app_source)

    def test_model_invoked_skill_does_not_become_thread_selection(self):
        """use_skill is stored on the message only; the composer selection is explicit."""
        app_source = read_app_source()
        self.assertIn("if not skill_uuid_val and _used_skills:", app_source)
        self.assertNotIn("th.last_skill_uuid = skill_uuid_val", app_source)
        self.assertNotIn("th.last_skill_uuid = _used_skills", app_source)


if __name__ == "__main__":
    unittest.main()
