"""Skill catalog, ``use_skill`` tool schema and execution.

Skills are reusable instruction sets owned by one user.  Only the name and
short description are shown to the model by default; the model calls
``use_skill`` to load the full instructions on demand.  Paths that cannot run
tools (Batch, local Python, unsupported providers) get the instructions
embedded in the catalog instead.
"""

SKILL_TOOL_NAME = "use_skill"
SKILL_DESCRIPTION_LIMIT = 300
SKILL_INSTRUCTION_LIMIT = 20_000
SKILL_CATALOG_TOTAL_LIMIT = 60_000

SKILL_TOOL_DESCRIPTION = (
    "Load the full instructions for one of the user's saved skills. "
    "Call this when the user asks to use a skill, or when a skill's short "
    "description clearly matches the request. Use the exact skill name from "
    "the available skills list. Never invent a skill that is not listed."
)

SKILL_TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Exact skill name from the available skills list.",
        }
    },
    "required": ["name"],
    "additionalProperties": False,
}


def _short_description(skill):
    description = str(getattr(skill, "description", "") or "").replace("\r\n", "\n").strip()
    if len(description) > SKILL_DESCRIPTION_LIMIT:
        description = description[:SKILL_DESCRIPTION_LIMIT].rstrip() + "…"
    return description


def _skill_instructions(skill):
    instruction = str(getattr(skill, "instruction", "") or "").strip()
    if len(instruction) > SKILL_INSTRUCTION_LIMIT:
        instruction = instruction[:SKILL_INSTRUCTION_LIMIT].rstrip() + "\n\n[Instructions truncated]"
    return instruction


def build_skill_catalog(skills, include_instructions=False, active_uuid=None):
    """Render the list shown to the model.

    ``include_instructions`` is used on paths without tool calling.  The
    explicitly selected skill is skipped there because its instructions are
    already part of the system prompt.
    """
    rows = list(skills or [])
    if not rows:
        return ""
    active = str(active_uuid or "")
    blocks = []
    used_chars = 0
    for skill in rows:
        name = str(getattr(skill, "name", "") or "").strip()
        if not name:
            continue
        description = _short_description(skill)
        summary = f"- {name}" + (f": {description}" if description else "")
        if include_instructions and str(getattr(skill, "uuid", "") or "") != active:
            instruction = _skill_instructions(skill)
            block = f"### {name}\n{description}\nInstructions:\n{instruction}" if description else (
                f"### {name}\nInstructions:\n{instruction}"
            )
            if used_chars + len(block) > SKILL_CATALOG_TOTAL_LIMIT:
                blocks.append("...(additional skill instructions omitted)")
                break
            used_chars += len(block)
            blocks.append(block)
        else:
            blocks.append(summary)
    return "\n".join(blocks)


def render_skill_notice(template_text, skills, include_instructions=False, active_uuid=None):
    """Expand ``{{skills}}`` / ``{{skill_count}}`` in the user's notice text."""
    rows = list(skills or [])
    catalog = build_skill_catalog(
        rows, include_instructions=include_instructions, active_uuid=active_uuid
    )
    rendered = str(template_text or "").replace("\r\n", "\n").strip()
    replaced = False
    for token in ("{{skills}}", "{skills}"):
        if token in rendered:
            rendered = rendered.replace(token, catalog)
            replaced = True
    for token in ("{{skill_count}}", "{skill_count}"):
        if token in rendered:
            rendered = rendered.replace(token, str(len(rows)))
            replaced = True
    if catalog and not replaced:
        rendered = f"{rendered}\n\n{catalog}".strip() if rendered else catalog
    return rendered.strip()


def build_skill_tool_definition():
    return {
        "name": SKILL_TOOL_NAME,
        "description": SKILL_TOOL_DESCRIPTION,
        "parameters": dict(SKILL_TOOL_PARAMETERS),
    }


def build_skill_tool_openai_chat():
    return {"type": "function", "function": build_skill_tool_definition()}


def build_skill_tool_openai_responses():
    definition = build_skill_tool_definition()
    return {
        "type": "function",
        "name": definition["name"],
        "description": definition["description"],
        "parameters": definition["parameters"],
    }


def build_skill_tool_anthropic():
    return {
        "name": SKILL_TOOL_NAME,
        "description": SKILL_TOOL_DESCRIPTION,
        "input_schema": dict(SKILL_TOOL_PARAMETERS),
    }


def execute_skill_tool(skills, name):
    """Return ``(text_for_model, meta)`` for one ``use_skill`` call.

    The instruction body is never logged or stored by this module.  ``meta``
    contains only the canonical name/UUID and whether the call succeeded.
    """
    rows = list(skills or [])
    query = str(name or "").strip()
    target = next((skill for skill in rows if str(getattr(skill, "name", "") or "") == query), None)
    if target is None and query:
        lowered = query.lower()
        target = next(
            (skill for skill in rows if str(getattr(skill, "name", "") or "").lower() == lowered),
            None,
        )
    if target is None:
        available = ", ".join(
            str(getattr(skill, "name", "") or "").strip()
            for skill in rows[:50]
            if str(getattr(skill, "name", "") or "").strip()
        )
        message = f"Error: no skill named {query!r}."
        if available:
            message += f" Available skills: {available}"
        return message, {"ok": False, "found": False, "name": query}

    instruction = _skill_instructions(target)
    text = f"Skill: {target.name}\n\n{instruction}" if instruction else f"Skill: {target.name}"
    return text, {
        "ok": True,
        "found": True,
        "name": target.name,
        "uuid": target.uuid,
    }


def build_gemini_skill_tool(skills, on_result=None):
    """Build the google-genai callable used by Automatic Function Calling."""

    def use_skill(name: str) -> str:
        text, meta = execute_skill_tool(skills, name)
        if on_result is not None:
            try:
                on_result(text, meta)
            except Exception:
                pass
        return text

    use_skill.__name__ = SKILL_TOOL_NAME
    use_skill.__doc__ = SKILL_TOOL_DESCRIPTION
    use_skill.__annotations__ = {"name": str, "return": str}
    return use_skill


def skill_tool_supported(model_key, batch_mode=False, gemini_local_python=False):
    """Whether the current route can run the ``use_skill`` tool.

    The remaining Chat Completions routes (Kimi / Z.AI / OpenAI search models)
    and Batch are handled by the embedded-instructions fallback.
    """
    if batch_mode:
        return False
    model = str(model_key or "").strip().lower()
    if not model:
        return False
    # Gemini local Python only replaces the code-execution tool; function tools
    # (including use_skill) are still available on the same route.
    if model.startswith(("claude", "deepseek", "gemini", "grok")):
        return True
    if model.startswith(("kimi", "glm-")):
        return True
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        # OpenAI Search API models keep their own search-only tool contract and
        # receive the embedded instructions instead.
        return "search" not in model
    return False
