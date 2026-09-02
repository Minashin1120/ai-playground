"""MCP tools/list 結果の変換・名前空間付与・読み取り/変更の分類。"""
from __future__ import annotations

import hashlib
import re

from . import config


def _sanitize_segment(seg):
    s = re.sub(r"[^A-Za-z0-9_]+", "_", str(seg or ""))
    s = s.strip("_")
    return s


def make_internal_tool_name(server_slug, tool_name):
    """内部ツール名 ``mcp__{slug}__{tool}`` を返す（長さ上限付き・一意化）。"""
    a = _sanitize_segment(server_slug) or "srv"
    b = _sanitize_segment(tool_name) or "tool"
    raw = f"{config.MCP_TOOL_PREFIX}{a}__{b}"
    limit = config.MCP_INTERNAL_TOOL_NAME_MAX_LEN
    if len(raw) <= limit:
        return raw
    digest = hashlib.sha1(f"{server_slug}\x00{tool_name}".encode("utf-8")).hexdigest()[:8]
    keep = limit - len(digest) - 1
    return f"{raw[:keep]}_{digest}"


def is_mcp_tool_name(name):
    return isinstance(name, str) and name.startswith(config.MCP_TOOL_PREFIX)


def classify_readonly(tool_name, description=""):
    """ツール名と説明から読み取り/変更を初期分類する。"""
    name = str(tool_name or "").lower()
    desc = str(description or "").lower()
    # 書き込み動詞が名前に含まれていたら変更扱い
    for verb in config.MCP_WRITE_VERBS:
        if re.search(r"(^|_|[^a-z])" + re.escape(verb) + r"([^a-z]|$)", name):
            return False
    # 読み取り動詞で始まる・含むなら読み取り
    for verb in config.MCP_READONLY_VERBS:
        if name.startswith(verb) or re.search(r"(^|_|[^a-z])" + re.escape(verb) + r"([^a-z]|$)", name):
            return True
    # 説明文の強い書き込み語
    if re.search(r"\b(send|create|write|delete|update|edit|upload|share)\b", desc):
        return False
    if re.search(r"\b(read-only|read only|readonly)\b", desc):
        return True
    # デフォルトは読み取りとして扱わず、確認対象（安全側）
    return False


def tool_spec_from_sdk_tool(tool):
    """SDKの Tool オブジェクト/辞書から正規化 dict を作る。"""
    if hasattr(tool, "model_dump"):
        try:
            raw = tool.model_dump(exclude_none=False)
        except Exception:
            raw = {}
    elif isinstance(tool, dict):
        raw = tool
    else:
        raw = getattr(tool, "__dict__", {}) or {}
    name = raw.get("name") if isinstance(raw, dict) else getattr(tool, "name", None)
    title = (
        raw.get("title")
        if isinstance(raw, dict)
        else getattr(tool, "title", None)
    )
    description = (
        raw.get("description")
        if isinstance(raw, dict)
        else getattr(tool, "description", None)
    )
    input_schema = (
        raw.get("input_schema")
        if isinstance(raw, dict)
        else getattr(tool, "input_schema", None)
    )
    if isinstance(input_schema, dict) and "type" not in input_schema:
        input_schema = {"type": "object", "properties": input_schema.get("properties") or {}}
    annotations = (
        raw.get("annotations")
        if isinstance(raw, dict)
        else getattr(tool, "annotations", None)
    )
    return {
        "name": name,
        "title": title or "",
        "description": description or "",
        "input_schema": input_schema if isinstance(input_schema, dict) else {"type": "object"},
        "annotations": annotations,
    }


def to_openai_function_schema(internal_name, tool):
    """OpenAI互換 function 定義（Responses API / Chat Completions 共通）。"""
    return {
        "type": "function",
        "name": internal_name,
        "description": tool.get("description") or "",
        "parameters": tool.get("input_schema") or {"type": "object"},
    }


def to_chat_completions_function_schema(internal_name, tool):
    """Chat Completions（DeepSeek等）用の入れ子 function 定義。"""
    return {
        "type": "function",
        "function": {
            "name": internal_name,
            "description": tool.get("description") or "",
            "parameters": tool.get("input_schema") or {"type": "object"},
        },
    }


def to_anthropic_tool(internal_name, tool):
    """Anthropic（Claude）ツール定義。"""
    schema = tool.get("input_schema") or {"type": "object"}
    if not isinstance(schema, dict) or schema.get("type") != "object":
        schema = {"type": "object", "properties": schema.get("properties") or {}}
    return {
        "name": internal_name,
        "description": (tool.get("description") or "")[:1024],
        "input_schema": schema,
    }


def content_blocks_to_text(result_dict):
    """CallToolResult の content ブロックをモデルへ返す文字列へ変換する。"""
    if not result_dict:
        return ""
    blocks = result_dict.get("content") or []
    parts = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text = block.get("text")
            if text:
                parts.append(text)
        elif btype == "image":
            mime = block.get("mimeType") or block.get("mime_type") or "image/*"
            data = block.get("data")
            parts.append(f"[MCP image ({mime}, {len(data or '')} bytes of base64 data) - image data not embedded]")
        elif btype == "audio":
            mime = block.get("mimeType") or block.get("mime_type") or "audio/*"
            parts.append(f"[MCP audio ({mime}) - audio data not embedded]")
        elif btype == "resource":
            uri = block.get("uri") or block.get("resource", {}).get("uri")
            text = block.get("text") or block.get("resource", {}).get("text")
            label = f"[MCP resource {uri}]"
            if text:
                parts.append(f"{label}\n{text}")
            else:
                parts.append(label)
        elif btype == "embedded":
            parts.append("[MCP embedded resource]")
        else:
            # 未知ブロックはJSON化
            try:
                import json
                parts.append(json.dumps(block, ensure_ascii=False))
            except Exception:
                parts.append(str(block))
    text = "\n".join(parts)
    structured = result_dict.get("structured_content")
    if structured is not None and not text:
        import json
        try:
            text = json.dumps(structured, ensure_ascii=False)
        except Exception:
            text = str(structured)
    return text
