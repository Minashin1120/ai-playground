"""Gemini（google-genai）向けMCPツールの組み立て。

google-genai の Automatic Function Calling（AFC）は ``function_declarations``
を含むツールがあると無効化されるため、MCPツールは「Python呼び出し可能ツール」
（callable）として ``conf['tools']`` へ追加する。SDK が関数シグネチャから
declaration を生成するため、JSON Schema のプロパティを Python パラメータへ
写像した ``__signature__`` を持つ関数を動的生成する。

注意:
- 名前は64文字以内・先頭は英字/アンダースコアである必要がある。
- 不正な識別子（ハイフン等）は ``_`` へ置換し、実行時に元の引数名へ戻す。
- 詳細な nested object/array のスキーマ情報は Gemini の AFC 制約により
  単純化される（型のみ）。エッジケースの精度が必要なツールは、説明文で
  パラメータ構造を補足する。
"""
from __future__ import annotations

import inspect
import json
import re
from typing import Any, Optional

from . import tools as mcp_tools

_FUNC_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]*$")
_PROP_SAFE_RE = re.compile(r"\W")


def _is_valid_py_ident(name):
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name))


def _sanitize_prop_name(name, used):
    """JSON Schema property 名をPython識別子へ写像する。"""
    base = _PROP_SAFE_RE.sub("_", str(name or "arg"))
    if not base or base[0].isdigit():
        base = "_" + base
    if not _is_valid_py_ident(base):
        base = "_arg"
    cand = base
    i = 2
    while cand in used:
        cand = f"{base}_{i}"
        i += 1
    used.add(cand)
    return cand


def _python_type_for(schema):
    """JSON Schema の型を Python annotation へ写像する（単純化）。"""
    if not isinstance(schema, dict):
        return Any
    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "string")
    if t == "string":
        return str
    if t == "integer":
        return int
    if t == "number":
        return float
    if t == "boolean":
        return bool
    if t == "array":
        return list
    if t == "object":
        return dict
    # anyOf/oneOf: 含まれる型から選択
    for key in ("anyOf", "oneOf"):
        subs = schema.get(key)
        if isinstance(subs, list) and subs:
            for sub in subs:
                if isinstance(sub, dict) and sub.get("type") not in (None, "null"):
                    return _python_type_for(sub)
    return Any


def _optional(t):
    try:
        return Optional[t]
    except Exception:
        return Any


def build_gemini_mcp_tools(runtime, on_result=None):
    """McpRuntime から Gemini 用の callable リストを組み立てる。

    on_result(text, meta_out, internal_name) が与えられた場合、ツール実行後に
    呼び出される（実行サマリの可視化などに使用）。callable 自体は常に文字列を返す
    （google-genai の Automatic Function Calling が関数戻り値をモデルへ渡すため）。
    """
    callables = []
    for meta in runtime.tool_metas():
        fn = _make_callable(runtime, meta, on_result=on_result)
        if fn is None:
            continue
        callables.append(fn)
    return callables


def _make_callable(runtime, meta, on_result=None):
    input_schema = meta.input_schema or {}
    properties = input_schema.get("properties") or {}
    required = input_schema.get("required") or []
    if not isinstance(properties, dict):
        properties = {}
    if not isinstance(required, list):
        required = []

    annotations = {}
    sig_params = []
    used = set()
    name_to_orig = {}
    defaults_ok = True
    for prop_name, prop_schema in properties.items():
        py_name = _sanitize_prop_name(prop_name, used)
        name_to_orig[py_name] = prop_name
        ann = _python_type_for(prop_schema)
        if prop_name not in required:
            ann = _optional(ann)
            sig_params.append(
                inspect.Parameter(py_name, inspect.Parameter.KEYWORD_ONLY, default=None, annotation=ann)
            )
        else:
            sig_params.append(
                inspect.Parameter(py_name, inspect.Parameter.KEYWORD_ONLY, annotation=ann)
            )
        annotations[py_name] = ann

    def _exec(**kwargs):
        # 送られてきた引数を元のJSON Schemaプロパティ名へ戻す
        args = {}
        for k, v in kwargs.items():
            orig = name_to_orig.get(k, k)
            if v is not None:
                args[orig] = v
        text, meta_out = runtime.execute(meta.internal_name, args)
        if on_result is not None:
            try:
                on_result(text, meta_out, meta.internal_name)
            except Exception:
                pass
        return text

    _exec.__name__ = meta.internal_name
    # モデルへ伝える説明（MCPであることと接続先を明示する）
    doc = mcp_tools.description_for_model(
        meta.internal_name,
        {"name": meta.name, "title": meta.title, "description": meta.description},
        server_name=meta.server_name,
        original_name=meta.name,
    )
    if not _FUNC_NAME_RE.match(meta.internal_name):
        # AFC / API が受け付けない名前はスキップ
        return None
    try:
        _exec.__signature__ = inspect.Signature(sig_params)
    except Exception:
        return None
    _exec.__annotations__ = annotations
    _exec.__doc__ = doc[:2000]
    return _exec
