"""background_stream_gemini_embedding branch of `background_chat_task` (provider dispatch chain).

Extracted from `server/background.py`.  The branch used to run inline, so it
shared `background_chat_task`'s locals.  It now receives a `locals()` snapshot
as ``local`` and returns full_res.

Module-level helpers are resolved through the `app` namespace because `app.py`
execs every server part into one module.
"""


def background_stream_gemini_embedding(local):
    """Generate a Gemini embedding and return the updated answer text."""

    _mark_provider_request_started = local.get('_mark_provider_request_started')
    final_message_text = local.get('final_message_text')
    full_res = local.get('full_res')
    g_client = local.get('g_client')
    model_key = local.get('model_key')
    pub = local.get('pub')

    try:
        _mark_provider_request_started()
        emb = g_client.models.embed_content(
            model=model_key,
            contents=final_message_text,
        )
        emb_list = emb.embeddings if getattr(emb, "embeddings", None) else None
        values = emb_list[0].values if emb_list else None
        if values is None:
            pub("error", "Gemini Embedding returned no values.")
        else:
            dims = len(values)
            preview = ", ".join(f"{float(v):.6f}" for v in values[:12])
            out = (
                f"**Gemini Embedding 2**\n\n"
                f"- 次元数: **{dims}**\n"
                f"- 先頭12次元: `[{preview}{', ...' if dims > 12 else ''}]`\n\n"
                f"*入力テキストの埋め込みベクトルを生成しました。*"
            )
            pub("content", out)
            full_res += out
    except Exception as e:
        logger.exception("Gemini Embedding Error")
        pub("error", f"Gemini Embedding Error: {str(e)}")
    return full_res
