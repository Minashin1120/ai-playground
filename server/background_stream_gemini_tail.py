"""Post-stream tail of the Gemini chat branch of `background_chat_task`.

Extracted from `server/background_stream_gemini.py`.  The branch used to run
inline in `background_chat_task`, so it shared that function's locals.  It now
receives a `locals()` snapshot as ``local`` and returns the updated answer text.

Module-level helpers are resolved through the `app` namespace because `app.py`
execs every server part into one module.
"""


def background_stream_gemini_tail(
    local,
    full_res,
    agentic_saved_urls,
    agentic_consumed_urls,
    agentic_filename_url_map,
    sandbox_text_buffer,
    auto_enable_url_context,
    gemini_local_python,
    grounding_chunks,
    grounding_supports,
    grounding_model_text,
    url_context_chunks,
    last_file_tool_error,
    prior_sandbox_names,
    thread_id,
    options,
    _chunk_grounding_info,
    _add_gemini_citations,
    _url_retrieval_status_label,
):
    """Finish the Gemini answer (sandbox refs, citations, Python output)."""

    pub = local.get("pub")
    re = local.get("re")

    if sandbox_text_buffer[0]:
        _tail = sandbox_text_buffer[0]
        sandbox_text_buffer[0] = ""
        if _tail:
            full_res += _tail
            pub("content", _tail)

    if agentic_saved_urls or agentic_consumed_urls or agentic_filename_url_map or "sandbox:" in full_res:
        full_res = _rewrite_sandbox_image_refs(
            full_res, agentic_saved_urls, agentic_consumed_urls, agentic_filename_url_map
        )
        for _consumed_url in list(agentic_consumed_urls):
            full_res = full_res.replace(
                f"![Agentic View]({_consumed_url})", ""
            )

    if _has_bare_sandbox_image_ref(full_res):
        if prior_sandbox_names is None:
            prior_sandbox_names = _prior_sandbox_image_names(thread_id)
        full_res = _resolve_prior_sandbox_image_refs(full_res, prior_sandbox_names)

    if grounding_chunks and (options.get('enable_search') or options.get('enable_maps')):
        if grounding_supports:
            full_res = _add_gemini_citations(
                full_res, grounding_supports, grounding_chunks, "".join(grounding_model_text)
            )
        sources_lines = []
        has_sources = False
        for i, chunk in enumerate(grounding_chunks):
            title, uri = _chunk_grounding_info(chunk)
            if title or uri:
                has_sources = True
            if uri:
                label = title or uri
                sources_lines.append(f"- [{i + 1}] [{label}]({uri})")
            elif title:
                sources_lines.append(f"- [{i + 1}] {title}")
            else:
                sources_lines.append(f"- [{i + 1}] (source unavailable)")
        if has_sources:
            sources_text = "\n\n**Sources:**\n" + "\n".join(sources_lines) + "\n"
            full_res += sources_text
            pub("content", sources_text)

    if url_context_chunks and (options.get('enable_url_context') or auto_enable_url_context):
        url_sources = []
        has_url_sources = False
        for i, chunk in enumerate(url_context_chunks):
            uri = None
            status = None
            if isinstance(chunk, dict):
                uri = chunk.get('retrieved_url') or chunk.get('retrievedUrl')
                status = chunk.get('url_retrieval_status') or chunk.get('urlRetrievalStatus')
            else:
                uri = getattr(chunk, 'retrieved_url', None) or getattr(chunk, 'retrievedUrl', None)
                status = getattr(chunk, 'url_retrieval_status', None) or getattr(chunk, 'urlRetrievalStatus', None)
            if uri:
                has_url_sources = True
                st_label = _url_retrieval_status_label(status)
                st_str = f" ({st_label})" if st_label else ""
                url_sources.append(f"- [{uri}]({uri}){st_str}")
        if has_url_sources:
            url_sources_text = "\n\n**URL Context:**\n" + "\n".join(url_sources) + "\n"
            full_res += url_sources_text
            pub("content", url_sources_text)

    if gemini_local_python and options.get('enable_python'):
        try:
            def _extract_exec_blocks(text):
                blocks = []
                if not text:
                    return blocks
                for m in re.finditer(r"```python\\s*\\n(.*?)```", text, flags=re.S|re.I):
                    code = m.group(1) or ""
                    lines = code.splitlines()
                    marker_idx = None
                    for i, line in enumerate(lines):
                        if not line.strip():
                            continue
                        if line.strip().upper() in ("# EXECUTE", "#EXECUTE", "# EXEC"):
                            marker_idx = i
                        break
                    if marker_idx is None:
                        continue
                    run_code = "\n".join(lines[marker_idx + 1:]).strip()
                    if run_code:
                        blocks.append(run_code)
                return blocks

            exec_blocks = _extract_exec_blocks(full_res)
            for b in exec_blocks:
                result = safe_execute_python(b)
                out_txt = f"\\n**Output:**\\n```\\n{result}\\n```\\n"
                full_res += out_txt
                pub("content", out_txt)
                pub("python", {"id": f"gem_local_py_{int(time.time()*1000)}_{os.urandom(3).hex()}", "code": b, "output": result})
        except Exception as e:
            log_force(f"Gemini local python failed: {e}")

    if not full_res.strip() and last_file_tool_error:
        full_res = last_file_tool_error
        pub("content", full_res)


    return full_res
