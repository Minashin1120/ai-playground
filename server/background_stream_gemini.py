"""Gemini branch of `background_chat_task`.

Extracted from `server/background.py`.  The branch used to run inline, so it
shared that function's locals.  It now receives a `locals()` snapshot as
``local`` and returns the values the shared stream tail still needs.

Module-level helpers are resolved through the `app` namespace because `app.py`
execs every server part into one module.
"""

# Returned to background_chat_task when this branch ended the whole task.
BACKGROUND_STREAM_EARLY_RETURN = object()


def background_stream_gemini(local):
    """Run the Gemini branch and report the values the shared tail needs."""
    # region: Gemini setup and local bindings
    full_res = local.get("full_res")
    m = local.get("m")
    model_key = local.get("model_key")
    final_usage_metadata = local.get("final_usage_metadata")
    thought_accumulated = local.get("thought_accumulated")

    log_force("Routing: Gemini Branch")
    # endregion


    # region: Branch dispatch (media parts, chat part, and the shared tail)
    if is_gemini_transcribe_model_key(model_key):
        _transcribe_result = background_stream_gemini_transcribe(local)
        if _transcribe_result is BACKGROUND_STREAM_EARLY_RETURN:
            return BACKGROUND_STREAM_EARLY_RETURN
        full_res = _transcribe_result
    elif "tts" in model_key:
        _tts_result = background_stream_gemini_tts(local)
        if _tts_result is BACKGROUND_STREAM_EARLY_RETURN:
            return BACKGROUND_STREAM_EARLY_RETURN
        full_res = _tts_result
    elif is_gemini_image_model_key(model_key):
        _image_result = background_stream_gemini_image(local)
        if _image_result is BACKGROUND_STREAM_EARLY_RETURN:
            return BACKGROUND_STREAM_EARLY_RETURN
        full_res = _image_result
    else:
        _chat_result = background_stream_gemini_chat(local)
        if _chat_result is BACKGROUND_STREAM_EARLY_RETURN:
            return BACKGROUND_STREAM_EARLY_RETURN
        full_res, thought_accumulated = _chat_result
    # endregion

    return (full_res, m, thought_accumulated, final_usage_metadata)
