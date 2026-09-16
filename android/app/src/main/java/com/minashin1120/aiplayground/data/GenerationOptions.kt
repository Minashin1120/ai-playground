package com.minashin1120.aiplayground.data

import org.json.JSONObject

data class GenerationOption(
    val key: String, val label: String, val defaultValue: String = "",
    val choices: List<String> = emptyList(), val kind: String = "text",
    val min: Double? = null, val max: Double? = null,
) {
    fun error(value: String): String? {
        if (kind == "boolean") return if (value in listOf("true", "false")) null else "$label を選択してください。"
        if (choices.isNotEmpty()) return if (value in choices) null else "$label の選択値が無効です。"
        if (value.isBlank()) return null
        if (kind == "number" || kind == "integer") {
            val number = value.toDoubleOrNull()
            if (number == null || !number.isFinite() || (min != null && number < min) || (max != null && number > max) ||
                (kind == "integer" && number % 1.0 != 0.0)) return "$label の数値を確認してください。"
        }
        return null
    }
}

private fun choice(key: String, label: String, default: String, vararg values: String) = GenerationOption(key, label, default, values.toList())
private fun number(key: String, label: String, default: String = "", min: Double? = null, max: Double? = null, integer: Boolean = false) =
    GenerationOption(key, label, default, kind = if (integer) "integer" else "number", min = min, max = max)
private fun toggle(key: String, label: String, default: Boolean = false) = GenerationOption(key, label, default.toString(), kind = "boolean")

/** Only existing /chat_stream fields are serialized; hidden provider settings never leak to another model. */
fun generationOptions(model: ModelInfo): List<GenerationOption> = buildList {
    val id = model.id.lowercase()
    val gemini = id.startsWith("gemini-")
    val grok = id.startsWith("grok-")
    if (model.mode == "chat" || model.mode == "agent") {
        add(toggle("enable_file_creation", "File", true))
        add(toggle("enable_system_prompt", "SysPrompt"))
        if (gemini) {
            add(toggle("enable_url_context", "URLs"))
            add(choice("thinking_level", "Thinking level", "high", "minimal", "low", "medium", "high"))
            if (id.startsWith("gemini-2.5")) add(number("thinking_budget", "Thinking Budget", "4096", 0.0, 32768.0, true))
            add(choice("safety_setting", "Safety", "default", "default", "none"))
        } else if (model.supports("thinking")) {
            add(choice("reasoning_effort", "Effort", "medium", "none", "low", "medium", "high", "xhigh", "max"))
        }
        if (grok) {
            add(number("xai_temperature", "Temperature", min = 0.0, max = 2.0))
            add(number("xai_top_p", "Top P", min = 0.0, max = 1.0))
            add(number("xai_max_completion_tokens", "Max output", min = 1.0, integer = true))
            add(number("xai_seed", "Seed", integer = true))
            add(number("xai_presence_penalty", "Presence", min = -2.0, max = 2.0))
            add(number("xai_frequency_penalty", "Frequency", min = -2.0, max = 2.0))
            add(GenerationOption("xai_stop", "Stop（カンマ区切り・最大4件）"))
            add(choice("xai_response_format", "Format", "text", "text", "json_object"))
            add(choice("xai_tool_choice", "Tools", "auto", "auto", "none", "required"))
            add(toggle("xai_parallel_tool_calls", "Parallel tools", true))
            add(toggle("xai_logprobs", "Logprobs"))
            add(number("xai_top_logprobs", "Top logs", min = 0.0, max = 8.0, integer = true))
        }
    }
    if (model.mode == "image") {
        when {
            id.startsWith("gpt-image") -> {
                add(choice("image_size", "Size", "1024x1024", "1024x1024", "1536x1024", "1024x1536", "auto"))
                add(choice("image_quality", "Quality", "medium", "low", "medium", "high", "xhigh", "max", "auto"))
                add(choice("image_format", "Format", "jpeg", "jpeg", "png", "webp"))
                add(number("image_compression", "Compression", "85", 0.0, 100.0, true))
            }
            gemini -> {
                add(choice("gemini_image_aspect", "Aspect", "1:1", "1:1", "auto", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9"))
                add(choice("gemini_image_size", "Size", "1K", "1K", "2K", "4K"))
            }
            grok -> {
                add(number("grok_image_count", "Count", "1", 1.0, 10.0, true))
                add(choice("grok_image_aspect", "Aspect", "1:1", "auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "2:1", "1:2", "19.5:9", "9:19.5", "20:9", "9:20"))
                add(choice("grok_image_resolution", "Resolution", "1k", "1k", "2k"))
                add(choice("grok_image_format", "Format", "url", "url", "b64_json"))
            }
        }
    }
    if (model.mode == "video" && grok) {
        add(number("grok_video_duration", "Duration（秒）", "5", 1.0, 15.0, true))
        add(choice("grok_video_aspect", "Aspect", "16:9", "16:9", "4:3", "1:1", "9:16", "3:4", "3:2", "2:3"))
        add(choice("grok_video_resolution", "Resolution", "720p", "720p", "480p", "1080p"))
    }
    if (model.mode == "ocr") {
        add(choice("ocr_table_format", "Table format", "", "", "markdown", "html"))
        add(GenerationOption("ocr_pages", "Pages（例: 0,2-4）"))
        add(toggle("ocr_extract_header", "Header"))
        add(toggle("ocr_extract_footer", "Footer"))
        add(toggle("ocr_include_blocks", "Blocks"))
        add(toggle("ocr_include_image_base64", "画像抽出", true))
    }
    if (model.mode == "tts") {
        val voices = when {
            gemini -> listOf("Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar", "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi", "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat")
            grok -> listOf("Eve", "Ara", "Rex", "Sal", "Leo")
            else -> listOf("alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse", "marin", "cedar")
        }
        add(GenerationOption("tts_voice", "Voice", voices.first(), voices))
        add(GenerationOption("tts_voice_custom", "Custom voice（任意）"))
        add(GenerationOption("tts_language", "Language（例: ja-JP）"))
        add(number("tts_speed", "Speed", "1", 0.25, 4.0))
    }
}

fun generationOptionsPayload(model: ModelInfo, values: Map<String, String>): JSONObject = JSONObject().apply {
    generationOptions(model).forEach { option ->
        val value = values[option.key] ?: option.defaultValue
        require(option.error(value) == null) { option.error(value).orEmpty() }
        if (option.kind == "boolean") put(option.key, value.toBooleanStrict())
        else if (value.isNotBlank()) put(option.key, value)
    }
}
