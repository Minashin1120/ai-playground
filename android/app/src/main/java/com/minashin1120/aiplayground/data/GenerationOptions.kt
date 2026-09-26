package com.minashin1120.aiplayground.data

import org.json.JSONObject
import java.util.Locale

/**
 * One control of a Web generation panel (`composer_gen_image.html` / `composer_gen_media.html`).
 * [options] are (value, label) pairs as in the Web `<select>`; [disabledValues] are `<option disabled>`.
 */
data class GenField(
    val key: String,
    val label: String,
    val kind: Kind,
    val defaultValue: String = "",
    val options: List<Pair<String, String>> = emptyList(),
    val disabledValues: Set<String> = emptySet(),
    val min: Double? = null,
    val max: Double? = null,
    val step: Double = 1.0,
    val integer: Boolean = false,
    val placeholder: String = "",
    val suffix: String = "",
    /** Tailwind input width (`w-16` = 64dp); 0 lets the control size itself. */
    val width: Int = 0,
    /** Hidden by the Web rules but still sent with its value, like a hidden DOM input. */
    val hidden: Boolean = false,
    val enabled: Boolean = true,
    /** Sent to the server (a few Web inputs are display-only). */
    val sent: Boolean = true,
) {
    enum class Kind { Select, Number, Text, Check, Range }

    fun error(value: String): String? {
        if (!sent) return null
        return when (kind) {
            Kind.Check -> if (value in listOf("true", "false")) null else "$label を選択してください。"
            Kind.Select -> if (options.any { it.first == value }) null else "$label の選択値が無効です。"
            Kind.Number, Kind.Range -> {
                if (value.isBlank()) return null
                val number = value.toDoubleOrNull()
                if (number == null || !number.isFinite() || (min != null && number < min) || (max != null && number > max) ||
                    (integer && number % 1.0 != 0.0)) "$label の数値を確認してください。" else null
            }
            Kind.Text -> null
        }
    }
}

/**
 * A Web options panel (`bg-gray-800/70 border rounded-lg text-[10px]`), or the Lyria RealTime studio bar
 * ([studioBar]) and the input-limit notes ([limits]) that sit in the same place.
 */
data class GenPanel(
    val id: String,
    val fields: List<GenField> = emptyList(),
    val note: String = "",
    /** `text-xs` panels (TTS) instead of `text-[10px]`. */
    val large: Boolean = false,
    val studioBar: Boolean = false,
    val limitsTitle: String = "",
    val limits: List<String> = emptyList(),
)

private fun select(key: String, label: String, default: String, vararg options: Pair<String, String>, disabled: Set<String> = emptySet(), hidden: Boolean = false) =
    GenField(key, label, GenField.Kind.Select, default, options.toList(), disabled, hidden = hidden)
private fun plain(vararg values: String) = values.map { it to it }.toTypedArray()
private fun number(key: String, label: String, default: String = "", min: Double? = null, max: Double? = null, step: Double = 1.0,
                   placeholder: String = "", width: Int = 64, suffix: String = "", hidden: Boolean = false, enabled: Boolean = true) =
    GenField(key, label, GenField.Kind.Number, default, min = min, max = max, step = step, integer = step == 1.0,
        placeholder = placeholder, width = width, suffix = suffix, hidden = hidden, enabled = enabled)
private fun check(key: String, label: String, default: Boolean = false, enabled: Boolean = true) =
    GenField(key, label, GenField.Kind.Check, default.toString(), enabled = enabled)

private val VIDEO_ASPECTS = plain("16:9", "4:3", "1:1", "9:16", "3:4", "3:2", "2:3")
private val OPENAI_TTS_VOICES = listOf("alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse", "marin", "cedar")
val GEMINI_TTS_VOICES = listOf(
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe",
    "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
    "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi", "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
)
private val GROK_TTS_VOICES = listOf("Eve", "Ara", "Rex", "Sal", "Leo")

// Web model predicates (p05).
fun isGptImageModel(m: String) = m.contains("gpt-image")
fun isGeminiImageModel(m: String) = m.contains("gemini") && (m.contains("image") || m.contains("nano"))
fun isGrokImageModel(m: String) = m.contains("grok") && (m.contains("imagine") || m.contains("image")) && !m.contains("video")
fun isGrokVideoModel(m: String) = m.contains("grok") && m.contains("video")
fun isGeminiVideoModel(m: String) = m.startsWith("veo-") || m.contains("omni-flash") || m.contains("omni-1.1-flash")
fun isGeminiMusicModel(m: String) = m.startsWith("lyria-")
fun isLyriaRealtimeModel(m: String) = m == "lyria-realtime-exp"

/** Web `getTtsProvider`. */
fun ttsProvider(model: String): String? {
    val m = model.lowercase(Locale.ROOT)
    return when {
        m.contains("google-tts") -> "google"
        m.contains("gemini") && m.contains("tts") -> "gemini"
        m.contains("grok-tts") || m.contains("xai-tts") -> "xai"
        m.contains("tts") -> "openai"
        else -> null
    }
}

/** Web `updateTtsUi` defaults: the voice and language a provider starts with. */
fun ttsDefaults(provider: String): Pair<String, String> = when (provider) {
    "gemini" -> "Kore" to ""
    "google" -> "auto" to "ja-JP"
    "xai" -> "Eve" to "ja"
    else -> "alloy" to ""
}

/**
 * The Web panels shown for [modelId] with the Web `update*Ui` rules applied (`p07`). [values] is the
 * shared panel state; like the Web DOM inputs, values carry over between models that use the same panel.
 */
fun generationPanels(modelId: String, values: Map<String, String> = emptyMap()): List<GenPanel> = buildList {
    val m = modelId.lowercase(Locale.ROOT)
    if (isGptImageModel(m)) {
        val png = (values["image_format"] ?: "jpeg") == "png"
        add(GenPanel("gpt-image", listOf(
            select("image_size", "Size", "1024x1024", "1024x1024" to "1024x1024", "1536x1024" to "1536x1024", "1024x1536" to "1024x1536", "auto" to "Auto"),
            select("image_quality", "Quality", "medium", "low" to "Low", "medium" to "Medium", "high" to "High", "xhigh" to "X-High", "max" to "Max", "auto" to "Auto"),
            select("image_format", "Format", "jpeg", "jpeg" to "JPEG", "png" to "PNG", "webp" to "WebP"),
            number("image_compression", "Compression", "85", 0.0, 100.0, hidden = png),
        ), "※ GPT-Image のみ有効"))
    }
    if (isGeminiImageModel(m)) {
        val lite = m.contains("gemini-3.1-flash-lite-image")
        add(GenPanel("gemini-image", listOf(
            select("gemini_image_aspect", "Aspect", "1:1", "1:1" to "1:1", "auto" to "Auto", *plain("1:4", "1:8", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9")),
            select("gemini_image_size", "Size", "1K", *plain("1K", "2K", "4K"), disabled = if (lite) setOf("2K", "4K") else emptySet()),
        ), "※ Nano Banana 系のみ有効"))
    }
    if (isGrokImageModel(m)) {
        val resolution = m == "grok-imagine-image-quality" || m == "grok-imagine-image-2.0"
        add(GenPanel("grok-image", listOf(
            number("grok_image_count", "Count", "1", 1.0, 10.0, width = 48),
            select("grok_image_aspect", "Aspect", "1:1", "auto" to "Auto", "1:1" to "1:1 (Square)", "16:9" to "16:9 (Landscape)", "9:16" to "9:16 (Portrait)",
                *plain("4:3", "3:4", "3:2", "2:3", "2:1", "1:2", "19.5:9", "9:19.5", "20:9", "9:20")),
            select("grok_image_resolution", "Res", "1k", "1k" to "1K", "2k" to "2K", hidden = !resolution),
            select("grok_image_quality", "Quality", "medium", "medium" to "Medium", "low" to "Low", hidden = m != "grok-imagine-image-2.0"),
            select("grok_image_format", "Format", "url", "url" to "URL", "b64_json" to "Base64"),
        ), "※ Grok Imagine"))
    }
    if (m.startsWith("grok-") && !isGrokImageModel(m) && !isGrokVideoModel(m) && !m.contains("voice")) {
        val noLogprobs = m.contains("grok-4.20")
        add(GenPanel("xai-chat", listOf(
            number("xai_temperature", "Temp", "", 0.0, 2.0, 0.1, "default"),
            number("xai_top_p", "Top P", "", 0.0, 1.0, 0.05, "default"),
            number("xai_max_completion_tokens", "Max output", "", 1.0, null, placeholder = "default", width = 80),
            number("xai_seed", "Seed", "", null, null, placeholder = "default", width = 80),
            number("xai_presence_penalty", "Presence", "", -2.0, 2.0, 0.1, "default"),
            number("xai_frequency_penalty", "Frequency", "", -2.0, 2.0, 0.1, "default"),
            GenField("xai_stop", "Stop", GenField.Kind.Text, placeholder = "comma separated (max 4)", width = 160),
            select("xai_response_format", "Format", "text", "text" to "Text", "json_object" to "JSON object"),
            select("xai_tool_choice", "Tools", "auto", "auto" to "Auto", "none" to "None", "required" to "Required"),
            check("xai_parallel_tool_calls", "Parallel tools", true),
            check("xai_logprobs", "Logprobs", enabled = !noLogprobs),
            number("xai_top_logprobs", "Top logs", "", 0.0, 8.0, placeholder = "0-8", width = 56, enabled = !noLogprobs),
        ), "※ xAI Chat Completions"))
    }
    if (isGrokVideoModel(m)) {
        add(GenPanel("grok-video", listOf(
            number("grok_video_duration", "Duration", "5", 1.0, 15.0, width = 48, suffix = "sec"),
            select("grok_video_aspect", "Aspect", "16:9", *VIDEO_ASPECTS),
            select("grok_video_resolution", "Resolution", "720p", *plain("720p", "480p", "1080p"),
                disabled = if (m != "grok-imagine-video-1.5") setOf("1080p") else emptySet()),
        ), "※ Grok Imagine Video"))
    }
    if (isGeminiVideoModel(m)) {
        val no4k = m == "veo-3.1-lite-generate-preview" || m == "veo-3.1-fast-generate-preview" || m == "gemini-omni-flash"
        add(GenPanel("gemini-video", listOf(
            // Gemini Omni 1.1 Flash makes fixed-length clips, so the Web hides the duration input.
            number("gemini_video_duration", "Duration", "8", 1.0, 12.0, width = 48, suffix = "sec", hidden = m == "gemini-omni-1.1-flash"),
            select("gemini_video_aspect", "Aspect", "16:9", *VIDEO_ASPECTS),
            select("gemini_video_resolution", "Resolution", "720p", *plain("720p", "1080p", "4K"), disabled = if (no4k) setOf("4K") else emptySet()),
        ), "※ 4Kは Veo 3.1 / Gemini Omni 1.1 Flash のみ対応"))
    }
    if (isGeminiMusicModel(m) && !isLyriaRealtimeModel(m)) {
        add(GenPanel("gemini-music", listOf(check("music_instrumental", "インストゥルメンタル（歌詞なし）")),
            "※ Lyria 3.5 / Lyria 3（歌詞・楽曲構造テキストも返却）"))
    }
    if (isLyriaRealtimeModel(m)) add(GenPanel("lyria-realtime", studioBar = true))
    if (isMistralOcrModel(m)) {
        add(GenPanel("mistral-ocr", listOf(
            select("ocr_table_format", "Table", "", "" to "本文内", "markdown" to "Markdown", "html" to "HTML"),
            GenField("ocr_pages", "Pages", GenField.Kind.Text, placeholder = "0-2 / 0,2-4", width = 112),
            check("ocr_extract_header", "Header"),
            check("ocr_extract_footer", "Footer"),
            check("ocr_include_blocks", "Blocks"),
            check("ocr_include_image_base64", "画像抽出", true),
        ), "※ 履歴は送信しません"))
    }
    imageInputLimits(m)?.let { (title, lines) -> add(GenPanel("image-input-limits", limitsTitle = title, limits = lines)) }
    ttsProvider(m)?.let { provider -> add(ttsPanel(provider, values)) }
}

private fun ttsPanel(provider: String, values: Map<String, String>): GenPanel {
    val (defaultVoice, defaultLanguage) = ttsDefaults(provider)
    val voices = when (provider) {
        "gemini" -> GEMINI_TTS_VOICES.map { it to it }
        "google" -> listOf("auto" to "Auto (Studio/Neural2)", "custom" to "Custom Voice Name")
        "xai" -> GROK_TTS_VOICES.map { it to it }
        else -> OPENAI_TTS_VOICES.map { it to it }
    }
    val customShown = provider == "xai" || (provider == "google" && values["tts_voice"] == "custom")
    val (min, max) = when (provider) {
        "google" -> 0.25 to 2.0
        "xai" -> 0.7 to 1.5
        else -> 0.25 to 4.0
    }
    return GenPanel("audio-gen", listOf(
        GenField("tts_voice", "Voice", GenField.Kind.Select, defaultVoice, voices),
        GenField("tts_voice_custom", "Custom Voice", GenField.Kind.Text, placeholder = "e.g. en-US-Wavenet-D", width = 192, hidden = !customShown),
        GenField("tts_language", "Lang", GenField.Kind.Text, defaultLanguage, placeholder = "ja-JP", width = 80, hidden = provider != "google" && provider != "xai"),
        GenField("tts_speed", "Speed", GenField.Kind.Range, "1", min = min, max = max, step = 0.05, enabled = provider != "gemini"),
    ), note = when (provider) {
        "gemini" -> "(Gemini TTSは速度変更非対応)"
        "xai" -> "xAI TTS supports speed 0.7–1.5 and speech tags"
        else -> ""
    }, large = true)
}

/** Web `updateImageInputLimits`: the note shown under the panels for image-capable models. */
fun imageInputLimits(model: String): Pair<String, List<String>>? {
    val m = model.lowercase(Locale.ROOT)
    return when {
        m.contains("gpt-image") -> "GPT-Image 入力制限" to listOf(
            "最大 16 枚 / 画像1枚あたり 50MB 未満 / PNG・JPG・WEBP", "マスク使用時: PNGのみ、4MB未満、元画像と同サイズ")
        m == "deepseek-v4.1-flash" || m == "deepseek-v4-flash-vision-exp" -> "DeepSeek V4.1 Flash 入力制限" to listOf(
            "JPEG・PNG・GIF・WebP / 画像1枚あたり最大32MB / リクエスト合計48MB", "画像は約800×800相当へ自動リサイズ（1枚あたり最大384トークン）")
        m.contains("deepseek") -> null
        isGeminiImageModel(m) -> when {
            m.contains("gemini-3.1-flash-lite-image") -> "Nano Banana 2 Lite 入力目安" to listOf(
                "画像生成・編集 / 1K出力 / 最大14枚の参照画像に対応", "複数参照や連続編集より、低遅延・大量生成向けです")
            m.contains("gemini-3.1-flash-image") -> "Nano Banana 2 入力目安" to listOf("画像入力は最大3枚程度を推奨（Gemini 3.1 Flash Image）")
            m.contains("gemini-2.5") && m.contains("image") -> "Nano Banana 入力目安" to listOf("画像入力は最大3枚までが推奨")
            else -> "Nano Banana Pro 入力目安" to listOf("高精度は最大5枚 / 合計14枚まで対応")
        }
        isMistralOcrModel(m) -> "Mistral OCR 4 入力" to listOf(
            "PDF / PNG / JPEG / TIFF / BMP / GIF / WEBP / DOCX / PPTX、または公開URL",
            "最大 512MB / 会話履歴は送信しません / チャット補完・Search・Python・Canvas 非対応")
        m.contains("grok") -> "Grok 画像入力制限" to listOf("最大 20MiB / PNG・JPG のみ / 枚数制限なし")
        else -> null
    }
}

/**
 * The value a field sends: an unset field uses its default, a disabled `<option>` falls back like the
 * Web (`1080p` for Veo 4K, `720p` for Grok 1080p, `1K` for Nano Banana 2 Lite), and a disabled input sends
 * nothing (Grok 4.20 logprobs).
 */
fun genFieldValue(field: GenField, values: Map<String, String>): String {
    val stored = values[field.key] ?: field.defaultValue
    // A value from another model's list (an OpenAI voice after switching to Gemini TTS) uses this list's default.
    val raw = if (field.kind == GenField.Kind.Select && field.options.none { it.first == stored }) field.defaultValue else stored
    if (field.kind == GenField.Kind.Select && raw in field.disabledValues) {
        return when (field.key) {
            "gemini_video_resolution" -> "1080p"
            "grok_video_resolution" -> "720p"
            else -> field.defaultValue
        }
    }
    if (!field.enabled && field.kind != GenField.Kind.Range) return if (field.kind == GenField.Kind.Check) "false" else ""
    if (field.kind == GenField.Kind.Range) {
        val number = raw.toDoubleOrNull() ?: 1.0
        return number.coerceIn(field.min ?: number, field.max ?: number).toString()
    }
    return raw
}

/** Only the fields of the panels shown for [modelId] are serialized; other panels never leak into the request. */
fun generationOptionsPayload(modelId: String, values: Map<String, String>): JSONObject = JSONObject().apply {
    generationPanels(modelId, values).flatMap { it.fields }.filter { it.sent }.forEach { field ->
        val value = genFieldValue(field, values)
        field.error(value)?.let { throw IllegalArgumentException(it) }
        when (field.kind) {
            GenField.Kind.Check -> put(field.key, value.toBooleanStrict())
            else -> if (value.isNotBlank()) put(field.key, value)
        }
    }
}
