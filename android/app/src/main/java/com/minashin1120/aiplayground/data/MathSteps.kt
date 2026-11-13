package com.minashin1120.aiplayground.data

import org.json.JSONObject

/**
 * Per-user「数学ステップ解説」settings (server `math_steps_config`). Values and defaults follow the
 * shared contract: basics OFF, level easy, step size normal, diagrams auto, animation ON.
 */
data class MathStepsConfig(
    val basics: Boolean = false,
    val level: String = "easy",
    val stepSize: String = "normal",
    val diagrams: String = "auto",
    val animation: Boolean = true,
) {
    fun toJson(): JSONObject = JSONObject()
        .put("basics", basics)
        .put("level", level)
        .put("step_size", stepSize)
        .put("diagrams", diagrams)
        .put("animation", animation)

    /** Filled values for the serverless prompt placeholders (same wording as the server). */
    fun promptValues(): Map<String, String> = mapOf(
        "math_basics" to if (basics) {
            "ON。最初のステップで、この問題に必要な前提の定義・公式を、小さな具体例つきで確認する。その後に問題を解き始める。"
        } else {
            "OFF。前提の復習は最小限にし、問題の整理から始める。"
        },
        "math_level" to when (level) {
            "detailed" -> "式の意味と根拠を丁寧に。ただし1ステップ1話題は守る。"
            "normal" -> "高校生くらいの標準的な説明。"
            else -> "小学高学年でも追えるやさしさ。言い換えを多めに。"
        },
        "math_step_size" to when (stepSize) {
            "small" -> "小さめ（1ステップは短く、式変形を細かく分ける）"
            "large" -> "大きめ（関連する計算をまとめるが、途中式は省略しない）"
            else -> "標準（1ステップ1つの考え方）"
        },
        "math_diagrams" to if (diagrams == "off") "図は使わない（mathsvg を出力しない）" else "必要なときだけ使う（最大2個）",
        "math_animation" to if (animation) "使ってよい" else "使わない（静止画のみ）",
    )

    companion object {
        val LEVELS = listOf("easy", "normal", "detailed")
        val STEP_SIZES = listOf("small", "normal", "large")
        val DIAGRAMS = listOf("auto", "off")

        /** Missing or unknown values fall back to the defaults instead of the raw server value. */
        fun parse(raw: Any?): MathStepsConfig {
            val json = when (raw) {
                is JSONObject -> raw
                is String -> runCatching { JSONObject(raw) }.getOrNull()
                else -> null
            } ?: return MathStepsConfig()
            return MathStepsConfig(
                basics = json.optBoolean("basics", false),
                level = json.optString("level").takeIf { it in LEVELS } ?: "easy",
                stepSize = json.optString("step_size").takeIf { it in STEP_SIZES } ?: "normal",
                diagrams = json.optString("diagrams").takeIf { it in DIAGRAMS } ?: "auto",
                animation = if (json.has("animation")) json.optBoolean("animation", true) else true,
            )
        }
    }
}

/** `math_steps_config` of the mobile preferences payload. */
fun JSONObject.mathStepsConfig(): MathStepsConfig = MathStepsConfig.parse(opt("math_steps_config"))

/**
 * Per-message step progress (server `math_steps_state`):
 * `{"current": 0, "max_reached": 0, "simplified": {"2": "..."}}`.
 * `current` is 0-based; `max_reached` never decreases.
 */
data class MathStepsState(
    val current: Int = 0,
    val maxReached: Int = 0,
    val simplified: Map<Int, String> = emptyMap(),
) {
    fun withCurrent(index: Int): MathStepsState =
        copy(current = index.coerceIn(0, MAX_STEP), maxReached = maxOf(maxReached, index.coerceIn(0, MAX_STEP)))

    fun withSimplified(index: Int, text: String): MathStepsState =
        copy(simplified = simplified + (index to text.take(MAX_SIMPLIFIED_CHARS)))

    fun toJson(): JSONObject = JSONObject()
        .put("current", current)
        .put("max_reached", maxReached)
        .put("simplified", JSONObject().apply {
            var budget = MAX_SIMPLIFIED_TOTAL
            simplified.toSortedMap().forEach { (index, text) ->
                if (budget <= 0) return@forEach
                val value = text.take(minOf(MAX_SIMPLIFIED_CHARS, budget))
                put(index.toString(), value)
                budget -= value.length
            }
        })

    companion object {
        const val MAX_STEP = 50
        const val MAX_SIMPLIFIED_CHARS = 20_000
        const val MAX_SIMPLIFIED_TOTAL = 100_000

        /** Parses the message payload field; null when the server has no state for this message. */
        fun parse(raw: Any?): MathStepsState? {
            val json = when (raw) {
                is JSONObject -> raw.takeIf { it !== JSONObject.NULL }
                is String -> runCatching { JSONObject(raw) }.getOrNull()
                else -> null
            } ?: return null
            val current = json.optInt("current", 0).coerceIn(0, MAX_STEP)
            val maxReached = maxOf(json.optInt("max_reached", current).coerceIn(0, MAX_STEP), current)
            var budget = MAX_SIMPLIFIED_TOTAL
            val simplified = LinkedHashMap<Int, String>()
            val rows = json.optJSONObject("simplified")
            if (rows != null) {
                rows.keys().asSequence().sortedBy { it.toIntOrNull() ?: Int.MAX_VALUE }.forEach { key ->
                    val index = key.toIntOrNull() ?: return@forEach
                    if (index < 0 || index > MAX_STEP || budget <= 0) return@forEach
                    val text = rows.optString(key)
                    if (text.isBlank() || text == "null") return@forEach
                    val value = text.take(minOf(MAX_SIMPLIFIED_CHARS, budget))
                    simplified[index] = value
                    budget -= value.length
                }
            }
            return MathStepsState(current, maxReached, simplified)
        }
    }
}

/** One step of an answer: the heading title and the Markdown body below it. */
data class MathStep(val index: Int, val number: Int, val title: String, val body: String)

private val STEP_HEADING = Regex(
    """^\s{0,3}#{1,6}\s*(?:ステップ|Step)\s*(\d+)\s*(.*)$""",
    RegexOption.IGNORE_CASE,
)
private val FENCE_LINE = Regex("""^\s{0,3}(`{3,}|~{3,})""")

/**
 * Splits an answer on `## ステップ N` / `## Step N` headings outside fenced code, as the server and
 * the Web part do. A preamble before the first heading is merged into step 1 so a model-written
 * title/intro is not lost; when there is no heading the list is empty and normal Markdown is kept.
 */
fun parseMathSteps(text: String): List<MathStep> {
    if (text.isBlank()) return emptyList()
    val lines = text.replace("\r\n", "\n").replace('\r', '\n').split('\n')
    val steps = ArrayList<MathStep>()
    var fence: String? = null
    var headingNumber = 0
    var title = ""
    val body = StringBuilder()
    val preamble = StringBuilder()

    fun flush() {
        if (headingNumber == 0) return
        val text = body.toString().trim('\n')
        // A model may write a title/intro before the first heading; keep it in step 1.
        val merged = if (steps.isEmpty() && preamble.isNotBlank()) {
            val prefix = preamble.toString().trim()
            if (text.isBlank()) prefix else "$prefix\n\n$text"
        } else text
        steps += MathStep(steps.size, headingNumber, title, merged)
        body.setLength(0)
    }

    for (line in lines) {
        val fenceMatch = FENCE_LINE.find(line)
        if (fence == null && fenceMatch != null) {
            fence = fenceMatch.groupValues[1].take(3)
            if (headingNumber != 0) body.append(line).append('\n') else preamble.append(line).append('\n')
            continue
        }
        if (fence != null) {
            if (headingNumber != 0) body.append(line).append('\n') else preamble.append(line).append('\n')
            if (fenceMatch != null && fenceMatch.groupValues[1].startsWith(fence)) fence = null
            continue
        }
        val heading = STEP_HEADING.matchEntire(line)
        if (heading != null) {
            flush()
            headingNumber = heading.groupValues[1].toIntOrNull()?.coerceIn(1, 999) ?: (steps.size + 1)
            title = heading.groupValues[2].trim()
        } else if (headingNumber != 0) {
            body.append(line).append('\n')
        } else {
            preamble.append(line).append('\n')
        }
    }
    flush()
    return steps
}

private val MATHSVG_FENCE = Regex("""(?s)(`{3,}|~{3,})mathsvg[^\n]*\n.*?\n\1[ \t]*""")

/** Web `prepareMathStepsCopyText`: keep readable headings, replace mathsvg fences with ［図］. */
fun mathStepsCopyText(text: String): String {
    val source = String(text ?: "")
    val withoutSvg = MATHSVG_FENCE.replace(source, "［図］")
    val steps = parseMathSteps(withoutSvg)
    if (steps.isEmpty()) return withoutSvg
    return steps.joinToString("\n\n") { step ->
        val title = step.title.takeIf { it.isNotBlank() }?.let { " $it" }.orEmpty()
        "## ステップ ${step.index + 1}$title\n\n${step.body.trim()}"
    }.trim()
}

/** Android `ThreadPdf`: all steps, preferring the saved simplified body, with ［図］ for mathsvg. */
fun mathStepsPdfText(text: String, state: MathStepsState?): String {
    val steps = parseMathSteps(text)
    if (steps.isEmpty()) return mathStepsCopyText(text)
    return steps.joinToString("\n\n") { step ->
        val title = step.title.takeIf { it.isNotBlank() }?.let { " $it" }.orEmpty()
        val body = state?.simplified?.get(step.index) ?: step.body
        "## ステップ ${step.index + 1}$title\n\n${MATHSVG_FENCE.replace(body, "［図］").trim()}"
    }.trim()
}
