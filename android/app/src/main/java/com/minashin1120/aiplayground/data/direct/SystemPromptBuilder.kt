package com.minashin1120.aiplayground.data.direct

import org.json.JSONArray
import org.json.JSONObject
import java.time.ZonedDateTime
import java.time.format.DateTimeFormatter
import com.minashin1120.aiplayground.data.MathStepsConfig
import com.minashin1120.aiplayground.data.mathStepsConfig

/**
 * Port of the server's system prompt assembly (`background.py` "System Prompt Construction"): the Skill
 * instruction sent by the client, the operator's global prompt (or the current-time notice), the user's
 * own prompt, the chat-specific instructions, Coding Mode, then the automatic notices (Python, marker,
 * MathJax, search). [preferences] is the `/api/mobile/v1/preferences` shape (cached from the server, or
 * the device profile's settings); [defaults] holds the notice texts shipped with the app.
 */
object SystemPromptBuilder {
    data class ThreadSettings(val customInstruction: String, val includeGlobalInstruction: Boolean)

    fun build(
        body: JSONObject,
        preferences: JSONObject,
        thread: ThreadSettings,
        defaults: ServerlessDefaults,
        provider: String,
        now: ZonedDateTime = ZonedDateTime.now(),
        cacheableConversation: Boolean = true,
    ): String {
        val includeGlobal = thread.includeGlobalInstruction
        val applyGlobal = preferences.optBoolean("apply_global_system_prompt", true)
        var globalPrompt: String? = null
        var useTimeNotice = false
        if (applyGlobal && includeGlobal) {
            val effective = preferences.optString("global_system_prompt_effective")
            val timeFallback = preferences.optBoolean("global_system_prompt_uses_time_fallback", !preferences.has("global_system_prompt_effective"))
            if (timeFallback) useTimeNotice = true
            else if (effective.isNotBlank()) globalPrompt = effective
        }
        var userPrompt: String? = null
        if (body.flag("enable_system_prompt") && includeGlobal && preferences.optBoolean("system_prompt_enabled", true)) {
            userPrompt = preferences.optString("system_prompt").takeIf { it.isNotBlank() }
        }
        val local = (if (body.has("thread_custom_instruction")) body.optString("thread_custom_instruction") else thread.customInstruction).trim()
        var combined = listOf(body.optString("system_prompt"), globalPrompt.orEmpty(), userPrompt.orEmpty())
            .filter { it.isNotBlank() }.joinToString("\n\n") { it.trim() }
        if (local.isNotEmpty()) combined = if (combined.isNotEmpty()) "$combined\n\n[Chat Specific Instructions]:\n$local" else local
        if (body.flag("coding_mode") && defaults.codingModePrompt.isNotBlank()) {
            combined = if (combined.isNotEmpty()) "$combined\n\n${defaults.codingModePrompt}" else defaults.codingModePrompt
        }
        if (body.flag("math_steps") && defaults.mathStepsPrompt.isNotBlank()) {
            val math = fillMathStepsPrompt(defaults.mathStepsPrompt, preferences.mathStepsConfig())
            combined = if (combined.isNotEmpty()) "$combined\n\n$math" else math
        }
        val notices = Notices(preferences, defaults)
        if (notices.enabled("skills") && body.optBoolean("enable_skills", true)) {
            val skills = body.optJSONArray("skills")
            if (skills != null && skills.length() > 0) {
                val catalog = buildSkillCatalog(skills)
                if (catalog.isNotBlank()) {
                    val rendered = renderSkillNotice(notices.text("skills"), catalog, skills.length())
                    combined = if (combined.isBlank()) rendered else "$rendered\n\n$combined"
                }
            }
        }
        if (body.flag("enable_python") && notices.enabled("python")) {
            val text = notices.text("python")
            if (combined.isBlank()) combined = text
            else if (!combined.lowercase().contains(text.lowercase())) combined = "$text\n\n$combined"
        }
        if (body.optString("marker_system_prompt").let { it.isNotBlank() && it != "null" } && notices.enabled("marker")) {
            val text = notices.text("marker")
            if (combined.isBlank()) combined = text else if (!combined.contains(text.trim())) combined = "$combined\n\n$text"
        }
        if (useTimeNotice) {
            if (cacheableConversation) {
                if (!body.has("turn_time_notice")) body.put("turn_time_notice", timeNotice(now))
            } else {
                val notice = timeNotice(now)
                combined = if (combined.isBlank()) notice else "$notice\n\n$combined"
            }
        }
        if (notices.enabled("mathjax") && !combined.contains("MathJax")) {
            val text = notices.text("mathjax")
            combined = if (combined.isBlank()) text else "$combined\n\n$text"
        }
        if (body.flag("enable_search")) {
            val key = when (provider) { "xai" -> "grok_search"; "openai" -> "openai_search"; else -> null }
            if (key != null && notices.enabled(key)) {
                val text = notices.text(key)
                combined = if (combined.isBlank()) text else "$text\n\n$combined"
            }
        }
        return combined
    }

    /** Server `_render_attachment_names_notice`: appended to the message when images are attached. */
    fun attachmentNamesBlock(names: List<String>, preferences: JSONObject, defaults: ServerlessDefaults): String {
        val cleaned = names.map { it.substringAfterLast('/').trim() }.filter { it.isNotEmpty() }
        val notices = Notices(preferences, defaults)
        if (cleaned.isEmpty() || !notices.enabled("attachment_names")) return ""
        val block = cleaned.mapIndexed { index, name -> "画像${index + 1}: $name" }.joinToString("\n")
        var rendered = notices.text("attachment_names").replace("\r\n", "\n").trim()
        var replaced = false
        listOf("{{attachment_names}}", "{attachment_names}", "{{attachment_list}}", "{attachment_list}").forEach { token ->
            if (rendered.contains(token)) { rendered = rendered.replace(token, block); replaced = true }
        }
        listOf("{{attachment_count}}", "{attachment_count}").forEach { token ->
            if (rendered.contains(token)) { rendered = rendered.replace(token, cleaned.size.toString()); replaced = true }
        }
        Regex("\\{\\{\\s*(attachment_names|attachment_list)\\s*\\}\\}").let { regex ->
            if (regex.containsMatchIn(rendered)) { rendered = regex.replace(rendered, Regex.escapeReplacement(block)); replaced = true }
        }
        Regex("\\{\\{\\s*attachment_count\\s*\\}\\}").let { regex ->
            if (regex.containsMatchIn(rendered)) { rendered = regex.replace(rendered, cleaned.size.toString()); replaced = true }
        }
        if (!replaced) rendered = when {
            rendered.endsWith(":") || rendered.endsWith("：") || rendered.contains('\n') -> "$rendered\n$block"
            else -> "$rendered:\n$block"
        }
        return rendered.trim()
    }

    /** Server `_render_quote_source_notice`: whose message a quote came from and its position in the conversation. */
    fun quoteSourceBlock(role: String, number: Int?, preferences: JSONObject, defaults: ServerlessDefaults): String {
        val notices = Notices(preferences, defaults)
        if ((role != "user" && role != "assistant") || !notices.enabled("quote_source")) return ""
        val source = if (number == null || number <= 0) role else "$role (message #$number)"
        val rendered = notices.text("quote_source").replace("\r\n", "\n").trim()
        val regex = Regex("\\{\\{\\s*quote_source\\s*\\}\\}|\\{quote_source\\}")
        return (if (regex.containsMatchIn(rendered)) regex.replace(rendered, Regex.escapeReplacement(source)) else "$rendered $source").trim()
    }

    /** Server `skills.py`: name + short description for tool-capable serverless engines. */
    private fun buildSkillCatalog(skills: JSONArray): String =
        (0 until skills.length()).mapNotNull { index ->
            val row = skills.optJSONObject(index) ?: return@mapNotNull null
            val name = row.optString("name").trim()
            if (name.isEmpty()) return@mapNotNull null
            val description = row.optString("description").replace("\r\n", "\n").trim()
                .let { if (it.length > 300) it.take(300).trimEnd() + "…" else it }
            if (description.isEmpty()) "- $name" else "- $name: $description"
        }.joinToString("\n")

    /** Server `render_skill_notice`: expands `{{skills}}` / `{{skill_count}}`. */
    private fun renderSkillNotice(template: String, catalog: String, count: Int): String {
        var rendered = template.replace("\r\n", "\n").trim()
        var replaced = false
        listOf("{{skills}}", "{skills}").forEach { token ->
            if (rendered.contains(token)) { rendered = rendered.replace(token, catalog); replaced = true }
        }
        listOf("{{skill_count}}", "{skill_count}").forEach { token ->
            if (rendered.contains(token)) { rendered = rendered.replace(token, count.toString()); replaced = true }
        }
        return if (!replaced && catalog.isNotBlank()) {
            if (rendered.isEmpty()) catalog else "$rendered\n\n$catalog"
        } else rendered
    }

    /** Server `build_global_system_prompt`. */
    fun timeNotice(now: ZonedDateTime): String =
        "Current time: ${now.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss zzz"))} (UTC${now.format(DateTimeFormatter.ofPattern("xx"))})"

    /** Server `build_math_steps_system_prompt`: fills the five config placeholders. */
    fun fillMathStepsPrompt(template: String, config: MathStepsConfig): String {
        var rendered = template.replace("\r\n", "\n")
        config.promptValues().forEach { (key, value) ->
            val regex = Regex("\\{\\{\\s*${Regex.escape(key)}\\s*\\}\\}|\\{${Regex.escape(key)}\\}")
            rendered = regex.replace(rendered, Regex.escapeReplacement(value))
        }
        // The server template also caps the number of steps (MATH_STEPS_MAX_STEPS).
        rendered = Regex("\\{\\{\\s*math_max_steps\\s*\\}\\}|\\{math_max_steps\\}").replace(rendered, "50")
        return rendered.trim()
    }

    private class Notices(private val preferences: JSONObject, private val defaults: ServerlessDefaults) {
        private val config = preferences.optJSONObject("auto_system_prompt_notices_config")
        private val all = preferences.optBoolean("apply_auto_system_prompt_notices", true)
        fun enabled(key: String): Boolean = all && (key == "mcp" || config?.optJSONObject(key)?.optBoolean("enabled", true) ?: true)
        fun text(key: String): String = config?.optJSONObject(key)?.optString("text")?.trim()?.takeIf { it.isNotEmpty() }
            ?: defaults.noticeText(key)
    }
}

/**
 * Values generated from the server for serverless mode (`assets/serverless-defaults.json`, kept in sync
 * by `android/ci/sync-serverless-defaults.py --check`): the model catalog in the `/api/mobile/v1/me`
 * shape, the automatic notice texts and the Coding Mode prompt.
 */
class ServerlessDefaults(val json: JSONObject) {
    val codingModePrompt: String get() = json.optString("coding_mode_system_prompt")
    /** 「数学ステップ解説」format-critical prompt (added to the asset once the server module exists). */
    val mathStepsPrompt: String get() = json.optString("math_steps_system_prompt")
    fun noticeText(key: String): String =
        json.optJSONObject("auto_system_prompt_notices_config")?.optJSONObject(key)?.optString("text").orEmpty()
    val noticesConfig: JSONObject get() = json.optJSONObject("auto_system_prompt_notices_config") ?: JSONObject()
}
