package com.minashin1120.aiplayground.data.direct

/**
 * Chooses the provider API for a model in serverless mode (server `providers.py` classification).
 * Returns null for models whose generation is not available on the device yet. [bases] overrides a
 * provider's API origin (tests only).
 */
class DirectRouter(private val http: DirectHttp, private val bases: Map<String, String> = emptyMap()) {
    data class Route(val provider: String, val engine: DirectEngine)

    private fun base(provider: String, default: String) = bases[provider] ?: default

    /** Gemini through Vertex AI with a service account kept on the device. */
    fun vertexGemini(target: VertexTarget): DirectEngine = GeminiDirect(http, vertex = target)

    fun vertexAuth(credentialsJson: String): VertexAuth = VertexAuth(http, credentialsJson)

    fun route(modelId: String, mode: String): Route? {
        val id = modelId.lowercase()
        return when {
            id.startsWith("claude") -> Route("anthropic", AnthropicDirect(http, base("anthropic", "https://api.anthropic.com")))
            // Media models (server image / TTS / transcription / video branches).
            mode == "image" && id.startsWith("gpt-image") -> Route("openai", OpenAiImageDirect(http, base("openai", "https://api.openai.com")))
            mode == "image" && id.startsWith("grok-imagine-image") -> Route("xai", XaiImageDirect(http, base("xai", "https://api.x.ai")))
            mode == "tts" && id.startsWith("gemini") -> Route("gemini", TtsDirect(http, "gemini", bases["gemini"]))
            mode == "tts" && id.startsWith("google-tts") -> Route("google", TtsDirect(http, "google", bases["google"]))
            mode == "tts" && id.startsWith("grok") -> Route("xai", TtsDirect(http, "xai", bases["xai"]))
            mode == "tts" && id.startsWith("gpt") -> Route("openai", TtsDirect(http, "openai", bases["openai"]))
            mode == "transcription" && id.startsWith("gpt") -> Route("openai", TranscriptionDirect(http, base("openai", "https://api.openai.com")))
            mode == "video" && id.startsWith("veo-") -> Route("gemini", VideoDirect(http, "gemini", bases["gemini"]))
            mode == "video" && id.startsWith("grok-imagine-video") -> Route("xai", VideoDirect(http, "xai", bases["xai"]))
            id.startsWith("gemini") && (mode == "chat" || mode == "image") ->
                Route("gemini", GeminiDirect(http, base("gemini", "https://generativelanguage.googleapis.com")))
            id.startsWith("grok") && mode == "chat" -> Route("xai", OpenAiResponsesDirect(http, xai = true, baseUrl = base("xai", "https://api.x.ai")))
            id.startsWith("gpt") && id.contains("search") -> Route("openai",
                ChatCompletionsDirect(http, base("openai", "https://api.openai.com") + "/v1", acceptsImages = false, openAiSearch = true))
            (id.startsWith("gpt") || id.startsWith("o1") || id.startsWith("o3") || id.startsWith("o4") || id.startsWith("chatgpt")) && mode == "chat" ->
                Route("openai", OpenAiResponsesDirect(http, baseUrl = base("openai", "https://api.openai.com")))
            id.startsWith("deepseek") && mode == "chat" -> Route("deepseek", ChatCompletionsDirect(http, base("deepseek", "https://api.deepseek.com"), acceptsImages = false))
            id.startsWith("kimi") && mode == "chat" -> Route("kimi", ChatCompletionsDirect(http, base("kimi", "https://api.moonshot.ai") + "/v1", acceptsImages = false))
            id.startsWith("mistral") && mode == "chat" -> Route("mistral", ChatCompletionsDirect(http, base("mistral", "https://api.mistral.ai") + "/v1", acceptsImages = true))
            else -> null
        }
    }
}
