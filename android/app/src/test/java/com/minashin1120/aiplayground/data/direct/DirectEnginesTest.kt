package com.minashin1120.aiplayground.data.direct

import kotlinx.coroutines.runBlocking
import mockwebserver3.MockResponse
import mockwebserver3.MockWebServer
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class DirectEnginesTest {
    private val http = DirectHttp(allowedHosts = emptySet(), allowCleartextLocalhost = true)
    private fun sse(vararg events: String) = MockResponse.Builder().addHeader("Content-Type", "text/event-stream")
        .body(events.joinToString("") { "data: $it\n\n" }).build()
    private fun request(options: JSONObject = JSONObject(), turns: List<DirectTurn> = listOf(DirectTurn("user", "こんにちは"))) =
        DirectRequest("model-x", "secret-key", "system text", turns, options)

    private fun run(engine: DirectEngine, req: DirectRequest): Pair<DirectResult, List<JSONObject>> = runBlocking {
        val events = mutableListOf<JSONObject>()
        val result = engine.run(req, { events += it }) { _, _ -> }
        result to events
    }

    @Test fun geminiStreamsTextThoughtsPythonAndUsage() {
        MockWebServer().use { server ->
            server.start()
            server.enqueue(sse(
                """{"candidates":[{"content":{"parts":[{"text":"考え","thought":true}]}}]}""",
                """{"candidates":[{"content":{"parts":[{"executableCode":{"code":"print(1)"}},{"codeExecutionResult":{"output":"1"}}]}}]}""",
                """{"candidates":[{"content":{"parts":[{"text":"答え","thoughtSignature":"sig1"}]}}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"thoughtsTokenCount":2}}""",
            ))
            val engine = GeminiDirect(http, server.url("/").toString().trimEnd('/'))
            val (result, events) = run(engine, request(JSONObject().put("enable_thinking", true).put("thinking_level", "low").put("enable_search", true)))
            assertTrue(result.content.endsWith("答え"))
            assertTrue(result.content.contains("```python\nprint(1)"))
            assertEquals("考え", result.thought)
            assertEquals(listOf("sig1"), result.thoughtSignatures)
            assertEquals(10, result.tokensIn)
            assertEquals(7, result.tokensOut)
            assertTrue(events.any { it.optString("type") == "python" && it.getJSONObject("content").optString("output") == "1" })
            val sent = server.takeRequest()
            assertEquals("secret-key", sent.headers["x-goog-api-key"])
            assertTrue(sent.target.contains("/v1beta/models/model-x:streamGenerateContent?alt=sse"))
            val body = engine.buildPayload(request(JSONObject().put("enable_thinking", true).put("thinking_level", "low").put("enable_search", true)))
            assertEquals("system text", body.getJSONObject("systemInstruction").getJSONArray("parts").getJSONObject(0).getString("text"))
            assertEquals("LOW", body.getJSONObject("generationConfig").getJSONObject("thinkingConfig").getString("thinkingLevel"))
            assertTrue(body.getJSONArray("tools").toString().contains("google_search"))
        }
    }

    @Test fun geminiSendsImagesInlineAndHistoryRoles() {
        val turns = listOf(
            DirectTurn("user", "前", listOf(DirectAttachment("a.png", "image/png", bytes = byteArrayOf(1, 2)))),
            DirectTurn("assistant", "返答"),
            DirectTurn("user", "次", listOf(DirectAttachment("n.txt", "text/plain", text = "中身"))),
        )
        val body = GeminiDirect(http).buildPayload(request(turns = turns))
        val contents = body.getJSONArray("contents")
        assertEquals("model", contents.getJSONObject(1).getString("role"))
        assertEquals("AQI=", contents.getJSONObject(0).getJSONArray("parts").getJSONObject(0).getJSONObject("inlineData").getString("data"))
        assertTrue(contents.getJSONObject(2).getJSONArray("parts").getJSONObject(0).getString("text").contains("中身"))
    }

    @Test fun geminiUploadsVideosThroughFilesApiInsteadOfInline() {
        MockWebServer().use { server ->
            server.start()
            val json = { body: String -> MockResponse.Builder().addHeader("Content-Type", "application/json").body(body).build() }
            server.enqueue(MockResponse.Builder().addHeader("X-Goog-Upload-URL", server.url("/upload-session?upload_id=1").toString()).body("").build())
            server.enqueue(json("""{"file":{"name":"files/v1","uri":"https://example.invalid/files/v1","mimeType":"video/mp4","state":"PROCESSING"}}"""))
            server.enqueue(json("""{"name":"files/v1","uri":"https://example.invalid/files/v1","mimeType":"video/mp4","state":"ACTIVE"}"""))
            server.enqueue(sse("""{"candidates":[{"content":{"parts":[{"text":"動画の内容"}]}}]}"""))
            val engine = GeminiDirect(http, server.url("/").toString().trimEnd('/'))
            val turns = listOf(DirectTurn("user", "これは？", listOf(DirectAttachment("clip.mp4", "video/mp4", bytes = byteArrayOf(9, 8, 7)))))
            val (result, events) = run(engine, request(turns = turns))
            assertEquals("動画の内容", result.content)
            assertTrue(events.any { it.optString("content") == "動画をGeminiへアップロード中..." })
            val start = server.takeRequest()
            assertTrue(start.target.startsWith("/upload/v1beta/files"))
            assertEquals("start", start.headers["X-Goog-Upload-Command"])
            assertEquals("3", start.headers["X-Goog-Upload-Header-Content-Length"])
            val upload = server.takeRequest()
            assertEquals("upload, finalize", upload.headers["X-Goog-Upload-Command"])
            assertEquals(3L, upload.bodySize)
            assertTrue(server.takeRequest().target.endsWith("/v1beta/files/v1"))
            val body = JSONObject(server.takeRequest().body!!.utf8())
            val parts = body.getJSONArray("contents").getJSONObject(0).getJSONArray("parts")
            assertEquals("https://example.invalid/files/v1", parts.getJSONObject(0).getJSONObject("fileData").getString("fileUri"))
            assertFalse(body.toString().contains("inlineData"))
        }
    }

    @Test fun gpt6AstraAndSolMapUnsupportedNoneToMedium() {
        val engine = OpenAiResponsesDirect(http, baseUrl = "https://api.openai.com")
        val astra = DirectRequest("gpt-6-astra", "sk-test", "", listOf(DirectTurn("user", "hi")),
            JSONObject().put("reasoning_effort", "none").put("enable_thinking", true))
        assertEquals("medium", engine.buildPayload(astra).getJSONObject("reasoning").getString("effort"))
        val sol = DirectRequest("gpt-6.1-sol", "sk-test", "", listOf(DirectTurn("user", "hi")),
            JSONObject().put("reasoning_effort", "max").put("enable_thinking", true))
        assertEquals("max", engine.buildPayload(sol).getJSONObject("reasoning").getString("effort"))
        val luna = DirectRequest("gpt-6-luna", "sk-test", "", listOf(DirectTurn("user", "hi")),
            JSONObject().put("reasoning_effort", "none").put("enable_thinking", true))
        assertEquals("none", engine.buildPayload(luna).getJSONObject("reasoning").getString("effort"))
    }

    @Test fun openAiResponsesStreamsAndBuildsReasoning() {
        MockWebServer().use { server ->
            server.start()
            server.enqueue(MockResponse.Builder().addHeader("Content-Type", "text/event-stream").body(
                "event: response.reasoning_summary_text.delta\ndata: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"思考\"}\n\n" +
                "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello\"}\n\n" +
                "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":4,\"output_tokens_details\":{\"reasoning_tokens\":1}}}}\n\n"
            ).build())
            val engine = OpenAiResponsesDirect(http, baseUrl = server.url("/").toString().trimEnd('/'))
            val req = DirectRequest("gpt-5.5", "sk-test", "sys", listOf(DirectTurn("user", "hi")), JSONObject().put("reasoning_effort", "high").put("enable_search", true))
            val (result, _) = run(engine, req)
            assertEquals("Hello", result.content)
            assertEquals("思考", result.thought)
            assertEquals(3, result.tokensIn)
            val sent = server.takeRequest()
            assertEquals("Bearer sk-test", sent.headers["Authorization"])
            val body = engine.buildPayload(req)
            assertEquals("high", body.getJSONObject("reasoning").getString("effort"))
            assertEquals("sys", body.getString("instructions"))
            assertFalse(body.getBoolean("store"))
            assertEquals("web_search", body.getJSONArray("tools").getJSONObject(0).getString("type"))
        }
    }

    @Test fun providerErrorsCarryTheProviderMessage() {
        MockWebServer().use { server ->
            server.start()
            server.enqueue(MockResponse.Builder().code(401).addHeader("Content-Type", "application/json")
                .body("""{"error":{"message":"Incorrect API key provided"}}""").build())
            val engine = OpenAiResponsesDirect(http, baseUrl = server.url("/").toString().trimEnd('/'))
            val failure = runCatching { run(engine, request()) }.exceptionOrNull()
            assertTrue(failure is DirectApiException)
            assertEquals(401, (failure as DirectApiException).status)
            assertEquals("Incorrect API key provided", failure.message)
        }
    }

    @Test fun anthropicStreamsThinkingAndText() {
        MockWebServer().use { server ->
            server.start()
            server.enqueue(sse(
                """{"type":"message_start","message":{"usage":{"input_tokens":12}}}""",
                """{"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"ふむ"}}""",
                """{"type":"content_block_delta","delta":{"type":"text_delta","text":"結論"}}""",
                """{"type":"message_delta","usage":{"output_tokens":6}}""",
                """{"type":"message_stop"}""",
            ))
            val engine = AnthropicDirect(http, server.url("/").toString().trimEnd('/'))
            val (result, _) = run(engine, request(JSONObject().put("enable_thinking", true).put("thinking_budget", "100")))
            assertEquals("結論", result.content)
            assertEquals("ふむ", result.thought)
            assertEquals(12, result.tokensIn)
            assertEquals(6, result.tokensOut)
            val sent = server.takeRequest()
            assertEquals("secret-key", sent.headers["x-api-key"])
            assertEquals("true", sent.headers["anthropic-dangerous-direct-browser-access"])
            val body = engine.buildPayload(request(JSONObject().put("enable_thinking", true).put("thinking_budget", "100")))
            assertEquals(1024, body.getJSONObject("thinking").getInt("budget_tokens"))
            assertEquals("system text", body.getString("system"))
        }
    }

    @Test fun chatCompletionsReadsReasoningContentAndUsage() {
        MockWebServer().use { server ->
            server.start()
            server.enqueue(sse(
                """{"choices":[{"delta":{"reasoning_content":"r"}}]}""",
                """{"choices":[{"delta":{"content":"ok"}}]}""",
                """{"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":3}}""",
                "[DONE]",
            ))
            val engine = ChatCompletionsDirect(http, server.url("/v1").toString(), acceptsImages = false)
            val turns = listOf(DirectTurn("user", "画像", listOf(DirectAttachment("x.png", "image/png", bytes = byteArrayOf(1)))))
            val (result, _) = run(engine, request(turns = turns))
            assertEquals("ok", result.content)
            assertEquals("r", result.thought)
            assertEquals(3, result.tokensOut)
            val body = engine.buildPayload(request(turns = turns))
            assertEquals("system", body.getJSONArray("messages").getJSONObject(0).getString("role"))
            assertTrue(body.getJSONArray("messages").getJSONObject(1).getString("content").contains("読み取れない形式"))
        }
    }

    @Test fun onlyProviderHostsAreReachable() {
        val strict = DirectHttp()
        assertTrue(runCatching { strict.request("https://evil.example/v1", emptyMap()) }.isFailure)
        assertTrue(runCatching { strict.request("http://api.openai.com/v1", emptyMap()) }.isFailure)
        assertTrue(runCatching { strict.request("https://api.openai.com/v1/responses", emptyMap()) }.isSuccess)
    }

    @Test fun routerPicksTheProviderApi() {
        val router = DirectRouter(http)
        assertEquals("gemini", router.route("gemini-3.6-flash", "chat")?.provider)
        assertEquals("gemini", router.route("gemini-3-pro-image", "image")?.provider)
        assertEquals("anthropic", router.route("claude-opus-4-6", "chat")?.provider)
        assertEquals("xai", router.route("grok-4.6", "chat")?.provider)
        assertEquals("openai", router.route("gpt-5-search-api", "chat")?.provider)
        assertEquals("deepseek", router.route("deepseek-v4-pro", "chat")?.provider)
        // Media models run on the device since 1.40.0; music and realtime sessions still do not.
        assertEquals("gemini", router.route("veo-3.1-generate-preview", "video")?.provider)
        assertEquals("openai", router.route("gpt-image-2", "image")?.provider)
        assertNull(router.route("lyria-3.5", "music"))
    }
}
