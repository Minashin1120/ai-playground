package com.minashin1120.aiplayground.data.direct

import kotlinx.coroutines.runBlocking
import mockwebserver3.MockResponse
import mockwebserver3.MockWebServer
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.util.Base64

class MediaDirectTest {
    private val http = DirectHttp(allowedHosts = emptySet(), allowCleartextLocalhost = true)
    private fun request(model: String, prompt: String, options: JSONObject = JSONObject(), attachments: List<DirectAttachment> = emptyList()) =
        DirectRequest(model, "key-1", "SYSTEM", listOf(DirectTurn("user", prompt, attachments)), options)
    private fun run(engine: DirectEngine, req: DirectRequest) = runBlocking { engine.run(req, {}) { _, _ -> } }

    @Test fun routerCoversMediaModels() {
        val router = DirectRouter(http)
        assertEquals("openai", router.route("gpt-image-2", "image")?.provider)
        assertEquals("xai", router.route("grok-imagine-image-2.0", "image")?.provider)
        assertEquals("gemini", router.route("gemini-2.5-flash-preview-tts", "tts")?.provider)
        assertEquals("google", router.route("google-tts-studio", "tts")?.provider)
        assertEquals("xai", router.route("grok-tts", "tts")?.provider)
        assertEquals("openai", router.route("gpt-4o-mini-tts", "tts")?.provider)
        assertEquals("openai", router.route("gpt-transcribe", "transcription")?.provider)
        assertEquals("gemini", router.route("veo-3.1-generate-preview", "video")?.provider)
        assertEquals("xai", router.route("grok-imagine-video", "video")?.provider)
        assertNull(router.route("lyria-3.5", "music"))
        assertNull(router.route("gpt-realtime-2", "realtime_audio"))
    }

    @Test fun gptImageUsesTheImageGenerationToolAndEditsWithImages() {
        val engine = OpenAiImageDirect(http)
        val plain = engine.buildPayload(request("gpt-image-2", "猫", JSONObject().put("image_size", "1024x1536").put("image_quality", "max")))
        val tool = plain.getJSONArray("tools").getJSONObject(0)
        assertEquals("generate", tool.getString("action"))
        assertEquals("1024x1536", tool.getString("size"))
        assertEquals("auto", tool.getString("quality"))
        assertFalse(plain.toString().contains("SYSTEM"))
        val edit = engine.buildPayload(request("gpt-image-2.5-flare", "猫", JSONObject().put("image_quality", "max"),
            listOf(DirectAttachment("a.png", "image/png", bytes = byteArrayOf(1)))))
        assertEquals("edit", edit.getJSONArray("tools").getJSONObject(0).getString("action"))
        assertEquals("max", edit.getJSONArray("tools").getJSONObject(0).getString("quality"))
        assertEquals(2, edit.getJSONArray("input").getJSONObject(0).getJSONArray("content").length())
    }

    @Test fun gptImageReturnsTheGeneratedFile() {
        MockWebServer().use { server ->
            server.start()
            val png = Base64.getEncoder().encodeToString(byteArrayOf(9, 9))
            server.enqueue(MockResponse.Builder().addHeader("Content-Type", "application/json")
                .body("""{"output":[{"type":"image_generation_call","result":"$png"}]}""").build())
            val result = run(OpenAiImageDirect(http, server.url("/").toString().trimEnd('/')), request("gpt-image-2", "猫"))
            assertArrayEquals(byteArrayOf(9, 9), result.files.single().bytes)
            assertEquals("image/png", result.files.single().mime)
            assertEquals("Bearer key-1", server.takeRequest().headers["Authorization"])
        }
    }

    @Test fun grokImagePayloadAndResponse() {
        val payload = XaiImageDirect(http).buildPayload(request("grok-imagine-image-2.0", "犬", JSONObject().put("grok_image_resolution", "2K")))
        assertEquals("2k", payload.getString("resolution"))
        assertEquals("b64_json", payload.getString("response_format"))
        MockWebServer().use { server ->
            server.start()
            server.enqueue(MockResponse.Builder().addHeader("Content-Type", "application/json")
                .body("""{"data":[{"b64_json":"${Base64.getEncoder().encodeToString(byteArrayOf(5))}"}]}""").build())
            val result = run(XaiImageDirect(http, server.url("/").toString().trimEnd('/')), request("grok-imagine-image", "犬"))
            assertArrayEquals(byteArrayOf(5), result.files.single().bytes)
            assertTrue(server.takeRequest().target.endsWith("/v1/images/generations"))
        }
    }

    @Test fun openAiSpeechIsSavedAsMp3() {
        MockWebServer().use { server ->
            server.start()
            server.enqueue(MockResponse.Builder().addHeader("Content-Type", "audio/mpeg").body("ID3data").build())
            val result = run(TtsDirect(http, "openai", server.url("/").toString().trimEnd('/')), request("gpt-4o-mini-tts", "こんにちは"))
            assertEquals("audio/mpeg", result.files.single().mime)
            assertEquals("ID3data", String(result.files.single().bytes))
            assertTrue(server.takeRequest().target.endsWith("/v1/audio/speech"))
        }
    }

    @Test fun pcmIsWrappedAsWav() {
        val wav = pcmToWav(ByteArray(4), 24000)
        assertEquals(48, wav.size)
        assertEquals("RIFF", String(wav.copyOfRange(0, 4)))
        assertEquals("WAVE", String(wav.copyOfRange(8, 12)))
        assertEquals("data", String(wav.copyOfRange(36, 40)))
    }

    @Test fun downloadsFollowRedirectsWithoutTheKey() {
        MockWebServer().use { server ->
            server.start()
            server.enqueue(MockResponse.Builder().code(302).addHeader("Location", "/final.mp4").build())
            server.enqueue(MockResponse.Builder().body("VIDEO").build())
            val bytes = runBlocking { http.download(server.url("/start").toString(), mapOf("x-goog-api-key" to "secret"), 1024) }
            assertEquals("VIDEO", String(bytes))
            assertEquals("secret", server.takeRequest().headers["x-goog-api-key"])
            assertNull(server.takeRequest().headers["x-goog-api-key"])
        }
    }
}
