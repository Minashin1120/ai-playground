package com.minashin1120.aiplayground.data

import kotlinx.coroutines.runBlocking
import mockwebserver3.MockResponse
import mockwebserver3.MockWebServer
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.IOException

class PlaygroundApiTest {
    @Test fun bearerDoesNotBecomeBrowserCookie() = runBlocking {
        MockWebServer().use { server ->
            server.start()
            server.enqueue(MockResponse.Builder().addHeader("Content-Type", "application/json")
                .addHeader("Set-Cookie", "session=browser; Path=/").body("{}").build())
            server.enqueue(MockResponse.Builder().addHeader("Content-Type", "application/json").body("{}").build())
            val api = PlaygroundApi(server.url("/"))
            api.get("/api/mobile/v1/me", "test-token")
            api.post("/api/mobile/v1/device", JSONObject())
            val first = server.takeRequest()
            assertEquals("Bearer test-token", first.headers["Authorization"])
            assertNull(first.headers["Cookie"])
            val second = server.takeRequest()
            assertNull(second.headers["Authorization"])
            assertNull(second.headers["Cookie"])
        }
    }

    @Test fun redirectsAreNotFollowed() = runBlocking {
        MockWebServer().use { server ->
            server.start()
            server.enqueue(MockResponse.Builder().code(302).addHeader("Location", "https://example.org/").build())
            val failure = runCatching { PlaygroundApi(server.url("/")).get("/api/threads", "test") }.exceptionOrNull()
            assertTrue(failure is ApiException)
            assertEquals(302, (failure as ApiException).status)
            assertEquals(1, server.requestCount)
        }
    }

    @Test fun phaseTwoUpdatesUseBearerPut() = runBlocking {
        MockWebServer().use { server ->
            server.start()
            server.enqueue(MockResponse.Builder().addHeader("Content-Type", "application/json")
                .body("{\"status\":\"ok\"}").build())
            PlaygroundApi(server.url("/")).put("/api/threads/t1/settings", JSONObject().put("is_temporary", true), "test-token")
            val request = server.takeRequest()
            assertEquals("PUT", request.method)
            assertEquals("Bearer test-token", request.headers["Authorization"])
            assertNull(request.headers["Cookie"])
        }
    }

    @Test fun streamRequiresTerminalEvent() = runBlocking {
        MockWebServer().use { server ->
            server.start()
            val api = PlaygroundApi(server.url("/"))
            server.enqueue(MockResponse.Builder().addHeader("Content-Type", "application/x-ndjson")
                .body("{\"type\":\"content\",\"content\":\"こんにちは\"}\n{\"type\":\"done\"}\n").build())
            val events = mutableListOf<String>()
            api.stream("/chat_stream", JSONObject(), "test") { events.add(it.getString("type")) }
            assertEquals(listOf("content", "done"), events)
            server.enqueue(MockResponse.Builder().addHeader("Content-Type", "application/x-ndjson")
                .body("{\"type\":\"content\"}\n").build())
            assertTrue(runCatching { api.stream("/chat_stream", JSONObject(), "test") {} }.exceptionOrNull() is IOException)
        }
    }

    @Test fun retryAfterIsPreserved() = runBlocking {
        MockWebServer().use { server ->
            server.start()
            server.enqueue(MockResponse.Builder().code(429).addHeader("Content-Type", "application/json")
                .addHeader("Retry-After", "17").body("{\"error\":\"slow_down\"}").build())
            val failure = runCatching { PlaygroundApi(server.url("/")).get("/api/threads", "test") }.exceptionOrNull()
            assertEquals(17L, (failure as ApiException).retryAfter)
        }
    }

    @Test fun externalPathsAreRejected() {
        MockWebServer().use { server ->
            server.start()
            val api = PlaygroundApi(server.url("/"))
            for (path in listOf("https://example.org/", "//example.org/")) {
                assertTrue(runCatching { api.url(path) }.exceptionOrNull() is IllegalArgumentException)
            }
        }
    }

    @Test fun responseSizeIsBounded() {
        assertEquals("日本語", readBoundedUtf8(ByteArrayInputStream("日本語".toByteArray()), 9))
        assertTrue(runCatching { readBoundedUtf8(ByteArrayInputStream(ByteArray(10)), 9) }.exceptionOrNull() is IOException)
    }

    @Test fun historyParsesLegacyAndMultipleAttachments() {
        val history = JSONObject("""{"messages":[{"id":1,"role":"user","content":null,"image_url":"one.txt"},{"id":2,"role":"assistant","content":"回答","image_url":"[\"a.png\",\"b.png\"]"}]}""")
        val rows = parseMessages(history)
        assertEquals("", rows[0].content)
        assertEquals(listOf("one.txt"), rows[0].files)
        assertEquals(listOf("a.png", "b.png"), rows[1].files)
    }

    @Test fun threadMetadataParsesPhaseTwoState() {
        val thread = parseThreadItem(JSONObject("""{"id":"t1","title":"Pinned","last_model":null,"is_bookmarked":true,"is_temporary":true}"""))
        assertEquals("t1", thread.id)
        assertEquals("", thread.model)
        assertTrue(thread.isBookmarked)
        assertTrue(thread.isTemporary)
    }

    @Test fun modelCatalogParsesMetadataAndSupportsLegacyFallback() {
        val catalog = parseModels(JSONObject("""{"models":[{"id":"gpt-5.6-sol","name":"GPT-5.6 Sol","provider":"openai","provider_label":"OpenAI","mode":"chat","capabilities":["chat","thinking"],"deprecated":false,"selectable":true}]}"""))
        assertEquals("GPT-5.6 Sol", catalog.single().name)
        assertTrue(catalog.single().supports("thinking"))
        val legacy = parseModels(JSONObject("""{"model_ids":["legacy-model"]}"""))
        assertTrue(legacy.single().selectable)
        assertEquals("legacy-model", legacy.single().name)
    }

    @Test fun attachmentReferencesStayOnTheFixedOrigin() {        assertEquals("123/pic.png", fileReferencePath("123/pic.png"))
        assertEquals("123/pic.png", fileReferencePath("/files/123/pic.png"))
        assertEquals("123/pic.png", fileReferencePath("files/123/pic.png"))
        assertEquals("123/pic.png", fileReferencePath("https://ai.minashin1120.com/files/123/pic.png"))
        assertNull(fileReferencePath("https://example.org/files/123/pic.png"))
        assertNull(fileReferencePath("/files/../secret"))
        assertNull(fileReferencePath("javascript:alert(1)"))
        assertNull(fileReferencePath(""))
        assertTrue(isImageReference("123/pic.PNG"))
        assertFalse(isImageReference("123/report.pdf"))
    }

    @Test fun attachmentKindsAndByteSizesRenderForPreview() {
        assertEquals(AttachmentKind.IMAGE, attachmentKind("photo.PNG"))
        assertEquals(AttachmentKind.IMAGE, attachmentKind("blob", "image/heic"))
        assertEquals(AttachmentKind.AUDIO, attachmentKind("voice.m4a"))
        assertEquals(AttachmentKind.VIDEO, attachmentKind("clip.mp4"))
        assertEquals(AttachmentKind.PDF, attachmentKind("report.pdf"))
        assertEquals(AttachmentKind.TEXT, attachmentKind("notes.md"))
        assertEquals(AttachmentKind.FILE, attachmentKind("archive.zip"))
        assertEquals("512 B", formatByteSize(512))
        assertEquals("1 KB", formatByteSize(1024))
        assertEquals("1.5 MB", formatByteSize(1024L * 1024 * 3 / 2))
        assertEquals("2 GB", formatByteSize(2L * 1024 * 1024 * 1024))
        assertEquals("", formatByteSize(-1))
        assertEquals("jpg", extensionForMime("image/jpeg"))
        assertEquals("m4a", extensionForMime("audio/x-m4a; charset=binary"))
        assertEquals("", extensionForMime("application/zip"))
    }

    @Test fun libraryGemsAndMentionsParse() {
        val library = parseLibraryFiles(JSONObject("""{"files":[{"filename":"renamed.txt","original_filename":"note.txt","filepath":"1/abc.txt","url":"/files/1/abc.txt","thumbnail_url":null,"type":"file","ext":"txt","is_favorite":true,"ts":123}],"total":1,"has_more":false}"""))
        assertEquals("renamed.txt", library.single().displayName)
        assertEquals("1/abc.txt", library.single().filepath)
        assertTrue(library.single().isFavorite)
        assertFalse(library.single().isImage)
        val gems = parseGems(JSONArray("""[{"uuid":"g1","name":"Brief","description":"d","instruction":"be brief","default_model":"gpt-5.6-sol"}]"""))
        assertEquals("Brief", gems.single().name)
        assertEquals("gpt-5.6-sol", gems.single().defaultModel)
        assertEquals("bri", gemMentionQuery("hello @bri"))
        assertNull(gemMentionQuery("hello world"))
        assertEquals("hello", replaceGemMention("hello @bri", "bri"))
    }

    @Test fun preferencesParseSafeFields() {
        val prefs = parsePreferences(JSONObject("""{"username":"u","default_model":"gpt-5.6-sol","default_enable_thinking":true,"default_enable_search":false,"enter_to_send":true,"light_mode_enabled":true,"auto_search_on_links":false,"theme_color":"#123456","temp_chat_timeout_seconds":900,"enable_e2ee":true,"is_2fa_enabled":true,"has_totp":true,"has_webauthn":false,"session_created_at":"2026-09-16T00:00:00Z","session_expires_at":"2026-10-16T00:00:00Z","device_name":"Pixel"}"""))
        assertEquals("gpt-5.6-sol", prefs.defaultModel)
        assertTrue(prefs.defaultEnableThinking)
        assertTrue(prefs.lightModeEnabled)
        assertFalse(prefs.autoSearchOnLinks)
        assertEquals(900, prefs.tempChatTimeoutSeconds)
        assertTrue(prefs.e2eeEnabled)
        assertTrue(prefs.twoFactorEnabled)
        assertEquals("Pixel", prefs.deviceName)
    }

    @Test fun branchPathRebuildsFromLeafAndSiblings() {
        val messages = listOf(
            ChatMessage("1", "user", "q1", parentId = null),
            ChatMessage("2", "assistant", "a1", parentId = 1),
            ChatMessage("3", "user", "q2", parentId = 2),
            ChatMessage("4", "assistant", "a2", parentId = 3),
            ChatMessage("5", "user", "q2b", parentId = 2),
            ChatMessage("6", "assistant", "a2b", parentId = 5),
        )
        assertEquals(listOf("1", "2", "3", "4"), activeBranchPath(messages, 4).map { it.id })
        assertEquals(listOf("1", "2", "5", "6"), activeBranchPath(messages, 6).map { it.id })
        assertEquals(4, latestLeafId(messages, 3))
        assertEquals(6, latestLeafId(messages, 5))
        assertEquals(listOf("3", "5"), siblingGroup(messages, messages[4]).map { it.id })
        assertTrue(activeBranchPath(emptyList(), null).isEmpty())
        val pdf = parsePdfMessages(JSONObject("""{"messages":[{"role":"user","content":"こんにちは"},{"role":"assistant","content":"はい"}]}"""))
        assertEquals(2, pdf.size)
        assertEquals("user", pdf[0].role)
        assertEquals("こんにちは", pdf[0].content)
    }

    @Test fun compressionSettingsComputeByteBudget() {
        val settings = CompressionSettings(maxSizeMB = 1.0f, maxDimension = 1920)
        assertEquals(1024L * 1024L, settings.maxSizeBytes)
        assertTrue(CompressionSettings(maxSizeMB = 0.001f).maxSizeBytes >= 64L * 1024L)
        assertFalse(CompressionSettings().formatOnly)
        assertEquals("original", CompressionSettings().outputType)
    }

    @Test fun binaryReadIsBounded() {
        assertEquals(5, readBoundedBytes(ByteArrayInputStream(ByteArray(5)), 5L).size)
        assertTrue(runCatching { readBoundedBytes(ByteArrayInputStream(ByteArray(6)), 5L) }.exceptionOrNull() is IOException)
    }

    @Test fun liveCardsMergeSearchAndPythonEvents() {
        val search = upsertSearchCard(emptyList(), "searching")
        assertFalse(search.single().done)
        assertTrue(upsertSearchCard(search, "done").single().done)
        val code = upsertPythonCard(emptyList(), JSONObject("""{"id":"p1","code":"print(1)"}"""))
        assertEquals("print(1)", code.single().code)
        val output = upsertPythonCard(code, JSONObject("""{"id":"p1","output":"1"}"""))
        assertEquals("print(1)", output.single().code)
        assertTrue(output.single().done)
    }

    @Test fun batchJobsAndStructuredToolCardsParse() {
        val jobs = parseBatchJobs(JSONObject("""{"jobs":[{"job_id":"b1","thread_id":"t1","thread_title":"Batch","model":"gpt-5.6-sol","provider":"openai","state":"JOB_STATE_RUNNING","status_text":"実行中","error":null,"is_active":true,"can_cancel":true}]}"""))
        assertEquals("b1", jobs.single().id)
        assertTrue(jobs.single().active)
        assertTrue(jobs.single().canCancel)
        val mcp = upsertToolCard(emptyList(), "mcp", JSONObject("""{"tool_call_id":"m1","tool_name":"search","status":"running"}"""))
        assertEquals(CardKind.MCP, mcp.single().kind)
        val coding = upsertToolCard(emptyList(), "coding_diff", JSONObject("""{"target_id":"c1","diff":"@@ -1 +1 @@"}"""))
        assertEquals(CardKind.CODING, coding.single().kind)
        assertTrue(coding.single().done)
    }
}
