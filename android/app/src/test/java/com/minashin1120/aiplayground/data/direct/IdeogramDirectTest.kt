package com.minashin1120.aiplayground.data.direct

import kotlinx.coroutines.runBlocking
import mockwebserver3.MockResponse
import mockwebserver3.MockWebServer
import okio.Buffer
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class IdeogramDirectTest {
    private val http = DirectHttp(allowedHosts = emptySet(), allowCleartextLocalhost = true)
    private val png = DirectAttachment("a.png", "image/png", bytes = byteArrayOf(1, 2, 3))
    private fun request(model: String, prompt: String, options: JSONObject = JSONObject(), attachments: List<DirectAttachment> = emptyList()) =
        DirectRequest(model, "key-1", "SYSTEM", listOf(DirectTurn("user", prompt, attachments)), options)
    private fun bodyText(call: IdeogramCall) = Buffer().also { call.body.writeTo(it) }.readUtf8()

    @Test fun routerSelectsIdeogramForImageModels() {
        val router = DirectRouter(http)
        listOf("ideogram-4.5", "ideogram-4.0", "ideogram-3.0", "ideogram-2a", "ideogram-2.0").forEach { id ->
            assertEquals("ideogram", router.route(id, "image")?.provider)
        }
    }

    @Test fun ideogram4UsesAnExactResolutionOfTheChosenTier() {
        val engine = IdeogramImageDirect(http)
        val call = engine.buildCall(request("ideogram-4.0", "猫", JSONObject().put("ideogram_aspect", "16:9").put("ideogram_resolution", "2k")
            .put("ideogram_speed", "turbo").put("ideogram_count", "3")), emptyList())
        val body = JSONObject(bodyText(call))
        assertTrue(call.url.endsWith("/v2/image/generate/ideogram-4"))
        assertEquals("2560x1440", body.getString("resolution"))
        assertEquals("turbo", body.getString("rendering_speed"))
        assertEquals(3, body.getInt("num_images"))
        assertFalse(call.edit)
    }

    @Test fun legacyModelsSendAspectRatioAndOnlyIdeogram2TakesANegativePrompt() {
        val engine = IdeogramImageDirect(http)
        val options = JSONObject().put("ideogram_aspect", "4:3").put("ideogram_negative_prompt", "blur").put("ideogram_style_type", "anime")
        val v2 = JSONObject(bodyText(engine.buildCall(request("ideogram-2.0", "猫", options), emptyList())))
        assertEquals("4x3", v2.getString("aspect_ratio"))
        assertEquals("blur", v2.getString("negative_prompt"))
        assertEquals("anime", v2.getString("style_type"))
        val v2a = JSONObject(bodyText(engine.buildCall(request("ideogram-2a", "猫", options), emptyList())))
        assertFalse(v2a.has("negative_prompt"))
        val v3 = bodyText(engine.buildCall(request("ideogram-3.0", "猫", options), emptyList()))
        assertTrue(v3.contains("name=\"negative_prompt\""))
        assertFalse(v3.contains("name=\"style_type\""))
    }

    @Test fun ideogram45EditsWithAttachedImagesAndOthersRejectThem() {
        val engine = IdeogramImageDirect(http)
        val edit = engine.buildCall(request("ideogram-4.5", "青くして", attachments = listOf(png, png)), listOf(png, png))
        assertTrue(edit.edit)
        assertTrue(edit.url.endsWith("/v2/image/precise-edit/ideogram-4-5"))
        val text = bodyText(edit)
        assertTrue(text.contains("name=\"image\""))
        assertTrue(text.contains("name=\"reference_images\""))
        assertFalse(text.contains("name=\"size\""))
        val generate = engine.buildCall(request("ideogram-4.5", "猫", JSONObject().put("ideogram_aspect", "1:1").put("ideogram_quality", "very_low")), emptyList())
        assertFalse(generate.edit)
        assertTrue(bodyText(generate).contains("1024x1024"))
        assertFalse(bodyText(generate).contains("very_low"))
        assertTrue(runCatching { engine.buildCall(request("ideogram-4.0", "猫"), listOf(png)) }.exceptionOrNull() is DirectApiException)
    }

    @Test fun generatedImagesAreDownloadedAndSafetyBlockedOnesSkipped() {
        MockWebServer().use { server ->
            server.start()
            val base = server.url("/").toString().trimEnd('/')
            server.enqueue(MockResponse.Builder().addHeader("Content-Type", "application/json").body(
                """{"data":[{"is_image_safe":true,"url":"$base/out.png"},{"is_image_safe":false,"url":null}]}""").build())
            server.enqueue(MockResponse.Builder().body("PNGDATA").build())
            val result = runBlocking {
                IdeogramImageDirect(http, base).run(request("ideogram-4.5", "猫"), {}) { _, _ -> }
            }
            assertEquals(1, result.files.size)
            assertEquals("PNGDATA", String(result.files.single().bytes))
            assertEquals("image/png", result.files.single().mime)
            val first = server.takeRequest()
            assertEquals("key-1", first.headers["Api-Key"])
            assertTrue(first.requestLine.contains("/v2/image/generate/ideogram-4-5"))
        }
    }

    @Test fun lastModelOutputRemainsTheTargetAcrossReferenceAndTextOnlyTurns() {
        val engine = IdeogramImageDirect(http)
        val reference = png.copy(name = "reference.png", bytes = byteArrayOf(4, 5, 6))
        val generated = png.copy(name = "generated.png")
        val turns = listOf(
            DirectTurn("user", "edit", listOf(png)),
            DirectTurn("assistant", "done", generated = listOf(generated)),
            DirectTurn("user", "use this reference", listOf(reference))
        )
        val followUp = request("ideogram-4.5", "").copy(turns = turns)
        assertEquals(listOf(generated, reference), engine.sourceImages(followUp))
        val body = bodyText(engine.buildCall(followUp, engine.sourceImages(followUp)))
        assertTrue(body.contains("name=\"image\"; filename=\"generated.png\""))
        assertTrue(body.contains("name=\"reference_images\"; filename=\"reference.png\""))
        assertFalse(body.contains("filename=\"a.png\""))
        assertEquals(listOf(generated), engine.sourceImages(followUp.copy(
            turns = turns + DirectTurn("assistant", "done", generated = listOf(generated)) + DirectTurn("user", "again")
        )))
    }

    @Test fun imageSelectionHandlesFirstTurnDuplicatesLimitsAndGeneratedFallback() {
        val engine = IdeogramImageDirect(http)
        assertEquals(listOf(png), engine.sourceImages(request("ideogram-4.5", "edit", attachments = listOf(png, png))))
        assertTrue(engine.sourceImages(request("ideogram-4.5", "generate")).isEmpty())
        assertEquals(listOf(png), engine.sourceImages(request("ideogram-4.5", "").copy(turns = listOf(
            DirectTurn("user", "edit", listOf(png)), DirectTurn("user", "again")
        ))))
        val many = (1..8).map { png.copy(name = "$it.png") }
        val followUp = request("ideogram-4.5", "").copy(turns = listOf(
            DirectTurn("user", "edit", listOf(png)), DirectTurn("assistant", "done", generated = listOf(png)), DirectTurn("user", "reference", many)
        ))
        assertEquals(listOf(png) + many.take(4), engine.sourceImages(followUp))
        assertEquals(listOf(png), engine.sourceImages(followUp.copy(turns = listOf(
            DirectTurn("user", "generate"), DirectTurn("assistant", "done", generated = listOf(png)), DirectTurn("user", "edit")
        ))))
    }

    @Test fun selectionFollowsTheDisplayedBranchAndLastImageInAnAnswer() {
        val engine = IdeogramImageDirect(http)
        val older = png.copy(name = "old.png")
        val a = png.copy(name = "branch-a.png")
        val b = png.copy(name = "branch-b.png")
        val reference = png.copy(name = "reference.png")
        val common = listOf(DirectTurn("user", "edit", listOf(png)),
            DirectTurn("assistant", "done", generated = listOf(older)))
        fun branch(output: DirectAttachment) = request("ideogram-4.5", "").copy(turns = common + listOf(
            DirectTurn("assistant", "done", generated = listOf(older, output)),
            DirectTurn("assistant", "text only"), DirectTurn("user", "reference", listOf(reference))))
        assertEquals(listOf(a, reference), engine.sourceImages(branch(a)))
        assertEquals(listOf(b, reference), engine.sourceImages(branch(b)))
    }

    @Test fun errorsUseIdeogramMessages() {
        assertTrue(ideogramError(401, "{}").message.contains("APIキー"))
        assertTrue(ideogramError(422, "{}").message.contains("安全性"))
        assertTrue(ideogramError(402, """{"error":"no credit","reject_reason":"insufficient_funds"}""").message.contains("insufficient_funds"))
        assertEquals("HTTP 500: boom", ideogramError(500, """{"error":"boom"}""").message)
    }
}
