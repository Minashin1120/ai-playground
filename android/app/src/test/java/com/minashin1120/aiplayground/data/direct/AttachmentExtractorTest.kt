package com.minashin1120.aiplayground.data.direct

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

class AttachmentExtractorTest {
    private fun minimalPptx(slides: List<List<String>>): ByteArray {
        val out = ByteArrayOutputStream()
        ZipOutputStream(out).use { zip ->
            slides.forEachIndexed { index, paragraphs ->
                val body = paragraphs.joinToString("") { "<a:p><a:r><a:t>$it</a:t></a:r></a:p>" }
                val xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>""" +
                    """<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" """ +
                    """xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">""" +
                    """<p:cSld><p:spTree><p:sp><p:txBody>$body</p:txBody></p:sp></p:spTree></p:cSld></p:sld>"""
                zip.putNextEntry(ZipEntry("ppt/slides/slide${index + 1}.xml"))
                zip.write(xml.toByteArray(Charsets.UTF_8))
                zip.closeEntry()
            }
        }
        return out.toByteArray()
    }

    @Test fun pptxExtractsSlideTextInOrder() {
        val bytes = minimalPptx(listOf(listOf("First", "Line"), listOf("Second")))
        val prepared = AttachmentExtractor.prepare(
            "deck.pptx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            bytes,
        )
        assertEquals("unexpected pptx extracted text: ${prepared.text}", "[Slide 1]\nFirst\nLine\n\n[Slide 2]\nSecond", prepared.text)
        assertTrue("expected binary bytes to be omitted for pptx", prepared.bytes == null)
    }
}
