package com.minashin1120.aiplayground.data

import android.graphics.Color
import android.graphics.pdf.PdfDocument
import android.text.StaticLayout
import android.text.TextPaint
import java.io.File

private const val PAGE_WIDTH = 595
private const val PAGE_HEIGHT = 842
private const val MARGIN = 36f

/** Renders a plain, printable A4 PDF of the given messages without a WebView. */
fun writeThreadPdf(title: String, generatedAt: String, messages: List<ChatMessage>, output: File) {
    val document = PdfDocument()
    val titlePaint = TextPaint().apply { textSize = 16f; isFakeBoldText = true; color = Color.BLACK }
    val metaPaint = TextPaint().apply { textSize = 9f; color = Color.DKGRAY }
    val headerPaint = TextPaint().apply { textSize = 11f; isFakeBoldText = true; color = Color.BLACK }
    val bodyPaint = TextPaint().apply { textSize = 11f; color = Color.BLACK }
    val width = (PAGE_WIDTH - MARGIN * 2).toInt()
    val bottom = PAGE_HEIGHT - MARGIN

    var pageNumber = 1
    var page = document.startPage(PdfDocument.PageInfo.Builder(PAGE_WIDTH, PAGE_HEIGHT, pageNumber).create())
    var canvas = page.canvas
    var y = MARGIN

    fun newPageIfNeeded(needed: Int) {
        if (y + needed <= bottom) return
        document.finishPage(page)
        pageNumber++
        page = document.startPage(PdfDocument.PageInfo.Builder(PAGE_WIDTH, PAGE_HEIGHT, pageNumber).create())
        canvas = page.canvas
        y = MARGIN
    }

    fun drawLayout(text: String, paint: TextPaint, spacingAfter: Int) {
        if (text.isBlank()) return
        val layout = StaticLayout.Builder.obtain(text, 0, text.length, paint, width).build()
        newPageIfNeeded(layout.height + spacingAfter)
        canvas.save()
        canvas.translate(MARGIN, y)
        layout.draw(canvas)
        canvas.restore()
        y += layout.height + spacingAfter
    }

    drawLayout(title.ifBlank { "AI Chat" }, titlePaint, 8)
    if (generatedAt.isNotBlank()) drawLayout(generatedAt, metaPaint, 16)
    messages.forEach { message ->
        drawLayout(if (message.role == "user") "あなた" else "AI", headerPaint, 4)
        drawLayout(message.content.ifBlank { "（本文なし）" }, bodyPaint, 16)
    }

    document.finishPage(page)
    output.outputStream().use { document.writeTo(it) }
    document.close()
}
