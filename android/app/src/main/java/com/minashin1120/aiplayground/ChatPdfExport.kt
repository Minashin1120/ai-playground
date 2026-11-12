package com.minashin1120.aiplayground

import android.app.Application
import com.minashin1120.aiplayground.data.backend.ChatBackend
import com.minashin1120.aiplayground.data.parsePdfMessages
import com.minashin1120.aiplayground.data.writeThreadPdf
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * スレッドのPDF出力（Web `openThreadPdfPrintDialog`）だけを持つ：スレッドの分岐（`leaf_id`）の
 * サーバーペイロードを取得し、共有キャッシュへ端末生成のA4 PDFを書き出す。
 */
internal class ChatPdfExport(
    private val application: Application,
    private val state: StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val scope: CoroutineScope,
    private val backend: () -> ChatBackend,
    private val token: () -> String,
    private val notify: (String) -> Unit,
    private val notifyChatOffline: (String?, String) -> Boolean,
) {
    /** Fetches the server thread payload and writes a native A4 PDF into the share cache. */
    fun export(onReady: (File) -> Unit) {
        val id = state.value.selected?.id
        if (id == null) { notify("PDF化するスレッドを開いてください"); return }
        if (state.value.pdfExporting) { notify("PDF出力の準備中です。しばらくお待ちください。"); return }
        if (notifyChatOffline(id, "PDF出力に失敗しました")) return
        mutable.update { it.copy(pdfExporting = true) }
        // Web `openThreadPdfPrintDialog`: the branch on screen (`leaf_id`) with the 準備中 progress toast.
        notify("PDF出力の準備中です")
        val leaf = state.value.leafId
        scope.launch {
            mutable.update { it.copy(busy = true) }
            try {
                val payload = backend().get("/c/$id/pdf" + (leaf?.let { "?leaf_id=$it" } ?: ""), token())
                val messages = parsePdfMessages(payload)
                val title = payload.optJSONObject("thread")?.optString("title").orEmpty().ifBlank { "AI Chat" }
                val safeId = id.filter { it.isLetterOrDigit() || it == '-' || it == '_' }.take(24).ifBlank { "thread" }
                val directory = File(application.cacheDir, "shared").apply { mkdirs() }
                val target = File(directory, "thread-$safeId.pdf")
                withContext(Dispatchers.IO) {
                    writeThreadPdf(title, payload.optString("generated_at"), messages, target)
                }
                onReady(target)
            } catch (e: CancellationException) { throw e }
            catch (ignored: Exception) { notify("PDF出力に失敗しました") }
            finally { mutable.update { it.copy(pdfExporting = false, busy = false) } }
        }
    }
}
