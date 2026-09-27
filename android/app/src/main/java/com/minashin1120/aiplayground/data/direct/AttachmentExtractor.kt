package com.minashin1120.aiplayground.data.direct

import org.xmlpull.v1.XmlPullParser
import org.xmlpull.v1.XmlPullParserFactory
import java.io.ByteArrayInputStream
import java.util.zip.ZipInputStream

/**
 * Turns an attachment into what a provider accepts (server `storage.py` text extraction): images and
 * PDFs stay binary, DOCX paragraphs and XLSX cells become text, plain text files are decoded, and
 * anything else is reported as not readable on the device.
 */
object AttachmentExtractor {
    private const val MAX_TEXT_CHARS = 400_000
    private val TEXT_EXTENSIONS = setOf(
        "txt", "md", "markdown", "csv", "tsv", "json", "xml", "html", "htm", "css", "js", "ts", "kt", "java", "py",
        "rb", "go", "rs", "c", "h", "cpp", "hpp", "cs", "swift", "php", "sh", "yaml", "yml", "toml", "ini", "log", "sql",
    )

    fun prepare(name: String, mime: String, bytes: ByteArray): DirectAttachment {
        val ext = name.substringAfterLast('.', "").lowercase()
        val type = mime.lowercase()
        return when {
            type.startsWith("image/") || type == "application/pdf" || type.startsWith("audio/") || type.startsWith("video/") ->
                DirectAttachment(name, type, bytes = bytes)
            ext == "pdf" -> DirectAttachment(name, "application/pdf", bytes = bytes)
            ext == "docx" -> DirectAttachment(name, type, text = runCatching { docxText(bytes) }.getOrElse { "（DOCXを読み取れませんでした）" })
            ext == "xlsx" -> DirectAttachment(name, type, text = runCatching { xlsxText(bytes) }.getOrElse { "（XLSXを読み取れませんでした）" })
            type.startsWith("text/") || ext in TEXT_EXTENSIONS -> DirectAttachment(name, type.ifBlank { "text/plain" },
                text = String(bytes, Charsets.UTF_8).take(MAX_TEXT_CHARS))
            else -> DirectAttachment(name, type, text = "（この形式の内容は端末では読み取れません）")
        }
    }

    private fun zipEntries(bytes: ByteArray, wanted: (String) -> Boolean): Map<String, ByteArray> {
        val out = LinkedHashMap<String, ByteArray>()
        var total = 0L
        ZipInputStream(ByteArrayInputStream(bytes)).use { zip ->
            while (true) {
                val entry = zip.nextEntry ?: break
                if (!entry.isDirectory && wanted(entry.name)) {
                    val data = zip.readNBytesCompat(32L * 1024 * 1024)
                    total += data.size
                    if (total > 64L * 1024 * 1024) break
                    out[entry.name] = data
                }
            }
        }
        return out
    }

    private fun parser(data: ByteArray): XmlPullParser = XmlPullParserFactory.newInstance().apply { isNamespaceAware = false }
        .newPullParser().apply { setInput(ByteArrayInputStream(data), "UTF-8") }

    /** Numbered paragraphs, like the server's DOCX extraction. */
    fun docxText(bytes: ByteArray): String {
        val document = zipEntries(bytes) { it == "word/document.xml" }["word/document.xml"] ?: return ""
        val paragraphs = mutableListOf<String>()
        val xml = parser(document)
        val current = StringBuilder()
        while (xml.next() != XmlPullParser.END_DOCUMENT) {
            when (xml.eventType) {
                XmlPullParser.START_TAG -> when (xml.name) { "w:tab" -> current.append('\t'); "w:br" -> current.append('\n') }
                XmlPullParser.TEXT -> if (xml.text != null) current.append(xml.text)
                XmlPullParser.END_TAG -> if (xml.name == "w:p") {
                    val text = current.toString().trim()
                    if (text.isNotEmpty()) paragraphs += text
                    current.clear()
                }
            }
        }
        return paragraphs.mapIndexed { index, text -> "[${index + 1}] $text" }.joinToString("\n").take(MAX_TEXT_CHARS)
    }

    /** Each sheet as tab-separated rows, like the server's XLSX → TSV conversion. */
    fun xlsxText(bytes: ByteArray): String {
        val entries = zipEntries(bytes) { it == "xl/sharedStrings.xml" || (it.startsWith("xl/worksheets/sheet") && it.endsWith(".xml")) }
        val shared = mutableListOf<String>()
        entries["xl/sharedStrings.xml"]?.let { data ->
            val xml = parser(data)
            val current = StringBuilder()
            var inside = false
            while (xml.next() != XmlPullParser.END_DOCUMENT) {
                when (xml.eventType) {
                    XmlPullParser.START_TAG -> if (xml.name == "si") { inside = true; current.clear() }
                    XmlPullParser.TEXT -> if (inside && xml.text != null) current.append(xml.text)
                    XmlPullParser.END_TAG -> if (xml.name == "si") { inside = false; shared += current.toString() }
                }
            }
        }
        val sheets = entries.keys.filter { it.startsWith("xl/worksheets/") }
            .sortedBy { it.removePrefix("xl/worksheets/sheet").removeSuffix(".xml").toIntOrNull() ?: Int.MAX_VALUE }
        val out = StringBuilder()
        sheets.forEachIndexed { index, key ->
            out.append("# Sheet ").append(index + 1).append('\n')
            val xml = parser(entries.getValue(key))
            val row = mutableListOf<String>()
            var cellType = ""
            var value = StringBuilder()
            var inValue = false
            while (xml.next() != XmlPullParser.END_DOCUMENT) {
                when (xml.eventType) {
                    XmlPullParser.START_TAG -> when (xml.name) {
                        "row" -> row.clear()
                        "c" -> { cellType = xml.getAttributeValue(null, "t").orEmpty(); value = StringBuilder() }
                        "v", "t" -> inValue = true
                    }
                    XmlPullParser.TEXT -> if (inValue && xml.text != null) value.append(xml.text)
                    XmlPullParser.END_TAG -> when (xml.name) {
                        "v", "t" -> inValue = false
                        "c" -> row += if (cellType == "s") shared.getOrNull(value.toString().trim().toIntOrNull() ?: -1).orEmpty() else value.toString()
                        "row" -> out.append(row.joinToString("\t")).append('\n')
                    }
                }
            }
            if (out.length > MAX_TEXT_CHARS) return out.toString().take(MAX_TEXT_CHARS)
        }
        return out.toString().trimEnd()
    }
}
