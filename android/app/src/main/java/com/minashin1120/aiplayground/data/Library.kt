package com.minashin1120.aiplayground.data

import java.text.Collator
import java.util.Locale

/** Web `#lib-sort` options in their order. */
val LIBRARY_SORTS = listOf("newest", "oldest", "name_asc", "name_desc")

/** One row of `/api/files/usage`. */
data class FileUsageChat(val id: String, val title: String, val updatedAt: String)

private val AUDIO_EXTS = setOf("mp3", "wav", "aac", "ogg", "flac", "aiff", "aif", "m4a", "opus", "oga", "weba", "webm")
private val VIDEO_EXTS = setOf("mp4", "mov", "avi", "mkv", "m4v", "webm", "mpg", "mpeg", "wmv", "3gp", "3gpp", "flv")

private fun fileExt(path: String) = path.substringBefore('?').substringAfterLast('/').substringAfterLast('.', "").lowercase(Locale.ROOT)

fun isAudioPath(path: String) = fileExt(path) in AUDIO_EXTS
fun isVideoPath(path: String) = fileExt(path) in VIDEO_EXTS

/** Web `getModelMediaSupport`: (audio, video) input support; only general Gemini models take both. */
fun modelMediaSupport(model: String): Pair<Boolean, Boolean> {
    val m = model.lowercase(Locale.ROOT)
    if (!m.contains("gemini")) return false to false
    if (listOf("image", "nano", "tts", "native-audio", "live").any { m.contains(it) }) return false to false
    if (m.contains("embedding") || m.startsWith("veo-") || m.contains("omni-flash") || m.contains("omni-1.1-flash") || m.startsWith("lyria-")) {
        return false to false
    }
    return true to true
}

/** Web `sortLibraryFiles`: the chosen order with the Web tie-breakers. */
fun sortLibraryFiles(files: List<LibraryFile>, order: String): List<LibraryFile> {
    val collator = Collator.getInstance(Locale.JAPANESE).apply { strength = Collator.PRIMARY }
    val nameAsc = Comparator<LibraryFile> { a, b -> collator.compare(a.displayName, b.displayName) }
    val newest = compareByDescending<LibraryFile> { it.timestamp }
    val comparator = when (order) {
        "name_asc" -> nameAsc.then(newest)
        "name_desc" -> nameAsc.reversed().then(newest)
        "oldest" -> compareBy<LibraryFile> { it.timestamp }.then(nameAsc)
        else -> newest.then(nameAsc)
    }
    return files.sortedWith(comparator)
}
