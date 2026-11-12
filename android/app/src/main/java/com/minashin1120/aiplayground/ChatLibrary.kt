package com.minashin1120.aiplayground

import android.content.SharedPreferences
import com.minashin1120.aiplayground.data.ApiException
import com.minashin1120.aiplayground.data.Attachment
import com.minashin1120.aiplayground.data.FileUsageChat
import com.minashin1120.aiplayground.data.LIBRARY_SORTS
import com.minashin1120.aiplayground.data.LibraryFile
import com.minashin1120.aiplayground.data.OfflineCacheStore
import com.minashin1120.aiplayground.data.backend.ChatBackend
import com.minashin1120.aiplayground.data.isAudioPath
import com.minashin1120.aiplayground.data.isVideoPath
import com.minashin1120.aiplayground.data.modelMediaSupport
import com.minashin1120.aiplayground.data.nullableString
import com.minashin1120.aiplayground.data.parseLibraryFiles
import com.minashin1120.aiplayground.data.sortLibraryFiles
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.net.URLEncoder

/**
 * The file library (Web `#files-modal`): list with search, sort and favorites, rename, delete,
 * attach to the composer without re-uploading, download, and "where is it used". State lives in the
 * view model; this class only reads it and reports through `mutable`.
 */
internal class ChatLibrary(
    private val state: StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val scope: CoroutineScope,
    private val backend: () -> ChatBackend,
    private val token: () -> String,
    private val prefs: SharedPreferences,
    private val offlineCache: OfflineCacheStore,
    private val attachmentFiles: ChatAttachmentFiles,
    private val notify: (String) -> Unit,
    private val notifyOffline: (String) -> Boolean,
    private val refreshCacheStats: suspend (Int) -> Unit,
) {
    private var job: Job? = null

    /** The running list request (search, reload, more), cancelled at the BAN and logout boundaries. */
    val activeJob: Job? get() = job

    private fun libraryPath(offset: Int): String {
        val current = state.value
        val query = URLEncoder.encode(current.libraryQuery, "UTF-8")
        val favorites = if (current.libraryFavoritesOnly) "&favorites_only=1" else ""
        return "/api/files?limit=40&offset=$offset&sort=${current.librarySort}&q=$query$favorites"
    }

    private suspend fun fetchLibrary(append: Boolean) {
        if (state.value.offline) {
            val accountId = state.value.account?.id ?: return
            val files = withContext(Dispatchers.IO) { offlineCache.loadLibrary(accountId) }
                .filter { file ->
                    (state.value.libraryQuery.isBlank() || file.displayName.contains(state.value.libraryQuery, true)) &&
                        (!state.value.libraryFavoritesOnly || file.isFavorite)
                }
                .let { sortLibraryFiles(it, state.value.librarySort) }
            mutable.update { it.copy(library = files, libraryHasMore = false, libraryTotal = files.size, libraryBusy = false) }
            return
        }
        val offset = if (append) state.value.library.size else 0
        val reply = backend().get(libraryPath(offset), token())
        val files = parseLibraryFiles(reply)
        state.value.account?.let { account ->
            withContext(Dispatchers.IO) { offlineCache.saveLibrary(account.id, files) }
            refreshCacheStats(account.id)
        }
        mutable.update {
            it.copy(
                library = if (append) (it.library + files).distinctBy { file -> file.filepath } else files,
                libraryHasMore = reply.optBoolean("has_more"),
                libraryTotal = reply.optInt("total"),
                libraryBusy = false,
                libraryFailed = false,
            )
        }
    }

    fun refresh() {
        job?.cancel()
        job = scope.launch {
            mutable.update { it.copy(libraryBusy = true) }
            try { fetchLibrary(false) }
            catch (e: CancellationException) { throw e }
            catch (e: Exception) { mutable.update { it.copy(libraryFailed = true) } }
            finally { mutable.update { it.copy(libraryBusy = false) } }
        }
    }

    fun search(query: String) {
        mutable.update { it.copy(libraryQuery = query) }
        job?.cancel()
        job = scope.launch {
            delay(300)
            mutable.update { it.copy(libraryBusy = true) }
            try { fetchLibrary(false) }
            catch (e: CancellationException) { throw e }
            catch (e: Exception) { mutable.update { it.copy(libraryFailed = true) } }
            finally { mutable.update { it.copy(libraryBusy = false) } }
        }
    }

    fun setFavoritesOnly(value: Boolean) {
        if (state.value.libraryFavoritesOnly == value) return
        prefs.edit().putBoolean(LIB_FAVORITES_ONLY_KEY, value).apply()
        mutable.update { it.copy(libraryFavoritesOnly = value) }
        refresh()
    }

    private fun isMoreBlocked(): Boolean = state.value.offline || !state.value.libraryHasMore || state.value.libraryBusy

    fun more() {
        if (isMoreBlocked()) return
        job = scope.launch {
            mutable.update { it.copy(libraryBusy = true) }
            try { fetchLibrary(true) }
            catch (e: CancellationException) { throw e }
            catch (e: Exception) { notify("追加読み込みに失敗しました。もう一度お試しください。") }
            finally { mutable.update { it.copy(libraryBusy = false) } }
        }
    }

    fun toggleFavorite(file: LibraryFile) { scope.launch {
        if (notifyOffline("オフライン中はファイルのお気に入りを変更できません。")) return@launch
        try {
            val reply = backend().post("/api/files/favorite", JSONObject().put("filepath", file.filepath), token())
            val favorite = reply.optBoolean("is_favorite")
            mutable.update { current -> current.copy(
                library = current.library.map { if (it.filepath == file.filepath) it.copy(isFavorite = favorite) else it },
                notice = if (favorite) "お気に入りに追加しました" else "お気に入りから外しました",
            ) }
            state.value.account?.let { account -> withContext(Dispatchers.IO) { offlineCache.saveLibrary(account.id, state.value.library) } }
        } catch (e: Exception) { notify("お気に入りの更新に失敗しました") }
    } }

    fun rename(file: LibraryFile, name: String) { scope.launch {
        if (notifyOffline("オフライン中はファイル名を変更できません。")) return@launch
        try {
            if (name.isBlank()) { notify("ファイル名を入力してください"); return@launch }
            val reply = backend().post("/api/files/rename", JSONObject().put("filepath", file.filepath).put("filename", name.trim()), token())
            val display = reply.optString("filename", name.trim())
            mutable.update { current -> current.copy(
                library = current.library.map { if (it.filepath == file.filepath) it.copy(displayName = display) else it },
                attachments = current.attachments.map { if (it.reference == file.filepath) it.copy(name = display) else it },
                notice = "ファイル名を変更しました",
            ) }
            state.value.account?.let { account -> withContext(Dispatchers.IO) { offlineCache.saveLibrary(account.id, state.value.library) } }
        } catch (e: ApiException) { notify(e.payload.optString("error").ifBlank { "名前変更に失敗しました" }) }
        catch (e: Exception) { notify("名前変更に失敗しました") }
    } }

    fun delete(file: LibraryFile) { scope.launch {
        if (notifyOffline("オフライン中はファイルを削除できません。")) return@launch
        try {
            backend().post("/api/files/delete", JSONObject().put("filenames", JSONArray().put(file.filepath)), token())
            state.value.account?.let { account -> withContext(Dispatchers.IO) { offlineCache.deleteLibraryFile(account.id, file.filepath) } }
            mutable.update { current -> current.copy(library = current.library.filterNot { it.filepath == file.filepath }) }
        } catch (e: Exception) { notify("削除に失敗しました") }
    } }

    /** Web `deleteSelectedFiles`: one batch request, then the list is reloaded. */
    fun deleteAll(filepaths: List<String>) { scope.launch {
        if (notifyOffline("オフライン中はファイルを削除できません。")) return@launch
        try {
            backend().post("/api/files/delete", JSONObject().put("filenames", JSONArray(filepaths)), token())
            state.value.account?.let { account ->
                withContext(Dispatchers.IO) { filepaths.forEach { offlineCache.deleteLibraryFile(account.id, it) } }
            }
            mutable.update { it.copy(libraryBusy = true) }
            try { fetchLibrary(false) } finally { mutable.update { it.copy(libraryBusy = false) } }
        } catch (e: Exception) { notify("削除エラー") }
    } }

    fun setSort(order: String) {
        if (order !in LIBRARY_SORTS || state.value.librarySort == order) return
        prefs.edit().putString(LIB_SORT_KEY, order).apply()
        mutable.update { it.copy(librarySort = order) }
        refresh()
    }

    /**
     * Web `attachSelectedLibraryFiles`: adds the files without uploading again, skipping audio or video
     * the current model cannot take (`getModelMediaSupport`).
     */
    fun attach(files: List<LibraryFile>) {
        val support = modelMediaSupport(state.value.model)
        var skippedAudio = 0
        var skippedVideo = 0
        val added = mutableListOf<Attachment>()
        files.forEach { file ->
            val audio = isAudioPath(file.filepath)
            val video = isVideoPath(file.filepath)
            if ((audio && !support.first) || (video && !support.second)) {
                if (audio) skippedAudio += 1
                if (video) skippedVideo += 1
                return@forEach
            }
            if (state.value.attachments.none { it.reference == file.filepath } && added.none { it.reference == file.filepath }) {
                added += Attachment(file.displayName, file.filepath, "", source = "library")
            }
        }
        val parts = listOfNotNull(skippedAudio.takeIf { it > 0 }?.let { "${it}件の音声" }, skippedVideo.takeIf { it > 0 }?.let { "${it}件の動画" })
        mutable.update { it.copy(
            attachments = it.attachments + added,
            notice = if (parts.isNotEmpty()) "このモデルは${parts.joinToString("・")}入力に非対応のため除外しました" else "ライブラリから添付しました",
        ) }
    }

    /** Web `showSelectedFileUsage`: chats that reference the file (up to 100) and whether more exist. */
    suspend fun usage(file: LibraryFile): Pair<List<FileUsageChat>, Boolean> {
        val reply = backend().get("/api/files/usage?filepath=" + URLEncoder.encode(file.filepath, "UTF-8"), token())
        val rows = reply.optJSONArray("chats") ?: JSONArray()
        val chats = (0 until rows.length()).map { index ->
            val row = rows.getJSONObject(index)
            FileUsageChat(row.optString("id"), row.nullableString("title").ifBlank { "新しいチャット" }, row.nullableString("updated_at"))
        }
        return chats to reply.optBoolean("has_more")
    }

    /** Web `downloadSelectedLibraryFiles`: fetches each file for the chosen save location. */
    fun saveFiles(files: List<LibraryFile>, target: suspend (LibraryFile, File, String) -> Unit) { scope.launch {
        var saved = 0
        files.forEach { file ->
            runCatching {
                val (local, mime) = attachmentFiles.download(file.url.ifBlank { file.filepath })
                target(file, local, mime)
                local.delete()
            }.onSuccess { saved += 1 }
        }
        if (saved == files.size) notify("${files.size}件のファイルをダウンロードしました")
        else notify("ダウンロードに失敗しました（${files.size - saved}件）")
    } }

    /** Web `reuseCurrentImage`: the viewer's image becomes an attachment of the next message. */
    fun reuseImage(reference: String, name: String = ""): Boolean {
        if (state.value.attachments.any { it.reference == reference }) { notify("この画像は既に添付されています"); return false }
        mutable.update { it.copy(attachments = it.attachments + Attachment(name.ifBlank { reference.substringAfterLast('/') }, reference, "", source = "library")) }
        notify("画像を添付ファイルに追加しました")
        return true
    }

    /** Reuses a library file as a composer attachment without re-uploading it. */
    fun reuse(file: LibraryFile) {
        if (state.value.attachments.any { it.reference == file.filepath }) {
            mutable.update { it.copy(notice = "この添付はすでに追加されています。") }
            return
        }
        mutable.update { it.copy(attachments = it.attachments + Attachment(file.displayName, file.filepath, "", source = "library")) }
    }

    internal companion object {
        /** Web keeps the library order and favorites filter in localStorage. */
        const val LIB_SORT_KEY = "lib_sort_order"
        const val LIB_FAVORITES_ONLY_KEY = "lib_favorites_only"
    }
}
