package com.minashin1120.aiplayground

import android.net.Uri
import com.minashin1120.aiplayground.data.OfflineCacheStore
import com.minashin1120.aiplayground.data.ProjectItem
import com.minashin1120.aiplayground.data.ThreadItem
import com.minashin1120.aiplayground.data.backend.ChatBackend
import com.minashin1120.aiplayground.data.parseProject
import com.minashin1120.aiplayground.data.parseProjects
import com.minashin1120.aiplayground.data.parseThreadItem
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
import java.net.URLEncoder

/**
 * Projects (Web `chat_core.part33_projects.js`): project list, dedicated project view,
 * per-project Vision toggle/model, shared files (upload / library attach / note / remove),
 * and moving chats between projects. State lives in [ChatState]; this collaborator updates `mutable`.
 */
internal class ChatProjects(
    private val state: StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val scope: CoroutineScope,
    private val backend: () -> ChatBackend,
    private val token: () -> String,
    private val offlineCache: OfflineCacheStore,
    private val uploader: ChatAttachmentUploader,
    private val isChatLocal: () -> Boolean,
    private val notify: (String) -> Unit,
    private val notifyOffline: (String) -> Boolean,
    private val notifyChatOffline: (String?, String) -> Boolean,
    private val fetchThreads: suspend (Boolean) -> Unit,
    private val startNewChatForProject: (String, String) -> Unit,
    private val refreshCacheStats: suspend (Int) -> Unit,
    private val report: suspend (Throwable, String?) -> Unit,
) {
    private var searchJob: Job? = null

    suspend fun fetch() {
        val items = parseProjects(backend().getArray("/api/projects", token()))
        if (!isChatLocal()) state.value.account?.let { account ->
            withContext(Dispatchers.IO) { offlineCache.saveProjects(account.id, items) }
            refreshCacheStats(account.id)
        }
        mutable.update { current ->
            val updatedViewing = current.viewingProject?.let { v ->
                items.firstOrNull { it.uuid == v.uuid }?.let { summary ->
                    v.copy(
                        name = summary.name,
                        description = summary.description,
                        instruction = summary.instruction,
                        useVisionForImages = summary.useVisionForImages,
                        visionModel = summary.visionModel,
                        fileCount = summary.fileCount,
                        threadCount = summary.threadCount,
                    )
                }
            }
            val updatedActiveName = current.activeProjectId?.let { id ->
                items.firstOrNull { it.uuid == id }?.name
            } ?: current.activeProjectName
            current.copy(
                projects = items,
                viewingProject = updatedViewing,
                activeProjectName = updatedActiveName,
            )
        }
    }

    fun load() {
        scope.launch {
            if (notifyOffline("オフライン中はプロジェクトを読み込めません。")) return@launch
            mutable.update { it.copy(projectsBusy = true) }
            try { fetch() } catch (e: Exception) { report(e, null) }
            finally { mutable.update { it.copy(projectsBusy = false) } }
        }
    }

    fun reload(onDone: () -> Unit) {
        scope.launch {
            if (notifyOffline("オフライン中はプロジェクトを更新できません。")) { onDone(); return@launch }
            try { fetch() } catch (e: Exception) { report(e, null) }
            finally { onDone() }
        }
    }

    fun openView(uuid: String) {
        val summary = state.value.projects.firstOrNull { it.uuid == uuid }
        mutable.update { current ->
            current.copy(
                viewingProject = summary ?: current.viewingProject,
                activeProjectId = uuid,
                activeProjectName = summary?.name ?: current.activeProjectName,
                projectThreads = emptyList(),
                projectThreadsNextPage = null,
                projectThreadsQuery = "",
                projectsBusy = true,
            )
        }
        scope.launch {
            try {
                if (state.value.offline && !isChatLocal()) {
                    val accountId = state.value.account?.id
                    val cached = accountId?.let { withContext(Dispatchers.IO) { offlineCache.loadProjectDetail(it, uuid) } }
                    val cachedThreads = accountId?.let {
                        withContext(Dispatchers.IO) { offlineCache.loadThreads(it).filter { t -> t.projectId == uuid } }
                    }.orEmpty()
                    if (cached != null) {
                        mutable.update { it.copy(viewingProject = cached, activeProjectName = cached.name, projectThreads = cachedThreads, projectsBusy = false) }
                    } else {
                        mutable.update { it.copy(projectThreads = cachedThreads, projectsBusy = false) }
                    }
                    return@launch
                }
                val reply = backend().get("/api/projects/$uuid", token())
                val detail = (reply.optJSONObject("project") ?: reply).takeIf { it.optString("uuid").isNotBlank() }?.let { parseProject(it) }
                if (detail != null) {
                    if (!isChatLocal()) state.value.account?.let { account ->
                        withContext(Dispatchers.IO) { offlineCache.saveProjectDetail(account.id, detail) }
                    }
                    mutable.update { current ->
                        current.copy(
                            viewingProject = detail,
                            activeProjectId = detail.uuid,
                            activeProjectName = detail.name,
                            projects = current.projects.map { if (it.uuid == detail.uuid) detail else it },
                        )
                    }
                }
                fetchProjectThreads(uuid, more = false)
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                report(e, "プロジェクトの読み込みに失敗しました")
            } finally {
                mutable.update { it.copy(projectsBusy = false) }
            }
        }
    }

    fun closeView(keepContext: Boolean = false) {
        mutable.update { current ->
            current.copy(
                viewingProject = null,
                activeProjectId = if (keepContext || current.selected != null) current.activeProjectId else null,
                activeProjectName = if (keepContext || current.selected != null) current.activeProjectName else "",
            )
        }
    }

    fun startNewChatInProject(project: ProjectItem) {
        mutable.update { current ->
            current.copy(
                viewingProject = null,
                activeProjectId = project.uuid,
                activeProjectName = project.name,
                includeProjectInstruction = true,
                projectInstructionPreview = project.instruction,
            )
        }
        startNewChatForProject(project.uuid, project.name)
    }

    fun save(
        uuid: String?,
        name: String,
        description: String,
        instruction: String,
        useVisionForImages: Boolean,
        visionModel: String?,
        onDone: (Boolean) -> Unit,
    ) {
        scope.launch {
            if (notifyOffline("オフライン中はプロジェクトを保存できません。")) { onDone(false); return@launch }
            mutable.update { it.copy(projectsBusy = true) }
            try {
                val payload = JSONObject()
                    .put("name", name.trim())
                    .put("description", description.trim())
                    .put("instruction", instruction)
                    .put("use_vision_for_images", useVisionForImages)
                    .put("vision_model", visionModel?.takeIf { it.isNotBlank() } ?: JSONObject.NULL)
                val reply = if (uuid.isNullOrBlank()) backend().post("/api/projects", payload, token())
                else backend().put("/api/projects/$uuid", payload, token())
                val saved = (reply.optJSONObject("project") ?: reply).takeIf { it.optString("uuid").isNotBlank() }?.let { parseProject(it) }
                fetch()
                if (saved != null) {
                    if (!isChatLocal()) state.value.account?.let { account ->
                        withContext(Dispatchers.IO) { offlineCache.saveProjectDetail(account.id, saved) }
                    }
                    if (uuid.isNullOrBlank()) {
                        notify("プロジェクトを作成しました")
                        openView(saved.uuid)
                    } else {
                        notify("プロジェクトを保存しました")
                        mutable.update { current ->
                            current.copy(
                                viewingProject = if (current.viewingProject?.uuid == saved.uuid) saved else current.viewingProject,
                                activeProjectName = if (current.activeProjectId == saved.uuid) saved.name else current.activeProjectName,
                                projectInstructionPreview = if (current.activeProjectId == saved.uuid) saved.instruction else current.projectInstructionPreview,
                            )
                        }
                    }
                }
                onDone(true)
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                report(e, "プロジェクトの保存に失敗しました")
                onDone(false)
            } finally {
                mutable.update { it.copy(projectsBusy = false) }
            }
        }
    }

    fun updateVisionSettings(uuid: String, useVisionForImages: Boolean, visionModel: String?) {
        scope.launch {
            if (notifyOffline("オフライン中はプロジェクト設定を変更できません。")) return@launch
            try {
                val payload = JSONObject()
                    .put("use_vision_for_images", useVisionForImages)
                    .put("vision_model", visionModel?.takeIf { it.isNotBlank() } ?: JSONObject.NULL)
                val reply = backend().put("/api/projects/$uuid", payload, token())
                val saved = (reply.optJSONObject("project") ?: reply).takeIf { it.optString("uuid").isNotBlank() }?.let { parseProject(it) }
                if (saved != null) {
                    if (!isChatLocal()) state.value.account?.let { account ->
                        withContext(Dispatchers.IO) { offlineCache.saveProjectDetail(account.id, saved) }
                    }
                    mutable.update { current ->
                        current.copy(
                            viewingProject = if (current.viewingProject?.uuid == saved.uuid) saved else current.viewingProject,
                            projects = current.projects.map { if (it.uuid == saved.uuid) saved else it },
                        )
                    }
                }
                notify("画像のVisionモデル設定を更新しました")
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                report(e, "Vision設定の更新に失敗しました")
            }
        }
    }

    fun delete(project: ProjectItem) {
        scope.launch {
            if (notifyOffline("オフライン中はプロジェクトを削除できません。")) return@launch
            try {
                backend().delete("/api/projects/${project.uuid}", token())
                if (!isChatLocal()) state.value.account?.let { account ->
                    withContext(Dispatchers.IO) { offlineCache.deleteProject(account.id, project.uuid) }
                }
                mutable.update { current ->
                    current.copy(
                        projects = current.projects.filterNot { it.uuid == project.uuid },
                        viewingProject = current.viewingProject?.takeIf { it.uuid != project.uuid },
                        activeProjectId = current.activeProjectId?.takeIf { it != project.uuid },
                        activeProjectName = if (current.activeProjectId == project.uuid) "" else current.activeProjectName,
                        selected = current.selected?.let { sel ->
                            if (sel.projectId == project.uuid) sel.copy(projectId = null, projectName = "") else sel
                        },
                    )
                }
                fetchThreads(false)
                notify("プロジェクトを削除しました")
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                report(e, "プロジェクトの削除に失敗しました")
            }
        }
    }

    fun searchProjectThreads(query: String) {
        val uuid = state.value.viewingProject?.uuid ?: return
        mutable.update { it.copy(projectThreadsQuery = query, projectThreadsNextPage = null) }
        searchJob?.cancel()
        searchJob = scope.launch {
            delay(250)
            try { fetchProjectThreads(uuid, more = false) }
            catch (e: CancellationException) { throw e }
            catch (e: Exception) { report(e, null) }
        }
    }

    fun loadMoreProjectThreads() {
        val uuid = state.value.viewingProject?.uuid ?: return
        if (state.value.projectThreadsNextPage == null) return
        scope.launch {
            try { fetchProjectThreads(uuid, more = true) }
            catch (e: CancellationException) { throw e }
            catch (e: Exception) { report(e, null) }
        }
    }

    suspend fun fetchProjectThreads(uuid: String, more: Boolean) {
        val current = state.value
        val page = if (more) current.projectThreadsNextPage ?: return else 1
        val q = URLEncoder.encode(current.projectThreadsQuery, "UTF-8")
        val reply = backend().get("/api/projects/$uuid/threads?page=$page&q=$q", token())
        val rows = reply.optJSONArray("threads") ?: JSONArray()
        val items = (0 until rows.length()).mapNotNull { rows.optJSONObject(it)?.let(::parseThreadItem) }
        val nextPage = if (reply.optBoolean("has_next")) reply.optInt("next_page", page + 1) else null
        mutable.update { cur ->
            if (cur.viewingProject?.uuid != uuid) cur
            else cur.copy(
                projectThreads = if (more) (cur.projectThreads + items).distinctBy { it.id } else items,
                projectThreadsNextPage = nextPage,
            )
        }
    }

    fun addFilesFromLibrary(uuid: String, filepaths: List<String>, onDone: (Boolean) -> Unit = {}) {
        if (filepaths.isEmpty()) { onDone(false); return }
        scope.launch {
            if (notifyOffline("オフライン中は共有ファイルを追加できません。")) { onDone(false); return@launch }
            mutable.update { it.copy(projectsBusy = true) }
            try {
                val payload = JSONObject().put("filepaths", JSONArray(filepaths))
                backend().post("/api/projects/$uuid/files", payload, token())
                refreshDetail(uuid)
                notify("共有ファイルを追加しました")
                onDone(true)
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                report(e, "共有ファイルの追加に失敗しました")
                onDone(false)
            } finally {
                mutable.update { it.copy(projectsBusy = false) }
            }
        }
    }

    fun uploadFiles(uuid: String, uris: List<Uri>) {
        if (uris.isEmpty()) return
        scope.launch {
            if (notifyOffline("オフライン中は共有ファイルをアップロードできません。")) return@launch
            mutable.update { it.copy(projectsBusy = true) }
            try {
                val uploadedPaths = mutableListOf<String>()
                for (uri in uris) {
                    val path = uploader.uploadSingleUri(uri)
                    if (path.isNotBlank()) uploadedPaths += path
                }
                if (uploadedPaths.isNotEmpty()) {
                    backend().post("/api/projects/$uuid/files", JSONObject().put("filepaths", JSONArray(uploadedPaths)), token())
                    refreshDetail(uuid)
                    notify("${uploadedPaths.size}件のファイルをプロジェクトに追加しました")
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                report(e, "共有ファイルのアップロードに失敗しました")
            } finally {
                mutable.update { it.copy(projectsBusy = false) }
            }
        }
    }

    fun updateFileNote(uuid: String, fileId: Int, note: String, onDone: (Boolean) -> Unit = {}) {
        scope.launch {
            if (notifyOffline("オフライン中はメモを保存できません。")) { onDone(false); return@launch }
            try {
                backend().put("/api/projects/$uuid/files/$fileId", JSONObject().put("note", note.trim()), token())
                refreshDetail(uuid)
                notify("メモを保存しました")
                onDone(true)
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                report(e, "メモの保存に失敗しました")
                onDone(false)
            }
        }
    }

    fun removeFile(uuid: String, fileId: Int) {
        scope.launch {
            if (notifyOffline("オフライン中は共有ファイルを解除できません。")) return@launch
            try {
                backend().delete("/api/projects/$uuid/files/$fileId", token())
                refreshDetail(uuid)
                notify("共有ファイルを解除しました")
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                report(e, "共有ファイルの解除に失敗しました")
            }
        }
    }

    fun moveThread(thread: ThreadItem, projectUuid: String?, onDone: (Boolean) -> Unit = {}) {
        scope.launch {
            if (notifyChatOffline(thread.id, "オフライン中はチャットを移動できません。")) { onDone(false); return@launch }
            try {
                val cleanUuid = projectUuid?.trim()?.takeIf { it.isNotEmpty() }
                val reply = backend().put(
                    "/api/threads/${thread.id}/settings",
                    JSONObject().put("project_id", cleanUuid ?: JSONObject.NULL),
                    token(),
                )
                val targetName = reply.optString("project_name").takeIf { !reply.isNull("project_name") && it != "null" }
                    ?: cleanUuid?.let { id -> state.value.projects.firstOrNull { it.uuid == id }?.name }.orEmpty()
                mutable.update { current ->
                    val updatedSelected = current.selected?.let { sel ->
                        if (sel.id == thread.id) sel.copy(projectId = cleanUuid, projectName = targetName) else sel
                    }
                    current.copy(
                        selected = updatedSelected,
                        activeProjectId = if (current.selected?.id == thread.id) cleanUuid else current.activeProjectId,
                        activeProjectName = if (current.selected?.id == thread.id) targetName else current.activeProjectName,
                    )
                }
                fetch()
                fetchThreads(false)
                state.value.viewingProject?.uuid?.let { fetchProjectThreads(it, more = false) }
                notify(if (cleanUuid != null) "プロジェクト「$targetName」に移動しました" else "プロジェクトから外しました")
                onDone(true)
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                report(e, "チャットの移動に失敗しました")
                onDone(false)
            }
        }
    }

    suspend fun refreshAfterThreadChange() {
        runCatching { fetch() }
        state.value.viewingProject?.uuid?.let { uuid ->
            runCatching { fetchProjectThreads(uuid, more = false) }
        }
    }

    private suspend fun refreshDetail(uuid: String) {
        val reply = backend().get("/api/projects/$uuid", token())
        val detail = (reply.optJSONObject("project") ?: reply).takeIf { it.optString("uuid").isNotBlank() }?.let { parseProject(it) } ?: return
        if (!isChatLocal()) state.value.account?.let { account ->
            withContext(Dispatchers.IO) { offlineCache.saveProjectDetail(account.id, detail) }
        }
        mutable.update { current ->
            current.copy(
                viewingProject = if (current.viewingProject?.uuid == uuid) detail else current.viewingProject,
                projects = current.projects.map { if (it.uuid == uuid) detail else it },
            )
        }
    }
}
