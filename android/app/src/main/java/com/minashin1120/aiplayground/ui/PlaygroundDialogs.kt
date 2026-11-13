package com.minashin1120.aiplayground.ui

import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.key
import com.minashin1120.aiplayground.AppChangelogUiState
import com.minashin1120.aiplayground.AppUpdateUiState
import com.minashin1120.aiplayground.ChatViewModel
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.data.ImageSplitOptions
import com.minashin1120.aiplayground.data.ImageSplitRequest
import com.minashin1120.aiplayground.data.apiKeyInfoFor

/**
 * Dialogs, sheets and confirmations of the chat screen (Web `#model-picker`, `#file-viewer`,
 * `#settings-modal`, the upload sheet, the image tools and their confirm()` / `prompt()` dialogs).
 *
 * The flags live in [PlaygroundOverlays] so [PlaygroundScreen] only wires the chat content.
 * Every dialog's body is defined in its own file; this file keeps the opening conditions, the
 * callbacks and the order in one place.
 */
@Composable
internal fun PlaygroundDialogs(
    state: ChatState,
    model: ChatViewModel,
    overlays: PlaygroundOverlays,
    loader: FileBytesLoader,
    imageActions: ImageViewerActions,
    imageSplit: ImageSplitRequest?,
    sharedTextPreview: String?,
    codeActions: MarkdownCodeActions,
    sidebarActions: (() -> Unit) -> SidebarActions,
    closeDrawer: () -> Unit,
    openInApp: (String) -> Unit,
    openPendingAttachment: (String) -> Unit,
    onWeb: (String) -> Unit,
    onFile: (String) -> Unit,
    onRetryChangelog: () -> Unit,
    appChangelog: AppChangelogUiState,
    appUpdate: AppUpdateUiState?,
    onCheckForUpdate: () -> Unit,
    openBubble: () -> Unit,
    startupSplashFinished: Boolean,
    launchCamera: () -> Unit,
    startRealtimeWithPermission: () -> Unit,
    onSaveSplitOnly: (ImageSplitOptions) -> Unit,
) {
    // region: Pickers, viewers and dialogs
    ModalHost(overlays.modelPicker) { ModelPicker(state, onDismiss = { overlays.modelPicker = false }, onSelect = { model.chooseModelByUser(it); overlays.modelPicker = false }) }

    ModalHost(overlays.libraryOpen) { LibraryDialog(state, model, onOpenFile = { overlays.viewingFile = it }, onDismiss = { overlays.libraryOpen = false }) }
    ModalValueHost(overlays.viewingFile) { request ->
        FileViewerDialog(
            request, loader,
            download = { model.downloadAttachment(it) },
            onDismiss = { overlays.viewingFile = null },
            onOpenExternal = onFile,
            imageActions = imageActions,
        )
    }
    ModalHost(overlays.skillEditorOpen) {
        val editing = overlays.skillEditing
        SkillEditorDialog(skill = editing, onDismiss = { overlays.skillEditorOpen = false }) { n, d, i, done ->
            model.saveSkill(editing?.uuid, n, d, i) { ok -> if (ok) overlays.skillEditorOpen = false; done(ok) }
        }
    }
    state.mcpDecision?.let { decision -> key(decision.id) { McpDecisionDialog(decision, model::resolveMcpDecision) } }
    state.accountLock?.let { lock -> AccountLockOverlay(lock, model::accountLockExpired) }
    state.sessionTurnstileUrl?.let { url ->
        LaunchedEffect(url) { onWeb(url) }
        SessionTurnstileOverlay(onOpen = { onWeb(url) }, onDismiss = model::dismissSessionTurnstile)
    }
    state.apiKeyPrompt?.let { modelId ->
        val useFree = state.geminiFreeTier && state.preferences?.geminiBackend != "vertex_ai"
        apiKeyInfoFor(modelId, useFree)?.let { info ->
            ApiKeyRequiredDialog(model.modelDisplayName(modelId), modelId, info, onSave = model::saveApiKeyAndResend,
                onSwitch = { model.dismissApiKeyPrompt(false); overlays.modelPicker = true },
                onCancel = { model.dismissApiKeyPrompt(true) })
        }
    }
    // endregion

    // region: Thread, history, skill and settings dialogs
    ModalHost(overlays.branchOpen && state.selected != null) {
        BranchManagerDialog(state, onDismiss = { overlays.branchOpen = false }, onSwitch = model::switchBranch,
            onDelete = model::deleteMessage, notify = model::notify)
    }
    ModalHost(overlays.historyOpen) { HistoryDialog(state, sidebarActions(closeDrawer), onDismiss = { overlays.historyOpen = false }) }
    ModalHost(overlays.alphaOpen) { AlphaInfoDialog(onDismiss = { overlays.alphaOpen = false }) }
    ModalValueHost(overlays.legalKind) { kind -> LegalDialog(kind, model::legalMarkdown, onDismiss = { overlays.legalKind = null }) }
    // Web `confirm("Delete?")` / `prompt("Title:")` for Skills and threads.
    overlays.skillDeleting?.let { skill -> BrowserConfirmDialog("Delete?") { ok -> if (ok) model.deleteSkill(skill); overlays.skillDeleting = null } }
    overlays.renaming?.let { thread ->
        ThreadRenameDialog(thread.title) { title -> if (!title.isNullOrBlank()) model.renameThread(thread, title.trim()); overlays.renaming = null }
    }
    // endregion

    // region: Settings, studio and input dialogs
    ModalHost(overlays.settingsOpen) {
        SettingsDialog(state, model, onDismiss = { overlays.settingsOpen = false }, initialTab = overlays.settingsTab,
            initialCard = overlays.settingsCard,
            onLogout = { model.logout(); overlays.settingsOpen = false; closeDrawer() }, onWeb = onWeb,
            appUpdate = appUpdate ?: AppUpdateUiState(), onCheckForUpdate = onCheckForUpdate,
            onBubble = { openBubble(); overlays.settingsOpen = false })
    }
    ModalHost(overlays.compressionOpen) { CompressionDialog(state, model) { overlays.compressionOpen = false } }
    ModalHost(overlays.changelogOpen) {
        AppChangelogDialog(
            state = appChangelog,
            onDismiss = { overlays.changelogOpen = false },
            onRetry = onRetryChangelog,
        )
    }
    ModalHost(overlays.advancedOpen) {
        BatchDialog(
            state, onLoad = model::loadBatchJobs,
            onOpen = { job -> overlays.advancedOpen = false; model.openThreadId(job.threadId) },
            onCancel = model::cancelBatchJob, onDelete = model::deleteBatchJob,
            onDismiss = { overlays.advancedOpen = false },
        )
    }
    ModalHost(overlays.realtimeOpen) {
        RealtimeStudioDialog(state, model, overlays.realtimeOptions, { overlays.realtimeOptions = it }, onDismiss = { overlays.realtimeOpen = false },
            onStart = startRealtimeWithPermission)
    }
    ModalHost(overlays.lyriaOpen) { LyriaStudioDialog(state, model, onDismiss = { overlays.lyriaOpen = false }) }
    ModalHost(overlays.richPasteOpen) {
        RichPasteDialog(state.preferences, onDismiss = { overlays.richPasteOpen = false },
            onSavePrompt = model::saveRichPastePrompt, notify = model::notify) { text ->
            model.draft(if (state.draft.isBlank()) text else state.draft.trimEnd() + "\n\n" + text)
        }
    }
    // endregion

    // region: Attachments, media and image tools
    if (overlays.attachMenu) UploadSheet(
        state, model, loader,
        onDismiss = { overlays.attachMenu = false },
        onPickFiles = { overlays.attachMenu = false; overlays.pendingAttachmentPicker = AttachmentPicker.Files },
        onCamera = launchCamera,
        onPhotos = { overlays.attachMenu = false; overlays.pendingAttachmentPicker = AttachmentPicker.Photos },
        onSplitImage = {
            overlays.attachMenu = false
            overlays.pendingAttachmentPicker = AttachmentPicker.Split
        },
        onLibrary = { overlays.libraryOpen = true },
        onChangeVisionModel = { overlays.visionPicker = true },
        onOpenFile = openPendingAttachment,
        onEditImage = { overlays.markerMaskMode = false; overlays.markerTarget = it },
    )
    ModalValueHost(sharedTextPreview.takeIf { startupSplashFinished }) { text ->
        // The chat's code-block actions, so code blocks get the same buttons as in an answer.
        ProvideMarkdownCodeActions(codeActions) {
            SharedTextPreviewDialog(text, loader, openInApp, onDismiss = model::closeSharedTextPreview)
        }
    }
    ModalValueHost(imageSplit) { request ->
        ImageSplitDialog(request, onDismiss = model::closeImageSplit,
            onAttach = { options -> model.runImageSplit(options, attach = true) },
            onSaveOnly = onSaveSplitOnly)
    }
    ModalHost(overlays.cameraOpen) {
        InAppCameraDialog(
            onDismiss = { overlays.cameraOpen = false },
            onCaptured = { uris, split ->
                overlays.cameraOpen = false
                if (split) model.openImageSplit(uris) else model.upload(uris)
            },
            onError = model::notify,
        )
    }
    ModalHost(overlays.visionPicker) {
        ModelPicker(state, onDismiss = { overlays.visionPicker = false },
            onSelect = { model.setVisionModel(it); overlays.visionPicker = false },
            selectedId = state.visionModel ?: state.preferences?.defaultVisionModel.orEmpty(), lockToPromptCache = false)
    }
    ModalValueHost(overlays.markerTarget) { target ->
        ImageMarkerEditor(target, loader, maskMode = overlays.markerMaskMode, onDismiss = { overlays.markerTarget = null; overlays.markerMaskMode = false },
            onSave = { png, attachOriginal ->
                if (overlays.markerMaskMode) model.uploadImageMask(png) else model.applyImageEdit(target.reference, png, attachOriginal)
                overlays.markerTarget = null
                overlays.markerMaskMode = false
            },
            onError = model::notify)
    }
    ModalHost(overlays.threadSettings && state.selected != null) {
        ChatInstructionsDialog(state, onRefresh = model::loadPreferences, onDismiss = { overlays.threadSettings = false }) {
            instruction, includeGlobal, userPrompt, done ->
            model.saveChatInstructions(instruction, includeGlobal, userPrompt) { ok -> if (ok) overlays.threadSettings = false; done(ok) }
        }
    }
    overlays.deleting?.let { thread -> BrowserConfirmDialog("Delete?") { ok -> if (ok) model.deleteThread(thread, onNewChat = closeDrawer); overlays.deleting = null } }
    // endregion
}
