package com.minashin1120.aiplayground.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.IntentSenderRequest
import androidx.activity.result.ActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.Crossfade
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.scaleIn
import androidx.compose.animation.EnterTransition
import androidx.compose.animation.ExitTransition
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.animation.togetherWith
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.tween
import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.app.Activity
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.layout.boundsInWindow
import androidx.compose.ui.layout.positionInWindow
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.LocalIndication
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.autofill.ContentType
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentType
import androidx.compose.ui.semantics.semantics
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.scrollBy
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.input.nestedscroll.NestedScrollConnection
import androidx.compose.ui.input.nestedscroll.NestedScrollSource
import androidx.compose.ui.input.nestedscroll.nestedScroll
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.ui.input.pointer.PointerEventPass
import androidx.compose.ui.input.pointer.PointerInputChange
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalLayoutDirection
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalAccessibilityManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.core.content.FileProvider
import androidx.core.content.ContextCompat
import com.minashin1120.aiplayground.ANDROID_17_APP_BUBBLE_API
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.ChatViewModel
import com.minashin1120.aiplayground.ChatTransitionKind
import com.minashin1120.aiplayground.ImportSettingChange
import com.minashin1120.aiplayground.AppChangelogUiState
import com.minashin1120.aiplayground.AppUpdateUiState
import com.minashin1120.aiplayground.data.ChatAutoScroll
import com.minashin1120.aiplayground.data.isImageReference
import com.minashin1120.aiplayground.data.isIdeogramObjectRemoval
import com.minashin1120.aiplayground.data.IMAGE_SPLIT_MAX_IMAGES
import com.minashin1120.aiplayground.data.ImageSplitOptions
import com.minashin1120.aiplayground.data.ActivityLog
import com.minashin1120.aiplayground.data.Diagnostics
import com.minashin1120.aiplayground.data.ChatMessage
import com.minashin1120.aiplayground.data.buildTokenTotals
import com.minashin1120.aiplayground.data.AI_SETTING_JUMP_TARGETS
import com.minashin1120.aiplayground.data.apiKeyInfoFor
import com.minashin1120.aiplayground.data.recentWebModels
import com.minashin1120.aiplayground.data.ConnectionStatus
import com.minashin1120.aiplayground.data.defaultMessage

import com.minashin1120.aiplayground.data.attachmentKind
import com.minashin1120.aiplayground.data.attachmentKindIcon
import com.minashin1120.aiplayground.data.numericId
import com.minashin1120.aiplayground.data.siblingGroup
import com.minashin1120.aiplayground.data.PasskeyClient
import com.minashin1120.aiplayground.data.GoogleAuthClient
import com.minashin1120.aiplayground.data.ServerOrigin
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.launch
import kotlin.math.abs
import android.content.Context
import android.content.ContextWrapper
import java.io.File

/** Phone layout keeps the history drawer closed until it has settled off-screen. */
internal fun shouldCoverPhoneHistoryUntilClosed(
    starting: Boolean,
    showThreads: Boolean,
    wideLayout: Boolean,
    drawerSettledClosed: Boolean,
): Boolean = starting || (showThreads && !wideLayout && !drawerSettledClosed)

/** Credential Manager needs the hosting Activity, not an arbitrary wrapper Context. */
private fun Context.findActivity(): Activity? {
    var current: Context? = this
    while (current is ContextWrapper) {
        if (current is Activity) return current
        current = current.baseContext
    }
    return current as? Activity
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PlaygroundScreen(
    model: ChatViewModel,
    onWeb: (String) -> Unit,
    onFile: (String) -> Unit,
    appChangelog: AppChangelogUiState = AppChangelogUiState(),
    appUpdate: AppUpdateUiState? = null,
    playStartupAnimation: Boolean = true,
    onDismissUpdate: () -> Unit = {},
    onDownloadUpdate: () -> Unit = {},
    onCancelDownload: () -> Unit = {},
    onHideUpdate: () -> Unit = {},
    onShowUpdate: () -> Unit = {},
    onRetryUpdate: () -> Unit = {},
    onInstallUpdate: () -> Unit = {},
    onCheckForUpdate: () -> Unit = {},
    onOpenChangelog: () -> Unit = {},
    onRetryChangelog: () -> Unit = {},
    onOpenBubble: () -> Unit = {},
) {
    val state by model.state.collectAsStateWithLifecycle()
    val imageSplit by model.imageSplit.collectAsStateWithLifecycle()
    val sharedTextPreview by model.sharedTextPreview.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val minimalPromptMode = state.preferences?.effectivePromptBarMode == "minimal"
    val gestureNavigation = remember(context) {
        val modeId = context.resources.getIdentifier("config_navBarInteractionMode", "integer", "android")
        modeId != 0 && context.resources.getInteger(modeId) == 2
    }
    val edgeToEdgeMinimal = gestureNavigation && minimalPromptMode
    val contentInsets = if (edgeToEdgeMinimal) {
        WindowInsets.safeDrawing.only(WindowInsetsSides.Top + WindowInsetsSides.Horizontal)
    } else {
        WindowInsets.safeDrawing
    }
    // Web applies the light theme for a light system setting (theme-light.css) or manual light mode.
    PlaygroundTheme(darkTheme = isSystemInDarkTheme() && state.preferences?.lightModeEnabled != true,
        themeColor = state.preferences?.themeColor, liquidGlass = state.preferences?.liquidGlassEnabled == true) {
        val colors = MaterialTheme.colorScheme
        val reduceMotion = rememberReduceMotion()
        val animationsEnabled = !reduceMotion
        val startupSplashEnabled = playStartupAnimation && !reduceMotion
        // The message preview opens as a separate full-screen dialog window, which would hide the intro.
        // Keep it closed until the intro has finished (or immediately when the intro is disabled).
        var startupSplashFinished by remember(startupSplashEnabled) { mutableStateOf(!startupSplashEnabled) }
        CompositionLocalProvider(
            LocalContentColor provides colors.onBackground,
            LocalReduceMotion provides reduceMotion,
            LocalAttachmentDownloader provides model::downloadAttachment,
            LocalImageCacheKey provides model::attachmentImageCacheKey,
        ) {
            InAppNoticeProvider {
            BoxWithConstraints(
                Modifier.fillMaxSize().background(
                    Brush.verticalGradient(listOf(colors.background, colors.surfaceContainerLow))
                )
            ) {
            // Match the Web breakpoint while retaining Android's portrait/landscape semantics.
            val layoutClass = playgroundLayoutClass(maxWidth, maxHeight)
            val wide = layoutClass == PlaygroundLayoutClass.Tablet
            var allowDrawerOpen by remember { mutableStateOf(false) }
            // Do not use rememberDrawerState: its saveable state can restore Open after process death.
            val drawer = remember {
                DrawerState(DrawerValue.Closed) { value ->
                    value == DrawerValue.Closed || allowDrawerOpen
                }
            }
            val scope = rememberCoroutineScope()
            var googleIdentityResult by remember { mutableStateOf<CompletableDeferred<ActivityResult>?>(null) }
            val googleIdentityLauncher = rememberLauncherForActivityResult(
                ActivityResultContracts.StartIntentSenderForResult(),
            ) { result -> googleIdentityResult?.complete(result) }
            // Runs the WebAuthn ceremony the ViewModel requested (sign-in, 2FA or passkey
            // registration) through Android Credential Manager, then hands the result back.
            LaunchedEffect(state.credentialRequest) {
                val request = state.credentialRequest ?: return@LaunchedEffect
                try {
                    val credentialJson = if (request.kind == "register") {
                        PasskeyClient.create(context.findActivity() ?: context, request.publicKeyJson)
                    } else {
                        PasskeyClient.get(context.findActivity() ?: context, request.publicKeyJson)
                    }
                    model.submitCredential(credentialJson)
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    model.cancelCredentialRequest("パスキー操作を完了できませんでした。")
                }
            }
            LaunchedEffect(state.googleLoginRequest) {
                if (state.googleLoginRequest == 0L) return@LaunchedEffect
                try {
                    val activity = context.findActivity() ?: error("Googleログインを開始できません。")
                    val credential = GoogleAuthClient.getIdToken(
                        activity,
                        state.googleServerClientId,
                    ) { pendingIntent ->
                        val result = CompletableDeferred<ActivityResult>()
                        googleIdentityResult = result
                        try {
                            googleIdentityLauncher.launch(
                                IntentSenderRequest.Builder(pendingIntent).build(),
                            )
                            val activityResult = result.await()
                            if (activityResult.resultCode != Activity.RESULT_OK) {
                                throw IllegalStateException("Googleログインがキャンセルされました。")
                            }
                            activityResult.data
                        } finally {
                            if (googleIdentityResult === result) googleIdentityResult = null
                        }
                    }
                    model.completeGoogleLogin(credential)
                } catch (e: CancellationException) {
                    throw e
                } catch (e: GoogleAuthClient.AuthException) {
                    model.cancelGoogleLogin(e.message ?: "Googleログインに失敗しました。", e.diagnosticLog)
                } catch (e: Exception) {
                    model.cancelGoogleLogin(e.message ?: "Googleログインに失敗しました。")
                }
            }
            val openDrawer: () -> Unit = {
                allowDrawerOpen = true
                scope.launch { drawer.open() }
            }
            val overlays = remember { PlaygroundOverlays() }
            var composerHeight by remember { mutableIntStateOf(0) }
            var bubbleAfterNotificationPermission by remember { mutableStateOf(false) }
            // ログの収集を強化: which modals and panels are open (names only).
            val panelStates = listOf(
                "model_picker" to overlays.modelPicker, "thread_settings" to overlays.threadSettings, "attach_menu" to overlays.attachMenu,
                "library" to overlays.libraryOpen, "file_view" to (overlays.viewingFile != null), "skill_editor" to overlays.skillEditorOpen,
                "skill_delete" to (overlays.skillDeleting != null), "history" to overlays.historyOpen, "branch" to overlays.branchOpen, "alpha" to overlays.alphaOpen,
                "legal" to (overlays.legalKind != null), "rename" to (overlays.renaming != null), "delete_thread" to (overlays.deleting != null),
                "settings" to overlays.settingsOpen, "compression" to overlays.compressionOpen, "changelog" to overlays.changelogOpen,
                "advanced" to overlays.advancedOpen, "realtime" to overlays.realtimeOpen, "lyria" to overlays.lyriaOpen, "rich_paste" to overlays.richPasteOpen,
                "marker" to (overlays.markerTarget != null), "vision_picker" to overlays.visionPicker, "camera" to overlays.cameraOpen,
            )
            val openPanels = panelStates.filter { it.second }.joinToString(",") { it.first }
            LaunchedEffect(openPanels) { ActivityLog.log("ui.panels", *panelStates.map { it.first to it.second }.toTypedArray()) }
            LaunchedEffect(drawer) { snapshotFlow { drawer.isOpen }.collect { ActivityLog.log("ui.drawer", "open" to it) } }
            val notifications = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
                if (bubbleAfterNotificationPermission) {
                    bubbleAfterNotificationPermission = false
                    if (granted) onOpenBubble() else model.notify("バブルを使うには通知権限が必要です。")
                }
            }
            val openBubble: () -> Unit = {
                if (Build.VERSION.SDK_INT < ANDROID_17_APP_BUBBLE_API &&
                    Build.VERSION.SDK_INT >= 33 &&
                    ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
                    bubbleAfterNotificationPermission = true
                    notifications.launch(Manifest.permission.POST_NOTIFICATIONS)
                } else {
                    onOpenBubble()
                }
            }
            val microphone = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
                if (overlays.awaitingMic) {
                    overlays.awaitingMic = false
                    val startDock = overlays.startDockAfterMic
                    overlays.startDockAfterMic = false
                    if (!granted) model.notify("Realtime音声にはマイクの権限が必要です。")
                    else if (startDock) startRealtimeWith(model, model.state.value.model, overlays.realtimeOptions)
                    else overlays.realtimeOpen = true
                }
            }
            // The composer voice dock and the studio dialog share this start: a missing microphone
            // permission is requested first and remembered by the launcher above.
            val startRealtimeWithPermission: () -> Unit = {
                if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                    startRealtimeWith(model, state.model, overlays.realtimeOptions)
                } else {
                    overlays.awaitingMic = true
                    overlays.startDockAfterMic = true
                    microphone.launch(Manifest.permission.RECORD_AUDIO)
                }
            }
            // Batch results and answers finished while the app is away are notified.
            LaunchedEffect(state.account?.id, state.localProfile) {
                if ((state.account != null || state.localProfile) && Build.VERSION.SDK_INT >= 33 &&
                    ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
                    notifications.launch(Manifest.permission.POST_NOTIFICATIONS)
                }
            }
            val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { model.upload(it) }
            // Web recording mic: record, then transcribe with the STT API or the current LLM.
            val recordPermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
                if (granted) model.toggleMicRecording() else model.notify("Microphone access denied or not available.")
            }
            val launchSpeech: () -> Unit = {
                val selected = state.model
                when {
                    // Web `mic-btn` with a voice model starts (or stops) the voice session in the dock.
                    isRealtimeAudioModel(state.account?.models?.firstOrNull { it.id == selected }) -> when {
                        state.realtime.active -> model.stopRealtime(save = true)
                        ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED ->
                            startRealtimeWith(model, selected, overlays.realtimeOptions)
                        else -> { overlays.awaitingMic = true; overlays.startDockAfterMic = true; microphone.launch(Manifest.permission.RECORD_AUDIO) }
                    }
                    ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED ->
                        model.toggleMicRecording()
                    else -> recordPermission.launch(Manifest.permission.RECORD_AUDIO)
                }
            }
            val photoPicker = rememberLauncherForActivityResult(ActivityResultContracts.PickMultipleVisualMedia()) { model.upload(it) }
            val splitPicker = rememberLauncherForActivityResult(
                ActivityResultContracts.PickMultipleVisualMedia(IMAGE_SPLIT_MAX_IMAGES),
            ) { model.openImageSplit(it) }
            // Android 17 (API 37) can leave the app window without input focus after the system Files
            // picker (DocumentsUI) returns, which then looks like a crash / frozen app (feedback
            // fccc59662437). Start the picker only after the upload sheet's dialog window is gone.
            LaunchedEffect(overlays.pendingAttachmentPicker) {
                val action = overlays.pendingAttachmentPicker ?: return@LaunchedEffect
                withFrameNanos { }
                try {
                    when (action) {
                        AttachmentPicker.Files -> picker.launch(arrayOf("*/*"))
                        AttachmentPicker.Photos -> photoPicker.launch(
                            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageAndVideo))
                        AttachmentPicker.Split -> splitPicker.launch(
                            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                    }
                    ActivityLog.log("ui.show", "source" to "picker")
                } catch (e: Exception) {
                    Diagnostics.failure("error", e, "phase" to "error")
                    model.notify("ファイル選択を開けませんでした。もう一度お試しください。")
                } finally {
                    if (overlays.pendingAttachmentPicker == action) overlays.pendingAttachmentPicker = null
                }
            }
            // 画像分割 「保存のみ」 before Android 10: MediaStore needs a storage permission there, so the user picks a folder.
            val splitFolderPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { tree ->
                val options = overlays.pendingSplitSave
                overlays.pendingSplitSave = null
                if (tree != null && options != null) model.runImageSplit(options, attach = false, tree = tree)
            }
            val saveSplitOnly: (ImageSplitOptions) -> Unit = { options ->
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) model.runImageSplit(options, attach = false)
                else {
                    overlays.pendingSplitSave = options
                    try { splitFolderPicker.launch(null) }
                    catch (ignored: Exception) { overlays.pendingSplitSave = null; model.notify("保存先のフォルダーを選択できません。") }
                }
            }
            val maskPicker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
                if (uri != null) model.uploadImageMask(uri)
            }
            val launchCamera: () -> Unit = { overlays.attachMenu = false; overlays.cameraOpen = true }
            val notices = checkNotNull(LocalInAppNotices.current)
            val noticeAccessibility = LocalAccessibilityManager.current
            val loader: FileBytesLoader = { reference, thumbnail, limit ->
                model.loadAttachmentBytes(reference, thumbnail, limit)
            }
            val openInApp: (String) -> Unit = { reference ->
                // Web `openImageViewer` steps through every `.chat-image` on screen: reply images, then attachments.
                val gallery = state.messages.flatMap { message ->
                    (if (message.role == "user") emptyList() else markdownImageReferences(message.content)) +
                        message.files.filter { isImageReference(it) }
                }.plus(markdownImageReferences(state.liveContent)).distinct()
                overlays.viewingFile = FileViewRequest(reference, gallery = gallery)
            }
            val openPendingAttachment: (String) -> Unit = { reference ->
                overlays.viewingFile = FileViewRequest(reference,
                    gallery = state.attachments.filter { isImageReference(it.name) || it.mime.startsWith("image/") }.map { it.reference },
                    editAttachment = state.attachments.any { it.reference == reference },
                )
            }
            // Web image viewer "Download": Android asks where to save it (ANDROID_ONLY.md §2).
            var pendingFileDownload by remember { mutableStateOf<String?>(null) }
            val fileSaver = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/octet-stream")) { uri ->
                val reference = pendingFileDownload
                pendingFileDownload = null
                if (uri != null && reference != null) scope.launch {
                    runCatching {
                        val (local, _) = model.downloadAttachment(reference)
                        kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
                            context.contentResolver.openOutputStream(uri)?.use { out -> local.inputStream().use { it.copyTo(out) } }
                            local.delete()
                        }
                    }.onFailure { model.notify("ダウンロードに失敗しました") }
                }
            }
            val viewerClipboard = LocalClipboardManager.current
            val imageActions = ImageViewerActions(
                onDownload = { reference -> pendingFileDownload = reference; fileSaver.launch(fileViewerTitle(reference)) },
                onCopyUrl = { reference ->
                    viewerClipboard.setText(androidx.compose.ui.text.AnnotatedString(
                        com.minashin1120.aiplayground.data.ServerOrigin.base + "/files/" + reference.removePrefix("/files/")))
                    model.notify("画像URLをコピーしました")
                },
                onReuse = { reference -> model.reuseImage(reference) },
                onEdit = { reference -> overlays.markerMaskMode = false; overlays.markerTarget = state.attachments.firstOrNull { it.reference == reference } },
            )
            val sharePdf: () -> Unit = {
                model.exportPdf { file ->
                    try {
                        val uri = FileProvider.getUriForFile(context, "${context.packageName}.files", file)
                        val intent = Intent(Intent.ACTION_SEND).setType("application/pdf")
                            .putExtra(Intent.EXTRA_STREAM, uri)
                            .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                        context.startActivity(Intent.createChooser(intent, "PDFを共有"))
                    } catch (_: Exception) { model.notify("PDFを共有できません。") }
                }
            }
            // Code-block download: Web saves `code.<ext>`; Android asks where to save it (ANDROID_ONLY.md §2).
            var pendingCodeDownload by remember { mutableStateOf<String?>(null) }
            val codeSaver = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("text/plain")) { uri ->
                val code = pendingCodeDownload
                pendingCodeDownload = null
                if (uri != null && code != null) scope.launch(kotlinx.coroutines.Dispatchers.IO) {
                    runCatching { context.contentResolver.openOutputStream(uri)?.use { it.write(code.toByteArray(Charsets.UTF_8)) } }
                        .onFailure { model.notify("ダウンロードに失敗しました") }
                }
            }
            val codingKey = state.codingTarget?.id
            val codeActions = remember(model, codingKey) {
                MarkdownCodeActions(
                    onDownload = { code, language -> pendingCodeDownload = code; codeSaver.launch(codeDownloadName(language)) },
                    onCodingTarget = { code, language ->
                        val lang = language.ifBlank { "text" }
                        model.selectCodingTarget(com.minashin1120.aiplayground.data.CodingTarget(
                            com.minashin1120.aiplayground.data.codingTargetKey(lang, code), code, lang, ""))
                    },
                    selectedCodingKey = codingKey,
                )
            }
            LaunchedEffect(state.settingsRequest) {
                if (state.settingsRequest > 0L) {
                    state.settingsRequestTab?.let { overlays.settingsTab = it }
                    overlays.settingsCard = state.settingsRequestCard
                    overlays.settingsOpen = true
                }
            }
            LaunchedEffect(overlays.settingsOpen) { if (!overlays.settingsOpen) overlays.settingsCard = null }
            LaunchedEffect(state.notice) {
                state.notice?.let {
                    val duration = noticeAccessibility?.calculateRecommendedTimeoutMillis(
                        NOTICE_DURATION_MILLIS, containsText = true, containsControls = true,
                    ) ?: NOTICE_DURATION_MILLIS
                    notices.add(it, duration)
                    model.dismissNotice()
                }
            }
            val closeDrawer: () -> Unit = { scope.launch { drawer.close() } }
            val showThreads = state.account != null
            LaunchedEffect(state.banned) {
                if (state.banned) {
                    allowDrawerOpen = false
                    overlays.deleting = null
                    overlays.renaming = null
                    overlays.skillDeleting = null
                    overlays.modelPicker = false
                    overlays.threadSettings = false
                    overlays.attachMenu = false
                    overlays.libraryOpen = false
                    overlays.viewingFile = null
                    overlays.skillEditorOpen = false
                    overlays.historyOpen = false
                    overlays.branchOpen = false
                    overlays.alphaOpen = false
                    overlays.legalKind = null
                    overlays.settingsOpen = false
                    overlays.changelogOpen = false
                    overlays.advancedOpen = false
                    overlays.realtimeOpen = false
                    overlays.lyriaOpen = false
                    overlays.richPasteOpen = false
                    overlays.markerTarget = null
                    overlays.visionPicker = false
                    closeDrawer()
                }
            }
            val hasOverlay = overlays.modelPicker || overlays.threadSettings || overlays.attachMenu || overlays.libraryOpen || overlays.viewingFile != null ||
                overlays.skillEditorOpen || overlays.historyOpen || overlays.branchOpen || overlays.alphaOpen || overlays.legalKind != null ||
                overlays.settingsOpen || overlays.changelogOpen || overlays.advancedOpen || overlays.realtimeOpen || overlays.lyriaOpen || overlays.richPasteOpen || overlays.markerTarget != null || overlays.visionPicker || overlays.compressionOpen
            BackHandler(enabled = hasOverlay || (!wide && drawer.currentValue == DrawerValue.Open) || state.canGoBackInChats) {
                when {
                    overlays.markerTarget != null -> overlays.markerTarget = null
                    overlays.visionPicker -> overlays.visionPicker = false
                    overlays.attachMenu -> overlays.attachMenu = false
                    overlays.compressionOpen -> overlays.compressionOpen = false
                    overlays.modelPicker -> overlays.modelPicker = false
                    overlays.threadSettings -> overlays.threadSettings = false
                    overlays.settingsOpen -> overlays.settingsOpen = false
                    overlays.changelogOpen -> overlays.changelogOpen = false
                    overlays.advancedOpen -> overlays.advancedOpen = false
                    overlays.realtimeOpen -> overlays.realtimeOpen = false
                    overlays.lyriaOpen -> overlays.lyriaOpen = false
                    overlays.richPasteOpen -> overlays.richPasteOpen = false
                    overlays.viewingFile != null -> overlays.viewingFile = null
                    overlays.libraryOpen -> overlays.libraryOpen = false
                    overlays.skillEditorOpen -> overlays.skillEditorOpen = false
                    overlays.alphaOpen -> overlays.alphaOpen = false
                    overlays.legalKind != null -> overlays.legalKind = null
                    overlays.historyOpen -> overlays.historyOpen = false
                    overlays.branchOpen -> overlays.branchOpen = false
                    !wide && drawer.currentValue == DrawerValue.Open -> closeDrawer()
                    state.canGoBackInChats -> model.goBackInChats()
                }
            }
            LaunchedEffect(showThreads, wide, state.starting) {
                allowDrawerOpen = false
                if (!wide) {
                    // Close under the startup spinner so the open-then-close animation is never shown.
                    repeat(3) {
                        drawer.snapTo(DrawerValue.Closed)
                        withFrameNanos { }
                    }
                }
                if (!state.starting && showThreads && !wide) allowDrawerOpen = true
            }

            val versionLabel = "V${com.minashin1120.aiplayground.BuildConfig.VERSION_NAME}"
            val openExternal: (String) -> Unit = { url ->
                try { context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url))) }
                catch (_: Exception) { model.notify("ブラウザーを開けません。") }
            }
            fun sidebarActions(onNavigate: () -> Unit) = SidebarActions(
                onNavigate = onNavigate,
                onChangelog = { overlays.changelogOpen = true; onOpenChangelog() },
                onHistory = { overlays.historyOpen = true },
                onLowBandwidth = model::cycleLowBandwidth,
                onSettings = { overlays.settingsOpen = true },
                onLibrary = { overlays.libraryOpen = true },
                onNewChat = { model.newChat() },
                onBatch = { overlays.advancedOpen = true },
                onBranches = { if (state.selected == null) model.notify("チャットを選択してください") else overlays.branchOpen = true },
                onPdf = sharePdf,
                onExternal = openExternal,
                onSearch = model::search,
                // ANDROID_ONLY.md: the Drawer closes on tap and the chat shows its loading skeleton until the history arrives.
                onOpenThread = { thread -> overlays.historyOpen = false; onNavigate(); model.openThread(thread) },
                onBookmark = model::toggleBookmark,
                onRenameThread = { overlays.renaming = it },
                onDeleteThread = { overlays.deleting = it },
                onMoreThreads = model::moreThreads,
                onRefreshThreads = model::reloadThreads,
                onChooseSkill = { skill -> model.chooseSkill(skill) },
                onNewSkill = { overlays.skillEditing = null; overlays.skillEditorOpen = true },
                onEditSkill = { skill -> overlays.skillEditing = skill; overlays.skillEditorOpen = true },
                onDeleteSkill = { overlays.skillDeleting = it },
                onRefreshSkills = model::reloadSkills,
                onHelp = { onWeb("/help") },
                onLegal = { overlays.legalKind = it },
                onAlphaInfo = { overlays.alphaOpen = true },
                onLogout = { model.logout(); closeDrawer() },
            )

            val content: @Composable () -> Unit = {
                PromptSendMotionHost(state) {
                Scaffold(
                    modifier = Modifier.imePadding(),
                    containerColor = Color.Transparent,
                    contentWindowInsets = contentInsets,
                    topBar = {
                        // Web shows `header.main-chrome-header` below the md breakpoint only.
                        if (showThreads && !wide) MobileChatHeader(state, onMenu = openDrawer, onNewChat = { model.newChat() }, onPdf = sharePdf,
                            onOpenWeb = { onWeb(webChatPath(state)) })
                    }, snackbarHost = {},
                    bottomBar = {
                        if (showThreads) CompositionLocalProvider(LocalComposerEstimator provides model::estimatePromptTokens,
                            LocalComposerDetailsEstimator provides model::estimatePromptTokenDetails) {
                        Composer(state, model, { overlays.modelPicker = true }, { overlays.attachMenu = true }, launchSpeech,
                            onRichPaste = { overlays.richPasteOpen = true }, onMask = {
                                if (isIdeogramObjectRemoval(state.model, state.generationValues)) {
                                    val image = state.attachments.firstOrNull { it.mime.startsWith("image/") || isImageReference(it.name) }
                                    if (image == null) model.notify("先にマスク対象の画像を添付してください")
                                    else { overlays.markerMaskMode = true; overlays.markerTarget = image }
                                } else maskPicker.launch("image/*")
                            },
                            onSettings = { overlays.settingsTab = "一般"; overlays.settingsOpen = true },
                            onChatInstructions = { model.ensureThread { overlays.threadSettings = true } },
                            onCompressionSettings = { overlays.compressionOpen = true },
                            onTemporarySettings = { model.requestSettings(SettingsTab.General.id, "temp-chat") },
                            loader = loader, onOpenFile = openPendingAttachment, onDockHeight = { composerHeight = it },
                            onRealtime = {
                                if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) overlays.realtimeOpen = true
                                else { overlays.awaitingMic = true; microphone.launch(Manifest.permission.RECORD_AUDIO) }
                            },
                            onLyria = { overlays.lyriaOpen = true },
                            realtimeOptions = overlays.realtimeOptions,
                            onRealtimeOptions = { overlays.realtimeOptions = it },
                            onRealtimeStart = startRealtimeWithPermission)
                        }
                    }
                ) { padding ->
                    // The minimal composer floats above the list. Reserving bottomBar height here
                    // would clip the list before it can draw behind the transparent gesture area.
                    Column(Modifier.fillMaxSize().padding(
                        top = padding.calculateTopPadding(),
                        bottom = if (edgeToEdgeMinimal) 0.dp else padding.calculateBottomPadding(),
                        start = padding.calculateStartPadding(LocalLayoutDirection.current),
                        end = padding.calculateEndPadding(LocalLayoutDirection.current),
                    )) {
                        val screen = when {
                            state.starting -> PlaygroundScreenKind.Starting
                            state.setupRequired -> PlaygroundScreenKind.Setup
                            state.account == null -> PlaygroundScreenKind.Auth
                            else -> PlaygroundScreenKind.Chat
                        }
                        // Web `#top-model-bar` (minimal prompt bar): the model button under the header.
                        AnimatedVisibility(screen == PlaygroundScreenKind.Chat && state.preferences?.effectivePromptBarMode == "minimal",
                            enter = expandFadeIn(reduceMotion), exit = shrinkFadeOut(reduceMotion)) {
                            TopModelBar(state) { overlays.modelPicker = true }
                        }
                        AnimatedContent(
                            targetState = screen,
                            modifier = Modifier.weight(1f).fillMaxWidth(),
                            transitionSpec = {
                                if (reduceMotion) EnterTransition.None togetherWith ExitTransition.None
                                else (fadeIn(tween(PlaygroundMotion.LONG, delayMillis = 60, easing = PlaygroundMotion.Standard)) +
                                    scaleIn(tween(PlaygroundMotion.LONG, delayMillis = 60, easing = PlaygroundMotion.Emphasized), initialScale = 0.98f)) togetherWith
                                    fadeOut(tween(PlaygroundMotion.SHORT, easing = PlaygroundMotion.Exit))
                            },
                            label = "playground screen",
                        ) { target ->
                            Box(Modifier.fillMaxSize()) {
                                when (target) {
                                    PlaygroundScreenKind.Starting -> CircularProgressIndicator(Modifier.align(Alignment.Center))
                                    PlaygroundScreenKind.Setup -> SetupScreen(state, model, onWeb)
                                    PlaygroundScreenKind.Auth -> AuthScreen(state, model, onWeb)
                                    PlaygroundScreenKind.Chat -> ProvideMarkdownCodeActions(codeActions) {
                                        CompositionLocalProvider(LocalCanvasMode provides state.canvasMode) {
                                        Conversation(
                                            state, model, openInApp, loader,
                                            animationsEnabled = animationsEnabled,
                                            bottomOverlay = if (edgeToEdgeMinimal) with(LocalDensity.current) { composerHeight.toDp() } else 0.dp,
                                            onSettingJump = { key ->
                                                val target = AI_SETTING_JUMP_TARGETS[key]
                                                if (target?.richPaste == true) overlays.richPasteOpen = true
                                                else {
                                                    overlays.settingsTab = SettingsTab.entries.firstOrNull { it.id == target?.tab }?.label ?: "一般"
                                                    overlays.settingsOpen = true
                                                }
                                            },
                                        )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            }
            // region: Responsive layout and drawer
            if (showThreads && wide) {
                Row(Modifier.fillMaxSize()) {
                    Sidebar(state, versionLabel, sidebarActions {})
                    Box(Modifier.weight(1f)) { content() }
                }
            } else {
                val web = LocalWebPalette.current
                // Keep the moving drawer inexpensive to render; a full-screen blur makes swipe
                // tracking janky on devices where the chat content is already GPU-heavy.
                ModalNavigationDrawer(drawerState = drawer, gesturesEnabled = showThreads && allowDrawerOpen,
                    scrimColor = if (web.isLight) Color(15, 23, 42).copy(alpha = 0.32f) else Color(4, 8, 20).copy(alpha = 0.55f),
                    drawerContent = {
                        ModalDrawerSheet(
                            Modifier.width(PlaygroundDimens.drawerPane),
                            drawerShape = androidx.compose.ui.graphics.RectangleShape,
                            drawerContainerColor = Color.Transparent,
                            drawerTonalElevation = 0.dp,
                            windowInsets = WindowInsets(0, 0, 0, 0),
                        ) {
                            if (showThreads) Sidebar(state, versionLabel, sidebarActions(closeDrawer))
                        }
                    }) { Box { content() } }
            }

            // region: Canvas and transition overlays
            // Web `#canvas-panel` below 1024px: a full-screen layer over the header and the composer.
            if (showThreads && state.canvasMode && LocalConfiguration.current.screenWidthDp.dp < CANVAS_SIDE_PANEL_MIN_WIDTH) {
                androidx.compose.ui.window.Dialog(onDismissRequest = model::toggleCanvas,
                    properties = androidx.compose.ui.window.DialogProperties(usePlatformDefaultWidth = false, decorFitsSystemWindows = false)) {
                    NoticeWindowHost()
                    CanvasPanel(canvasSourceText(state), fullScreen = true, notify = model::notify, onClose = model::toggleCanvas,
                        modifier = Modifier.fillMaxSize().safeDrawingPadding())
                }
            }
            ChatTransitionVeil(
                navigationId = state.chatNavigationId,
                kind = state.chatNavigationKind,
                animationsEnabled = animationsEnabled,
            )

            AnimatedVisibility(
                visible = shouldCoverPhoneHistoryUntilClosed(state.starting && !startupSplashEnabled, showThreads, wide, allowDrawerOpen),
                enter = EnterTransition.None,
                exit = if (reduceMotion) ExitTransition.None else fadeOut(tween(PlaygroundMotion.MEDIUM, easing = PlaygroundMotion.Standard)),
                label = "startup cover",
            ) {
                Box(
                    Modifier.fillMaxSize().background(
                        Brush.verticalGradient(listOf(colors.background, colors.surfaceContainerLow))
                    )
                ) {
                    CircularProgressIndicator(Modifier.align(Alignment.Center))
                }
            }
            // endregion

            // region: Dialogs, sheets and confirmations
            PlaygroundDialogs(
                state = state,
                model = model,
                overlays = overlays,
                loader = loader,
                imageActions = imageActions,
                imageSplit = imageSplit,
                sharedTextPreview = sharedTextPreview,
                codeActions = codeActions,
                sidebarActions = { onNavigate -> sidebarActions(onNavigate) },
                closeDrawer = closeDrawer,
                openInApp = openInApp,
                openPendingAttachment = openPendingAttachment,
                onWeb = onWeb,
                onFile = onFile,
                onRetryChangelog = onRetryChangelog,
                appChangelog = appChangelog,
                appUpdate = appUpdate,
                onCheckForUpdate = onCheckForUpdate,
                openBubble = openBubble,
                startupSplashFinished = startupSplashFinished,
                launchCamera = launchCamera,
                startRealtimeWithPermission = startRealtimeWithPermission,
                onSaveSplitOnly = saveSplitOnly,
            )
            // endregion
            // region: Status overlays
            if (state.banned) BannedScreen(
                reason = state.banReason,
                bannedAt = state.banAt,
                appeal = state.banAppeal,
                onSubmitAppeal = model::submitBanAppeal,
                onLogout = model::logout,
            )
            if (!state.banned) appUpdate?.let { update ->
                if (!update.dialogHidden) AppUpdateDialog(
                    update,
                    onDismiss = onDismissUpdate,
                    onDownload = onDownloadUpdate,
                    onCancelDownload = onCancelDownload,
                    onHide = onHideUpdate,
                    onRetry = onRetryUpdate,
                    onInstall = onInstallUpdate,
                )
                // ANDROID_ONLY.md: a hidden update download stays visible as a bar over the top edge.
                AppUpdateProgressBar(update, onShowUpdate, Modifier.align(Alignment.TopCenter))
            }
            StartupSplash(startupSplashEnabled, ready = !state.starting, onFinished = { startupSplashFinished = true })
            // endregion
            }
            }
        }
    }
}

/** Top-level screens swapped with a shared cross-fade. */
private enum class PlaygroundScreenKind { Starting, Setup, Auth, Chat }
