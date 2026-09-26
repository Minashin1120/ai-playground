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
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.LocalIndication
import androidx.compose.foundation.clickable
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
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.blur
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
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
import com.minashin1120.aiplayground.data.ThreadItem
import com.minashin1120.aiplayground.data.Attachment
import com.minashin1120.aiplayground.data.isImageReference
import com.minashin1120.aiplayground.BuildConfig
import com.minashin1120.aiplayground.data.LibraryFile
import com.minashin1120.aiplayground.data.Gem
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
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.launch
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
    onRetryUpdate: () -> Unit = {},
    onInstallUpdate: () -> Unit = {},
    onCheckForUpdate: () -> Unit = {},
    onOpenChangelog: () -> Unit = {},
    onRetryChangelog: () -> Unit = {},
    onOpenBubble: () -> Unit = {},
) {
    val state by model.state.collectAsStateWithLifecycle()
    // Web applies the light theme for a light system setting (theme-light.css) or manual light mode.
    PlaygroundTheme(darkTheme = isSystemInDarkTheme() && state.preferences?.lightModeEnabled != true,
        themeColor = state.preferences?.themeColor, liquidGlass = state.preferences?.liquidGlassEnabled == true) {
        val colors = MaterialTheme.colorScheme
        val context = LocalContext.current
        val reduceMotion = rememberReduceMotion()
        val animationsEnabled = !reduceMotion
        val startupSplashEnabled = playStartupAnimation && !reduceMotion
        CompositionLocalProvider(LocalContentColor provides colors.onBackground, LocalReduceMotion provides reduceMotion) {
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
            var deleting by remember { mutableStateOf<ThreadItem?>(null) }
            var modelPicker by remember { mutableStateOf(false) }
            var threadSettings by remember { mutableStateOf(false) }
            var attachMenu by remember { mutableStateOf(false) }
            var libraryOpen by remember { mutableStateOf(false) }
            var viewingFile by remember { mutableStateOf<FileViewRequest?>(null) }
            var gemEditorOpen by remember { mutableStateOf(false) }
            var gemEditing by remember { mutableStateOf<Gem?>(null) }
            var gemDeleting by remember { mutableStateOf<Gem?>(null) }
            var historyOpen by remember { mutableStateOf(false) }
            var branchOpen by remember { mutableStateOf(false) }
            var alphaOpen by remember { mutableStateOf(false) }
            var legalKind by remember { mutableStateOf<String?>(null) }
            var renaming by remember { mutableStateOf<ThreadItem?>(null) }
            var settingsOpen by remember { mutableStateOf(false) }
            var settingsTab by remember { mutableStateOf("一般") }
            var compressionOpen by remember { mutableStateOf(false) }
            var changelogOpen by remember { mutableStateOf(false) }
            var advancedOpen by remember { mutableStateOf(false) }
            var realtimeOpen by remember { mutableStateOf(false) }
            var lyriaOpen by remember { mutableStateOf(false) }
            var awaitingMic by remember { mutableStateOf(false) }
            var realtimeOptions by remember { mutableStateOf(RealtimeOptions()) }
            var startDockAfterMic by remember { mutableStateOf(false) }
            var richPasteOpen by remember { mutableStateOf(false) }
            var markerTarget by remember { mutableStateOf<Attachment?>(null) }
            var composerHeight by remember { mutableIntStateOf(0) }
            var visionPicker by remember { mutableStateOf(false) }
            var cameraUri by remember { mutableStateOf<Uri?>(null) }
            var bubbleAfterNotificationPermission by remember { mutableStateOf(false) }
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
                if (awaitingMic) {
                    awaitingMic = false
                    val startDock = startDockAfterMic
                    startDockAfterMic = false
                    if (!granted) model.notify("Realtime音声にはマイクの権限が必要です。")
                    else if (startDock) startRealtimeWith(model, model.state.value.model, realtimeOptions)
                    else realtimeOpen = true
                }
            }
            LaunchedEffect(state.account?.id) {
                if (state.account != null && Build.VERSION.SDK_INT >= 33 &&
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
                            startRealtimeWith(model, selected, realtimeOptions)
                        else -> { awaitingMic = true; startDockAfterMic = true; microphone.launch(Manifest.permission.RECORD_AUDIO) }
                    }
                    ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED ->
                        model.toggleMicRecording()
                    else -> recordPermission.launch(Manifest.permission.RECORD_AUDIO)
                }
            }
            val photoPicker = rememberLauncherForActivityResult(ActivityResultContracts.PickMultipleVisualMedia(30)) { model.upload(it) }
            val maskPicker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
                if (uri != null) model.uploadImageMask(uri)
            }
            val camera = rememberLauncherForActivityResult(ActivityResultContracts.TakePicture()) { success ->
                val uri = cameraUri
                cameraUri = null
                if (success && uri != null) model.upload(listOf(uri))
            }
            val launchCamera: () -> Unit = {
                try {
                    val directory = File(context.cacheDir, "shared").apply { mkdirs() }
                    val file = File(directory, "camera_${System.currentTimeMillis()}.jpg").apply { createNewFile() }
                    val uri = FileProvider.getUriForFile(context, "${context.packageName}.files", file)
                    cameraUri = uri
                    camera.launch(uri)
                } catch (e: Exception) { model.notify("カメラを起動できません。") }
            }
            val snackbar = remember { SnackbarHostState() }
            val loader: FileBytesLoader = { reference, thumbnail, limit ->
                model.loadAttachmentBytes(reference, thumbnail, limit)
            }
            val openInApp: (String) -> Unit = { reference ->
                // Web `openImageViewer` steps through the chat's images.
                val gallery = state.messages.flatMap { it.files }.filter { isImageReference(it) }.distinct()
                viewingFile = FileViewRequest(reference, gallery = gallery)
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
                        BuildConfig.BASE_URL.trimEnd('/') + "/files/" + reference.removePrefix("/files/")))
                    model.notify("画像URLをコピーしました")
                },
                onReuse = { reference -> model.reuseImage(reference) },
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
            LaunchedEffect(state.settingsRequest) { if (state.settingsRequest > 0L) settingsOpen = true }
            LaunchedEffect(state.notice) {
                state.notice?.let { snackbar.showSnackbar(it, duration = SnackbarDuration.Long); model.dismissNotice() }
            }
            val closeDrawer: () -> Unit = { scope.launch { drawer.close() } }
            val showThreads = state.account != null
            LaunchedEffect(state.banned) {
                if (state.banned) {
                    allowDrawerOpen = false
                    deleting = null
                    renaming = null
                    gemDeleting = null
                    modelPicker = false
                    threadSettings = false
                    attachMenu = false
                    libraryOpen = false
                    viewingFile = null
                    gemEditorOpen = false
                    historyOpen = false
                    branchOpen = false
                    alphaOpen = false
                    legalKind = null
                    settingsOpen = false
                    changelogOpen = false
                    advancedOpen = false
                    realtimeOpen = false
                    lyriaOpen = false
                    richPasteOpen = false
                    markerTarget = null
                    visionPicker = false
                    closeDrawer()
                }
            }
            val hasOverlay = modelPicker || threadSettings || attachMenu || libraryOpen || viewingFile != null ||
                gemEditorOpen || historyOpen || branchOpen || alphaOpen || legalKind != null ||
                settingsOpen || changelogOpen || advancedOpen || realtimeOpen || lyriaOpen || richPasteOpen || markerTarget != null || visionPicker || compressionOpen
            BackHandler(enabled = hasOverlay || (!wide && drawer.currentValue == DrawerValue.Open)) {
                when {
                    markerTarget != null -> markerTarget = null
                    visionPicker -> visionPicker = false
                    attachMenu -> attachMenu = false
                    compressionOpen -> compressionOpen = false
                    modelPicker -> modelPicker = false
                    threadSettings -> threadSettings = false
                    settingsOpen -> settingsOpen = false
                    changelogOpen -> changelogOpen = false
                    advancedOpen -> advancedOpen = false
                    realtimeOpen -> realtimeOpen = false
                    lyriaOpen -> lyriaOpen = false
                    richPasteOpen -> richPasteOpen = false
                    viewingFile != null -> viewingFile = null
                    libraryOpen -> libraryOpen = false
                    gemEditorOpen -> gemEditorOpen = false
                    alphaOpen -> alphaOpen = false
                    legalKind != null -> legalKind = null
                    historyOpen -> historyOpen = false
                    branchOpen -> branchOpen = false
                    !wide -> closeDrawer()
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
                onChangelog = { changelogOpen = true; onOpenChangelog() },
                onHistory = { historyOpen = true },
                onLowBandwidth = model::cycleLowBandwidth,
                onSettings = { settingsOpen = true },
                onLibrary = { libraryOpen = true },
                onNewChat = { model.newChat() },
                onBatch = { advancedOpen = true },
                onBranches = { if (state.selected == null) model.notify("チャットを選択してください") else branchOpen = true },
                onPdf = sharePdf,
                onExternal = openExternal,
                onSearch = model::search,
                onOpenThread = { thread -> historyOpen = false; model.openThread(thread) },
                onBookmark = model::toggleBookmark,
                onRenameThread = { renaming = it },
                onDeleteThread = { deleting = it },
                onMoreThreads = model::moreThreads,
                onRefreshThreads = model::reloadThreads,
                onChooseGem = { gem -> model.chooseGem(gem) },
                onNewGem = { gemEditing = null; gemEditorOpen = true },
                onEditGem = { gem -> gemEditing = gem; gemEditorOpen = true },
                onDeleteGem = { gemDeleting = it },
                onRefreshGems = model::reloadGems,
                onHelp = { onWeb("/help") },
                onLegal = { legalKind = it },
                onAlphaInfo = { alphaOpen = true },
                onLogout = { model.logout(); closeDrawer() },
            )

            val content: @Composable () -> Unit = {
                Scaffold(
                    modifier = Modifier.imePadding(),
                    containerColor = Color.Transparent,
                    topBar = {
                        // Web shows `header.main-chrome-header` below the md breakpoint only.
                        if (showThreads && !wide) MobileChatHeader(state, onMenu = openDrawer, onNewChat = { model.newChat() }, onPdf = sharePdf)
                    }, snackbarHost = { SnackbarHost(snackbar) },
                    bottomBar = {
                        if (showThreads) CompositionLocalProvider(LocalComposerEstimator provides model::estimatePromptTokens) {
                        Composer(state, model, { modelPicker = true }, { attachMenu = true }, launchSpeech,
                            onRichPaste = { richPasteOpen = true }, onMask = { maskPicker.launch("image/*") },
                            onSettings = { settingsTab = "一般"; settingsOpen = true },
                            onChatInstructions = { model.ensureThread { threadSettings = true } },
                            onCompressionSettings = { compressionOpen = true },
                            onTemporarySettings = { settingsTab = "一般"; settingsOpen = true },
                            loader = loader, onOpenFile = openInApp, onDockHeight = { composerHeight = it },
                            onRealtime = {
                                if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) realtimeOpen = true
                                else { awaitingMic = true; microphone.launch(Manifest.permission.RECORD_AUDIO) }
                            },
                            onLyria = { lyriaOpen = true },
                            realtimeOptions = realtimeOptions,
                            onRealtimeOptions = { realtimeOptions = it },
                            onRealtimeStart = {
                                if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                                    startRealtimeWith(model, state.model, realtimeOptions)
                                } else { awaitingMic = true; startDockAfterMic = true; microphone.launch(Manifest.permission.RECORD_AUDIO) }
                            })
                        }
                    }
                ) { padding ->
                    Column(Modifier.fillMaxSize().padding(padding)) {
                        val screen = when {
                            state.starting -> PlaygroundScreenKind.Starting
                            state.setupRequired -> PlaygroundScreenKind.Setup
                            state.account == null -> PlaygroundScreenKind.Auth
                            else -> PlaygroundScreenKind.Chat
                        }
                        // Web `#top-model-bar` (minimal prompt bar): the model button under the header.
                        AnimatedVisibility(screen == PlaygroundScreenKind.Chat && state.preferences?.effectivePromptBarMode == "minimal",
                            enter = expandFadeIn(reduceMotion), exit = shrinkFadeOut(reduceMotion)) {
                            TopModelBar(state) { modelPicker = true }
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
                                            onSettingJump = { key ->
                                                val target = AI_SETTING_JUMP_TARGETS[key]
                                                if (target?.richPaste == true) richPasteOpen = true
                                                else {
                                                    settingsTab = SettingsTab.entries.firstOrNull { it.id == target?.tab }?.label ?: "一般"
                                                    settingsOpen = true
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

            if (showThreads && wide) {
                Row(Modifier.fillMaxSize()) {
                    Sidebar(state, versionLabel, sidebarActions {})
                    Box(Modifier.weight(1f)) { content() }
                }
            } else {
                val web = LocalWebPalette.current
                // `#overlay` (rgba(4,8,20,.55), light rgba(15,23,42,.32)) with its 4px backdrop blur.
                val contentBlur by androidx.compose.animation.core.animateDpAsState(
                    if (drawer.targetValue == DrawerValue.Open && !reduceMotion) 4.dp else 0.dp, label = "drawer blur")
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
                    }) { Box(Modifier.blur(contentBlur)) { content() } }
            }

            // Web `#canvas-panel` below 1024px: a full-screen layer over the header and the composer.
            if (showThreads && state.canvasMode && LocalConfiguration.current.screenWidthDp.dp < CANVAS_SIDE_PANEL_MIN_WIDTH) {
                androidx.compose.ui.window.Dialog(onDismissRequest = model::toggleCanvas,
                    properties = androidx.compose.ui.window.DialogProperties(usePlatformDefaultWidth = false, decorFitsSystemWindows = false)) {
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

            ModalHost(modelPicker) { ModelPicker(state, onDismiss = { modelPicker = false }, onSelect = { model.chooseModel(it); modelPicker = false }) }
            ModalHost(libraryOpen) { LibraryDialog(state, model, onOpenFile = { viewingFile = it }, onDismiss = { libraryOpen = false }) }
            ModalValueHost(viewingFile) { request ->
                FileViewerDialog(
                    request, loader,
                    download = { model.downloadAttachment(it) },
                    onDismiss = { viewingFile = null },
                    onOpenExternal = onFile,
                    imageActions = imageActions,
                )
            }
            ModalHost(gemEditorOpen) {
                val editing = gemEditing
                GemEditorDialog(gem = editing, models = state.account?.models.orEmpty(), onDismiss = { gemEditorOpen = false }) { n, d, i, m, prompts, done ->
                    model.saveGem(editing?.uuid, n, d, i, m, prompts) { ok -> if (ok) gemEditorOpen = false; done(ok) }
                }
            }
            state.mcpDecision?.let { decision -> key(decision.id) { McpDecisionDialog(decision, model::resolveMcpDecision) } }
            state.accountLock?.let { lock -> AccountLockOverlay(lock, model::accountLockExpired) }
            state.sessionTurnstileUrl?.let { url ->
                LaunchedEffect(url) { onWeb(url) }
                SessionTurnstileOverlay(onOpen = { onWeb(url) }, onDismiss = model::dismissSessionTurnstile)
            }
            state.apiKeyPrompt?.let { modelId ->
                apiKeyInfoFor(modelId)?.let { info ->
                    ApiKeyRequiredDialog(model.modelDisplayName(modelId), modelId, info, onSave = model::saveApiKeyAndResend,
                        onSwitch = { model.dismissApiKeyPrompt(false); modelPicker = true },
                        onCancel = { model.dismissApiKeyPrompt(true) })
                }
            }
            val progressLabel by model.progressLabel.collectAsState()
            // Web `#offline-banner` and `#global-progress-spinner`: bottom-right, above the composer.
            val bannerShown = showThreads && (state.connectionBannerVisible || state.offline)
            val aboveComposer = with(LocalDensity.current) { composerHeight.toDp() }
            AnimatedVisibility(bannerShown, Modifier.align(Alignment.BottomEnd).imePadding().padding(end = 16.dp, bottom = aboveComposer + 16.dp),
                enter = fadeIn(tween(if (reduceMotion) 0 else 280)) + slideInVertically(tween(if (reduceMotion) 0 else 280)) { it / 5 },
                exit = fadeOut(tween(if (reduceMotion) 0 else 280)) + slideOutVertically(tween(if (reduceMotion) 0 else 280)) { it / 5 }) {
                ConnectionBanner(
                    status = if (state.connectionStatus == ConnectionStatus.UNKNOWN) ConnectionStatus.OFFLINE else state.connectionStatus,
                    message = state.connectionMessage,
                    onRetry = model::reconnect,
                )
            }
            GlobalProgressSpinner(progressLabel, Modifier.align(Alignment.BottomEnd).imePadding()
                .padding(end = 16.dp, bottom = aboveComposer + if (bannerShown) 64.dp else 16.dp))
            ModalHost(branchOpen && state.selected != null) {
                BranchManagerDialog(state, onDismiss = { branchOpen = false }, onSwitch = model::switchBranch,
                    onDelete = model::deleteMessage, notify = model::notify)
            }
            ModalHost(historyOpen) { HistoryDialog(state, sidebarActions {}, onDismiss = { historyOpen = false }) }
            ModalHost(alphaOpen) { AlphaInfoDialog(onDismiss = { alphaOpen = false }) }
            ModalValueHost(legalKind) { kind -> LegalDialog(kind, model::legalMarkdown, onDismiss = { legalKind = null }) }
            // Web `confirm("Delete?")` / `prompt("Title:")` for Gems and threads.
            gemDeleting?.let { gem -> BrowserConfirmDialog("Delete?") { ok -> if (ok) model.deleteGem(gem); gemDeleting = null } }
            renaming?.let { thread ->
                BrowserPromptDialog("Title:") { title -> if (!title.isNullOrEmpty()) model.renameThread(thread, title); renaming = null }
            }
            ModalHost(settingsOpen) {
                SettingsDialog(state, model, onDismiss = { settingsOpen = false }, initialTab = settingsTab,
                    onLogout = { model.logout(); settingsOpen = false; closeDrawer() }, onWeb = onWeb,
                    appUpdate = appUpdate ?: AppUpdateUiState(), onCheckForUpdate = onCheckForUpdate,
                    onBubble = { openBubble(); settingsOpen = false })
            }
            ModalHost(compressionOpen) { CompressionDialog(state, model) { compressionOpen = false } }
            ModalHost(changelogOpen) {
                AppChangelogDialog(
                    state = appChangelog,
                    onDismiss = { changelogOpen = false },
                    onRetry = onRetryChangelog,
                )
            }
            ModalHost(advancedOpen) {
                BatchDialog(
                    state, onLoad = model::loadBatchJobs,
                    onOpen = { job -> advancedOpen = false; model.openThreadId(job.threadId) },
                    onCancel = model::cancelBatchJob, onDelete = model::deleteBatchJob,
                    onDismiss = { advancedOpen = false },
                )
            }
            ModalHost(realtimeOpen) {
                RealtimeStudioDialog(state, model, realtimeOptions, { realtimeOptions = it }, onDismiss = { realtimeOpen = false },
                    onStart = {
                        if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                            startRealtimeWith(model, state.model, realtimeOptions)
                        } else { awaitingMic = true; startDockAfterMic = true; microphone.launch(Manifest.permission.RECORD_AUDIO) }
                    })
            }
            ModalHost(lyriaOpen) { LyriaStudioDialog(state, model, onDismiss = { lyriaOpen = false }) }
            ModalHost(richPasteOpen) {
                RichPasteDialog(state.preferences, onDismiss = { richPasteOpen = false },
                    onSavePrompt = model::saveRichPastePrompt, notify = model::notify) { text ->
                    model.draft(if (state.draft.isBlank()) text else state.draft.trimEnd() + "\n\n" + text)
                }
            }
            if (attachMenu) UploadSheet(
                state, model, loader,
                onDismiss = { attachMenu = false },
                onPickFiles = { picker.launch(arrayOf("*/*")) },
                onCamera = launchCamera,
                onPhotos = { photoPicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageAndVideo)) },
                onLibrary = { libraryOpen = true },
                onChangeVisionModel = { visionPicker = true },
                onOpenFile = openInApp,
                onEditImage = { markerTarget = it },
            )
            ModalHost(visionPicker) {
                ModelPicker(state, onDismiss = { visionPicker = false },
                    onSelect = { model.setVisionModel(it); visionPicker = false },
                    selectedId = state.visionModel ?: state.preferences?.defaultVisionModel.orEmpty(), lockToPromptCache = false)
            }
            ModalValueHost(markerTarget) { target ->
                ImageMarkerEditor(target, loader, onDismiss = { markerTarget = null },
                    onSave = { png, attachOriginal -> model.applyImageEdit(target.reference, png, attachOriginal); markerTarget = null },
                    onError = model::notify)
            }
            ModalHost(threadSettings && state.selected != null) {
                ChatInstructionsDialog(state, onRefresh = model::loadPreferences, onDismiss = { threadSettings = false }) {
                    instruction, includeGlobal, userPrompt, done ->
                    model.saveChatInstructions(instruction, includeGlobal, userPrompt) { ok -> if (ok) threadSettings = false; done(ok) }
                }
            }
            deleting?.let { thread -> BrowserConfirmDialog("Delete?") { ok -> if (ok) model.deleteThread(thread); deleting = null } }
            if (state.banned) BannedScreen(
                reason = state.banReason,
                bannedAt = state.banAt,
                onWeb = { onWeb("/banned") },
                onLogout = model::logout,
            )
            if (!state.banned) appUpdate?.let { update ->
                AppUpdateDialog(
                    update,
                    onDismiss = onDismissUpdate,
                    onDownload = onDownloadUpdate,
                    onCancelDownload = onCancelDownload,
                    onRetry = onRetryUpdate,
                    onInstall = onInstallUpdate,
                )
            }
            StartupSplash(startupSplashEnabled)
            }
        }
    }
}

/** Top-level screens swapped with a shared cross-fade. */
private enum class PlaygroundScreenKind { Starting, Setup, Auth, Chat }

@Composable
private fun BannedScreen(
    reason: String,
    bannedAt: String,
    onWeb: () -> Unit,
    onLogout: () -> Unit,
) {
    val colors = MaterialTheme.colorScheme
    Surface(color = colors.background, contentColor = colors.onBackground, modifier = Modifier.fillMaxSize()) {
        Box(
            Modifier.fillMaxSize().padding(24.dp),
            contentAlignment = Alignment.Center,
        ) {
            Card(
                modifier = Modifier.fillMaxWidth().widthIn(max = 520.dp),
                colors = CardDefaults.cardColors(containerColor = colors.surfaceContainer),
                border = androidx.compose.foundation.BorderStroke(1.dp, colors.error.copy(alpha = 0.35f)),
            ) {
                Column(
                    Modifier.padding(24.dp).verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    Text("アカウントはBANされています", style = MaterialTheme.typography.headlineSmall,
                        color = colors.error, fontWeight = FontWeight.Bold)
                    Text("ボット検出により、このアカウントは停止中です。", color = colors.onSurfaceVariant)
                    Text("解除できるのは管理者のみです。通常のチャットや設定操作は利用できません。",
                        color = colors.onSurfaceVariant)
                    if (reason.isNotBlank()) Text("理由: $reason", style = MaterialTheme.typography.bodySmall)
                    if (bannedAt.isNotBlank()) Text("日時: $bannedAt", style = MaterialTheme.typography.bodySmall)
                    Spacer(Modifier.height(4.dp))
                    Button(onClick = onWeb, modifier = Modifier.fillMaxWidth()) {
                        Text("Web版で詳細・異議申し立てを確認")
                    }
                    OutlinedButton(onClick = onLogout, modifier = Modifier.fillMaxWidth()) {
                        Text("ログアウト")
                    }
                }
            }
        }
    }
}

/**
 * Web `#offline-banner` (`connection_monitor.js`): a pill with the state's colours, icon and wording.
 * The "再試行" button is Android's own (ANDROID_ONLY.md).
 */
@Composable
private fun ConnectionBanner(status: ConnectionStatus, message: String, onRetry: () -> Unit) {
    val (fg, border, background) = when (status) {
        ConnectionStatus.MAINTENANCE -> Triple(Color(0xFFE9D5FF), Color(192, 132, 252).copy(alpha = 0.62f), Color(32, 12, 52).copy(alpha = 0.94f))
        ConnectionStatus.UNSTABLE -> Triple(Color(0xFFFEF3C7), Color(245, 158, 11).copy(alpha = 0.55f), Color(40, 24, 0).copy(alpha = 0.92f))
        ConnectionStatus.ONLINE -> Triple(Color(0xFFD1FAE5), Color(52, 211, 153).copy(alpha = 0.58f), Color(3, 34, 26).copy(alpha = 0.92f))
        ConnectionStatus.SERVER_DOWN -> Triple(Color(0xFFFECACA), Color(239, 68, 68).copy(alpha = 0.75f), Color(48, 5, 12).copy(alpha = 0.95f))
        ConnectionStatus.OFFLINE, ConnectionStatus.UNKNOWN -> Triple(Color(0xFFFECACA), Color(248, 113, 113).copy(alpha = 0.6f), Color(28, 8, 14).copy(alpha = 0.92f))
    }
    val icon = when (status) {
        ConnectionStatus.MAINTENANCE -> com.minashin1120.aiplayground.R.drawable.fa_solid_screwdriver_wrench
        ConnectionStatus.UNSTABLE -> com.minashin1120.aiplayground.R.drawable.fa_solid_exclamation_triangle
        ConnectionStatus.ONLINE -> com.minashin1120.aiplayground.R.drawable.fa_solid_check_circle
        ConnectionStatus.SERVER_DOWN -> com.minashin1120.aiplayground.R.drawable.fa_solid_server
        ConnectionStatus.OFFLINE, ConnectionStatus.UNKNOWN -> com.minashin1120.aiplayground.R.drawable.fa_solid_unlink
    }
    val reduce = LocalReduceMotion.current
    // Web `bannerPulse`: the icon pulses once whenever the connection gets worse.
    val pulse = remember { Animatable(1f) }
    LaunchedEffect(status) {
        if (reduce || status == ConnectionStatus.ONLINE) return@LaunchedEffect
        pulse.animateTo(1.25f, tween(PlaygroundMotion.SHORT, easing = PlaygroundMotion.Emphasized))
        pulse.animateTo(1f, tween(PlaygroundMotion.MEDIUM, easing = PlaygroundMotion.Standard))
    }
    val screen = LocalConfiguration.current.screenWidthDp.dp
    Row(
        Modifier.widthIn(max = minOf(screen - 32.dp, 420.dp)).shadow(20.dp, CircleShape, ambientColor = Color.Black.copy(alpha = 0.35f))
            .clip(CircleShape).background(background).border(1.dp, border, CircleShape).padding(horizontal = 12.dp, vertical = 9.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        FaIcon(icon, null, size = 12.dp, tint = fg, modifier = Modifier.graphicsLayer { scaleX = pulse.value; scaleY = pulse.value })
        Text(message.ifBlank { status.defaultMessage() }, fontSize = 12.sp, lineHeight = 16.sp, color = fg, modifier = Modifier.weight(1f, fill = false))
        if (status != ConnectionStatus.ONLINE) Text("再試行", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = fg,
            modifier = Modifier.clip(CircleShape).clickable(role = Role.Button, onClick = onRetry).padding(horizontal = 6.dp, vertical = 2.dp))
    }
}

@Composable
private fun AuthScreen(state: ChatState, model: ChatViewModel, onWeb: (String) -> Unit) {
    var signup by rememberSaveable { mutableStateOf(false) }
    var username by rememberSaveable { mutableStateOf("") }
    var password by rememberSaveable { mutableStateOf("") }
    var confirmation by rememberSaveable { mutableStateOf("") }
    var totp by rememberSaveable { mutableStateOf("") }
    val clipboard = LocalClipboardManager.current
    val colors = MaterialTheme.colorScheme
    LaunchedEffect(state.authTurnstileUrl) {
        state.authTurnstileUrl?.let { path -> onWeb(path) }
    }
    if (state.pairing) {
        PairingScreen(state, model, onWeb)
        return
    }
    LazyColumn(
        Modifier.fillMaxSize(),
        contentPadding = PaddingValues(24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        item {
            Surface(shape = RoundedCornerShape(28.dp), color = colors.primaryContainer, contentColor = colors.onPrimaryContainer) {
                Column(Modifier.fillMaxWidth().padding(28.dp)) {
                    Text("✦", fontSize = 48.sp, color = colors.primary)
                    Text("ひとつの場所で、\nいろいろなAIと。", style = MaterialTheme.typography.headlineLarge, fontWeight = FontWeight.Bold)
                    Spacer(Modifier.height(12.dp))
                    Text("アプリ内でアカウントを作成・ログインできます。", color = colors.onPrimaryContainer)
                }
            }
        }
        item {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                if (state.authTwoFactorTransaction != null) {
                    Text("2段階認証", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
                    Text("認証アプリに表示されたコードを入力してください。", color = colors.onSurfaceVariant)
                    OutlinedTextField(
                        value = totp, onValueChange = { totp = it }, label = { Text("認証コード") },
                        singleLine = true, modifier = Modifier.fillMaxWidth(), enabled = !state.authBusy,
                    )
                    state.authError?.let { Text(it, color = colors.error) }
                    Button(
                        onClick = { model.verifyTotp(totp) },
                        enabled = !state.authBusy && totp.trim().isNotEmpty(),
                        modifier = Modifier.fillMaxWidth().heightIn(min = 50.dp),
                    ) { Text(if (state.authBusy) "確認しています…" else "ログイン") }
                    if (state.auth2faMethod == "webauthn") {
                        OutlinedButton(
                            onClick = { model.beginWebauthnTwoFactor() },
                            enabled = !state.authBusy,
                            modifier = Modifier.fillMaxWidth().heightIn(min = 50.dp),
                        ) { Text("パスキーで認証") }
                    }
                } else {
                    Text(if (signup) "アカウントを作成" else "ログイン", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
                    OutlinedTextField(
                        value = username, onValueChange = { username = it }, label = { Text("ユーザー名") },
                        singleLine = true, modifier = Modifier.fillMaxWidth(), enabled = !state.authBusy,
                    )
                    OutlinedTextField(
                        value = password, onValueChange = { password = it }, label = { Text("パスワード") },
                        singleLine = true, visualTransformation = PasswordVisualTransformation(),
                        modifier = Modifier.fillMaxWidth(), enabled = !state.authBusy,
                    )
                    if (signup) {
                        OutlinedTextField(
                            value = confirmation, onValueChange = { confirmation = it }, label = { Text("パスワード（確認）") },
                            singleLine = true, visualTransformation = PasswordVisualTransformation(),
                            modifier = Modifier.fillMaxWidth(), enabled = !state.authBusy,
                        )
                    }
                    state.authError?.let { Text(it, color = colors.error) }
                    state.googleAuthDiagnostics?.let { diagnostics ->
                        Surface(
                            color = colors.surfaceVariant,
                            shape = RoundedCornerShape(12.dp),
                            modifier = Modifier.fillMaxWidth(),
                        ) {
                            Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                Text("Googleログイン診断ログ", style = MaterialTheme.typography.titleSmall)
                                Text(
                                    "IDトークンやnonceは含まれていません。OAuth設定の確認に利用できます。",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = colors.onSurfaceVariant,
                                )
                                SelectionContainer {
                                    Text(diagnostics, style = MaterialTheme.typography.bodySmall)
                                }
                                OutlinedButton(
                                    onClick = {
                                        clipboard.setText(AnnotatedString(diagnostics))
                                        model.notify("診断ログをコピーしました。")
                                    },
                                    modifier = Modifier.fillMaxWidth(),
                                ) { Text("診断ログをコピー") }
                            }
                        }
                    }
                    Button(
                        onClick = {
                            if (signup && password != confirmation) model.notify("パスワードが一致しません。")
                            else if (signup) model.signup(username, password) else model.login(username, password)
                        },
                        enabled = !state.authBusy && username.isNotBlank() && password.isNotBlank(),
                        modifier = Modifier.fillMaxWidth().heightIn(min = 50.dp),
                    ) { Text(if (state.authBusy) "処理しています…" else if (signup) "アカウントを作成" else "ログイン") }
                    TextButton(onClick = { signup = !signup; confirmation = "" }) {
                        Text(if (signup) "すでにアカウントをお持ちですか？ログイン" else "アカウントを新規作成")
                    }
                    if (!signup) {
                        OutlinedButton(
                            onClick = { model.beginPasskeyLogin(username) },
                            enabled = !state.authBusy && username.isNotBlank(),
                            modifier = Modifier.fillMaxWidth().heightIn(min = 50.dp),
                        ) { Text("パスキーでログイン") }
                        Button(
                            onClick = model::beginGoogleLogin,
                            enabled = !state.authBusy,
                            modifier = Modifier.fillMaxWidth().heightIn(min = 50.dp),
                        ) { Text(if (state.authBusy) "Googleログインを処理しています…" else "Googleでログイン") }
                    }
                }
            }
        }
        item {
            HorizontalDivider()
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("既存のブラウザー連携", style = MaterialTheme.typography.titleMedium)
                Text("旧方式です。現在も利用できますが、アプリ内ログインを推奨します。", color = colors.onSurfaceVariant, style = MaterialTheme.typography.bodySmall)
                OutlinedButton(onClick = model::pair, modifier = Modifier.fillMaxWidth()) { Text("ブラウザーで連携（旧方式）") }
                OutlinedButton(onClick = { onWeb(model.browserLoginPath("google")) }, modifier = Modifier.fillMaxWidth()) { Text("Google（ブラウザー・旧方式）") }
                OutlinedButton(onClick = { onWeb(model.browserLoginPath("minashin")) }, modifier = Modifier.fillMaxWidth()) { Text("Minashinでログイン") }
                TextButton(onClick = { onWeb("/") }, modifier = Modifier.fillMaxWidth()) { Text("Web版を開く") }
            }
        }
    }
}

@Composable
private fun SetupScreen(state: ChatState, model: ChatViewModel, onWeb: (String) -> Unit) {
    var selectedModel by rememberSaveable(state.setupDefaultModel) { mutableStateOf(state.setupDefaultModel) }
    var modelMenu by remember { mutableStateOf(false) }
    var openai by rememberSaveable { mutableStateOf("") }
    var gemini by rememberSaveable { mutableStateOf("") }
    var anthropic by rememberSaveable { mutableStateOf("") }
    var deepseek by rememberSaveable { mutableStateOf("") }
    var kimi by rememberSaveable { mutableStateOf("") }
    var mistral by rememberSaveable { mutableStateOf("") }
    var xai by rememberSaveable { mutableStateOf("") }
    var google by rememberSaveable { mutableStateOf("") }
    var googleProject by rememberSaveable { mutableStateOf("") }
    var vertexProject by rememberSaveable { mutableStateOf("") }
    var vertexLocation by rememberSaveable { mutableStateOf("global") }
    var vertexJson by rememberSaveable { mutableStateOf("") }
    var e2ee by rememberSaveable { mutableStateOf(false) }
    val zipPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) model.importAccountZip(uri)
    }
    val colors = MaterialTheme.colorScheme
    if (state.setupImportSettingsChanges.isNotEmpty()) {
        AlertDialog(
            onDismissRequest = { if (!state.setupImportBusy) model.cancelSetupImportConfirmation() },
            title = { Text("設定の変更を確認") },
            text = {
                LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.heightIn(max = 360.dp)) {
                    item {
                        Text("ZIPに含まれる設定で、現在の設定を上書きします。内容を確認してください。")
                    }
                    items(state.setupImportSettingsChanges) { change ->
                        ImportSettingChangeRow(change)
                    }
                }
            },
            confirmButton = {
                TextButton(
                    onClick = model::confirmSetupImportSettings,
                    enabled = !state.setupImportBusy,
                ) { Text(if (state.setupImportBusy) "適用中…" else "設定を上書きして続行") }
            },
            dismissButton = {
                TextButton(
                    onClick = model::cancelSetupImportConfirmation,
                    enabled = !state.setupImportBusy,
                ) { Text("取り消す") }
            },
        )
    }
    LazyColumn(
        Modifier.fillMaxSize(), contentPadding = PaddingValues(24.dp), verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        item {
            Text("はじめに設定しましょう", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
            Spacer(Modifier.height(6.dp))
            Text("${state.account?.name ?: "アカウント"} の初回セットアップです。後からWebの設定でも変更できます。", color = colors.onSurfaceVariant)
        }
        item {
            Text("既定のモデル", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Box {
                OutlinedButton(onClick = { modelMenu = true }, modifier = Modifier.fillMaxWidth()) { Text(selectedModel) }
                DropdownMenu(expanded = modelMenu, onDismissRequest = { modelMenu = false }) {
                    state.setupModels.filter { it.selectable }.take(40).forEach { option ->
                        DropdownMenuItem(text = { Text(option.name) }, onClick = { selectedModel = option.id; modelMenu = false })
                    }
                }
            }
        }
        item { Text("APIキー（必要なプロバイダーだけ入力）", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold) }
        item { SetupSecretField("OpenAI", openai, { openai = it }) }
        item { SetupSecretField("Gemini", gemini, { gemini = it }) }
        item { SetupSecretField("Anthropic", anthropic, { anthropic = it }) }
        item { SetupSecretField("DeepSeek", deepseek, { deepseek = it }) }
        item { SetupSecretField("Kimi", kimi, { kimi = it }) }
        item { SetupSecretField("Mistral", mistral, { mistral = it }) }
        item { SetupSecretField("xAI", xai, { xai = it }) }
        item { SetupSecretField("Google API", google, { google = it }) }
        item { SetupSecretField("Google Cloudプロジェクト", googleProject, { googleProject = it }) }
        item { SetupSecretField("Vertex AIプロジェクト", vertexProject, { vertexProject = it }) }
        item { SetupSecretField("Vertex AIリージョン", vertexLocation, { vertexLocation = it }) }
        item {
            OutlinedTextField(
                vertexJson, { vertexJson = it }, label = { Text("Vertex AIサービスアカウントJSON（任意）") },
                minLines = 3, maxLines = 8, modifier = Modifier.fillMaxWidth(),
            )
        }
        item {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Checkbox(checked = e2ee, onCheckedChange = { e2ee = it })
                Column(Modifier.weight(1f)) {
                    Text("保存データを暗号化する")
                    Text("既存データがある場合はバックグラウンドで移行します。", style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
                }
            }
        }
        item {
            state.authError?.let { Text(it, color = colors.error) }
            Button(
                onClick = {
                    model.finishSetup(selectedModel, openai, gemini, anthropic, deepseek, kimi, mistral, xai,
                        google, googleProject, vertexProject, vertexLocation, vertexJson, e2ee)
                }, enabled = !state.authBusy, modifier = Modifier.fillMaxWidth().heightIn(min = 52.dp),
            ) { Text(if (state.authBusy) "保存しています…" else "セットアップを完了") }
        }
        item {
            Text("アカウントZIPのインポート（任意）", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Text("Web版で書き出したZIPから、履歴・設定・APIキーなどを取り込めます。", color = colors.onSurfaceVariant, style = MaterialTheme.typography.bodySmall)
            if (state.setupImportBusy) {
                Text(
                    "${state.setupImportName} をアップロード中… (${state.setupImportProgress}/${state.setupImportTotalChunks})",
                    style = MaterialTheme.typography.bodySmall,
                )
                val progress = if (state.setupImportTotalChunks > 0) {
                    state.setupImportProgress.toFloat() / state.setupImportTotalChunks
                } else 0f
                LinearProgressIndicator(progress = { progress.coerceIn(0f, 1f) }, modifier = Modifier.fillMaxWidth())
                TextButton(onClick = model::cancelSetupImport) { Text("インポートをキャンセル") }
            } else if (state.setupImportDone) {
                Text("インポートが完了しました。", color = colors.primary, style = MaterialTheme.typography.bodySmall)
            } else {
                Button(onClick = {
                    zipPicker.launch(arrayOf("application/zip", "application/x-zip-compressed", "application/octet-stream"))
                }, modifier = Modifier.fillMaxWidth()) { Text("ZIPを選択してインポート") }
            }
            state.setupImportError?.let { Text(it, color = colors.error, style = MaterialTheme.typography.bodySmall) }
        }
        item {
            TextButton(onClick = { onWeb("/setup") }, modifier = Modifier.fillMaxWidth()) {
                Text("Webのセットアップ画面を開く")
            }
        }
    }
}

@Composable
private fun ImportSettingChangeRow(change: ImportSettingChange) {
    val label = mapOf(
        "default_model" to "既定のモデル",
        "theme_color" to "テーマ色",
        "system_prompt" to "システムプロンプト",
        "system_prompt_enabled" to "システムプロンプトの有効化",
        "light_mode_enabled" to "ライトモード",
        "liquid_glass_enabled" to "Liquid Glass",
        "gemini_backend" to "Geminiバックエンド",
        "gemini_vertex_location" to "Vertex AIリージョン",
    )[change.field] ?: change.field
    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
        Text(label, fontWeight = FontWeight.SemiBold)
        Text("現在: ${shortImportValue(change.current)}", style = MaterialTheme.typography.bodySmall)
        Text("取り込み後: ${shortImportValue(change.incoming)}", style = MaterialTheme.typography.bodySmall)
    }
}

private fun shortImportValue(value: String): String {
    val normalized = value.replace("\n", " ").trim()
    return if (normalized.length > 180) normalized.take(180) + "…" else normalized
}

@Composable
private fun SetupSecretField(label: String, value: String, onValueChange: (String) -> Unit) {
    OutlinedTextField(
        value = value, onValueChange = onValueChange, label = { Text(label) },
        singleLine = true, visualTransformation = PasswordVisualTransformation(), modifier = Modifier.fillMaxWidth(),
    )
}

@Composable
private fun PairingScreen(state: ChatState, model: ChatViewModel, onWeb: (String) -> Unit) {
    val colors = MaterialTheme.colorScheme
    LazyColumn(Modifier.fillMaxSize(), contentPadding = PaddingValues(24.dp), verticalArrangement = Arrangement.spacedBy(24.dp)) {
        item {
            Surface(shape = RoundedCornerShape(28.dp), color = colors.primaryContainer, contentColor = colors.onPrimaryContainer) {
                Column(Modifier.fillMaxWidth().padding(28.dp)) {
                    Text("✦", fontSize = 48.sp, color = colors.primary)
                    Text("ひとつの場所で、\nいろいろなAIと。", color = colors.onPrimaryContainer, style = MaterialTheme.typography.headlineLarge, fontWeight = FontWeight.Bold)
                    Spacer(Modifier.height(16.dp))
                    Text("いつものアカウントとチャット履歴を、そのままAndroidで。", color = colors.onPrimaryContainer)
                }
            }
        }
        item {
            Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                if (state.userCode.isNotBlank() && state.pairing) {
                    Text("1. この確認コードを確認してください", color = colors.onSurface, fontWeight = FontWeight.SemiBold)
                    SelectionContainer { Text(state.userCode.chunked(4).joinToString(" − "), color = colors.primary, fontSize = 28.sp, fontWeight = FontWeight.Bold) }
                    Text("2. ブラウザーでログインすると、コードが自動入力された連携画面が開きます。", color = colors.onSurfaceVariant)
                    Button(onClick = { onWeb("/android/connect?code=${Uri.encode(state.userCode)}") }, modifier = Modifier.fillMaxWidth()) { Text("ブラウザーで連携を許可") }
                    Text("3. このアプリに戻ると連携を確認します。コードの有効期限は10分です。", color = colors.onSurfaceVariant, style = MaterialTheme.typography.bodySmall)
                    LinearProgressIndicator(Modifier.fillMaxWidth())
                    TextButton(onClick = model::cancelPairing) { Text("連携をキャンセル", color = colors.onSurfaceVariant) }
                } else {
                    Text("ブラウザーでいつものログイン方法を使えます。パスワードやAPIキーを、このアプリに入力する必要はありません。", color = colors.onSurface)
                    Button(onClick = model::pair, enabled = !state.pairing, modifier = Modifier.fillMaxWidth().heightIn(min = 52.dp)) {
                        Text(if (state.pairing) "連携を準備しています…" else "アカウントを連携")
                    }
                }
                Text("接続先: ai.minashin1120.com", color = colors.onSurfaceVariant, style = MaterialTheme.typography.labelMedium)
                TextButton(onClick = { onWeb("/") }) { Text("アカウント作成・Webアプリを開く", color = colors.primary) }
            }
        }
    }
}

@OptIn(androidx.compose.animation.ExperimentalAnimationApi::class)
@Composable
private fun Conversation(
    state: ChatState,
    model: ChatViewModel,
    onFile: (String) -> Unit,
    loader: FileBytesLoader?,
    animationsEnabled: Boolean,
    onSettingJump: (String) -> Unit = {},
) {
    var outgoingState by remember { mutableStateOf<ChatState?>(null) }
    var keepOutgoingUntilLoaded by remember { mutableStateOf(false) }
    LaunchedEffect(model) {
        var previous = model.state.value
        var pendingOutgoing: ChatState? = null
        model.state.collect { current ->
            val selectedChanged = current.selected?.id != previous.selected?.id
            val navigationStarted = current.chatTransitionKind == com.minashin1120.aiplayground.ChatTransitionKind.NONE &&
                current.busy &&
                (selectedChanged || (current.messages.isEmpty() && previous.messages.isNotEmpty()))
            if (navigationStarted) {
                pendingOutgoing = previous
                outgoingState = previous
                keepOutgoingUntilLoaded = true
            }
            if (current.chatTransitionKind != com.minashin1120.aiplayground.ChatTransitionKind.NONE &&
                current.chatTransitionId != previous.chatTransitionId) {
                outgoingState = pendingOutgoing ?: previous
                pendingOutgoing = null
                keepOutgoingUntilLoaded = false
            } else if (keepOutgoingUntilLoaded && !current.busy && pendingOutgoing != null) {
                pendingOutgoing = null
                keepOutgoingUntilLoaded = false
            }
            previous = current
        }
    }
    val slideOffsetPx = with(LocalDensity.current) { 22.dp.roundToPx() }
    AnimatedContent(
        targetState = state.chatTransitionId,
        modifier = Modifier.fillMaxSize(),
        transitionSpec = {
            if (!animationsEnabled || targetState == initialState) {
                EnterTransition.None togetherWith ExitTransition.None
            } else {
                // Symmetric dissolve with a shared vertical drift so the swap reads as one motion.
                (fadeIn(animationSpec = tween(260, easing = FastOutSlowInEasing)) +
                    slideInVertically(animationSpec = tween(260, easing = FastOutSlowInEasing)) { slideOffsetPx }) togetherWith
                    (fadeOut(animationSpec = tween(260, easing = FastOutSlowInEasing)) +
                        slideOutVertically(animationSpec = tween(260, easing = FastOutSlowInEasing)) { -slideOffsetPx / 2 })
            }
        },
        label = "chat conversation transition",
    ) { transitionId ->
        val showOutgoing = keepOutgoingUntilLoaded || transitionId != state.chatTransitionId
        // The collector above reacts a frame late; until it does, keep what this pane last showed
        // instead of flashing the loading thread's empty list and refilling it.
        val lastShown = remember { arrayOfNulls<ChatState>(1) }
        val loadingOther = state.busy && state.chatNavigationId != state.chatTransitionId
        val contentState = when {
            showOutgoing -> outgoingState ?: state
            loadingOther -> lastShown[0] ?: state
            else -> state
        }
        lastShown[0] = contentState
        ProvideQuoteSelection(model::quoteMessage) { ConversationContent(contentState, model, onFile, loader, onSettingJump) }
    }
}

/** Duration of the Web-parity chat transition veil. */
private const val CHAT_TRANSITION_VEIL_MS = 460

/**
 * Piecewise-linear opacity keyframes matching the Web CSS `chatTransitionVeil` / `chatTransitionNew`.
 * New-chat transitions peak earlier and higher than history transitions.
 */
internal fun chatTransitionVeilAlpha(progress: Float, newChat: Boolean): Float {
    val frames = if (newChat) {
        listOf(0f to 0f, 0.36f to 0.58f, 1f to 0f)
    } else {
        listOf(0f to 0f, 0.30f to 0.22f, 0.52f to 0.72f, 1f to 0f)
    }
    val p = progress.coerceIn(0f, 1f)
    for (index in 0 until frames.lastIndex) {
        val (start, startAlpha) = frames[index]
        val (end, endAlpha) = frames[index + 1]
        if (p <= end) return startAlpha + (endAlpha - startAlpha) * ((p - start) / (end - start))
    }
    return frames.last().second
}

/**
 * A quiet theme-colored sweep over the whole screen when a history item is opened or a new chat
 * starts, mirroring the Web `#chat-transition-veil` motion so the swap reads as one continuous move.
 */
@Composable
private fun ChatTransitionVeil(
    navigationId: Long,
    kind: ChatTransitionKind,
    animationsEnabled: Boolean,
) {
    val accent = MaterialTheme.colorScheme.primary
    val progress = remember { Animatable(0f) }
    var active by remember { mutableStateOf(false) }
    LaunchedEffect(navigationId) {
        if (navigationId <= 0L || !animationsEnabled) return@LaunchedEffect
        active = true
        progress.snapTo(0f)
        progress.animateTo(
            targetValue = 1f,
            animationSpec = tween(durationMillis = CHAT_TRANSITION_VEIL_MS, easing = LinearEasing),
        )
        active = false
    }
    if (!active) return
    val alpha = chatTransitionVeilAlpha(progress.value, kind == ChatTransitionKind.NEW_CHAT)
    if (alpha <= 0.001f) return
    Canvas(Modifier.fillMaxSize()) {
        val width = size.width
        val center = width * (-0.7f + 1.9f * progress.value)
        val half = width * 0.8f
        drawRect(
            brush = Brush.horizontalGradient(
                colors = listOf(Color.Transparent, accent.copy(alpha = 0.30f), Color.Transparent),
                startX = center - half,
                endX = center + half,
            ),
            alpha = alpha,
        )
    }
}

@Composable
private fun ConversationContent(
    state: ChatState,
    model: ChatViewModel,
    onFile: (String) -> Unit,
    loader: FileBytesLoader?,
    onSettingJump: (String) -> Unit = {},
) {
    val reduce = LocalReduceMotion.current
    val scroll = rememberLazyListState()
    val scope = rememberCoroutineScope()
    var showScrollToBottom by remember { mutableStateOf(false) }
    val keys = remember { ConversationKeyTracker() }
    val streamActive = state.streaming || state.liveContent.isNotEmpty() || state.liveThought.isNotEmpty() || state.cards.isNotEmpty()
    keys.update(state.messages, state.streaming, streamActive)
    // Once the stored reply has taken over the streamed row, the placeholder must not reappear.
    val live = streamActive && !keys.liveConsumed
    LaunchedEffect(scroll.firstVisibleItemIndex, scroll.layoutInfo.totalItemsCount) {
        val info = scroll.layoutInfo
        val lastVisible = info.visibleItemsInfo.lastOrNull()?.index ?: 0
        showScrollToBottom = info.totalItemsCount > 0 && lastVisible < info.totalItemsCount - 2
    }
    var positioned by remember { mutableStateOf(false) }
    LaunchedEffect(state.messages.size, state.liveContent.length, state.cards.size) {
        val info = scroll.layoutInfo
        val nearBottom = (info.visibleItemsInfo.lastOrNull()?.index ?: 0) >= info.totalItemsCount - 3
        val count = state.messages.size + (if (state.hasOlder) 1 else 0) + (if (live) 1 else 0)
        if (count <= 0) return@LaunchedEffect
        if (!positioned) {
            // Open a freshly shown conversation at the latest message without scrolling through it.
            positioned = true
            scroll.scrollToItem(count - 1)
        } else if (nearBottom) {
            if (reduce) scroll.scrollToItem(count - 1) else scroll.animateScrollToItem(count - 1)
        }
    }
    val temporary = state.selected?.isTemporary == true || (state.selected == null && state.newThreadTemporary)
    // Web shows the hover controls of the bubble the user last tapped.
    var activeMessageId by remember { mutableStateOf<String?>(null) }
    var deletingMessage by remember { mutableStateOf<ChatMessage?>(null) }
    var tokenDetail by remember { mutableStateOf<TokenDetail?>(null) }
    var pythonRuns by remember { mutableStateOf<List<com.minashin1120.aiplayground.data.PythonExecution>?>(null) }
    var encryptionState by remember { mutableStateOf<Boolean?>(null) }
    // Choosing a quick-access model hides the welcome screen until the next chat, as on Web.
    var welcomeDismissed by remember(state.chatTransitionId) { mutableStateOf(false) }
    val pathTotals = remember(state.messages) { buildTokenTotals(state.messages) }
    val allTotals = remember(state.allMessages) { buildTokenTotals(state.allMessages) }
    val actions = MessageActions(
        onEdit = { model.beginEdit(it) },
        onRegenerate = { model.regenerate(it) },
        onDelete = { deletingMessage = it },
        onTokenDetail = { tokenDetail = it },
        onEncryption = { encryptionState = it },
        onPython = { runs -> if (runs.isEmpty()) model.notify("Python実行結果がありません") else pythonRuns = runs },
    )
    Column(Modifier.fillMaxSize()) {
        TotalTokenBar(pathTotals, allTotals) { tokenDetail = it }
        state.batchBanner?.let { (text, _) ->
            BatchCompletionBanner(text, onOpen = { model.dismissBatchBanner(open = true) }, onClose = { model.dismissBatchBanner(open = false) })
        }
        val screenWidth = LocalConfiguration.current.screenWidthDp.dp
        val sideCanvas = state.canvasMode && screenWidth >= CANVAS_SIDE_PANEL_MIN_WIDTH
        Row(Modifier.weight(1f).fillMaxWidth()) {
        Box(Modifier.weight(1f).fillMaxHeight()) {
            LazyColumn(
                state = scroll,
                modifier = Modifier.fillMaxSize().widthIn(max = 832.dp).align(Alignment.TopCenter),
                // `#chat-container`: 12px padding on phones, 20px between message groups.
                contentPadding = PaddingValues(12.dp),
                verticalArrangement = Arrangement.spacedBy(20.dp),
            ) {
            if (state.hasOlder) item(key = "older") {
                Box(Modifier.fillMaxWidth().padding(bottom = 12.dp), contentAlignment = Alignment.Center) {
                    val web = LocalWebPalette.current
                    val enabled = !state.busy && !state.offline
                    Row(
                        Modifier
                            .clip(RoundedCornerShape(4.dp))
                            .border(1.dp, web.twBorder(Tw.gray600), RoundedCornerShape(4.dp))
                            .clickable(enabled = enabled, onClick = model::olderMessages)
                            .graphicsLayer { alpha = if (enabled) 1f else 0.5f }
                            .padding(horizontal = 12.dp, vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        FaIcon(com.minashin1120.aiplayground.R.drawable.fa_solid_clock_rotate_left, null, size = 12.dp,
                            tint = web.twText(Tw.gray200), modifier = Modifier.padding(end = 4.dp))
                        Text(if (state.busy) "読み込み中..." else "過去メッセージを読み込む", color = web.twText(Tw.gray200), fontSize = 12.sp, lineHeight = 16.sp)
                    }
                }
            }
            itemsIndexed(state.messages, key = { row, message -> keys.keyAt(row, message) }) { row, message ->
                val entering = remember { keys.consumeFresh(keys.keyAt(row, message)) }
                val siblings = siblingGroup(state.allMessages, message)
                val index = siblings.indexOfFirst { it.id == message.id }
                Box(Modifier.animateItem(fadeInSpec = null, placementSpec = listPlacement(reduce), fadeOutSpec = null)) {
                    StaggerIn(0, animate = entering) {
                        MessageBubble(
                            message = message,
                            onFile = onFile,
                            loader = loader,
                            actions = MessageActions(
                                onEdit = actions.onEdit, onRegenerate = actions.onRegenerate, onDelete = actions.onDelete,
                                onSwitchBranch = { target -> model.switchBranchByIndex(siblings, target) },
                                onTokenDetail = actions.onTokenDetail, onEncryption = actions.onEncryption,
                            ),
                            controlsVisible = activeMessageId == message.id,
                            onToggleControls = { activeMessageId = if (activeMessageId == message.id) null else message.id },
                            branchIndex = if (index < 0) 0 else index,
                            branchCount = if (numericId(message) != null) siblings.size else 0,
                        )
                    }
                }
            }
                if (live) item(key = keys.liveKey) {
                    val entering = remember { keys.consumeFresh(keys.liveKey) }
                    Box(Modifier.animateItem(fadeInSpec = null, placementSpec = listPlacement(reduce), fadeOutSpec = null)) {
                        StaggerIn(0, animate = entering) {
                            LiveMessage(state, onFile, model::quoteMessage, loader, model::resolveMcpDecision)
                        }
                    }
                }
                items(state.settingsBubbles, key = { it.id }) { bubble ->
                    Box(Modifier.animateItem(fadeInSpec = null, placementSpec = listPlacement(reduce), fadeOutSpec = null)) {
                        SettingsBubbleView(bubble, state.model, onFile, loader, onSettingJump)
                    }
                }
            }
            val showWelcome = state.messages.isEmpty() && !state.busy && !live && !welcomeDismissed
            androidx.compose.animation.AnimatedVisibility(
                visible = showWelcome,
                enter = fadeIn(motionTween(reduce)),
                exit = fadeOut(motionTween(reduce, PlaygroundMotion.SHORT)),
                label = "welcome",
            ) {
                if (temporary) TemporaryChatWelcome(state.preferences?.tempChatTimeoutSeconds ?: 90)
                else WelcomeScreen(
                    recentWebModels(state.account?.models.orEmpty()).map { info -> info.id to "${info.emoji} ${info.name}".trim() },
                    onChoose = { id -> model.chooseModel(id); welcomeDismissed = true },
                )
            }
            // Qualified so the outer Column's scoped overload is not picked inside this Box.
            androidx.compose.animation.AnimatedVisibility(
                visible = showScrollToBottom && !showWelcome,
                modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 16.dp),
                enter = popIn(reduce),
                exit = popOut(reduce),
                label = "scroll to bottom",
            ) {
                ScrollToBottomPill(onClick = {
                    scope.launch {
                        val last = (scroll.layoutInfo.totalItemsCount - 1).coerceAtLeast(0)
                        if (reduce) scroll.scrollToItem(last) else scroll.animateScrollToItem(last)
                    }
                })
            }
        }
        if (sideCanvas) CanvasPanel(canvasSourceText(state), fullScreen = false, notify = model::notify, onClose = model::toggleCanvas,
            modifier = Modifier.width(canvasSidePanelWidth(screenWidth)).fillMaxHeight())
        }
    }
    deletingMessage?.let { message ->
        BrowserConfirmDialog("Delete this message and subsequent history?") { ok ->
            if (ok) model.deleteMessage(message)
            deletingMessage = null
        }
    }
    ModalValueHost(tokenDetail) { detail -> TokenDetailDialog(detail, onDismiss = { tokenDetail = null }) }
    ModalValueHost(pythonRuns) { runs -> PythonExecutionDialog(runs, onDismiss = { pythonRuns = null }) }
    ModalValueHost(encryptionState) { encrypted ->
        EncryptionStatusDialog(encrypted, onSettings = { encryptionState = null; model.requestSettings() }, onDismiss = { encryptionState = null })
    }
}

/** `#welcome-temporary-content`: amber card that replaces the welcome screen in a temporary chat. */
@Composable
private fun TemporaryChatWelcome(timeoutSeconds: Int) {
    val web = LocalWebPalette.current
    val seconds = timeoutSeconds.coerceIn(10, 3600)
    Box(Modifier.fillMaxSize().padding(horizontal = 8.dp, vertical = 24.dp), contentAlignment = Alignment.Center) {
        Column(
            Modifier
                .widthIn(max = 448.dp)
                .padding(horizontal = 16.dp)
                .clip(RoundedCornerShape(12.dp))
                .background(web.twBg(Tw.amber900, 0.2f))
                .border(1.dp, Tw.amber500.copy(alpha = 0.5f), RoundedCornerShape(12.dp))
                .padding(16.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                FaIcon(com.minashin1120.aiplayground.R.drawable.fa_solid_user_secret, null, size = 14.dp, tint = web.twText(Tw.amber300),
                    modifier = Modifier.padding(end = 8.dp))
                Text("一時チャットモード", color = web.twText(Tw.amber300), fontSize = 14.sp, lineHeight = 20.sp, fontWeight = FontWeight.Bold)
            }
            Text(
                "このページが非表示/切断の状態で $seconds 秒経過すると、この一時チャットとこのチャットでアップロードした添付を自動削除します（ライブラリ添付は除外）。",
                color = if (web.isLight) Color(0xFF92400E) else Color(0xFFFEF3C7).copy(alpha = 0.9f), fontSize = 12.sp, lineHeight = 19.5.sp,
                modifier = Modifier.padding(top = 8.dp),
            )
        }
    }
}

