package com.minashin1120.aiplayground.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.IntentSenderRequest
import androidx.activity.result.ActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
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
import android.speech.RecognizerIntent
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalClipboardManager
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
import com.minashin1120.aiplayground.data.LibraryFile
import com.minashin1120.aiplayground.data.Gem
import com.minashin1120.aiplayground.data.FixedPrompt
import com.minashin1120.aiplayground.data.ChatMessage
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
    PlaygroundTheme(darkTheme = state.preferences?.let { !it.lightModeEnabled } ?: isSystemInDarkTheme(),
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
            var logout by remember { mutableStateOf(false) }
            var modelPicker by remember { mutableStateOf(false) }
            var threadSettings by remember { mutableStateOf(false) }
            var attachMenu by remember { mutableStateOf(false) }
            var libraryOpen by remember { mutableStateOf(false) }
            var viewingFile by remember { mutableStateOf<FileViewRequest?>(null) }
            var gemsOpen by remember { mutableStateOf(false) }
            var settingsOpen by remember { mutableStateOf(false) }
            var changelogOpen by remember { mutableStateOf(false) }
            var advancedOpen by remember { mutableStateOf(false) }
            var realtimeOpen by remember { mutableStateOf(false) }
            var lyriaOpen by remember { mutableStateOf(false) }
            var awaitingMic by remember { mutableStateOf(false) }
            var richPasteOpen by remember { mutableStateOf(false) }
            var maskOpen by remember { mutableStateOf(false) }
            var maskSource by remember { mutableStateOf<Uri?>(null) }
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
                    if (granted) realtimeOpen = true else model.notify("Realtime音声にはマイクの権限が必要です。")
                }
            }
            LaunchedEffect(state.account?.id) {
                if (state.account != null && Build.VERSION.SDK_INT >= 33 &&
                    ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
                    notifications.launch(Manifest.permission.POST_NOTIFICATIONS)
                }
            }
            val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { model.upload(it) }
            val speech = rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
                if (result.resultCode == Activity.RESULT_OK) {
                    result.data?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)?.firstOrNull()?.let { text ->
                        val current = model.state.value
                        if (!current.streaming) model.draft(if (current.draft.isBlank()) text else current.draft.trimEnd() + "\n" + text)
                    }
                }
            }
            val launchSpeech: () -> Unit = {
                val selected = state.model
                val useStudio = state.preferences?.voiceStudioUi != false
                when {
                    useStudio && isRealtimeAudioModel(state.account?.models?.firstOrNull { it.id == selected }) -> {
                        if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) realtimeOpen = true
                        else { awaitingMic = true; microphone.launch(Manifest.permission.RECORD_AUDIO) }
                    }
                    useStudio && selected.startsWith("lyria") -> lyriaOpen = true
                    else -> try {
                        speech.launch(Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
                            .putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                            .putExtra(RecognizerIntent.EXTRA_PROMPT, "メッセージを話してください"))
                    } catch (_: Exception) { model.notify("音声入力に対応するアプリが見つかりません。") }
                }
            }
            val photoPicker = rememberLauncherForActivityResult(ActivityResultContracts.PickMultipleVisualMedia(30)) { model.upload(it) }
            val maskPicker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
                if (uri != null) { maskSource = uri; maskOpen = true }
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
                viewingFile = FileViewRequest(reference)
            }
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
            LaunchedEffect(state.notice) {
                state.notice?.let { snackbar.showSnackbar(it, duration = SnackbarDuration.Long); model.dismissNotice() }
            }
            val closeDrawer: () -> Unit = { scope.launch { drawer.close() } }
            val showThreads = state.account != null
            LaunchedEffect(state.banned) {
                if (state.banned) {
                    allowDrawerOpen = false
                    deleting = null
                    logout = false
                    modelPicker = false
                    threadSettings = false
                    attachMenu = false
                    libraryOpen = false
                    viewingFile = null
                    gemsOpen = false
                    settingsOpen = false
                    changelogOpen = false
                    advancedOpen = false
                    realtimeOpen = false
                    lyriaOpen = false
                    richPasteOpen = false
                    maskOpen = false
                    maskSource = null
                    closeDrawer()
                }
            }
            val hasOverlay = modelPicker || threadSettings || attachMenu || libraryOpen || viewingFile != null ||
                gemsOpen || settingsOpen || changelogOpen || advancedOpen || realtimeOpen || lyriaOpen || richPasteOpen || maskOpen || logout
            BackHandler(enabled = hasOverlay || (!wide && drawer.currentValue == DrawerValue.Open)) {
                when {
                    attachMenu -> attachMenu = false
                    modelPicker -> modelPicker = false
                    threadSettings -> threadSettings = false
                    settingsOpen -> settingsOpen = false
                    changelogOpen -> changelogOpen = false
                    advancedOpen -> advancedOpen = false
                    realtimeOpen -> realtimeOpen = false
                    lyriaOpen -> lyriaOpen = false
                    richPasteOpen -> richPasteOpen = false
                    maskOpen -> { maskOpen = false; maskSource = null }
                    viewingFile != null -> viewingFile = null
                    libraryOpen -> libraryOpen = false
                    gemsOpen -> gemsOpen = false
                    logout -> logout = false
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

            val content: @Composable () -> Unit = {
                Scaffold(
                    modifier = Modifier.imePadding(),
                    containerColor = Color.Transparent,
                    topBar = {
                        TopAppBar(colors = TopAppBarDefaults.topAppBarColors(containerColor = colors.surface.copy(alpha = 0.94f)), title = { Column {
                            Text(state.selected?.title?.ifBlank { "新しいチャット" } ?: "AI Playground", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold, maxLines = 1, overflow = TextOverflow.Ellipsis)
                            if (state.selected?.isTemporary == true || state.newThreadTemporary) Text("一時チャット", style = MaterialTheme.typography.labelSmall, color = colors.secondary)
                        } }, navigationIcon = {
                            if (showThreads && !wide) IconButton(onClick = openDrawer) {
                                Icon(Icons.Rounded.Menu, contentDescription = "履歴メニュー")
                            }
                        }, actions = {
                            if (showThreads) IconButton(onClick = { model.newChat() }, enabled = !state.busy && !state.streaming) {
                                Icon(Icons.Rounded.Add, contentDescription = "新規チャット", tint = colors.primary)
                            }
                            if (state.selected != null) IconButton(onClick = { threadSettings = true }, enabled = !state.busy && !state.streaming) {
                                Icon(Icons.Rounded.Tune, contentDescription = "チャット設定")
                            }
                            if (state.selected != null) IconButton(onClick = {
                                sharePdf()
                            }, enabled = !state.offline && !state.busy && !state.streaming) { Icon(Icons.Rounded.PictureAsPdf, contentDescription = "PDFを共有") }
                            if (showThreads) IconButton(onClick = model::refresh, enabled = !state.busy && !state.streaming) {
                                Icon(Icons.Rounded.Refresh, contentDescription = "更新")
                            }
                        })
                    }, snackbarHost = { SnackbarHost(snackbar) },
                    bottomBar = {
                        if (showThreads) Composer(state, model, { modelPicker = true }, { attachMenu = true }, launchSpeech,
                            onRichPaste = { richPasteOpen = true }, onMask = { maskPicker.launch("image/*") },
                            onSettings = { settingsOpen = true },
                            onRealtime = {
                                if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) realtimeOpen = true
                                else { awaitingMic = true; microphone.launch(Manifest.permission.RECORD_AUDIO) }
                            },
                            onLyria = { lyriaOpen = true })
                    }
                ) { padding ->
                    Column(Modifier.fillMaxSize().padding(padding)) {
                        if (state.connectionBannerVisible || state.offline) {
                            ConnectionBanner(
                                status = if (state.connectionStatus == ConnectionStatus.UNKNOWN) ConnectionStatus.OFFLINE else state.connectionStatus,
                                message = state.connectionMessage,
                                onRetry = model::reconnect,
                            )
                        }
                        val screen = when {
                            state.starting -> PlaygroundScreenKind.Starting
                            state.setupRequired -> PlaygroundScreenKind.Setup
                            state.account == null -> PlaygroundScreenKind.Auth
                            else -> PlaygroundScreenKind.Chat
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
                                    PlaygroundScreenKind.Chat -> Conversation(
                                        state, model, openInApp, loader,
                                        animationsEnabled = animationsEnabled,
                                    )
                                }
                            }
                        }
                    }
                }
            }

            if (showThreads && wide) {
                Row(Modifier.fillMaxSize()) {
                    Surface(Modifier.width(PlaygroundDimens.sidePane).fillMaxHeight(),
                        color = colors.surface.copy(alpha = 0.94f), contentColor = colors.onSurface) {
                        ThreadPanel(state, model, onLogout = { logout = true }, onDelete = { deleting = it }, onNavigate = {}, onLibrary = { libraryOpen = true }, onGems = { gemsOpen = true }, onSettings = { settingsOpen = true }, onAdvanced = { advancedOpen = true }, onPdf = sharePdf, onBubble = openBubble, onWeb = onWeb, onChangelog = { changelogOpen = true; onOpenChangelog() })
                    }
                    VerticalDivider()
                    Box(Modifier.weight(1f)) { content() }
                }
            } else {
                ModalNavigationDrawer(drawerState = drawer, gesturesEnabled = showThreads && allowDrawerOpen,
                    drawerContent = {
                        ModalDrawerSheet(Modifier.width(PlaygroundDimens.drawerPane), drawerContainerColor = colors.surface) {
                            if (showThreads) {
                                ThreadPanel(state, model, onLogout = { logout = true }, onDelete = { deleting = it }, onNavigate = closeDrawer, onLibrary = { libraryOpen = true }, onGems = { gemsOpen = true }, onSettings = { settingsOpen = true }, onAdvanced = { advancedOpen = true }, onPdf = sharePdf, onBubble = { openBubble(); closeDrawer() }, onWeb = { path -> onWeb(path); closeDrawer() }, onChangelog = { changelogOpen = true; onOpenChangelog(); closeDrawer() })
                            }
                        }
                    }) { content() }
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
                )
            }
            ModalHost(gemsOpen) { GemsDialog(state, model, onDismiss = { gemsOpen = false }) }
            ModalHost(settingsOpen) {
                SettingsDialog(state, model, onDismiss = { settingsOpen = false },
                    onLogout = { model.logout(); settingsOpen = false; closeDrawer() }, onWeb = onWeb,
                    appUpdate = appUpdate ?: AppUpdateUiState(), onCheckForUpdate = onCheckForUpdate)
            }
            ModalHost(changelogOpen) {
                AppChangelogDialog(
                    state = appChangelog,
                    onDismiss = { changelogOpen = false },
                    onRetry = onRetryChangelog,
                )
            }
            ModalHost(advancedOpen) {
                AdvancedToolsDialog(state, model, onDismiss = { advancedOpen = false }, onWebPath = onWeb,
                    onRealtime = {
                        advancedOpen = false
                        if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) realtimeOpen = true
                        else { awaitingMic = true; microphone.launch(Manifest.permission.RECORD_AUDIO) }
                    }, onLyria = { advancedOpen = false; lyriaOpen = true })
            }
            ModalHost(realtimeOpen) { RealtimeStudioDialog(state, model, onDismiss = { realtimeOpen = false }) }
            ModalHost(lyriaOpen) { LyriaStudioDialog(state, model, onDismiss = { lyriaOpen = false }) }
            ModalHost(richPasteOpen) {
                RichPasteDialog(state.draft, onDismiss = { richPasteOpen = false }) { text ->
                    model.draft(if (state.draft.isBlank()) text else state.draft.trimEnd() + "\n\n" + text)
                }
            }
            ModalValueHost(if (maskOpen) maskSource else null) { uri ->
                ImageMaskEditor(uri, onDismiss = { maskOpen = false; maskSource = null }) { bytes ->
                    model.uploadImageMask("mask_${System.currentTimeMillis()}.png", bytes)
                    maskOpen = false
                    maskSource = null
                }
            }
            if (attachMenu) {
                val attachSheet = rememberModalBottomSheetState(skipPartiallyExpanded = true)
                // Slide the sheet away first, then run the chosen action so the motion is never cut.
                val closeAttachMenu: (() -> Unit) -> Unit = { then ->
                    scope.launch { attachSheet.hide() }.invokeOnCompletion {
                        attachMenu = false
                        then()
                    }
                }
                ModalBottomSheet(onDismissRequest = { attachMenu = false }, sheetState = attachSheet) {
                    Column(
                        Modifier.fillMaxWidth().padding(start = 16.dp, end = 16.dp, bottom = 24.dp),
                        verticalArrangement = Arrangement.spacedBy(4.dp),
                    ) {
                        Text("添付を追加", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold,
                            modifier = Modifier.padding(start = 12.dp, bottom = 8.dp))
                        DialogAction(Icons.Rounded.FolderOpen, "ファイルを選択") { closeAttachMenu { picker.launch(arrayOf("*/*")) } }
                        DialogAction(Icons.Rounded.PhotoLibrary, "写真・動画を選択") { closeAttachMenu { photoPicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageAndVideo)) } }
                        DialogAction(Icons.Rounded.PhotoCamera, "カメラで撮影") { closeAttachMenu { launchCamera() } }
                        DialogAction(Icons.Rounded.FolderShared, "ライブラリから選択") { closeAttachMenu { libraryOpen = true } }
                    }
                }
            }
            ModalHost(threadSettings && state.selected != null) {
                ThreadSettingsDialog(state, onDismiss = { threadSettings = false }) {
                    title, instruction, includeGlobal, temporary ->
                    model.saveThreadSettings(title, instruction, includeGlobal, temporary)
                    threadSettings = false
                }
            }
            deleting?.let { thread -> AlertDialog(onDismissRequest = { deleting = null }, title = { Text("チャットを削除しますか？") },
                text = { Text("「${thread.title}」の履歴と紐付く添付ファイルを削除します。この操作は取り消せません。") },
                confirmButton = { TextButton(onClick = { model.deleteThread(thread); deleting = null }) { Text("削除") } },
                dismissButton = { TextButton(onClick = { deleting = null }) { Text("キャンセル") } }) }
            if (logout) AlertDialog(onDismissRequest = { logout = false }, title = { Text("この端末からログアウト") },
                text = { Text("このAndroid端末の連携を取り消します。Webや他の端末のログインは継続します。") },
                confirmButton = { TextButton(onClick = { model.logout(); logout = false; closeDrawer() }) { Text("ログアウト") } },
                dismissButton = { TextButton(onClick = { logout = false }) { Text("キャンセル") } })
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

@Composable
private fun DialogAction(icon: androidx.compose.ui.graphics.vector.ImageVector, label: String, onClick: () -> Unit) {
    TextButton(onClick = onClick, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp), shape = RoundedCornerShape(12.dp)) {
        Icon(icon, contentDescription = null)
        Text(label, modifier = Modifier.padding(start = 12.dp).weight(1f), textAlign = androidx.compose.ui.text.style.TextAlign.Start)
    }
}

@Composable
private fun ConnectionBanner(status: ConnectionStatus, message: String, onRetry: () -> Unit) {
    val colors = MaterialTheme.colorScheme
    val container = when (status) {
        ConnectionStatus.MAINTENANCE -> colors.tertiaryContainer
        ConnectionStatus.UNSTABLE -> colors.secondaryContainer
        ConnectionStatus.ONLINE -> colors.primaryContainer
        ConnectionStatus.SERVER_DOWN, ConnectionStatus.OFFLINE, ConnectionStatus.UNKNOWN -> colors.errorContainer
    }
    val content = when (status) {
        ConnectionStatus.MAINTENANCE -> colors.onTertiaryContainer
        ConnectionStatus.UNSTABLE -> colors.onSecondaryContainer
        ConnectionStatus.ONLINE -> colors.onPrimaryContainer
        ConnectionStatus.SERVER_DOWN, ConnectionStatus.OFFLINE, ConnectionStatus.UNKNOWN -> colors.onErrorContainer
    }
    val icon = when (status) {
        ConnectionStatus.MAINTENANCE -> Icons.Rounded.Build
        ConnectionStatus.UNSTABLE -> Icons.Rounded.WarningAmber
        ConnectionStatus.ONLINE -> Icons.Rounded.CheckCircle
        ConnectionStatus.SERVER_DOWN -> Icons.Rounded.Dns
        ConnectionStatus.OFFLINE, ConnectionStatus.UNKNOWN -> Icons.Rounded.CloudOff
    }
    Surface(color = container, modifier = Modifier.fillMaxWidth()) {
        Row(Modifier.padding(horizontal = 16.dp, vertical = 4.dp), verticalAlignment = Alignment.CenterVertically) {
            Icon(icon, contentDescription = null, tint = content, modifier = Modifier.size(18.dp))
            Text(message.ifBlank { status.defaultMessage() }, modifier = Modifier.weight(1f).padding(start = 8.dp),
                style = MaterialTheme.typography.bodySmall, color = content)
            if (status != ConnectionStatus.ONLINE) TextButton(onClick = onRetry) { Text("再試行") }
        }
    }
}

@Composable
private fun ThreadPanel(
    state: ChatState,
    model: ChatViewModel,
    onLogout: () -> Unit,
    onDelete: (ThreadItem) -> Unit,
    onNavigate: () -> Unit,
    onLibrary: () -> Unit,
    onGems: () -> Unit,
    onSettings: () -> Unit,
    onAdvanced: () -> Unit,
    onPdf: () -> Unit,
    onBubble: () -> Unit,
    onWeb: (String) -> Unit,
    onChangelog: () -> Unit,
) {
    val colors = MaterialTheme.colorScheme
    Column(Modifier.fillMaxSize().statusBarsPadding().padding(horizontal = 12.dp, vertical = 12.dp)) {
        Text(state.selected?.title?.ifBlank { "AI Chat" } ?: "AI Chat", style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold, maxLines = 1, overflow = TextOverflow.Ellipsis)
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            IconButton(onClick = { onSettings(); onNavigate() }) { Icon(Icons.Rounded.Settings, "設定", modifier = Modifier.size(20.dp)) }
            IconButton(onClick = { onLibrary(); onNavigate() }) { Icon(Icons.Rounded.FolderOpen, "ライブラリ", modifier = Modifier.size(20.dp)) }
            IconButton(onClick = { model.newChat(); onNavigate() }, enabled = !state.busy && !state.streaming) { Icon(Icons.Rounded.Add, "新規チャット", tint = colors.primary) }
            IconButton(onClick = { onPdf(); onNavigate() }, enabled = state.selected != null && !state.offline && !state.busy && !state.streaming) { Icon(Icons.Rounded.PictureAsPdf, "PDFを共有") }
            IconButton(onClick = { onAdvanced(); onNavigate() }) { Icon(Icons.Rounded.Layers, "Batch処理・高度な機能", modifier = Modifier.size(20.dp)) }
            IconButton(onClick = model::refresh, enabled = !state.busy && !state.streaming) { Icon(Icons.Rounded.Refresh, "更新", modifier = Modifier.size(20.dp)) }
        }
        TextButton(onClick = onBubble, modifier = Modifier.fillMaxWidth()) {
            Icon(Icons.Rounded.ChatBubble, contentDescription = null, modifier = Modifier.size(18.dp))
            Text(
                if (Build.VERSION.SDK_INT >= ANDROID_17_APP_BUBBLE_API) "バブルに追加する方法"
                else "バブルで開く",
                modifier = Modifier.padding(start = 6.dp),
            )
        }
        OutlinedTextField(
            state.search, model::search, singleLine = true, placeholder = { Text("チャットを検索...", style = MaterialTheme.typography.bodySmall) },
            leadingIcon = { Icon(Icons.Rounded.Search, contentDescription = null) },
            shape = RoundedCornerShape(PlaygroundDimens.controlRadius), modifier = Modifier.fillMaxWidth(),
            colors = OutlinedTextFieldDefaults.colors(unfocusedContainerColor = colors.surfaceContainerLow.copy(alpha = 0.72f), focusedContainerColor = colors.surfaceContainerLow),
        )
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text("Gems", style = MaterialTheme.typography.labelLarge, color = colors.onSurfaceVariant, modifier = Modifier.weight(1f))
            TextButton(onClick = { onGems(); onNavigate() }) { Text("＋ New / 編集") }
        }
        LazyColumn(Modifier.heightIn(max = 140.dp)) {
            items(state.gems, key = { it.uuid }) { gem ->
                TextButton(onClick = { model.chooseGem(gem); onNavigate() }, enabled = !state.streaming,
                    modifier = Modifier.fillMaxWidth()) {
                    Icon(Icons.Rounded.AutoAwesome, null, modifier = Modifier.size(16.dp))
                    Text(gem.name, modifier = Modifier.weight(1f).padding(start = 8.dp), maxLines = 1, overflow = TextOverflow.Ellipsis)
                    if (state.selectedGem?.uuid == gem.uuid) Icon(Icons.Rounded.Check, "適用中", modifier = Modifier.size(16.dp))
                }
            }
        }
        HorizontalDivider(color = colors.outlineVariant.copy(alpha = 0.6f))
        LazyColumn(Modifier.weight(1f), contentPadding = PaddingValues(vertical = 8.dp), verticalArrangement = Arrangement.spacedBy(3.dp)) {
            items(state.threads, key = { it.id }) { thread ->
                val selected = state.selected?.id == thread.id
                Row(
                    Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp))
                        .background(if (selected) colors.primary.copy(alpha = 0.13f) else Color.Transparent)
                        .border(1.dp, if (selected) colors.primary.copy(alpha = 0.25f) else Color.Transparent, RoundedCornerShape(12.dp)),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Row(modifier = Modifier.weight(1f).clickable { model.openThread(thread); onNavigate() }.padding(horizontal = 10.dp, vertical = 9.dp), verticalAlignment = Alignment.CenterVertically) {
                        Icon(if (thread.isTemporary) Icons.Rounded.Schedule else Icons.Rounded.ChatBubbleOutline, contentDescription = null, tint = if (selected) colors.primary else colors.onSurfaceVariant, modifier = Modifier.size(18.dp))
                        Column(Modifier.fillMaxWidth()) {
                            Text(buildString {
                                if (thread.isBookmarked) append("★  ")
                                append(thread.title.ifBlank { "新しいチャット" })
                            }, modifier = Modifier.padding(start = 9.dp), style = MaterialTheme.typography.bodyMedium, fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal, maxLines = 2, overflow = TextOverflow.Ellipsis)
                            if (thread.model.isNotBlank()) Text(thread.model, style = MaterialTheme.typography.labelSmall, color = colors.onSurfaceVariant, modifier = Modifier.padding(start = 9.dp), maxLines = 1, overflow = TextOverflow.Ellipsis)
                        }
                    }
                    IconButton(onClick = { model.toggleBookmark(thread) }, enabled = !state.offline, modifier = Modifier.size(38.dp)) {
                        Icon(if (thread.isBookmarked) Icons.Rounded.Star else Icons.Rounded.StarBorder, contentDescription = if (thread.isBookmarked) "ブックマーク解除" else "ブックマーク", modifier = Modifier.size(18.dp), tint = if (thread.isBookmarked) colors.secondary else colors.onSurfaceVariant)
                    }
                    IconButton(onClick = { onDelete(thread) }, enabled = !state.offline, modifier = Modifier.size(38.dp)) {
                        Icon(Icons.Rounded.DeleteOutline, contentDescription = "削除", modifier = Modifier.size(18.dp), tint = colors.onSurfaceVariant)
                    }
                }
            }
            if (state.threads.isEmpty()) item(key = "empty") {
                Text(
                    if (state.search.isBlank()) "チャット履歴はまだありません。"
                    else "「${state.search}」に一致する履歴はありません。",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(12.dp),
                )
            }
            if (state.nextPage != null) item(key = "more") {
                TextButton(onClick = model::moreThreads, modifier = Modifier.fillMaxWidth()) { Text("もっと読み込む") }
            }
        }
        HorizontalDivider(color = colors.outlineVariant.copy(alpha = 0.6f), modifier = Modifier.padding(vertical = 6.dp))
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.Center,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            SidebarFooterLink("ヘルプ") { onWeb("/help"); onNavigate() }
            Text("·", color = colors.onSurfaceVariant, style = MaterialTheme.typography.labelSmall)
            SidebarFooterLink("更新履歴") { onChangelog(); onNavigate() }
        }
        Button(
            onClick = onLogout,
            modifier = Modifier.fillMaxWidth().height(40.dp),
            shape = RoundedCornerShape(10.dp),
            contentPadding = PaddingValues(horizontal = 12.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = colors.surfaceContainerHigh,
                contentColor = colors.onSurface,
            ),
        ) {
            Text("ログアウト", style = MaterialTheme.typography.labelLarge)
        }
        Spacer(Modifier.navigationBarsPadding())
    }
}

@Composable
private fun SidebarFooterLink(
    label: String,
    onClick: () -> Unit,
) {
    TextButton(
        onClick = onClick,
        modifier = Modifier.height(32.dp),
        contentPadding = PaddingValues(horizontal = 6.dp),
    ) { Text(label, color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.labelSmall) }
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

@Composable
private fun ThreadSettingsDialog(
    state: ChatState,
    onDismiss: () -> Unit,
    onSave: (String, String, Boolean, Boolean) -> Unit,
) {
    val thread = state.selected ?: return
    var title by remember(thread.id) { mutableStateOf(thread.title) }
    var instruction by remember(thread.id) { mutableStateOf(state.customInstruction) }
    var includeGlobal by remember(thread.id) { mutableStateOf(state.includeGlobalInstruction) }
    var temporary by remember(thread.id) { mutableStateOf(thread.isTemporary) }
    PlaygroundDialog(
        onDismissRequest = onDismiss,
        title = { Text("チャット設定") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedTextField(title, { title = it }, label = { Text("タイトル") }, singleLine = true,
                    modifier = Modifier.fillMaxWidth())
                OutlinedTextField(instruction, { instruction = it }, label = { Text("このチャットの指示") },
                    minLines = 3, maxLines = 7, modifier = Modifier.fillMaxWidth())
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Checkbox(includeGlobal, { includeGlobal = it })
                    Text("アカウント共通の指示も使う", modifier = Modifier.weight(1f))
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Checkbox(temporary, { temporary = it })
                    Column(Modifier.weight(1f)) {
                        Text("一時チャット")
                        Text("アプリが在席更新を停止すると、設定時間後に自動削除されます。",
                            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
                if (thread.isTemporary && state.tempChatRemainingSeconds != null) {
                    Text("現在の自動削除目安: ${state.tempChatRemainingSeconds}秒",
                        style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
        },
        confirmButton = { TextButton(onClick = { onSave(title, instruction, includeGlobal, temporary) }, enabled = !state.offline && title.length <= 200 && instruction.length <= 100_000) { Text("保存") } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("キャンセル") } },
    )
}

@OptIn(androidx.compose.animation.ExperimentalAnimationApi::class)
@Composable
private fun Conversation(
    state: ChatState,
    model: ChatViewModel,
    onFile: (String) -> Unit,
    loader: FileBytesLoader?,
    animationsEnabled: Boolean,
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
        val contentState = if (showOutgoing) outgoingState ?: state else state
        ConversationContent(contentState, model, onFile, loader)
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
) {
    val scroll = rememberLazyListState()
    val scope = rememberCoroutineScope()
    var showScrollToBottom by remember { mutableStateOf(false) }
    val live = state.streaming || state.liveContent.isNotEmpty() || state.liveThought.isNotEmpty() || state.cards.isNotEmpty()
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
            scroll.animateScrollToItem(count - 1)
        }
    }
    Column(Modifier.fillMaxSize()) {
        if (state.busy) LinearProgressIndicator(Modifier.fillMaxWidth())
        if (state.selected?.isTemporary == true || (state.selected == null && state.newThreadTemporary)) {
            Surface(color = MaterialTheme.colorScheme.tertiaryContainer, modifier = Modifier.fillMaxWidth()) {
                Text("一時チャット・離席後に自動削除されます",
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onTertiaryContainer)
            }
        }
        if (state.jobId != null && !state.streaming) Row(Modifier.fillMaxWidth().padding(horizontal = 16.dp), verticalAlignment = Alignment.CenterVertically) {
            Text("生成の状態を確認できます", modifier = Modifier.weight(1f), style = MaterialTheme.typography.bodySmall)
            TextButton(onClick = model::resume) { Text("再接続") }
        }
        if (state.retryAvailable) TextButton(onClick = model::retry, modifier = Modifier.fillMaxWidth(), enabled = !state.streaming) { Text("同じ送信を再試行（二重送信を防止）") }
        if (state.canvasMode) CanvasPreview(state.messages + if (state.liveContent.isNotBlank()) listOf(ChatMessage("canvas-live", "assistant", state.liveContent)) else emptyList(),
            onUse = { code -> model.draft(code) }, onClose = model::toggleCanvas)
        Box(Modifier.weight(1f).fillMaxWidth()) {
            LazyColumn(
                state = scroll,
                modifier = Modifier.fillMaxSize().widthIn(max = PlaygroundDimens.contentMax).align(Alignment.Center),
                contentPadding = PaddingValues(horizontal = PlaygroundDimens.conversationHorizontalPadding, vertical = 20.dp),
                verticalArrangement = Arrangement.spacedBy(18.dp),
            ) {
            if (state.hasOlder) item(key = "older") { TextButton(onClick = model::olderMessages, enabled = !state.busy && !state.offline, modifier = Modifier.fillMaxWidth()) { Text("以前のメッセージ（オンラインで取得）") } }
            if (state.messages.isEmpty() && !state.busy && !live) item(key = "welcome") {
                Column(Modifier.fillMaxWidth().padding(vertical = 34.dp), verticalArrangement = Arrangement.spacedBy(14.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                    Surface(shape = RoundedCornerShape(18.dp), color = MaterialTheme.colorScheme.primary.copy(alpha = 0.13f), modifier = Modifier.size(58.dp)) {
                        Box(contentAlignment = Alignment.Center) { Text("✦", color = MaterialTheme.colorScheme.primary, fontSize = 30.sp, fontWeight = FontWeight.Bold) }
                    }
                    Text("AI Gems & Chat", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
                    Text("使いたいモデルを選んで、すぐに会話を始められます", color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.bodyMedium, textAlign = androidx.compose.ui.text.style.TextAlign.Center)
                    Spacer(Modifier.height(6.dp))
                    recentWebModels(state.account?.models.orEmpty()).forEach { info ->
                        Surface(
                            onClick = { model.chooseModel(info.id) },
                            shape = RoundedCornerShape(PlaygroundDimens.cardRadius),
                            color = MaterialTheme.colorScheme.surface.copy(alpha = 0.84f),
                            contentColor = MaterialTheme.colorScheme.onSurface,
                            border = androidx.compose.foundation.BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant),
                            modifier = Modifier.fillMaxWidth(),
                        ) {
                            Row(Modifier.padding(horizontal = 16.dp, vertical = 15.dp), verticalAlignment = Alignment.CenterVertically) {
                                Text("${info.emoji} ${info.name}".trim(), modifier = Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
                                Icon(if (state.model == info.id) Icons.Rounded.Check else Icons.Rounded.ArrowForward, contentDescription = if (state.model == info.id) "選択中" else null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(18.dp))
                            }
                        }
                    }
                }
            }
            items(state.messages, key = { it.id }) { message ->
                val siblings = siblingGroup(state.allMessages, message)
                val index = siblings.indexOfFirst { it.id == message.id }
                MessageCard(
                    message = message,
                    onFile = onFile,
                    onQuote = model::quoteMessage,
                    loader = loader,
                    onEdit = { model.beginEdit(it) },
                    onRegenerate = { model.regenerate(it) },
                    branchIndex = if (index < 0) 0 else index,
                    branchCount = if (numericId(message) != null) siblings.size else 0,
                    onSwitchBranch = { target -> model.switchBranchByIndex(siblings, target) },
                )
            }
                if (live) item(key = "live") { LiveMessage(state, onFile, model::quoteMessage, loader, model::resolveMcpDecision) }
            }
            if (showScrollToBottom) {
                SmallFloatingActionButton(
                    onClick = {
                        scope.launch {
                            val last = (scroll.layoutInfo.totalItemsCount - 1).coerceAtLeast(0)
                            scroll.animateScrollToItem(last)
                        }
                    },
                    modifier = Modifier.align(Alignment.BottomEnd).padding(end = 20.dp, bottom = 16.dp),
                    containerColor = MaterialTheme.colorScheme.surfaceContainerHighest,
                    contentColor = MaterialTheme.colorScheme.primary,
                ) {
                    Icon(Icons.Rounded.KeyboardArrowDown, contentDescription = "一番下へ")
                }
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun LibraryDialog(state: ChatState, model: ChatViewModel, onOpenFile: (FileViewRequest) -> Unit, onDismiss: () -> Unit) {
    val loader: FileBytesLoader = { reference, thumbnail, limit -> model.loadAttachmentBytes(reference, thumbnail, limit) }
    var renameTarget by remember { mutableStateOf<LibraryFile?>(null) }
    var deleteTarget by remember { mutableStateOf<LibraryFile?>(null) }
    LaunchedEffect(Unit) { model.refreshLibrary() }
    PlaygroundDialog(
        onDismissRequest = onDismiss,
        title = { Text("ファイルライブラリ") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedTextField(state.libraryQuery, model::librarySearch, singleLine = true,
                    label = { Text("ファイル名で検索") }, modifier = Modifier.fillMaxWidth())
                Row(verticalAlignment = Alignment.CenterVertically) {
                    FilterChip(state.libraryFavoritesOnly, { model.setLibraryFavoritesOnly(!state.libraryFavoritesOnly) }, { Text("★ お気に入り") })
                    Spacer(Modifier.width(12.dp))
                    Text("${state.libraryTotal}件", style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                if (state.libraryBusy) LinearProgressIndicator(Modifier.fillMaxWidth())
                LazyColumn(Modifier.heightIn(max = 360.dp)) {
                    items(state.library, key = { it.filepath }) { file ->
                        Column(Modifier.fillMaxWidth().padding(vertical = 2.dp)) {
                            val reference = file.url.ifBlank { file.filepath }
                            if (file.isImage) ProtectedImage(reference, loader, {
                                onOpenFile(FileViewRequest(it, file.displayName, file.ext, "image/"))
                            },
                                modifier = Modifier.fillMaxWidth().heightIn(max = 160.dp), thumbnail = true, contentDescription = file.displayName)
                            Text("${attachmentKindIcon(attachmentKind(file.displayName))} ${file.displayName}",
                                maxLines = 1, overflow = TextOverflow.Ellipsis)
                            FlowRow {
                                TextButton(onClick = {
                                    onOpenFile(FileViewRequest(reference, file.displayName, file.ext,
                                        if (file.type == "image") "image/" else ""))
                                }) { Text("開く") }
                                TextButton(onClick = { model.reuseLibraryFile(file); onDismiss() }) { Text("再利用") }
                                TextButton(onClick = { model.toggleLibraryFavorite(file) }, enabled = !state.offline) { Text(if (file.isFavorite) "★" else "☆") }
                                TextButton(onClick = { renameTarget = file }, enabled = !state.offline) { Text("名前変更") }
                                TextButton(onClick = { deleteTarget = file }, enabled = !state.offline) { Text("削除") }
                            }
                        }
                    }
                    if (state.library.isEmpty() && !state.libraryBusy) item(key = "empty") {
                        Text("ファイルはありません。チャットで送った添付がここに表示されます。",
                            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.padding(12.dp))
                    }
                    if (state.libraryHasMore) item(key = "more") {
                        TextButton(onClick = model::moreLibrary, modifier = Modifier.fillMaxWidth()) { Text("もっと読み込む") }
                    }
                }
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text("閉じる") } },
    )
    renameTarget?.let { file ->
        RenameLibraryFileDialog(file, onDismiss = { renameTarget = null }) { name ->
            model.renameLibraryFile(file, name)
            renameTarget = null
        }
    }
    deleteTarget?.let { file ->
        AlertDialog(onDismissRequest = { deleteTarget = null }, title = { Text("ファイルを削除しますか？") },
            text = { Text("「${file.displayName}」を削除します。この操作は取り消せません。") },
            confirmButton = { TextButton(onClick = { model.deleteLibraryFile(file); deleteTarget = null }) { Text("削除") } },
            dismissButton = { TextButton(onClick = { deleteTarget = null }) { Text("キャンセル") } })
    }
}

@Composable
private fun RenameLibraryFileDialog(file: LibraryFile, onDismiss: () -> Unit, onRename: (String) -> Unit) {
    var name by remember(file.filepath) { mutableStateOf(file.displayName) }
    AlertDialog(onDismissRequest = onDismiss, title = { Text("ファイル名を変更") },
        text = { OutlinedTextField(name, { name = it }, singleLine = true, label = { Text("表示名") }, modifier = Modifier.fillMaxWidth()) },
        confirmButton = { TextButton(onClick = { onRename(name.trim()) }, enabled = name.isNotBlank() && name.length <= 200) { Text("保存") } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("キャンセル") } })
}

@Composable
private fun GemsDialog(state: ChatState, model: ChatViewModel, onDismiss: () -> Unit) {
    var editor by remember { mutableStateOf<Gem?>(null) }
    var creating by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) { model.loadGems() }
    PlaygroundDialog(
        onDismissRequest = onDismiss,
        title = { Text("Gems") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("${state.gems.size}件", style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.weight(1f))
                    TextButton(onClick = { creating = true }) { Text("新規作成") }
                }
                if (state.gemsBusy) LinearProgressIndicator(Modifier.fillMaxWidth())
                LazyColumn(Modifier.heightIn(max = 360.dp)) {
                    items(state.gems, key = { it.uuid }) { gem ->
                        Column(Modifier.fillMaxWidth().padding(vertical = 2.dp)) {
                            Text((if (state.selectedGem?.uuid == gem.uuid) "✓ " else "") + gem.name, fontWeight = FontWeight.SemiBold)
                            if (gem.description.isNotBlank()) Text(gem.description, maxLines = 2, overflow = TextOverflow.Ellipsis,
                                style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            Row {
                                TextButton(onClick = { model.chooseGem(gem); onDismiss() }) { Text("適用") }
                                TextButton(onClick = { editor = gem }) { Text("編集") }
                                TextButton(onClick = { model.deleteGem(gem) }) { Text("削除") }
                            }
                        }
                    }
                    if (state.gems.isEmpty() && !state.gemsBusy) item(key = "empty") {
                        Text("Gemはまだありません。繰り返す指示を登録して、入力欄の @ から呼び出せます。",
                            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.padding(12.dp))
                    }
                }
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text("閉じる") } },
    )
    if (creating) GemEditorDialog(gem = null, onDismiss = { creating = false }) { n, d, i, m, prompts, done ->
        model.saveGem(null, n, d, i, m, prompts) { ok -> if (ok) creating = false; done(ok) }
    }
    editor?.let { gem -> GemEditorDialog(gem = gem, onDismiss = { editor = null }) { n, d, i, m, prompts, done ->
        model.saveGem(gem.uuid, n, d, i, m, prompts) { ok -> if (ok) editor = null; done(ok) }
    } }
}

@Composable
private fun GemEditorDialog(
    gem: Gem?,
    onDismiss: () -> Unit,
    onSave: (String, String, String, String, List<FixedPrompt>, (Boolean) -> Unit) -> Unit,
) {
    var name by remember { mutableStateOf(gem?.name.orEmpty()) }
    var description by remember { mutableStateOf(gem?.description.orEmpty()) }
    var instruction by remember { mutableStateOf(gem?.instruction.orEmpty()) }
    var defaultModel by remember { mutableStateOf(gem?.defaultModel.orEmpty()) }
    var saving by remember { mutableStateOf(false) }
    var prompts by remember { mutableStateOf(gem?.fixedPrompts.orEmpty()) }
    PlaygroundDialog(
        onDismissRequest = { if (!saving) onDismiss() },
        title = { Text(if (gem == null) "Gemを作成" else "Gemを編集") },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedTextField(name, { name = it }, singleLine = true, label = { Text("名前") }, modifier = Modifier.fillMaxWidth())
                OutlinedTextField(description, { description = it }, label = { Text("説明") },
                    minLines = 1, maxLines = 3, modifier = Modifier.fillMaxWidth())
                OutlinedTextField(instruction, { instruction = it }, label = { Text("指示") },
                    minLines = 3, maxLines = 8, modifier = Modifier.fillMaxWidth())
                OutlinedTextField(defaultModel, { defaultModel = it }, singleLine = true,
                    label = { Text("既定モデル（任意・モデルID）") }, modifier = Modifier.fillMaxWidth())
                HorizontalDivider()
                Text("Fixed Prompts", style = MaterialTheme.typography.titleSmall)
                prompts.forEachIndexed { index, prompt ->
                    OutlinedTextField(prompt.name, { value -> prompts = prompts.mapIndexed { i, old -> if (i == index) old.copy(name = value) else old } },
                        label = { Text("プロンプト名") }, singleLine = true, enabled = !saving, modifier = Modifier.fillMaxWidth())
                    OutlinedTextField(prompt.content, { value -> prompts = prompts.mapIndexed { i, old -> if (i == index) old.copy(content = value) else old } },
                        label = { Text("プロンプト内容") }, maxLines = 4, enabled = !saving, modifier = Modifier.fillMaxWidth())
                    TextButton(onClick = { prompts = prompts.filterIndexed { i, _ -> i != index } }, enabled = !saving) { Text("このプロンプトを削除") }
                }
                TextButton(onClick = { prompts = prompts + FixedPrompt("", "") }, enabled = !saving && prompts.size < 50) { Text("＋ プロンプトを追加") }
            }
        },
        confirmButton = {
            TextButton(
                onClick = { saving = true; onSave(name.trim(), description.trim(), instruction.trim(), defaultModel.trim(), prompts) { saving = false } },
                enabled = !saving && name.isNotBlank() && name.length <= 100 && description.length <= 4000 && instruction.length <= 100_000 &&
                    prompts.all { it.name.isNotBlank() && it.name.length <= 100 && it.content.isNotBlank() && it.content.length <= 20_000 },
            ) { Text("保存") }
        },
        dismissButton = { TextButton(onClick = { if (!saving) onDismiss() }) { Text("キャンセル") } },
    )
}
