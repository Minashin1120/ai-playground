package com.minashin1120.aiplayground.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.core.content.FileProvider
import androidx.core.content.ContextCompat
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.ChatViewModel
import com.minashin1120.aiplayground.AppUpdateUiState
import com.minashin1120.aiplayground.data.ThreadItem
import com.minashin1120.aiplayground.data.LibraryFile
import com.minashin1120.aiplayground.data.Gem
import com.minashin1120.aiplayground.data.FixedPrompt
import com.minashin1120.aiplayground.data.ChatMessage
import com.minashin1120.aiplayground.data.recentWebModels

import com.minashin1120.aiplayground.data.attachmentKind
import com.minashin1120.aiplayground.data.attachmentKindIcon
import com.minashin1120.aiplayground.data.numericId
import com.minashin1120.aiplayground.data.siblingGroup
import kotlinx.coroutines.launch
import java.io.File

/** Phone layout keeps the history drawer closed until it has settled off-screen. */
internal fun shouldCoverPhoneHistoryUntilClosed(
    starting: Boolean,
    showThreads: Boolean,
    wideLayout: Boolean,
    drawerSettledClosed: Boolean,
): Boolean = starting || (showThreads && !wideLayout && !drawerSettledClosed)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PlaygroundScreen(
    model: ChatViewModel,
    onWeb: (String) -> Unit,
    onFile: (String) -> Unit,
    appUpdate: AppUpdateUiState? = null,
    playStartupAnimation: Boolean = true,
    onDismissUpdate: () -> Unit = {},
    onDownloadUpdate: () -> Unit = {},
    onCancelDownload: () -> Unit = {},
    onRetryUpdate: () -> Unit = {},
    onInstallUpdate: () -> Unit = {},
) {
    val state by model.state.collectAsStateWithLifecycle()
    PlaygroundTheme(darkTheme = state.preferences?.let { !it.lightModeEnabled } ?: isSystemInDarkTheme(),
        themeColor = state.preferences?.themeColor, liquidGlass = state.preferences?.liquidGlassEnabled == true) {
        val colors = MaterialTheme.colorScheme
        val context = LocalContext.current
        val startupSplashEnabled = playStartupAnimation && !areSystemAnimationsDisabled(context)
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
            var advancedOpen by remember { mutableStateOf(false) }
            var realtimeOpen by remember { mutableStateOf(false) }
            var lyriaOpen by remember { mutableStateOf(false) }
            var awaitingMic by remember { mutableStateOf(false) }
            var richPasteOpen by remember { mutableStateOf(false) }
            var maskOpen by remember { mutableStateOf(false) }
            var maskSource by remember { mutableStateOf<Uri?>(null) }
            var cameraUri by remember { mutableStateOf<Uri?>(null) }
            val notifications = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { }
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
                    useStudio && realtimeModels.any { it.first == selected } -> {
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
            val hasOverlay = modelPicker || threadSettings || attachMenu || libraryOpen || viewingFile != null ||
                gemsOpen || settingsOpen || advancedOpen || realtimeOpen || lyriaOpen || richPasteOpen || maskOpen || logout
            BackHandler(enabled = hasOverlay || (!wide && drawer.currentValue == DrawerValue.Open)) {
                when {
                    attachMenu -> attachMenu = false
                    modelPicker -> modelPicker = false
                    threadSettings -> threadSettings = false
                    settingsOpen -> settingsOpen = false
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
                            }, enabled = !state.busy && !state.streaming) { Icon(Icons.Rounded.PictureAsPdf, contentDescription = "PDFを共有") }
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
                        if (state.offline) OfflineBanner(model::reconnect)
                        Box(Modifier.weight(1f)) {
                            when {
                                state.starting -> CircularProgressIndicator(Modifier.align(Alignment.Center))
                                state.account == null -> PairingScreen(state, model, onWeb)
                                else -> Conversation(state, model, openInApp, loader)
                            }
                        }
                    }
                }
            }

            if (showThreads && wide) {
                Row(Modifier.fillMaxSize()) {
                    Surface(Modifier.width(PlaygroundDimens.sidePane).fillMaxHeight(), color = colors.surface.copy(alpha = 0.94f)) {
                        ThreadPanel(state, model, onLogout = { logout = true }, onDelete = { deleting = it }, onNavigate = {}, onLibrary = { libraryOpen = true }, onGems = { gemsOpen = true }, onSettings = { settingsOpen = true }, onAdvanced = { advancedOpen = true }, onPdf = sharePdf, onWeb = onWeb)
                    }
                    VerticalDivider()
                    Box(Modifier.weight(1f)) { content() }
                }
            } else {
                ModalNavigationDrawer(drawerState = drawer, gesturesEnabled = showThreads && allowDrawerOpen,
                    drawerContent = {
                        ModalDrawerSheet(Modifier.width(PlaygroundDimens.drawerPane), drawerContainerColor = colors.surface) {
                            if (showThreads) {
                                ThreadPanel(state, model, onLogout = { logout = true }, onDelete = { deleting = it }, onNavigate = closeDrawer, onLibrary = { libraryOpen = true }, onGems = { gemsOpen = true }, onSettings = { settingsOpen = true }, onAdvanced = { advancedOpen = true }, onPdf = sharePdf, onWeb = { path -> onWeb(path); closeDrawer() })
                            }
                        }
                    }) { content() }
            }

            if (shouldCoverPhoneHistoryUntilClosed(state.starting && !startupSplashEnabled, showThreads, wide, allowDrawerOpen)) {
                Box(
                    Modifier.fillMaxSize().background(
                        Brush.verticalGradient(listOf(colors.background, colors.surfaceContainerLow))
                    )
                ) {
                    CircularProgressIndicator(Modifier.align(Alignment.Center))
                }
            }

            if (modelPicker) ModelPicker(state, onDismiss = { modelPicker = false }, onSelect = { model.chooseModel(it); modelPicker = false })
            if (libraryOpen) LibraryDialog(state, model, onOpenFile = { viewingFile = it }, onDismiss = { libraryOpen = false })
            viewingFile?.let { request ->
                FileViewerDialog(
                    request, loader,
                    download = { model.downloadAttachment(it) },
                    onDismiss = { viewingFile = null },
                    onOpenExternal = onFile,
                )
            }
            if (gemsOpen) GemsDialog(state, model, onDismiss = { gemsOpen = false })
            if (settingsOpen) SettingsDialog(state, model, onDismiss = { settingsOpen = false },
                onLogout = { model.logout(); settingsOpen = false; closeDrawer() }, onWeb = onWeb)
            if (advancedOpen) AdvancedToolsDialog(state, model, onDismiss = { advancedOpen = false }, onWebPath = onWeb,
                onRealtime = {
                    advancedOpen = false
                    if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) realtimeOpen = true
                    else { awaitingMic = true; microphone.launch(Manifest.permission.RECORD_AUDIO) }
                }, onLyria = { advancedOpen = false; lyriaOpen = true })
            if (realtimeOpen) RealtimeStudioDialog(state, model, onDismiss = { realtimeOpen = false })
            if (lyriaOpen) LyriaStudioDialog(state, model, onDismiss = { lyriaOpen = false })
            if (richPasteOpen) RichPasteDialog(state.draft, onDismiss = { richPasteOpen = false }) { text ->
                model.draft(if (state.draft.isBlank()) text else state.draft.trimEnd() + "\n\n" + text)
            }
            maskSource?.let { uri ->
                if (maskOpen) ImageMaskEditor(uri, onDismiss = { maskOpen = false; maskSource = null }) { bytes ->
                    model.uploadImageMask("mask_${System.currentTimeMillis()}.png", bytes)
                    maskOpen = false
                    maskSource = null
                }
            }
            if (attachMenu) AlertDialog(onDismissRequest = { attachMenu = false }, title = { Text("添付を追加") },
                text = { Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    DialogAction(Icons.Rounded.FolderOpen, "ファイルを選択") { attachMenu = false; picker.launch(arrayOf("*/*")) }
                    DialogAction(Icons.Rounded.PhotoLibrary, "写真・動画を選択") { attachMenu = false; photoPicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageAndVideo)) }
                    DialogAction(Icons.Rounded.PhotoCamera, "カメラで撮影") { attachMenu = false; launchCamera() }
                    DialogAction(Icons.Rounded.FolderShared, "ライブラリから選択") { attachMenu = false; libraryOpen = true }
                } },
                confirmButton = { TextButton(onClick = { attachMenu = false }) { Text("閉じる") } })
            if (threadSettings && state.selected != null) ThreadSettingsDialog(state, onDismiss = { threadSettings = false }) {
                title, instruction, includeGlobal, temporary ->
                model.saveThreadSettings(title, instruction, includeGlobal, temporary)
                threadSettings = false
            }
            deleting?.let { thread -> AlertDialog(onDismissRequest = { deleting = null }, title = { Text("チャットを削除しますか？") },
                text = { Text("「${thread.title}」の履歴と紐付く添付ファイルを削除します。この操作は取り消せません。") },
                confirmButton = { TextButton(onClick = { model.deleteThread(thread); deleting = null }) { Text("削除") } },
                dismissButton = { TextButton(onClick = { deleting = null }) { Text("キャンセル") } }) }
            if (logout) AlertDialog(onDismissRequest = { logout = false }, title = { Text("この端末からログアウト") },
                text = { Text("このAndroid端末の連携を取り消します。Webや他の端末のログインは継続します。") },
                confirmButton = { TextButton(onClick = { model.logout(); logout = false; closeDrawer() }) { Text("ログアウト") } },
                dismissButton = { TextButton(onClick = { logout = false }) { Text("キャンセル") } })
            appUpdate?.let { update ->
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

@Composable
private fun DialogAction(icon: androidx.compose.ui.graphics.vector.ImageVector, label: String, onClick: () -> Unit) {
    TextButton(onClick = onClick, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp), shape = RoundedCornerShape(12.dp)) {
        Icon(icon, contentDescription = null)
        Text(label, modifier = Modifier.padding(start = 12.dp).weight(1f), textAlign = androidx.compose.ui.text.style.TextAlign.Start)
    }
}

@Composable
private fun OfflineBanner(onRetry: () -> Unit) {
    Surface(color = MaterialTheme.colorScheme.errorContainer, modifier = Modifier.fillMaxWidth()) {
        Row(Modifier.padding(horizontal = 16.dp, vertical = 4.dp), verticalAlignment = Alignment.CenterVertically) {
            Text("オフラインのようです。接続を確認してください。", modifier = Modifier.weight(1f),
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onErrorContainer)
            TextButton(onClick = onRetry) { Text("再試行") }
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
    onWeb: (String) -> Unit,
) {
    val colors = MaterialTheme.colorScheme
    Column(Modifier.fillMaxSize().statusBarsPadding().padding(horizontal = 12.dp, vertical = 12.dp)) {
        Text(state.selected?.title?.ifBlank { "AI Chat" } ?: "AI Chat", style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold, maxLines = 1, overflow = TextOverflow.Ellipsis)
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            IconButton(onClick = { onSettings(); onNavigate() }) { Icon(Icons.Rounded.Settings, "設定", modifier = Modifier.size(20.dp)) }
            IconButton(onClick = { onLibrary(); onNavigate() }) { Icon(Icons.Rounded.FolderOpen, "ライブラリ", modifier = Modifier.size(20.dp)) }
            IconButton(onClick = { model.newChat(); onNavigate() }, enabled = !state.busy && !state.streaming) { Icon(Icons.Rounded.Add, "新規チャット", tint = colors.primary) }
            IconButton(onClick = { onPdf(); onNavigate() }, enabled = state.selected != null && !state.busy && !state.streaming) { Icon(Icons.Rounded.PictureAsPdf, "PDFを共有") }
            IconButton(onClick = { onAdvanced(); onNavigate() }) { Icon(Icons.Rounded.Layers, "Batch処理・高度な機能", modifier = Modifier.size(20.dp)) }
            IconButton(onClick = model::refresh, enabled = !state.busy && !state.streaming) { Icon(Icons.Rounded.Refresh, "更新", modifier = Modifier.size(20.dp)) }
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
                    IconButton(onClick = { model.toggleBookmark(thread) }, modifier = Modifier.size(38.dp)) {
                        Icon(if (thread.isBookmarked) Icons.Rounded.Star else Icons.Rounded.StarBorder, contentDescription = if (thread.isBookmarked) "ブックマーク解除" else "ブックマーク", modifier = Modifier.size(18.dp), tint = if (thread.isBookmarked) colors.secondary else colors.onSurfaceVariant)
                    }
                    IconButton(onClick = { onDelete(thread) }, modifier = Modifier.size(38.dp)) {
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
        SidebarAction(Icons.Rounded.HelpOutline, "ヘルプ", { onWeb("/help"); onNavigate() })
        SidebarAction(Icons.Rounded.History, "更新履歴", { onWeb("/changelog"); onNavigate() })
        SidebarAction(Icons.Rounded.Logout, "この端末からログアウト", onLogout, danger = true)
        Spacer(Modifier.navigationBarsPadding())
    }
}

@Composable
private fun SidebarAction(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    label: String,
    onClick: () -> Unit,
    danger: Boolean = false,
) {
    val tint = if (danger) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant
    TextButton(onClick = onClick, modifier = Modifier.fillMaxWidth().heightIn(min = 44.dp), shape = RoundedCornerShape(10.dp), contentPadding = PaddingValues(horizontal = 10.dp)) {
        Icon(icon, contentDescription = null, tint = tint, modifier = Modifier.size(19.dp))
        Text(label, modifier = Modifier.padding(start = 11.dp).weight(1f), color = tint, textAlign = androidx.compose.ui.text.style.TextAlign.Start, style = MaterialTheme.typography.bodyMedium)
    }
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
        confirmButton = { TextButton(onClick = { onSave(title, instruction, includeGlobal, temporary) }, enabled = title.length <= 200 && instruction.length <= 100_000) { Text("保存") } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("キャンセル") } },
    )
}

@Composable
private fun Conversation(state: ChatState, model: ChatViewModel, onFile: (String) -> Unit, loader: FileBytesLoader?) {
    val scroll = rememberLazyListState()
    val scope = rememberCoroutineScope()
    var showScrollToBottom by remember { mutableStateOf(false) }
    val live = state.streaming || state.liveContent.isNotEmpty() || state.liveThought.isNotEmpty() || state.cards.isNotEmpty()
    LaunchedEffect(scroll.firstVisibleItemIndex, scroll.layoutInfo.totalItemsCount) {
        val info = scroll.layoutInfo
        val lastVisible = info.visibleItemsInfo.lastOrNull()?.index ?: 0
        showScrollToBottom = info.totalItemsCount > 0 && lastVisible < info.totalItemsCount - 2
    }
    LaunchedEffect(state.messages.size, state.liveContent.length, state.cards.size) {
        val info = scroll.layoutInfo
        val nearBottom = (info.visibleItemsInfo.lastOrNull()?.index ?: 0) >= info.totalItemsCount - 3
        val count = state.messages.size + (if (state.hasOlder) 1 else 0) + (if (live) 1 else 0)
        if (nearBottom && count > 0) scroll.animateScrollToItem(count - 1)
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
            if (state.hasOlder) item(key = "older") { TextButton(onClick = model::olderMessages, enabled = !state.busy, modifier = Modifier.fillMaxWidth()) { Text("以前のメッセージ") } }
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
                MessageCard(message, onFile, model::quoteMessage, loader,
                    onEdit = { model.beginEdit(it) },
                    onRegenerate = { model.regenerate(it) },
                    branchIndex = if (index < 0) 0 else index,
                    branchCount = if (numericId(message) != null) siblings.size else 0,
                    onSwitchBranch = { target -> model.switchBranchByIndex(siblings, target) })
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
                                TextButton(onClick = { model.toggleLibraryFavorite(file) }) { Text(if (file.isFavorite) "★" else "☆") }
                                TextButton(onClick = { renameTarget = file }) { Text("名前変更") }
                                TextButton(onClick = { deleteTarget = file }) { Text("削除") }
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
