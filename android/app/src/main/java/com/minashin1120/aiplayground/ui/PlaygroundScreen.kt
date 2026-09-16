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
import com.minashin1120.aiplayground.data.ThreadItem
import com.minashin1120.aiplayground.data.LibraryFile
import com.minashin1120.aiplayground.data.Gem
import com.minashin1120.aiplayground.data.FixedPrompt
import com.minashin1120.aiplayground.data.recentWebModels
import com.minashin1120.aiplayground.data.CompressionSettings
import com.minashin1120.aiplayground.data.attachmentKind
import com.minashin1120.aiplayground.data.attachmentKindIcon
import com.minashin1120.aiplayground.data.numericId
import com.minashin1120.aiplayground.data.siblingGroup
import kotlinx.coroutines.launch
import java.io.File

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PlaygroundScreen(model: ChatViewModel, onWeb: (String) -> Unit, onFile: (String) -> Unit) {
    val state by model.state.collectAsStateWithLifecycle()
    PlaygroundTheme(darkTheme = state.preferences?.let { !it.lightModeEnabled } ?: isSystemInDarkTheme()) {
        val colors = MaterialTheme.colorScheme
        BoxWithConstraints(
            Modifier.fillMaxSize().background(
                Brush.verticalGradient(listOf(colors.background, colors.surfaceContainerLow))
            )
        ) {
            // Tablets and wide screens keep the history pane beside the conversation.
            val wide = maxWidth >= PlaygroundDimens.breakpoint
            val drawer = rememberDrawerState(DrawerValue.Closed)
            val scope = rememberCoroutineScope()
            LaunchedEffect(Unit) {
                // Do not restore an open history drawer as the app's startup state.
                drawer.snapTo(DrawerValue.Closed)
            }
            var deleting by remember { mutableStateOf<ThreadItem?>(null) }
            var logout by remember { mutableStateOf(false) }
            var modelPicker by remember { mutableStateOf(false) }
            var threadSettings by remember { mutableStateOf(false) }
            var attachMenu by remember { mutableStateOf(false) }
            var libraryOpen by remember { mutableStateOf(false) }
            var gemsOpen by remember { mutableStateOf(false) }
            var settingsOpen by remember { mutableStateOf(false) }
            var advancedOpen by remember { mutableStateOf(false) }
            var cameraUri by remember { mutableStateOf<Uri?>(null) }
            val context = LocalContext.current
            val notifications = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { }
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
                try {
                    speech.launch(Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
                        .putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                        .putExtra(RecognizerIntent.EXTRA_PROMPT, "メッセージを話してください"))
                } catch (_: Exception) { model.notify("音声入力に対応するアプリが見つかりません。") }
            }
            val photoPicker = rememberLauncherForActivityResult(ActivityResultContracts.PickMultipleVisualMedia(30)) { model.upload(it) }
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
            LaunchedEffect(state.notice) {
                state.notice?.let { snackbar.showSnackbar(it, duration = SnackbarDuration.Long); model.dismissNotice() }
            }
            val closeDrawer: () -> Unit = { scope.launch { drawer.close() } }
            val showThreads = state.account != null

            val content: @Composable () -> Unit = {
                Scaffold(
                    modifier = Modifier.imePadding(),
                    containerColor = Color.Transparent,
                    topBar = {
                        TopAppBar(colors = TopAppBarDefaults.topAppBarColors(containerColor = colors.surface.copy(alpha = 0.94f)), title = { Column {
                            Text(state.selected?.title?.ifBlank { "新しいチャット" } ?: "AI Playground", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold, maxLines = 1, overflow = TextOverflow.Ellipsis)
                            if (state.selected?.isTemporary == true || state.newThreadTemporary) Text("一時チャット", style = MaterialTheme.typography.labelSmall, color = colors.secondary)
                        } }, navigationIcon = {
                            if (showThreads && !wide) IconButton(onClick = { scope.launch { drawer.open() } }) {
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
                                model.exportPdf { file ->
                                    try {
                                        val uri = FileProvider.getUriForFile(context, "${context.packageName}.files", file)
                                        val intent = Intent(Intent.ACTION_SEND).setType("application/pdf")
                                            .putExtra(Intent.EXTRA_STREAM, uri)
                                            .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                                        context.startActivity(Intent.createChooser(intent, "PDFを共有"))
                                    } catch (e: Exception) { model.notify("PDFを共有できません。") }
                                }
                            }, enabled = !state.busy && !state.streaming) { Icon(Icons.Rounded.PictureAsPdf, contentDescription = "PDFを共有") }
                            if (showThreads) IconButton(onClick = model::refresh, enabled = !state.busy && !state.streaming) {
                                Icon(Icons.Rounded.Refresh, contentDescription = "更新")
                            }
                        })
                    }, snackbarHost = { SnackbarHost(snackbar) },
                    bottomBar = {
                        if (showThreads) Composer(state, model, { modelPicker = true }, { attachMenu = true }, launchSpeech)
                    }
                ) { padding ->
                    Column(Modifier.fillMaxSize().padding(padding)) {
                        if (state.offline) OfflineBanner(model::reconnect)
                        Box(Modifier.weight(1f)) {
                            when {
                                state.starting -> CircularProgressIndicator(Modifier.align(Alignment.Center))
                                state.account == null -> PairingScreen(state, model, onWeb)
                                else -> Conversation(state, model, onFile, loader)
                            }
                        }
                    }
                }
            }

            if (showThreads && wide) {
                Row(Modifier.fillMaxSize()) {
                    Surface(Modifier.width(PlaygroundDimens.sidePane).fillMaxHeight(), color = colors.surface.copy(alpha = 0.94f)) {
                        ThreadPanel(state, model, onWeb, onLogout = { logout = true }, onDelete = { deleting = it }, onNavigate = {}, onLibrary = { libraryOpen = true }, onGems = { gemsOpen = true }, onSettings = { settingsOpen = true }, onAdvanced = { advancedOpen = true })
                    }
                    VerticalDivider()
                    Box(Modifier.weight(1f)) { content() }
                }
            } else {
                ModalNavigationDrawer(drawerState = drawer, gesturesEnabled = showThreads,
                    drawerContent = {
                        if (showThreads) ModalDrawerSheet(Modifier.width(PlaygroundDimens.drawerPane), drawerContainerColor = colors.surface) {
                            ThreadPanel(state, model, onWeb, onLogout = { logout = true }, onDelete = { deleting = it }, onNavigate = closeDrawer, onLibrary = { libraryOpen = true }, onGems = { gemsOpen = true }, onSettings = { settingsOpen = true }, onAdvanced = { advancedOpen = true })
                        }
                    }) { content() }
            }

            if (modelPicker) ModelPicker(state, onDismiss = { modelPicker = false }, onSelect = { model.chooseModel(it); modelPicker = false })
            if (libraryOpen) LibraryDialog(state, model, onFile, onDismiss = { libraryOpen = false })
            if (gemsOpen) GemsDialog(state, model, onDismiss = { gemsOpen = false })
            if (settingsOpen) SettingsDialog(state, model, onDismiss = { settingsOpen = false }, onLogout = { model.logout(); settingsOpen = false; closeDrawer() })
            if (advancedOpen) AdvancedToolsDialog(state, model, onDismiss = { advancedOpen = false }, onWebPath = onWeb)
            if (attachMenu) AlertDialog(onDismissRequest = { attachMenu = false }, title = { Text("添付を追加") },
                text = { Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    DialogAction(Icons.Rounded.FolderOpen, "ファイルを選択") { attachMenu = false; picker.launch(arrayOf("*/*")) }
                    DialogAction(Icons.Rounded.PhotoLibrary, "写真・動画を選択") { attachMenu = false; photoPicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageAndVideo)) }
                    DialogAction(Icons.Rounded.PhotoCamera, "カメラで撮影") { attachMenu = false; launchCamera() }
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
    onWeb: (String) -> Unit,
    onLogout: () -> Unit,
    onDelete: (ThreadItem) -> Unit,
    onNavigate: () -> Unit,
    onLibrary: () -> Unit,
    onGems: () -> Unit,
    onSettings: () -> Unit,
    onAdvanced: () -> Unit,
) {
    val colors = MaterialTheme.colorScheme
    Column(Modifier.fillMaxSize().statusBarsPadding().padding(horizontal = 12.dp, vertical = 12.dp)) {
        Text(state.selected?.title?.ifBlank { "AI Chat" } ?: "AI Chat", style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold, maxLines = 1, overflow = TextOverflow.Ellipsis)
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            IconButton(onClick = { onSettings(); onNavigate() }) { Icon(Icons.Rounded.Settings, "設定", modifier = Modifier.size(20.dp)) }
            IconButton(onClick = { onLibrary(); onNavigate() }) { Icon(Icons.Rounded.FolderOpen, "ライブラリ", modifier = Modifier.size(20.dp)) }
            IconButton(onClick = { model.newChat(); onNavigate() }, enabled = !state.busy && !state.streaming) { Icon(Icons.Rounded.Add, "新規チャット", tint = colors.primary) }
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
        SidebarAction(Icons.Rounded.Schedule, "一時チャット", { model.newChat(temporary = true); onNavigate() })
        SidebarAction(
            Icons.Rounded.Security,
            "Web設定・安全性確認",
            onClick = { onWeb("/settings") },
        )
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
    LazyColumn(Modifier.fillMaxSize(), contentPadding = PaddingValues(24.dp), verticalArrangement = Arrangement.spacedBy(24.dp)) {
        item {
            Surface(shape = RoundedCornerShape(28.dp), color = MaterialTheme.colorScheme.primaryContainer) {
                Column(Modifier.fillMaxWidth().padding(28.dp)) {
                    Text("✦", fontSize = 48.sp, color = MaterialTheme.colorScheme.primary)
                    Text("ひとつの場所で、\nいろいろなAIと。", style = MaterialTheme.typography.headlineLarge, fontWeight = FontWeight.Bold)
                    Spacer(Modifier.height(16.dp))
                    Text("いつものアカウントとチャット履歴を、そのままAndroidで。")
                }
            }
        }
        item {
            Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                if (state.userCode.isNotBlank() && state.pairing) {
                    Text("1. この確認コードを覚えてください", fontWeight = FontWeight.SemiBold)
                    SelectionContainer { Text(state.userCode.chunked(4).joinToString(" − "), fontSize = 28.sp, fontWeight = FontWeight.Bold) }
                    Text("2. ブラウザーでログインし、コードを入力して端末連携を許可します。")
                    Button(onClick = { onWeb("/android/connect") }, modifier = Modifier.fillMaxWidth()) { Text("ブラウザーで連携を許可") }
                    Text("3. このアプリに戻ると連携を確認します。コードの有効期限は10分です。", style = MaterialTheme.typography.bodySmall)
                    LinearProgressIndicator(Modifier.fillMaxWidth())
                    TextButton(onClick = model::cancelPairing) { Text("連携をキャンセル") }
                } else {
                    Text("ブラウザーでいつものログイン方法を使えます。パスワードやAPIキーを、このアプリに入力する必要はありません。")
                    Button(onClick = model::pair, enabled = !state.pairing, modifier = Modifier.fillMaxWidth().heightIn(min = 52.dp)) {
                        Text(if (state.pairing) "連携を準備しています…" else "アカウントを連携")
                    }
                }
                Text("接続先: ai.minashin1120.com", style = MaterialTheme.typography.labelMedium)
                TextButton(onClick = { onWeb("/") }) { Text("アカウント作成・Webアプリを開く") }
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
    val live = state.streaming || state.liveContent.isNotEmpty() || state.liveThought.isNotEmpty() || state.cards.isNotEmpty()
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
        LazyColumn(
            state = scroll,
            modifier = Modifier.weight(1f).widthIn(max = PlaygroundDimens.contentMax).fillMaxWidth().align(Alignment.CenterHorizontally),
            contentPadding = PaddingValues(horizontal = 16.dp, vertical = 20.dp),
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
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun LibraryDialog(state: ChatState, model: ChatViewModel, onFile: (String) -> Unit, onDismiss: () -> Unit) {
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
                            if (file.isImage) ProtectedImage(file.url.ifBlank { file.filepath }, loader, onFile,
                                modifier = Modifier.fillMaxWidth().heightIn(max = 160.dp), thumbnail = true, contentDescription = file.displayName)
                            Text("${attachmentKindIcon(attachmentKind(file.displayName))} ${file.displayName}",
                                maxLines = 1, overflow = TextOverflow.Ellipsis)
                            FlowRow {
                                TextButton(onClick = { onFile(file.url.ifBlank { file.filepath }) }) { Text("開く") }
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

@Composable
private fun CheckboxRow(label: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Checkbox(checked, onChange)
        Text(label, modifier = Modifier.weight(1f))
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun SettingsDialog(
    state: ChatState,
    model: ChatViewModel,
    onDismiss: () -> Unit,
    onLogout: () -> Unit,
) {
    val prefs = state.preferences
    var defaultModel by remember(prefs?.defaultModel) { mutableStateOf(prefs?.defaultModel.orEmpty()) }
    var thinking by remember(prefs?.defaultEnableThinking) { mutableStateOf(prefs?.defaultEnableThinking ?: false) }
    var search by remember(prefs?.defaultEnableSearch) { mutableStateOf(prefs?.defaultEnableSearch ?: false) }
    var enterToSend by remember(prefs?.enterToSend) { mutableStateOf(prefs?.enterToSend ?: false) }
    var lightMode by remember(prefs?.lightModeEnabled) { mutableStateOf(prefs?.lightModeEnabled ?: false) }
    var autoSearch by remember(prefs?.autoSearchOnLinks) { mutableStateOf(prefs?.autoSearchOnLinks ?: true) }
    var timeout by remember(prefs?.tempChatTimeoutSeconds) { mutableStateOf((prefs?.tempChatTimeoutSeconds ?: 90).toString()) }
    var compressionEnabled by remember(state.compression) { mutableStateOf(state.compression.enabled) }
    var maxSizeMB by remember(state.compression) { mutableStateOf(state.compression.maxSizeMB.toString()) }
    var maxDim by remember(state.compression) { mutableStateOf(state.compression.maxDimension.toString()) }
    var formatOnly by remember(state.compression) { mutableStateOf(state.compression.formatOnly) }
    var outputType by remember(state.compression) { mutableStateOf(state.compression.outputType) }
    var modelPicker by remember { mutableStateOf(false) }
    var confirmLogout by remember { mutableStateOf(false) }
    var settingsTab by remember { mutableStateOf("一般") }
    LaunchedEffect(Unit) { model.loadPreferences() }
    PlaygroundDialog(
        onDismissRequest = onDismiss,
        title = { Text("設定") },
        text = {
            Column(Modifier.heightIn(max = 460.dp).verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                if (state.prefsBusy) LinearProgressIndicator(Modifier.fillMaxWidth())
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    listOf("一般", "画像圧縮", "セッション").forEach { tab ->
                        FilterChip(settingsTab == tab, { settingsTab = tab }, { Text(tab) })
                    }
                }
                if (settingsTab == "一般") {
                Text(state.account?.name.orEmpty(), style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                TextButton(onClick = { modelPicker = true }, modifier = Modifier.fillMaxWidth()) {
                    Text("既定モデル: ${state.account?.models?.firstOrNull { it.id == defaultModel }?.name ?: defaultModel.ifBlank { "未設定" }} ▾")
                }
                CheckboxRow("既定でThinkingを使う", thinking) { thinking = it }
                CheckboxRow("既定でWeb検索を使う", search) { search = it }
                CheckboxRow("Enterで送信", enterToSend) { enterToSend = it }
                CheckboxRow("ライトモード", lightMode) { lightMode = it }
                CheckboxRow("リンクのX投稿を自動検索", autoSearch) { autoSearch = it }
                OutlinedTextField(timeout, { timeout = it.filter { c -> c.isDigit() }.take(6) }, singleLine = true,
                    label = { Text("一時チャットの自動削除（秒）") }, modifier = Modifier.fillMaxWidth())
                }
                if (settingsTab == "画像圧縮") {
                Text("画像の圧縮", fontWeight = FontWeight.SemiBold)
                CheckboxRow("画像を圧縮して送信", compressionEnabled) { compressionEnabled = it }
                if (compressionEnabled) {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedTextField(maxSizeMB, { maxSizeMB = it.filter { c -> c.isDigit() || c == '.' }.take(5) },
                            singleLine = true, label = { Text("最大サイズ (MB)") }, modifier = Modifier.weight(1f))
                        OutlinedTextField(maxDim, { maxDim = it.filter { c -> c.isDigit() }.take(4) },
                            singleLine = true, label = { Text("最大辺 (px)") }, modifier = Modifier.weight(1f))
                    }
                    CheckboxRow("形式のみ変換（サイズ・寸法は変更しない）", formatOnly) { formatOnly = it }
                    FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        listOf("original" to "元の形式", "image/jpeg" to "JPEG", "image/png" to "PNG", "image/webp" to "WebP").forEach { (value, label) ->
                            FilterChip(outputType == value, { outputType = value }, { Text(label) })
                        }
                    }
                }
                }
                if (settingsTab == "セッション") {
                Text("この端末のセッション", fontWeight = FontWeight.SemiBold)
                Text("端末: ${prefs?.deviceName?.ifBlank { "このAndroid端末" } ?: "このAndroid端末"}",
                    style = MaterialTheme.typography.labelSmall)
                if (prefs != null) {
                    Text("連携日時: ${prefs.sessionCreatedAt}", style = MaterialTheme.typography.labelSmall)
                    Text("有効期限: ${prefs.sessionExpiresAt}", style = MaterialTheme.typography.labelSmall)
                }
                Text("暗号化: ${if (prefs?.e2eeEnabled == true) "有効（サーバー管理鍵）" else "無効"} / 2FA: ${if (prefs?.twoFactorEnabled == true) "有効" else "無効"}",
                    style = MaterialTheme.typography.labelSmall)
                Text("APIキー・パスワード・2FAの変更はWeb設定で行います。",
                    style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                TextButton(onClick = { confirmLogout = true }) { Text("この端末の連携を取り消す") }
                }
            }
        },
        confirmButton = {
            TextButton(
                onClick = {
                    model.saveCompressionSettings(CompressionSettings(
                        enabled = compressionEnabled,
                        maxSizeMB = maxSizeMB.toFloatOrNull()?.coerceIn(0.05f, 50f) ?: 1.0f,
                        maxDimension = maxDim.toIntOrNull()?.coerceIn(256, 8192) ?: 1920,
                        outputType = outputType,
                        formatOnly = formatOnly,
                    ))
                    model.savePreferences(defaultModel, thinking, search, enterToSend, lightMode, autoSearch, timeout.toIntOrNull() ?: 90)
                },
                enabled = !state.prefsBusy,
            ) { Text("保存") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("閉じる") } },
    )
    if (modelPicker) ModelPicker(state, onDismiss = { modelPicker = false }, onSelect = { defaultModel = it; modelPicker = false }, selectedId = defaultModel)
    if (confirmLogout) AlertDialog(onDismissRequest = { confirmLogout = false }, title = { Text("この端末からログアウト") },
        text = { Text("このAndroid端末の連携を取り消します。Webや他の端末のログインは継続します。") },
        confirmButton = { TextButton(onClick = { confirmLogout = false; onLogout() }) { Text("ログアウト") } },
        dismissButton = { TextButton(onClick = { confirmLogout = false }) { Text("キャンセル") } })
}
