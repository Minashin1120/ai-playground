package com.minashin1120.aiplayground.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
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
        BoxWithConstraints(Modifier.fillMaxSize()) {
            // Tablets and wide screens keep the history pane beside the conversation.
            val wide = maxWidth >= PlaygroundDimens.breakpoint
            val drawer = rememberDrawerState(DrawerValue.Closed)
            val scope = rememberCoroutineScope()
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
                    topBar = {
                        TopAppBar(title = { Column {
                            Text(state.selected?.title?.ifBlank { "新しいチャット" } ?: "AI Playground", maxLines = 1, overflow = TextOverflow.Ellipsis)
                            Text(if (state.account == null) "あなたのAIを、ポケットに。" else "Android · ${state.account?.name}", style = MaterialTheme.typography.labelSmall)
                        } }, navigationIcon = {
                            if (showThreads && !wide) TextButton(onClick = { scope.launch { drawer.open() } }) { Text("履歴") }
                        }, actions = {
                            if (state.selected != null) TextButton(onClick = { threadSettings = true }, enabled = !state.busy && !state.streaming) { Text("設定") }
                            if (state.selected != null) TextButton(onClick = {
                                model.exportPdf { file ->
                                    try {
                                        val uri = FileProvider.getUriForFile(context, "${context.packageName}.files", file)
                                        val intent = Intent(Intent.ACTION_SEND).setType("application/pdf")
                                            .putExtra(Intent.EXTRA_STREAM, uri)
                                            .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                                        context.startActivity(Intent.createChooser(intent, "PDFを共有"))
                                    } catch (e: Exception) { model.notify("PDFを共有できません。") }
                                }
                            }, enabled = !state.busy && !state.streaming) { Text("PDF") }
                            if (showThreads) TextButton(onClick = model::refresh, enabled = !state.busy && !state.streaming) { Text("更新") }
                        })
                    }, snackbarHost = { SnackbarHost(snackbar) },
                    bottomBar = {
                        if (showThreads) Composer(state, model, { modelPicker = true }, { attachMenu = true })
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
                    Surface(Modifier.width(PlaygroundDimens.sidePane).fillMaxHeight(), tonalElevation = 1.dp) {
                        ThreadPanel(state, model, onWeb, onLogout = { logout = true }, onDelete = { deleting = it }, onNavigate = {}, onLibrary = { libraryOpen = true }, onGems = { gemsOpen = true }, onSettings = { settingsOpen = true }, onAdvanced = { advancedOpen = true })
                    }
                    VerticalDivider()
                    Box(Modifier.weight(1f)) { content() }
                }
            } else {
                ModalNavigationDrawer(drawerState = drawer, gesturesEnabled = showThreads,
                    drawerContent = {
                        if (showThreads) ModalDrawerSheet(Modifier.width(PlaygroundDimens.drawerPane)) {
                            ThreadPanel(state, model, onWeb, onLogout = { logout = true }, onDelete = { deleting = it }, onNavigate = closeDrawer, onLibrary = { libraryOpen = true }, onGems = { gemsOpen = true }, onSettings = { settingsOpen = true }, onAdvanced = { advancedOpen = true })
                        }
                    }) { content() }
            }

            if (modelPicker) ModelPicker(state, onDismiss = { modelPicker = false }, onSelect = { model.chooseModel(it); modelPicker = false })
            if (libraryOpen) LibraryDialog(state, model, onDismiss = { libraryOpen = false })
            if (gemsOpen) GemsDialog(state, model, onDismiss = { gemsOpen = false })
            if (settingsOpen) SettingsDialog(state, model, onDismiss = { settingsOpen = false }, onLogout = { model.logout(); settingsOpen = false; closeDrawer() })
            if (advancedOpen) AdvancedToolsDialog(state, model, onDismiss = { advancedOpen = false }, onWebPath = onWeb)
            if (attachMenu) AlertDialog(onDismissRequest = { attachMenu = false }, title = { Text("添付を追加") },
                text = { Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    TextButton(onClick = { attachMenu = false; picker.launch(arrayOf("*/*")) }, modifier = Modifier.fillMaxWidth()) { Text("📂 ファイルを選択") }
                    TextButton(onClick = { attachMenu = false; photoPicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageAndVideo)) }, modifier = Modifier.fillMaxWidth()) { Text("🖼 写真・動画を選択") }
                    TextButton(onClick = { attachMenu = false; launchCamera() }, modifier = Modifier.fillMaxWidth()) { Text("📷 カメラで撮影") }
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
    Column(Modifier.fillMaxSize().padding(20.dp)) {
        Column(Modifier.fillMaxWidth()) {
            Text("AI Playground", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
            Text(state.account?.name.orEmpty(), color = MaterialTheme.colorScheme.onSurfaceVariant)
            Spacer(Modifier.height(20.dp))
            Button(onClick = { model.newChat(); onNavigate() }, modifier = Modifier.fillMaxWidth()) { Text("＋ 新しいチャット") }
            TextButton(onClick = { model.newChat(temporary = true); onNavigate() }, modifier = Modifier.fillMaxWidth()) { Text("◷ 一時チャット") }
            Spacer(Modifier.height(12.dp))
            OutlinedTextField(state.search, model::search, singleLine = true, label = { Text("履歴を検索") }, modifier = Modifier.fillMaxWidth())
        }
        LazyColumn(Modifier.weight(1f), contentPadding = PaddingValues(vertical = 12.dp)) {
            items(state.threads, key = { it.id }) { thread ->
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    TextButton(onClick = { model.openThread(thread); onNavigate() }, modifier = Modifier.weight(1f)) {
                        Column(Modifier.fillMaxWidth()) {
                            Text(buildString {
                                if (thread.isBookmarked) append("★ ")
                                if (thread.isTemporary) append("◷ ")
                                append(thread.title.ifBlank { "新しいチャット" })
                            }, maxLines = 2, overflow = TextOverflow.Ellipsis)
                            if (thread.model.isNotBlank()) Text(thread.model, style = MaterialTheme.typography.labelSmall, maxLines = 1)
                        }
                    }
                    TextButton(onClick = { model.toggleBookmark(thread) }) {
                        Text(if (thread.isBookmarked) "★" else "☆", style = MaterialTheme.typography.titleMedium)
                    }
                    TextButton(onClick = { onDelete(thread) }) { Text("削除", style = MaterialTheme.typography.labelSmall) }
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
        TextButton(onClick = onSettings, modifier = Modifier.fillMaxWidth()) { Text("一般設定") }
        TextButton(onClick = onLibrary, modifier = Modifier.fillMaxWidth()) { Text("ファイルライブラリ") }
        TextButton(onClick = onGems, modifier = Modifier.fillMaxWidth()) { Text("Gems") }
        TextButton(onClick = onAdvanced, modifier = Modifier.fillMaxWidth()) { Text("高度な機能・Batch") }
        TextButton(onClick = { onWeb("/settings") }, modifier = Modifier.fillMaxWidth()) { Text("Web設定・安全性確認") }
        TextButton(onClick = onLogout, modifier = Modifier.fillMaxWidth()) { Text("この端末からログアウト") }
        Spacer(Modifier.navigationBarsPadding())
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
    AlertDialog(
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
        LazyColumn(state = scroll, modifier = Modifier.weight(1f), contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
            if (state.hasOlder) item(key = "older") { TextButton(onClick = model::olderMessages, enabled = !state.busy, modifier = Modifier.fillMaxWidth()) { Text("以前のメッセージ") } }
            if (state.messages.isEmpty() && !state.busy && !live) item(key = "welcome") {
                Column(Modifier.fillMaxWidth().padding(vertical = 32.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
                    Text("今日は何を考えよう？", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
                    Text("モデルを選んで、気になっていることを話してみましょう。", color = MaterialTheme.colorScheme.onSurfaceVariant)
                    listOf("アイデアを一緒に考えて", "難しい内容をわかりやすく説明して", "文章の改善を手伝って").forEach { suggestion ->
                        OutlinedButton(onClick = { model.draft(suggestion) }, modifier = Modifier.fillMaxWidth()) { Text(suggestion) }
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

@Composable
private fun ModelPicker(state: ChatState, onDismiss: () -> Unit, onSelect: (String) -> Unit, selectedId: String = state.model) {
    var query by remember { mutableStateOf("") }
    AlertDialog(onDismissRequest = onDismiss, title = { Text("モデルを選択") }, text = {
        Column {
            OutlinedTextField(query, { query = it }, singleLine = true, label = { Text("モデル名で検索") })
            Text("チャット・画像・動画・OCR・音声モデルを選べます。APIキーはWeb設定を利用します。", style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(vertical = 12.dp))
            LazyColumn(Modifier.heightIn(max = 380.dp)) {
                val models = state.account?.models.orEmpty().filter {
                    it.name.contains(query, ignoreCase = true) || it.id.contains(query, ignoreCase = true) || it.providerLabel.contains(query, ignoreCase = true)
                }.sortedWith(compareBy({ !it.selectable }, { it.providerLabel }, { it.name }))
                items(models, key = { it.id }) { info ->
                    TextButton(onClick = { onSelect(info.id) }, enabled = info.selectable, modifier = Modifier.fillMaxWidth().heightIn(min = 56.dp)) {
                        Column(Modifier.fillMaxWidth()) {
                            Text((if (info.id == selectedId) "✓ " else "") + info.name, fontWeight = FontWeight.SemiBold)
                            Text(buildString {
                                append(info.providerLabel)
                                append(" · ${info.mode}")
                                if (!info.selectable) append(if (info.deprecated) " · 提供終了" else " · Webで利用")
                            }, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                }
            }
        }
    }, confirmButton = { TextButton(onClick = onDismiss) { Text("閉じる") } })
}

@Composable
private fun LibraryDialog(state: ChatState, model: ChatViewModel, onDismiss: () -> Unit) {
    var renameTarget by remember { mutableStateOf<LibraryFile?>(null) }
    var deleteTarget by remember { mutableStateOf<LibraryFile?>(null) }
    LaunchedEffect(Unit) { model.refreshLibrary() }
    AlertDialog(
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
                            Text("${attachmentKindIcon(attachmentKind(file.displayName))} ${file.displayName}",
                                maxLines = 1, overflow = TextOverflow.Ellipsis)
                            Row {
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
    AlertDialog(
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
    if (creating) GemEditorDialog(gem = null, onDismiss = { creating = false }) { n, d, i, m, done ->
        model.saveGem(null, n, d, i, m) { ok -> if (ok) creating = false; done(ok) }
    }
    editor?.let { gem -> GemEditorDialog(gem = gem, onDismiss = { editor = null }) { n, d, i, m, done ->
        model.saveGem(gem.uuid, n, d, i, m) { ok -> if (ok) editor = null; done(ok) }
    } }
}

@Composable
private fun GemEditorDialog(
    gem: Gem?,
    onDismiss: () -> Unit,
    onSave: (String, String, String, String, (Boolean) -> Unit) -> Unit,
) {
    var name by remember { mutableStateOf(gem?.name.orEmpty()) }
    var description by remember { mutableStateOf(gem?.description.orEmpty()) }
    var instruction by remember { mutableStateOf(gem?.instruction.orEmpty()) }
    var defaultModel by remember { mutableStateOf(gem?.defaultModel.orEmpty()) }
    var saving by remember { mutableStateOf(false) }
    AlertDialog(
        onDismissRequest = { if (!saving) onDismiss() },
        title = { Text(if (gem == null) "Gemを作成" else "Gemを編集") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedTextField(name, { name = it }, singleLine = true, label = { Text("名前") }, modifier = Modifier.fillMaxWidth())
                OutlinedTextField(description, { description = it }, label = { Text("説明") },
                    minLines = 1, maxLines = 3, modifier = Modifier.fillMaxWidth())
                OutlinedTextField(instruction, { instruction = it }, label = { Text("指示") },
                    minLines = 3, maxLines = 8, modifier = Modifier.fillMaxWidth())
                OutlinedTextField(defaultModel, { defaultModel = it }, singleLine = true,
                    label = { Text("既定モデル（任意・モデルID）") }, modifier = Modifier.fillMaxWidth())
            }
        },
        confirmButton = {
            TextButton(
                onClick = { saving = true; onSave(name.trim(), description.trim(), instruction.trim(), defaultModel.trim()) { saving = false } },
                enabled = !saving && name.isNotBlank() && name.length <= 100 && instruction.length <= 100_000,
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
    LaunchedEffect(Unit) { model.loadPreferences() }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("一般設定") },
        text = {
            Column(Modifier.heightIn(max = 460.dp).verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                if (state.prefsBusy) LinearProgressIndicator(Modifier.fillMaxWidth())
                Text(state.account?.name.orEmpty(), style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                TextButton(onClick = { modelPicker = true }, modifier = Modifier.fillMaxWidth()) {
                    Text("既定モデル: ${defaultModel.ifBlank { "未設定" }} ▾")
                }
                CheckboxRow("既定でThinkingを使う", thinking) { thinking = it }
                CheckboxRow("既定でWeb検索を使う", search) { search = it }
                CheckboxRow("Enterで送信", enterToSend) { enterToSend = it }
                CheckboxRow("ライトモード", lightMode) { lightMode = it }
                CheckboxRow("リンクのX投稿を自動検索", autoSearch) { autoSearch = it }
                OutlinedTextField(timeout, { timeout = it.filter { c -> c.isDigit() }.take(6) }, singleLine = true,
                    label = { Text("一時チャットの自動削除（秒）") }, modifier = Modifier.fillMaxWidth())
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
                HorizontalDivider(Modifier.padding(vertical = 4.dp))
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
