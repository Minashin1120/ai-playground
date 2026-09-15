package com.minashin1120.aiplayground.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.ChatViewModel
import com.minashin1120.aiplayground.data.ThreadItem
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PlaygroundScreen(model: ChatViewModel, onWeb: (Boolean) -> Unit, onFile: (String) -> Unit) {
    val state by model.state.collectAsStateWithLifecycle()
    PlaygroundTheme {
        val drawer = rememberDrawerState(DrawerValue.Closed)
        val scope = rememberCoroutineScope()
        var deleting by remember { mutableStateOf<ThreadItem?>(null) }
        var logout by remember { mutableStateOf(false) }
        var modelPicker by remember { mutableStateOf(false) }
        val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { model.upload(it) }
        val snackbar = remember { SnackbarHostState() }
        LaunchedEffect(state.notice) {
            state.notice?.let { snackbar.showSnackbar(it, duration = SnackbarDuration.Long); model.dismissNotice() }
        }
        ModalNavigationDrawer(drawerState = drawer, gesturesEnabled = state.account != null,
            drawerContent = {
                if (state.account != null) ModalDrawerSheet(Modifier.width(320.dp)) {
                    Column(Modifier.padding(20.dp).fillMaxWidth()) {
                        Text("AI Playground", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
                        Text(state.account?.name.orEmpty(), color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Spacer(Modifier.height(20.dp))
                        Button(onClick = { model.newChat(); scope.launch { drawer.close() } }, modifier = Modifier.fillMaxWidth()) { Text("＋ 新しいチャット") }
                        Spacer(Modifier.height(12.dp))
                        OutlinedTextField(state.search, model::search, singleLine = true, label = { Text("履歴を検索") }, modifier = Modifier.fillMaxWidth())
                    }
                    LazyColumn(Modifier.weight(1f), contentPadding = PaddingValues(horizontal = 12.dp)) {
                        items(state.threads, key = { it.id }) { thread ->
                            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                                TextButton(onClick = { model.openThread(thread); scope.launch { drawer.close() } }, modifier = Modifier.weight(1f)) {
                                    Column(Modifier.fillMaxWidth()) {
                                        Text(thread.title.ifBlank { "新しいチャット" }, maxLines = 2, overflow = TextOverflow.Ellipsis)
                                        if (thread.model.isNotBlank()) Text(thread.model, style = MaterialTheme.typography.labelSmall, maxLines = 1)
                                    }
                                }
                                TextButton(onClick = { deleting = thread }) { Text("削除", style = MaterialTheme.typography.labelSmall) }
                            }
                        }
                        if (state.nextPage != null) item { TextButton(onClick = model::moreThreads, modifier = Modifier.fillMaxWidth()) { Text("もっと読み込む") } }
                    }
                    TextButton(onClick = { onWeb(false) }, modifier = Modifier.fillMaxWidth()) { Text("Web設定・安全性確認") }
                    TextButton(onClick = { logout = true }, modifier = Modifier.fillMaxWidth()) { Text("この端末からログアウト") }
                    Spacer(Modifier.navigationBarsPadding())
                }
            }) {
            Scaffold(
                modifier = Modifier.imePadding(),
                topBar = {
                    TopAppBar(title = { Column {
                        Text(state.selected?.title?.ifBlank { "新しいチャット" } ?: "AI Playground", maxLines = 1, overflow = TextOverflow.Ellipsis)
                        Text(if (state.account == null) "あなたのAIを、ポケットに。" else "Android · ${state.account?.name}", style = MaterialTheme.typography.labelSmall)
                    } }, navigationIcon = {
                        if (state.account != null) TextButton(onClick = { scope.launch { drawer.open() } }) { Text("履歴") }
                    }, actions = {
                        if (state.account != null) TextButton(onClick = model::refresh, enabled = !state.busy && !state.streaming) { Text("更新") }
                    })
                }, snackbarHost = { SnackbarHost(snackbar) },
                bottomBar = {
                    if (state.account != null) Composer(state, model, { modelPicker = true }, { picker.launch(arrayOf("*/*")) })
                }
            ) { padding ->
                Box(Modifier.fillMaxSize().padding(padding)) {
                    when {
                        state.starting -> CircularProgressIndicator(Modifier.align(Alignment.Center))
                        state.account == null -> PairingScreen(state, model, onWeb)
                        else -> Conversation(state, model, onFile)
                    }
                }
            }
        }
        if (modelPicker) ModelPicker(state, onDismiss = { modelPicker = false }, onSelect = { model.chooseModel(it); modelPicker = false })
        deleting?.let { thread -> AlertDialog(onDismissRequest = { deleting = null }, title = { Text("チャットを削除しますか？") },
            text = { Text("「${thread.title}」の履歴と紐付く添付ファイルを削除します。この操作は取り消せません。") },
            confirmButton = { TextButton(onClick = { model.deleteThread(thread); deleting = null }) { Text("削除") } },
            dismissButton = { TextButton(onClick = { deleting = null }) { Text("キャンセル") } }) }
        if (logout) AlertDialog(onDismissRequest = { logout = false }, title = { Text("この端末からログアウト") },
            text = { Text("このAndroid端末の連携を取り消します。Webや他の端末のログインは継続します。") },
            confirmButton = { TextButton(onClick = { model.logout(); logout = false; scope.launch { drawer.close() } }) { Text("ログアウト") } },
            dismissButton = { TextButton(onClick = { logout = false }) { Text("キャンセル") } })
    }
}

@Composable
private fun PairingScreen(state: ChatState, model: ChatViewModel, onWeb: (Boolean) -> Unit) {
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
                    Button(onClick = { onWeb(true) }, modifier = Modifier.fillMaxWidth()) { Text("ブラウザーで連携を許可") }
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
                TextButton(onClick = { onWeb(false) }) { Text("アカウント作成・Webアプリを開く") }
            }
        }
    }
}

@Composable
private fun Conversation(state: ChatState, model: ChatViewModel, onFile: (String) -> Unit) {
    val scroll = rememberLazyListState()
    val live = state.streaming || state.liveContent.isNotEmpty() || state.liveThought.isNotEmpty()
    LaunchedEffect(state.messages.size, state.liveContent.length) {
        val info = scroll.layoutInfo
        val nearBottom = (info.visibleItemsInfo.lastOrNull()?.index ?: 0) >= info.totalItemsCount - 3
        val count = state.messages.size + (if (state.hasOlder) 1 else 0) + (if (live) 1 else 0)
        if (nearBottom && count > 0) scroll.animateScrollToItem(count - 1)
    }
    Column(Modifier.fillMaxSize()) {
        if (state.busy) LinearProgressIndicator(Modifier.fillMaxWidth())
        if (state.jobId != null && !state.streaming) Row(Modifier.fillMaxWidth().padding(horizontal = 16.dp), verticalAlignment = Alignment.CenterVertically) {
            Text("生成の状態を確認できます", modifier = Modifier.weight(1f), style = MaterialTheme.typography.bodySmall)
            TextButton(onClick = model::resume) { Text("再接続") }
        }
        if (state.retryAvailable) TextButton(onClick = model::retry, modifier = Modifier.fillMaxWidth(), enabled = !state.streaming) { Text("同じ送信を再試行（二重送信を防止）") }
        LazyColumn(state = scroll, modifier = Modifier.weight(1f), contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
            if (state.hasOlder) item(key = "older") { TextButton(onClick = model::olderMessages, enabled = !state.busy, modifier = Modifier.fillMaxWidth()) { Text("以前のメッセージ") } }
            if (state.messages.isEmpty() && !state.busy) item(key = "welcome") {
                Column(Modifier.fillMaxWidth().padding(vertical = 32.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
                    Text("今日は何を考えよう？", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
                    Text("モデルを選んで、気になっていることを話してみましょう。", color = MaterialTheme.colorScheme.onSurfaceVariant)
                    listOf("アイデアを一緒に考えて", "難しい内容をわかりやすく説明して", "文章の改善を手伝って").forEach { suggestion ->
                        OutlinedButton(onClick = { model.draft(suggestion) }, modifier = Modifier.fillMaxWidth()) { Text(suggestion) }
                    }
                }
            }
            items(state.messages, key = { it.id }) { message -> MessageCard(message, onFile) }
            if (live) item(key = "live") { LiveMessage(state, onFile) }
        }
    }
}

@Composable
private fun ModelPicker(state: ChatState, onDismiss: () -> Unit, onSelect: (String) -> Unit) {
    var query by remember { mutableStateOf("") }
    AlertDialog(onDismissRequest = onDismiss, title = { Text("モデルを選択") }, text = {
        Column {
            OutlinedTextField(query, { query = it }, singleLine = true, label = { Text("モデル名で検索") })
            Text("通常チャット用のモデルを選んでください。APIキーはWeb設定を利用します。", style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(vertical = 12.dp))
            LazyColumn(Modifier.heightIn(max = 380.dp)) {
                val models = state.account?.models.orEmpty().filter {
                    it.name.contains(query, ignoreCase = true) || it.id.contains(query, ignoreCase = true) || it.providerLabel.contains(query, ignoreCase = true)
                }.sortedWith(compareBy({ !it.selectable }, { it.providerLabel }, { it.name }))
                items(models, key = { it.id }) { info ->
                    TextButton(onClick = { onSelect(info.id) }, enabled = info.selectable, modifier = Modifier.fillMaxWidth().heightIn(min = 56.dp)) {
                        Column(Modifier.fillMaxWidth()) {
                            Text((if (info.id == state.model) "✓ " else "") + info.name, fontWeight = FontWeight.SemiBold)
                            Text(buildString {
                                append(info.providerLabel)
                                if (!info.selectable) append(if (info.deprecated) " · 提供終了" else " · ${info.mode}はWebで利用")
                            }, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                }
            }
        }
    }, confirmButton = { TextButton(onClick = onDismiss) { Text("閉じる") } })
}
