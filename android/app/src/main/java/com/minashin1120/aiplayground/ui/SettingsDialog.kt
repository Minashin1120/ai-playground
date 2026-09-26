package com.minashin1120.aiplayground.ui

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.EnterTransition
import androidx.compose.animation.ExitTransition
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.relocation.BringIntoViewRequester
import androidx.compose.foundation.relocation.bringIntoViewRequester
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.material3.Text
import androidx.lifecycle.viewModelScope
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.AppUpdateUiState
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.ChatViewModel
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.CacheCategory
import kotlinx.coroutines.delay

/** SharedPreferences key of the Web `GEMINI_LOCAL_PY_DIALOG_KEY` (stored per device, like the browser). */
internal const val GEMINI_LOCAL_PY_DIALOG_PREF = "gemini_local_py_dialog"

/**
 * Web settings modal (`templates/chat/overlay_settings.html`): header, search, ten tabs, cards and
 * the キャンセル／保存 footer. Phones show the tabs as a scrolling row, wider screens as a 188dp column.
 */
@Composable
fun SettingsDialog(
    state: ChatState,
    model: ChatViewModel,
    onDismiss: () -> Unit,
    onLogout: () -> Unit,
    onWeb: (String) -> Unit,
    appUpdate: AppUpdateUiState = AppUpdateUiState(),
    onCheckForUpdate: () -> Unit = {},
    onBubble: () -> Unit = {},
    initialTab: String = "一般",
) {
    // `onLogout` stays in the signature for callers; like Web, logout lives in the sidebar footer only.
    val context = LocalContext.current
    val devicePrefs = remember { context.getSharedPreferences("settings_local", 0) }
    var localPythonDialog by remember { mutableStateOf(devicePrefs.getBoolean(GEMINI_LOCAL_PY_DIALOG_PREF, true)) }
    var tab by remember { mutableStateOf(SettingsTab.entries.firstOrNull { it.label == initialTab || it.id == initialTab } ?: SettingsTab.General) }
    var query by remember { mutableStateOf("") }
    var jumpTarget by remember { mutableStateOf<String?>(null) }
    var colorPicker by remember { mutableStateOf(false) }
    var confirmCache by remember { mutableStateOf<CacheCategory?>(null) }
    // The form is filled from the payload fetched when the modal opens (Web `openSettingsModal`).
    var loaded by remember { mutableStateOf(state.offline) }
    var sawBusy by remember { mutableStateOf(false) }
    LaunchedEffect(state.account?.id, state.offline) {
        if (state.account != null) {
            model.loadPreferences()
            if (!state.offline) {
                model.loadStorageUsage()
                model.loadFeedback()
                model.loadMcpServers()
                model.loadSecurity()
            }
        }
    }
    LaunchedEffect(state.prefsBusy) {
        if (state.prefsBusy) sawBusy = true
        else if (sawBusy || state.offline) loaded = true
    }
    LaunchedEffect(Unit) { delay(4000); loaded = true }
    val prefs = state.preferences
    val form = remember(if (loaded) "loaded" else prefs) { SettingsForm(prefs) }
    val notify: (String) -> Unit = model::notify
    val extras = SettingsExtras(appUpdate, onCheckForUpdate, onBubble, onWeb) { confirmCache = it }
    val ops = remember(model) { AccountOps(model.viewModelScope, model::accountApi, notify) }
    var confirmRequest by remember { mutableStateOf<Pair<String, () -> Unit>?>(null) }
    val transfer by model.accountTransfer.state.collectAsState()
    val transferReauth by model.accountTransfer.reauth.collectAsState()
    val importForm = remember { ImportFormState() }
    val importPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            importForm.fileUri = uri
            importForm.fileName = context.contentResolver.query(uri, arrayOf(android.provider.OpenableColumns.DISPLAY_NAME), null, null, null)
                ?.use { cursor -> if (cursor.moveToFirst()) cursor.getString(0) else null } ?: uri.lastPathSegment
        }
    }
    val exportSaver = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/zip")) { uri ->
        if (uri != null) model.accountTransfer.download(uri)
    }
    LaunchedEffect(Unit) { if (!state.offline) model.accountTransfer.refreshLatestExport() }
    val accountCtx = AccountSettingsContext(
        model = model, ops = ops, confirm = { message, action -> confirmRequest = message to action }, notify = notify, onWeb = onWeb,
        pickImportFile = { importPicker.launch(arrayOf("application/zip", "application/x-zip-compressed", "application/octet-stream")) },
        saveExport = { exportSaver.launch("ai-playground-account.zip") },
    )
    val dataTab = dataCards(state, model, form, extras).toMutableList()
        .apply { add(indexOfFirst { it.key == "debug" }.coerceAtLeast(0), accountDataCard(transfer, importForm, accountCtx)) }

    val cards = generalCards(state, form, localPythonDialog, { localPythonDialog = it }, notify) + androidCard(state, extras) +
        apiCards(state, form, notify) + promptCards(state, form) + displayCards(form) { colorPicker = true } +
        dataTab + accountCards(state, form, accountCtx) + securityCards(state, form, accountCtx) +
        twoFactorCards(state, form, accountCtx) + feedbackCards(state, model, notify) + mcpCards(state, model, extras, ops)

    PlaygroundDialog(
        onDismissRequest = onDismiss,
        title = { Text("設定") },
        icon = R.drawable.fa_solid_cog,
        subtitle = "アプリの動作・表示・セキュリティを管理",
        fillHeight = true,
        padBody = false,
        text = {
            val phone = LocalConfiguration.current.screenWidthDp < 768
            Column(Modifier.fillMaxSize()) {
                SettingsSearchBox(query, { query = it }, phone)
                val content: @Composable (Modifier) -> Unit = { modifier ->
                    SettingsContent(
                        cards = cards, tab = tab, query = query.trim(), phone = phone, modifier = modifier,
                        jumpTarget = jumpTarget, onJumpDone = { jumpTarget = null },
                        onJump = { spec -> query = ""; tab = spec.tab; jumpTarget = spec.key },
                    )
                }
                if (phone) {
                    SettingsTabRow(tab, cards, query.trim(), phone = true) { query = ""; tab = it }
                    content(Modifier.weight(1f))
                } else Row(Modifier.weight(1f).drawBehind {
                    drawLine(Color(0xFF1A2338), androidx.compose.ui.geometry.Offset(0f, 0f), androidx.compose.ui.geometry.Offset(size.width, 0f), 1.dp.toPx())
                }) {
                    SettingsTabRow(tab, cards, query.trim(), phone = false) { query = ""; tab = it }
                    content(Modifier.weight(1f))
                }
            }
        },
        dismissButton = {
            WebButton(onClick = onDismiss, variant = WebButtonVariant.Ghost, contentPadding = PaddingValues(horizontal = 14.dp, vertical = 10.dp)) {
                Text("キャンセル")
            }
        },
        confirmButton = {
            WebButton(
                onClick = {
                    if (!loaded) notify("設定を読み込み中です。完了するまでお待ちください")
                    else {
                        devicePrefs.edit().putBoolean(GEMINI_LOCAL_PY_DIALOG_PREF, localPythonDialog).apply()
                        val username = form.newUsername.trim()
                        val password = form.newPassword
                        val e2eeChanged = form.e2ee != form.e2eeLoaded
                        if (username.isEmpty() && password.isEmpty() && !e2eeChanged) {
                            model.savePreferences(form.payload())
                            onDismiss()
                        } else ops.run("設定の保存に失敗しました") {
                            // Web sends these with the same save; a taken user name is left unchanged there too.
                            if (username.isNotEmpty() || password.isNotEmpty()) {
                                try { changeCredentials(username, password) } catch (e: com.minashin1120.aiplayground.data.ApiException) {
                                    when (e.code) {
                                        "username_unavailable" -> if (password.isNotEmpty()) changeCredentials("", password)
                                        "invalid_username" -> throw com.minashin1120.aiplayground.data.ApiException(e.status, org.json.JSONObject().put("error", "Username must be 3-80 characters"))
                                        "invalid_password" -> throw com.minashin1120.aiplayground.data.ApiException(e.status, org.json.JSONObject().put("error", "Password must be 8-256 characters"))
                                        else -> throw e
                                    }
                                }
                                form.newUsername = ""
                                form.newPassword = ""
                            }
                            // Web shows the migration notice instead of 「設定を保存しました」.
                            val e2eeMessage = if (e2eeChanged) setE2ee(form.e2ee) else null
                            model.savePreferences(form.payload(), e2eeMessage ?: "設定を保存しました")
                            onDismiss()
                        }
                    }
                },
                variant = WebButtonVariant.Primary,
                enabled = !state.prefsBusy,
                contentPadding = PaddingValues(horizontal = 14.dp, vertical = 10.dp),
            ) {
                FaIcon(R.drawable.fa_solid_save, null, size = 13.dp)
                Text("保存")
            }
        },
    )
    val pendingReauth = ops.reauthRetry ?: transferReauth
    if (pendingReauth != null) ReauthDialog(model::accountApi, onDismiss = {
        ops.reauthRetry = null
        model.accountTransfer.reauth.value = null
    }) {
        ops.reauthRetry = null
        model.accountTransfer.reauth.value = null
        pendingReauth()
    }
    confirmRequest?.let { (message, action) ->
        BrowserConfirmDialog(message) { ok -> confirmRequest = null; if (ok) action() }
    }
    transfer.dedupePreview?.let { preview ->
        val kept = if (preview.keptReferenced > 0) "\n※チャットから参照されているため、ファイル ${preview.keptReferenced}件は削除せず残します。" else ""
        BrowserConfirmDialog("重複データが ${preview.total}件 見つかりました。\n\n${preview.parts}$kept\n\n同じ内容のデータは最も古い1件を残して削除します。続行しますか？") { ok ->
            model.accountTransfer.confirmDedupe(ok)
        }
    }
    transfer.settingsChanges?.let { changes ->
        ImportSettingsConfirmDialog(changes) { ok -> model.accountTransfer.answerSettingsChanges(ok) }
    }
    transfer.fileSelection?.let { request ->
        ImportFileSelectionDialog(request) { selected -> model.accountTransfer.answerFileSelection(selected) }
    }
    if (colorPicker) ColorPickerDialog(normalizeWebHex(form.themeColor) ?: THEME_DEFAULT, onDismiss = { colorPicker = false }) {
        form.themeColor = it
        colorPicker = false
    }
    confirmCache?.let { category ->
        val label = if (category == CacheCategory.CHAT_HISTORY) "チャット履歴" else "ファイル"
        BrowserConfirmDialog("端末に保存された${label}キャッシュだけを削除します。サーバー上のデータは削除されません。") { ok ->
            if (ok) model.clearOfflineCache(category)
            confirmCache = null
        }
    }
}

/** `.settings-toolbar` search box with the clear button. */
@Composable
private fun SettingsSearchBox(value: String, onChange: (String) -> Unit, @Suppress("UNUSED_PARAMETER") phone: Boolean) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(12.dp)
    val style = TextStyle(fontSize = 13.sp, lineHeight = 18.85.sp, color = web.text, fontFamily = WebFonts.sans)
    Box(Modifier.fillMaxWidth().padding(start = 14.dp, end = 14.dp, bottom = 10.dp)) {
        BasicTextField(
            value = value, onValueChange = onChange, singleLine = true, textStyle = style, cursorBrush = SolidColor(web.text),
            modifier = Modifier.fillMaxWidth().clip(shape)
                .background(if (web.isLight) Color.White else Color.White.copy(alpha = 0.04f))
                .border(1.dp, if (web.isLight) Color(15, 23, 42).copy(alpha = 0.14f) else web.line, shape)
                .semantics { contentDescription = "設定を検索" },
            decorationBox = { inner ->
                Row(Modifier.padding(horizontal = 12.dp, vertical = 10.dp), verticalAlignment = Alignment.CenterVertically) {
                    FaIcon(R.drawable.fa_solid_search, null, size = 12.dp, tint = web.muted)
                    Box(Modifier.weight(1f).padding(start = 12.dp)) {
                        if (value.isEmpty()) Text("設定を検索...", style = style.copy(color = web.muted))
                        inner()
                    }
                    if (value.isNotEmpty()) Box(
                        Modifier.size(22.dp).clip(RoundedCornerShape(999.dp)).background(Color.White.copy(alpha = 0.06f))
                            .clickable(role = Role.Button) { onChange("") }.semantics { contentDescription = "検索をクリア" },
                        contentAlignment = Alignment.Center,
                    ) { FaIcon(R.drawable.fa_solid_times, null, size = 10.dp, tint = web.muted) }
                }
            },
        )
    }
}

/** `#settings-tabs`: a scrolling pill row on phones, a vertical list beside the content otherwise. */
@Composable
private fun SettingsTabRow(active: SettingsTab, cards: List<SettingsCardSpec>, query: String, phone: Boolean, onSelect: (SettingsTab) -> Unit) {
    val web = LocalWebPalette.current
    val wrapBg = if (web.isLight) Color(15, 23, 42).copy(alpha = 0.035f) else Color(8, 12, 22).copy(alpha = if (phone) 0.45f else 0.55f)
    val lineColor = if (web.isLight) Color(15, 23, 42).copy(alpha = 0.10f) else web.line
    val hits = if (query.isEmpty()) emptyMap() else cards.filter { it.matches(query) }.groupingBy { it.tab }.eachCount()
    val tabs: @Composable (Modifier) -> Unit = { itemModifier ->
        SettingsTab.entries.forEach { entry ->
            SettingsTabButton(entry, entry == active, hits[entry] ?: 0, phone, itemModifier) { onSelect(entry) }
        }
    }
    if (phone) {
        Row(
            Modifier.fillMaxWidth().background(wrapBg)
                .drawBehind {
                    drawLine(lineColor, androidx.compose.ui.geometry.Offset(0f, 0f), androidx.compose.ui.geometry.Offset(size.width, 0f), 1.dp.toPx())
                    drawLine(lineColor, androidx.compose.ui.geometry.Offset(0f, size.height), androidx.compose.ui.geometry.Offset(size.width, size.height), 1.dp.toPx())
                }
                .horizontalScroll(rememberScrollState()).padding(8.dp),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
        ) { tabs(Modifier) }
    } else {
        Column(
            Modifier.width(188.dp).fillMaxHeight().background(wrapBg)
                .drawBehind { drawLine(lineColor, androidx.compose.ui.geometry.Offset(size.width, 0f), androidx.compose.ui.geometry.Offset(size.width, size.height), 1.dp.toPx()) }
                .verticalScroll(rememberScrollState()).padding(10.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp),
        ) { tabs(Modifier.fillMaxWidth()) }
    }
}

@Composable
private fun SettingsTabButton(tab: SettingsTab, active: Boolean, hits: Int, phone: Boolean, modifier: Modifier, onClick: () -> Unit) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(12.dp)
    val fg = when {
        !active -> web.muted
        web.isLight -> web.theme.t600
        else -> web.theme200
    }
    Row(
        modifier.clip(shape)
            .background(if (active) web.theme.rgb(0.14f) else Color.Transparent)
            .border(1.dp, if (active) web.theme.rgb(0.28f) else Color.Transparent, shape)
            .clickable(role = Role.Tab, onClick = onClick)
            .padding(horizontal = 11.dp, vertical = if (phone) 7.dp else 9.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Box(Modifier.width(16.dp), contentAlignment = Alignment.Center) {
            tab.icon?.let { FaIcon(it, null, size = 12.dp, tint = if (active) web.theme300 else fg.copy(alpha = 0.85f)) }
        }
        Text(tab.label, fontSize = if (phone) 12.sp else 13.sp, fontWeight = FontWeight.SemiBold, letterSpacing = 0.12.sp, color = fg, maxLines = 1)
        if (hits > 0) Text(
            hits.toString(), fontSize = 9.sp, fontWeight = FontWeight.Bold, color = web.theme200,
            modifier = Modifier.clip(RoundedCornerShape(999.dp)).background(web.theme.rgb(0.2f)).padding(horizontal = 6.dp, vertical = 1.dp),
        )
    }
}

internal fun SettingsCardSpec.matches(query: String): Boolean =
    query.isNotEmpty() && ((title ?: "") + " " + search).contains(query, ignoreCase = true)

/** Web `getSectionSnippet`: 25 characters before and 35 after the match. */
internal fun settingsSnippet(text: String, query: String): String {
    val index = text.lowercase().indexOf(query.lowercase())
    if (index < 0) return ""
    val start = (index - 25).coerceAtLeast(0)
    val end = (index + query.length + 35).coerceAtMost(text.length)
    var snippet = text.substring(start, end).replace(Regex("\\s+"), " ").trim()
    if (start > 0) snippet = "…$snippet"
    if (end < text.length) snippet = "$snippet…"
    return snippet
}

/** The active tab's cards, or the Web search overlay while a query is typed. */
@Composable
private fun SettingsContent(
    cards: List<SettingsCardSpec>,
    tab: SettingsTab,
    query: String,
    phone: Boolean,
    modifier: Modifier,
    jumpTarget: String?,
    onJumpDone: () -> Unit,
    onJump: (SettingsCardSpec) -> Unit,
) {
    val web = LocalWebPalette.current
    val reduce = LocalReduceMotion.current
    Box(
        modifier.fillMaxWidth().drawBehind {
            // `radial-gradient(520px 180px at 100% 0%, rgba(theme,.06), transparent 70%)`
            drawRect(Brush.radialGradient(listOf(web.theme.rgb(0.06f), Color.Transparent),
                center = androidx.compose.ui.geometry.Offset(size.width, 0f), radius = 364.dp.toPx()))
        },
    ) {
        if (query.isNotEmpty()) {
            SearchResults(cards, tab, query, phone, onJump)
            return@Box
        }
        AnimatedContent(
            targetState = tab,
            transitionSpec = {
                val forward = if (targetState.ordinal >= initialState.ordinal) 1 else -1
                if (reduce) EnterTransition.None togetherWith ExitTransition.None
                else (slideInHorizontally(tween(PlaygroundMotion.MEDIUM, easing = PlaygroundMotion.Emphasized)) { w -> forward * w / 6 } +
                    fadeIn(tween(PlaygroundMotion.MEDIUM))) togetherWith
                    (slideOutHorizontally(tween(PlaygroundMotion.SHORT, easing = PlaygroundMotion.Exit)) { w -> -forward * w / 6 } +
                        fadeOut(tween(PlaygroundMotion.SHORT)))
            },
            label = "settings tab",
        ) { target ->
            Column(
                Modifier.fillMaxSize().verticalScroll(rememberScrollState())
                    .padding(start = 12.dp, end = 12.dp, top = 12.dp, bottom = 14.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                cards.filter { it.tab == target }.forEach { spec -> SettingsCardView(spec, jumpTarget == spec.key, onJumpDone) }
            }
        }
    }
}

@OptIn(androidx.compose.foundation.ExperimentalFoundationApi::class)
@Composable
private fun SettingsCardView(spec: SettingsCardSpec, highlight: Boolean, onJumpDone: () -> Unit) {
    val web = LocalWebPalette.current
    val requester = remember { BringIntoViewRequester() }
    val flash = remember { Animatable(0f) }
    LaunchedEffect(highlight) {
        if (highlight) {
            delay(260)
            requester.bringIntoView()
            flash.snapTo(1f)
            flash.animateTo(0f, tween(1800))
            onJumpDone()
        }
    }
    val titleContent: String? = spec.title
    WebSettingsCard(
        title = null,
        modifier = Modifier.bringIntoViewRequester(requester).drawBehind {
            if (flash.value > 0f) drawRoundRect(web.theme.rgb(0.45f * flash.value),
                topLeft = androidx.compose.ui.geometry.Offset(-3.dp.toPx(), -3.dp.toPx()),
                size = androidx.compose.ui.geometry.Size(size.width + 6.dp.toPx(), size.height + 6.dp.toPx()),
                cornerRadius = androidx.compose.ui.geometry.CornerRadius(19.dp.toPx()))
        },
        danger = spec.danger,
        compact = true,
    ) {
        if (titleContent != null) Row(Modifier.padding(bottom = 10.dp), verticalAlignment = Alignment.CenterVertically) {
            spec.titleIcon?.let { FaIcon(it, null, size = 12.dp, tint = spec.titleIconTint ?: if (spec.danger) Tw.rose300 else web.theme300, modifier = Modifier.padding(end = 8.dp)) }
            Text(titleContent, color = if (spec.danger) Tw.rose300 else web.theme300, fontSize = 13.sp, fontWeight = FontWeight.Bold, letterSpacing = 0.26.sp)
        }
        spec.content(this)
    }
}

@Composable
private fun SearchResults(cards: List<SettingsCardSpec>, current: SettingsTab, query: String, @Suppress("UNUSED_PARAMETER") phone: Boolean, onJump: (SettingsCardSpec) -> Unit) {
    val web = LocalWebPalette.current
    val results = cards.filter { it.matches(query) }
    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(start = 12.dp, end = 12.dp, top = 12.dp, bottom = 14.dp),
    ) {
        if (results.isEmpty()) {
            Column(Modifier.fillMaxWidth().padding(vertical = 48.dp, horizontal = 16.dp), horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Box(Modifier.size(56.dp).clip(RoundedCornerShape(16.dp)).background(web.theme.rgb(0.10f)), contentAlignment = Alignment.Center) {
                    FaIcon(R.drawable.fa_solid_search, null, size = 20.dp, tint = web.theme300)
                }
                Text("一致する設定はありません", fontSize = 14.sp, fontWeight = FontWeight.SemiBold, color = web.text)
                Text("「$query」に一致する設定項目はありません。", fontSize = 12.sp, lineHeight = 19.2.sp, color = web.muted)
            }
            return@Column
        }
        Text("${results.size}件の一致", fontSize = 11.sp, fontWeight = FontWeight.SemiBold, color = web.muted,
            modifier = Modifier.padding(start = 2.dp, end = 2.dp, bottom = 8.dp))
        // The Web lists the results of the current tab first when it has any.
        val target = if (results.any { it.tab == current }) current else results.first().tab
        val ordered = results.filter { it.tab == target } + results.filter { it.tab != target }
        var previous: SettingsTab? = null
        ordered.forEach { spec ->
            if (spec.tab != previous) {
                if (previous != null) Box(Modifier.fillMaxWidth().padding(vertical = 6.dp).height(1.dp).background(web.twBorder(Tw.gray700).copy(alpha = 0.5f)))
                if (spec.tab != target) Text("▼ ${spec.tab.label}", fontSize = 10.sp, fontWeight = FontWeight.Bold, color = Tw.gray500,
                    modifier = Modifier.padding(start = 4.dp, bottom = 4.dp))
                previous = spec.tab
            }
            val shape = RoundedCornerShape(8.dp)
            Row(
                Modifier.fillMaxWidth().padding(bottom = 4.dp).clip(shape)
                    .background(if (web.isLight) Color(15, 23, 42).copy(alpha = 0.03f) else Color(10, 16, 30).copy(alpha = 0.75f))
                    .border(1.dp, Color.White.copy(alpha = 0.04f), shape)
                    .clickable { onJump(spec) }
                    .padding(horizontal = 12.dp, vertical = 10.dp),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Text(spec.tab.label, fontSize = 10.sp, fontWeight = FontWeight.Bold, color = web.theme200, maxLines = 1,
                    modifier = Modifier.padding(top = 2.dp).clip(RoundedCornerShape(999.dp)).background(web.theme.rgb(0.14f))
                        .border(1.dp, web.theme.rgb(0.28f), RoundedCornerShape(999.dp)).padding(horizontal = 8.dp, vertical = 1.dp))
                Column(Modifier.weight(1f)) {
                    Text(spec.title ?: spec.search.substringBefore(' ').ifBlank { spec.tab.label }, fontSize = 14.sp, fontWeight = FontWeight.Bold,
                        color = if (web.isLight) web.text else Color.White, maxLines = 1, overflow = TextOverflow.Ellipsis)
                    Text(settingsSnippet(spec.search, query), fontSize = 11.sp, color = Tw.gray400, maxLines = 1, overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.padding(top = 2.dp))
                }
            }
        }
    }
}
