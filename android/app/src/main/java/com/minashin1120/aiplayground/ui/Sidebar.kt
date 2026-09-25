package com.minashin1120.aiplayground.ui

import androidx.annotation.DrawableRes
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Text
import androidx.compose.material3.pulltorefresh.PullToRefreshBox
import androidx.compose.material3.pulltorefresh.rememberPullToRefreshState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.Gem
import com.minashin1120.aiplayground.data.ThreadItem
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.filter

/**
 * Colors of the Web `#sidebar` and its children as computed from `chat.custom.v*.css` (dark) and
 * `theme-light-manual.css` (light). Light values keep Web quirks, e.g. every thread row receives the
 * `bg-gray-700` remap because its `hover:bg-gray-700` class matches the light-theme attribute selector.
 */
internal data class SidebarColors(
    val background: Color,
    val edge: Color,
    val rule: Color,
    val title: Color,
    val toolIcon: Color,
    val searchBackground: Color,
    val searchBorder: Color,
    val searchText: Color,
    val searchIcon: Color,
    val gemsLabel: Color,
    val addGem: Color,
    val gemBackground: Color,
    val gemText: Color,
    val gemAction: Color,
    val threadText: Color,
    val threadSelected: Color,
    val threadIdle: Color,
    val starOff: Color,
    val threadAction: Color,
    val footerShade: Color,
    val footerText: Color,
    val legal: Color,
    val logoutBackground: Color,
    val logoutText: Color,
    val amberText: Color,
    val amberFaint: Color,
    val lowBandwidthBackground: Color,
)

@Composable
internal fun sidebarColors(): SidebarColors {
    val web = LocalWebPalette.current
    val slate = Color(15, 23, 42)
    return if (web.isLight) SidebarColors(
        background = Color.White,
        edge = web.line,
        rule = slate.copy(alpha = 0.10f),
        title = Color(0xFF2B3546),
        toolIcon = Color(0xFF5B6675),
        searchBackground = Color.White,
        searchBorder = slate.copy(alpha = 0.10f),
        searchText = web.text,
        searchIcon = Color(0xFF697588),
        gemsLabel = web.muted,
        addGem = Color(0xFF1D4ED8),
        gemBackground = slate.copy(alpha = 0.035f),
        gemText = Color(0xFF3F4A5C),
        gemAction = Color(0xFF5B6675),
        threadText = Color(0xFF3F4A5C),
        threadSelected = Color(0xFFE7EDF6),
        threadIdle = Color(0xFFE7EDF6),
        starOff = Color(0xFF697588),
        threadAction = Color(0xFF697588),
        footerShade = slate.copy(alpha = 0.05f),
        footerText = Color(0xFF697588),
        legal = Color(0xFF64748B),
        logoutBackground = slate.copy(alpha = 0.05f),
        logoutText = Color(0xFF475569),
        amberText = Color(0xFF92400E),
        amberFaint = Color(0xFF92400E),
        lowBandwidthBackground = Color(245, 158, 11).copy(alpha = 0.14f),
    ) else SidebarColors(
        // rgba(7,10,20,.76) over a 24px backdrop blur; Android cannot blur behind the drawer, so it is denser.
        background = Color(7, 10, 20).copy(alpha = 0.93f),
        edge = web.lineSoft,
        rule = web.lineSoft,
        title = Tw.gray200,
        toolIcon = Tw.gray400,
        searchBackground = Color(6, 9, 18).copy(alpha = 0.78f),
        searchBorder = Color.White.copy(alpha = 0.07f),
        searchText = web.text,
        searchIcon = Tw.gray500,
        gemsLabel = web.muted,
        addGem = web.theme300,
        gemBackground = Color(10, 14, 26).copy(alpha = 0.55f),
        gemText = Tw.gray300,
        gemAction = Tw.gray400,
        threadText = Tw.gray300,
        threadSelected = Tw.gray700.copy(alpha = 0.6f),
        threadIdle = Color.Transparent,
        starOff = Tw.gray500,
        threadAction = Tw.gray500,
        footerShade = Color(6, 10, 22).copy(alpha = 0.35f),
        footerText = Tw.gray500,
        legal = Color(0xFF7C889E),
        logoutBackground = Color(10, 16, 30).copy(alpha = 0.78f),
        logoutText = Color(0xFFA8B3C7),
        amberText = Tw.amber200,
        amberFaint = Tw.amber200.copy(alpha = 0.5f),
        lowBandwidthBackground = Tw.amber900.copy(alpha = 0.2f),
    )
}

/** Callbacks of the sidebar controls; each mirrors the Web button of the same name. */
internal class SidebarActions(
    val onNavigate: () -> Unit,
    val onChangelog: () -> Unit,
    val onHistory: () -> Unit,
    val onLowBandwidth: () -> Unit,
    val onSettings: () -> Unit,
    val onLibrary: () -> Unit,
    val onNewChat: () -> Unit,
    val onBatch: () -> Unit,
    val onPdf: () -> Unit,
    val onExternal: (String) -> Unit,
    val onSearch: (String) -> Unit,
    val onOpenThread: (ThreadItem) -> Unit,
    val onBookmark: (ThreadItem) -> Unit,
    val onRenameThread: (ThreadItem) -> Unit,
    val onDeleteThread: (ThreadItem) -> Unit,
    val onMoreThreads: () -> Unit,
    val onRefreshThreads: (() -> Unit) -> Unit,
    val onChooseGem: (Gem) -> Unit,
    val onNewGem: () -> Unit,
    val onEditGem: (Gem) -> Unit,
    val onDeleteGem: (Gem) -> Unit,
    val onRefreshGems: (() -> Unit) -> Unit,
    val onHelp: () -> Unit,
    val onLegal: (String) -> Unit,
    val onAlphaInfo: () -> Unit,
    val onLogout: () -> Unit,
)

internal const val WEB_GITHUB_URL = "https://github.com/Minashin1120/ai-playground"

/** The Web `#sidebar` (templates/chat/sidebar.html): header with toolbar and search, Gems, threads, footer. */
@Composable
internal fun Sidebar(state: ChatState, versionName: String, actions: SidebarActions, modifier: Modifier = Modifier) {
    val colors = sidebarColors()
    Row(modifier.fillMaxHeight().width(PlaygroundDimens.drawerPane)) {
        Column(
            Modifier.weight(1f).fillMaxHeight().background(colors.background),
        ) {
            SidebarHeader(state, colors, actions)
            SidebarGems(state, colors, actions)
            Box(Modifier.fillMaxWidth().height(1.dp).graphicsLayer { alpha = 0.85f }.background(colors.rule))
            ThreadList(
                state = state,
                colors = colors,
                actions = actions,
                modifier = Modifier.weight(1f),
                contentPadding = PaddingValues(8.dp),
            )
            SidebarFooter(versionName, colors, actions)
        }
        // `border-r` of the sidebar.
        Box(Modifier.width(1.dp).fillMaxHeight().background(colors.edge))
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun SidebarHeader(state: ChatState, colors: SidebarColors, actions: SidebarActions) {
    val temporary = state.selected?.isTemporary == true || state.newThreadTemporary
    Column(
        Modifier
            .fillMaxWidth()
            .background(Brush.verticalGradient(listOf(Color(12, 20, 40).copy(alpha = 0.35f), Color.Transparent)))
            .statusBarsPadding()
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Column {
            if (temporary) {
                Text("一時チャット", color = colors.amberFaint, fontSize = 10.sp, lineHeight = 10.sp,
                    letterSpacing = 0.025.em, modifier = Modifier.padding(bottom = 2.dp))
            }
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(
                    state.selected?.title?.ifBlank { "No Title" } ?: "AI Chat",
                    color = colors.title, fontSize = 16.sp, lineHeight = 24.sp, fontWeight = FontWeight.Bold,
                    letterSpacing = (-0.01).em, maxLines = 1, overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.weight(1f, fill = false),
                )
                state.tempChatRemainingSeconds?.takeIf { temporary }?.let { seconds ->
                    Text(
                        "${seconds}秒", color = colors.amberText, fontSize = 10.sp, lineHeight = 10.sp,
                        modifier = Modifier.border(1.dp, Tw.amber500.copy(alpha = 0.4f), CircleShape)
                            .padding(horizontal = 6.dp, vertical = 2.dp),
                    )
                }
            }
        }
        FlowRow(horizontalArrangement = Arrangement.spacedBy(2.dp), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            SidebarIconButton(R.drawable.fa_brands_github, "GitHub", colors, iconSize = 14.dp) { actions.onExternal(WEB_GITHUB_URL) }
            SidebarIconButton(R.drawable.fa_solid_history, "更新履歴", colors) { actions.onChangelog(); actions.onNavigate() }
            SidebarIconButton(R.drawable.fa_solid_clock_rotate_left, "チャット履歴", colors) { actions.onHistory() }
            SidebarIconButton(
                R.drawable.fa_solid_tachometer_alt, "低速回線モード", colors,
                tint = if (state.lowBandwidthMode) colors.amberText else colors.toolIcon,
            ) { actions.onLowBandwidth() }
            SidebarIconButton(R.drawable.fa_solid_cog, "設定", colors) { actions.onSettings(); actions.onNavigate() }
            SidebarIconButton(R.drawable.fa_solid_folder, "ライブラリ", colors) { actions.onLibrary(); actions.onNavigate() }
            NewChatToolButton(enabled = true) { actions.onNewChat(); actions.onNavigate() }
            SidebarIconButton(R.drawable.fa_solid_layer_group, "Batch処理", colors) { actions.onBatch(); actions.onNavigate() }
            SidebarIconButton(R.drawable.fa_solid_file_pdf, "PDF出力", colors) { actions.onPdf() }
        }
        SidebarSearch(state.search, "チャットを検索...", colors, actions.onSearch)
        AnimatedVisibility(state.lowBandwidthMode) {
            Row(
                Modifier
                    .fillMaxWidth()
                    .clip(CircleShape)
                    .background(colors.lowBandwidthBackground)
                    .border(1.dp, Tw.amber600.copy(alpha = 0.4f), CircleShape)
                    .padding(horizontal = 10.dp, vertical = 4.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                FaIcon(R.drawable.fa_solid_wifi, null, size = 10.dp, tint = colors.amberText)
                Spacer(Modifier.width(4.dp))
                Text(
                    lowBandwidthPillText(state.lowBandwidthPreference, state.lowBandwidthReason),
                    color = colors.amberText, fontSize = 10.sp, lineHeight = 20.sp,
                )
            }
        }
    }
    Box(Modifier.fillMaxWidth().height(1.dp).background(colors.rule))
}

/** `.sidebar-icon-btn`: 27px square, 11px corners, faint vertical sheen, 12px icon. */
@Composable
private fun SidebarIconButton(
    @DrawableRes icon: Int,
    label: String,
    colors: SidebarColors,
    iconSize: Dp = 12.dp,
    tint: Color = colors.toolIcon,
    enabled: Boolean = true,
    onClick: () -> Unit,
) {
    val shape = RoundedCornerShape(11.dp)
    Box(
        Modifier
            .size(27.dp)
            .clip(shape)
            .background(Brush.verticalGradient(listOf(Color.White.copy(alpha = 0.04f), Color.Black.copy(alpha = 0.06f))))
            .clickable(enabled = enabled, onClickLabel = label, role = Role.Button, onClick = onClick)
            .graphicsLayer { alpha = if (enabled) 1f else 0.5f },
        contentAlignment = Alignment.Center,
    ) { FaIcon(icon, label, size = iconSize, tint = tint) }
}

/** `#new-chat-btn`: the theme-gradient square with a white plus. */
@Composable
private fun NewChatToolButton(enabled: Boolean, onClick: () -> Unit) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(10.dp)
    Box(
        Modifier
            .size(28.8.dp)
            .clip(shape)
            .background(Brush.linearGradient(listOf(web.theme.t500, web.theme.t600)))
            .border(1.dp, web.theme.rgb(0.4f), shape)
            .clickable(enabled = enabled, onClickLabel = "新規チャット", role = Role.Button, onClick = onClick)
            .graphicsLayer { alpha = if (enabled) 1f else 0.5f },
        contentAlignment = Alignment.Center,
    ) { FaIcon(R.drawable.fa_solid_plus, "新規チャット", size = 12.dp, tint = Color.White) }
}

/** `#search-box`: 32px pill-rounded input with the search glyph on the right. */
@Composable
internal fun SidebarSearch(value: String, placeholder: String, colors: SidebarColors, onValueChange: (String) -> Unit) {
    val shape = RoundedCornerShape(12.dp)
    Box(
        Modifier
            .fillMaxWidth()
            .height(32.dp)
            .clip(shape)
            .background(colors.searchBackground)
            .border(1.dp, colors.searchBorder, shape)
            .padding(start = 12.dp, end = 12.dp),
        contentAlignment = Alignment.CenterStart,
    ) {
        BasicTextField(
            value, onValueChange, singleLine = true,
            textStyle = TextStyle(color = colors.searchText, fontSize = 13.sp, lineHeight = 20.sp, fontFamily = WebFonts.sans),
            cursorBrush = SolidColor(colors.searchText),
            modifier = Modifier.fillMaxWidth().padding(end = 20.dp),
            decorationBox = { inner ->
                if (value.isEmpty()) Text(placeholder, color = Tw.gray400, fontSize = 13.sp, lineHeight = 20.sp, maxLines = 1)
                inner()
            },
        )
        FaIcon(R.drawable.fa_solid_search, null, size = 12.dp, tint = colors.searchIcon, modifier = Modifier.align(Alignment.CenterEnd))
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SidebarGems(state: ChatState, colors: SidebarColors, actions: SidebarActions) {
    Column(Modifier.fillMaxWidth().padding(vertical = 8.dp)) {
        Row(
            Modifier.fillMaxWidth().padding(horizontal = 6.dp).padding(bottom = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("GEMS", color = colors.gemsLabel, fontSize = 10.sp, lineHeight = 16.sp, fontWeight = FontWeight.Bold,
                letterSpacing = 0.14.em, modifier = Modifier.weight(1f))
            Row(
                Modifier
                    .border(1.dp, Color.White.copy(alpha = 0.06f))
                    .background(Brush.verticalGradient(listOf(Color.White.copy(alpha = 0.04f), Color.Black.copy(alpha = 0.06f))))
                    .clickable(onClickLabel = "Gemを作成", role = Role.Button) { actions.onNewGem() }
                    .padding(horizontal = 2.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                FaIcon(R.drawable.fa_solid_plus, null, size = 12.dp, tint = colors.addGem)
                Text(" New", color = colors.addGem, fontSize = 12.sp, lineHeight = 16.sp)
            }
        }
        PullToRefreshArea(onRefresh = actions.onRefreshGems, modifier = Modifier.heightIn(max = 160.dp)) {
            LazyColumn(Modifier.fillMaxWidth()) {
                items(state.gems, key = { it.uuid }) { gem -> GemRow(gem, colors, actions) }
            }
        }
    }
}

/** `.gem-item`: gem glyph, name, then edit / delete glyphs (always visible on touch screens). */
@Composable
private fun GemRow(gem: Gem, colors: SidebarColors, actions: SidebarActions) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(12.dp)
    Row(
        Modifier
            .fillMaxWidth()
            .padding(top = 4.dp)
            .clip(shape)
            .background(colors.gemBackground)
            .border(1.dp, colors.rule, shape)
            .clickable(interactionSource = remember { MutableInteractionSource() }, indication = null) {
                actions.onChooseGem(gem); actions.onNavigate()
            }
            .padding(horizontal = 8.dp, vertical = 5.6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        FaIcon(R.drawable.fa_solid_gem, null, size = 14.dp, tint = web.theme300)
        Spacer(Modifier.width(8.dp))
        Text(gem.name, color = colors.gemText, fontSize = 14.sp, lineHeight = 20.sp, maxLines = 1,
            overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f))
        RowGlyphButton(R.drawable.fa_solid_pencil_alt, "Gemを編集", 10.dp, colors.gemAction, horizontal = 8.dp) {
            actions.onEditGem(gem); actions.onNavigate()
        }
        Spacer(Modifier.width(4.dp))
        RowGlyphButton(R.drawable.fa_solid_trash, "Gemを削除", 10.dp, colors.gemAction, horizontal = 8.dp) { actions.onDeleteGem(gem) }
    }
}

/**
 * `#thread-list`: star, title, "一時" badge and the edit / delete glyphs. Shared by the sidebar and the
 * history modal, where Web moves the same list element.
 */
@Composable
internal fun ThreadList(
    state: ChatState,
    colors: SidebarColors,
    actions: SidebarActions,
    modifier: Modifier = Modifier,
    contentPadding: PaddingValues = PaddingValues(8.dp),
) {
    val listState = rememberLazyListState()
    // Web loads the next page when `#scroll-sentinel` scrolls into view.
    LaunchedEffect(listState, state.nextPage) {
        snapshotFlow { listState.layoutInfo.let { info -> (info.visibleItemsInfo.lastOrNull()?.index ?: -1) >= info.totalItemsCount - 2 } }
            .distinctUntilChanged()
            .filter { it && state.nextPage != null }
            .collect { actions.onMoreThreads() }
    }
    PullToRefreshArea(onRefresh = actions.onRefreshThreads, modifier = modifier) {
        LazyColumn(
            Modifier.fillMaxSize(), state = listState, contentPadding = contentPadding,
            verticalArrangement = Arrangement.spacedBy(2.dp),
        ) {
            items(state.threads, key = { it.id }) { thread ->
                ThreadRow(thread, selected = state.selected?.id == thread.id, offline = state.offline, colors = colors, actions = actions)
            }
        }
    }
}

@Composable
private fun ThreadRow(thread: ThreadItem, selected: Boolean, offline: Boolean, colors: SidebarColors, actions: SidebarActions) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(12.dp)
    Row(
        Modifier
            .fillMaxWidth()
            .heightIn(min = 34.8.dp)
            .clip(shape)
            .background(if (selected) colors.threadSelected else colors.threadIdle)
            .border(1.dp, if (selected) web.theme.t500 else Color.Transparent, shape)
            .clickable(interactionSource = remember { MutableInteractionSource() }, indication = null) {
                actions.onOpenThread(thread); actions.onNavigate()
            }
            .padding(horizontal = 8.dp, vertical = 6.4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Row(Modifier.weight(1f), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
            RowGlyphButton(
                R.drawable.fa_solid_star, if (thread.isBookmarked) "ブックマーク解除" else "ブックマーク", 10.dp,
                if (thread.isBookmarked) Tw.yellow400 else colors.starOff, horizontal = 4.dp, enabled = !offline,
            ) { actions.onBookmark(thread) }
            Text(
                thread.title.ifBlank { "No Title" }, color = colors.threadText, fontSize = 14.sp, lineHeight = 20.sp,
                maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f, fill = false),
            )
            if (thread.isTemporary) {
                Text(
                    "一時", color = colors.amberText.takeIf { web.isLight } ?: Tw.amber300, fontSize = 9.sp, lineHeight = 20.sp,
                    modifier = Modifier.border(1.dp, Tw.amber500.copy(alpha = 0.5f), RoundedCornerShape(4.dp)).padding(horizontal = 4.dp),
                )
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(4.dp), verticalAlignment = Alignment.CenterVertically) {
            RowGlyphButton(R.drawable.fa_solid_pen, "名前を変更", 12.dp, colors.threadAction, horizontal = 4.dp, enabled = !offline) {
                actions.onRenameThread(thread)
            }
            RowGlyphButton(R.drawable.fa_solid_trash, "削除", 12.dp, colors.threadAction, horizontal = 4.dp, enabled = !offline) {
                actions.onDeleteThread(thread)
            }
        }
    }
}

/** A bare `<button class="px-…"><i class="fas …">` used inside list rows. */
@Composable
private fun RowGlyphButton(
    @DrawableRes icon: Int,
    label: String,
    size: Dp,
    tint: Color,
    horizontal: Dp,
    enabled: Boolean = true,
    onClick: () -> Unit,
) {
    Box(
        Modifier
            .clickable(enabled = enabled, onClickLabel = label, role = Role.Button, onClick = onClick)
            .padding(horizontal = horizontal, vertical = 4.dp)
            .graphicsLayer { alpha = if (enabled) 1f else 0.5f },
        contentAlignment = Alignment.Center,
    ) { FaIcon(icon, label, size = size, tint = tint) }
}

/**
 * Web pull-to-refresh (`.ptr-pull-indicator`): the list is pulled down, an arrow and
 * 「引っ張って更新」/「離して更新」 appear in the theme color, then a spinner while refreshing.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun PullToRefreshArea(onRefresh: (() -> Unit) -> Unit, modifier: Modifier = Modifier, content: @Composable BoxScope.() -> Unit) {
    val web = LocalWebPalette.current
    var refreshing by remember { mutableStateOf(false) }
    val pull = rememberPullToRefreshState()
    PullToRefreshBox(
        isRefreshing = refreshing,
        onRefresh = { refreshing = true; onRefresh { refreshing = false } },
        modifier = modifier,
        state = pull,
        indicator = {
            val fraction = if (refreshing) 1f else pull.distanceFraction.coerceIn(0f, 1.5f)
            if (fraction > 0f) {
                Row(
                    Modifier.align(Alignment.TopCenter).fillMaxWidth().height(40.dp * fraction.coerceAtMost(1f)),
                    horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterHorizontally),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    val color = web.theme.rgb(0.95f)
                    if (refreshing) {
                        CircularProgressIndicator(Modifier.size(12.dp), color = color, strokeWidth = 1.5.dp)
                    } else {
                        val ready = pull.distanceFraction >= 1f
                        val rotation by animateFloatAsState(if (ready) 180f else 0f, motionTween(LocalReduceMotion.current, 200), label = "ptr arrow")
                        FaIcon(R.drawable.fa_solid_arrow_down, null, size = 12.dp, tint = color, modifier = Modifier.rotate(rotation))
                        Text(if (ready) "離して更新" else "引っ張って更新", color = color, fontSize = 12.sp,
                            fontWeight = FontWeight.SemiBold, letterSpacing = 0.02.em)
                    }
                }
            }
        },
    ) {
        Box(Modifier.fillMaxWidth().graphicsLayer {
            translationY = (if (refreshing) 1f else pull.distanceFraction.coerceIn(0f, 1f)) * 40.dp.toPx()
        }) { content() }
    }
}

/** `.sidebar-footer`: version (tap for the alpha notice), legal links and the logout button. */
@Composable
private fun SidebarFooter(versionName: String, colors: SidebarColors, actions: SidebarActions) {
    val web = LocalWebPalette.current
    Column(
        Modifier
            .fillMaxWidth()
            .background(Brush.verticalGradient(listOf(Color.Transparent, colors.footerShade)))
            .padding(top = 0.dp),
    ) {
        Box(Modifier.fillMaxWidth().height(1.dp).background(colors.rule))
        Column(
            Modifier.fillMaxWidth().padding(12.dp).navigationBarsPadding(),
            verticalArrangement = Arrangement.spacedBy(8.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(
                buildAnnotatedString {
                    append("$versionName ")
                    withStyle(SpanStyle(color = web.theme300, fontWeight = FontWeight.Bold)) { append("Stable") }
                },
                color = colors.footerText, fontSize = 12.sp, lineHeight = 16.sp, textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth().clickable(
                    interactionSource = remember { MutableInteractionSource() }, indication = null,
                    onClickLabel = "アルファ版に関するご注意",
                ) { actions.onAlphaInfo() },
            )
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
                LegalLink("ヘルプ", colors) { actions.onHelp(); actions.onNavigate() }
                Text("·", color = colors.footerText, fontSize = 12.sp, lineHeight = 16.sp)
                LegalLink("利用規約", colors) { actions.onLegal("terms") }
                Text("·", color = colors.footerText, fontSize = 12.sp, lineHeight = 16.sp)
                LegalLink("プライバシー", colors) { actions.onLegal("privacy") }
            }
            val shape = RoundedCornerShape(8.dp)
            Box(
                Modifier
                    .fillMaxWidth()
                    .height(34.dp)
                    .clip(shape)
                    .background(colors.logoutBackground)
                    .border(1.dp, colors.rule, shape)
                    .clickable(role = Role.Button, onClick = actions.onLogout),
                contentAlignment = Alignment.Center,
            ) {
                Text("ログアウト", color = colors.logoutText, fontSize = 12.sp, lineHeight = 16.sp,
                    fontWeight = FontWeight.SemiBold, letterSpacing = 0.02.em)
            }
        }
    }
}

@Composable
private fun LegalLink(label: String, colors: SidebarColors, onClick: () -> Unit) {
    Text(
        label, color = colors.legal, fontSize = 10.5.sp, lineHeight = 16.sp,
        modifier = Modifier.clickable(role = Role.Button, onClick = onClick),
    )
}

/** Web `updateLowBandwidthModeUi()` pill text. */
internal fun lowBandwidthPillText(preference: String, reason: String): String {
    val badge = if (preference == "auto") " (自動)" else " (手動)"
    return "低速回線モード$badge" + if (reason.isNotBlank()) ": $reason" else ""
}
