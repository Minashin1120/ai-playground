package com.minashin1120.aiplayground.ui

import android.content.Context
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.BRANCH_GAP
import com.minashin1120.aiplayground.data.BRANCH_NODE_HEIGHT
import com.minashin1120.aiplayground.data.BranchNode
import com.minashin1120.aiplayground.data.ChatMessage
import com.minashin1120.aiplayground.data.branchModelBreakdown
import com.minashin1120.aiplayground.data.buildTokenTotals
import com.minashin1120.aiplayground.data.layoutBranchTree
import org.json.JSONObject

/** Web `branch_names_<thread>` / `fixed_branch_<thread>` (localStorage), kept on the device. */
private class BranchStore(context: Context, private val threadId: String) {
    private val prefs = context.getSharedPreferences("branches", Context.MODE_PRIVATE)
    val names = mutableStateMapOf<String, String>().apply {
        runCatching { JSONObject(prefs.getString("branch_names_$threadId", null) ?: "{}") }.getOrNull()?.let { json ->
            json.keys().forEach { key -> put(key, json.optString(key)) }
        }
    }
    var fixed by mutableStateOf(prefs.getString("fixed_branch_$threadId", null))

    fun save() {
        prefs.edit()
            .putString("branch_names_$threadId", JSONObject(names.toMap()).toString())
            .apply { if (fixed != null) putString("fixed_branch_$threadId", fixed) else remove("fixed_branch_$threadId") }
            .apply()
    }
}

/** Web card width (`min-w-[120px] max-w-[180px]`, sized by its text). */
private fun branchCardWidth(name: String, model: String, tokens: Int): Float {
    val nameWidth = name.length * 6.4f
    val metaWidth = model.ifBlank { "-" }.length * 5.2f + 8f + tokens.toString().length * 5.6f
    return (24f + maxOf(nameWidth, metaWidth) + 2f).coerceIn(120f, 180f)
}

private fun branchName(names: Map<String, String>, message: ChatMessage) = names[message.id] ?: if (message.role == "user") "User" else "AI"

/** `#branch-modal` (ブランチ管理): the message tree, the selected node's details and actions, legend and totals. */
@Composable
internal fun BranchManagerDialog(
    state: ChatState,
    onDismiss: () -> Unit,
    onSwitch: (Int) -> Unit,
    onDelete: (ChatMessage) -> Unit,
    notify: (String) -> Unit,
) {
    val web = LocalWebPalette.current
    val context = LocalContext.current
    val thread = state.selected ?: return
    val store = remember(thread.id) { BranchStore(context, thread.id) }
    var selectedId by remember(thread.id) { mutableStateOf<String?>(null) }
    var confirmBranchDelete by remember { mutableStateOf(false) }
    var confirmMessageDelete by remember { mutableStateOf(false) }
    val all = state.allMessages
    val total = remember(all) { buildTokenTotals(all).total }
    WebOverlayModal(onDismiss, grayOverlay(), 4.dp) { phone ->
        BoxWithConstraints(Modifier.fillMaxSize()) {
            val panelHeight = if (phone) Modifier else Modifier.heightIn(max = maxHeight * 0.9f)
            Column(
                Modifier.fillMaxSize().then(if (phone) Modifier.verticalScroll(rememberScrollState()) else Modifier),
                verticalArrangement = if (phone) Arrangement.Top else Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                val shape = RoundedCornerShape(8.dp)
                Column(
                    Modifier.padding(16.dp).widthIn(max = 896.dp).fillMaxWidth().then(panelHeight).clip(shape)
                        .background(web.twBg(Tw.gray800)).border(1.dp, web.twBorder(Tw.gray700), shape).padding(24.dp),
                ) {
                    BranchHeader(onDismiss)
                    val selected = all.firstOrNull { it.id == selectedId }
                    val tree: @Composable (Modifier) -> Unit = { modifier ->
                        BranchTree(state, store, selectedId, phone, modifier) { selectedId = it.id }
                    }
                    val side: @Composable (Modifier) -> Unit = { modifier ->
                        if (selected == null) BranchEmptyPanel(modifier)
                        else BranchDetail(
                            state, selected, store, modifier,
                            onSaveName = { name ->
                                if (name.isNotEmpty()) store.names[selected.id] = name else store.names.remove(selected.id)
                                store.save()
                                notify("名前を保存しました")
                            },
                            onSwitch = {
                                selected.id.toIntOrNull()?.let(onSwitch)
                                onDismiss()
                                notify("ブランチを切り替えました")
                            },
                            onFix = {
                                if (store.fixed == selected.id) { store.fixed = null; notify("固定を解除しました") }
                                else { store.fixed = selected.id; notify("メインルートに固定しました") }
                                store.save()
                            },
                            onDelete = { confirmBranchDelete = true },
                        )
                    }
                    if (phone) {
                        Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                            tree(Modifier.fillMaxWidth())
                            side(Modifier.fillMaxWidth())
                        }
                    } else {
                        Row(Modifier.weight(1f, fill = false), horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                            tree(Modifier.weight(1f).fillMaxHeight())
                            side(Modifier.width(256.dp))
                        }
                    }
                    BranchFooter(total)
                }
            }
        }
    }
    if (confirmBranchDelete) BrowserConfirmDialog("このブランチを削除してもよろしいですか？（その後の全てのメッセージも削除されます）") { ok ->
        confirmBranchDelete = false
        if (ok) confirmMessageDelete = true
    }
    // Web `deleteMessage` asks again with its own confirm().
    if (confirmMessageDelete) BrowserConfirmDialog("Delete this message and subsequent history?") { ok ->
        confirmMessageDelete = false
        val target = all.firstOrNull { it.id == selectedId }
        if (ok && target != null) { onDelete(target); selectedId = null }
    }
}

@Composable
private fun BranchHeader(onDismiss: () -> Unit) {
    val web = LocalWebPalette.current
    val rule = web.twBorder(Tw.gray700)
    Row(
        Modifier.fillMaxWidth().padding(bottom = 12.dp)
            .drawBehind { drawLine(rule, Offset(0f, size.height), Offset(size.width, size.height), 1.dp.toPx()) }
            .padding(bottom = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        FaIcon(R.drawable.fa_solid_sitemap, null, size = 18.dp, tint = web.twText(Tw.purple400), modifier = Modifier.padding(end = 8.dp))
        Text("ブランチ管理", fontSize = 18.sp, lineHeight = 28.sp, fontWeight = FontWeight.Bold, color = web.twText(Tw.white), modifier = Modifier.weight(1f))
        Box(
            Modifier.size(32.dp).clip(CircleShape).clickable(role = Role.Button, onClick = onDismiss),
            contentAlignment = Alignment.Center,
        ) { FaIcon(R.drawable.fa_solid_times, "閉じる", size = 16.dp, tint = web.twText(Tw.gray400)) }
    }
}

/** `#branch-tree-container`: the tree, scrollable in both directions (at least 400px tall). */
@Composable
private fun BranchTree(
    state: ChatState,
    store: BranchStore,
    selectedId: String?,
    phone: Boolean,
    modifier: Modifier,
    onSelect: (ChatMessage) -> Unit,
) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(8.dp)
    val all = state.allMessages
    BoxWithConstraints(
        modifier.heightIn(min = 400.dp).clip(shape).background(web.twBg(Tw.gray900)).border(1.dp, web.twBorder(Tw.gray700), shape),
    ) {
        val inner = (maxWidth - 32.dp).value.coerceAtLeast(0f)
        val names = store.names.toMap()
        val layout = remember(all, names, inner) {
            layoutBranchTree(all, inner) { message, tokens -> branchCardWidth(branchName(names, message), message.model, tokens) }
        }
        val connector = web.twBorder(Tw.gray700)
        Box(
            Modifier.then(if (phone) Modifier else Modifier.verticalScroll(rememberScrollState()))
                .horizontalScroll(rememberScrollState()).padding(16.dp),
        ) {
            Box(
                Modifier.size(layout.width.dp, layout.height.dp).drawBehind {
                    layout.nodes.forEach { node ->
                        if (node.hasChildren) {
                            val x = (node.x + node.width / 2f).dp.toPx()
                            val top = (node.y + BRANCH_NODE_HEIGHT).dp.toPx()
                            drawLine(connector, Offset(x, top), Offset(x, top + BRANCH_GAP.dp.toPx()), 1.dp.toPx())
                        }
                    }
                },
            ) {
                layout.nodes.forEach { node ->
                    key(node.id) {
                        BranchNodeCard(node, branchName(names, node.message), node.message.id == selectedId,
                            current = node.id == state.leafId, fixed = store.fixed == node.message.id) { onSelect(node.message) }
                    }
                }
            }
        }
    }
}

@Composable
private fun BranchNodeCard(node: BranchNode, name: String, selected: Boolean, current: Boolean, fixed: Boolean, onClick: () -> Unit) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(8.dp)
    val border = when {
        selected -> web.twBorder(Tw.purple400)
        current -> web.twBorder(Tw.blue500).copy(alpha = 0.5f)
        else -> web.twBorder(Tw.gray700)
    }
    val ring = web.twBorder(Tw.purple500)
    Box(Modifier.offset(node.x.dp, node.y.dp).size(node.width.dp, BRANCH_NODE_HEIGHT.dp)) {
        Column(
            Modifier.fillMaxSize()
                .drawBehind {
                    if (selected) {
                        val spread = 2.dp.toPx()
                        drawRoundRect(ring, topLeft = Offset(-spread, -spread),
                            size = androidx.compose.ui.geometry.Size(size.width + spread * 2, size.height + spread * 2),
                            cornerRadius = androidx.compose.ui.geometry.CornerRadius(10.dp.toPx(), 10.dp.toPx()))
                    }
                }
                .clip(shape)
                .background(if (current) web.twBg(Tw.blue900).copy(alpha = 0.4f) else web.twBg(Tw.gray800))
                .border(1.dp, border, shape)
                .clickable(role = Role.Button, onClick = onClick)
                .padding(horizontal = 12.dp, vertical = 8.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(name, fontSize = 10.sp, lineHeight = 15.sp, fontWeight = FontWeight.Bold, color = web.twText(Tw.white),
                maxLines = 1, overflow = TextOverflow.Ellipsis, textAlign = TextAlign.Center)
            Row(Modifier.fillMaxWidth().padding(top = 4.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(node.message.model.ifBlank { "-" }, fontSize = 9.sp, lineHeight = 13.5.sp, color = web.twText(Tw.gray500),
                    maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f))
                Text(node.pathTokens.toString(), fontSize = 9.sp, lineHeight = 13.5.sp, fontFamily = FontFamily.Monospace,
                    fontWeight = FontWeight.Bold, color = web.twText(Tw.blue400))
            }
        }
        val dotBorder = web.twBorder(Tw.gray900)
        if (fixed) Box(Modifier.align(Alignment.TopEnd).offset(4.dp, (-4).dp).size(12.dp).clip(CircleShape)
            .background(Tw.amber500).border(1.dp, dotBorder, CircleShape))
        if (current) Box(Modifier.align(Alignment.TopStart).offset((-4).dp, (-4).dp).size(12.dp).clip(CircleShape)
            .background(Tw.blue500).border(1.dp, dotBorder, CircleShape))
    }
}

@Composable
private fun BranchEmptyPanel(modifier: Modifier) {
    val web = LocalWebPalette.current
    val border = web.twBorder(Tw.gray700)
    Column(
        modifier.heightIn(min = 120.dp).clip(RoundedCornerShape(8.dp)).background(web.twBg(Tw.gray900))
            .drawBehind {
                drawRoundRect(border, style = androidx.compose.ui.graphics.drawscope.Stroke(1.dp.toPx(),
                    pathEffect = PathEffect.dashPathEffect(floatArrayOf(4.dp.toPx(), 3.dp.toPx()))),
                    cornerRadius = androidx.compose.ui.geometry.CornerRadius(8.dp.toPx(), 8.dp.toPx()))
            }
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        FaIcon(R.drawable.fa_solid_mouse_pointer, null, size = 18.dp, tint = web.twText(Tw.gray500), modifier = Modifier.padding(bottom = 8.dp))
        Text("ツリー上のノードを選択して\n詳細を表示・管理します", fontSize = 12.sp, lineHeight = 16.sp, color = web.twText(Tw.gray500), textAlign = TextAlign.Center)
    }
}

/** `#branch-detail-panel`. */
@Composable
private fun BranchDetail(
    state: ChatState,
    node: ChatMessage,
    store: BranchStore,
    modifier: Modifier,
    onSaveName: (String) -> Unit,
    onSwitch: () -> Unit,
    onFix: () -> Unit,
    onDelete: () -> Unit,
) {
    val web = LocalWebPalette.current
    val all = state.allMessages
    val id = node.id.toIntOrNull() ?: return
    val pathTokens = remember(all, id) { com.minashin1120.aiplayground.data.branchPathTokens(all, id) }
    val breakdown = remember(all, id) { branchModelBreakdown(all, id) }
    val nodeTokens = node.tokens ?: ((node.tokensIn ?: 0) + (node.tokensOut ?: 0))
    var name by remember(node.id) { mutableStateOf(store.names[node.id].orEmpty()) }
    val rule = web.twBorder(Tw.gray800)
    val shape = RoundedCornerShape(8.dp)
    Column(modifier.clip(shape).background(web.twBg(Tw.gray900)).border(1.dp, web.twBorder(Tw.gray700), shape).padding(16.dp)) {
        Text("ブランチ詳細", fontSize = 14.sp, lineHeight = 20.sp, fontWeight = FontWeight.Bold, color = web.twText(Tw.purple300),
            modifier = Modifier.fillMaxWidth().drawBehind { drawLine(rule, Offset(0f, size.height), Offset(size.width, size.height), 1.dp.toPx()) }
                .padding(bottom = 4.dp))
        Column(Modifier.padding(top = 8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            val label = SpanStyle(color = web.twText(Tw.gray500))
            val value = SpanStyle(color = web.twText(Tw.gray300))
            Text(buildAnnotatedString {
                withStyle(label) { append("ID: ") }; withStyle(value.copy(fontFamily = FontFamily.Monospace)) { append(node.id) }
            }, fontSize = 12.sp, lineHeight = 16.sp)
            // The Web reads `created_at`, which the message payload does not carry, so it always shows "-".
            Text(buildAnnotatedString { withStyle(label) { append("作成: ") }; withStyle(value) { append("-") } }, fontSize = 12.sp, lineHeight = 16.sp)
            Text(buildAnnotatedString {
                withStyle(label) { append("モデル: ") }; withStyle(value) { append(node.model.ifBlank { "-" }) }
            }, fontSize = 12.sp, lineHeight = 16.sp)
            Text(buildAnnotatedString {
                withStyle(label) { append("トークン: ") }
                withStyle(SpanStyle(color = web.twText(Tw.blue300), fontWeight = FontWeight.Bold)) { append(nodeTokens.toString()) }
                withStyle(SpanStyle(color = web.twText(Tw.gray500))) { append(" / ") }
                withStyle(SpanStyle(color = web.twText(Tw.purple400), fontWeight = FontWeight.Bold)) { append("$pathTokens total") }
            }, fontSize = 12.sp, lineHeight = 16.sp)
        }
        SmallCaps("モデル別内訳 (パス累計)", Modifier.padding(top = 16.dp, bottom = 8.dp), rule)
        Column(Modifier.heightIn(max = 160.dp).verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            breakdown.forEach { (model, stats) ->
                val rowShape = RoundedCornerShape(4.dp)
                Column(
                    Modifier.fillMaxWidth().clip(rowShape).background(web.twBg(Tw.gray800).copy(alpha = 0.5f))
                        .border(1.dp, web.twBorder(Tw.gray700).copy(alpha = 0.5f), rowShape).padding(8.dp),
                ) {
                    Row(Modifier.fillMaxWidth().padding(bottom = 4.dp)) {
                        Text(model, fontSize = 10.sp, lineHeight = 15.sp, fontWeight = FontWeight.Bold, color = web.twText(Tw.gray300),
                            maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f).padding(end = 8.dp))
                        Text(stats[0].toString(), fontSize = 10.sp, lineHeight = 15.sp, fontWeight = FontWeight.Bold, color = web.twText(Tw.blue400))
                    }
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                        listOf("In: ${stats[1]}", "Out: ${stats[2]}", if (stats[3] > 0) "Th: ${stats[3]}" else "").forEach { cell ->
                            Text(cell, fontSize = 9.sp, lineHeight = 13.5.sp, fontFamily = FontFamily.Monospace, color = web.twText(Tw.gray500),
                                modifier = Modifier.weight(1f))
                        }
                    }
                }
            }
        }
        Column(Modifier.padding(top = 16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("名前・ラベル", fontSize = 10.sp, lineHeight = 15.sp, fontWeight = FontWeight.Bold, color = web.twText(Tw.gray500))
            TwInput(name, { name = it.take(200) }, "未命名", fontSize = 12.sp, padding = 6.dp, background = Tw.gray800)
            BranchButton("名前を保存", Tw.blue600, Tw.white, vertical = 6.dp) { onSaveName(name.trim()) }
        }
        Column(
            Modifier.padding(top = 16.dp).drawBehind { drawLine(rule, Offset.Zero, Offset(size.width, 0f), 1.dp.toPx()) }.padding(top = 16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            BranchButton("このブランチに切り替え", Tw.purple600, Tw.white, onClick = onSwitch)
            val isFixed = store.fixed == node.id
            BranchButton(if (isFixed) "固定を解除" else "メインルートに固定", if (isFixed) Tw.gray600 else Tw.amber600, Tw.white, onClick = onFix)
            BranchButton("ブランチを削除", Tw.red600.copy(alpha = 0.2f), Tw.red400, border = Tw.red600.copy(alpha = 0.5f), enabled = !state.offline, onClick = onDelete)
        }
    }
}

@Composable
private fun SmallCaps(text: String, modifier: Modifier, rule: Color) {
    Text(text.uppercase(), fontSize = 10.sp, lineHeight = 15.sp, fontWeight = FontWeight.Bold, color = LocalWebPalette.current.twText(Tw.gray500),
        modifier = modifier.fillMaxWidth().drawBehind { drawLine(rule, Offset(0f, size.height), Offset(size.width, size.height), 1.dp.toPx()) }
            .padding(bottom = 4.dp))
}

@Composable
private fun BranchButton(
    label: String,
    background: Color,
    color: Color,
    border: Color? = null,
    vertical: androidx.compose.ui.unit.Dp = 8.dp,
    enabled: Boolean = true,
    onClick: () -> Unit,
) {
    val shape = RoundedCornerShape(4.dp)
    Text(
        label, fontSize = 12.sp, lineHeight = 16.sp, fontWeight = FontWeight.Bold, color = color, textAlign = TextAlign.Center,
        modifier = Modifier.fillMaxWidth().clip(shape).background(background)
            .then(if (border != null) Modifier.border(1.dp, border, shape) else Modifier)
            .clickable(enabled = enabled, role = Role.Button, onClick = onClick)
            .padding(vertical = vertical),
    )
}

@Composable
private fun BranchFooter(total: Int) {
    val web = LocalWebPalette.current
    val muted = web.twText(Tw.gray500)
    Row(Modifier.fillMaxWidth().padding(top = 16.dp), verticalAlignment = Alignment.CenterVertically) {
        Row(Modifier.weight(1f), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            listOf(Tw.blue500 to "現在のブランチ", Tw.amber500 to "固定済み").forEach { (dot, label) ->
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                    Box(Modifier.size(8.dp).clip(CircleShape).background(dot))
                    Text(label, fontSize = 10.sp, lineHeight = 15.sp, color = muted)
                }
            }
        }
        Text(buildAnnotatedString {
            append("全ブランチ合計: ")
            withStyle(SpanStyle(color = web.twText(Tw.gray300), fontWeight = FontWeight.Bold)) { append(total.toString()) }
            append(" tokens")
        }, fontSize = 10.sp, lineHeight = 15.sp, color = muted)
    }
}
