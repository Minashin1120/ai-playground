@file:OptIn(androidx.compose.foundation.layout.ExperimentalLayoutApi::class)

package com.minashin1120.aiplayground.ui

import androidx.annotation.DrawableRes
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.BatchJob
import com.minashin1120.aiplayground.data.batchFormatTime
import com.minashin1120.aiplayground.data.batchProviderLabel
import com.minashin1120.aiplayground.data.batchStateShort
import kotlinx.coroutines.delay

/** `#batch-modal` (Batch処理): filter tabs, job cards with 開く／停止／履歴から削除, refreshed every 5 seconds while open. */
@Composable
internal fun BatchDialog(
    state: ChatState,
    onLoad: (silent: Boolean) -> Unit,
    onOpen: (BatchJob) -> Unit,
    onCancel: (BatchJob) -> Unit,
    onDelete: (BatchJob) -> Unit,
    onDismiss: () -> Unit,
) {
    val web = LocalWebPalette.current
    var filter by remember { mutableStateOf("all") }
    var cancelTarget by remember { mutableStateOf<BatchJob?>(null) }
    var deleteTarget by remember { mutableStateOf<BatchJob?>(null) }
    LaunchedEffect(Unit) {
        onLoad(false)
        while (true) {
            delay(5_000)
            onLoad(true)
        }
    }
    val jobs = state.batchJobs.filter {
        when (filter) {
            "active" -> it.active
            "done" -> !it.active
            else -> true
        }
    }
    WebOverlayModal(onDismiss, if (web.isLight) Color(15, 23, 42).copy(alpha = 0.35f) else Color(3, 7, 16).copy(alpha = 0.72f), 10.dp) { phone ->
        BoxWithConstraints(Modifier.fillMaxSize().padding(if (phone) 8.dp else 16.dp), contentAlignment = Alignment.Center) {
            val shape = RoundedCornerShape(if (phone) 18.dp else 22.dp)
            val panelSize = if (phone) Modifier.fillMaxSize() else Modifier.widthIn(max = 820.dp).fillMaxWidth().height(minOf(maxHeight * 0.88f, 820.dp))
            Column(
                panelSize.clip(shape)
                    .background(Brush.verticalGradient(
                        if (web.isLight) listOf(Color.White, Color(0xFFF7F9FC))
                        else listOf(Color(10, 14, 28).copy(alpha = 0.98f), Color(6, 9, 18).copy(alpha = 0.99f)),
                    ))
                    .border(1.dp, if (web.isLight) Color(15, 23, 42).copy(alpha = 0.10f) else Color.White.copy(alpha = 0.08f), shape),
            ) {
                BatchHeader(phone, jobs.size, onRefresh = { onLoad(false) }, onClose = onDismiss)
                Row(
                    Modifier.padding(start = if (phone) 14.dp else 20.dp, end = if (phone) 14.dp else 20.dp, bottom = if (phone) 10.dp else 12.dp),
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    listOf("all" to "すべて", "active" to "実行中", "done" to "終了").forEach { (value, label) ->
                        BatchFilterTab(label, filter == value) { filter = value }
                    }
                }
                val line = web.line
                val glow = web.theme.rgb(if (web.isLight) 0.05f else 0.06f)
                LazyColumn(
                    Modifier.weight(1f).fillMaxWidth()
                        .drawBehind {
                            drawLine(line, Offset.Zero, Offset(size.width, 0f), 1.dp.toPx())
                            drawRect(Brush.radialGradient(listOf(glow, Color.Transparent), center = Offset(size.width, 0f), radius = 364.dp.toPx()))
                        },
                    contentPadding = if (phone) PaddingValues(start = 14.dp, end = 14.dp, top = 14.dp, bottom = 16.dp)
                        else PaddingValues(start = 18.dp, end = 18.dp, top = 16.dp, bottom = 18.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    if (jobs.isEmpty()) item(key = "empty") {
                        Column(
                            Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 48.dp),
                            horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            FaIcon(R.drawable.fa_solid_layer_group, null, size = 28.dp, tint = web.muted, modifier = Modifier.alpha(0.4f))
                            Text("Batch処理の履歴はありません", fontSize = 13.sp, color = web.muted)
                        }
                    }
                    items(jobs, key = { it.id }) { job ->
                        BatchJobCard(job, onOpen = { onOpen(job) }, onCancel = { cancelTarget = job }, onDelete = { deleteTarget = job })
                    }
                }
            }
        }
    }
    cancelTarget?.let { job ->
        BrowserConfirmDialog("このBatch処理を停止しますか？") { ok -> cancelTarget = null; if (ok) onCancel(job) }
    }
    deleteTarget?.let { job ->
        BrowserConfirmDialog("このBatch処理の履歴を削除しますか？") { ok -> deleteTarget = null; if (ok) onDelete(job) }
    }
}

@Composable
private fun BatchHeader(phone: Boolean, count: Int, onRefresh: () -> Unit, onClose: () -> Unit) {
    val web = LocalWebPalette.current
    Row(
        Modifier.fillMaxWidth().background(Brush.verticalGradient(listOf(web.theme.rgb(0.12f), Color.Transparent)))
            .padding(start = if (phone) 14.dp else 20.dp, end = if (phone) 14.dp else 20.dp, top = if (phone) 14.dp else 16.dp, bottom = if (phone) 8.dp else 12.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Row(Modifier.weight(1f), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(14.dp)) {
            val tile = RoundedCornerShape(14.dp)
            Box(
                Modifier.size(44.dp).clip(tile)
                    .background(Brush.linearGradient(listOf(web.theme.rgb(0.30f), web.theme.rgb(0.08f)), start = Offset.Zero, end = Offset.Infinite))
                    .border(1.dp, web.theme.rgb(0.30f), tile),
                contentAlignment = Alignment.Center,
            ) { FaIcon(R.drawable.fa_solid_layer_group, null, size = 18.dp, tint = web.theme300) }
            Column(Modifier.weight(1f)) {
                Text("Batch処理", fontSize = 18.sp, lineHeight = 25.sp, fontWeight = FontWeight.Bold, letterSpacing = 0.18.sp, color = web.text)
                Text("実行中・完了したジョブを確認・管理", fontSize = 12.sp, lineHeight = 17.sp, color = web.muted, modifier = Modifier.padding(top = 2.dp))
            }
        }
        val chrome = if (web.isLight) Color(15, 23, 42).copy(alpha = 0.04f) else Color.White.copy(alpha = 0.03f)
        val chromeBorder = if (web.isLight) Color(15, 23, 42).copy(alpha = 0.08f) else web.lineSoft
        Text("${count}件", fontSize = 11.sp, fontWeight = FontWeight.SemiBold, letterSpacing = 0.22.sp, color = web.muted,
            modifier = Modifier.clip(CircleShape)
                .background(if (web.isLight) chrome else Color.White.copy(alpha = 0.05f)).border(1.dp, chromeBorder, CircleShape)
                .padding(horizontal = 11.dp, vertical = 5.dp))
        listOf(Triple(R.drawable.fa_solid_rotate_right, "最新の状態に更新", onRefresh), Triple(R.drawable.fa_solid_times, "閉じる", onClose)).forEach { (icon, label, action) ->
            val btn = RoundedCornerShape(12.dp)
            Box(
                Modifier.size(38.dp).clip(btn).background(chrome).border(1.dp, chromeBorder, btn)
                    .clickable(role = Role.Button, onClick = action).semantics { contentDescription = label },
                contentAlignment = Alignment.Center,
            ) { FaIcon(icon, null, size = 14.dp, tint = web.muted) }
        }
    }
}

@Composable
private fun BatchFilterTab(label: String, active: Boolean, onClick: () -> Unit) {
    val web = LocalWebPalette.current
    val bg = when {
        active -> web.theme.rgb(0.16f)
        web.isLight -> Color(15, 23, 42).copy(alpha = 0.04f)
        else -> Color.White.copy(alpha = 0.04f)
    }
    val border = when {
        active -> web.theme.rgb(0.34f)
        web.isLight -> Color(15, 23, 42).copy(alpha = 0.10f)
        else -> web.line
    }
    Text(label, fontSize = 12.sp, fontWeight = FontWeight.SemiBold, letterSpacing = 0.12.sp, color = if (active) web.theme200 else web.muted,
        modifier = Modifier.clip(CircleShape).background(bg).border(1.dp, border, CircleShape)
            .clickable(role = Role.Tab, onClick = onClick).padding(horizontal = 14.dp, vertical = 7.dp))
}

/** Web `batchStateTone`: (border, background, text). */
@Composable
private fun batchTone(state: String): Triple<Color, Color, Color> {
    val web = LocalWebPalette.current
    fun tone(border: Color, bg: Color, bgAlpha: Float, text: Color) =
        Triple(web.twBorder(border, 0.4f), web.twBg(bg, bgAlpha), web.twText(text))
    return when (state.uppercase()) {
        "JOB_STATE_SUCCEEDED" -> tone(Tw.emerald500, Tw.emerald900, 0.2f, Tw.emerald200)
        "JOB_STATE_FAILED" -> tone(Tw.red500, Tw.red900, 0.2f, Tw.red200)
        "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED" -> tone(Tw.gray500, Tw.gray700, 0.3f, Tw.gray300)
        "JOB_STATE_CANCELLING" -> tone(Tw.amber500, Tw.amber900, 0.2f, Tw.amber200)
        else -> tone(Tw.violet500, Tw.violet900, 0.2f, Tw.violet200)
    }
}

@Composable
private fun BatchJobCard(job: BatchJob, onOpen: () -> Unit, onCancel: () -> Unit, onDelete: () -> Unit) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(16.dp)
    Column(
        Modifier.fillMaxWidth().clip(shape)
            .background(if (web.isLight) Color.White else Color(10, 16, 30).copy(alpha = 0.78f))
            .background(Brush.verticalGradient(
                if (web.isLight) listOf(Color.White.copy(alpha = 0.9f), Color(248, 250, 252).copy(alpha = 0.94f))
                else listOf(Color.White.copy(alpha = 0.045f), Color.White.copy(alpha = 0.018f)),
            ))
            .border(1.dp, if (web.isLight) Color(15, 23, 42).copy(alpha = 0.08f) else Color.White.copy(alpha = 0.08f), shape)
            .padding(horizontal = 18.dp, vertical = 16.dp),
    ) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp), verticalAlignment = Alignment.Top) {
            Column(Modifier.weight(1f)) {
                Text(job.threadTitle, fontSize = 14.sp, lineHeight = 20.sp, fontWeight = FontWeight.Bold, color = web.text,
                    maxLines = 1, overflow = TextOverflow.Ellipsis)
                FlowRow(Modifier.padding(top = 4.dp), horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                        FaIcon(R.drawable.fa_solid_layer_group, null, size = 10.dp, tint = web.muted)
                        Text(batchProviderLabel(job.provider), fontSize = 10.sp, lineHeight = 15.sp, color = web.muted)
                    }
                    if (job.model.isNotBlank()) Text(job.model, fontSize = 10.sp, lineHeight = 15.sp, color = web.muted, maxLines = 1,
                        overflow = TextOverflow.Ellipsis, modifier = Modifier.widthIn(max = 256.dp))
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        FaIcon(R.drawable.fa_solid_history, null, size = 10.dp, tint = web.muted, modifier = Modifier.padding(end = 4.dp))
                        Text(batchFormatTime(job.createdAt), fontSize = 10.sp, lineHeight = 15.sp, color = web.muted)
                    }
                }
            }
            val (border, bg, fg) = batchTone(job.state)
            Text(batchStateShort(job.state), fontSize = 10.sp, fontWeight = FontWeight.Bold, letterSpacing = 0.2.sp, color = fg,
                modifier = Modifier.clip(CircleShape).background(bg).border(1.dp, border, CircleShape).padding(horizontal = 10.dp, vertical = 3.dp))
        }
        if (job.status.isNotBlank()) Text(job.status, fontSize = 11.sp, lineHeight = 17.sp,
            color = if (web.isLight) web.text else Color(226, 232, 240).copy(alpha = 0.9f), modifier = Modifier.padding(top = 8.dp))
        if (job.error.isNotBlank()) Text(job.error, fontSize = 10.sp, lineHeight = 15.sp, color = Color(0xFFFDA4AF), modifier = Modifier.padding(top = 4.dp))
        val buttons = buildList {
            if (job.threadExists) add(Triple(R.drawable.fa_solid_comment_dots, "開く", 0))
            if (job.canCancel) add(Triple(R.drawable.fa_solid_stop, "停止", 1))
            if (!job.active) add(Triple(R.drawable.fa_solid_trash, "履歴から削除", 2))
        }
        if (buttons.isNotEmpty()) {
            FlowRow(Modifier.padding(top = 12.dp), horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                buttons.forEach { (icon, label, kind) ->
                    BatchActionButton(icon, label, kind, when (kind) { 0 -> onOpen; 1 -> onCancel; else -> onDelete })
                }
            }
        }
    }
}

@Composable
private fun BatchActionButton(@DrawableRes icon: Int, label: String, kind: Int, onClick: () -> Unit) {
    val (border, bg, fg) = when (kind) {
        0 -> Triple(Color(96, 165, 250).copy(alpha = 0.45f), Color(37, 99, 235).copy(alpha = 0.18f), Color(0xFFBFDBFE))
        1 -> Triple(Color(251, 191, 36).copy(alpha = 0.45f), Color(180, 83, 9).copy(alpha = 0.18f), Color(0xFFFDE68A))
        else -> Triple(Color(248, 113, 113).copy(alpha = 0.40f), Color(153, 27, 27).copy(alpha = 0.18f), Color(0xFFFECACA))
    }
    val web = LocalWebPalette.current
    val text = web.twText(fg)
    val shape = RoundedCornerShape(11.dp)
    Row(
        Modifier.clip(shape).background(bg).border(1.dp, border, shape).clickable(role = Role.Button, onClick = onClick)
            .padding(horizontal = 12.dp, vertical = 7.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        FaIcon(icon, null, size = 11.dp, tint = text)
        Text(label, fontSize = 11.sp, lineHeight = 14.sp, fontWeight = FontWeight.Bold, letterSpacing = 0.11.sp, color = text)
    }
}
