package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.R

/**
 * `header.main-chrome-header` of Web (`md:hidden`, phones only): menu, chat title with the temporary
 * label / TTL chip, the "+" new chat button and PDF export.
 */
@Composable
internal fun MobileChatHeader(
    state: ChatState,
    onMenu: () -> Unit,
    onNewChat: () -> Unit,
    onPdf: () -> Unit,
) {
    val web = LocalWebPalette.current
    val temporary = state.selected?.isTemporary == true || state.newThreadTemporary
    val background = if (web.isLight) Color.White.copy(alpha = 0.84f) else Color(7, 10, 20).copy(alpha = 0.72f)
    val rule = if (web.isLight) Color(15, 23, 42).copy(alpha = 0.10f) else web.lineSoft
    Column(
        Modifier
            .fillMaxWidth()
            .shadow(24.dp, ambientColor = Color.Black.copy(alpha = 0.18f), spotColor = Color.Black.copy(alpha = 0.18f))
            .background(web.bg1)
            .background(background),
    ) {
        Row(
            Modifier.fillMaxWidth().statusBarsPadding().padding(horizontal = 12.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Box(
                Modifier.clickable(onClickLabel = "メニュー", role = Role.Button, onClick = onMenu).padding(8.dp),
                contentAlignment = Alignment.Center,
            ) { FaIcon(R.drawable.fa_solid_bars, "メニュー", size = 20.dp, tint = web.twText(Tw.gray300)) }
            Column(Modifier.weight(1f)) {
                if (temporary) {
                    Text("一時チャット", color = web.twText(Tw.amber200).let { if (web.isLight) it else it.copy(alpha = 0.5f) },
                        fontSize = 9.sp, lineHeight = 9.sp, modifier = Modifier.padding(bottom = 2.dp))
                }
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text(
                        state.selected?.title?.ifBlank { "No Title" } ?: "AI Chat",
                        color = web.text, fontSize = 14.sp, lineHeight = 20.sp, fontWeight = FontWeight.Bold,
                        maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f, fill = false),
                    )
                    state.tempChatRemainingSeconds?.takeIf { temporary }?.let { seconds ->
                        Text(
                            "${seconds}秒", color = web.twText(Tw.amber200), fontSize = 9.sp, lineHeight = 9.sp,
                            modifier = Modifier.border(1.dp, Tw.amber500.copy(alpha = 0.4f), RoundedCornerShape(4.dp))
                                .padding(horizontal = 4.dp, vertical = 2.dp),
                        )
                    }
                }
            }
            val enabled = true  // Web starts a new chat at any time (it aborts the running stream).
            Box(
                Modifier
                    .height(44.dp)
                    .clip(RoundedCornerShape(14.dp))
                    .background(web.theme.rgb(0.14f))
                    .clickable(enabled = enabled, onClickLabel = "新規チャット", role = Role.Button, onClick = onNewChat)
                    .padding(horizontal = 8.dp)
                    .graphicsLayer { alpha = if (enabled) 1f else 0.5f },
                contentAlignment = Alignment.Center,
            ) { Text("+", color = web.theme300, fontSize = 20.sp, lineHeight = 28.sp, fontWeight = FontWeight.Bold) }
            val pdfEnabled = true
            val pdfShape = RoundedCornerShape(4.dp)
            Box(
                Modifier
                    .size(width = 30.dp, height = 38.dp)
                    .clip(pdfShape)
                    .background(Brush.verticalGradient(listOf(Color.White.copy(alpha = 0.04f), Color.Black.copy(alpha = 0.06f))))
                    .border(1.dp, Color.White.copy(alpha = 0.06f), pdfShape)
                    .clickable(enabled = pdfEnabled, onClickLabel = "PDF出力", role = Role.Button, onClick = onPdf)
                    .graphicsLayer { alpha = if (pdfEnabled) 1f else 0.5f },
                contentAlignment = Alignment.Center,
            ) {
                // Light theme remaps the `hover:text-red-300` class of this button to red-700.
                FaIcon(R.drawable.fa_solid_file_pdf, "PDF出力", size = 12.dp, tint = if (web.isLight) Color(0xFFB91C1C) else Tw.gray400)
            }
        }
        Box(Modifier.fillMaxWidth().height(1.dp).background(rule))
    }
}
