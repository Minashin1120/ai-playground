package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.ChatMessage
import com.minashin1120.aiplayground.data.SharedTextMath

/**
 * メッセージとしてプレビュー (ANDROID_ONLY.md): text shared to the 「メッセージとしてプレビュー」 share target,
 * drawn full screen by [MessageBubble] as an answer, over the chat background and in the chat column
 * (`#chat-container` width and padding), so it looks exactly like an answer in the chat. Nothing is sent.
 */
@Composable
internal fun SharedTextPreviewDialog(
    text: String,
    loader: FileBytesLoader?,
    onFile: (String) -> Unit,
    onDismiss: () -> Unit,
) {
    val web = LocalWebPalette.current
    val colors = MaterialTheme.colorScheme
    // Formulas whose sender dropped the `$` delimiters are drawn as math too (SharedTextMath).
    val message = remember(text) { ChatMessage("shared-preview", "assistant", SharedTextMath.addMissingDelimiters(text)) }
    WebOverlayModal(onDismiss, colors.background, 4.dp, dismissOnOutsideTap = false) { _ ->
        Column(
            Modifier.fillMaxSize().modalPanelTaps()
                // Same backdrop as the chat screen, so the bubble's translucent glass blends the same way.
                .background(Brush.verticalGradient(listOf(colors.background, colors.surfaceContainerLow))),
        ) {
            Row(
                Modifier.fillMaxWidth().drawBehind {
                    drawLine(web.twBorder(Tw.gray700), Offset(0f, size.height), Offset(size.width, size.height), 1.dp.toPx())
                }.padding(start = 16.dp, end = 8.dp, top = 8.dp, bottom = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                FaIcon(R.drawable.fa_solid_eye, null, size = 16.dp, tint = web.theme300)
                Text("メッセージとしてプレビュー", fontSize = 16.sp, fontWeight = FontWeight.Bold, maxLines = 1, overflow = TextOverflow.Ellipsis,
                    color = if (web.isLight) web.text else Color.White, modifier = Modifier.weight(1f))
                Box(
                    Modifier.clip(CircleShape).clickable(onClickLabel = "閉じる", role = Role.Button, onClick = onDismiss).padding(12.dp),
                ) { FaIcon(R.drawable.fa_solid_times, "閉じる", size = 16.dp, tint = web.twText(Tw.gray400)) }
            }
            Box(Modifier.weight(1f).fillMaxWidth()) {
                LazyColumn(
                    Modifier.fillMaxSize().widthIn(max = 832.dp).align(Alignment.TopCenter),
                    // `#chat-container`: 12px padding on phones, like the chat list.
                    contentPadding = PaddingValues(12.dp),
                ) {
                    item(key = "shared-preview") {
                        MessageBubble(message, onFile, loader, MessageActions(), controlsVisible = false, onToggleControls = {})
                    }
                }
            }
        }
    }
}
