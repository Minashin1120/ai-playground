package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.ChatMessage

/**
 * メッセージとしてプレビュー (ANDROID_ONLY.md): text shared to the 「メッセージとしてプレビュー」 share target,
 * drawn by [MessageBubble] as an answer (Markdown) or as the user's own message (raw text). Nothing is sent.
 */
@Composable
internal fun SharedTextPreviewDialog(
    text: String,
    loader: FileBytesLoader?,
    onFile: (String) -> Unit,
    onDismiss: () -> Unit,
    onInsert: () -> Unit,
) {
    val web = LocalWebPalette.current
    var asUser by rememberSaveable { mutableStateOf(false) }
    val message = remember(text, asUser) { ChatMessage("shared-preview", if (asUser) "user" else "assistant", text) }
    WebOverlayModal(onDismiss, grayOverlay(), 4.dp, alignment = Alignment.Center) { _ ->
        val shape = RoundedCornerShape(8.dp)
        Column(
            Modifier.padding(16.dp).widthIn(max = 832.dp).fillMaxWidth().modalPanelTaps().clip(shape)
                .background(web.twBg(Tw.gray800)).border(1.dp, web.twBorder(Tw.gray700), shape).padding(24.dp),
        ) {
            Row(Modifier.padding(bottom = 16.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FaIcon(R.drawable.fa_solid_eye, null, size = 18.dp, tint = web.theme300)
                Text("メッセージとしてプレビュー", fontSize = 20.sp, fontWeight = FontWeight.Bold, color = if (web.isLight) web.text else Color.White)
            }
            Row(Modifier.padding(bottom = 12.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                PreviewRoleButton("AIの回答", selected = !asUser) { asUser = false }
                PreviewRoleButton("自分のメッセージ", selected = asUser) { asUser = true }
            }
            val areaShape = RoundedCornerShape(4.dp)
            Box(
                Modifier.weight(1f, fill = false).fillMaxWidth().clip(areaShape).background(web.twBg(Tw.gray900))
                    .border(1.dp, web.twBorder(Tw.gray700), areaShape).verticalScroll(rememberScrollState())
                    // `#chat-container` padding, plus room for the bubble's shadow.
                    .padding(12.dp),
            ) {
                MessageBubble(message, onFile, loader, MessageActions(), controlsVisible = false, onToggleControls = {})
            }
            Row(
                Modifier.fillMaxWidth().padding(top = 24.dp).drawBehind {
                    drawLine(web.twBorder(Tw.gray700), Offset(0f, 0f), Offset(size.width, 0f), 1.dp.toPx())
                }.padding(top = 16.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.End),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text("閉じる", fontSize = 14.sp, color = Tw.gray400,
                    modifier = Modifier.clickable(role = Role.Button, onClick = onDismiss).padding(horizontal = 8.dp, vertical = 8.dp))
                Text("入力欄に入れる", fontSize = 14.sp, fontWeight = FontWeight.Bold, color = Color.White,
                    modifier = Modifier.clip(RoundedCornerShape(4.dp)).background(web.theme.t600)
                        .clickable(role = Role.Button, onClick = onInsert).padding(horizontal = 16.dp, vertical = 8.dp))
            }
        }
    }
}

@Composable
private fun PreviewRoleButton(label: String, selected: Boolean, onClick: () -> Unit) {
    val web = LocalWebPalette.current
    Text(label, fontSize = 12.sp, fontWeight = FontWeight.Bold,
        color = if (selected) Color.White else if (web.isLight) web.text else Tw.gray300,
        modifier = Modifier.clip(RoundedCornerShape(4.dp)).background(if (selected) web.theme.t600 else web.twBg(Tw.gray700))
            .clickable(role = Role.RadioButton, onClick = onClick).padding(horizontal = 12.dp, vertical = 6.dp))
}
