package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.McpDecision
import org.json.JSONArray
import org.json.JSONObject
import java.util.Locale

/** Web `isGeminiLocalPythonMode`: Gemini chat models with Python on and audio or video attached. */
internal fun isGeminiLocalPythonMode(model: String, hasAudio: Boolean, hasVideo: Boolean, python: Boolean): Boolean {
    val m = model.lowercase(Locale.ROOT)
    if (!m.contains("gemini")) return false
    if (listOf("image", "nano", "tts", "native-audio").any { m.contains(it) }) return false
    return python && (hasAudio || hasVideo)
}

/** Web `openMcpDecisionModal`: pretty-prints JSON arguments, otherwise shows them as sent. */
internal fun mcpArgsPreview(raw: String): String = runCatching {
    val trimmed = raw.trim()
    if (trimmed.startsWith("[")) JSONArray(trimmed).toString(2) else JSONObject(trimmed).toString(2)
}.getOrDefault(raw)

/** The plain Tailwind modal used by the Web dialogs (`bg-gray-800 rounded-lg p-6 border m-4`). */
@Composable
private fun TwPanelModal(onDismiss: () -> Unit, border: Color, maxWidth: Dp, content: @Composable ColumnScope.() -> Unit) {
    val web = LocalWebPalette.current
    WebOverlayModal(onDismiss, grayOverlay(), 4.dp) { phone ->
        Column(
            Modifier.fillMaxSize().verticalScroll(rememberScrollState()),
            verticalArrangement = if (phone) Arrangement.Top else Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            val shape = RoundedCornerShape(8.dp)
            Column(
                Modifier.padding(16.dp).widthIn(max = maxWidth).fillMaxWidth().clip(shape).background(web.twBg(Tw.gray800))
                    .border(1.dp, border, shape).padding(24.dp),
                content = content,
            )
        }
    }
}

/** `#mcp-decision-modal` (外部ツール操作の確認). Back or the overlay denies, like a Web timeout. */
@Composable
internal fun McpDecisionDialog(decision: McpDecision, onDecide: (Boolean) -> Unit) {
    val web = LocalWebPalette.current
    TwPanelModal({ onDecide(false) }, Tw.amber500.copy(alpha = 0.5f), 512.dp) {
        Row(Modifier.padding(bottom = 12.dp), verticalAlignment = Alignment.CenterVertically) {
            FaIcon(R.drawable.fa_solid_shield_alt, null, size = 18.dp, tint = web.twText(Tw.amber300), modifier = Modifier.padding(end = 8.dp))
            Text("外部ツール操作の確認", fontSize = 18.sp, lineHeight = 28.sp, fontWeight = FontWeight.Bold, color = web.twText(Tw.amber300))
        }
        Text(buildAnnotatedString {
            append("AIが、接続中の外部MCPサーバーで")
            withStyle(SpanStyle(fontWeight = FontWeight.Bold)) { append("変更を伴う操作") }
            append("を実行しようとしています。 内容を確認し、よければ「許可」を押してください（時間切れ・キャンセル時は実行されません）。")
        }, fontSize = 12.sp, lineHeight = 19.5.sp, color = web.twText(Tw.gray300), modifier = Modifier.padding(bottom = 12.dp))
        val box = RoundedCornerShape(4.dp)
        Column(
            Modifier.fillMaxWidth().clip(box).background(web.twBg(Tw.gray950).copy(alpha = 0.7f)).border(1.dp, web.twBorder(Tw.gray700), box).padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            val label = SpanStyle(color = web.twText(Tw.gray500))
            Text(buildAnnotatedString {
                withStyle(label) { append("サーバー: ") }
                withStyle(SpanStyle(color = web.twText(Tw.white), fontWeight = FontWeight.Bold)) { append(decision.serverName.ifBlank { "不明なサーバー" }) }
            }, fontSize = 12.sp, lineHeight = 16.sp)
            Text(buildAnnotatedString {
                withStyle(label) { append("ツール: ") }
                withStyle(SpanStyle(color = web.twText(Tw.cyan300), fontFamily = FontFamily.Monospace)) { append(decision.toolName) }
            }, fontSize = 12.sp, lineHeight = 16.sp)
            Text("入力:", fontSize = 12.sp, lineHeight = 16.sp, color = web.twText(Tw.gray500))
            Text(
                mcpArgsPreview(decision.argsPreview), fontSize = 11.sp, lineHeight = 16.5.sp, fontFamily = FontFamily.Monospace,
                color = web.twText(Tw.gray400),
                modifier = Modifier.fillMaxWidth().heightIn(max = 160.dp).clip(box).background(Color.Black.copy(alpha = 0.3f))
                    .verticalScroll(rememberScrollState()).padding(8.dp),
            )
        }
        Row(Modifier.fillMaxWidth().padding(top = 16.dp), horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.End)) {
            WebButton(onClick = { onDecide(false) }, variant = WebButtonVariant.Ghost) { Text("拒否") }
            Row(
                Modifier.clip(RoundedCornerShape(12.dp)).background(Color(0xFFB45309)).clickable(role = Role.Button) { onDecide(true) }
                    .padding(horizontal = 16.dp, vertical = 9.dp),
                verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(7.dp),
            ) {
                FaIcon(R.drawable.fa_solid_check, null, size = 13.dp, tint = web.textInverse)
                Text("許可", fontSize = 13.sp, fontWeight = FontWeight.Bold, color = web.textInverse)
            }
        }
    }
}

/** `#gemini-local-python-modal` (ローカルPython実行に切替); [onResult] gets (proceed, dontShowAgain). */
@Composable
internal fun GeminiLocalPythonDialog(onResult: (Boolean, Boolean) -> Unit) {
    val web = LocalWebPalette.current
    var dontShow by remember { mutableStateOf(false) }
    TwPanelModal({ onResult(false, dontShow) }, Tw.yellow500.copy(alpha = 0.6f), 448.dp) {
        Row(Modifier.fillMaxWidth().padding(bottom = 12.dp), verticalAlignment = Alignment.CenterVertically) {
            FaIcon(R.drawable.fa_solid_terminal, null, size = 18.dp, tint = web.twText(Tw.yellow300), modifier = Modifier.padding(end = 8.dp))
            Text("ローカルPython実行に切替", fontSize = 18.sp, lineHeight = 28.sp, fontWeight = FontWeight.Bold, color = web.twText(Tw.yellow300),
                modifier = Modifier.weight(1f))
            Box(Modifier.size(32.dp).clip(CircleShape).clickable(role = Role.Button) { onResult(false, dontShow) }, contentAlignment = Alignment.Center) {
                FaIcon(R.drawable.fa_solid_times, "閉じる", size = 16.dp, tint = web.twText(Tw.gray400))
            }
        }
        val body = web.twText(Tw.gray300)
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Geminiの音声/動画入力ではPython実行がローカルモードに切り替わります。", fontSize = 14.sp, lineHeight = 20.sp, color = body)
            Text(buildAnnotatedString {
                append("実行したい場合は ")
                withStyle(SpanStyle(fontFamily = FontFamily.Monospace)) { append("```python") }
                append(" ブロックの先頭行に ")
                withStyle(SpanStyle(fontFamily = FontFamily.Monospace)) { append("# EXECUTE") }
                append(" を入れてください。")
            }, fontSize = 14.sp, lineHeight = 20.sp, color = body)
        }
        Row(
            Modifier.padding(top = 16.dp).toggleable(dontShow, role = Role.Checkbox) { dontShow = it },
            verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            WebCheckbox(dontShow, null)
            Text("次回から表示しない", fontSize = 12.sp, lineHeight = 16.sp, color = web.twText(Tw.gray400))
        }
        Row(Modifier.fillMaxWidth().padding(top = 20.dp), horizontalArrangement = Arrangement.spacedBy(12.dp, Alignment.End)) {
            Text("キャンセル", fontSize = 16.sp, lineHeight = 24.sp, color = web.twText(Tw.gray400),
                modifier = Modifier.clip(RoundedCornerShape(4.dp)).clickable(role = Role.Button) { onResult(false, dontShow) }
                    .padding(horizontal = 16.dp, vertical = 8.dp))
            Text("このまま送信", fontSize = 16.sp, lineHeight = 24.sp, fontWeight = FontWeight.Bold, color = Tw.white,
                modifier = Modifier.clip(RoundedCornerShape(4.dp)).background(Tw.yellow600).clickable(role = Role.Button) { onResult(true, dontShow) }
                    .padding(horizontal = 16.dp, vertical = 8.dp))
        }
    }
}

/**
 * Web `#bot-lock-overlay`: blocks the whole app until the lock ends, counting down `ロック解除まで: m:ss`.
 * [onExpired] runs when the countdown reaches zero (the Web reloads the page).
 */
@Composable
internal fun AccountLockOverlay(lock: com.minashin1120.aiplayground.data.AccountLock, onExpired: () -> Unit) {
    var remaining by remember(lock) { mutableStateOf(((lock.untilMillis - System.currentTimeMillis()) / 1000).coerceAtLeast(0)) }
    LaunchedEffect(lock) {
        while (true) {
            remaining = ((lock.untilMillis - System.currentTimeMillis() + 999) / 1000).coerceAtLeast(0)
            if (remaining <= 0) { onExpired(); break }
            kotlinx.coroutines.delay(1000)
        }
    }
    androidx.compose.ui.window.Dialog(
        onDismissRequest = {},
        properties = androidx.compose.ui.window.DialogProperties(dismissOnBackPress = false, dismissOnClickOutside = false, usePlatformDefaultWidth = false),
    ) {
        WebModalWindow(0.dp)
        Box(Modifier.fillMaxSize().background(Color(3, 7, 18).copy(alpha = 0.94f)).padding(24.dp), contentAlignment = Alignment.Center) {
            val shape = RoundedCornerShape(12.dp)
            Column(
                Modifier.widthIn(max = 440.dp).fillMaxWidth().clip(shape).background(Color(0xFF0F172A)).border(1.dp, Color(0xFFF59E0B), shape)
                    .padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                FaIcon(R.drawable.fa_solid_lock, null, size = 26.dp, tint = Color(0xFFFBBF24))
                Text("アカウントが一時的にロックされました", fontSize = 16.sp, fontWeight = FontWeight.Bold, color = Color(0xFFFBBF24),
                    textAlign = androidx.compose.ui.text.style.TextAlign.Center)
                Text(lock.message, fontSize = 13.sp, lineHeight = 22.1.sp, color = Color(0xFFF1F5F9), textAlign = androidx.compose.ui.text.style.TextAlign.Center)
                Text("ロック解除まで: ${remaining / 60}:${(remaining % 60).toString().padStart(2, '0')}", fontSize = 12.sp, color = Color(0xFF94A3B8),
                    modifier = Modifier.padding(top = 2.dp))
                Text("ロック解除までしばらくお待ちください。同じ操作を繰り返すとBANされる場合があります。", fontSize = 11.sp, lineHeight = 17.6.sp,
                    color = Color(0xFF94A3B8), textAlign = androidx.compose.ui.text.style.TextAlign.Center)
            }
        }
    }
}

/**
 * Web `#bot-detection-overlay` for the chat Turnstile check. The check runs on the site in the browser
 * (Android cannot host the widget), so the card offers to open it again or to close.
 */
@Composable
internal fun SessionTurnstileOverlay(onOpen: () -> Unit, onDismiss: () -> Unit) {
    androidx.compose.ui.window.Dialog(
        onDismissRequest = onDismiss,
        properties = androidx.compose.ui.window.DialogProperties(dismissOnClickOutside = false, usePlatformDefaultWidth = false),
    ) {
        WebModalWindow(0.dp)
        Box(Modifier.fillMaxSize().background(Color(3, 7, 18).copy(alpha = 0.92f)).padding(24.dp), contentAlignment = Alignment.Center) {
            val shape = RoundedCornerShape(12.dp)
            Column(
                Modifier.widthIn(max = 420.dp).fillMaxWidth().clip(shape).background(Color(0xFF0F172A)).border(1.dp, Color(0xFF334155), shape)
                    .padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Text("安全性の確認中...", fontSize = 15.sp, fontWeight = FontWeight.Bold, color = Color(0xFFF1F5F9))
                Text("自動アクセス防止のため、確認を完了してください。", fontSize = 12.sp, lineHeight = 19.2.sp, color = Color(0xFF94A3B8),
                    textAlign = androidx.compose.ui.text.style.TextAlign.Center)
                Row(Modifier.padding(top = 8.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    WebButton(onClick = onDismiss) { Text("閉じる") }
                    WebButton(onClick = onOpen, variant = WebButtonVariant.Primary) { Text("確認を開く") }
                }
            }
        }
    }
}

/**
 * Web `#api-key-required-modal`: save the missing key and send again, switch to another model,
 * or cancel and show the error.
 */
@Composable
internal fun ApiKeyRequiredDialog(
    modelName: String,
    modelId: String,
    info: com.minashin1120.aiplayground.data.ApiKeyInfo,
    onSave: (String) -> Unit,
    onSwitch: () -> Unit,
    onCancel: () -> Unit,
) {
    val web = LocalWebPalette.current
    var key by remember { mutableStateOf("") }
    TwPanelModal(onCancel, web.twBorder(Tw.gray700), 448.dp) {
        Row(Modifier.padding(bottom = 16.dp), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Box(Modifier.padding(top = 2.dp).size(40.dp).clip(CircleShape).background(web.twBg(Tw.red900, 0.5f)), contentAlignment = Alignment.Center) {
                FaIcon(R.drawable.fa_solid_key, null, size = 18.dp, tint = web.twText(Tw.red400))
            }
            Column(Modifier.weight(1f)) {
                Text("APIキーが必要です", fontSize = 18.sp, lineHeight = 28.sp, fontWeight = FontWeight.Bold, color = web.twText(Tw.white))
                Text("$modelName（$modelId）", fontSize = 12.sp, lineHeight = 16.sp, color = web.twText(Tw.gray400), modifier = Modifier.padding(top = 4.dp))
            }
        }
        Text("このモデルを使用するには${info.label}の設定が必要です。", fontSize = 14.sp, lineHeight = 20.sp, color = web.twText(Tw.gray300),
            modifier = Modifier.padding(bottom = 16.dp))
        Text(info.label, fontSize = 12.sp, lineHeight = 16.sp, color = web.twText(Tw.gray500), modifier = Modifier.padding(bottom = 4.dp))
        val shape = RoundedCornerShape(4.dp)
        val textColor = web.twText(Tw.white)
        val style = androidx.compose.ui.text.TextStyle(fontSize = 14.sp, lineHeight = 20.sp, color = textColor, fontFamily = WebFonts.sans)
        androidx.compose.foundation.text.BasicTextField(
            key, { key = it.take(4096) }, singleLine = true, textStyle = style, cursorBrush = androidx.compose.ui.graphics.SolidColor(textColor),
            visualTransformation = androidx.compose.ui.text.input.PasswordVisualTransformation(),
            keyboardOptions = androidx.compose.foundation.text.KeyboardOptions(keyboardType = androidx.compose.ui.text.input.KeyboardType.Password,
                imeAction = androidx.compose.ui.text.input.ImeAction.Send),
            keyboardActions = androidx.compose.foundation.text.KeyboardActions(onSend = { onSave(key) }),
            modifier = Modifier.fillMaxWidth().clip(shape).background(web.twBg(Tw.gray800)).border(1.dp, web.twBorder(Tw.gray600), shape),
            decorationBox = { inner ->
                Box(Modifier.padding(horizontal = 12.dp, vertical = 8.dp)) {
                    if (key.isEmpty()) Text("APIキーを入力", style = style.copy(color = Tw.gray500))
                    inner()
                }
            },
        )
        Column(Modifier.padding(top = 12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            val buttonShape = RoundedCornerShape(8.dp)
            @Composable
            fun Action(icon: Int?, label: String, background: Color, color: Color, bold: Boolean, vertical: androidx.compose.ui.unit.Dp, onClick: () -> Unit) {
                Row(
                    Modifier.fillMaxWidth().clip(buttonShape).background(background).clickable(role = Role.Button, onClick = onClick).padding(vertical = vertical),
                    horizontalArrangement = Arrangement.Center, verticalAlignment = Alignment.CenterVertically,
                ) {
                    icon?.let { FaIcon(it, null, size = 13.dp, tint = color, modifier = Modifier.padding(end = 4.dp)) }
                    Text(label, fontSize = 14.sp, lineHeight = 20.sp, fontWeight = if (bold) FontWeight.Bold else FontWeight.Normal, color = color)
                }
            }
            Action(R.drawable.fa_solid_save, "APIキーを保存して送信", Tw.blue600, Color.White, true, 10.dp) { onSave(key) }
            Action(R.drawable.fa_solid_exchange_alt, "他のモデルに切り替え", web.twBg(Tw.gray700), web.twText(Tw.white), true, 8.dp, onSwitch)
            Action(null, "キャンセル（エラーメッセージを表示）", Color.Transparent, web.twText(Tw.gray400), false, 8.dp, onCancel)
        }
    }
}
