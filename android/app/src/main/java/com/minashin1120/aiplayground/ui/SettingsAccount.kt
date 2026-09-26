@file:OptIn(androidx.compose.foundation.layout.ExperimentalLayoutApi::class)

package com.minashin1120.aiplayground.ui

import android.graphics.BitmapFactory
import android.util.Base64
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.ChatViewModel
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.LoginSession
import com.minashin1120.aiplayground.data.TRANSFER_PHASE_LABELS
import com.minashin1120.aiplayground.data.TransferUiState
import java.time.Instant
import java.time.LocalDateTime
import java.time.OffsetDateTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter

/** Web `formatSessionTime` (`new Date(value).toLocaleString()` in ja-JP); naive times are local like JS. */
internal fun webLocaleTime(value: String?): String {
    if (value.isNullOrBlank()) return "不明"
    val formatter = DateTimeFormatter.ofPattern("yyyy/M/d H:mm:ss")
    return runCatching { OffsetDateTime.parse(value).atZoneSameInstant(ZoneId.systemDefault()).format(formatter) }
        .recoverCatching { Instant.parse(value).atZone(ZoneId.systemDefault()).format(formatter) }
        .recoverCatching { LocalDateTime.parse(value).format(formatter) }
        .getOrDefault(value)
}

/** `ZIP` sizes as the Web formats them (`renderAccountExportAvailability`). */
internal fun exportSizeLabel(bytes: Long): String =
    if (bytes >= 1024L * 1024 * 1024) String.format("%.2f GB", bytes / (1024.0 * 1024 * 1024))
    else String.format("%.1f MB", bytes / (1024.0 * 1024))

/** Callbacks shared by the account-related tabs. */
internal class AccountSettingsContext(
    val model: ChatViewModel,
    val ops: AccountOps,
    val confirm: (String, () -> Unit) -> Unit,
    val notify: (String) -> Unit,
    val onWeb: (String) -> Unit,
    val pickImportFile: () -> Unit,
    val saveExport: () -> Unit,
)

internal fun accountCards(state: ChatState, form: SettingsForm, ctx: AccountSettingsContext): List<SettingsCardSpec> {
    val prefs = state.preferences
    return listOf(
        SettingsCardSpec(SettingsTab.Account, "account", "アカウント設定", "アカウント設定 ユーザー名変更 パスワード変更 変更しない場合は空欄") {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Column {
                    SettingsFieldLabel("ユーザー名変更")
                    SettingsTextField(form.newUsername, { form.newUsername = it.take(80) }, Modifier.fillMaxWidth(), placeholder = "変更しない場合は空欄")
                }
                Column {
                    SettingsFieldLabel("パスワード変更")
                    SettingsTextField(form.newPassword, { form.newPassword = it.take(256) }, Modifier.fillMaxWidth(), placeholder = "変更しない場合は空欄", password = true)
                }
            }
        },
        SettingsCardSpec(SettingsTab.Account, "easy-login", "簡易ログイン",
            "簡易ログイン 警告: 有効期間中は一時パスワードでログインでき、2FAはスキップされます。 分間有効 一時パスワード発行 発行をキャンセル") {
            var minutes by remember { mutableStateOf("5") }
            var result by remember { mutableStateOf<com.minashin1120.aiplayground.data.EasyLogin?>(null) }
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                SettingsDesc("警告: 有効期間中は一時パスワードでログインでき、2FAはスキップされます。ログイン後は自動で無効化されます。", color = Tw.red300)
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    SettingsTextField(minutes, { minutes = it.filter(Char::isDigit).take(3) }, Modifier.width(80.dp).align(Alignment.CenterVertically),
                        number = true, fontSize = 12.sp)
                    Text("分間有効", fontSize = 12.sp, color = settingsLabelColor(), modifier = Modifier.align(Alignment.CenterVertically))
                    SettingsSmallButton("一時パスワード発行", {
                        val mins = minutes.toIntOrNull() ?: 5
                        ctx.confirm("簡易ログインを${mins}分間有効にしますか？") {
                            ctx.ops.run("簡易ログインの発行に失敗しました") { result = easyLogin(mins) }
                        }
                    }, tone = SettingsButtonTone.Orange)
                    SettingsSmallButton("発行をキャンセル", {
                        ctx.confirm("現在の一時パスワード発行をキャンセルしますか？") {
                            ctx.ops.run("キャンセルに失敗しました") {
                                if (cancelEasyLogin()) { result = null; ctx.notify("簡易ログインをキャンセルしました") }
                                else ctx.notify("キャンセルに失敗しました")
                            }
                        }
                    })
                }
                result?.let { issued ->
                    val shape = RoundedCornerShape(4.dp)
                    Column(Modifier.fillMaxWidth().clip(shape).background(LocalWebPalette.current.twBg(Tw.gray800))
                        .border(1.dp, Tw.orange500.copy(alpha = 0.4f), shape).padding(8.dp)) {
                        SelectionContainer {
                            Text("一時パスワード: ${issued.password}", fontSize = 11.sp, color = Tw.orange200, fontFamily = FontFamily.Monospace)
                        }
                        Text("有効期限: ${issued.expiresAt}", fontSize = 11.sp, color = Tw.orange200, fontFamily = FontFamily.Monospace)
                    }
                }
            }
        },
        linkCard("google", "Google 連携", R.drawable.fa_brands_google, prefs?.googleLinked == true, prefs?.googleEmail.orEmpty(),
            "Google アカウントでログインできるようになります。", "連携中の Google アカウント", "Google と連携する",
            "Google 連携を解除しますか？\n解除後は Google ログインが利用できなくなります（パスワードが設定されていない場合はログインできなくなる可能性があります）。",
            "Google 連携を解除しました", ctx) { unlinkGoogle() },
        linkCard("minashin", "Minashin 連携", null, prefs?.minashinLinked == true, prefs?.minashinEmail.orEmpty(),
            "Minashin アカウントでログインできるようになります。", "連携中の Minashin アカウント", "Minashin と連携する",
            "Minashin 連携を解除しますか？\n解除後は Minashin ログインが利用できなくなります（パスワードが設定されていない場合はログインできなくなる可能性があります）。",
            "Minashin 連携を解除しました", ctx) { unlinkMinashin() },
    )
}

private fun linkCard(
    key: String, title: String, icon: Int?, linked: Boolean, email: String, hint: String, linkedHint: String,
    linkLabel: String, unlinkConfirm: String, unlinked: String, ctx: AccountSettingsContext,
    unlink: suspend com.minashin1120.aiplayground.data.AccountApi.() -> Any,
) = SettingsCardSpec(SettingsTab.Account, key, title, "$title 未連携 連携済み $hint $linkLabel 連携を解除", titleIcon = icon) {
    val web = LocalWebPalette.current
    Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Box(
                Modifier.size(40.dp).clip(CircleShape).background(if (linked) Tw.green900.copy(alpha = 0.3f) else web.twBg(Tw.gray800)),
                contentAlignment = Alignment.Center,
            ) {
                if (icon != null) FaIcon(icon, null, size = 20.dp, tint = if (linked) Tw.green400 else web.text)
                else MinashinMark()
            }
            Column {
                Text(if (linked) "連携済み" else "未連携", fontSize = 14.sp, fontWeight = FontWeight.Bold,
                    color = if (linked) Tw.green400 else if (web.isLight) web.text else Tw.gray200)
                SettingsDesc(if (linked) email.ifBlank { linkedHint } else hint)
            }
        }
        if (linked) {
            val shape = RoundedCornerShape(4.dp)
            Text("連携を解除", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = Tw.red400,
                modifier = Modifier.clip(shape).background(Tw.red900.copy(alpha = 0.2f)).border(1.dp, Tw.red800, shape)
                    .clickableRole { ctx.confirm(unlinkConfirm) {
                        ctx.ops.run("解除に失敗しました") { unlink(); ctx.notify(unlinked); ctx.model.loadPreferences() }
                    } }
                    .padding(horizontal = 16.dp, vertical = 8.dp))
        } else {
            // Linking needs the provider sign-in in a browser tab (Phase 3c); until then it opens Web.
            Text(linkLabel, fontSize = 12.sp, fontWeight = FontWeight.Bold, color = Color.White,
                modifier = Modifier.clip(RoundedCornerShape(4.dp)).background(web.theme.t600)
                    .clickableRole { ctx.onWeb("/settings") }.padding(horizontal = 16.dp, vertical = 8.dp))
        }
    }
}

/** The Minashin "M" tile from the Web card (gradient square with a white M). */
@Composable
private fun MinashinMark() {
    androidx.compose.foundation.Canvas(Modifier.size(24.dp)) {
        val brush = androidx.compose.ui.graphics.Brush.linearGradient(listOf(Color(0xFF6366F1), Color(0xFFA855F7), Color(0xFFD946EF)))
        drawRoundRect(brush, cornerRadius = androidx.compose.ui.geometry.CornerRadius(size.width * 14f / 64f))
        val s = size.width / 64f
        val path = androidx.compose.ui.graphics.Path().apply {
            moveTo(18 * s, 45 * s); lineTo(18 * s, 20 * s); lineTo(32 * s, 37 * s); lineTo(46 * s, 20 * s); lineTo(46 * s, 45 * s)
        }
        drawPath(path, Color.White, style = androidx.compose.ui.graphics.drawscope.Stroke(5 * s,
            cap = androidx.compose.ui.graphics.StrokeCap.Round, join = androidx.compose.ui.graphics.StrokeJoin.Round))
    }
}

private fun Modifier.clickableRole(onClick: () -> Unit): Modifier = clickableButton(onClick)

internal fun Modifier.clickableButton(onClick: () -> Unit): Modifier =
    this.clickable(role = androidx.compose.ui.semantics.Role.Button, onClick = onClick)

internal fun securityCards(state: ChatState, form: SettingsForm, ctx: AccountSettingsContext): List<SettingsCardSpec> {
    val prefs = state.preferences
    return buildList {
        if (prefs?.migrationStatus == "processing") add(SettingsCardSpec(SettingsTab.Security, "migration", null, "暗号化データの移行処理中です...") {
            val parts = prefs.migrationProgress.split('/')
            val done = parts.getOrNull(0)?.toIntOrNull() ?: 0
            val total = parts.getOrNull(1)?.toIntOrNull() ?: 0
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FaIcon(R.drawable.fa_solid_sync, null, size = 12.dp, tint = Tw.blue200)
                Text("暗号化データの移行処理中です...", fontSize = 12.sp, color = Tw.blue200)
            }
            if (parts.size == 2) Text("$done / $total", fontSize = 10.sp, color = Tw.blue200.copy(alpha = 0.8f), modifier = Modifier.padding(top = 4.dp))
            Box(Modifier.padding(top = 8.dp).fillMaxWidth().height(8.dp).clip(RoundedCornerShape(4.dp)).background(Tw.blue900.copy(alpha = 0.4f))) {
                Box(Modifier.fillMaxHeight().fillMaxWidth(if (total > 0) (done.toFloat() / total).coerceIn(0f, 1f) else 0f).background(Tw.blue400.copy(alpha = 0.8f)))
            }
        })
        add(SettingsCardSpec(SettingsTab.Security, "e2ee", null, "E2EE (End-to-End Encryption) チャット履歴とファイルを暗号化します。") {
            val web = LocalWebPalette.current
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("E2EE (End-to-End Encryption)", fontSize = 14.sp, fontWeight = FontWeight.Bold, color = if (web.isLight) web.text else Color.White)
                    Text("チャット履歴とファイルを暗号化します。", fontSize = 12.sp, color = if (web.isLight) Color(92, 103, 121) else Tw.gray500)
                }
                WebToggle(form.e2ee, { form.e2ee = it }, activeColor = Tw.green600)
            }
        })
        add(SettingsCardSpec(SettingsTab.Security, "scan", "暗号化スキャン", "暗号化スキャン 全体スキャン このスレッド") {
            var result by remember { mutableStateOf("未実行") }
            var samples by remember { mutableStateOf("") }
            fun scan(threadId: String?) {
                result = "スキャン中..."
                samples = ""
                ctx.ops.run("失敗しました") {
                    try {
                        val data = encryptionScan(threadId)
                        result = "Total: ${data.optInt("total")} / Encrypted: ${data.optInt("encrypted")} / Plain: ${data.optInt("unencrypted")}"
                        val rows = data.optJSONArray("samples")
                        if (rows != null && rows.length() > 0) samples = "例: " + (0 until minOf(8, rows.length())).joinToString(" / ") { i ->
                            val row = rows.getJSONObject(i)
                            val time = row.optString("timestamp").takeIf { it.isNotBlank() && it != "null" }?.let(::webLocaleTime).orEmpty()
                            "#${row.optInt("id")} (${row.optString("role")}) $time"
                        }
                    } catch (e: Exception) {
                        result = serverErrorText(e, "失敗しました")
                        if (e is kotlinx.coroutines.CancellationException) throw e
                    }
                }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                SettingsSmallButton("全体スキャン", { scan(null) })
                SettingsSmallButton("このスレッド", {
                    val thread = state.selected?.id
                    if (thread == null) ctx.notify("スレッドがありません") else scan(thread)
                })
            }
            Text(result, fontSize = 11.sp, color = if (LocalWebPalette.current.isLight) LocalWebPalette.current.text else Tw.gray300,
                modifier = Modifier.padding(top = 8.dp))
            if (samples.isNotBlank()) Text(samples, fontSize = 10.sp, color = Tw.gray400, modifier = Modifier.padding(top = 4.dp))
        })
        add(SettingsCardSpec(SettingsTab.Security, "sessions", "ログインセッション",
            "ログインセッション 現在ログイン中の端末を確認し、不要なセッションをログアウトできます。 更新 他のセッションをログアウト 全セッションを強制ログアウト") {
            SessionsCard(ctx)
        })
        if (prefs?.isAdmin == true) add(SettingsCardSpec(SettingsTab.Security, "delete", "アカウント削除", "アカウント削除") {
            Text("管理者アカウントのため、ここからはアカウント削除できません。", fontSize = 12.sp, color = Tw.gray400)
        }) else add(SettingsCardSpec(SettingsTab.Security, "delete", "アカウント削除", "アカウント削除 アカウントを完全に削除", danger = true) {
            val shape = RoundedCornerShape(4.dp)
            Text("アカウントを完全に削除", fontSize = 14.sp, color = Tw.red200,
                modifier = Modifier.fillMaxWidth().clip(shape).background(Tw.red800.copy(alpha = 0.5f)).border(1.dp, Tw.red600, shape)
                    .clickableButton {
                        ctx.confirm("本当にアカウントを削除しますか？\nこの操作は取り消せません。") {
                            ctx.ops.run("アカウントを削除できませんでした。時間をおいて再度お試しください。") {
                                deleteAccount()
                                ctx.model.signedOutRemotely()
                            }
                        }
                    }
                    .padding(vertical = 8.dp),
                textAlign = androidx.compose.ui.text.style.TextAlign.Center)
        })
    }
}

@Composable
private fun SessionsCard(ctx: AccountSettingsContext) {
    val web = LocalWebPalette.current
    var rows by remember { mutableStateOf<List<LoginSession>?>(null) }
    var failed by remember { mutableStateOf(false) }
    fun load() {
        failed = false
        rows = null
        ctx.ops.run("セッションの取得に失敗しました。") {
            try { rows = sessions().filterNot { it.revoked } } catch (e: Exception) {
                if (e is kotlinx.coroutines.CancellationException) throw e
                failed = true
            }
        }
    }
    LaunchedEffect(Unit) { load() }
    SettingsDesc("現在ログイン中の端末を確認し、不要なセッションをログアウトできます。")
    FlowRow(Modifier.padding(top = 12.dp), horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        SettingsSmallButton("更新", { load() })
        SettingsSmallButton("他のセッションをログアウト", {
            ctx.confirm("現在の端末以外をログアウトしますか？") { ctx.ops.run("操作に失敗しました") { revokeOtherSessions(); load() } }
        }, tone = SettingsButtonTone.Orange)
        SettingsSmallButton("全セッションを強制ログアウト", {
            ctx.confirm("全セッションを強制ログアウトします。よろしいですか？") {
                ctx.ops.run("操作に失敗しました") { revokeAllSessions(); ctx.model.signedOutRemotely() }
            }
        }, tone = SettingsButtonTone.Red)
    }
    Column(Modifier.padding(top = 12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        val list = rows
        when {
            failed -> Text("セッションの取得に失敗しました。", fontSize = 12.sp, color = Tw.red400)
            list == null -> Text("読み込み中...", fontSize = 12.sp, color = Tw.gray500)
            list.isEmpty() -> Text("アクティブなセッションはありません。", fontSize = 12.sp, color = Tw.gray500)
            else -> list.forEach { row ->
                val shape = RoundedCornerShape(4.dp)
                Row(
                    Modifier.fillMaxWidth().clip(shape).background(web.twBg(Tw.gray800).copy(alpha = 0.6f))
                        .border(1.dp, web.twBorder(Tw.gray700), shape).padding(12.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    Column(Modifier.weight(1f)) {
                        Row(Modifier.padding(bottom = 4.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            if (row.current) Text("現在", fontSize = 10.sp, color = Color.White,
                                modifier = Modifier.clip(RoundedCornerShape(4.dp)).background(web.theme.t600).padding(horizontal = 6.dp, vertical = 2.dp))
                            Text(row.ipAddress.ifBlank { "Unknown" }, fontSize = 12.sp, color = if (web.isLight) web.text else Tw.gray200)
                        }
                        Text(row.userAgent.ifBlank { "Unknown" }.take(120), fontSize = 11.sp, color = Tw.gray400, maxLines = 1, overflow = TextOverflow.Ellipsis)
                        Text("最終アクセス: ${webLocaleTime(row.lastSeenAt)} / 作成: ${webLocaleTime(row.createdAt)}", fontSize = 10.sp, color = Tw.gray500,
                            modifier = Modifier.padding(top = 4.dp))
                    }
                    if (!row.current) SettingsSmallButton("ログアウト", {
                        ctx.confirm("このセッションをログアウトしますか？") {
                            ctx.ops.run("ログアウトに失敗しました") {
                                if (revokeSession(row.id)) ctx.model.signedOutRemotely() else load()
                            }
                        }
                    }, fontSize = 11.sp)
                }
            }
        }
    }
}

internal fun twoFactorCards(state: ChatState, form: SettingsForm, ctx: AccountSettingsContext): List<SettingsCardSpec> {
    val security = state.security
    val model = ctx.model
    return listOf(
        SettingsCardSpec(SettingsTab.TwoFactor, "status", "Status", "Status Two-Factor Authentication ENABLED DISABLED Disable 2FA") {
            val web = LocalWebPalette.current
            val enabled = security?.is2faEnabled == true
            Row(Modifier.padding(bottom = 16.dp), verticalAlignment = Alignment.CenterVertically) {
                Text("Two-Factor Authentication", fontSize = 14.sp, color = if (web.isLight) web.text else Tw.gray300, modifier = Modifier.weight(1f))
                Text(if (enabled) "ENABLED" else "DISABLED", fontSize = 12.sp, fontWeight = FontWeight.Bold,
                    color = if (enabled) Color.White else Tw.gray400,
                    modifier = Modifier.clip(RoundedCornerShape(4.dp)).background(if (enabled) Tw.green600 else web.twBg(Tw.gray700))
                        .padding(horizontal = 8.dp, vertical = 2.dp))
            }
            if (enabled) Text("Disable 2FA", fontSize = 14.sp, fontWeight = FontWeight.Bold, color = Color.White,
                textAlign = androidx.compose.ui.text.style.TextAlign.Center,
                modifier = Modifier.fillMaxWidth().padding(bottom = 8.dp).clip(RoundedCornerShape(4.dp)).background(Tw.red600)
                    .clickableButton {
                        ctx.confirm("Disable 2FA?") {
                            ctx.ops.run("2FAの無効化に失敗しました") { disable2fa(); ctx.notify("2FAを無効化しました"); model.loadSecurity(); model.loadPreferences() }
                        }
                    }.padding(vertical = 8.dp))
            state.securityError?.let { SettingsDesc(it, color = Tw.red300) }
        },
        SettingsCardSpec(SettingsTab.TwoFactor, "totp", "Authenticator App (TOTP)", "Authenticator App (TOTP) Google Authenticator, Authy, etc. Setup TOTP Verify",
            titleIcon = R.drawable.fa_solid_mobile_alt) {
            var code by remember { mutableStateOf("") }
            val web = LocalWebPalette.current
            Text("Google Authenticator, Authy, etc.", fontSize = 12.sp, color = Tw.gray400, modifier = Modifier.padding(bottom = 12.dp))
            SettingsSmallButton("Setup TOTP", model::startTotpSetup, tone = SettingsButtonTone.Purple, enabled = !state.securityBusy, fill = true)
            state.securityTotpSecret?.let { secret ->
                Column(
                    Modifier.padding(top = 16.dp).fillMaxWidth().clip(RoundedCornerShape(4.dp)).background(web.twBg(Tw.gray800)).padding(16.dp),
                    horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    state.securityTotpQr?.let { qr ->
                        val bitmap = remember(qr) {
                            runCatching {
                                val bytes = Base64.decode(qr.substringAfter("base64,"), Base64.DEFAULT)
                                BitmapFactory.decodeByteArray(bytes, 0, bytes.size)?.asImageBitmap()
                            }.getOrNull()
                        }
                        bitmap?.let {
                            Image(it, "TOTP QR", Modifier.size(150.dp).border(4.dp, Color.White, RoundedCornerShape(4.dp)),
                                filterQuality = androidx.compose.ui.graphics.FilterQuality.None)
                        }
                    }
                    SelectionContainer { Text(secret, fontSize = 12.sp, fontFamily = FontFamily.Monospace, color = Tw.gray400) }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                        SettingsTextField(code, { code = it.filter(Char::isDigit).take(8) }, Modifier.weight(1f), placeholder = "000000", number = true)
                        SettingsSmallButton("Verify", { if (code.isNotBlank()) { model.enableTotp(code); code = "" } }, tone = SettingsButtonTone.Blue,
                            enabled = !state.securityBusy)
                    }
                }
            }
        },
        SettingsCardSpec(SettingsTab.TwoFactor, "passkey", "Security Key / Passkey",
            "Security Key / Passkey YubiKey, Touch ID, Windows Hello. 複数のパスキーを登録できます。 Register Key 登録済みパスキー",
            titleIcon = R.drawable.fa_solid_key) {
            val web = LocalWebPalette.current
            var name by remember { mutableStateOf("") }
            Text("YubiKey, Touch ID, Windows Hello. 複数のパスキーを登録できます。", fontSize = 12.sp, color = Tw.gray400,
                modifier = Modifier.padding(bottom = 12.dp))
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                SettingsTextField(name, { name = it.take(80) }, Modifier.fillMaxWidth(), placeholder = "パスキー名 (例: MacBook Touch ID)", fontSize = 12.sp)
                SettingsSmallButton("Register Key", { model.beginPasskeyRegistration(name); name = "" }, tone = SettingsButtonTone.Green,
                    enabled = !state.securityBusy, fill = true)
            }
            val keys = security?.passkeys.orEmpty()
            Text("登録済みパスキー: ${keys.size}", fontSize = 11.sp, color = Tw.gray400, modifier = Modifier.padding(top = 12.dp, bottom = 8.dp))
            if (keys.isEmpty()) Text("登録済みのパスキーはありません。", fontSize = 11.sp, color = Tw.gray500)
            keys.forEachIndexed { index, key ->
                val shape = RoundedCornerShape(4.dp)
                Row(
                    Modifier.fillMaxWidth().padding(bottom = 8.dp).clip(shape).background(web.twBg(Tw.gray800).copy(alpha = 0.6f))
                        .border(1.dp, web.twBorder(Tw.gray700), shape).padding(8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(Modifier.weight(1f)) {
                        Text(key.name.ifBlank { "Security Key ${index + 1}" }, fontSize = 12.sp, color = if (web.isLight) web.text else Tw.gray200,
                            maxLines = 1, overflow = TextOverflow.Ellipsis)
                        Text(if (key.createdAt.isNullOrBlank()) "登録日時: 不明" else "登録日時: ${webLocaleTime(key.createdAt)}", fontSize = 10.sp,
                            color = Tw.gray500, modifier = Modifier.padding(top = 4.dp))
                    }
                    SettingsSmallButton("削除", { model.removePasskey(key.id) }, tone = SettingsButtonTone.Red, fontSize = 10.sp,
                        enabled = key.id.isNotBlank())
                }
            }
        },
        SettingsCardSpec(SettingsTab.TwoFactor, "auth-prefs", "Authentication Preferences",
            "Authentication Preferences Google ログイン時に2FAをスキップ 既定の2要素認証方式 パスキーのみログインを有効化",
            titleIcon = R.drawable.fa_solid_user_shield) {
            val web = LocalWebPalette.current
            val labelColor = if (web.isLight) web.text else Tw.gray300
            Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Google ログイン時に2FAをスキップ", fontSize = 12.sp, color = labelColor, modifier = Modifier.weight(1f))
                    WebToggle(form.skip2faGoogle, { form.skip2faGoogle = it }, activeColor = Tw.emerald500)
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("既定の2要素認証方式", fontSize = 12.sp, color = labelColor, modifier = Modifier.weight(1f))
                    SettingsSelect(form.default2fa, webOptions("totp" to "Authenticator App (TOTP)", "webauthn" to "Security Key / Passkey"),
                        { form.default2fa = it }, fillWidth = false, fontSize = 12.sp)
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("パスキーのみログインを有効化", fontSize = 12.sp, color = labelColor, modifier = Modifier.weight(1f))
                    WebToggle(form.passkeyOnly, { form.passkeyOnly = it }, activeColor = Tw.emerald500)
                }
            }
            if (security?.passkeys.isNullOrEmpty()) SettingsDesc("パスキー未登録のため有効化できません。", Modifier.padding(top = 8.dp), color = Tw.red300)
        },
    )
}

private val IMPORT_CATEGORIES = listOf(
    "settings" to "設定", "api_credentials" to "APIキー・クラウド認証", "chats" to "チャット履歴", "gems" to "Gem",
    "files" to "アップロードファイル", "feedback" to "フィードバック", "diagnostics" to "診断メトリクス",
)

/** Web `#account-*` controls of the Data tab. */
internal class ImportFormState {
    var fileName by mutableStateOf<String?>(null)
    var fileUri by mutableStateOf<android.net.Uri?>(null)
    val categories = mutableStateMapOf<String, Boolean>().apply { IMPORT_CATEGORIES.forEach { put(it.first, true) } }
    var inplace by mutableStateOf(false)
    var settingsBypass by mutableStateOf(false)
}

internal fun accountDataCard(transfer: TransferUiState, importForm: ImportFormState, ctx: AccountSettingsContext): SettingsCardSpec =
    SettingsCardSpec(SettingsTab.Data, "account-data", "アカウントデータ",
        "アカウントデータ 設定、チャット、Gem、ファイルなどをZIPで保存・復元できます。 エクスポートZIPを作成 インポートするデータを選択 選択したデータをインポート 重複データの修復 重複データを確認して修復") {
        val web = LocalWebPalette.current
        val controller = ctx.model.accountTransfer
        val rule = web.twBorder(Tw.gray700)
        SettingsDesc("設定、チャット、Gem、ファイルなどをZIPで保存・復元できます。ユーザー名、パスワード、2FA、パスキー、ログイン連携・セッション、権限、BAN情報は対象外です。",
            Modifier.padding(bottom = 12.dp), color = Tw.gray400)
        SettingsSmallButton("エクスポートZIPを作成", controller::startExport, tone = SettingsButtonTone.Blue, enabled = !transfer.running,
            icon = R.drawable.fa_solid_file_export, fill = true)
        SettingsDesc("作成はバックグラウンドで続行します。ページを離れたりリロードしたりしても、完成後1時間はここからダウンロードできます。", Modifier.padding(top = 4.dp))
        SettingsDesc("ZIPにはAPIキー・クラウド認証情報が復号された状態で含まれます。安全な場所で保管してください。", Modifier.padding(top = 8.dp), color = Tw.amber300)
        transfer.exportReady?.let { ready ->
            val shape = RoundedCornerShape(4.dp)
            Column(Modifier.padding(top = 12.dp).fillMaxWidth().clip(shape).background(Tw.emerald950.copy(alpha = 0.2f))
                .border(1.dp, Tw.emerald500.copy(alpha = 0.4f), shape).padding(12.dp)) {
                val warning = if (ready.unreadable > 0) "（読取不能 ${ready.unreadable}件を復旧用として収録）" else ""
                Text("エクスポートZIPをダウンロードできます：${exportSizeLabel(ready.sizeBytes)}$warning", fontSize = 11.sp, color = Tw.emerald200)
                Text(if (ready.expiresAt.isNotBlank()) "保存期限：${webLocaleTime(ready.expiresAt)}（期限後に自動削除）" else "完成から1時間後に自動削除されます。",
                    fontSize = 10.sp, color = Tw.gray400, modifier = Modifier.padding(top = 4.dp))
                SettingsSmallButton(if (transfer.downloading) "ダウンロード中 ${exportSizeLabel(transfer.downloadedBytes)}" else "エクスポートZIPをダウンロード",
                    ctx.saveExport, Modifier.padding(top = 8.dp), tone = SettingsButtonTone.Emerald, enabled = !transfer.downloading,
                    icon = R.drawable.fa_solid_download, fill = true)
            }
        }
        Column(Modifier.padding(top = 16.dp).fillMaxWidth().settingsTopRule(rule), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                SettingsSmallButton("ファイルを選択", ctx.pickImportFile, fontSize = 12.sp)
                Text(importForm.fileName ?: "選択されていません", fontSize = 12.sp, color = if (web.isLight) web.text else Tw.gray300,
                    maxLines = 1, overflow = TextOverflow.Ellipsis)
            }
            Text("インポートするデータを選択", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = if (web.isLight) web.text else Tw.gray300)
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                IMPORT_CATEGORIES.forEach { (key, label) ->
                    SettingsCheck(label, importForm.categories[key] == true, { importForm.categories[key] = it }, fontSize = 12.sp, boxSize = 16.dp)
                }
            }
            SettingsCheck("元の場所へ復元（このアカウントのファイルを元のパスへ上書き。キー不整合で見えなくなったファイルの復旧用）", importForm.inplace,
                { importForm.inplace = it }, fontSize = 12.sp, boxSize = 16.dp)
            if (importForm.inplace) SettingsDesc("⚠ 既存の同名ファイルを上書きします。同じパスに別のファイルがある場合も置き換わります。", color = Tw.amber300)
            SettingsCheck("設定の確認をスキップ（「設定」をインポートするとき、変更内容の確認画面を表示せずにインポート）", importForm.settingsBypass,
                { importForm.settingsBypass = it }, fontSize = 12.sp, boxSize = 16.dp)
            SettingsDesc("既存データは削除しません。設定と認証情報は選択時のみ上書きし、それ以外は追加します。すでに同じ内容のデータが存在する場合は、重複を避けるためスキップされます。ユーザー名は変更されません。")
            SettingsSmallButton("選択したデータをインポート", {
                val uri = importForm.fileUri
                val selected = IMPORT_CATEGORIES.filter { importForm.categories[it.first] == true }
                when {
                    uri == null -> ctx.notify("インポートするZIPファイルを選択してください")
                    selected.isEmpty() -> ctx.notify("インポートするデータを1つ以上選択してください")
                    else -> ctx.confirm("次のデータをインポートします。既存データは削除されません。すでに同じ内容のデータがある場合はスキップされます。\n\n" +
                        selected.joinToString("、") { it.second } +
                        (if (importForm.inplace) "\n※「元の場所へ復元」: このアカウントの同名ファイルを上書きします" else "") + "\n\n続行しますか？") {
                        controller.import(uri, selected.map { it.first }, importForm.inplace, importForm.settingsBypass)
                    }
                }
            }, tone = SettingsButtonTone.Emerald, enabled = !transfer.running, icon = R.drawable.fa_solid_file_import, fill = true)
            transfer.progress?.let { progress -> TransferProgressBox(progress, transfer.running, controller::cancel) }
            transfer.importResult?.let { Text(it, fontSize = 11.sp, color = if (transfer.importResultError) Tw.red300 else Tw.emerald300) }
        }
        Column(Modifier.padding(top = 16.dp).fillMaxWidth().settingsTopRule(rule), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                FaIcon(R.drawable.fa_solid_trash, null, size = 12.dp, tint = Tw.amber300)
                Text("重複データの修復", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = if (web.isLight) web.text else Tw.gray300)
            }
            SettingsDesc("インポートの重複で作成されたチャット、Gem、ファイル、フィードバック、診断データを検出して削除します。同じ内容のデータは最も古い1件だけ残し、チャットやファイルへの参照は壊れないよう保護します。")
            SettingsSmallButton("重複データを確認して修復", controller::previewDedupe, tone = SettingsButtonTone.Orange, enabled = !transfer.dedupeBusy,
                icon = R.drawable.fa_solid_trash, fill = true)
            transfer.dedupeResult?.let { Text(it, fontSize = 11.sp, color = if (transfer.dedupeError) Tw.red300 else Tw.emerald300) }
        }
    }

@Composable
private fun TransferProgressBox(progress: com.minashin1120.aiplayground.data.TransferProgress, running: Boolean, onCancel: () -> Unit) {
    val shape = RoundedCornerShape(4.dp)
    Column(
        Modifier.fillMaxWidth().clip(shape).background(Tw.blue950.copy(alpha = 0.2f)).border(1.dp, Tw.blue500.copy(alpha = 0.4f), shape).padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(progress.message, fontSize = 11.sp, fontWeight = FontWeight.Bold, color = Tw.blue200, modifier = Modifier.weight(1f))
            Text("${progress.progress}%", fontSize = 10.sp, fontFamily = FontFamily.Monospace, color = Tw.blue300)
        }
        Box(Modifier.fillMaxWidth().height(8.dp).clip(RoundedCornerShape(4.dp)).background(LocalWebPalette.current.twBg(Tw.gray800))
            .border(1.dp, LocalWebPalette.current.twBorder(Tw.gray700), RoundedCornerShape(4.dp))) {
            Box(Modifier.fillMaxHeight().fillMaxWidth(progress.progress / 100f).background(Tw.blue500))
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(TRANSFER_PHASE_LABELS[progress.phase] ?: "処理状況を確認しています。", fontSize = 10.sp, color = Tw.gray400, modifier = Modifier.weight(1f))
            if (running && progress.phase !in setOf("ready", "completed", "failed", "cancelled", "expired")) {
                SettingsSmallButton("キャンセル", onCancel, tone = SettingsButtonTone.Red, fontSize = 10.sp)
            }
        }
    }
}
