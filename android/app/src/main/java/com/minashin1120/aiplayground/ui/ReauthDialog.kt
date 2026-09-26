package com.minashin1120.aiplayground.ui

import android.app.Activity
import android.content.Context
import android.content.ContextWrapper
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.AccountApi
import com.minashin1120.aiplayground.data.ApiException
import com.minashin1120.aiplayground.data.PasskeyClient
import com.minashin1120.aiplayground.data.ReauthOptions
import com.minashin1120.aiplayground.data.needsReauth
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.launch

/**
 * Runs settings operations against [AccountApi]. An answer of `reauth_required` parks the operation
 * in [reauthRetry]; [ReauthDialog] replays it after the user confirms their identity.
 */
@Stable
internal class AccountOps(
    private val scope: CoroutineScope,
    private val api: () -> AccountApi,
    private val notify: (String) -> Unit,
) {
    var reauthRetry by mutableStateOf<(() -> Unit)?>(null)

    fun run(fallbackError: String, op: suspend AccountApi.() -> Unit) {
        scope.launch {
            try {
                api().op()
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                if (e.needsReauth()) reauthRetry = { run(fallbackError, op) }
                else notify(serverErrorText(e, fallbackError))
            }
        }
    }
}

/** Web toasts show `data.error || fallback`. */
internal fun serverErrorText(e: Exception, fallback: String): String =
    (e as? ApiException)?.payload?.optString("error")?.ifBlank { null } ?: fallback

internal fun Context.hostActivity(): Activity? {
    var current: Context? = this
    while (current is ContextWrapper) {
        if (current is Activity) return current
        current = current.baseContext
    }
    return current as? Activity
}

/**
 * ANDROID_ONLY.md: before account deletion, password change, easy login, 2FA reset, revoking every
 * session and account archives, the app asks for the password, an authenticator code or a passkey.
 */
@Composable
internal fun ReauthDialog(api: () -> AccountApi, onDismiss: () -> Unit, onConfirmed: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var options by remember { mutableStateOf<ReauthOptions?>(null) }
    var password by remember { mutableStateOf("") }
    var code by remember { mutableStateOf("") }
    var error by remember { mutableStateOf<String?>(null) }
    var busy by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) {
        try {
            val loaded = api().reauthOptions()
            if (loaded.reauthenticated) onConfirmed() else options = loaded
        } catch (e: CancellationException) { throw e } catch (e: Exception) {
            error = serverErrorText(e, "本人確認を開始できませんでした。")
        }
    }
    fun attempt(block: suspend AccountApi.() -> Unit) {
        if (busy) return
        busy = true
        error = null
        scope.launch {
            try {
                api().block()
                onConfirmed()
            } catch (e: CancellationException) { throw e } catch (e: Exception) {
                error = if ((e as? ApiException)?.code == "invalid_credentials") "確認できませんでした。入力内容を確認してください。"
                    else serverErrorText(e, "確認できませんでした。")
            } finally { busy = false }
        }
    }
    val methods = options?.methods.orEmpty()
    PlaygroundDialog(
        onDismissRequest = onDismiss,
        title = { Text("本人確認") },
        icon = R.drawable.fa_solid_user_shield,
        subtitle = "この操作を続けるには本人確認が必要です",
        panelMaxWidth = 440.dp,
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                when {
                    options == null && error == null -> SettingsDesc("読み込み中...")
                    options?.signInAgain == true -> SettingsDesc("この端末でもう一度ログインしてから操作してください。")
                }
                if ("password" in methods) Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    SettingsFieldLabel("現在のパスワード")
                    SettingsTextField(password, { password = it.take(256) }, Modifier.fillMaxWidth(), password = true)
                    SettingsSmallButton("パスワードで確認", { attempt { reauthPassword(password) } }, enabled = password.isNotEmpty() && !busy,
                        tone = SettingsButtonTone.Blue, fill = true)
                }
                if ("totp" in methods) Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    SettingsFieldLabel("認証アプリのコード")
                    SettingsTextField(code, { code = it.filter(Char::isDigit).take(8) }, Modifier.fillMaxWidth(), placeholder = "000000", number = true)
                    SettingsSmallButton("コードで確認", { attempt { reauthTotp(code) } }, enabled = code.length >= 6 && !busy,
                        tone = SettingsButtonTone.Blue, fill = true)
                }
                if ("passkey" in methods) options?.passkeyJson?.let { json ->
                    SettingsSmallButton("パスキーで確認", {
                        attempt {
                            val response = PasskeyClient.get(context.hostActivity() ?: context, json)
                            reauthPasskey(response)
                        }
                    }, enabled = !busy, tone = SettingsButtonTone.Green, icon = R.drawable.fa_solid_key, fill = true)
                }
                error?.let { SettingsDesc(it, color = Tw.red300) }
            }
        },
        confirmButton = {
            WebButton(onClick = onDismiss, variant = WebButtonVariant.Ghost) { Text("キャンセル", fontSize = 13.sp) }
        },
    )
}
