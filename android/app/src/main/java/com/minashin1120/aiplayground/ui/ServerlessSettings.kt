package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.ChatViewModel
import com.minashin1120.aiplayground.data.ServerOrigin
import com.minashin1120.aiplayground.data.originLabel

/** Tabs that need a server account and are hidden in the no-account profile (ANDROID_ONLY.md). */
internal val LOCAL_PROFILE_HIDDEN_TABS = setOf(SettingsTab.Account, SettingsTab.Security, SettingsTab.TwoFactor, SettingsTab.Feedback, SettingsTab.Mcp)

/**
 * ANDROID_ONLY.md: the "接続" card after the Android card in the General tab. Shows the server (or the
 * no-account profile) and switches serverless mode of the signed-in account.
 */
internal fun connectionCard(state: ChatState, model: ChatViewModel): SettingsCardSpec =
    SettingsCardSpec(SettingsTab.General, "connection", "接続",
        "接続 接続先 サーバー不使用モード サーバーを経由せずに送信 アカウントなし この端末 サーバーにログイン") {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            if (state.localProfile) {
                SettingsFieldLabel("接続先")
                Text("この端末（サーバーを使用しない）", fontSize = 12.sp, color = settingsLabelColor())
                SettingsDesc("チャット・添付・APIキーは端末内に暗号化して保存し、AIの各社APIへ直接送信します。アカウント、セキュリティ、2要素認証、フィードバック、MCP、アカウントデータはサーバーにログインすると使えます。")
                SettingsSmallButton("サーバーにログイン", model::leaveLocalProfile, fill = true)
            } else {
                SettingsFieldLabel("接続先")
                Text(originLabel(ServerOrigin.current), fontSize = 12.sp, color = settingsLabelColor())
                SettingsSwitchRow(
                    title = "サーバー不使用モード",
                    description = "サーバーを経由せず、端末からAIの各社APIへ直接送信します。チャットは端末に保存され、APIキーは端末に設定したものを使います（APIキータブで設定）。",
                    checked = state.serverless,
                    onChange = model::setServerless,
                    modifier = Modifier.padding(top = 4.dp),
                )
                if (state.serverless) {
                    SettingsDesc("このモードでは、Batch、Realtime、Lyria、動画生成、MCP、ファイル作成、サーバーでのPython実行など、サーバーで処理する機能は使えません。")
                }
            }
        }
    }
