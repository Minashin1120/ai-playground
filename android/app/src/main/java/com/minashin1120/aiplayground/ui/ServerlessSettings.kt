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
internal fun connectionCard(state: ChatState, model: ChatViewModel, ctx: AccountSettingsContext): SettingsCardSpec =
    SettingsCardSpec(SettingsTab.General, "connection", "接続",
        "接続 接続先 サーバー不使用モード サーバーを経由せずに送信 アカウントなし この端末 サーバーにログイン 同期 今すぐ同期 未送信 APIキーを取り込む") {
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
                    description = "サーバーを経由せず、端末からAIの各社APIへ直接送信します。履歴はサーバーから読み、回答が終わるとアカウントに保存します。APIキーは端末に設定したものを使います（APIキータブで設定）。",
                    checked = state.serverless,
                    onChange = model::setServerless,
                    modifier = Modifier.padding(top = 4.dp),
                )
                if (state.serverless) {
                    SettingsDesc("このモードでは、Batch、Realtime、Lyria、動画生成、MCP、ファイル作成、サーバーでのPython実行など、サーバーで処理する機能は使えません。")
                    SettingsFieldLabel("同期", Modifier.padding(top = 4.dp))
                    SettingsDesc("サーバーに接続できないときの回答は端末に残し、接続が戻ると自動でアカウントに保存します。")
                    SettingsDesc(listOfNotNull(
                        if (state.syncing) "同期しています…" else if (state.pendingCount > 0) "未送信 ${state.pendingCount}件" else "未送信はありません",
                        state.lastSyncAt?.let { "最終同期: " + formatSyncTime(it) },
                    ).joinToString("、"))
                    state.syncMessage?.let { SettingsDesc(it) }
                    SettingsSmallButton(if (state.syncing) "同期中…" else "今すぐ同期", model::syncNow, enabled = !state.syncing, fill = true)
                    if (state.serverInfo?.secretsExport != false) {
                        SettingsFieldLabel("APIキー", Modifier.padding(top = 4.dp))
                        SettingsDesc("このモードでは端末に保存したAPIキーを使います。サーバーに保存済みのキーを、本人確認のうえで端末へ取り込めます。")
                        SettingsSmallButton("サーバーのAPIキーを端末へ取り込む", {
                            ctx.ops.run("APIキーを取り込めませんでした") { model.importServerSecrets(exportSecrets()) }
                        }, fill = true)
                    }
                    if (state.localImportAvailable) {
                        SettingsFieldLabel("アカウントなしのチャット", Modifier.padding(top = 4.dp))
                        SettingsDesc("「サーバーを使わずに始める」で作成したチャットを、このアカウント（${state.account?.name.orEmpty()} / ${originLabel(ServerOrigin.current)}）へ取り込み、サーバーにアップロードします。")
                        SettingsSmallButton("端末のチャットをこのアカウントへ取り込む", {
                            ctx.confirm("アカウントなしで作成したチャットを「${state.account?.name.orEmpty()}」（${originLabel(ServerOrigin.current)}）へ取り込み、サーバーにアップロードしますか？") {
                                model.importLocalProfileChats()
                            }
                        }, fill = true)
                    }
                }
            }
        }
    }

private fun formatSyncTime(millis: Long): String =
    java.text.SimpleDateFormat("yyyy/MM/dd HH:mm", java.util.Locale.JAPAN).format(java.util.Date(millis))
