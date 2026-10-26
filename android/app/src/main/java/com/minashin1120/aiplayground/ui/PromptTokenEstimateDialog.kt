package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.data.ActivityLog
import kotlinx.coroutines.CancellationException
import org.json.JSONObject

internal data class PromptTokenEstimateSnapshot(
    val model: String, val message: String, val quote: String, val files: List<String>, val estimate: JSONObject,
)

internal val LocalComposerDetailsEstimator = staticCompositionLocalOf<(suspend (String, String, String, List<String>) -> JSONObject)?> { null }

@Composable
internal fun PromptTokenEstimateDialog(snapshot: PromptTokenEstimateSnapshot, onDismiss: () -> Unit) {
    val estimator = LocalComposerDetailsEstimator.current
    var data by remember(snapshot) { mutableStateOf(snapshot.estimate) }
    var loading by remember(snapshot) { mutableStateOf(true) }
    var failed by remember(snapshot) { mutableStateOf(false) }
    var retry by remember(snapshot) { mutableIntStateOf(0) }
    DisposableEffect(Unit) {
        ActivityLog.log("ui.show", "token_modal" to true)
        onDispose { ActivityLog.log("ui.hide", "token_modal" to false) }
    }
    LaunchedEffect(snapshot, retry) {
        loading = true
        failed = false
        try {
            if (estimator != null) {
                val reply = estimator(snapshot.model, snapshot.message, snapshot.quote, snapshot.files)
                if (!reply.optBoolean("countable") || (snapshot.estimate.optString("estimate_method") == "provider" &&
                    reply.optString("estimate_method") != "provider")) throw IllegalStateException("Unavailable")
                data = reply
            }
        } catch (e: CancellationException) { throw e }
        catch (e: Exception) { failed = true }
        finally { loading = false }
    }
    val web = LocalWebPalette.current
    SmallInfoModal(onDismiss) {
        Column(Modifier.heightIn(max = 560.dp).verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("入力トークン見積の詳細", modifier = Modifier.padding(end = 24.dp), color = Tw.blue300,
                fontSize = 14.sp, lineHeight = 20.sp, fontWeight = FontWeight.Bold)
            EstimateNote(snapshot.model, accent = true)
            EstimateNote("開いた時点の入力")
            EstimateRow("合計", data.opt("tokens_total"))
            EstimateRow("本文・引用", data.opt("tokens_prompt"))
            EstimateRow("添付ファイル合計", data.opt("tokens_files"))
            EstimateNote("計測方法: ${if (data.optString("estimate_method") == "provider") "プロバイダー計測" else "概算"}")
            EstimateNote("対象: 入力欄の本文・引用・添付（履歴・SysPrompt・ツール除外）")
            EstimateNote("添付 ${data.optInt("files_total")}件 / 計測済み ${data.optInt("files_counted")}件 / 未計測 ${data.optInt("files_non_text")}件 / 未検出 ${data.optInt("files_missing")}件 / 失敗 ${data.optInt("files_error")}件")
            val detail = data.optJSONObject("breakdown")
            if (detail != null) {
                Text("内訳", color = web.twText(Tw.gray200), fontSize = 12.sp, fontWeight = FontWeight.Bold)
                EstimateRow("本文（形式情報を含む場合あり）", detail.opt("message_tokens"))
                EstimateRow("引用（区切りを含む）", detail.opt("quote_tokens"))
                val files = detail.optJSONArray("files")
                if (files != null) for (index in 0 until files.length()) {
                    val file = files.optJSONObject(index) ?: continue
                    EstimateRow("添付 ${index + 1}: ${file.optString("name").ifEmpty { "ファイル" }}", file.opt("tokens"))
                    val status = when (file.optString("reason")) {
                        "ok" -> "計測済み"
                        "missing" -> "未検出"
                        "error" -> "計測失敗"
                        "read_error" -> "読み込み失敗"
                        "unseparated" -> "分割未計測"
                        "no_text" -> "テキストなし"
                        else -> "未計測"
                    }
                    EstimateNote("${if (file.optString("method") == "provider") "プロバイダー計測" else "概算"} / $status")
                }
                EstimateRow("結合時の差・分割未計測分", detail.opt("adjustment_tokens"))
                EstimateNote("内訳は本文、引用、添付を順に加えたときの増分です。結合時の差や分割できない分は別に表示します。")
            }
            if (loading) EstimateNote("詳細な内訳を計算中...")
            if (failed) {
                Text("詳細な内訳を取得できませんでした。", color = Tw.red300, fontSize = 12.sp)
                Text("再試行", color = Tw.blue300, fontSize = 12.sp,
                    modifier = Modifier.clickable(role = Role.Button, onClickLabel = "再試行") { retry++ }.padding(vertical = 8.dp))
            }
            if (!loading && !failed && detail == null) EstimateNote("この接続先では詳細な内訳を取得できません。")
            EstimateNote("送信後に返る使用量や請求額の確定値とは異なります。")
        }
    }
}

@Composable
private fun EstimateNote(text: String, accent: Boolean = false) {
    val web = LocalWebPalette.current
    Text(text, color = if (accent) Tw.blue300 else web.twText(Tw.gray400), fontSize = 12.sp, lineHeight = 18.sp)
}

@Composable
private fun EstimateRow(label: String, value: Any?) {
    val web = LocalWebPalette.current
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        Text(label, color = web.twText(Tw.gray300), fontSize = 12.sp, lineHeight = 18.sp, modifier = Modifier.weight(1f))
        Text(if (value is Number) "$value tokens" else "未計測", color = web.twText(Tw.gray300),
            fontSize = 12.sp, lineHeight = 18.sp, fontFamily = FontFamily.Monospace)
    }
}
