package com.minashin1120.aiplayground

import com.minashin1120.aiplayground.data.ApiException
import com.minashin1120.aiplayground.data.SettingsBubble
import com.minashin1120.aiplayground.data.backend.ChatBackend
import com.minashin1120.aiplayground.data.formatAiSettingValue
import com.minashin1120.aiplayground.data.isMistralOcrModel
import com.minashin1120.aiplayground.data.summarizeAiSettings
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject

/**
 * AI設定コマンド（Web `runAiSettingsCommand` / `aiSettingsConversation` /
 * `showPendingSlashCommandIndicator`）だけを持つ：`/settings <指示>` の会話（直近10件）と、
 * モデル経由の設定の確認・変更、結果の一時バブル。状態の更新は `mutable` へ行い、
 * 設定の再読込は生成時に受け取った関数へ委ねる。
 */
internal class ChatAiSettings(
    private val state: StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val scope: CoroutineScope,
    private val backend: () -> ChatBackend,
    private val token: () -> String,
    private val notify: (String) -> Unit,
    private val refreshPreferences: suspend () -> Unit,
) {
    /** Web `aiSettingsConversation` (kept for this app session like the Web sessionStorage). */
    private val conversation = mutableListOf<Pair<String, String>>()

    /** Web `showPendingSlashCommandIndicator` / `hidePendingSlashCommandIndicator` (leaving `/settings` clears its conversation). */
    fun setPendingSlashCommand(id: String?) {
        if (id == null && state.value.pendingSlashCommand == "settings") conversation.clear()
        mutable.update { it.copy(pendingSlashCommand = id) }
    }

    private fun append(role: String, content: String) {
        val text = content.trim()
        if (text.isEmpty()) return
        conversation += role to text.take(1600)
        while (conversation.size > 10) conversation.removeAt(0)
    }

    /** Web `runAiSettingsCommand`: `/settings <instruction>` changes or inspects settings through the selected model. */
    fun run(instruction: String) {
        val modelId = state.value.model
        if (modelId.isBlank()) { notify("モデルを選択してください"); return }
        if (isMistralOcrModel(modelId)) { notify("Mistral OCR は設定変更コマンドに使えません。チャットモデルを選んでください。"); return }
        if (state.value.pendingSlashCommand != "settings") mutable.update { it.copy(pendingSlashCommand = "settings") }
        val history = JSONArray(conversation.map { (role, content) -> JSONObject().put("role", role).put("content", content) })
        append("user", instruction)
        val stamp = System.currentTimeMillis()
        val pendingId = "settings-pending-$stamp"
        mutable.update { it.copy(draft = "", settingsBubbles = it.settingsBubbles +
            SettingsBubble("settings-user-$stamp", "user", "/settings $instruction") +
            SettingsBubble(pendingId, "assistant", "", modelId, pending = true)) }
        scope.launch {
            fun finish(bubble: SettingsBubble?) = mutable.update { current ->
                current.copy(settingsBubbles = current.settingsBubbles.filterNot { b -> b.id == pendingId } + listOfNotNull(bubble))
            }
            try {
                val data = backend().post("/api/settings/apply-ai-prompt", JSONObject().put("prompt", instruction).put("model", modelId)
                    .put("conversation", history), token())
                val inspect = data.optString("mode") == "inspect"
                val values = if (inspect) data.optJSONObject("current") else data.optJSONObject("applied")
                if (data.optString("status") == "ok" && values != null) {
                    append("assistant", summarizeAiSettings(values, inspect))
                    val count = values.length()
                    notify(if (inspect) "現在の設定を確認しました（${count}項目）" else "設定を更新しました（${count}項目）")
                    if (!inspect) runCatching { refreshPreferences() }
                    val entries = values.keys().asSequence().map { key -> key to formatAiSettingValue(values.opt(key)) }.toList()
                    val text = when {
                        entries.isNotEmpty() && inspect -> "現在の設定を確認しました。\n\n確認した項目をタップすると、設定画面の該当箇所へ移動できます。"
                        entries.isNotEmpty() -> "設定を更新しました。\n\n変更した項目をタップすると、設定画面の該当箇所へ移動できます。"
                        inspect -> "確認できる設定項目がありませんでした。"
                        else -> "変更された設定項目はありませんでした。"
                    }
                    finish(SettingsBubble("settings-result-${System.currentTimeMillis()}", "assistant", text, modelId, entries))
                    return@launch
                }
                val message = data.optString("message").ifBlank { data.optString("error") }.ifBlank { "設定変更に失敗しました" }
                append("assistant", "設定操作に失敗しました: $message")
                finish(SettingsBubble("settings-error-${System.currentTimeMillis()}", "assistant", "設定変更に失敗しました。\n\n$message", modelId))
                notify(message)
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                val message = (e as? ApiException)?.payload?.let { p -> p.optString("message").ifBlank { p.optString("error") } }
                if (!message.isNullOrBlank()) {
                    append("assistant", "設定操作に失敗しました: $message")
                    finish(SettingsBubble("settings-error-${System.currentTimeMillis()}", "assistant", "設定変更に失敗しました。\n\n$message", modelId))
                    notify(message)
                } else {
                    append("assistant", "設定操作の通信に失敗しました。")
                    finish(SettingsBubble("settings-error-${System.currentTimeMillis()}", "assistant",
                        "設定変更の通信に失敗しました。時間をおいて再度お試しください。", modelId))
                    notify("設定変更の通信に失敗しました")
                }
            }
        }
    }
}
