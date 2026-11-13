package com.minashin1120.aiplayground

import com.minashin1120.aiplayground.data.backend.ChatBackend
import com.minashin1120.aiplayground.data.apiKeyInfoFor
import com.minashin1120.aiplayground.data.parsePreferences
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import org.json.JSONObject

/**
 * 送信の入口で使う確認（Xリンクの自動検索・APIキーの保存）だけを持つ協作者
 * （Web `applyXLinkAuto` / `#auto-search-banner` / `api-key-modal-save-btn`）。
 * 送信そのものは `send`、入力欄は `composer` へ委ねる。
 */
internal class ChatSendEntry(
    private val state: StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val scope: CoroutineScope,
    private val backend: () -> ChatBackend,
    private val token: () -> String,
    private val composer: ChatComposer,
    private val notify: (String) -> Unit,
    private val send: () -> Unit,
) {
    /** The original failure text of the API-key prompt (shown by `api-key-modal-cancel-btn`). */
    var apiKeyMissingMessage: String? = null

    /** Web `applyXLinkAuto`. */
    fun applyXLinkSearch() {
        mutable.update { it.copy(enableSearch = true) }
        if (state.value.model != "grok-4-fast-reasoning") composer.chooseModel("grok-4-fast-reasoning")
    }

    /** Web `#auto-search-banner`: 検索ONで続行 / 検索OFFで回答, optionally remembered (次回から尋ねない). */
    fun resolveXLinkPrompt(enable: Boolean, remember: Boolean) {
        mutable.update { it.copy(xLinkPrompt = false) }
        if (enable) {
            applyXLinkSearch()
            if (remember) scope.launch {
                runCatching {
                    val reply = backend().put("/api/mobile/v1/preferences", JSONObject().put("auto_search_on_links", true), token())
                    mutable.update { it.copy(preferences = parsePreferences(reply)) }
                }
            }
        }
        send()
    }

    /** Web `api-key-modal-save-btn`: saves the key for the model's provider, then sends again. */
    fun saveApiKeyAndResend(key: String) {
        val modelId = state.value.apiKeyPrompt ?: return
        val trimmed = key.trim()
        if (trimmed.isEmpty()) { notify("APIキーを入力してください"); return }
        val useFree = state.value.geminiFreeTier && state.value.preferences?.geminiBackend != "vertex_ai"
        val info = apiKeyInfoFor(modelId, useFree) ?: return
        scope.launch {
            try {
                val reply = backend().put("/api/mobile/v1/preferences", JSONObject().put(info.keyField, trimmed), token())
                mutable.update { it.copy(preferences = parsePreferences(reply), apiKeyPrompt = null) }
                send()
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) { notify("APIキーの保存に失敗しました") }
        }
    }

    /** `api-key-modal-cancel-btn` shows the original error; `api-key-modal-fallback-btn` leaves the model picker to the UI. */
    fun dismissApiKeyPrompt(showError: Boolean) {
        val modelId = state.value.apiKeyPrompt ?: return
        mutable.update { it.copy(apiKeyPrompt = null) }
        if (showError) notify(apiKeyMissingMessage ?: "${modelDisplayName(modelId)} のAPIキーが設定されていません")
    }

    fun modelDisplayName(modelId: String): String =
        state.value.account?.models?.firstOrNull { it.id == modelId }?.name ?: modelId
}
