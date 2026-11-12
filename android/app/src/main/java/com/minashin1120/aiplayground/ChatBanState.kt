package com.minashin1120.aiplayground

import com.minashin1120.aiplayground.data.ApiException
import com.minashin1120.aiplayground.data.BanAppealState
import com.minashin1120.aiplayground.data.LyriaState
import com.minashin1120.aiplayground.data.RealtimeState
import com.minashin1120.aiplayground.data.backend.ChatBackend
import com.minashin1120.aiplayground.data.nullableString
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import org.json.JSONObject

/**
 * The BAN boundary of the Web server: the account work stopped in the same order, the reason/date
 * screen (`/banned`) and the appeal form. State lives in the view model; this class only reads it
 * and reports through `mutable`.
 */
internal class ChatBanState(
    private val state: StateFlow<ChatState>,
    private val mutable: MutableStateFlow<ChatState>,
    private val scope: CoroutineScope,
    private val backend: () -> ChatBackend,
    private val token: () -> String,
    private val stopAccountWork: () -> Unit,
    private val reloadAccount: suspend () -> Unit,
    private val report: suspend (Throwable) -> Unit,
) {
    /** Stops account work at the same BAN boundary enforced by the Web server. */
    fun enter(error: ApiException) {
        stopAccountWork()
        val alreadyBanned = state.value.banned
        mutable.update { current -> current.copy(
            banned = true,
            banReason = error.payload.optString("reason").ifBlank { error.payload.optString("message") },
            banAt = error.payload.optString("banned_at"),
            pairing = false,
            busy = false,
            uploading = false,
            streaming = false,
            realtime = RealtimeState(),
            lyria = LyriaState(),
            jobId = null,
            retryAvailable = false,
            mcpDecision = null,
            status = "",
        ) }
        if (!alreadyBanned) loadDetails()
    }

    /** Web `/banned`: the reason, date and appeal box come from `/api/mobile/v1/ban`. */
    private fun loadDetails() { scope.launch {
        try { applyDetails(backend().get("/api/mobile/v1/ban", token())) }
        catch (e: CancellationException) { throw e }
        catch (_: Exception) { /* Keep the reason from the BAN response; the form still works. */ }
    } }

    private fun applyDetails(json: JSONObject, submitted: Boolean = false, error: String = "") {
        if (!json.optBoolean("banned", true)) {
            // Web `/banned` redirects to the chat once the BAN is lifted.
            mutable.update { it.copy(banned = false, banReason = "", banAt = "", banAppeal = BanAppealState()) }
            scope.launch { runCatching { reloadAccount() }.onFailure { report(it) } }
            return
        }
        val latest = json.optJSONObject("latest_appeal")
        mutable.update { current -> current.copy(
            banReason = json.nullableString("reason").ifBlank { current.banReason },
            banAt = json.nullableString("banned_at").ifBlank { current.banAt },
            banAppeal = BanAppealState(
                latestAt = latest?.nullableString("created_at").orEmpty(),
                latestStatus = latest?.nullableString("status").orEmpty(),
                adminReply = latest?.nullableString("admin_reply").orEmpty(),
                blocked = json.optBoolean("appeal_blocked"),
                blockReason = json.nullableString("appeal_block_reason"),
                submitted = submitted,
                error = error,
            ),
        ) }
    }

    /** Web `submit_ban_appeal`; returns true when the appeal was stored (the form is cleared). */
    suspend fun submitAppeal(message: String): Boolean {
        if (!state.value.banned || state.value.banAppeal.sending) return false
        mutable.update { it.copy(banAppeal = it.banAppeal.copy(sending = true, submitted = false, error = "")) }
        return try {
            applyDetails(backend().post("/api/mobile/v1/ban/appeal", JSONObject().put("message", message), token()), submitted = true)
            true
        } catch (e: CancellationException) {
            mutable.update { it.copy(banAppeal = it.banAppeal.copy(sending = false)) }
            throw e
        } catch (e: Exception) {
            val payload = (e as? ApiException)?.payload
            when {
                e is ApiException && e.code == "not_banned" -> applyDetails(JSONObject().put("banned", false))
                payload != null && payload.has("banned") ->
                    applyDetails(payload, error = payload.optString("message").ifBlank { e.message.orEmpty() })
                else -> mutable.update { it.copy(banAppeal = it.banAppeal.copy(sending = false,
                    error = (e as? ApiException)?.message ?: "異議申し立てを送信できませんでした。通信状態を確認してください。")) }
            }
            false
        }
    }
}
