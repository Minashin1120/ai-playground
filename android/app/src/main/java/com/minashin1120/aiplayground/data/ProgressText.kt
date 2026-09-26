package com.minashin1120.aiplayground.data

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.concurrent.atomic.AtomicLong

/** Web `progress_spinner.js`: the shared request spinner's wording. */
object ProgressText {
    const val DEFAULT = "通信中..."

    /** `PHASE_LABELS`. */
    val PHASE_LABELS = mapOf(
        "communicating" to DEFAULT,
        "sending" to "送信中...",
        "waiting" to "モデルの応答待機中...",
        "receiving" to "受信中...",
        "reconnecting" to "再接続中...",
        "saving" to "保存中...",
        "uploading" to "アップロード中...",
        "generating" to "生成中...",
    )

    /** `FLOW_INITIAL_PHASES`. */
    val FLOW_INITIAL_PHASES = mapOf("chat" to "sending", "chatResume" to "reconnecting")

    private val PASSIVE = Regex("""(?:/api/version(?:[/?]|$)|/api/(?:debug|metrics)(?:[/?]|$)|/api/bot-telemetry(?:[/?]|$)|/api/temporary_chat/heartbeat(?:[/?]|$))""", RegexOption.IGNORE_CASE)

    fun phase(key: String): String = PHASE_LABELS[key] ?: DEFAULT

    /** `PASSIVE_REQUEST_RE`: heartbeats and telemetry never show the spinner. */
    fun isPassive(path: String): Boolean = PASSIVE.containsMatchIn(path)

    /** `inferSpinnerTextFromUrl`: the label for a request, classified by HTTP method first. */
    fun forRequest(method: String, path: String): String {
        val m = method.uppercase()
        val combined = "$m ${path.lowercase()}"
        return when {
            m == "GET" || m == "HEAD" -> "読み込み中..."
            Regex("delete|remove").containsMatchIn(combined) || m == "DELETE" -> "削除中..."
            Regex("message|chat|prompt|stream|reply").containsMatchIn(combined) -> "送信中..."
            Regex("setting|config|preference").containsMatchIn(combined) -> "保存中..."
            Regex("login|signin").containsMatchIn(combined) -> "ログイン中..."
            Regex("verify|2fa|totp|webauthn|passkey").containsMatchIn(combined) -> "認証中..."
            Regex("upload|attachment|/photo").containsMatchIn(combined) -> "アップロード中..."
            Regex("generate|image|imagine").containsMatchIn(combined) -> "生成中..."
            Regex("save|update").containsMatchIn(combined) || m == "PUT" || m == "PATCH" -> "保存中..."
            m == "POST" -> "送信中..."
            else -> DEFAULT
        }
    }
}

/**
 * The Web global spinner's operation list: every tracked request adds an operation and the newest
 * operation's label is shown; the spinner hides when none remain. [label] is null while idle.
 */
class ProgressTracker {
    inner class Operation internal constructor(val id: Long, label: String) {
        @Volatile var label: String = label
            private set
        @Volatile private var finished = false

        fun setPhase(phase: String) = setLabel(ProgressText.phase(phase))

        fun setLabel(value: String) {
            if (finished) return
            label = value.ifBlank { ProgressText.DEFAULT }
            publish()
        }

        /** Safe to call more than once, like the Web finish function. */
        fun finish() {
            if (finished) return
            finished = true
            synchronized(operations) { operations.remove(id) }
            publish()
        }
    }

    private val nextId = AtomicLong(1)
    private val operations = LinkedHashMap<Long, Operation>()
    private val mutableLabel = MutableStateFlow<String?>(null)
    val label: StateFlow<String?> = mutableLabel.asStateFlow()

    fun start(label: String): Operation {
        val operation = Operation(nextId.getAndIncrement(), label.ifBlank { ProgressText.DEFAULT })
        synchronized(operations) { operations[operation.id] = operation }
        publish()
        return operation
    }

    /** `startFlow(name)`: a manually driven operation starting at the flow's first phase. */
    fun startFlow(name: String): Operation = start(ProgressText.phase(ProgressText.FLOW_INITIAL_PHASES[name] ?: "communicating"))

    private fun publish() {
        mutableLabel.value = synchronized(operations) { operations.values.maxByOrNull { it.id }?.label }
    }
}
