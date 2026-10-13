package com.minashin1120.aiplayground

import android.app.Application
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.runtime.getValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewModelScope
import com.minashin1120.aiplayground.data.ApiException
import com.minashin1120.aiplayground.data.SelectionTranslator
import com.minashin1120.aiplayground.data.direct.DirectApiException
import com.minashin1120.aiplayground.ui.PlaygroundTheme
import com.minashin1120.aiplayground.ui.SelectionTranslationDialog
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/** Asks [MainActivity] to open the settings at this card (the 「AIで翻訳」 model when none is chosen). */
const val EXTRA_OPEN_SETTINGS_CARD = "com.minashin1120.aiplayground.OPEN_SETTINGS_CARD"

/** Longest selection that is sent for translation. */
private const val MAX_TRANSLATION_CHARS = 20_000

data class SelectionTranslationState(
    val source: String = "",
    val setting: TranslationSetting? = null,
    val result: String = "",
    val running: Boolean = false,
    val error: String? = null,
    /** The error is fixed in the app (no model chosen, not signed in), so the modal offers to open it. */
    val openApp: Boolean = false,
)

class SelectionTranslationViewModel(application: Application) : AndroidViewModel(application) {
    private val mutable = MutableStateFlow(SelectionTranslationState())
    val state = mutable.asStateFlow()
    private var job: Job? = null

    /** Translates [text] once; a recreated activity (rotation) keeps the running translation. */
    fun start(text: String) {
        if (state.value.source.isNotEmpty()) return
        mutable.value = SelectionTranslationState(source = text)
        run()
    }

    fun retry() = run()

    private fun run() {
        job?.cancel()
        val setting = loadTranslationSetting(getApplication())
        mutable.update { it.copy(setting = setting, result = "", running = true, error = null, openApp = false) }
        job = viewModelScope.launch {
            try {
                val answer = SelectionTranslator(getApplication()).translate(state.value.source, setting) { piece ->
                    mutable.update { it.copy(result = it.result + piece) }
                }
                mutable.update { it.copy(result = answer.ifBlank { it.result }.trim(), running = false) }
            } catch (e: CancellationException) {
                throw e
            } catch (e: SelectionTranslator.Unavailable) {
                mutable.update { it.copy(running = false, error = e.message, openApp = e.openApp) }
            } catch (e: Exception) {
                val message = when (e) {
                    is SelectionTranslator.Failed -> e.message
                    is DirectApiException -> "API Error${if (e.status > 0) " (${e.status})" else ""}: ${e.message}"
                    is ApiException -> e.message
                    else -> "Connection Error: ${e.message ?: e.javaClass.simpleName}"
                }
                mutable.update { it.copy(running = false, error = message) }
            }
        }
    }
}

/**
 * ANDROID_ONLY.md 「AIで翻訳」 in the system text-selection menu (`Intent.ACTION_PROCESS_TEXT`) and share sheet
 * (`Intent.ACTION_SEND` of text): a translucent activity over the app the text came from, so that app stays open
 * behind the translation. The selection is never replaced.
 */
class TranslateOverlayActivity : ComponentActivity() {
    private val model: SelectionTranslationViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setResult(RESULT_CANCELED)
        val extra = if (intent?.action == Intent.ACTION_SEND) Intent.EXTRA_TEXT else Intent.EXTRA_PROCESS_TEXT
        val text = runCatching {
            intent?.getCharSequenceExtra(extra)?.toString()
        }.getOrNull()?.trim().orEmpty().take(MAX_TRANSLATION_CHARS)
        if (text.isEmpty() && model.state.value.source.isEmpty()) { finish(); return }
        model.start(text)
        setContent {
            PlaygroundTheme {
                val state by model.state.collectAsStateWithLifecycle()
                SelectionTranslationDialog(state, onDismiss = ::finish, onCopy = { copy(state.result) },
                    onRetry = model::retry, onOpenApp = { openApp(state.setting?.model.isNullOrBlank()) })
            }
        }
    }

    private fun copy(text: String) {
        if (text.isBlank()) return
        getSystemService(ClipboardManager::class.java)?.setPrimaryClip(ClipData.newPlainText("翻訳", text))
        // Android 13+ confirms the copy itself.
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) Toast.makeText(this, "コピーしました", Toast.LENGTH_SHORT).show()
    }

    private fun openApp(settings: Boolean) {
        startActivity(Intent(this, MainActivity::class.java).apply {
            if (settings) putExtra(EXTRA_OPEN_SETTINGS_CARD, "android")
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
        })
        finish()
    }
}
