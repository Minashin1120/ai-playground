package com.minashin1120.aiplayground

import android.app.Application
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Parcelable
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
import com.minashin1120.aiplayground.data.loadTranslationImage
import com.minashin1120.aiplayground.ui.PlaygroundTheme
import com.minashin1120.aiplayground.ui.SelectionTranslationDialog
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** Asks [MainActivity] to open the settings at this card (the 「AIで翻訳」 model when none is chosen). */
const val EXTRA_OPEN_SETTINGS_CARD = "com.minashin1120.aiplayground.OPEN_SETTINGS_CARD"

/** Longest selection that is sent for translation. */
private const val MAX_TRANSLATION_CHARS = 20_000

data class SelectionTranslationState(
    val source: String = "",
    /** Images shared to be translated (the text in them), once read; [imageCount] were shared. */
    val images: List<SelectionTranslator.Image> = emptyList(),
    val imageCount: Int = 0,
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
    private var uris: List<Uri> = emptyList()
    var started = false
        private set

    /**
     * Translates [text], or the text in the images at [uris] when there are any, once; a recreated activity
     * (rotation) keeps the running translation. The images are read while the activity holds the share's
     * read permission.
     */
    fun start(text: String, uris: List<Uri> = emptyList()) {
        if (started) return
        started = true
        this.uris = uris.take(SelectionTranslator.MAX_IMAGES)
        mutable.value = SelectionTranslationState(source = if (uris.isEmpty()) text else "", imageCount = this.uris.size)
        run()
    }

    fun retry() = run()

    private fun run() {
        job?.cancel()
        val setting = loadTranslationSetting(getApplication())
        mutable.update { it.copy(setting = setting, result = "", running = true, error = null, openApp = false) }
        job = viewModelScope.launch {
            try {
                if (uris.isNotEmpty() && state.value.images.isEmpty()) {
                    val images = try {
                        withContext(Dispatchers.IO) {
                            uris.mapIndexedNotNull { index, uri -> loadTranslationImage(getApplication(), uri, index, setting.compression) }
                        }
                    } catch (e: java.io.IOException) {
                        throw SelectionTranslator.Unavailable(e.message ?: "画像を読み込めませんでした。", openApp = false)
                    }
                    if (images.isEmpty()) throw SelectionTranslator.Unavailable("画像を読み込めませんでした。", openApp = false)
                    mutable.update { it.copy(images = images) }
                }
                val current = state.value
                val answer = SelectionTranslator(getApplication()).translate(current.source, current.images, setting) { piece ->
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
 * (`Intent.ACTION_SEND` of text, `ACTION_SEND`/`ACTION_SEND_MULTIPLE` of images): a translucent activity over the
 * app the text came from, so that app stays open behind the translation. The selection is never replaced.
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
        val images = runCatching { sharedImages(intent) }.getOrDefault(emptyList())
        if (text.isEmpty() && images.isEmpty() && !model.started) { finish(); return }
        model.start(text, images)
        setContent {
            PlaygroundTheme {
                val state by model.state.collectAsStateWithLifecycle()
                SelectionTranslationDialog(state, onDismiss = ::finish, onCopy = { copy(state.result) },
                    onRetry = model::retry, onOpenApp = { openApp(state.setting?.model.isNullOrBlank()) })
            }
        }
    }

    /** Images shared to 「AIで翻訳」 (the share sheet offers it only for images). */
    @Suppress("DEPRECATION")
    private fun sharedImages(intent: Intent?): List<Uri> {
        if (intent?.type?.startsWith("image/") != true) return emptyList()
        val streams: List<Parcelable?> = when (intent.action) {
            Intent.ACTION_SEND -> listOf(intent.getParcelableExtra<Parcelable>(Intent.EXTRA_STREAM))
            Intent.ACTION_SEND_MULTIPLE -> intent.getParcelableArrayListExtra<Parcelable>(Intent.EXTRA_STREAM).orEmpty()
            else -> emptyList()
        }
        return streams.filterIsInstance<Uri>()
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
