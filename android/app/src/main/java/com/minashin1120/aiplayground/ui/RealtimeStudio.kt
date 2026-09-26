package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.selection.toggleable
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.withStyle
import com.minashin1120.aiplayground.R
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.ChatViewModel
import com.minashin1120.aiplayground.data.ModelInfo
import com.minashin1120.aiplayground.data.RealtimeState

private const val REALTIME_AUDIO_MODE = "realtime_audio"
private const val REALTIME_TRANSCRIPTION_ONLY_MODEL = "gpt-realtime-whisper"

internal fun isRealtimeAudioModel(model: ModelInfo?): Boolean =
    model != null && model.mode == REALTIME_AUDIO_MODE && model.selectable &&
        model.id != REALTIME_TRANSCRIPTION_ONLY_MODEL

internal fun realtimeModels(state: ChatState): List<Pair<String, String>> =
    state.account?.models.orEmpty()
        .filter(::isRealtimeAudioModel)
        .map { it.id to it.name }

private fun isGeminiLive(model: String): Boolean = model.startsWith("gemini-3.1-flash-live") ||
    model == "gemini-3.8-live" || model == "gemini-3.8-live-extended-thinking" ||
    model == "gemini-3.5-live-translate-preview" || model == "gemini-3.5-transcribe-live"

private fun isGeminiExtendedThinking(model: String): Boolean = model == "gemini-3.8-live-extended-thinking"

private fun isGrokLiveTranscribe(model: String): Boolean = model == "grok-voice-transcribe-2.0"

// Same voice lists as the Web STS panel (chat_core part05: *_STS_VOICES).
internal val OPENAI_REALTIME_VOICES = listOf("alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse", "marin", "cedar")
internal val GROK_REALTIME_VOICES = listOf("Ara", "Rex", "Sal", "Eve", "Leo")
internal val GEMINI_REALTIME_VOICES = listOf(
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe",
    "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
    "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi", "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
)
internal val REALTIME_TARGET_LANGUAGES = listOf(
    "ja" to "日本語", "en" to "English", "ko" to "한국어", "zh-CN" to "中文（簡体）", "zh-TW" to "中文（繁体）",
    "fr" to "Français", "de" to "Deutsch", "es" to "Español", "it" to "Italiano", "pt" to "Português",
    "ru" to "Русский", "hi" to "हिन्दी", "th" to "ไทย", "vi" to "Tiếng Việt", "id" to "Bahasa Indonesia", "ar" to "العربية",
)

/** Settings shared by the composer voice dock and the enlarged studio dialog (Web `#sts-settings`). */
data class RealtimeOptions(
    val voice: String = "",
    val targetLanguage: String = "ja",
    val thinkingLevel: String = "",
    val transcriptionMode: String = "VERBATIM",
    val customVocabulary: String = "",
    /** Web `#sts-include-thoughts` (Thoughts). */
    val includeThoughts: Boolean = false,
    /** `#sts-auto-play`: play the reply audio. */
    val autoPlay: Boolean = true,
    /** `#sts-auto-restart` / `#sts-auto-send` / `#sts-silence-sec` (turn-by-turn recording on the Web). */
    val autoRestart: Boolean = true,
    val autoSend: Boolean = false,
    val silenceSeconds: String = "2.5",
    /** `#sts-speed` (OpenAI Realtime, 0.25–1.5). */
    val speed: Float = 1f,
    /** `#sts-rate-in` / `#sts-rate-out` (xAI PCM sample rates). */
    val rateIn: Int = 24000,
    val rateOut: Int = 24000,
)

/** Web `GROK_PCM_RATES`. */
internal val GROK_PCM_RATES = listOf(8000, 16000, 21050, 24000, 32000, 44100, 48000)

internal fun isRealtimeTranscription(model: String): Boolean =
    model == "gemini-3.5-transcribe-live" || isGrokLiveTranscribe(model) || model.contains("transcribe")

/** Web `getStsProvider`. */
internal fun realtimeProvider(model: String): String = when {
    model.contains("gpt-realtime") || model.startsWith("gpt-") -> "openai"
    model.contains("grok") -> "xai"
    else -> "gemini"
}

internal fun realtimeVoices(model: String): List<String> = when {
    isRealtimeTranscription(model) || model == "gemini-3.5-live-translate-preview" -> emptyList()
    model.contains("gpt-realtime") -> OPENAI_REALTIME_VOICES
    model.contains("grok-voice") -> GROK_REALTIME_VOICES
    isGeminiLive(model) || model.startsWith("gemini-") -> GEMINI_REALTIME_VOICES
    else -> emptyList()
}

/** Thinking levels the Web panel offers for this model; empty when it has no Thinking control. */
internal fun realtimeThinkingLevels(model: String): List<String> = when {
    isGeminiExtendedThinking(model) -> listOf("low", "medium", "high")
    model == "gemini-3.8-live" || model == "gemini-3.5-live-translate-preview" || isRealtimeTranscription(model) -> emptyList()
    model.startsWith("gemini-") -> listOf("minimal", "low", "medium", "high")
    else -> emptyList()
}

/** Web `setSelectOptions(voiceSel, …, value || default)`: alloy, Ara or Kore until a voice is chosen. */
internal fun resolvedRealtimeVoice(model: String, options: RealtimeOptions): String {
    val voices = realtimeVoices(model)
    val fallback = when (realtimeProvider(model)) { "openai" -> "alloy"; "xai" -> "Ara"; else -> "Kore" }
    return voices.firstOrNull { it == options.voice } ?: fallback.takeIf { it in voices } ?: voices.firstOrNull() ?: "Kore"
}

internal fun resolvedRealtimeThinking(model: String, options: RealtimeOptions): String {
    val levels = realtimeThinkingLevels(model)
    return when {
        options.thinkingLevel in levels -> options.thinkingLevel
        isGeminiExtendedThinking(model) -> "medium"
        else -> levels.firstOrNull() ?: "minimal"
    }
}

internal fun startRealtimeWith(model: ChatViewModel, modelId: String, options: RealtimeOptions) {
    val provider = realtimeProvider(modelId)
    model.startRealtime(modelId, resolvedRealtimeVoice(modelId, options), options.targetLanguage,
        resolvedRealtimeThinking(modelId, options), options.transcriptionMode, options.customVocabulary,
        includeThoughts = realtimeThinkingLevels(modelId).isNotEmpty() && options.includeThoughts,
        speed = if (provider == "openai" && !isRealtimeTranscription(modelId)) options.speed else null,
        rateIn = if (provider == "xai" && !isRealtimeTranscription(modelId)) options.rateIn else null,
        rateOut = if (provider == "xai" && !isRealtimeTranscription(modelId)) options.rateOut else null,
        autoPlay = options.autoPlay)
}

/** Web `#sts-mode-label`. */
internal fun realtimeModeLabel(model: String): String = when {
    isRealtimeTranscription(model) -> "Realtime Speech-to-Text"
    model == "gemini-3.5-live-translate-preview" -> "Realtime Translation"
    else -> "Speech-to-Speech Live"
}

/** Web `#sts-note`. */
internal fun realtimeNote(model: String): String = when {
    model == "gemini-3.5-transcribe-live" -> "リアルタイム低遅延文字起こし（16kHz PCM / 最大10分）"
    isGrokLiveTranscribe(model) -> "xAI ストリーミング文字起こし（16kHz PCM）"
    model == "gpt-live-transcribe" -> "低遅延ライブ文字起こし（24kHz PCM）"
    isRealtimeTranscription(model) -> "高精度なコミット単位の文字起こし（24kHz PCM）"
    model == "gemini-3.5-live-translate-preview" -> "70以上の言語に対応するリアルタイム音声翻訳（Think非対応・音声選択不可）"
    model == "gemini-3.8-live" -> "Gemini 3.8 Flash Liveは固定レイテンシのLive APIモデル（Thinking level非対応）"
    isGeminiExtendedThinking(model) -> "Gemini 3.8 Live Extended Thinkingはlow / medium / highのバックグラウンド推論に対応"
    realtimeProvider(model) == "openai" -> "OpenAI Realtimeは24kHz PCM固定"
    realtimeProvider(model) == "xai" -> "xAIはPCMサンプルレート変更可"
    else -> "Gemini Liveは音声速度変更非対応"
}

/** Web voice studio title (`updateTitle`). */
private fun voiceStudioTitle(model: String): String = when (model) {
    "gpt-transcribe", "gpt-live-transcribe" -> "音声文字起こしスタジオ"
    "gemini-3.5-live-translate-preview" -> "リアルタイム音声翻訳スタジオ"
    else -> "音声スタジオ"
}

private const val DOCK_SETTINGS_OPEN_PREF = "voice_dock_settings_open"

/**
 * Web `#sts-panel`: mic button, mode label, status and hint, Cancel, and in studio mode (`voice-dock`)
 * the 設定 toggle, the expand button and the live transcript. With the studio UI off the settings are
 * always shown and the studio-only controls are hidden.
 */
@Composable
fun RealtimeVoiceDock(
    state: ChatState,
    model: ChatViewModel,
    options: RealtimeOptions,
    onOptions: (RealtimeOptions) -> Unit,
    onStart: () -> Unit,
    onExpand: () -> Unit,
    inStudio: Boolean = false,
) {
    val web = LocalWebPalette.current
    val context = androidx.compose.ui.platform.LocalContext.current
    val prefs = remember { context.getSharedPreferences("settings_local", 0) }
    val studioMode = state.preferences?.voiceStudioUi != false
    var settingsOpen by remember { mutableStateOf(prefs.getBoolean(DOCK_SETTINGS_OPEN_PREF, false)) }
    val realtime = state.realtime
    val modelId = if (realtime.active) realtime.model else state.model
    val phone = androidx.compose.ui.platform.LocalConfiguration.current.screenWidthDp <= 640
    val shape = RoundedCornerShape(8.dp)
    Column(
        Modifier.fillMaxWidth().clip(shape)
            .background(Brush.radialGradient(listOf(web.theme.rgb(0.15f), Color(8, 14, 28).copy(alpha = 0.7f))))
            .border(1.dp, web.theme.rgb(0.2f), shape).padding(horizontal = 16.dp, vertical = 12.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(if (phone) 9.6.dp else 13.6.dp)) {
            val recording = realtime.active
            Box(
                Modifier.size(56.dp).clip(CircleShape).background(if (recording) Tw.red600 else Tw.cyan600)
                    .clickable(role = Role.Button) { if (recording) model.stopRealtime(save = true) else onStart() }
                    .semantics { contentDescription = "録音を開始/停止" },
                contentAlignment = Alignment.Center,
            ) { FaIcon(R.drawable.fa_solid_microphone, null, size = 20.dp, tint = Color.White) }
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.4.dp)) {
                Text(realtimeModeLabel(modelId).uppercase(), fontSize = 10.sp, letterSpacing = 1.sp, color = Tw.cyan300.copy(alpha = 0.8f),
                    maxLines = 1, overflow = TextOverflow.Ellipsis)
                Text(realtime.status.ifBlank { "Tap to speak" }, fontSize = 14.sp, lineHeight = 20.sp, color = web.twText(Tw.gray200),
                    maxLines = 1, overflow = TextOverflow.Ellipsis)
                if (!phone) Text("マイクをタップして開始/停止。停止するとチャットに保存されます", fontSize = 10.sp, lineHeight = 15.sp,
                    color = web.twText(Tw.gray400))
            }
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.4.dp)) {
                Text("Cancel", fontSize = 10.sp, color = web.twText(Tw.gray300),
                    modifier = Modifier.clip(RoundedCornerShape(4.dp)).background(web.twBg(Tw.gray800)).border(1.dp, web.twBorder(Tw.gray700), RoundedCornerShape(4.dp))
                        .clickable(role = Role.Button) { model.stopRealtime(save = false) }.padding(horizontal = 8.dp, vertical = 4.dp))
                if (studioMode) {
                    DockIconButton(R.drawable.fa_solid_sliders_h, "設定", showLabel = !phone, active = settingsOpen) {
                        settingsOpen = !settingsOpen
                        prefs.edit().putBoolean(DOCK_SETTINGS_OPEN_PREF, settingsOpen).apply()
                    }
                    if (!inStudio) DockIconButton(R.drawable.fa_solid_up_right_from_square, "音声スタジオで拡大表示", showLabel = false, active = false, onClick = onExpand)
                }
            }
        }
        if (studioMode && !inStudio) VoiceTranscript(realtime, dock = true)
        if (!studioMode || settingsOpen) {
            val rule = if (studioMode) web.lineSoft else Color.Transparent
            Box(Modifier.padding(top = 12.dp).fillMaxWidth().drawBehind { drawLine(rule, Offset.Zero, Offset(size.width, 0f), 1.dp.toPx()) }
                .padding(top = if (studioMode) 12.dp else 0.dp)) {
                VoiceSettings(modelId, options, onOptions, studioMode)
            }
        }
    }
}

@Composable
private fun DockIconButton(@androidx.annotation.DrawableRes icon: Int, label: String, showLabel: Boolean, active: Boolean, onClick: () -> Unit) {
    val web = LocalWebPalette.current
    Row(
        Modifier.heightIn(min = 36.dp).widthIn(min = 36.dp).clip(CircleShape)
            .background(if (active) web.theme.rgb(0.14f) else Color.White.copy(alpha = 0.04f))
            .border(1.dp, if (active) web.theme.rgb(0.45f) else web.lineStrong, CircleShape)
            .clickable(role = Role.Button, onClick = onClick).semantics { contentDescription = label }
            .padding(horizontal = 10.4.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(5.6.dp, Alignment.CenterHorizontally),
    ) {
        FaIcon(icon, null, size = 12.dp, tint = web.text)
        if (showLabel) Text(label, fontSize = 12.sp, color = web.text)
    }
}

/** `#sts-live-transcript` (dock) or `#voice-studio-transcript` (studio): あなた: … / AI: … */
@Composable
private fun VoiceTranscript(realtime: RealtimeState, dock: Boolean) {
    val web = LocalWebPalette.current
    val lines = listOfNotNull(
        realtime.userText.takeIf { it.isNotBlank() }?.let { true to it },
        realtime.assistantText.takeIf { it.isNotBlank() }?.let { false to it },
    )
    if (dock && lines.isEmpty()) return
    val shape = RoundedCornerShape(if (dock) 10.dp else 8.dp)
    Column(
        Modifier.padding(top = if (dock) 10.4.dp else 0.dp).fillMaxWidth().heightIn(min = if (dock) 0.dp else 160.dp, max = if (dock) 104.dp else 360.dp)
            .clip(shape).background(if (dock) Color.Black.copy(alpha = 0.18f) else web.twBg(Tw.gray900, 0.4f))
            .border(1.dp, if (dock) web.lineSoft else web.twBorder(Tw.gray700, 0.5f), shape)
            .verticalScroll(rememberScrollState(), reverseScrolling = true)
            .padding(horizontal = if (dock) 10.4.dp else 12.dp, vertical = if (dock) 8.dp else 12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        if (lines.isEmpty()) Text("会話の文字起こしがここに表示されます。", fontSize = 10.sp, color = web.twText(Tw.gray500))
        lines.forEach { (user, text) ->
            Text(buildAnnotatedString {
                withStyle(SpanStyle(color = if (user) web.twText(Tw.cyan300) else web.twText(Tw.gray100), fontWeight = FontWeight.Bold)) {
                    append(if (user) "あなた:" else "AI:")
                }
                append(" ")
                withStyle(SpanStyle(color = web.twText(Tw.gray200))) { append(text) }
            }, fontSize = 12.sp, lineHeight = 18.sp)
        }
    }
}

/** `#sts-settings`: Auto Play / Auto Restart / Auto Send / Silence and the provider-specific options. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun VoiceSettings(modelId: String, options: RealtimeOptions, onChange: (RealtimeOptions) -> Unit, dock: Boolean) {
    val web = LocalWebPalette.current
    val size = if (dock) 12.sp else 10.sp
    val transcription = isRealtimeTranscription(modelId)
    val provider = realtimeProvider(modelId)
    val labelColor = web.twText(Tw.gray300)
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        FlowRow(horizontalArrangement = Arrangement.spacedBy(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp),
            itemVerticalAlignment = Alignment.CenterVertically) {
            if (!transcription) VoiceCheck("Auto Play", options.autoPlay, size) { onChange(options.copy(autoPlay = it)) }
            VoiceCheck("Auto Restart", options.autoRestart, size) { onChange(options.copy(autoRestart = it)) }
            VoiceCheck("Auto Send", options.autoSend, size) { onChange(options.copy(autoSend = it)) }
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                Text("Silence", fontSize = size, color = labelColor)
                TwInput(options.silenceSeconds, { value -> onChange(options.copy(silenceSeconds = value.filter { c -> c.isDigit() || c == '.' }.take(4))) },
                    "2.5", modifier = Modifier.width(48.dp), fontSize = size, padding = 4.dp, background = Tw.gray800, borderColor = Tw.gray700)
                Text("s", fontSize = size, color = labelColor)
            }
        }
        FlowRow(horizontalArrangement = Arrangement.spacedBy(16.dp), verticalArrangement = Arrangement.spacedBy(9.6.dp),
            itemVerticalAlignment = Alignment.CenterVertically) {
            val voices = realtimeVoices(modelId)
            if (voices.isNotEmpty()) VoiceSelect("Voice", resolvedRealtimeVoice(modelId, options), voices.map { WebOption(it, it) }, size) {
                onChange(options.copy(voice = it))
            }
            if (modelId == "gemini-3.5-live-translate-preview") VoiceSelect("Target Lang", options.targetLanguage,
                REALTIME_TARGET_LANGUAGES.map { WebOption(it.first, it.second) }, size) { onChange(options.copy(targetLanguage = it)) }
            if (modelId == "gemini-3.5-transcribe-live") VoiceSelect("Mode", options.transcriptionMode,
                webOptions("VERBATIM" to "Verbatim", "SMART" to "Smart"), size) { onChange(options.copy(transcriptionMode = it)) }
            if (modelId == "gemini-3.5-transcribe-live" || isGrokLiveTranscribe(modelId)) Row(verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                Text("Custom Vocab", fontSize = size, color = labelColor)
                TwInput(options.customVocabulary, { onChange(options.copy(customVocabulary = it.take(4000))) }, "Gemini, Kubernetes, BigQuery",
                    modifier = Modifier.width(176.dp), fontSize = size, padding = 4.dp, background = Tw.gray800, borderColor = Tw.gray700)
            }
            if (!transcription && provider == "openai") Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                Text("Speed", fontSize = size, color = labelColor)
                Slider(options.speed.coerceIn(0.25f, 1.5f), { onChange(options.copy(speed = (Math.round(it * 20) / 20f))) }, valueRange = 0.25f..1.5f,
                    colors = SliderDefaults.colors(thumbColor = Tw.cyan400, activeTrackColor = Tw.cyan400, inactiveTrackColor = web.twBg(Tw.gray600)),
                    modifier = Modifier.width(120.dp).height(24.dp))
                Text(String.format(java.util.Locale.ROOT, "%.2fx", options.speed), fontSize = size, fontFamily = FontFamily.Monospace, color = web.twText(Tw.gray200))
            }
            if (!transcription && provider == "xai") Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                val rates = GROK_PCM_RATES.map { WebOption(it.toString(), it.toString()) }
                VoiceSelect("Rate In", options.rateIn.toString(), rates, size) { onChange(options.copy(rateIn = it.toInt())) }
                VoiceSelect("Out", options.rateOut.toString(), rates, size) { onChange(options.copy(rateOut = it.toInt())) }
            }
            val levels = realtimeThinkingLevels(modelId)
            if (levels.isNotEmpty()) Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                Text("Thinking", fontSize = size, color = web.twText(Tw.purple300))
                WebSelect(resolvedRealtimeThinking(modelId, options),
                    webOptions("minimal" to "Min", "low" to "Low", "medium" to "Mid", "high" to "High"),
                    { onChange(options.copy(thinkingLevel = it)) }, fontSize = size, background = web.twBg(Tw.gray800), borderColor = web.twBorder(Tw.gray700),
                    disabledValues = setOf("minimal", "low", "medium", "high") - levels.toSet())
                VoiceCheck("Thoughts", options.includeThoughts, size) { onChange(options.copy(includeThoughts = it)) }
            }
            Text(realtimeNote(modelId), fontSize = size, color = web.twText(Tw.gray500))
        }
    }
}

@Composable
private fun VoiceCheck(label: String, checked: Boolean, size: androidx.compose.ui.unit.TextUnit, onChange: (Boolean) -> Unit) {
    Row(
        Modifier.toggleable(checked, role = Role.Checkbox, onValueChange = onChange),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        WebCheckbox(checked, null, size = 12.dp)
        Text(label, fontSize = size, color = LocalWebPalette.current.twText(Tw.gray300))
    }
}

@Composable
private fun VoiceSelect(label: String, value: String, options: List<WebOption>, size: androidx.compose.ui.unit.TextUnit, onChange: (String) -> Unit) {
    val web = LocalWebPalette.current
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
        Text(label, fontSize = size, color = web.twText(Tw.gray300))
        WebSelect(value, options, onChange, fontSize = size, background = web.twBg(Tw.gray800), borderColor = web.twBorder(Tw.gray700),
            contentDescription = label)
    }
}

/**
 * `#voice-studio-modal`: the enlarged view with the transcript above the same dock controls. The session
 * keeps running when it is closed.
 */
@Composable
fun RealtimeStudioDialog(
    state: ChatState,
    model: ChatViewModel,
    options: RealtimeOptions,
    onOptions: (RealtimeOptions) -> Unit,
    onDismiss: () -> Unit,
    onStart: () -> Unit = { startRealtimeWith(model, state.model, options) },
) {
    val web = LocalWebPalette.current
    val modelId = if (state.realtime.active) state.realtime.model else state.model
    WebOverlayModal(onDismiss, grayOverlay(), 4.dp) { phone ->
        Column(
            Modifier.fillMaxSize().verticalScroll(rememberScrollState()),
            verticalArrangement = if (phone) Arrangement.Top else Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            val shape = RoundedCornerShape(8.dp)
            Column(
                Modifier.padding(16.dp).widthIn(max = 672.dp).fillMaxWidth().clip(shape).background(web.twBg(Tw.gray800))
                    .border(1.dp, Tw.cyan500.copy(alpha = 0.3f), shape).padding(if (phone) 16.dp else 24.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    FaIcon(R.drawable.fa_solid_microphone, null, size = 16.dp, tint = web.twText(Tw.cyan300), modifier = Modifier.padding(end = 8.dp))
                    Text(voiceStudioTitle(modelId), fontSize = 18.sp, lineHeight = 28.sp, fontWeight = FontWeight.Bold, color = web.twText(Tw.white),
                        modifier = Modifier.weight(1f))
                    Box(Modifier.size(32.dp).clip(CircleShape).clickable(role = Role.Button, onClick = onDismiss).semantics { contentDescription = "閉じる" },
                        contentAlignment = Alignment.Center) { FaIcon(R.drawable.fa_solid_times, null, size = 16.dp, tint = web.twText(Tw.gray400)) }
                }
                VoiceTranscript(state.realtime, dock = false)
                RealtimeVoiceDock(state, model, options, onOptions, onStart = onStart, onExpand = {}, inStudio = true)
                val rule = web.twBorder(Tw.gray700, 0.5f)
                Text("マイクをタップして話し始め / もう一度タップで停止。停止後はチャットへ自動保存されます。", fontSize = 10.sp, lineHeight = 15.sp,
                    color = web.twText(Tw.gray500), textAlign = TextAlign.Center,
                    modifier = Modifier.fillMaxWidth().drawBehind { drawLine(rule, Offset.Zero, Offset(size.width, 0f), 1.dp.toPx()) }.padding(top = 8.dp))
            }
        }
    }
}
