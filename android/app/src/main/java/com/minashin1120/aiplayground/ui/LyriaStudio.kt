package com.minashin1120.aiplayground.ui

import androidx.annotation.DrawableRes
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.runtime.snapshots.SnapshotStateList
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.ChatViewModel
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.LyriaPrompt
import kotlinx.coroutines.delay
import org.json.JSONObject
import java.util.Locale

private val LYRIA_SCALES = listOf(
    "" to "自動", "C_MAJOR_A_MINOR" to "C major / A minor", "D_FLAT_MAJOR_B_FLAT_MINOR" to "D♭ major / B♭ minor",
    "D_MAJOR_B_MINOR" to "D major / B minor", "E_FLAT_MAJOR_C_MINOR" to "E♭ major / C minor",
    "E_MAJOR_D_FLAT_MINOR" to "E major / C♯/D♭ minor", "F_MAJOR_D_MINOR" to "F major / D minor",
    "G_FLAT_MAJOR_E_FLAT_MINOR" to "G♭ major / E♭ minor", "G_MAJOR_E_MINOR" to "G major / E minor",
    "A_FLAT_MAJOR_F_MINOR" to "A♭ major / F minor", "A_MAJOR_G_FLAT_MINOR" to "A major / F♯/G♭ minor",
    "B_FLAT_MAJOR_G_MINOR" to "B♭ major / G minor", "B_MAJOR_A_FLAT_MINOR" to "B major / G♯/A♭ minor",
)

/** The studio's music settings (Web `collectConfig`). */
@Stable
private class LyriaConfig {
    var bpm by mutableFloatStateOf(120f)
    var guidance by mutableFloatStateOf(4f)
    var density by mutableFloatStateOf(0.5f)
    var brightness by mutableFloatStateOf(0.5f)
    var temperature by mutableFloatStateOf(1.1f)
    var scale by mutableStateOf("")
    var mode by mutableStateOf("QUALITY")
    var muteBass by mutableStateOf(false)
    var muteDrums by mutableStateOf(false)
    var onlyBassDrums by mutableStateOf(false)

    fun json(): JSONObject = JSONObject()
        .put("bpm", Math.round(bpm))
        .put("guidance", round1(guidance))
        .put("density", round2(density))
        .put("brightness", round2(brightness))
        .put("temperature", round1(temperature))
        .apply { if (scale.isNotEmpty()) put("scale", scale) }
        .put("music_generation_mode", mode)
        .put("mute_bass", muteBass)
        .put("mute_drums", muteDrums)
        .put("only_bass_and_drums", onlyBassDrums)

    private fun round1(value: Float) = Math.round(value * 10) / 10.0
    private fun round2(value: Float) = Math.round(value * 100) / 100.0
}

/**
 * `#lyria-studio-modal` (Lyria RealTime Studio): weighted prompts, music settings, transport buttons,
 * status with elapsed time and 「チャットへ保存」. Closing it cancels the session, like the Web.
 */
@Composable
internal fun LyriaStudioDialog(state: ChatState, model: ChatViewModel, onDismiss: () -> Unit) {
    val web = LocalWebPalette.current
    val lyria = state.lyria
    val prompts = remember { mutableStateListOf(LyriaPrompt(lyria.prompt, 1f)) }
    val config = remember { LyriaConfig() }
    var lastConfig by remember { mutableStateOf<JSONObject?>(null) }
    val close = { model.stopLyria(save = false); onDismiss() }
    TwPanelFrame(onDismiss = close, border = web.twBorder(Tw.purple700, 0.5f), panelMaxWidth = 768.dp) { wide ->
        Row(verticalAlignment = Alignment.CenterVertically) {
            FaIcon(R.drawable.fa_solid_music, null, size = 16.dp, tint = web.twText(Tw.purple300), modifier = Modifier.padding(end = 8.dp))
            Text("Lyria RealTime Studio", fontSize = 18.sp, lineHeight = 28.sp, fontWeight = FontWeight.Bold, color = web.twText(Tw.white),
                modifier = Modifier.weight(1f))
            val saveVisible = lyria.active && lyria.kind !in setOf("idle", "connecting", "error")
            if (saveVisible) Row(
                Modifier.padding(end = 8.dp).clip(RoundedCornerShape(4.dp)).background(Tw.emerald600)
                    .clickable(enabled = !lyria.busy, role = Role.Button) { model.lyriaSave(onDismiss) }.padding(horizontal = 12.dp, vertical = 6.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                FaIcon(R.drawable.fa_solid_save, null, size = 11.dp, tint = Color.White, modifier = Modifier.padding(end = 4.dp))
                Text("チャットへ保存", fontSize = 11.sp, fontWeight = FontWeight.Bold, color = Color.White)
            }
            Box(Modifier.size(32.dp).clip(CircleShape).clickable(role = Role.Button, onClick = close).semantics { contentDescription = "閉じる" },
                contentAlignment = Alignment.Center) {
                FaIcon(R.drawable.fa_solid_times, null, size = 16.dp, tint = web.twText(Tw.gray400))
            }
        }
        LyriaStatusBar(lyria.status, lyria.kind, lyria.startedAt)
        val promptColumn: @Composable (Modifier) -> Unit = { modifier -> LyriaPromptColumn(prompts, modifier) { model.lyriaApplyPrompts(prompts.toList()) } }
        val configColumn: @Composable (Modifier) -> Unit = { modifier ->
            LyriaConfigColumn(config, modifier) {
                val next = config.json()
                val prev = lastConfig
                val reset = prev == null || prev.optInt("bpm") != next.optInt("bpm") || prev.optString("scale") != next.optString("scale")
                model.lyriaApplyConfig(next, resetContext = reset)
                lastConfig = next
            }
        }
        if (wide) Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
            promptColumn(Modifier.weight(1f))
            configColumn(Modifier.weight(1f))
        } else Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
            promptColumn(Modifier.fillMaxWidth())
            configColumn(Modifier.fillMaxWidth())
        }
        val playing = lyria.kind == "streaming" || lyria.kind == "connecting"
        val rule = web.twBorder(Tw.gray700, 0.5f)
        Row(
            Modifier.fillMaxWidth().drawTopRule(rule).padding(top = 4.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp, Alignment.CenterHorizontally), verticalAlignment = Alignment.CenterVertically,
        ) {
            TransportButton(R.drawable.fa_solid_play, "再生", 56.dp, Tw.purple600, enabled = !lyria.busy) {
                if (!lyria.active) {
                    lastConfig = config.json()
                    model.lyriaStart(prompts.toList(), config.json())
                } else model.lyriaControl("PLAY")
            }
            TransportButton(R.drawable.fa_solid_pause, "一時停止", 48.dp, web.twBg(Tw.gray700), enabled = !lyria.busy && playing) { model.lyriaControl("PAUSE") }
            TransportButton(R.drawable.fa_solid_stop, "停止", 48.dp, web.twBg(Tw.gray700), enabled = !lyria.busy && lyria.active && playing) { model.lyriaControl("STOP") }
            TransportButton(R.drawable.fa_solid_arrows_rotate, "コンテキストをリセット（BPM・スケール変更後）", 48.dp, web.twBg(Tw.gray700),
                enabled = !lyria.busy && lyria.active && playing) { model.lyriaControl("RESET_CONTEXT") }
        }
        Text("実験モデル: 楽器のみ・ボーカルなし / 48kHzステレオ / 再生内容は「チャットへ保存」でWAVとして保存できます",
            fontSize = 10.sp, lineHeight = 15.sp, color = web.twText(Tw.gray500), textAlign = TextAlign.Center, modifier = Modifier.fillMaxWidth())
    }
}

/** A Web Tailwind modal panel (`bg-gray-800 rounded-lg p-4 sm:p-6 m-4 gap-3`); [content] gets whether the md: layout applies. */
@Composable
private fun TwPanelFrame(onDismiss: () -> Unit, border: Color, panelMaxWidth: Dp, content: @Composable ColumnScope.(Boolean) -> Unit) {
    val web = LocalWebPalette.current
    WebOverlayModal(onDismiss, grayOverlay(), 4.dp) { phone ->
        BoxWithConstraints(Modifier.fillMaxSize()) {
            val wide = maxWidth >= 768.dp
            Column(
                Modifier.fillMaxSize().verticalScroll(rememberScrollState()),
                verticalArrangement = if (phone) Arrangement.Top else Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                val shape = RoundedCornerShape(8.dp)
                Column(
                    Modifier.padding(16.dp).widthIn(max = panelMaxWidth).fillMaxWidth().clip(shape)
                        .background(web.twBg(Tw.gray800)).border(1.dp, border, shape).padding(if (phone) 16.dp else 24.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) { content(wide) }
            }
        }
    }
}

private fun Modifier.drawTopRule(color: Color): Modifier = drawBehind {
    drawLine(color, androidx.compose.ui.geometry.Offset.Zero, androidx.compose.ui.geometry.Offset(size.width, 0f), 1.dp.toPx())
}

/** `#lyria-studio-status`: coloured dot (pulsing while connecting or streaming), text and `mm:ss`. */
@Composable
private fun LyriaStatusBar(status: String, kind: String, startedAt: Long) {
    val web = LocalWebPalette.current
    val reduce = LocalReduceMotion.current
    var now by remember { mutableLongStateOf(System.currentTimeMillis()) }
    LaunchedEffect(startedAt, kind) {
        while (startedAt > 0 && kind == "streaming") { now = System.currentTimeMillis(); delay(1000) }
        now = System.currentTimeMillis()
    }
    val dot = when (kind) {
        "connecting", "paused" -> Tw.amber500
        "streaming" -> Tw.emerald600
        "error" -> Tw.red600
        else -> Tw.gray600
    }
    val pulse = if ((kind == "connecting" || kind == "streaming") && !reduce) rememberInfiniteTransition(label = "lyria dot").animateFloat(
        1f, 0.5f, infiniteRepeatable(tween(1000), RepeatMode.Reverse), label = "lyria dot alpha",
    ).value else 1f
    val secs = if (startedAt > 0) ((now - startedAt) / 1000).coerceAtLeast(0) else 0
    val shape = RoundedCornerShape(8.dp)
    Row(
        Modifier.fillMaxWidth().clip(shape).background(web.twBg(Tw.gray900, 0.6f)).border(1.dp, web.twBorder(Tw.gray700, 0.7f), shape)
            .padding(horizontal = 12.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Box(Modifier.size(8.dp).alpha(pulse).clip(CircleShape).background(dot))
        Text(status, fontSize = 12.sp, lineHeight = 16.sp, color = web.twText(Tw.gray300), modifier = Modifier.weight(1f))
        Text(String.format(Locale.ROOT, "%02d:%02d", secs / 60, secs % 60), fontSize = 12.sp, fontFamily = FontFamily.Monospace,
            color = web.twText(Tw.gray400))
    }
}

@Composable
private fun LyriaSectionTitle(@DrawableRes icon: Int, title: String, modifier: Modifier = Modifier) {
    val web = LocalWebPalette.current
    Row(modifier, verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        FaIcon(icon, null, size = 11.dp, tint = web.twText(Tw.purple300))
        Text(title, fontSize = 12.sp, fontWeight = FontWeight.Bold, color = web.twText(Tw.purple300))
    }
}

@Composable
private fun LyriaPromptColumn(prompts: SnapshotStateList<LyriaPrompt>, modifier: Modifier, onApply: () -> Unit) {
    val web = LocalWebPalette.current
    Column(modifier, verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            LyriaSectionTitle(R.drawable.fa_solid_pen, "プロンプト（重み付き）", Modifier.weight(1f))
            Row(
                Modifier.clip(RoundedCornerShape(4.dp)).background(web.twBg(Tw.gray700)).clickable(role = Role.Button) { prompts.add(LyriaPrompt("", 1f)) }
                    .padding(horizontal = 8.dp, vertical = 4.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                FaIcon(R.drawable.fa_solid_plus, null, size = 10.dp, tint = web.twText(Tw.gray300), modifier = Modifier.padding(end = 4.dp))
                Text("追加", fontSize = 10.sp, color = web.twText(Tw.gray300))
            }
        }
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            prompts.forEachIndexed { index, prompt ->
                key(index) {
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        TwInput(prompt.text, { prompts[index] = prompts[index].copy(text = it.take(4000)) }, "例: minimal techno / warm acoustic guitar",
                            modifier = Modifier.weight(1f), fontSize = 11.sp, padding = 6.dp, background = Tw.gray700)
                        Text("w", fontSize = 10.sp, color = web.twText(Tw.gray400))
                        PurpleSlider(prompt.weight, 0.1f..5f, 0.1f, Modifier.width(64.dp)) { prompts[index] = prompts[index].copy(weight = it) }
                        Text(String.format(Locale.ROOT, "%.1f", prompt.weight), fontSize = 10.sp, fontFamily = FontFamily.Monospace,
                            color = web.twText(Tw.purple300), textAlign = TextAlign.End, modifier = Modifier.width(32.dp))
                        Box(
                            Modifier.size(24.dp).clip(CircleShape).background(web.twBg(Tw.gray800))
                                .clickable(role = Role.Button) { if (prompts.size > 1) prompts.removeAt(index) }.semantics { contentDescription = "削除" },
                            contentAlignment = Alignment.Center,
                        ) { FaIcon(R.drawable.fa_solid_times, null, size = 10.dp, tint = web.twText(Tw.gray400)) }
                    }
                }
            }
        }
        Text("ジャンル・楽器・雰囲気を詳しく書くと良い結果になります（例: 「minimal techno」「warm acoustic guitar」）。再生中も「適用」で滑らかに変化します。",
            fontSize = 10.sp, lineHeight = 16.sp, color = web.twText(Tw.gray500))
        LyriaWideButton(R.drawable.fa_solid_arrow_right, "プロンプトを適用", Tw.purple600, onApply)
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun LyriaConfigColumn(config: LyriaConfig, modifier: Modifier, onApply: () -> Unit) {
    val web = LocalWebPalette.current
    Column(modifier, verticalArrangement = Arrangement.spacedBy(12.dp)) {
        LyriaSectionTitle(R.drawable.fa_solid_sliders_h, "音楽設定")
        val rows = listOf<@Composable (Modifier) -> Unit>(
            { m -> LyriaRange("BPM", config.bpm, 60f..200f, 1f, m, integer = true) { config.bpm = it } },
            { m -> LyriaRange("ガイダンス", config.guidance, 0f..6f, 0.1f, m) { config.guidance = it } },
            { m -> LyriaRange("密度", config.density, 0f..1f, 0.05f, m) { config.density = it } },
            { m -> LyriaRange("明るさ", config.brightness, 0f..1f, 0.05f, m) { config.brightness = it } },
            { m -> LyriaRange("温度", config.temperature, 0f..3f, 0.1f, m) { config.temperature = it } },
            { m -> LyriaSelect("スケール", config.scale, LYRIA_SCALES, m) { config.scale = it } },
            { m -> LyriaSelect("モード", config.mode, listOf("QUALITY" to "Quality（品質優先）", "DIVERSITY" to "Diversity（多様性）"), m) { config.mode = it } },
        )
        rows.chunked(2).forEach { pair ->
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                pair.forEach { it(Modifier.weight(1f)) }
                if (pair.size == 1) Spacer(Modifier.weight(1f))
            }
        }
        FlowRow(horizontalArrangement = Arrangement.spacedBy(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            LyriaCheck("ベースをミュート", config.muteBass) { config.muteBass = it }
            LyriaCheck("ドラムをミュート", config.muteDrums) { config.muteDrums = it }
            LyriaCheck("ベースとドラムのみ", config.onlyBassDrums) { config.onlyBassDrums = it }
        }
        LyriaWideButton(R.drawable.fa_solid_sliders_h, "設定を適用", web.twBg(Tw.gray700), onApply)
        Text("BPM・スケールの変更はモデルの文脈をリセットするため、次のフレーズで反映されます。", fontSize = 10.sp, lineHeight = 15.sp,
            color = web.twText(Tw.gray500))
    }
}

@Composable
private fun LyriaRange(label: String, value: Float, range: ClosedFloatingPointRange<Float>, step: Float, modifier: Modifier,
                       integer: Boolean = false, onChange: (Float) -> Unit) {
    val web = LocalWebPalette.current
    Column(modifier, verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Row {
            Text(label, fontSize = 11.sp, color = web.twText(Tw.gray300), modifier = Modifier.weight(1f))
            Text(if (integer) Math.round(value).toString() else String.format(Locale.ROOT, "%.1f", value), fontSize = 11.sp,
                fontFamily = FontFamily.Monospace, color = web.twText(Tw.purple300))
        }
        PurpleSlider(value, range, step, Modifier.fillMaxWidth(), onChange)
    }
}

@Composable
private fun LyriaSelect(label: String, value: String, options: List<Pair<String, String>>, modifier: Modifier, onChange: (String) -> Unit) {
    val web = LocalWebPalette.current
    Column(modifier, verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text(label, fontSize = 11.sp, color = web.twText(Tw.gray300))
        WebSelect(value, options.map { WebOption(it.first, it.second) }, onChange, fontSize = 11.sp, fillWidth = true,
            contentPadding = PaddingValues(horizontal = 8.dp, vertical = 4.dp), contentDescription = label)
    }
}

@Composable
private fun LyriaCheck(label: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(
        Modifier.toggleable(checked, role = Role.Checkbox, onValueChange = onChange),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        WebCheckbox(checked, null, size = 12.dp)
        Text(label, fontSize = 11.sp, color = LocalWebPalette.current.twText(Tw.gray300))
    }
}

/** `input[type=range].accent-purple-400`. */
@Composable
private fun PurpleSlider(value: Float, range: ClosedFloatingPointRange<Float>, step: Float, modifier: Modifier, onChange: (Float) -> Unit) {
    val steps = (((range.endInclusive - range.start) / step).toInt() - 1).coerceAtLeast(0)
    Slider(
        value.coerceIn(range), { onChange((Math.round(it / step) * step).coerceIn(range)) }, valueRange = range, steps = steps,
        colors = SliderDefaults.colors(thumbColor = Tw.purple400, activeTrackColor = Tw.purple400, inactiveTrackColor = LocalWebPalette.current.twBg(Tw.gray600),
            activeTickColor = Color.Transparent, inactiveTickColor = Color.Transparent),
        modifier = modifier.height(24.dp),
    )
}

@Composable
private fun LyriaWideButton(@DrawableRes icon: Int, label: String, background: Color, onClick: () -> Unit) {
    Row(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(4.dp)).background(background).clickable(role = Role.Button, onClick = onClick)
            .padding(horizontal = 12.dp, vertical = 8.dp),
        horizontalArrangement = Arrangement.Center, verticalAlignment = Alignment.CenterVertically,
    ) {
        FaIcon(icon, null, size = 12.dp, tint = Color.White, modifier = Modifier.padding(end = 4.dp))
        Text(label, fontSize = 12.sp, fontWeight = FontWeight.Bold, color = Color.White)
    }
}

@Composable
private fun TransportButton(@DrawableRes icon: Int, label: String, size: Dp, background: Color, enabled: Boolean, onClick: () -> Unit) {
    Box(
        Modifier.size(size).alpha(if (enabled) 1f else 0.5f).clip(CircleShape).background(background)
            .clickable(enabled = enabled, role = Role.Button, onClick = onClick).semantics { contentDescription = label },
        contentAlignment = Alignment.Center,
    ) { FaIcon(icon, null, size = if (size > 50.dp) 18.dp else 16.dp, tint = Color.White) }
}
