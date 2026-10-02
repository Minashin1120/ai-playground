package com.minashin1120.aiplayground.ui

import android.media.MediaPlayer
import android.media.PlaybackParams
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.R
import java.io.File
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.launch

/**
 * Downloads an attachment reference into a local file (account cache first), as [FileViewerDialog] does.
 * Provided once by the screen so an answer's inline `<audio>` can play without opening the file panel.
 */
internal val LocalAttachmentDownloader = staticCompositionLocalOf<(suspend (String) -> Pair<File, String>)?> { null }

private enum class AudioPhase { Idle, Loading, Ready, Failed }

/** Web player's speed button cycle (`AUDIO_PLAYER_SPEEDS` in chat_core part01). */
private val AudioSpeeds = floatArrayOf(1f, 1.25f, 1.5f, 2f, 0.75f)

private fun speedLabel(speed: Float): String = "${speed.toString().removeSuffix(".0")}x"

/** Browser `audio` time label: `m:ss`. */
internal fun formatAudioTime(milliseconds: Int): String {
    val seconds = (milliseconds.coerceAtLeast(0) / 1000)
    return "${seconds / 60}:${"%02d".format(seconds % 60)}"
}

private class AudioHolder { var player: MediaPlayer? = null }

/**
 * Web `<audio controls class="w-full mt-2">` that the server writes into an answer for generated speech,
 * music and Lyria recordings, drawn like the Web's own player (`.aip`). The file is fetched on the first tap (like `preload="metadata"` without
 * spending mobile data for every audio in a long chat) and played with [MediaPlayer].
 */
@Composable
internal fun InlineAudioPlayer(reference: String, onOpen: (String) -> Unit) {
    val web = LocalWebPalette.current
    val download = LocalAttachmentDownloader.current
    val scope = rememberCoroutineScope()
    val holder = remember(reference) { AudioHolder() }
    var phase by remember(reference) { mutableStateOf(AudioPhase.Idle) }
    var playing by remember(reference) { mutableStateOf(false) }
    var durationMs by remember(reference) { mutableIntStateOf(0) }
    var positionMs by remember(reference) { mutableIntStateOf(0) }
    var scrubbing by remember(reference) { mutableStateOf(false) }
    var speedIndex by remember(reference) { mutableIntStateOf(0) }

    DisposableEffect(reference) {
        onDispose {
            holder.player?.let { player ->
                player.setOnPreparedListener(null)
                player.setOnCompletionListener(null)
                player.setOnErrorListener(null)
                runCatching { player.stop() }
                player.release()
            }
            holder.player = null
        }
    }
    LaunchedEffect(reference, playing) {
        while (playing) {
            if (!scrubbing) positionMs = runCatching { holder.player?.currentPosition ?: 0 }.getOrDefault(0)
            delay(250)
        }
    }

    // Setting playback params on a paused player can start it, so the speed is only applied right before playing.
    fun applySpeed(player: MediaPlayer) {
        runCatching { player.playbackParams = PlaybackParams().setSpeed(AudioSpeeds[speedIndex]) }
    }

    fun load() {
        val fetch = download
        if (fetch == null) { onOpen(reference); return }
        phase = AudioPhase.Loading
        scope.launch {
            val file = runCatching { fetch(reference).first }.getOrNull()
            ensureActive()
            if (file == null) { phase = AudioPhase.Failed; return@launch }
            val player = MediaPlayer()
            holder.player = player
            player.setOnPreparedListener {
                durationMs = it.duration.coerceAtLeast(0)
                phase = AudioPhase.Ready
                applySpeed(it)
                it.start()
                playing = true
            }
            player.setOnCompletionListener {
                playing = false
                positionMs = 0
                runCatching { it.seekTo(0) }
            }
            player.setOnErrorListener { _, _, _ -> playing = false; phase = AudioPhase.Failed; true }
            val started = runCatching {
                player.setDataSource(file.absolutePath)
                player.prepareAsync()
            }.isSuccess
            if (!started) phase = AudioPhase.Failed
        }
    }

    fun toggle() {
        when (phase) {
            AudioPhase.Idle, AudioPhase.Failed -> load()
            AudioPhase.Loading -> Unit
            AudioPhase.Ready -> holder.player?.let { player ->
                if (playing) { runCatching { player.pause() }; playing = false } else { applySpeed(player); runCatching { player.start() }; playing = true }
            }
        }
    }

    val shape = RoundedCornerShape(27.dp)
    Row(
        Modifier
            .padding(top = 8.dp)
            .fillMaxWidth()
            .widthIn(max = 400.dp)
            .height(54.dp)
            .clip(shape)
            .background(web.panel2)
            .border(1.dp, web.line, shape)
            .padding(horizontal = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Box(
            Modifier.size(32.dp).clip(CircleShape).background(web.theme.t500)
                .clickable(onClickLabel = if (playing) "一時停止" else "再生", role = Role.Button, onClick = ::toggle),
            contentAlignment = Alignment.Center,
        ) {
            if (phase == AudioPhase.Loading) CircularProgressIndicator(Modifier.size(16.dp), strokeWidth = 2.dp, color = web.textInverse)
            else FaIcon(if (playing) R.drawable.fa_solid_pause else R.drawable.fa_solid_play, if (playing) "一時停止" else "再生", size = 13.dp, tint = web.textInverse)
        }
        if (phase == AudioPhase.Failed) {
            Text("音声を再生できませんでした。", color = web.muted, fontSize = 12.sp)
        } else {
            Text(
                "${formatAudioTime(positionMs)} / ${formatAudioTime(durationMs)}",
                color = web.muted, fontSize = 12.sp, fontFamily = FontFamily.Monospace,
            )
            Slider(
                value = positionMs.coerceIn(0, durationMs.coerceAtLeast(1)).toFloat(),
                onValueChange = { scrubbing = true; positionMs = it.toInt() },
                onValueChangeFinished = {
                    runCatching { holder.player?.seekTo(positionMs) }
                    scrubbing = false
                },
                valueRange = 0f..durationMs.coerceAtLeast(1).toFloat(),
                enabled = phase == AudioPhase.Ready,
                colors = SliderDefaults.colors(
                    thumbColor = web.theme300, activeTrackColor = web.theme.t500, inactiveTrackColor = web.lineStrong,
                    disabledThumbColor = web.lineStrong, disabledActiveTrackColor = web.lineStrong,
                    disabledInactiveTrackColor = web.lineStrong,
                ),
                modifier = Modifier.weight(1f),
            )
            Box(
                Modifier.height(24.dp).widthIn(min = 38.dp).clip(CircleShape).border(1.dp, web.lineStrong, CircleShape)
                    .clickable(onClickLabel = "再生速度", role = Role.Button) {
                        speedIndex = (speedIndex + 1) % AudioSpeeds.size
                        if (playing) holder.player?.let(::applySpeed)
                    }
                    .padding(horizontal = 6.dp),
                contentAlignment = Alignment.Center,
            ) {
                Text(speedLabel(AudioSpeeds[speedIndex]), color = web.muted, fontSize = 11.sp)
            }
        }
    }
}
