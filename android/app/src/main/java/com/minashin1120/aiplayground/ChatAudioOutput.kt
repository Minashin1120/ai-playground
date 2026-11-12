package com.minashin1120.aiplayground

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack

/**
 * Streaming PCM output shared by the realtime voice session and the Lyria RealTime studio
 * (Web `AudioContext`). The caller writes the decoded chunks.
 */
internal fun newChatAudioTrack(rate: Int, stereo: Boolean): AudioTrack {
    val channels = if (stereo) AudioFormat.CHANNEL_OUT_STEREO else AudioFormat.CHANNEL_OUT_MONO
    val min = AudioTrack.getMinBufferSize(rate, channels, AudioFormat.ENCODING_PCM_16BIT).coerceAtLeast(4096)
    return AudioTrack.Builder().setAudioAttributes(AudioAttributes.Builder()
        .setUsage(AudioAttributes.USAGE_MEDIA).setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
        .setAudioFormat(AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_16BIT)
            .setSampleRate(rate).setChannelMask(channels).build())
        .setBufferSizeInBytes(min * 2).setTransferMode(AudioTrack.MODE_STREAM).build().also { it.play() }
}
