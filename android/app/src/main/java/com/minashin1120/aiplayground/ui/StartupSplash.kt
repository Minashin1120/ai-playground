package com.minashin1120.aiplayground.ui

import android.content.Context
import android.provider.Settings
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.EaseIn
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import com.minashin1120.aiplayground.R
import kotlinx.coroutines.delay

internal fun areSystemAnimationsDisabled(context: Context): Boolean =
    runCatching {
        Settings.Global.getFloat(
            context.contentResolver,
            Settings.Global.ANIMATOR_DURATION_SCALE,
            1f,
        ) == 0f
    }.getOrDefault(false)

@Composable
internal fun StartupSplash(enabled: Boolean, background: Color) {
    if (!enabled) return

    var visible by rememberSaveable { mutableStateOf(true) }
    if (!visible) return

    BoxWithConstraints(
        Modifier
            .fillMaxSize()
            .windowInsetsPadding(WindowInsets.systemBars)
            .clipToBounds(),
    ) {
        val progress = remember { Animatable(0f) }
        val logoSize = (maxWidth * 0.20f).coerceAtLeast(1.dp)
        val maxScale = (maxOf(maxWidth.value, maxHeight.value) / logoSize.value * 1.15f).coerceAtLeast(5f)
        val scale = 1f + (maxScale - 1f) * progress.value
        val backgroundAlpha = 1f - ((progress.value - 0.56f) / 0.44f).coerceIn(0f, 1f)

        LaunchedEffect(Unit) {
            progress.animateTo(1f, tween(durationMillis = 1_500, easing = EaseIn))
            delay(50)
            visible = false
        }

        Box(
            Modifier
                .fillMaxSize()
                .background(background.copy(alpha = backgroundAlpha)),
        )
        Image(
            painter = painterResource(R.drawable.ic_playground_mark),
            contentDescription = null,
            contentScale = ContentScale.Fit,
            modifier = Modifier
                .size(logoSize)
                .align(Alignment.Center)
                .offset(y = -(maxHeight * 0.08f))
                .graphicsLayer {
                    scaleX = scale
                    scaleY = scale
                },
        )
    }
}
