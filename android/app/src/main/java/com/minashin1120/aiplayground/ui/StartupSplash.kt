package com.minashin1120.aiplayground.ui

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.EaseIn
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Image
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
import androidx.compose.ui.graphics.CompositingStrategy
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import com.minashin1120.aiplayground.R

@Composable
internal fun StartupSplash(enabled: Boolean) {
    if (!enabled) return

    var visible by rememberSaveable { mutableStateOf(true) }
    if (!visible) return

    // Fades the fully zoomed mark away so the app is revealed instead of cut in.
    val fade = remember { Animatable(1f) }
    BoxWithConstraints(
        Modifier
            .fillMaxSize()
            .windowInsetsPadding(WindowInsets.systemBars)
            .clipToBounds()
            .graphicsLayer { alpha = fade.value },
    ) {
        val progress = remember { Animatable(0f) }
        val logoSize = (maxWidth * 0.20f).coerceAtLeast(1.dp)
        val maxScale = (maxOf(maxWidth.value, maxHeight.value) / logoSize.value * 1.15f).coerceAtLeast(5f)
        val zoomProgress = ((progress.value - 0.62f) / 0.38f).coerceIn(0f, 1f)
        val scale = 1f + (maxScale - 1f) * zoomProgress
        LaunchedEffect(Unit) {
            progress.animateTo(1f, tween(durationMillis = 1_500, easing = EaseIn))
            fade.animateTo(0f, tween(durationMillis = 180, easing = PlaygroundMotion.Exit))
            visible = false
        }

        androidx.compose.foundation.Canvas(
            Modifier
                .fillMaxSize()
                .graphicsLayer { compositingStrategy = CompositingStrategy.Offscreen },
        ) {
            drawRect(Color.Black)
        }
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
