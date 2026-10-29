package com.minashin1120.aiplayground.ui

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.AnimationVector1D
import androidx.compose.animation.core.tween
import androidx.compose.foundation.layout.Box
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.unit.dp
import com.minashin1120.aiplayground.data.ActivityLog

/** Shared by stable conversation keys, including the live-to-stored answer handoff. */
internal class AnswerBubbleMotions {
    private val entries = mutableMapOf<String, Animatable<Float, AnimationVector1D>>()

    fun entry(key: String, animate: Boolean): Animatable<Float, AnimationVector1D> =
        entries.getOrPut(key) { Animatable(if (animate) 0f else 1f) }
}

/** Keep the answer's layout in place while its prompt travels, then enter from the left. */
@Composable
internal fun ConversationMessageEntry(
    key: String,
    answer: Boolean,
    animate: Boolean,
    answers: AnswerBubbleMotions,
    content: @Composable () -> Unit,
) {
    if (!answer) {
        StaggerIn(0, animate = animate, content = content)
        return
    }
    val reduce = LocalReduceMotion.current
    val progress = remember(answers, key) { answers.entry(key, animate && !reduce) }
    val waiting = progress.value < 1f && LocalPromptSendMotion.current?.id != null
    LaunchedEffect(waiting, reduce) {
        if (reduce) progress.snapTo(1f)
        else if (!waiting && progress.value < 1f) {
            ActivityLog.log("ui.show")
            try { progress.animateTo(1f, tween(280, easing = PlaygroundMotion.WebEaseOut)) }
            finally { ActivityLog.log("ui.hide") }
        }
    }
    val distance = with(LocalDensity.current) { 24.dp.toPx() }
    Box(Modifier
        .then(if (waiting) Modifier.clearAndSetSemantics { } else Modifier)
        .graphicsLayer {
            alpha = if (waiting) 0f else progress.value
            translationX = -distance * (1f - progress.value)
        }) { content() }
}
