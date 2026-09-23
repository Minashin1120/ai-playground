package com.minashin1120.aiplayground.ui

import android.content.Context
import android.provider.Settings
import androidx.activity.compose.PredictiveBackHandler
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.EnterTransition
import androidx.compose.animation.ExitTransition
import androidx.compose.animation.core.CubicBezierEasing
import androidx.compose.animation.core.Easing
import androidx.compose.animation.core.FastOutLinearInEasing
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.FiniteAnimationSpec
import androidx.compose.animation.core.MutableTransitionState
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.VisibilityThreshold
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.snap
import androidx.compose.animation.core.tween
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.interaction.InteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.Box
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.compositionLocalOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.composed
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.LifecycleResumeEffect
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay

/** Shared motion tokens so every surface moves with the same rhythm as the Web `uiEnter*` family. */
internal object PlaygroundMotion {
    const val SHORT = 160
    const val MEDIUM = 260
    const val LONG = 340
    const val EMPHASIZED = 420

    /** Exit length of modal panels; hosts keep the dialog composed at least this long after closing. */
    const val MODAL_EXIT = 200

    val Standard: Easing = FastOutSlowInEasing
    val Emphasized: Easing = CubicBezierEasing(0.2f, 0f, 0f, 1f)
    val Exit: Easing = FastOutLinearInEasing
}

internal fun areSystemAnimationsDisabled(context: Context): Boolean =
    runCatching {
        Settings.Global.getFloat(
            context.contentResolver,
            Settings.Global.ANIMATOR_DURATION_SCALE,
            1f,
        ) == 0f
    }.getOrDefault(false)

/** True when the system "remove animations" setting is on; every Playground motion collapses to an instant change. */
internal val LocalReduceMotion = compositionLocalOf { false }

/** Reads the system animation setting and re-reads it on every resume so toggling it applies without a restart. */
@Composable
internal fun rememberReduceMotion(): Boolean {
    val context = LocalContext.current
    var reduce by remember { mutableStateOf(areSystemAnimationsDisabled(context)) }
    LifecycleResumeEffect(context) {
        reduce = areSystemAnimationsDisabled(context)
        onPauseOrDispose { }
    }
    return reduce
}

internal fun <T> motionTween(
    reduce: Boolean,
    durationMillis: Int = PlaygroundMotion.MEDIUM,
    easing: Easing = PlaygroundMotion.Standard,
): FiniteAnimationSpec<T> = if (reduce) snap() else tween(durationMillis, easing = easing)

/** Vertical reveal for panels and rows that push surrounding content. */
internal fun expandFadeIn(reduce: Boolean): EnterTransition =
    if (reduce) EnterTransition.None
    else expandVertically(tween(PlaygroundMotion.LONG, easing = PlaygroundMotion.Standard)) +
        fadeIn(tween(PlaygroundMotion.MEDIUM, easing = PlaygroundMotion.Standard))

internal fun shrinkFadeOut(reduce: Boolean): ExitTransition =
    if (reduce) ExitTransition.None
    else shrinkVertically(tween(PlaygroundMotion.LONG, easing = PlaygroundMotion.Standard)) +
        fadeOut(tween(PlaygroundMotion.MEDIUM, easing = PlaygroundMotion.Standard))

/** Small scale-and-fade used for chips, buttons and badges (Web `tagPopIn` / `btn-swap-pop`). */
internal fun popIn(reduce: Boolean): EnterTransition =
    if (reduce) EnterTransition.None
    else scaleIn(tween(PlaygroundMotion.MEDIUM, easing = PlaygroundMotion.Emphasized), initialScale = 0.92f) +
        fadeIn(tween(PlaygroundMotion.SHORT))

internal fun popOut(reduce: Boolean): ExitTransition =
    if (reduce) ExitTransition.None
    else scaleOut(tween(PlaygroundMotion.SHORT, easing = PlaygroundMotion.Exit), targetScale = 0.92f) +
        fadeOut(tween(PlaygroundMotion.SHORT))

/** Full-screen phone panels rise slightly (Web `uiEnterUp`); floating panels scale up (Web `uiEnterScale`). */
internal fun modalEnter(fullScreen: Boolean, reduce: Boolean): EnterTransition = when {
    reduce -> EnterTransition.None
    fullScreen -> slideInVertically(tween(PlaygroundMotion.LONG, easing = PlaygroundMotion.Emphasized)) { it / 10 } +
        fadeIn(tween(PlaygroundMotion.MEDIUM))
    else -> scaleIn(tween(PlaygroundMotion.MEDIUM, easing = PlaygroundMotion.Emphasized), initialScale = 0.94f) +
        fadeIn(tween(PlaygroundMotion.SHORT + 40))
}

internal fun modalExit(fullScreen: Boolean, reduce: Boolean): ExitTransition = when {
    reduce -> ExitTransition.None
    fullScreen -> slideOutVertically(tween(PlaygroundMotion.MODAL_EXIT, easing = PlaygroundMotion.Exit)) { it / 12 } +
        fadeOut(tween(PlaygroundMotion.MODAL_EXIT))
    else -> scaleOut(tween(PlaygroundMotion.MODAL_EXIT, easing = PlaygroundMotion.Exit), targetScale = 0.96f) +
        fadeOut(tween(PlaygroundMotion.MODAL_EXIT))
}

/** Scale applied to a modal panel while a predictive back gesture is in progress. */
internal fun predictiveBackScale(progress: Float): Float = 1f - 0.08f * progress.coerceIn(0f, 1f)

/** Whether the surrounding [ModalHost] wants its modal shown; false while the exit animation plays. */
internal val LocalModalVisible = compositionLocalOf { true }

/**
 * Keeps a modal composed until its exit animation has finished so closing animates instead of cutting.
 * Content inside is created fresh on every open, like the previous `if (open)` call sites.
 */
@Composable
internal fun ModalHost(visible: Boolean, content: @Composable () -> Unit) {
    val reduce = LocalReduceMotion.current
    var present by remember { mutableStateOf(visible) }
    LaunchedEffect(visible) {
        if (visible) {
            present = true
        } else {
            if (!reduce) delay(PlaygroundMotion.MODAL_EXIT + 40L)
            present = false
        }
    }
    if (!visible && !present) return
    CompositionLocalProvider(LocalModalVisible provides visible) { content() }
}

/** [ModalHost] for modals driven by a nullable value; the last value is kept while the modal animates out. */
@Composable
internal fun <T : Any> ModalValueHost(value: T?, content: @Composable (T) -> Unit) {
    val latest = remember { mutableStateOf<T?>(null) }
    SideEffect { if (value != null) latest.value = value }
    ModalHost(visible = value != null) {
        (value ?: latest.value)?.let { content(it) }
    }
}

/**
 * Animates a modal panel in and out and follows the predictive back gesture before dismissing.
 * Place it directly inside the `Dialog` content.
 */
@Composable
internal fun ModalPanelMotion(
    fullScreen: Boolean,
    onDismissRequest: () -> Unit,
    modifier: Modifier = Modifier,
    content: @Composable () -> Unit,
) {
    val reduce = LocalReduceMotion.current
    val visible = LocalModalVisible.current
    val transition = remember { MutableTransitionState(false) }
    transition.targetState = visible
    var backProgress by remember { mutableFloatStateOf(0f) }
    LaunchedEffect(visible) { if (visible) backProgress = 0f }
    PredictiveBackHandler(enabled = visible) { events ->
        try {
            events.collect { event -> backProgress = event.progress }
            onDismissRequest()
        } catch (e: CancellationException) {
            backProgress = 0f
            throw e
        }
    }
    AnimatedVisibility(
        visibleState = transition,
        modifier = modifier,
        enter = modalEnter(fullScreen, reduce),
        exit = modalExit(fullScreen, reduce),
        label = "modal panel",
    ) {
        Box(Modifier.graphicsLayer {
            val scale = if (reduce) 1f else predictiveBackScale(backProgress)
            scaleX = scale
            scaleY = scale
        }) { content() }
    }
}

/** Keeps the last non-null value so content can still render while it animates out. */
@Composable
internal fun <T : Any> rememberRetained(value: T?): T? {
    val latest = remember { mutableStateOf<T?>(null) }
    SideEffect { if (value != null) latest.value = value }
    return value ?: latest.value
}

/** Plays [enter] once when first composed; used for rows that only ever appear (cards, chips). */
@Composable
internal fun AppearOnce(
    enter: (Boolean) -> EnterTransition = ::expandFadeIn,
    modifier: Modifier = Modifier,
    content: @Composable () -> Unit,
) {
    val reduce = LocalReduceMotion.current
    val state = remember { MutableTransitionState(reduce).apply { targetState = true } }
    AnimatedVisibility(visibleState = state, modifier = modifier, enter = enter(reduce), exit = ExitTransition.None, label = "appear once") {
        content()
    }
}

/** Fades and lifts content in, delayed by [index] steps (Web `uiEnterUp` stagger). */
@Composable
internal fun StaggerIn(index: Int, animate: Boolean = true, content: @Composable () -> Unit) {
    val reduce = LocalReduceMotion.current
    val progress = remember { Animatable(if (reduce || !animate) 1f else 0f) }
    LaunchedEffect(Unit) {
        if (progress.value < 1f) {
            progress.animateTo(1f, tween(PlaygroundMotion.LONG, delayMillis = staggerDelay(index), easing = PlaygroundMotion.Emphasized))
        }
    }
    val lift = with(LocalDensity.current) { 12.dp.toPx() }
    Box(Modifier.graphicsLayer {
        alpha = progress.value
        translationY = (1f - progress.value) * lift
    }) { content() }
}

internal fun staggerDelay(index: Int): Int = index.coerceIn(0, 8) * 40

/** Slightly shrinks a pressable surface while it is held. */
internal fun Modifier.pressScale(interactionSource: InteractionSource): Modifier = composed {
    val reduce = LocalReduceMotion.current
    val pressed by interactionSource.collectIsPressedAsState()
    val scale by animateFloatAsState(
        targetValue = if (pressed && !reduce) 0.98f else 1f,
        animationSpec = motionTween(reduce, PlaygroundMotion.SHORT),
        label = "press scale",
    )
    graphicsLayer {
        scaleX = scale
        scaleY = scale
    }
}

/** Fade used when list rows are inserted or removed; null disables it under reduced motion. */
internal fun listFade(reduce: Boolean): FiniteAnimationSpec<Float>? =
    if (reduce) null else tween(PlaygroundMotion.MEDIUM, easing = PlaygroundMotion.Standard)

/** Movement of list rows that shift because others were inserted, removed or reordered. */
internal fun listPlacement(reduce: Boolean): FiniteAnimationSpec<IntOffset>? =
    if (reduce) null else spring(dampingRatio = Spring.DampingRatioNoBouncy, stiffness = Spring.StiffnessMediumLow, visibilityThreshold = IntOffset.VisibilityThreshold)
