package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.compositionLocalOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.ChatMessage
import com.minashin1120.aiplayground.data.MathStepPart
import com.minashin1120.aiplayground.data.MathStepsConfig
import com.minashin1120.aiplayground.data.buildMathStepViews
import com.minashin1120.aiplayground.data.parseMathSteps
import kotlinx.coroutines.delay

/**
 * 「数学ステップ解説」: one step of an answer at a time, with 次へ / 前へ / 最初から and
 * 「もっとやさしく」. Progress is saved on the server (`/api/math_steps/progress`) so another
 * device continues where this one stopped; the answer is never cut up on the server.
 */

/** Whether animated SVG diagrams may run (the user's animation option and reduced-motion). */
internal val LocalSvgAnimation = compositionLocalOf { true }

/** Server calls of the step view; [ChatViewModel] supplies them through the conversation. */
internal class MathStepsActions(
    val onProgress: (ChatMessage, Int, Int) -> Unit = { _, _, _ -> },
    /** Calls `/api/math_steps/simplify`; [done] gets the whole easier step list, or null on failure. */
    val onSimplify: (ChatMessage, Int, Int?, (List<MathStepPart>?) -> Unit) -> Unit =
        { _, _, _, done -> done(null) },
    /** Removes the saved replacement (expanded step or legacy simplified body) on the server. */
    val onRestoreOriginal: (ChatMessage, Int, Int) -> Unit = { _, _, _ -> },
)

/** The five gear options, shared by the settings card and the prompt-bar gear modal. */
@Composable
internal fun MathStepsConfigFields(config: MathStepsConfig, onChange: (MathStepsConfig) -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Column {
            SettingsCheck("基礎から説明する", config.basics, { onChange(config.copy(basics = it)) }, boxSize = 18.dp)
            SettingsDesc("最初に必要な用語や公式のおさらいを入れます。", Modifier.padding(start = 28.dp, top = 2.dp))
        }
        Column {
            SettingsFieldLabel("説明の詳しさ", Modifier.padding(bottom = 4.dp))
            SettingsSelect(
                value = config.level,
                options = webOptions(
                    "easy" to "やさしい（ことば中心）",
                    "normal" to "ふつう",
                    "detailed" to "くわしい（式変形も丁寧に）",
                ),
                onSelect = { onChange(config.copy(level = it)) },
            )
        }
        Column {
            SettingsFieldLabel("1ステップの大きさ", Modifier.padding(bottom = 4.dp))
            SettingsSelect(
                value = config.stepSize,
                options = webOptions(
                    "small" to "小さめ（多いステップ）",
                    "normal" to "ふつう",
                    "large" to "大きめ（少ないステップ）",
                ),
                onSelect = { onChange(config.copy(stepSize = it)) },
            )
        }
        Column {
            SettingsFieldLabel("図の使用", Modifier.padding(bottom = 4.dp))
            SettingsSelect(
                value = config.diagrams,
                options = webOptions("auto" to "必要なら使う（おすすめ）", "off" to "使わない"),
                onSelect = { onChange(config.copy(diagrams = it)) },
            )
        }
        Column {
            SettingsCheck("図のアニメーションを使う", config.animation, { onChange(config.copy(animation = it)) }, boxSize = 18.dp)
            SettingsDesc("端末が「視差効果を減らす」設定なら自動で静止します。", Modifier.padding(start = 28.dp, top = 2.dp))
        }
    }
}

/** `#math-steps-modal`: the prompt-bar gear. */
@Composable
internal fun MathStepsSettingsDialog(
    config: MathStepsConfig,
    onDismiss: () -> Unit,
    onSave: (MathStepsConfig) -> Unit,
) {
    var draft by remember { mutableStateOf(config) }
    PlaygroundDialog(
        onDismissRequest = onDismiss,
        title = { Text("数学ステップの設定") },
        text = {
            Column(Modifier.fillMaxWidth().verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                MathStepsConfigFields(draft, onChange = { draft = it })
            }
        },
        confirmButton = {
            WebButton(onClick = { onSave(draft); onDismiss() }, variant = WebButtonVariant.Primary) { Text("保存") }
        },
        dismissButton = { WebButton(onClick = onDismiss, variant = WebButtonVariant.Ghost) { Text("キャンセル") } },
        icon = R.drawable.fa_solid_lightbulb,
        subtitle = "説明の詳しさと進め方を選びます",
        tone = WebModalTone.Info,
        panelMaxWidth = 560.dp,
    )
}

/** One answer rendered as steps. Falls back to the normal Markdown when no heading is found. */
@Composable
internal fun MathStepsContent(
    text: String,
    message: ChatMessage,
    actions: MathStepsActions,
    config: MathStepsConfig,
    loader: FileBytesLoader?,
    onOpen: (String) -> Unit,
    startCollapsed: Boolean,
    canvasMode: Boolean,
    flashQuote: String? = null,
    flashAlpha: Float = 0f,
    onQuoteFlashLocated: ((Float) -> Unit)? = null,
) {
    val original = remember(text) { parseMathSteps(text) }
    if (original.isEmpty()) {
        MarkdownText(text, loader, onOpen, startCollapsed, canvasMode, flashQuote, flashAlpha, onQuoteFlashLocated)
        return
    }
    val saved = message.mathStepsState
    val stateKey = message.conversationKey ?: message.id
    var expanded by remember(stateKey) { mutableStateOf(saved?.expanded.orEmpty()) }
    var simplified by remember(stateKey) { mutableStateOf(saved?.simplified.orEmpty()) }
    val steps = remember(original, expanded) { buildMathStepViews(original, expanded) }
    val lastIndex = steps.lastIndex
    var current by remember(stateKey) { mutableIntStateOf((saved?.current ?: 0).coerceIn(0, lastIndex)) }
    var maxReached by remember(stateKey) { mutableIntStateOf((saved?.maxReached ?: current).coerceIn(0, lastIndex)) }
    var savedCurrent by remember(stateKey) { mutableIntStateOf(saved?.current ?: 0) }
    var simplifying by remember(stateKey) { mutableStateOf(false) }
    var simplifyFailed by remember(stateKey) { mutableStateOf(false) }

    // A regenerated/shortened answer must never index past the last step.
    LaunchedEffect(lastIndex) {
        if (current > lastIndex) current = lastIndex
        if (maxReached > lastIndex) maxReached = lastIndex
    }

    // Another device (or this one after a reload) may have moved on: follow the server state.
    LaunchedEffect(saved, lastIndex) {
        if (saved == null) return@LaunchedEffect
        val next = saved.current.coerceIn(0, lastIndex)
        if (next != current) current = next
        if (saved.maxReached > maxReached) maxReached = saved.maxReached.coerceAtMost(lastIndex)
        if (saved.expanded != expanded) expanded = saved.expanded
        if (saved.simplified != simplified) simplified = saved.simplified
        savedCurrent = next
    }

    // Optimistic: the step changes immediately; the progress request is debounced and quiet.
    LaunchedEffect(current, maxReached) {
        if (current == savedCurrent) return@LaunchedEffect
        delay(400)
        actions.onProgress(message, current, maxReached)
        savedCurrent = current
    }

    fun goTo(index: Int) {
        val next = index.coerceIn(0, lastIndex)
        current = next
        if (next > maxReached) maxReached = next
        simplifyFailed = false
    }

    val web = LocalWebPalette.current
    val view = steps[current]
    val title = view.title
    Column(Modifier.fillMaxWidth().semantics { contentDescription = "数学ステップ解説" },
        verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Row(
            Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp))
                .background(web.theme.rgb(if (web.isLight) 0.06f else 0.10f))
                .padding(horizontal = 10.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            FaIcon(R.drawable.fa_solid_lightbulb, null, size = 12.dp, tint = web.theme300)
            Column(Modifier.weight(1f)) {
                Text(
                    "ステップ ${current + 1} / ${steps.size}",
                    fontSize = 12.sp,
                    lineHeight = 17.sp,
                    fontWeight = FontWeight.Bold,
                    color = if (web.isLight) web.theme.t700 else web.theme200,
                )
                if (title.isNotBlank()) {
                    Text(title, fontSize = 11.sp, lineHeight = 16.sp, color = web.muted)
                }
            }
        }

        val easyBody = if (view.expanded) view.body else simplified[view.sourceIndex]
        val canRestore = view.expanded || easyBody != null
        CompositionLocalProvider(LocalSvgAnimation provides config.animation) {
            if (easyBody != null) {
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    MarkdownText(easyBody, loader, onOpen, startCollapsed, canvasMode)
                    if (canRestore) {
                        Text(
                            "元の説明に戻す",
                            fontSize = 11.sp,
                            color = web.theme300,
                            modifier = Modifier.clip(RoundedCornerShape(6.dp))
                                .clickable(role = Role.Button) {
                                    val sourceIndex = view.sourceIndex
                                    var restoredIndex = current
                                    if (view.expanded) {
                                        val without = expanded - sourceIndex
                                        val nextSteps = buildMathStepViews(original, without)
                                        val first = nextSteps.indexOfFirst { it.sourceIndex == sourceIndex }
                                            .takeIf { it >= 0 } ?: 0
                                        expanded = without
                                        simplified = simplified - sourceIndex
                                        current = first.coerceIn(0, nextSteps.lastIndex)
                                        if (maxReached > nextSteps.lastIndex) maxReached = nextSteps.lastIndex
                                        restoredIndex = current
                                    } else {
                                        simplified = simplified - sourceIndex
                                    }
                                    actions.onRestoreOriginal(message, sourceIndex, restoredIndex)
                                }
                                .padding(vertical = 6.dp, horizontal = 4.dp),
                        )
                    }
                }
            } else {
                MarkdownText(
                    view.body, loader, onOpen, startCollapsed, canvasMode,
                    flashQuote = flashQuote, flashAlpha = flashAlpha, onQuoteFlashLocated = onQuoteFlashLocated,
                )
            }
        }

        if (simplifyFailed) {
            Text("やさしい説明を作れませんでした",
                fontSize = 11.sp, lineHeight = 16.sp, color = Tw.rose300)
        }

        Row(
            Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            // Web `.math-steps-actions`: the large next/finish button first, then the small actions.
            WebButton(
                onClick = { if (current < lastIndex) goTo(current + 1) },
                modifier = Modifier.fillMaxWidth().heightIn(min = 44.dp),
                variant = if (current < lastIndex) WebButtonVariant.Primary else WebButtonVariant.Gray,
                enabled = current < lastIndex,
                contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 22.dp, vertical = 12.dp),
                fontSize = 14.sp,
            ) { Text(if (current < lastIndex) "わかった、次へ" else "完了") }
        }

        Row(
            Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            WebButton(
                onClick = { goTo(current - 1) },
                variant = WebButtonVariant.Gray,
                enabled = current > 0,
                contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 12.dp, vertical = 8.dp),
                fontSize = 12.sp,
            ) { Text("前へ") }
            Text(
                "最初から",
                fontSize = 11.sp,
                color = web.muted,
                modifier = Modifier.clip(RoundedCornerShape(6.dp))
                    .clickable(enabled = current > 0, role = Role.Button) { goTo(0) }
                    .padding(vertical = 10.dp, horizontal = 6.dp),
            )
            Spacer(Modifier.weight(1f))
            WebButton(
                onClick = {
                    if (simplifying) return@WebButton
                    val target = view
                    simplifying = true
                    simplifyFailed = false
                    actions.onSimplify(message, target.sourceIndex, target.partIndex) { parts ->
                        simplifying = false
                        if (parts.isNullOrEmpty()) simplifyFailed = true
                        else {
                            expanded = expanded + (target.sourceIndex to parts)
                            simplified = simplified - target.sourceIndex
                        }
                    }
                },
                variant = WebButtonVariant.Gray,
                enabled = !simplifying && message.id.toIntOrNull() != null,
                contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 12.dp, vertical = 8.dp),
                fontSize = 12.sp,
            ) {
                FaIcon(R.drawable.fa_solid_wand_magic_sparkles, null, size = 12.dp)
                Text(if (simplifying) "やさしくしています…" else "もっとやさしく")
            }
        }
    }
}

/** `#math-steps-gear` in the prompt bar: a cog inside the 数学ステップ chip. */
@Composable
internal fun MathStepsGear(onClick: () -> Unit) {
    val web = LocalWebPalette.current
    Box(
        Modifier.size(22.dp).clip(CircleShape)
            .clickable(role = Role.Button, onClick = onClick)
            .semantics { contentDescription = "数学ステップの設定" },
        contentAlignment = Alignment.Center,
    ) {
        FaIcon(R.drawable.fa_solid_cog, null, size = 10.dp,
            tint = if (web.isLight) Color(3, 105, 161) else Color(125, 211, 252))
    }
}

/** Only the sizes and names the server accepts are sent; the gear never writes free text. */
internal fun MathStepsConfig.withAllowedValues(): MathStepsConfig = copy(
    level = level.takeIf { it in MathStepsConfig.LEVELS } ?: "easy",
    stepSize = stepSize.takeIf { it in MathStepsConfig.STEP_SIZES } ?: "normal",
    diagrams = diagrams.takeIf { it in MathStepsConfig.DIAGRAMS } ?: "auto",
)
