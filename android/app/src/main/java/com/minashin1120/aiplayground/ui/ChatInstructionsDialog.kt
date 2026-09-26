package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.AutoSystemPrompt
import com.minashin1120.aiplayground.data.Preferences
import org.json.JSONObject

/** The user-level fields of `#thread-modal`, filled from the settings the modal loads. */
@Stable
private class ThreadPromptForm(prefs: Preferences?) {
    var userPrompt by mutableStateOf(prefs?.systemPrompt.orEmpty())
    var userPromptEnabled by mutableStateOf(prefs?.systemPromptEnabled ?: true)
    var applyAutoNotices by mutableStateOf(prefs?.applyAutoSystemPromptNotices ?: true)
    val autoPrompts = mutableStateListOf<AutoSystemPrompt>().apply { addAll(prefs?.autoSystemPrompts.orEmpty()) }

    fun update(key: String, transform: (AutoSystemPrompt) -> AutoSystemPrompt) {
        val index = autoPrompts.indexOfFirst { it.key == key }
        if (index >= 0) autoPrompts[index] = transform(autoPrompts[index])
    }

    /** Web `resetAutoSystemPromptConfigToCodeDefaults('thread', …)`. */
    fun reset() {
        applyAutoNotices = true
        for (i in autoPrompts.indices) {
            val row = autoPrompts[i]
            autoPrompts[i] = row.copy(enabled = if (row.mcpLocked) row.enabled else true, text = row.defaultText)
        }
    }

    /** Web `userPromptPayload`; the MCP row is always saved as enabled. */
    fun payload(): JSONObject = JSONObject()
        .put("system_prompt", userPrompt)
        .put("system_prompt_enabled", userPromptEnabled)
        .put("apply_auto_system_prompt_notices", applyAutoNotices)
        .put("auto_system_prompt_notices_config", JSONObject().apply {
            autoPrompts.forEach { row -> put(row.key, JSONObject().put("enabled", if (row.mcpLocked) true else row.enabled).put("text", row.text)) }
        })
}

/**
 * `#thread-modal` (Chat Instructions), opened from the SysPrompt ⚙: this chat's system prompt and
 * whether the global prompt is included, the global prompt preview, and the user system prompt with
 * the compact auto-injected prompt rows.
 */
@Composable
internal fun ChatInstructionsDialog(
    state: ChatState,
    onRefresh: () -> Unit,
    onDismiss: () -> Unit,
    onSave: (instruction: String, includeGlobal: Boolean, userPrompt: JSONObject, done: (Boolean) -> Unit) -> Unit,
) {
    val web = LocalWebPalette.current
    val thread = state.selected ?: return
    LaunchedEffect(thread.id) { onRefresh() }
    var instruction by remember(thread.id, state.customInstruction) { mutableStateOf(state.customInstruction) }
    var includeGlobal by remember(thread.id, state.includeGlobalInstruction) { mutableStateOf(state.includeGlobalInstruction) }
    val prefs = state.preferences
    val form = remember(prefs) { ThreadPromptForm(prefs) }
    var saving by remember { mutableStateOf(false) }
    val border = web.twBorder(Tw.gray700)
    WebOverlayModal(onDismiss, grayOverlay(), 4.dp) { phone ->
        Column(
            Modifier.fillMaxSize().verticalScroll(rememberScrollState()),
            verticalArrangement = if (phone) Arrangement.Top else Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            val shape = RoundedCornerShape(8.dp)
            Column(
                Modifier.padding(16.dp).widthIn(max = 448.dp).fillMaxWidth().clip(shape).background(web.twBg(Tw.gray800))
                    .border(1.dp, border, shape).padding(24.dp),
            ) {
                Row(Modifier.padding(bottom = 16.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    FaIcon(R.drawable.fa_solid_comment_dots, null, size = 20.dp, tint = web.twText(Tw.green400))
                    Text("Chat Instructions", fontSize = 20.sp, lineHeight = 28.sp, fontWeight = FontWeight.Bold, color = web.twText(Tw.white))
                }
                Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                    Column {
                        TwLabel("このチャット専用のシステムプロンプト", Modifier.padding(bottom = 4.dp))
                        TwInput(instruction, { instruction = it.take(100_000) }, "このチャットだけで有効にしたい指示（役割や出力形式など）を入力してください...",
                            singleLine = false, height = 192.dp, padding = 12.dp)
                        Text("※ 全体システムプロンプトが有効な場合はその後に追加されます。SysPromptのON/OFFに関わらず、このチャット専用の指示は適用されます。",
                            fontSize = 10.sp, lineHeight = 15.sp, color = web.twText(Tw.gray500), modifier = Modifier.padding(top = 8.dp))
                    }
                    Row(
                        Modifier.fillMaxWidth().clip(RoundedCornerShape(4.dp)).background(web.twBg(Tw.gray900).copy(alpha = 0.5f))
                            .border(1.dp, border.copy(alpha = border.alpha * 0.5f), RoundedCornerShape(4.dp))
                            .padding(horizontal = 4.dp, vertical = 8.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text("全体プロンプトをこのチャットに含める", fontSize = 11.sp, lineHeight = 16.sp, color = web.twText(Tw.gray300),
                            modifier = Modifier.weight(1f))
                        SmallGreenToggle(includeGlobal) { includeGlobal = it }
                    }
                    Column(Modifier.fillMaxWidth()) {
                        Box(Modifier.fillMaxWidth().height(1.dp).background(border))
                        TwLabel("全体システムプロンプト（参照のみ）", Modifier.padding(top = 16.dp, bottom = 8.dp))
                        TwInput(prefs?.globalSystemPromptEffective.orEmpty(), {}, "全体システムプロンプト", singleLine = false, height = 80.dp,
                            fontSize = 12.sp, background = Tw.gray950, borderColor = Tw.gray700, textColor = Tw.gray200, readOnly = true)
                        if (prefs != null) Text(
                            when {
                                !prefs.globalSystemPromptEnabled -> "現在は無効化されています。"
                                prefs.globalSystemPromptUsesTimeFallback -> "管理者設定が空欄のため、時刻の既定プロンプトが適用されています。"
                                else -> "管理者が設定した全体システムプロンプトが適用されています。"
                            },
                            fontSize = 10.sp, lineHeight = 15.sp, color = web.twText(Tw.gray500), modifier = Modifier.padding(top = 8.dp),
                        )
                        TwLabel("ユーザー設定のシステムプロンプト（この歯車からも編集可能）", Modifier.padding(top = 16.dp, bottom = 8.dp))
                        Row(Modifier.fillMaxWidth().padding(bottom = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                            Text("ユーザーシステムプロンプト", fontSize = 11.sp, lineHeight = 16.sp, color = web.twText(Tw.gray500), modifier = Modifier.weight(1f))
                            TwCheck("有効", form.userPromptEnabled) { form.userPromptEnabled = it }
                        }
                        TwInput(form.userPrompt, { form.userPrompt = it.take(100_000) }, "自分だけに適用する指示", singleLine = false,
                            height = 96.dp, fontSize = 12.sp)
                        if (form.autoPrompts.isNotEmpty()) ThreadAutoPrompts(form)
                    }
                }
                Row(Modifier.fillMaxWidth().padding(top = 24.dp), horizontalArrangement = Arrangement.spacedBy(12.dp, Alignment.End)) {
                    Text("キャンセル", fontSize = 16.sp, lineHeight = 24.sp, color = web.twText(Tw.gray400),
                        modifier = Modifier.clip(RoundedCornerShape(4.dp)).clickable(role = Role.Button, onClick = onDismiss)
                            .padding(horizontal = 16.dp, vertical = 8.dp))
                    Text(if (saving) "保存中..." else "保存", fontSize = 16.sp, lineHeight = 24.sp, fontWeight = FontWeight.Bold, color = Tw.white,
                        modifier = Modifier.alpha(if (saving) 0.5f else 1f).clip(RoundedCornerShape(4.dp)).background(Tw.green600)
                            .clickable(enabled = !saving, role = Role.Button) {
                                saving = true
                                onSave(instruction, includeGlobal, form.payload()) { saving = false }
                            }
                            .padding(horizontal = 24.dp, vertical = 8.dp))
                }
            }
        }
    }
}

/** `#thread-auto-sys-prompt-settings`: the compact rows (`buildAutoSystemPromptRows('thread', true)`). */
@Composable
private fun ThreadAutoPrompts(form: ThreadPromptForm) {
    val web = LocalWebPalette.current
    val border = web.twBorder(Tw.gray700)
    Column(Modifier.fillMaxWidth().padding(top = 12.dp)) {
        Box(Modifier.fillMaxWidth().height(1.dp).background(border))
        Row(Modifier.fillMaxWidth().padding(top = 12.dp, bottom = 8.dp), verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            TwLabel("自動注入システムプロンプト（ユーザー単位）", Modifier.weight(1f))
            Text("既定に戻す", fontSize = 10.sp, lineHeight = 15.sp, fontWeight = FontWeight.Bold, color = Tw.white,
                modifier = Modifier.clip(RoundedCornerShape(4.dp)).background(web.twBg(Tw.gray700))
                    .clickable(role = Role.Button, onClick = form::reset).padding(horizontal = 8.dp, vertical = 4.dp))
            TwCheck("全体適用", form.applyAutoNotices) { form.applyAutoNotices = it }
        }
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            form.autoPrompts.forEach { row ->
                val shape = RoundedCornerShape(4.dp)
                Column(
                    Modifier.fillMaxWidth().clip(shape).background(web.twBg(Tw.gray950).copy(alpha = 0.4f)).border(1.dp, border, shape).padding(8.dp),
                ) {
                    Row(Modifier.fillMaxWidth().padding(bottom = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                        Text(row.label, fontSize = 11.sp, lineHeight = 16.sp, color = web.twText(Tw.gray300), modifier = Modifier.weight(1f))
                        TwCheck("適用", row.enabled, enabled = !row.mcpLocked) { value -> form.update(row.key) { it.copy(enabled = value) } }
                    }
                    TwInput(row.text, { value -> form.update(row.key) { it.copy(text = value.take(100_000)) } },
                        row.defaultText.ifBlank { "自動注入文言" }, singleLine = false, height = 56.dp, fontSize = 11.sp,
                        background = Tw.gray950, borderColor = Tw.gray700, textColor = Tw.gray200)
                    if (row.hint.isNotBlank()) Text(row.hint, fontSize = 10.sp, lineHeight = 15.sp, color = web.twText(Tw.gray500),
                        modifier = Modifier.padding(top = 4.dp))
                    if (row.mcpLocked) Text(
                        "この項目のオン・オフはプロンプトバーのMCPスイッチに連動します（オフ時は案内文の注入とツール付与自体が無効）。文面は編集できます。",
                        fontSize = 10.sp, lineHeight = 15.sp, color = web.twText(Tw.cyan300).copy(alpha = 0.7f), modifier = Modifier.padding(top = 4.dp),
                    )
                }
            }
        }
    }
}

@Composable
private fun TwLabel(text: String, modifier: Modifier = Modifier) {
    Text(text, fontSize = 12.sp, lineHeight = 16.sp, color = LocalWebPalette.current.twText(Tw.gray400), modifier = modifier)
}

/** `label.flex.items-center.gap-1.text-[10px].text-gray-500` wrapping a checkbox. */
@Composable
private fun TwCheck(label: String, checked: Boolean, enabled: Boolean = true, onChange: (Boolean) -> Unit) {
    Row(
        Modifier.toggleable(checked, enabled = enabled, role = Role.Checkbox, onValueChange = onChange),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        WebCheckbox(checked, null, enabled = enabled, modifier = Modifier.alpha(if (enabled) 1f else 0.5f))
        Text(label, fontSize = 10.sp, lineHeight = 15.sp, color = LocalWebPalette.current.twText(Tw.gray500))
    }
}

/** `w-8 h-4` switch with a 12px knob and `peer-checked:bg-green-600`. */
@Composable
private fun SmallGreenToggle(checked: Boolean, onChange: (Boolean) -> Unit) {
    val web = LocalWebPalette.current
    Box(
        Modifier.size(width = 32.dp, height = 16.dp).clip(CircleShape)
            .background(if (checked) Tw.green600 else web.twBg(Tw.gray700))
            .toggleable(checked, role = Role.Switch, onValueChange = onChange),
    ) {
        Box(Modifier.offset(x = if (checked) 14.dp else 2.dp, y = 2.dp).size(12.dp).clip(CircleShape).background(Color.White))
    }
}
