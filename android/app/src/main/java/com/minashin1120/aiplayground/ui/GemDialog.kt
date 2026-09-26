package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.runtime.snapshots.SnapshotStateList
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.FixedPrompt
import com.minashin1120.aiplayground.data.Gem

/** Web `collectGemFixedPrompts()`: rows with both a name and content, trimmed; empty rows are ignored. */
internal fun collectGemFixedPrompts(rows: List<FixedPrompt>): List<FixedPrompt> =
    rows.map { FixedPrompt(it.name.trim(), it.content.trim()) }.filter { it.name.isNotEmpty() && it.content.isNotEmpty() }

/**
 * `#gem-modal`: Create New Gem / Edit Gem with Name, Description (Optional), System Instruction,
 * Default Model (optional) and the Gem's Fixed Prompts. [onSave] gets the collected values and a
 * callback telling whether the save finished.
 */
@Composable
internal fun GemEditorDialog(
    gem: Gem?,
    onDismiss: () -> Unit,
    onSave: (name: String, description: String, instruction: String, fixedPrompts: List<FixedPrompt>, done: (Boolean) -> Unit) -> Unit,
) {
    val web = LocalWebPalette.current
    var name by remember { mutableStateOf(gem?.name.orEmpty()) }
    var description by remember { mutableStateOf(gem?.description.orEmpty()) }
    var instruction by remember { mutableStateOf(gem?.instruction.orEmpty()) }
    // Web offers only "Use current model" here, so the option is always the empty value.
    var defaultModel by remember { mutableStateOf("") }
    val rows = remember { mutableStateListOf<FixedPrompt>().apply { addAll(gem?.fixedPrompts.orEmpty()) } }
    var saving by remember { mutableStateOf(false) }
    var missing by remember { mutableStateOf(false) }
    val border = web.twBorder(Tw.gray700)
    WebOverlayModal(onDismiss, grayOverlay(), 4.dp) { phone ->
        Column(
            Modifier.fillMaxSize().verticalScroll(rememberScrollState()),
            verticalArrangement = if (phone) Arrangement.Top else Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            val shape = RoundedCornerShape(8.dp)
            Column(
                Modifier.padding(16.dp).widthIn(max = 512.dp).fillMaxWidth().clip(shape).background(web.twBg(Tw.gray800))
                    .border(1.dp, border, shape).padding(24.dp),
            ) {
                Row(Modifier.padding(bottom = 16.dp), verticalAlignment = Alignment.CenterVertically) {
                    FaIcon(R.drawable.fa_solid_gem, null, size = 20.dp, tint = Tw.blue500, modifier = Modifier.padding(end = 8.dp))
                    Text(if (gem == null) "Create New Gem" else "Edit Gem", fontSize = 20.sp, lineHeight = 28.sp,
                        fontWeight = FontWeight.Bold, color = web.twText(Tw.white))
                }
                Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                    GemField("Name") { TwInput(name, { name = it }, "My Helper") }
                    GemField("Description (Optional)") { TwInput(description, { description = it }, "A short description...") }
                    GemField("System Instruction") {
                        TwInput(instruction, { instruction = it }, "You are a helpful assistant...", singleLine = false, height = 128.dp)
                    }
                    GemField("Default Model (optional)") {
                        WebSelect(
                            defaultModel, listOf(WebOption("", "Use current model")), { defaultModel = it },
                            fontSize = 14.sp, background = web.twBg(Tw.gray900), fillWidth = true,
                            contentPadding = PaddingValues(8.dp), contentDescription = "Default Model (optional)",
                        )
                    }
                    Column(Modifier.fillMaxWidth()) {
                        Box(Modifier.fillMaxWidth().height(1.dp).background(border))
                        Text("FIXED PROMPTS (GEM SPECIFIC)", fontSize = 12.sp, lineHeight = 16.sp, fontWeight = FontWeight.Bold,
                            letterSpacing = 0.6.sp, color = web.twText(Tw.indigo400), modifier = Modifier.padding(top = 16.dp, bottom = 8.dp))
                        GemFixedPromptRows(rows)
                        Row(
                            Modifier.clip(RoundedCornerShape(4.dp)).background(web.twBg(Tw.gray700))
                                .clickable(role = Role.Button) { rows.add(FixedPrompt("", "")) }
                                .padding(horizontal = 8.dp, vertical = 4.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            FaIcon(R.drawable.fa_solid_plus, null, size = 10.dp, tint = web.twText(Tw.gray300), modifier = Modifier.padding(end = 4.dp))
                            Text("プロンプトを追加", fontSize = 10.sp, lineHeight = 15.sp, color = web.twText(Tw.gray300))
                        }
                    }
                }
                Row(Modifier.fillMaxWidth().padding(top = 24.dp), horizontalArrangement = Arrangement.spacedBy(12.dp, Alignment.End)) {
                    Text("Cancel", fontSize = 16.sp, lineHeight = 24.sp, color = web.twText(Tw.gray400),
                        modifier = Modifier.clip(RoundedCornerShape(4.dp)).clickable(role = Role.Button, onClick = onDismiss)
                            .padding(horizontal = 16.dp, vertical = 8.dp))
                    Text(if (gem == null) "Create Gem" else "Save Changes", fontSize = 16.sp, lineHeight = 24.sp,
                        fontWeight = FontWeight.Bold, color = Tw.white,
                        modifier = Modifier.clip(RoundedCornerShape(4.dp)).background(Tw.blue600)
                            .clickable(enabled = !saving, role = Role.Button) {
                                if (name.isEmpty() || instruction.isEmpty()) missing = true
                                else {
                                    saving = true
                                    onSave(name, description, instruction, collectGemFixedPrompts(rows)) { saving = false }
                                }
                            }
                            .padding(horizontal = 16.dp, vertical = 8.dp))
                }
            }
        }
    }
    if (missing) BrowserAlertDialog("Name and Instruction are required.") { missing = false }
}

@Composable
private fun GemField(label: String, content: @Composable () -> Unit) {
    val web = LocalWebPalette.current
    Column(Modifier.fillMaxWidth()) {
        Text(label, fontSize = 12.sp, lineHeight = 16.sp, color = web.twText(Tw.gray400), modifier = Modifier.padding(bottom = 4.dp))
        content()
    }
}

/** `.gem-fixed-prompt-row`: 名前 (w-24), プロンプト内容 (h-9, fills the row) and the × remove button. */
@Composable
private fun GemFixedPromptRows(rows: SnapshotStateList<FixedPrompt>) {
    val web = LocalWebPalette.current
    Column(Modifier.fillMaxWidth().padding(bottom = if (rows.isEmpty()) 0.dp else 8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        rows.forEachIndexed { index, row ->
            key(index) {
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.Top) {
                    TwInput(row.name, { rows[index] = rows[index].copy(name = it) }, "名前", modifier = Modifier.width(96.dp),
                        fontSize = 10.sp, padding = 6.dp)
                    TwInput(row.content, { rows[index] = rows[index].copy(content = it) }, "プロンプト内容", singleLine = false,
                        height = 36.dp, modifier = Modifier.weight(1f), fontSize = 10.sp, padding = 6.dp)
                    Box(
                        Modifier.clip(RoundedCornerShape(4.dp)).clickable(role = Role.Button) { rows.removeAt(index) }
                            .semantics { contentDescription = "削除" }.padding(6.dp),
                    ) { FaIcon(R.drawable.fa_solid_times, null, size = 16.dp, tint = web.twText(Tw.gray500)) }
                }
            }
        }
    }
}

/** Tailwind form field: `bg-gray-900 border border-gray-600 rounded p-2 text-white text-sm` unless overridden. */
@Composable
internal fun TwInput(
    value: String,
    onChange: (String) -> Unit,
    placeholder: String,
    modifier: Modifier = Modifier.fillMaxWidth(),
    singleLine: Boolean = true,
    height: Dp? = null,
    fontSize: TextUnit = 14.sp,
    padding: Dp = 8.dp,
    background: Color = Tw.gray900,
    borderColor: Color = Tw.gray600,
    textColor: Color = Tw.white,
    readOnly: Boolean = false,
) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(4.dp)
    val color = web.twText(textColor)
    val style = TextStyle(fontSize = fontSize, lineHeight = fontSize * 1.5f, color = color, fontFamily = WebFonts.sans)
    BasicTextField(
        value, onChange, singleLine = singleLine, readOnly = readOnly, textStyle = style, cursorBrush = SolidColor(color),
        modifier = modifier.then(if (height != null) Modifier.height(height) else Modifier).clip(shape)
            .background(web.twBg(background)).border(1.dp, web.twBorder(borderColor), shape)
            .semantics { contentDescription = placeholder },
        decorationBox = { inner ->
            Box(Modifier.padding(padding)) {
                if (value.isEmpty()) Text(placeholder, style = style.copy(color = Tw.gray400))
                inner()
            }
        },
    )
}
