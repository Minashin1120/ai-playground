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
import com.minashin1120.aiplayground.data.Skill

/**
 * `#skill-modal`: Create New Skill / Edit Skill with Name, Short Description (Optional) and Instructions.
 * [onSave] gets the collected values and a callback telling whether the save finished.
 */
@Composable
internal fun SkillEditorDialog(
    skill: Skill?,
    onDismiss: () -> Unit,
    onSave: (name: String, description: String, instruction: String, done: (Boolean) -> Unit) -> Unit,
) {
    val web = LocalWebPalette.current
    var name by remember { mutableStateOf(skill?.name.orEmpty()) }
    var description by remember { mutableStateOf(skill?.description.orEmpty()) }
    var instruction by remember { mutableStateOf(skill?.instruction.orEmpty()) }
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
                Modifier.padding(16.dp).widthIn(max = 512.dp).fillMaxWidth().modalPanelTaps().clip(shape).background(web.twBg(Tw.gray800))
                    .border(1.dp, border, shape).padding(24.dp),
            ) {
                Row(Modifier.padding(bottom = 16.dp), verticalAlignment = Alignment.CenterVertically) {
                    FaIcon(R.drawable.fa_solid_wand_magic_sparkles, null, size = 20.dp, tint = Tw.blue500, modifier = Modifier.padding(end = 8.dp))
                    Text(if (skill == null) "Create New Skill" else "Edit Skill", fontSize = 20.sp, lineHeight = 28.sp,
                        fontWeight = FontWeight.Bold, color = web.twText(Tw.white))
                }
                Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                    SkillField("Name") { TwInput(name, { name = it }, "My Skill") }
                    SkillField("Short Description (Optional)") {
                        TwInput(description, { description = it }, "A short description the model uses to choose this skill...")
                    }
                    SkillField("Instructions") {
                        TwInput(instruction, { instruction = it }, "You are a helpful assistant...", singleLine = false, height = 128.dp)
                    }
                }
                Row(Modifier.fillMaxWidth().padding(top = 24.dp), horizontalArrangement = Arrangement.spacedBy(12.dp, Alignment.End)) {
                    Text("Cancel", fontSize = 16.sp, lineHeight = 24.sp, color = web.twText(Tw.gray400),
                        modifier = Modifier.clip(RoundedCornerShape(4.dp)).clickable(role = Role.Button, onClick = onDismiss)
                            .padding(horizontal = 16.dp, vertical = 8.dp))
                    Text(if (skill == null) "Create Skill" else "Save Changes", fontSize = 16.sp, lineHeight = 24.sp,
                        fontWeight = FontWeight.Bold, color = Tw.white,
                        modifier = Modifier.clip(RoundedCornerShape(4.dp)).background(Tw.blue600)
                            .clickable(enabled = !saving, role = Role.Button) {
                                if (name.isEmpty() || instruction.isEmpty()) missing = true
                                else {
                                    saving = true
                                    onSave(name, description, instruction) { saving = false }
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
private fun SkillField(label: String, content: @Composable () -> Unit) {
    val web = LocalWebPalette.current
    Column(Modifier.fillMaxWidth()) {
        Text(label, fontSize = 12.sp, lineHeight = 16.sp, color = web.twText(Tw.gray400), modifier = Modifier.padding(bottom = 4.dp))
        content()
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
