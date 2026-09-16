package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.KeyboardArrowDown
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.minashin1120.aiplayground.data.ModelInfo
import com.minashin1120.aiplayground.data.generationOptions

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun GenerationOptionsPanel(model: ModelInfo, values: Map<String, String>, enabled: Boolean, onChange: (String, String) -> Unit) {
    val options = remember(model.id) { generationOptions(model) }
    if (options.isEmpty()) return
    Surface(shape = MaterialTheme.shapes.small, color = MaterialTheme.colorScheme.surfaceContainerLow) {
        FlowRow(Modifier.fillMaxWidth().heightIn(max = 220.dp).verticalScroll(rememberScrollState()).padding(8.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            options.forEach { option ->
                key(model.id, option.key) {
                    val value = values[option.key] ?: option.defaultValue
                    when {
                        option.kind == "boolean" -> FilterChip(value == "true", { onChange(option.key, (value != "true").toString()) }, { Text(option.label) }, enabled = enabled)
                        option.choices.isNotEmpty() -> {
                            var expanded by remember { mutableStateOf(false) }
                            Box {
                                OutlinedButton(onClick = { expanded = true }, enabled = enabled, contentPadding = PaddingValues(horizontal = 10.dp)) {
                                    Text("${option.label}: ${value.ifEmpty { "Default" }}", style = MaterialTheme.typography.labelMedium)
                                    Icon(Icons.Rounded.KeyboardArrowDown, null, Modifier.size(16.dp))
                                }
                                DropdownMenu(expanded, { expanded = false }) {
                                    option.choices.forEach { choice ->
                                        DropdownMenuItem(text = { Text(choice.ifEmpty { "Default" }) }, onClick = { onChange(option.key, choice); expanded = false })
                                    }
                                }
                            }
                        }
                        else -> Column(Modifier.widthIn(min = 140.dp, max = 240.dp), horizontalAlignment = Alignment.Start) {
                            OutlinedTextField(value, { onChange(option.key, it) }, enabled = enabled, singleLine = true,
                                label = { Text(option.label, style = MaterialTheme.typography.labelSmall) },
                                placeholder = { Text("Default") }, isError = option.error(value) != null,
                                keyboardOptions = KeyboardOptions(keyboardType = if (option.kind in listOf("number", "integer")) KeyboardType.Decimal else KeyboardType.Text))
                            option.error(value)?.let { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.labelSmall) }
                        }
                    }
                }
            }
        }
    }
}
