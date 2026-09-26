package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.ChatViewModel
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.CompressionSettings
import com.minashin1120.aiplayground.data.isGeminiImageModelKey
import com.minashin1120.aiplayground.data.isMistralOcrModel

/**
 * `#compression-modal` 画像・圧縮詳細設定: client-side compression of attached images and, for the
 * current image / OCR model, the same generation options as the composer panel.
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
internal fun CompressionDialog(state: ChatState, model: ChatViewModel, onDismiss: () -> Unit) {
    val web = LocalWebPalette.current
    val current = state.compression
    var outputType by remember { mutableStateOf(current.outputType) }
    var formatOnly by remember { mutableStateOf(current.formatOnly) }
    var maxSize by remember { mutableStateOf(current.maxSizeMB.toString()) }
    var maxDim by remember { mutableStateOf(current.maxDimension.toString()) }
    val modelId = state.model.lowercase()
    val values = remember { mutableStateMapOf<String, String>().apply { putAll(state.generationValues) } }
    fun v(key: String, default: String) = values[key] ?: default
    WebOverlayModal(onDismiss, grayOverlay(), 4.dp, alignment = Alignment.Center) { _ ->
        val shape = RoundedCornerShape(8.dp)
        Column(
            Modifier.padding(16.dp).widthIn(max = 448.dp).fillMaxWidth().clip(shape)
                .background(web.twBg(Tw.gray800)).border(1.dp, web.twBorder(Tw.gray700), shape).padding(24.dp),
        ) {
            Row(Modifier.padding(bottom = 16.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FaIcon(R.drawable.fa_solid_compress_alt, null, size = 18.dp, tint = web.theme300)
                Text("画像・圧縮詳細設定", fontSize = 20.sp, fontWeight = FontWeight.Bold, color = if (web.isLight) web.text else Color.White)
            }
            Column(Modifier.weight(1f, fill = false).verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(24.dp)) {
                Section("添付画像のクライアント側圧縮", web.theme300, web.theme.t500) {
                    ModalField("出力形式 (変換)") {
                        ModalSelect(outputType, webOptions("original" to "元形式を維持 (自動)", "image/jpeg" to "JPEG に変換",
                            "image/png" to "PNG に変換", "image/webp" to "WebP に変換 (推奨)")) { outputType = it }
                    }
                    SettingsCheck("形式変換のみ (品質100・リサイズ無効)", formatOnly, { formatOnly = it }, boxSize = 16.dp,
                        labelColor = if (web.isLight) web.text else Tw.gray300, fontSize = 12.sp)
                    Column(Modifier.alpha(if (formatOnly) 0.5f else 1f)) {
                        ModalField("最大ファイルサイズ (MB)") {
                            SettingsTextField(maxSize, { if (!formatOnly) maxSize = it.filter { c -> c.isDigit() || c == '.' }.take(5) },
                                Modifier.fillMaxWidth(), number = true, readOnly = formatOnly, deep = true)
                        }
                    }
                    Column(Modifier.alpha(if (formatOnly) 0.5f else 1f)) {
                        ModalField("最大幅/高さ (px)") {
                            SettingsTextField(maxDim, { if (!formatOnly) maxDim = it.filter(Char::isDigit).take(4) },
                                Modifier.fillMaxWidth(), number = true, readOnly = formatOnly, deep = true)
                        }
                    }
                }
                when {
                    modelId.contains("gpt-image") -> RuledSection("GPT Image 生成設定", Tw.green300, Tw.green500) {
                        TwoColumns(
                            { ModalField("Size") { ModalSelect(v("image_size", "1024x1024"), webOptions("1024x1024" to "1024x1024", "1536x1024" to "1536x1024", "1024x1536" to "1024x1536", "auto" to "Auto")) { values["image_size"] = it } } },
                            { ModalField("Quality") { ModalSelect(v("image_quality", "medium"), webOptions("low" to "Low", "medium" to "Medium", "high" to "High", "xhigh" to "X-High", "max" to "Max", "auto" to "Auto")) { values["image_quality"] = it } } },
                        )
                        TwoColumns(
                            { ModalField("Format") { ModalSelect(v("image_format", "jpeg"), webOptions("jpeg" to "JPEG", "png" to "PNG", "webp" to "WebP")) { values["image_format"] = it } } },
                            { ModalField("Compression (%)") { SettingsTextField(v("image_compression", "85"), { values["image_compression"] = it.filter(Char::isDigit).take(3) }, Modifier.fillMaxWidth(), number = true, deep = true) } },
                        )
                    }
                    isGeminiImageModelKey(modelId) -> RuledSection("Gemini (Nano Banana) 生成設定", web.theme300, web.theme.t500) {
                        TwoColumns(
                            { ModalField("Aspect") { ModalSelect(v("gemini_image_aspect", "1:1"), listOf("1:1", "auto", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9").map { WebOption(it, if (it == "auto") "Auto" else it) }) { values["gemini_image_aspect"] = it } } },
                            { ModalField("Size") { ModalSelect(v("gemini_image_size", "1K"), webOptions("1K" to "1K", "2K" to "2K", "4K" to "4K")) { values["gemini_image_size"] = it } } },
                        )
                    }
                    modelId.contains("grok") && (modelId.contains("imagine") || modelId.contains("image")) && !modelId.contains("video") ->
                        RuledSection("Grok (Imagine) 生成設定", Tw.orange400, Tw.orange500) {
                            TwoColumns(
                                { ModalField("Aspect") { ModalSelect(v("grok_image_aspect", "1:1"), webOptions("auto" to "Auto", "1:1" to "1:1 (Square)", "16:9" to "16:9 (Landscape)", "9:16" to "9:16 (Portrait)", "4:3" to "4:3", "3:4" to "3:4", "3:2" to "3:2", "2:3" to "2:3", "2:1" to "2:1", "1:2" to "1:2", "19.5:9" to "19.5:9", "9:19.5" to "9:19.5", "20:9" to "20:9", "9:20" to "9:20")) { values["grok_image_aspect"] = it } } },
                                { ModalField("Resolution") { ModalSelect(v("grok_image_resolution", "1k"), webOptions("1k" to "1K", "2k" to "2K")) { values["grok_image_resolution"] = it } } },
                            )
                        }
                    isMistralOcrModel(modelId) -> RuledSection("Mistral OCR 4 設定", Tw.orange300, Tw.orange400) {
                        Text("会話履歴・Search・Python・Canvas は使いません。添付した文書または公開URLだけを OCR します。", fontSize = 11.sp, color = Tw.gray400)
                        TwoColumns(
                            { ModalField("Table format") { ModalSelect(v("ocr_table_format", ""), webOptions("" to "本文内 Markdown", "markdown" to "Markdown（別フィールド）", "html" to "HTML（別フィールド）")) { values["ocr_table_format"] = it } } },
                            { ModalField("Pages（0始まり）") { SettingsTextField(v("ocr_pages", ""), { values["ocr_pages"] = it.take(200) }, Modifier.fillMaxWidth(), placeholder = "0-2 または 0,2-4", deep = true) } },
                        )
                        FlowRow(horizontalArrangement = Arrangement.spacedBy(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            listOf("ocr_extract_header" to "ヘッダー抽出", "ocr_extract_footer" to "フッター抽出",
                                "ocr_include_blocks" to "構造ブロック", "ocr_include_image_base64" to "画像抽出").forEach { (key, label) ->
                                val default = if (key == "ocr_include_image_base64") "true" else "false"
                                SettingsCheck(label, v(key, default) == "true", { values[key] = it.toString() }, boxSize = 14.dp,
                                    labelColor = if (web.isLight) web.text else Tw.gray300)
                            }
                        }
                    }
                }
            }
            Row(
                Modifier.fillMaxWidth().padding(top = 24.dp).drawBehind {
                    drawLine(web.twBorder(Tw.gray700), androidx.compose.ui.geometry.Offset(0f, 0f), androidx.compose.ui.geometry.Offset(size.width, 0f), 1.dp.toPx())
                }.padding(top = 16.dp),
                horizontalArrangement = Arrangement.spacedBy(12.dp, Alignment.End),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text("閉じる", fontSize = 14.sp, color = Tw.gray400,
                    modifier = Modifier.clickable(role = Role.Button, onClick = onDismiss).padding(horizontal = 16.dp, vertical = 8.dp))
                Text("保存", fontSize = 14.sp, fontWeight = FontWeight.Bold, color = Color.White,
                    modifier = Modifier.clip(RoundedCornerShape(4.dp)).background(web.theme.t600)
                        .clickable(role = Role.Button) {
                            model.saveCompressionSettings(CompressionSettings(
                                enabled = current.enabled,
                                maxSizeMB = maxSize.toFloatOrNull()?.coerceIn(0.1f, 10f) ?: 1.0f,
                                maxDimension = maxDim.toIntOrNull()?.coerceIn(256, 8192) ?: 1920,
                                outputType = outputType,
                                formatOnly = formatOnly,
                            ))
                            values.forEach { (key, value) -> model.generationOption(key, value) }
                            model.notify("設定を保存しました")
                            onDismiss()
                        }
                        .padding(horizontal = 24.dp, vertical = 8.dp))
            }
        }
    }
}

@Composable
private fun Section(title: String, color: Color, rule: Color, content: @Composable ColumnScope.() -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
        Text(title, fontSize = 14.sp, fontWeight = FontWeight.Bold, color = color,
            modifier = Modifier.drawBehind { drawRect(rule, size = androidx.compose.ui.geometry.Size(2.dp.toPx(), size.height)) }.padding(start = 8.dp))
        content()
    }
}

@Composable
private fun RuledSection(title: String, color: Color, rule: Color, content: @Composable ColumnScope.() -> Unit) {
    val line = LocalWebPalette.current.twBorder(Tw.gray700)
    Column(Modifier.fillMaxWidth().settingsTopRule(line, 16.dp)) { Section(title, color, rule, content) }
}

@Composable
private fun TwoColumns(first: @Composable () -> Unit, second: @Composable () -> Unit) {
    Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
        Box(Modifier.weight(1f)) { first() }
        Box(Modifier.weight(1f)) { second() }
    }
}

@Composable
private fun ModalField(label: String, field: @Composable () -> Unit) {
    Column {
        Text(label, fontSize = 12.sp, color = Tw.gray400, modifier = Modifier.padding(bottom = 4.dp))
        field()
    }
}

@Composable
private fun ModalSelect(value: String, options: List<WebOption>, onSelect: (String) -> Unit) {
    val web = LocalWebPalette.current
    WebSelect(value, options, onSelect, fillWidth = true, fontSize = 14.sp,
        background = web.twBg(Tw.gray900), borderColor = web.twBorder(Tw.gray600), textColor = if (web.isLight) web.text else Color.White,
        contentPadding = PaddingValues(8.dp))
}
