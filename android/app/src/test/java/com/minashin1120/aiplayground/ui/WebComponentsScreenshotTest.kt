package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.github.takahirom.roborazzi.captureRoboImage
import com.minashin1120.aiplayground.R
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode

/**
 * Renders the Web-parity building blocks for side-by-side comparison with the Web UI.
 * CI records the PNGs (`-Proborazzi.test.record=true`) and uploads them with the reports.
 */
@RunWith(RobolectricTestRunner::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(sdk = [34], qualifiers = "w360dp-h800dp-xxhdpi")
class WebComponentsScreenshotTest {
    @Test
    fun componentsDarkPhone() {
        captureRoboImage("build/outputs/roborazzi/web_components_dark_phone.png") { Gallery(dark = true) }
    }

    @Test
    fun componentsLightPhone() {
        captureRoboImage("build/outputs/roborazzi/web_components_light_phone.png") { Gallery(dark = false) }
    }

    @Composable
    private fun Gallery(dark: Boolean) {
        PlaygroundTheme(darkTheme = dark) {
            val web = LocalWebPalette.current
            Box(Modifier.fillMaxSize().background(web.bg1), contentAlignment = Alignment.TopCenter) {
                WebModalPanel(Modifier.padding(8.dp).fillMaxWidth(), phone = true) {
                    WebModalHeader(
                        title = { Text("設定") },
                        onClose = {},
                        phone = true,
                        icon = R.drawable.fa_solid_cog,
                        subtitle = "アプリの動作・表示・セキュリティを管理",
                    )
                    Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        WebSettingsCard("ライトモード", compact = true) {
                            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                                Text("手動ライトモード", Modifier.weight(1f))
                                WebToggle(checked = true, onCheckedChange = {})
                            }
                        }
                        WebSettingsCard("Composer", compact = true) {
                            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                                WebCheckChip("Canvas", true, {}, ChipTones.Canvas.accent, composerChipStyle(true, ChipTones.Canvas))
                                WebCheckChip("Coding", false, {}, ChipTones.Coding.accent, composerChipStyle(false, ChipTones.Coding))
                            }
                            Row(Modifier.padding(top = 8.dp), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                                WebCheckChip("Search", true, {}, Tw.blue500, composerOptStyle(true))
                                WebCheckChip("Python", false, {}, Tw.yellow500, composerOptStyle(false), labelColor = Tw.yellow200)
                                WebSelect("medium", webOptions("minimal" to "Min", "low" to "Low", "medium" to "Mid", "high" to "High"), {})
                            }
                        }
                    }
                    WebModalFooter(phone = true) {
                        WebButton({}, variant = WebButtonVariant.Ghost) { Text("キャンセル") }
                        WebButton({}, variant = WebButtonVariant.Primary) {
                            FaIcon(R.drawable.fa_solid_save, null, size = 13.dp)
                            Text("保存")
                        }
                    }
                }
            }
        }
    }
}
