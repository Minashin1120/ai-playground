import re
import unittest
from pathlib import Path


from tests.app_source import read_app_source
ROOT = Path(__file__).resolve().parents[1]


class DeepSeekV4FlashVisionExpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_source = read_app_source()
        js_assets = list((ROOT / "static/js").glob("chat_core.v4.8.*.js"))
        assert len(js_assets) == 1, "Only the latest versioned chat core asset should remain"
        cls.js_source = js_assets[0].read_text(encoding="utf-8")
        cls.setup_source = (ROOT / "templates/setup.html").read_text(encoding="utf-8")

    def test_model_entry_is_registered_with_metadata(self):
        self.assertRegex(
            self.js_source,
            re.compile(
                r'id:\s*"deepseek-v4-flash-vision-exp"[^}]*implementedAt:\s*"2026-08-23"'
                r'[^}]*implementedRank:\s*8260[^}]*name:\s*"DeepSeek V4 Flash Vision Exp"'
            ),
        )
        self.assertIn('<option value="deepseek-v4-flash-vision-exp">DeepSeek V4 Flash Vision Exp</option>', self.setup_source)
        self.assertIn("'deepseek-v4-flash-vision-exp'", self.setup_source)

    def test_backend_sends_images_natively_for_vision_exp(self):
        branch = self.app_source[self.app_source.index('log_force("Routing: DeepSeek V4 Branch (Chat Completions)")') :]
        build = branch[: branch.index('messages.append({"role": "user", "content": user_text})')]
        # Native vision flag is derived from the official experimental model ID.
        self.assertIn('deepseek_native_vision = "vision-exp" in model_key_l', build)
        # History images are rebuilt as OpenAI-compatible image_url content blocks.
        self.assertIn('"type": "image_url"', build)
        self.assertIn('_load_message_history_images(', build)
        self.assertIn("m.get('role') == 'user' and m.get('image_url')", build)
        # Image-only sends must not be rejected as empty requests.
        self.assertIn('if not user_text.strip() and deepseek_user_content is None:', build)
        # The text-only vision-analysis fallback stays intact for other DeepSeek models.
        self.assertIn("# Image-only send: use the vision analysis as the user turn.", build)
        self.assertIn('messages.append({"role": "system", "content": analysis_block})', build)

    def test_text_only_models_keep_the_vision_analysis_fallback(self):
        branch = self.app_source[self.app_source.index('log_force("Routing: DeepSeek V4 Branch (Chat Completions)")') :]
        build = branch[: branch.index('messages.append({"role": "user", "content": user_text})')]
        self.assertIn("if image_files and not deepseek_native_vision:", build)
        self.assertIn("_analyze_image_with_vision_model(", build)

    def test_effort_mapping_and_ui_follow_flash_family(self):
        self.assertIn('"deepseek-v4-flash-0731", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp"}', self.app_source)
        self.assertIn("const isDeepSeekFlash0731 = modelLower === 'deepseek-v4-flash-0731'", self.js_source)
        self.assertIn("|| modelLower === 'deepseek-v4-flash-vision-exp'", self.js_source)
        self.assertIn("!isDeepSeekFlash0731 && !isDeepSeekPro", self.js_source)

    def test_input_limits_and_vision_notice_reflect_native_vision(self):
        self.assertIn("model === 'deepseek-v4-flash-vision-exp'", self.js_source)
        self.assertIn("DeepSeek V4 Flash Vision Exp 入力制限", self.js_source)
        self.assertIn("JPEG・PNG・GIF・WebP / 画像1枚あたり最大32MB / リクエスト合計48MB", self.js_source)
        self.assertIn("uploadModelLower !== 'deepseek-v4-flash-vision-exp'", self.js_source)
        self.assertIn("vmi.classList.toggle('hidden', modelLower === 'deepseek-v4-flash-vision-exp')", self.js_source)


if __name__ == "__main__":
    unittest.main()
