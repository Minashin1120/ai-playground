import json
import unittest
from unittest.mock import Mock, patch

import app as target


class BranchImageToolsTests(unittest.TestCase):
    def test_tool_schemas(self):
        list_schema_nested = target._build_list_branch_images_tool_schema(flat=False)
        self.assertEqual(list_schema_nested["type"], "function")
        self.assertEqual(list_schema_nested["function"]["name"], "list_branch_images")

        list_schema_flat = target._build_list_branch_images_tool_schema(flat=True)
        self.assertEqual(list_schema_flat["type"], "function")
        self.assertEqual(list_schema_flat["name"], "list_branch_images")

        import_schema_nested = target._build_import_branch_images_tool_schema(flat=False)
        self.assertEqual(import_schema_nested["type"], "function")
        self.assertEqual(import_schema_nested["function"]["name"], "import_branch_images")
        self.assertIn("image_ids", import_schema_nested["function"]["parameters"]["properties"])

        import_schema_flat = target._build_import_branch_images_tool_schema(flat=True)
        self.assertEqual(import_schema_flat["type"], "function")
        self.assertEqual(import_schema_flat["name"], "import_branch_images")

    def test_collect_branch_images_includes_current_loaded_files_as_imported(self):
        loaded_files = [
            {
                "name": "current_photo.png",
                "send_name": "current_photo.png",
                "mime": "image/png",
                "bytes": b"\x89PNG\r\n\x1a\nfake_bytes_1",
            },
            {
                "name": "notes.txt",
                "mime": "text/plain",
                "bytes": b"not an image",
            },
        ]
        images = target._collect_branch_images(history=[], loaded_files=loaded_files)
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]["image_id"], "img_1")
        self.assertEqual(images[0]["name"], "current_photo.png")
        self.assertTrue(images[0]["imported"])
        self.assertEqual(images[0]["execution_name"], "input_1.png")
        self.assertEqual(images[0]["turn"], "current turn attachment")

    def test_collect_branch_images_with_history(self):
        loaded_files = [
            {
                "name": "current.png",
                "mime": "image/png",
                "bytes": b"\x89PNGfake1",
            }
        ]
        history = [
            {
                "role": "user",
                "content": "Look at this past image",
                "image_url": json.dumps(["user1/past_1.jpg"]),
            },
            {
                "role": "assistant",
                "content": "I edited it: ![View](/files/user1/generated_1.png)",
                "image_url": None,
            },
        ]

        def fake_disk_info(norm):
            return {"exists": True, "size": 100, "mtime": 1000}

        def fake_load_bytes(norm, info):
            return b"fake_image_bytes"

        with patch("app._get_file_disk_info", side_effect=fake_disk_info), \
             patch("app._load_user_file_bytes", side_effect=fake_load_bytes):
            images = target._collect_branch_images(
                history=history,
                loaded_files=loaded_files,
                user_id="user1",
                imported_keys={"user1/past_1.jpg"},
            )

        # 1 from loaded_files, 2 from history (newest assistant generated, then older user upload)
        self.assertEqual(len(images), 3)

        # img_1: current turn
        self.assertEqual(images[0]["image_id"], "img_1")
        self.assertEqual(images[0]["name"], "current.png")
        self.assertTrue(images[0]["imported"])

        # img_2: assistant generated (from 1 turn ago)
        self.assertEqual(images[1]["image_id"], "img_2")
        self.assertEqual(images[1]["name"], "generated_1.png")
        self.assertEqual(images[1]["role"], "assistant")
        self.assertFalse(images[1]["imported"])

        # img_3: user upload (from 2 turns ago)
        self.assertEqual(images[2]["image_id"], "img_3")
        self.assertEqual(images[2]["name"], "past_1.jpg")
        self.assertEqual(images[2]["role"], "user")
        self.assertTrue(images[2]["imported"])  # Since it was in imported_keys

    def test_execute_list_branch_images(self):
        loaded_files = [
            {
                "name": "test.png",
                "mime": "image/png",
                "bytes": b"\x89PNGfake",
            }
        ]
        result_json_str = target._execute_list_branch_images(history=[], loaded_files=loaded_files)
        data = json.loads(result_json_str)
        self.assertEqual(data["total_images"], 1)
        self.assertEqual(data["imported_count"], 1)
        self.assertEqual(data["images"][0]["image_id"], "img_1")
        self.assertIn("already available", data["images"][0]["location"])

    def test_execute_import_branch_images(self):
        history = [
            {
                "role": "user",
                "content": "Check this",
                "image_url": json.dumps(["user1/prev.png"]),
            }
        ]

        def fake_disk_info(norm):
            return {"exists": True, "size": 100, "mtime": 1000}

        def fake_load_bytes(norm, info):
            return b"fake_bytes"

        with patch("app._get_file_disk_info", side_effect=fake_disk_info), \
             patch("app._load_user_file_bytes", side_effect=fake_load_bytes):

            imported_log = []

            def fake_on_import(img):
                imported_log.append(img["image_id"])
                return "input_2.png"

            # 1. Import by specific ID
            res_str = target._execute_import_branch_images(
                ["img_1"],
                history=history,
                loaded_files=[],
                user_id="user1",
                on_import_callback=fake_on_import,
            )
            data = json.loads(res_str)
            self.assertTrue(data["success"])
            self.assertEqual(data["imported_count"], 1)
            self.assertEqual(data["imported"][0]["execution_name"], "input_2.png")
            self.assertEqual(imported_log, ["img_1"])

            # 2. Import "all"
            imported_log.clear()
            res_all_str = target._execute_import_branch_images(
                ["all"],
                history=history,
                loaded_files=[],
                user_id="user1",
                on_import_callback=fake_on_import,
            )
            data_all = json.loads(res_all_str)
            self.assertTrue(data_all["success"])
            self.assertEqual(data_all["imported_count"], 1)

            # 3. Import nonexistent ID
            res_err_str = target._execute_import_branch_images(
                ["img_999"],
                history=history,
                loaded_files=[],
                user_id="user1",
            )
            data_err = json.loads(res_err_str)
            self.assertFalse(data_err["success"])
            self.assertIn("None of the requested image_ids", data_err["message"])


if __name__ == "__main__":
    unittest.main()
