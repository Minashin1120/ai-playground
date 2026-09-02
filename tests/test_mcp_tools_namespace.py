import unittest

from mcp_service import tools as mcp_tools


class McpToolsConversionTests(unittest.TestCase):
    """ツール名の名前空間・読み取り分類・コンテンツ変換のテスト。"""

    def test_internal_name_namespace(self):
        name = mcp_tools.make_internal_tool_name("google_gmail", "search_messages")
        self.assertEqual(name, "mcp__google_gmail__search_messages")
        self.assertTrue(mcp_tools.is_mcp_tool_name(name))
        self.assertFalse(mcp_tools.is_mcp_tool_name("create_file"))
        # ハイフン等はアンダースコア化される
        name2 = mcp_tools.make_internal_tool_name("my-srv", "do-thing")
        self.assertIn("my_srv", name2)
        self.assertIn("do_thing", name2)
        # 長い名前は末尾ハッシュで一意化・長さ上限内
        long_name = mcp_tools.make_internal_tool_name("a" * 60, "b" * 90)
        self.assertLessEqual(len(long_name), 64)
        other = mcp_tools.make_internal_tool_name("a" * 60 + "X", "b" * 90)
        self.assertNotEqual(long_name, other)

    def test_classify_readonly(self):
        self.assertTrue(mcp_tools.classify_readonly("search_files", "Search files"))
        self.assertTrue(mcp_tools.classify_readonly("messages.list", "List messages"))
        self.assertTrue(mcp_tools.classify_readonly("drive.files.get", "Fetch a file"))
        self.assertFalse(mcp_tools.classify_readonly("messages.send", "Send an email"))
        self.assertFalse(mcp_tools.classify_readonly("files.delete", ""))
        self.assertFalse(mcp_tools.classify_readonly("calendar.events.create", ""))
        # 書き込み動詞を名前に含む読み取り系の誤判定を避ける
        self.assertFalse(mcp_tools.classify_readonly("threads.trash", ""))
        # 説明文の強い語でも変更判定
        self.assertFalse(mcp_tools.classify_readonly("doAction", "Create a new draft email"))

    def test_openai_and_chat_completions_schema(self):
        tool = {"name": "t", "description": "desc", "input_schema": {"type": "object", "properties": {}}}
        resp = mcp_tools.to_openai_function_schema("mcp__s__t", tool)
        self.assertEqual(resp["type"], "function")
        self.assertEqual(resp["name"], "mcp__s__t")
        cc = mcp_tools.to_chat_completions_function_schema("mcp__s__t", tool)
        self.assertEqual(cc["function"]["name"], "mcp__s__t")
        anth = mcp_tools.to_anthropic_tool("mcp__s__t", tool)
        self.assertEqual(anth["name"], "mcp__s__t")
        self.assertEqual(anth["input_schema"]["type"], "object")

    def test_anthropic_non_object_schema_forced(self):
        tool = {"name": "t", "description": "d", "input_schema": {"type": "string"}}
        anth = mcp_tools.to_anthropic_tool("mcp__s__t", tool)
        self.assertEqual(anth["input_schema"]["type"], "object")

    def test_content_blocks_to_text(self):
        out = mcp_tools.content_blocks_to_text({
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "text", "text": "world"},
                {"type": "image", "mimeType": "image/png", "data": "aaaa"},
            ],
            "structured_content": None,
        })
        self.assertIn("hello\nworld", out)
        self.assertIn("image data not embedded", out)


if __name__ == "__main__":
    unittest.main()
