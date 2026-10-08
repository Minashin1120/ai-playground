package com.minashin1120.aiplayground.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class AppChangelogDialogTest {
    @Test
    fun splitsAggregateMarkdownIntoVersionSectionsAndDropsAggregateTitle() {
        val sections = parseAppChangelogSections("""# AI Playground for Android

# Android版更新履歴 - 1.13.33

- 読みやすくしました。

# Android版更新履歴 - 1.13.32

- Markdown表示に対応しました。
""")

        assertEquals(listOf("1.13.33", "1.13.32"), sections.map { it.version })
        assertEquals("- 読みやすくしました。", sections[0].markdown)
        assertTrue(sections.none { it.markdown.contains("AI Playground for Android") })
    }

    @Test
    fun keepsLegacyMarkdownWhenNoVersionHeadingExists() {
        val sections = parseAppChangelogSections("# 更新履歴\n\n- 変更点")

        assertEquals(1, sections.size)
        assertEquals(null, sections.single().version)
        assertEquals("# 更新履歴\n\n- 変更点", sections.single().markdown)
    }

    @Test
    fun filtersSectionsByVersionOrMarkdownCaseInsensitively() {
        val sections = listOf(
            AppChangelogSection("1.13.33", "- Added search"),
            AppChangelogSection("1.13.32", "- Fixed Markdown rendering"),
        )

        assertEquals(listOf("1.13.33"), filterAppChangelogSections(sections, "SEARCH").map { it.version })
        assertEquals(listOf("1.13.32"), filterAppChangelogSections(sections, "v1.13.32").map { it.version })
        assertEquals(sections, filterAppChangelogSections(sections, "  "))
        assertTrue(filterAppChangelogSections(sections, "missing").isEmpty())
    }

    @Test
    fun splitsFlatBulletListIntoChangeItems() {
        val items = appChangelogItems("- 検索を追加しました。\n  続きの行です。\n\n* **太字**も使えます。")

        assertEquals(listOf("検索を追加しました。 続きの行です。", "**太字**も使えます。"), items)
    }

    @Test
    fun fallsBackToMarkdownWhenSectionIsNotAFlatBulletList() {
        assertNull(appChangelogItems("# 更新履歴\n\n- 変更点"))
        assertNull(appChangelogItems("- 親\n  - 子"))
        assertNull(appChangelogItems("説明の段落です。"))
        assertNull(appChangelogItems("   "))
    }
}
