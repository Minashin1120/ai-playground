package com.minashin1120.aiplayground.data

import org.junit.Assert.*
import org.junit.Test

/** Coding Mode target texts and send candidates (chat_core part03 / part14). */
class CodingTargetsTest {
    private val history = historyCodingTargets(listOf(
        ChatMessage("1", "user", "```kotlin\nval ignored = 1\n```"),
        ChatMessage("2", "assistant", "```kotlin\nval a = 1\n```\n\n```diff\n-a\n+b\n```"),
        ChatMessage("3", "assistant", "```python\nprint(1)\n```"),
    ))

    @Test fun historySkipsUserMessagesAndDiffBlocks() {
        assertEquals(listOf("kotlin", "python"), history.map { it.language })
        assertEquals("2", history.first().messageId)
    }

    @Test fun promptFencesNeedAClosingMarkerAndKeepTheLanguage() {
        val targets = extractPromptCodingTargets("直して\n~~~js\nlet x\n~~~\n```py\nunfinished")
        assertEquals(1, targets.size)
        assertEquals("js", targets[0].language)
        assertEquals("let x", targets[0].code)
        assertEquals("prompt-1", targets[0].candidateId)
        assertTrue(targets[0].promptSource)
    }

    @Test fun barTextFollowsTheWebStates() {
        assertEquals("コードブロック生成後に自動有効化", codingBarText("", emptyList(), null))
        assertEquals("自動選択: 最新の kotlin コードブロック", codingBarText("", history.take(1), null))
        assertEquals("入力中: js コードブロック", codingBarText("```js\nx\n```", emptyList(), null))
        assertEquals("モデルが編集対象を判断: 入力1件 / 履歴2件", codingBarText("```js\nx\n```", history, null))
        val selected = CodingTarget(codingTargetKey("python", "print(1)"), "print(1)", "python", "3")
        assertEquals("編集対象: python コードブロック", codingBarText("```js\nx\n```", history, selected))
    }

    @Test fun sendPlanPrefersTheLatestPromptFenceAndKeepsHistoryCandidates() {
        val plan = planCodingSend("```js\nx\n```", history, null, "gpt-5.6")
        assertTrue(plan.active)
        assertEquals("js", plan.target!!.language)
        assertEquals(listOf("prompt-1", "history-1", "history-2"), plan.candidates.map { it.candidateId })
    }

    @Test fun explicitSelectionIsTheOnlyCandidate() {
        val selected = CodingTarget(codingTargetKey("python", "print(1)"), "print(1)", "python", "3")
        val plan = planCodingSend("```js\nx\n```", history, selected, "gpt-5.6")
        assertEquals(listOf("selected-1"), plan.candidates.map { it.candidateId })
        assertTrue(plan.target!!.explicit)
        assertEquals("3", plan.target!!.messageId)
    }

    @Test fun nothingToEditLeavesCodingInactiveAndMediaModelsAreRejected() {
        assertFalse(planCodingSend("hello", emptyList(), null, "gpt-5.6").active)
        val rejected = planCodingSend("```js\nx\n```", emptyList(), null, "gpt-image-2")
        assertEquals("Coding Modeではテキスト生成モデルを選択してください", rejected.error)
    }

    @Test fun oversizedPromptCodeIsRejected() {
        val huge = "```js\n" + "a".repeat(CODING_MAX_CHARS + 1) + "\n```"
        assertEquals("入力内の編集候補コード合計が大きすぎます（上限300,000文字）", planCodingSend(huge, emptyList(), null, "gpt-5.6").error)
    }
}
