package com.minashin1120.aiplayground

import org.junit.Assert.assertEquals
import org.junit.Test

class AnswerNotificationsTest {
    @Test fun theTitleSaysWhetherTheAnswerFinishedOrFailed() {
        assertEquals("回答が完了しました", answerNotificationTitle(failed = false))
        assertEquals("回答の生成でエラーが発生しました", answerNotificationTitle(failed = true))
    }
}
