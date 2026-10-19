package com.minashin1120.aiplayground.data.local

import com.minashin1120.aiplayground.data.direct.DirectTurn
import org.junit.Assert.assertEquals
import org.junit.Test

class InventedAgenticImagesTest {
    @Test fun agenticLinksNotInTheHistoryAreRemoved() {
        val earlier = "![Agentic View](/files/1/agentic_1792394007042_d323e19c.png)"
        val turns = listOf(DirectTurn("user", "見て"), DirectTurn("assistant", "前の回答\n\n$earlier"), DirectTurn("user", "この2つは"))
        val answer = "コード\n\n![Agentic View](/files/1/agentic_1792396614144_c51ef634.png)\n\n画像を確認しました。\n$earlier"
        assertEquals("コード\n\n\n\n画像を確認しました。\n$earlier", withoutInventedAgenticImages(answer, turns))
        // Other images are left alone.
        val plain = "![図](/files/1/upload.png)"
        assertEquals(plain, withoutInventedAgenticImages(plain, turns))
    }
}
