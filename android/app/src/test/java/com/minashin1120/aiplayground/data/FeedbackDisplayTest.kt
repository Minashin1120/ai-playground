package com.minashin1120.aiplayground.data

import org.junit.Assert.*
import org.junit.Test

class FeedbackDisplayTest {
    @Test fun thinkingMetadataNeverBecomesVisibleReasoning() {
        assertEquals("計算", reasoningText("""{"text":"計算","deepseek_tool_context":[{"role":"tool","content":"hidden"}]}"""))
        assertEquals("a\nb", reasoningText("""["a",{"text":"b"},{"signature":"hidden"}]"""))
        assertEquals("", reasoningText("""{"signature":"hidden"}"""))
        assertEquals("plain thought", reasoningText("plain thought"))
    }

    @Test fun veoAndSoraShowTheVideoGenerationSkeleton() {
        assertEquals("video", pendingSkeletonKind("veo-3.1-generate-preview"))
        assertEquals("video", pendingSkeletonKind("sora-2"))
        assertEquals("video", pendingSkeletonKind("grok-imagine-video"))
    }
}
