package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.ComposeFoundationFlags
import androidx.compose.foundation.ExperimentalFoundationApi
import com.minashin1120.aiplayground.PlaygroundApplication
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
@OptIn(ExperimentalFoundationApi::class)
class QuoteSelectionTest {
    /** The "Quote" item lives in [LocalTextToolbar][androidx.compose.ui.platform.LocalTextToolbar], which the new context menu bypasses. */
    @Test fun applicationKeepsTheTextToolbarThatOffersQuote() {
        assertTrue(RuntimeEnvironment.getApplication() is PlaygroundApplication)
        assertFalse(ComposeFoundationFlags.isNewContextMenuEnabled)
    }
}
