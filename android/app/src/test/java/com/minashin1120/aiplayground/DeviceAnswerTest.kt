package com.minashin1120.aiplayground

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DeviceAnswerTest {
    @Test fun anAnswerGeneratedOnTheDeviceIsNotCancelledByLeavingTheApp() {
        assertTrue(isDeviceAnswer(streaming = true, uploadsLocal = true, jobId = null))
    }

    @Test fun serverAnswersAndIdleScreensAreNotDeviceAnswers() {
        assertFalse(isDeviceAnswer(streaming = true, uploadsLocal = false, jobId = null))
        assertFalse(isDeviceAnswer(streaming = true, uploadsLocal = true, jobId = "job-1"))
        assertFalse(isDeviceAnswer(streaming = false, uploadsLocal = true, jobId = null))
    }
}
