package com.minashin1120.aiplayground.data

import mockwebserver3.MockResponse
import mockwebserver3.MockWebServer
import okhttp3.OkHttpClient
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class AppUpdateCheckerTest {
    @Test fun onlyNewestStableAndroidReleaseIsSelected() {
        val payload = """
            [
              {"tag_name":"v4.8.999","draft":false,"prerelease":false},
              {"tag_name":"android-v1.14.0","draft":false,"prerelease":false,"assets":[
                {"name":"app-release.apk","size":12345},
                {"name":"app-release.apk.sha256","size":80}
              ]},
              {"tag_name":"android-v1.15.0","draft":false,"prerelease":true,"assets":[]},
              {"tag_name":"android-v1.13.9","draft":false,"prerelease":false,"assets":[]}
            ]
        """.trimIndent()

        val update = latestAndroidUpdateFromJson(payload, "1.13.4")

        assertEquals("1.14.0", update?.versionName)
        assertEquals("android-v1.14.0", update?.tagName)
        assertEquals("https://github.com/Minashin1120/ai-playground/releases/download/android-v1.14.0/app-release.apk", update?.apkUrl)
        assertEquals("https://github.com/Minashin1120/ai-playground/releases/download/android-v1.14.0/app-release.apk.sha256", update?.checksumUrl)
        assertEquals(12345L, update?.apkSizeBytes)
    }

    @Test fun olderAndroidReleaseDoesNotTriggerUpdate() {
        val payload = """[{"tag_name":"android-v1.13.4","draft":false,"prerelease":false,"assets":[]}]"""

        assertNull(latestAndroidUpdateFromJson(payload, "1.13.4"))
    }

    @Test fun malformedVersionsAndDraftsAreIgnored() {
        val payload = """
            [
              {"tag_name":"android-vnext","draft":false,"prerelease":false},
              {"tag_name":"android-v1.20.0","draft":true,"prerelease":false,"assets":[]},
              {"tag_name":"android-v1.19.0","draft":false,"prerelease":false,"assets":[
                {"name":"app-release.apk"},
                {"name":"app-release.apk.sha256"}
              ]}
            ]
        """.trimIndent()

        assertEquals("1.19.0", latestAndroidUpdateFromJson(payload, "1.13.4")?.versionName)
    }

    @Test fun releaseWithoutRequiredAssetsIsIgnored() {
        val payload = """
            [
              {"tag_name":"android-v1.20.0","draft":false,"prerelease":false,"assets":[
                {"name":"app-release.apk"}
              ]},
              {"tag_name":"android-v1.19.0","draft":false,"prerelease":false,"assets":[
                {"name":"app-release.apk"},
                {"name":"app-release.apk.sha256"}
              ]}
            ]
        """.trimIndent()

        assertEquals("1.19.0", latestAndroidUpdateFromJson(payload, "1.13.4")?.versionName)
    }

    @Test fun timedOutReleaseLookupStillFinishes() {
        MockWebServer().use { server ->
            server.start()
            val checker = AppUpdateChecker(
                client = OkHttpClient.Builder()
                    .callTimeout(1, TimeUnit.SECONDS)
                    .connectTimeout(1, TimeUnit.SECONDS)
                    .readTimeout(1, TimeUnit.SECONDS)
                    .build(),
                releasesUrl = server.url("/releases").toString(),
            )
            val done = CountDownLatch(1)
            var result: AppUpdateCheckResult? = null
            checker.check("1.0.0") {
                result = it
                done.countDown()
            }
            assertTrue(done.await(5, TimeUnit.SECONDS))
            assertTrue(result is AppUpdateCheckResult.Failed)
        }
    }

    @Test fun emptyReleaseListIsUpToDate() {
        MockWebServer().use { server ->
            server.start()
            server.enqueue(MockResponse.Builder()
                .addHeader("Content-Type", "application/json")
                .body("[]").build())
            val checker = AppUpdateChecker(
                releasesUrl = server.url("/releases").toString(),
            )
            val done = CountDownLatch(1)
            var result: AppUpdateCheckResult? = null
            checker.check("1.0.0") {
                result = it
                done.countDown()
            }
            assertTrue(done.await(5, TimeUnit.SECONDS))
            assertTrue(result is AppUpdateCheckResult.UpToDate)
        }
    }
}
