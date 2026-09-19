package com.minashin1120.aiplayground.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

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
}
