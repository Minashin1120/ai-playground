package com.minashin1120.aiplayground.data

import okhttp3.Call
import okhttp3.Callback
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import java.io.IOException
import java.util.concurrent.TimeUnit

data class AppUpdate(
    val versionName: String,
    val releaseUrl: String,
)

private data class AppVersion(val major: Int, val minor: Int, val patch: Int) : Comparable<AppVersion> {
    override fun compareTo(other: AppVersion): Int = compareValuesBy(this, other, AppVersion::major, AppVersion::minor, AppVersion::patch)
}

private const val REPOSITORY = "Minashin1120/ai-playground"
private const val RELEASES_API_URL = "https://api.github.com/repos/$REPOSITORY/releases?per_page=100"
private val ANDROID_TAG = Regex("^android-v(\\d+)\\.(\\d+)\\.(\\d+)$")
private val VERSION_NAME = Regex("^(?:v)?(\\d+)\\.(\\d+)\\.(\\d+)(?:[-+].*)?$")

private fun parseVersion(value: String): AppVersion? {
    val match = VERSION_NAME.matchEntire(value.trim()) ?: return null
    val values = match.groupValues.drop(1).take(3).map { it.toIntOrNull() ?: return null }
    return AppVersion(values[0], values[1], values[2])
}

/** Finds the newest stable Android release without considering Web release tags. */
internal fun latestAndroidUpdateFromJson(payload: String, currentVersion: String): AppUpdate? {
    val current = parseVersion(currentVersion) ?: return null
    val releases = runCatching { org.json.JSONArray(payload) }.getOrNull() ?: return null
    var newest: Pair<AppVersion, AppUpdate>? = null
    for (index in 0 until releases.length()) {
        val release = releases.optJSONObject(index) ?: continue
        if (release.optBoolean("draft") || release.optBoolean("prerelease")) continue
        val tag = release.optString("tag_name")
        val tagMatch = ANDROID_TAG.matchEntire(tag) ?: continue
        val versionName = tagMatch.groupValues.drop(1).joinToString(".")
        val version = parseVersion(versionName) ?: continue
        if (version <= current || newest?.first?.let { version <= it } == true) continue
        newest = version to AppUpdate(
            versionName = versionName,
            releaseUrl = "https://github.com/$REPOSITORY/releases/tag/$tag",
        )
    }
    return newest?.second
}

class AppUpdateChecker(
    private val client: OkHttpClient = OkHttpClient.Builder()
        .followRedirects(false)
        .followSslRedirects(false)
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.SECONDS)
        .callTimeout(15, TimeUnit.SECONDS)
        .build(),
) {
    private var call: Call? = null

    fun check(currentVersion: String, onResult: (AppUpdate?) -> Unit) {
        call?.cancel()
        val request = Request.Builder()
            .url(RELEASES_API_URL)
            .header("Accept", "application/vnd.github+json")
            .header("X-GitHub-Api-Version", "2022-11-28")
            .header("User-Agent", "AIPlayground-Android/$currentVersion")
            .get()
            .build()
        call = client.newCall(request).also { pending ->
            pending.enqueue(object : Callback {
                override fun onFailure(call: Call, e: IOException) {
                    if (!call.isCanceled()) onResult(null)
                }

                override fun onResponse(call: Call, response: Response) {
                    val update = runCatching {
                        response.use {
                            if (!it.isSuccessful) null
                            else latestAndroidUpdateFromJson(readBoundedUtf8(it.body.byteStream(), 2 * 1024 * 1024), currentVersion)
                        }
                    }.getOrNull()
                    if (!call.isCanceled()) onResult(update)
                }
            })
        }
    }

    fun cancel() {
        call?.cancel()
        call = null
    }
}
