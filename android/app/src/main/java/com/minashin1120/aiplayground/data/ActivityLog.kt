package com.minashin1120.aiplayground.data

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * Web `static/js/activity_log.js` (ログの収集を強化): while the user turns it on in the feedback tab, the
 * app's own operations are written to the app's data directory (`noBackupFilesDir/activity_log/`, not the
 * cache directory, so clearing the cache keeps it). Requests (method, path without query, status, time),
 * screen and state changes, taps, lifecycle, errors and the [Diagnostics] steps. Never the user's message
 * text, answers, typed text or file contents.
 *
 * Sending a feedback attaches the last hour ([recent]). Entries older than an hour are dropped as new ones
 * are written. Without [init] (JVM unit tests) every call does nothing.
 */
object ActivityLog {
    @Volatile var retentionHours: Int = 1
        private set
    val WINDOW_MS: Long get() = retentionHours * 3600000L

    fun setRetentionHours(value: Int) {
        require(value in 1..8760)
        synchronized(lock) {
            file?.let { prune(it) }
            retentionHours = value
            prefs?.edit()?.putInt("retention_hours", value)?.apply()
            file?.let { prune(it) }
        }
    }
    private const val MAX_FILE_BYTES = 4 * 1024 * 1024L
    private const val PREFS = "activity_log"
    private const val PRUNE_EVERY = 200

    private val lock = Any()
    @Volatile private var file: File? = null
    @Volatile private var prefs: android.content.SharedPreferences? = null
    @Volatile var enabled: Boolean = false
        private set
    private var seq = 0L
    private var generation = 0L
    private var writesSincePrune = 0
    private val listeners = mutableListOf<(Boolean) -> Unit>()
    /** Writes leave the calling thread (taps and state changes are logged on the main thread). */
    private val writer = Executors.newSingleThreadScheduledExecutor { runnable -> Thread(runnable, "activity-log").apply { isDaemon = true } }

    fun init(app: Context) {
        val preferences = app.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        prefs = preferences
        retentionHours = preferences.getInt("retention_hours", 1).coerceIn(1, 8760)
        writer.scheduleWithFixedDelay({ synchronized(lock) { file?.let { runCatching { prune(it) } } } }, 30, 30, TimeUnit.SECONDS)
        initFile(File(app.noBackupFilesDir, "activity_log/events.jsonl"), preferences.getBoolean("enabled", false))
    }

    /** Tests: [target] holds the log instead of the app's data directory. */
    internal fun initFile(target: File, enabledAtStart: Boolean) {
        synchronized(lock) {
            generation++
            file = target
            enabled = enabledAtStart
            runCatching { if (enabled) prune(target) else target.delete() }
        }
    }

    fun setEnabled(value: Boolean) {
        if (enabled == value) return
        enabled = value
        prefs?.edit()?.putBoolean("enabled", value)?.apply()
        if (value) log("activity_log.enabled") else clear()
        synchronized(listeners) { listeners.toList() }.forEach { runCatching { it(value) } }
    }

    fun onChange(listener: (Boolean) -> Unit) { synchronized(listeners) { listeners += listener } }

    /**
     * The account (server and user) the log belongs to: another account's log is deleted so that its
     * operations are never sent with this account's feedback.
     */
    fun setOwner(owner: String) {
        val preferences = prefs ?: return
        val previous = preferences.getString("owner", null)
        if (previous == owner) return
        synchronized(lock) { generation++; if (previous != null) file?.delete() }
        preferences.edit().putString("owner", owner).apply()
    }

    fun log(event: String, vararg fields: Pair<String, Any?>) {
        runCatching {
            if (!enabled) return
            val target = file ?: return
            val entry = JSONObject().put("t", System.currentTimeMillis()).put("ev", event.take(80))
                .put("thread", Thread.currentThread().name)
            fields.forEach { (key, value) -> if (value != null) entry.put(key, value) }
            append(target, entry)
        }
    }

    /** An entry that was already built (the [Diagnostics] steps). */
    fun logEntry(entry: JSONObject) {
        runCatching {
            if (!enabled) return
            val target = file ?: return
            append(target, JSONObject(entry.toString()))
        }
    }

    private fun append(target: File, entry: JSONObject) {
        val safe = LogPolicy.entry(entry) ?: return
        val version = synchronized(lock) { (++seq) to generation }
        safe.put("seq", version.first)
        runCatching { writer.execute { write(target, safe, version.second) } }
    }

    private fun write(target: File, entry: JSONObject, expectedGeneration: Long) {
        synchronized(lock) {
            if (!enabled || generation != expectedGeneration) return
            runCatching {
                target.parentFile?.mkdirs()
                if (++writesSincePrune >= PRUNE_EVERY || target.length() > MAX_FILE_BYTES) {
                    writesSincePrune = 0
                    prune(target)
                }
                target.appendText("${entry.optLong("t")}\t$entry\n")
            }
        }
    }

    /** Drops entries older than [WINDOW_MS], then the oldest half while the file is still too large. */
    private fun prune(target: File) {
        if (!target.isFile) return
        val cutoff = System.currentTimeMillis() - WINDOW_MS
        var lines = target.readLines().mapNotNull { line ->
            runCatching {
                val safe = LogPolicy.entry(JSONObject(line.substringAfter('\t'))) ?: return@runCatching null
                val time = safe.optLong("t")
                if (time <= cutoff || time > System.currentTimeMillis()) null else "$time\t$safe"
            }.getOrNull()
        }
        while (lines.sumOf { it.length + 1 } > MAX_FILE_BYTES && lines.size > 1) lines = lines.takeLast(lines.size / 2)
        target.writeText(lines.joinToString("") { "$it\n" })
    }

    /** Each line is `<time>\t<JSON>`, so pruning reads the time without parsing the entry. */
    private fun timeOf(line: String): Long? = line.substringBefore('\t', "").toLongOrNull()

    /** Waits (up to 2 seconds) for the entries logged so far to reach the file; also before a crash ends the process. */
    fun drain() { runCatching { writer.submit(Runnable {}).get(2, TimeUnit.SECONDS) } }

    /** The last [windowMs] of entries (oldest first), at most about [maxChars] characters of JSON. */
    fun recent(windowMs: Long = WINDOW_MS, maxChars: Int = 3_000_000): JSONArray = drain().let { synchronized(lock) {
        val target = file ?: return JSONArray()
        runCatching { prune(target) }
        val cutoff = System.currentTimeMillis() - minOf(windowMs, WINDOW_MS)
        val lines = runCatching { if (target.isFile) target.readLines() else emptyList() }.getOrDefault(emptyList())
        val kept = ArrayDeque<String>()
        var chars = 0
        for (line in lines.asReversed()) {
            val time = timeOf(line) ?: continue
            if (time <= cutoff) break
            if (chars + line.length + 1 > maxChars) break
            chars += line.length + 1
            kept.addFirst(line)
        }
        JSONArray().also { out -> kept.forEach { line -> runCatching { out.put(JSONObject(line.substringAfter('\t'))) } } }
    } }

    /** Entries and bytes on the device (the settings' log size). */
    fun stats(): Pair<Int, Long> = drain().let { synchronized(lock) {
        val target = file ?: return 0 to 0L
        runCatching { prune(target) }
        if (!target.isFile) return 0 to 0L
        val count = runCatching { target.useLines { lines -> lines.count { it.isNotBlank() } } }.getOrDefault(0)
        count to target.length()
    } }

    fun clear() {
        drain()
        synchronized(lock) { generation++; file?.delete() }
    }


}

/**
 * Web `ActivityLog.reportFeedback`: the message after a feedback was accepted. [logCount] is the number of
 * log entries sent (null when the log is off); [chatCopy] is true when a copy of the open chat went with it;
 * [imageCount] is the number of attached images.
 */
fun feedbackSentText(
    logCount: Int?, logsSaved: Boolean, chatCopy: Boolean, chatCopySaved: Boolean,
    imageCount: Int = 0, imagesSaved: Boolean = true,
): String {
    val failed = buildList {
        if (logCount != null && !logsSaved) add("ログ")
        if (chatCopy && !chatCopySaved) add("チャットのコピー")
        if (imageCount > 0 && !imagesSaved) add("画像")
    }
    if (failed.isNotEmpty()) return "フィードバックを送信しました（${failed.joinToString("と")}は保存できませんでした）"
    val sent = buildList {
        add("フィードバック")
        if (logCount != null) add("保存期間内のログ（${logCount}件）")
        if (chatCopy) add("チャットのコピー")
        if (imageCount > 0) add("画像（${imageCount}枚）")
    }
    return sent.joinToString(if (sent.size > 2) "、" else "と") + "を送信しました"
}

/** Web `.fb-reply-send`: the message after a reply was accepted; [logCount] is null when no log went with it. */
fun feedbackReplySentText(logCount: Int?, logsSaved: Boolean, imageCount: Int = 0, imagesSaved: Boolean = true): String {
    val failed = buildList {
        if (logCount != null && !logsSaved) add("ログ")
        if (imageCount > 0 && !imagesSaved) add("画像")
    }
    if (failed.isNotEmpty()) return "返信を送信しました（${failed.joinToString("と")}は保存できませんでした）"
    val sent = buildList {
        add("返信")
        if (logCount != null) add("保存期間内のログ（${logCount}件）")
        if (imageCount > 0) add("画像（${imageCount}枚）")
    }
    return sent.joinToString(if (sent.size > 2) "、" else "と") + "を送信しました"
}
