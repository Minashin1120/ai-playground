package com.minashin1120.aiplayground.data

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * Administrator-only diagnostics of serverless mode: which step ran when, how long it took, sizes, MIME
 * types, fixed states, exception categories and app source locations when work stalls. Never free text,
 * titles, filenames, credentials, message text or file contents.
 *
 * Entries go to a file first, so they survive the app being killed; [ChatViewModel] sends them to
 * `/api/mobile/v1/diagnostics` (server `logs/android-diagnostics.log`) and removes what the server took.
 * Nothing is recorded until [setEnabled] marks the account as an administrator, and turning it off
 * deletes what was waiting. When the user turns on [ActivityLog], every entry is copied there as well
 * (for any account). Without [init] (JVM unit tests) every call does nothing.
 */
object Diagnostics {
    private const val MAX_FILE_BYTES = 768 * 1024
    private const val STACK_FRAMES = 24
    private const val PREFS = "diagnostics"

    private val lock = Any()
    @Volatile private var file: File? = null
    @Volatile private var context: Context? = null
    @Volatile var enabled: Boolean = false
        private set
    /** Time of the latest entry; the stall watchdog dumps thread stacks when nothing happens for a while. */
    @Volatile var lastEventAt: Long = 0L
        private set
    private var seq = 0L
    private var pid = 0

    fun init(app: Context) {
        context = app
        file = File(app.noBackupFilesDir, "diagnostics/pending.jsonl")
        synchronized(lock) {
            file?.let { target -> runCatching {
                if (target.isFile) {
                    val safe = target.readLines().mapNotNull { line -> runCatching { LogPolicy.entry(JSONObject(line)) }.getOrNull() }
                    target.writeText(safe.joinToString("") { "$it\n" })
                }
            } }
        }
        pid = runCatching { android.os.Process.myPid() }.getOrDefault(0)
        enabled = app.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getBoolean("admin", false)
        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, error ->
            runCatching { failure("app.crash", error, "thread" to thread.name, "memory" to memory()) }
            ActivityLog.drain()
            previous?.uncaughtException(thread, error)
        }
        log("app.process_start")
    }

    /** Whether the signed-in account is an administrator (remembered for the next start, before the account loads). */
    fun setEnabled(value: Boolean) {
        val app = context ?: return
        if (enabled == value) return
        enabled = value
        app.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putBoolean("admin", value).apply()
        if (!value) synchronized(lock) { file?.delete() }
    }

    fun log(event: String, vararg fields: Pair<String, Any?>) {
        runCatching {
            // The same steps also go to the user's activity log (ログの収集を強化) when it is on.
            val activity = ActivityLog.enabled
            if (!enabled && !activity) return
            val target = file
            val now = System.currentTimeMillis()
            val entry = JSONObject().put("t", now).put("ev", event).put("pid", pid).put("thread", Thread.currentThread().name)
            fields.forEach { (key, value) -> entry.put(key, value) }
            val safe = LogPolicy.entry(entry) ?: return
            if (activity) ActivityLog.logEntry(safe)
            lastEventAt = now
            if (!enabled || target == null) return
            synchronized(lock) {
                safe.put("seq", ++seq)
                runCatching {
                    target.parentFile?.mkdirs()
                    if (target.length() > MAX_FILE_BYTES) {
                        // Keep the newest half when the server has been out of reach for a long time.
                        val lines = target.readLines()
                        target.writeText(lines.takeLast(lines.size / 2).joinToString("") { "$it\n" })
                    }
                    target.appendText(safe.toString() + "\n")
                }
            }
        }
    }

    /** An exception category and the app's own source locations; never its message. */
    fun failure(event: String, error: Throwable, vararg fields: Pair<String, Any?>) {
        log(event, *fields, "error" to error.javaClass.simpleName, "failed" to true,
            "stack" to frames(error.stackTrace))
    }

    /** Stacks of the app's threads (and blocked ones), for work that stopped moving. */
    fun stacks(reason: String, vararg fields: Pair<String, Any?>) {
        if (!enabled && !ActivityLog.enabled) return
        val threads = JSONArray()
        runCatching {
            Thread.getAllStackTraces().entries
                .filter { (thread, trace) ->
                    thread.state == Thread.State.BLOCKED || thread.name == "main" ||
                        trace.any { it.className.startsWith("com.minashin1120.") }
                }
                .take(24)
                .forEach { (thread, trace) ->
                    threads.put(JSONObject().put("name", thread.name).put("state", thread.state.name).put("frames", frames(trace)))
                }
        }
        val combined = JSONArray()
        for (index in 0 until threads.length()) {
            val frames = threads.getJSONObject(index).getJSONArray("frames")
            for (frame in 0 until frames.length()) if (combined.length() < STACK_FRAMES) combined.put(frames.getJSONObject(frame))
        }
        log("stall.stacks", *fields, "stack" to combined, "used_mb" to memory().optLong("used_mb"), "max_mb" to memory().optLong("max_mb"))
    }

    /** Up to [max] waiting lines (oldest first): [lines] is how many to [drop] once the server took [entries]. */
    class Batch(val lines: Int, val entries: List<JSONObject>)

    fun pending(max: Int, maxChars: Int): Batch = synchronized(lock) {
        val target = file ?: return Batch(0, emptyList())
        val all = runCatching { if (target.isFile) target.readLines() else emptyList() }.getOrDefault(emptyList())
        var chars = 0
        val lines = all.take(max).takeWhile { line -> chars += line.length; chars <= maxChars }
            .ifEmpty { all.take(1) }
        Batch(lines.size, lines.mapNotNull { runCatching { LogPolicy.entry(JSONObject(it)) }.getOrNull() })
    }

    /** Waiting entries (not yet sent) that mention [ids], for the chat copy of a feedback. */
    fun pendingMentioning(ids: Collection<String>): List<JSONObject> = synchronized(lock) {
        val target = file ?: return emptyList()
        val lines = runCatching { if (target.isFile) target.readLines() else emptyList() }.getOrDefault(emptyList())
        lines.filter { line -> ids.any { it.isNotBlank() && line.contains(it) } }
            .mapNotNull { runCatching { JSONObject(it) }.getOrNull() }
    }

    /** Removes the first [count] lines after the server accepted them. */
    fun drop(count: Int) = synchronized(lock) {
        val target = file ?: return@synchronized
        runCatching {
            val lines = target.readLines()
            target.writeText(lines.drop(count).joinToString("") { "$it\n" })
        }
        Unit
    }

    fun memory(): JSONObject {
        val runtime = Runtime.getRuntime()
        return JSONObject().put("used_mb", (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024))
            .put("max_mb", runtime.maxMemory() / (1024 * 1024))
    }

    private fun frames(trace: Array<StackTraceElement>): JSONArray {
        val out = JSONArray()
        trace.filter { it.className.startsWith("com.minashin1120.aiplayground.") }.take(STACK_FRAMES).forEach {
            val name = it.className.substringAfterLast('.').substringBefore('$').removeSuffix("Kt") + ".kt"
            out.put(JSONObject().put("code_file", name).put("line", it.lineNumber))
        }
        return out
    }


}
