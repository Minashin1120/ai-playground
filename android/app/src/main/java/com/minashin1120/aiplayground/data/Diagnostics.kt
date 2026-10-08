package com.minashin1120.aiplayground.data

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * Administrator-only diagnostics of serverless mode: which step ran when, how long it took, sizes, MIME
 * types, states, exception names, the error shown in an answer (the server's or the AI provider's text,
 * which can name an attachment), and the stacks of the app's threads when work stalls. Never the user's
 * message text or file contents.
 *
 * Entries go to a file first, so they survive the app being killed; [ChatViewModel] sends them to
 * `/api/mobile/v1/diagnostics` (server `logs/android-diagnostics.log`) and removes what the server took.
 * Nothing is recorded until [setEnabled] marks the account as an administrator, and turning it off
 * deletes what was waiting. When the user turns on [ActivityLog], every entry is copied there as well
 * (for any account). Without [init] (JVM unit tests) every call does nothing.
 */
object Diagnostics {
    private const val MAX_FILE_BYTES = 768 * 1024
    private const val MAX_FIELD_CHARS = 2000
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
        // The same steps also go to the user's activity log (ログの収集を強化) when it is on.
        val activity = ActivityLog.enabled
        if (!enabled && !activity) return
        val target = file ?: return
        val now = System.currentTimeMillis()
        val entry = JSONObject().put("t", now).put("ev", event).put("pid", pid).put("thread", Thread.currentThread().name)
        fields.forEach { (key, value) -> entry.put(key, sanitize(value)) }
        if (activity) ActivityLog.logEntry(entry)
        lastEventAt = now
        if (!enabled) return
        synchronized(lock) {
            entry.put("seq", ++seq)
            runCatching {
                target.parentFile?.mkdirs()
                if (target.length() > MAX_FILE_BYTES) {
                    // Keep the newest half when the server has been out of reach for a long time.
                    val lines = target.readLines()
                    target.writeText(lines.takeLast(lines.size / 2).joinToString("") { "$it\n" })
                }
                target.appendText(entry.toString() + "\n")
            }
        }
    }

    /** An exception: its class, message and the app's own frames. */
    fun failure(event: String, error: Throwable, vararg fields: Pair<String, Any?>) {
        log(event, *fields, "error" to error.javaClass.name, "message" to error.message,
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
        log("stall.stacks", *fields, "reason" to reason, "memory" to memory(), "threads" to threads)
    }

    /** Up to [max] waiting lines (oldest first): [lines] is how many to [drop] once the server took [entries]. */
    class Batch(val lines: Int, val entries: List<JSONObject>)

    fun pending(max: Int, maxChars: Int): Batch = synchronized(lock) {
        val target = file ?: return Batch(0, emptyList())
        val all = runCatching { if (target.isFile) target.readLines() else emptyList() }.getOrDefault(emptyList())
        var chars = 0
        val lines = all.take(max).takeWhile { line -> chars += line.length; chars <= maxChars }
            .ifEmpty { all.take(1) }
        Batch(lines.size, lines.mapNotNull { runCatching { JSONObject(it) }.getOrNull() })
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
        trace.take(STACK_FRAMES).forEach { out.put("${it.className}.${it.methodName}:${it.lineNumber}") }
        return out
    }

    private fun sanitize(value: Any?): Any = when (value) {
        null -> JSONObject.NULL
        is Number, is Boolean, is JSONObject, is JSONArray -> value
        is Collection<*> -> JSONArray(value.map { sanitize(it) })
        else -> value.toString().take(MAX_FIELD_CHARS)
    }
}
