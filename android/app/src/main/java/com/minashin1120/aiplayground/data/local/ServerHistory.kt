package com.minashin1120.aiplayground.data.local

import com.minashin1120.aiplayground.data.ApiException
import com.minashin1120.aiplayground.data.backend.ChatBackend
import kotlinx.coroutines.CancellationException
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException

/**
 * What serverless mode of a signed-in account reads from the server to answer on the device: the
 * messages a new answer builds on and their attachments. When the server is out of reach, the offline
 * cache of the chat screen stands in ([cachedThread], [cachedFile]).
 */
class ServerHistory(
    private val server: ChatBackend,
    private val token: () -> String,
    private val serverFile: suspend (String, Long) -> ByteArray,
    private val cachedThread: suspend (String) -> JSONObject?,
    private val cachedFile: suspend (String, Long) -> ByteArray?,
    /** True while the app already knows the server is out of reach (no request is tried then). */
    val isOffline: () -> Boolean,
    /** Sends the outbox after an answer, waiting only briefly; failures stay in the outbox and are retried later. */
    val afterAnswer: suspend () -> Unit,
) {
    /** A server chat: its `GET /api/threads/<id>` settings and the messages loaded (enough for the parent's ancestors). */
    class Snapshot(val thread: JSONObject, val messages: List<JSONObject>)

    /**
     * Reads server chat [id] from the newest message back until every ancestor of [parentId] is loaded
     * (the whole chat is not read). Out of reach, the offline cache is used when it holds those ancestors.
     */
    suspend fun thread(id: String, parentId: Int?): Snapshot {
        if (!isOffline()) {
            try {
                return fromServer(id, parentId)
            } catch (e: Exception) {
                if (!unreachable(e)) throw e
            }
        }
        val cached = cachedThread(id)
        val rows = messageRows(cached?.optJSONArray("messages"))
        if (cached == null || !complete(rows, parentId)) throw ApiException(400, JSONObject().put("code", "serverless_history_unavailable")
            .put("error", "サーバーに接続できないため、このチャットの履歴を読み込めません。"))
        return Snapshot(cached, rows)
    }

    private suspend fun fromServer(id: String, parentId: Int?): Snapshot {
        val first = server.get("/api/threads/$id?limit=$PAGE", token())
        val rows = ArrayList<JSONObject>()
        var reply = first
        while (true) {
            rows += messageRows(reply.optJSONArray("messages"))
            val oldest = reply.opt("oldest_loaded_id")?.takeIf { it != JSONObject.NULL }?.toString()
            if (complete(rows, parentId) || !reply.optBoolean("has_older_messages") || oldest == null) break
            reply = server.get("/api/threads/$id?limit=$PAGE&include_meta=0&before_id=$oldest", token())
        }
        return Snapshot(first, rows)
    }

    /** Attachment bytes of a server reference, or null when neither the server nor the offline cache has them. */
    suspend fun file(reference: String, limit: Long): ByteArray? {
        if (!isOffline()) {
            try {
                return serverFile(reference, limit)
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                if (!unreachable(e)) return null
            }
        }
        return cachedFile(reference, limit)
    }

    companion object {
        private const val PAGE = 200

        /** The server did not answer (no connection or a server error), as opposed to a refusal. */
        fun unreachable(e: Exception): Boolean = if (e is ApiException) e.status >= 500 else e is IOException

        /** Whether every ancestor of [parentId] is among [rows]. */
        fun complete(rows: List<JSONObject>, parentId: Int?): Boolean {
            if (parentId == null) return true
            val byId = rows.associateBy { it.optInt("id") }
            var cursor: Int? = parentId
            val seen = HashSet<Int>()
            while (cursor != null && seen.add(cursor)) {
                val row = byId[cursor] ?: return false
                cursor = if (row.isNull("parent_id")) null else row.optInt("parent_id")
            }
            return true
        }

        fun messageRows(array: JSONArray?): List<JSONObject> = array?.let { a -> (0 until a.length()).mapNotNull { a.optJSONObject(it) } }.orEmpty()
    }
}
