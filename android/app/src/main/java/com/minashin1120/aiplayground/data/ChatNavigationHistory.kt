package com.minashin1120.aiplayground.data

/** A new-chat screen or a saved thread visited before the current chat. */
data class ChatLocation(val thread: ThreadItem?, val temporary: Boolean = false) {
    val id: String? get() = thread?.id
}

/** In-memory navigation history for the current account. */
class ChatNavigationHistory {
    private val previous = mutableListOf<ChatLocation>()

    val canGoBack: Boolean get() = previous.isNotEmpty()

    fun record(current: ChatLocation, next: ChatLocation) {
        if (current.id == next.id && (current.id != null || current.temporary == next.temporary)) return
        if (previous.size == 50) previous.removeAt(0)
        previous.add(current)
    }

    fun pop(): ChatLocation? = if (previous.isEmpty()) null else previous.removeAt(previous.lastIndex)

    fun removeThread(id: String) { previous.removeAll { it.id == id } }

    fun clear() { previous.clear() }
}
