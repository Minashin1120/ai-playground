package com.minashin1120.aiplayground.data

/** One node of the Web branch tree (`renderBranchTreeVisualization`), positioned in dp. */
data class BranchNode(
    val message: ChatMessage,
    val id: Int,
    /** Left edge and top of the node card. */
    val x: Float,
    val y: Float,
    val width: Float,
    val hasChildren: Boolean,
    /** Tokens from the root to this node (`getCumulativeTokensForNode`). */
    val pathTokens: Int,
)

data class BranchTreeLayout(val nodes: List<BranchNode>, val width: Float, val height: Float)

const val BRANCH_NODE_HEIGHT = 50f
/** `mt-4` above each node and the `h-4` connector under a parent. */
const val BRANCH_GAP = 16f

private fun rowTokens(m: ChatMessage): Int = m.tokens ?: ((m.tokensIn ?: 0) + (m.tokensOut ?: 0))

/** Tokens along the path from the root to [id] (`getCumulativeTokensForNode`). */
fun branchPathTokens(messages: List<ChatMessage>, id: Int): Int {
    val byId = messages.mapNotNull { m -> numericId(m)?.let { it to m } }.toMap()
    var total = 0
    var current: Int? = id
    val seen = HashSet<Int>()
    while (current != null && seen.add(current)) {
        val m = byId[current] ?: break
        total += rowTokens(m)
        current = m.parentId
    }
    return total
}

/** Web `getPerModelTokensForPath`: per model (total, in, out, thought), largest total first. */
fun branchModelBreakdown(messages: List<ChatMessage>, id: Int): List<Pair<String, IntArray>> {
    val byId = messages.mapNotNull { m -> numericId(m)?.let { it to m } }.toMap()
    val stats = LinkedHashMap<String, IntArray>()
    var current: Int? = id
    val seen = HashSet<Int>()
    while (current != null && seen.add(current)) {
        val m = byId[current] ?: break
        val row = stats.getOrPut(m.model.ifBlank { "Unknown" }) { IntArray(4) }
        row[0] += rowTokens(m)
        row[1] += m.tokensIn ?: 0
        row[2] += m.tokensOut ?: 0
        row[3] += m.tokensThought ?: 0
        current = m.parentId
    }
    return stats.entries.sortedByDescending { it.value[0] }.map { it.key to it.value }
}

/**
 * Lays the thread out like the Web flex tree: each parent is centered over its children, which sit
 * side by side 16dp apart; roots (messages without a parent) are stacked. Orphans are skipped as on Web.
 * [cardWidth] gives each node's width (the Web card is 120–180px wide depending on its text).
 */
fun layoutBranchTree(messages: List<ChatMessage>, minWidth: Float, cardWidth: (ChatMessage, Int) -> Float): BranchTreeLayout {
    val byId = LinkedHashMap<Int, ChatMessage>()
    messages.forEach { m -> numericId(m)?.let { byId[it] = m } }
    val children = HashMap<Int, MutableList<Int>>()
    val roots = mutableListOf<Int>()
    byId.forEach { (id, m) ->
        val parent = m.parentId
        if (parent != null && parent != 0 && byId.containsKey(parent)) children.getOrPut(parent) { mutableListOf() }.add(id)
        else if (parent == null || parent == 0) roots.add(id)
    }
    // Cumulative tokens top-down.
    val path = HashMap<Int, Int>()
    val widths = HashMap<Int, Float>()
    val subtree = HashMap<Int, Float>()
    val heights = HashMap<Int, Float>()
    // Post-order without recursion so long chats do not overflow the stack.
    val order = mutableListOf<Int>()
    roots.forEach { root ->
        val stack = ArrayDeque<Int>().apply { addLast(root) }
        path[root] = rowTokens(byId.getValue(root))
        while (stack.isNotEmpty()) {
            val id = stack.removeLast()
            order.add(id)
            children[id].orEmpty().forEach { child ->
                path[child] = path.getValue(id) + rowTokens(byId.getValue(child))
                stack.addLast(child)
            }
        }
    }
    order.asReversed().forEach { id ->
        val w = cardWidth(byId.getValue(id), path.getValue(id))
        widths[id] = w
        val kids = children[id].orEmpty()
        val kidsWidth = if (kids.isEmpty()) 0f else kids.sumOf { subtree.getValue(it).toDouble() }.toFloat() + BRANCH_GAP * (kids.size - 1)
        subtree[id] = maxOf(w, kidsWidth)
        val kidsHeight = if (kids.isEmpty()) 0f else BRANCH_GAP + kids.maxOf { heights.getValue(it) }
        heights[id] = BRANCH_GAP + BRANCH_NODE_HEIGHT + kidsHeight
    }
    val width = maxOf(minWidth, roots.maxOfOrNull { subtree.getValue(it) } ?: 0f)
    val nodes = mutableListOf<BranchNode>()
    var top = 0f
    roots.forEach { root ->
        val stack = ArrayDeque<Triple<Int, Float, Float>>() // id, slot left, top
        stack.addLast(Triple(root, (width - subtree.getValue(root)) / 2f, top))
        while (stack.isNotEmpty()) {
            val (id, slot, y) = stack.removeLast()
            val span = subtree.getValue(id)
            val center = slot + span / 2f
            val w = widths.getValue(id)
            val kids = children[id].orEmpty()
            nodes.add(BranchNode(byId.getValue(id), id, center - w / 2f, y + BRANCH_GAP, w, kids.isNotEmpty(), path.getValue(id)))
            if (kids.isNotEmpty()) {
                val total = kids.sumOf { subtree.getValue(it).toDouble() }.toFloat() + BRANCH_GAP * (kids.size - 1)
                var left = center - total / 2f
                val childTop = y + BRANCH_GAP + BRANCH_NODE_HEIGHT + BRANCH_GAP
                kids.forEach { child ->
                    stack.addLast(Triple(child, left, childTop))
                    left += subtree.getValue(child) + BRANCH_GAP
                }
            }
        }
        top += heights.getValue(root)
    }
    return BranchTreeLayout(nodes, width, top)
}
