package com.minashin1120.aiplayground.data

import org.json.JSONArray

/** Display metadata never expands the server's model or permission allowlist. */
fun applyWebModelCatalog(models: List<ModelInfo>, json: String): List<ModelInfo> {
    val rows = JSONArray(json)
    val metadata = (0 until rows.length()).associate { index ->
        val row = rows.getJSONObject(index)
        row.getString("id") to (index to row)
    }
    return models.map { model ->
        val (webCatalogOrder, row) = metadata[model.id] ?: return@map model
        val deprecated = model.deprecated || row.optBoolean("deprecated")
        model.copy(name = row.optString("name", model.name),
            description = row.optString("description"), price = row.optString("price"),
            category = row.optString("category"), implementedAt = row.optString("implementedAt"),
            implementedRank = row.optInt("implementedRank"), emoji = row.optString("emoji"),
            tags = row.optJSONArray("tags")?.let { tags -> (0 until tags.length()).map { tags.getString(it) }.toSet() }.orEmpty(),
            deprecated = deprecated, selectable = model.selectable && !deprecated,
            webCatalogOrder = webCatalogOrder,
            categoryIcon = row.optString("categoryIcon"), categoryDescription = row.optString("categoryDescription"),
            apiId = row.optString("apiId").ifBlank { model.id }, agenticView = row.optBoolean("agenticView"),
            searchTerms = row.optJSONArray("searchTerms")?.let { terms -> (0 until terms.length()).map { terms.getString(it) } }.orEmpty())
    }
}

fun recentWebModels(models: List<ModelInfo>, limit: Int = 5): List<ModelInfo> = models
    .filter { it.selectable && !it.deprecated && it.implementedAt.isNotBlank() }
    .sortedWith(compareByDescending<ModelInfo> { it.implementedAt }.thenByDescending { it.implementedRank }.thenBy { it.id })
    .take(limit.coerceAtLeast(0))
