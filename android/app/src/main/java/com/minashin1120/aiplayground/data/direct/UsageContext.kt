package com.minashin1120.aiplayground.data.direct

import com.minashin1120.aiplayground.data.ModelUsageLedger
import kotlin.coroutines.AbstractCoroutineContextElement
import kotlin.coroutines.CoroutineContext
import kotlin.coroutines.coroutineContext
import org.json.JSONObject

/** A ledger belongs to a generation coroutine, never to a shared HTTP client. */
class UsageContext(val ledger: ModelUsageLedger, val model: String) : AbstractCoroutineContextElement(Key) {
    companion object Key : CoroutineContext.Key<UsageContext>
}

internal suspend fun beginUsage(url: String, payload: JSONObject): ModelUsageLedger.Call? {
    val context = coroutineContext[UsageContext] ?: return null
    if (!url.substringBefore('?').let { it.endsWith(":generateContent") || it.endsWith(":streamGenerateContent") ||
                it.endsWith("/responses") || it.endsWith("/chat/completions") || it.endsWith("/messages") }) return null
    val pathModel = Regex("/models/([^/:]+):").find(url)?.groupValues?.get(1)
    val model = payload.optString("model").ifBlank { pathModel ?: context.model }
    val provider = when {
        model.startsWith("gemini") -> "google"
        model.startsWith("claude") -> "anthropic"
        model.startsWith("grok") -> "xai"
        else -> "openai"
    }
    return context.ledger.begin(model, provider).also { com.minashin1120.aiplayground.data.ModelUsagePricing.configure(it, payload) }
}
