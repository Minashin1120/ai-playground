package com.minashin1120.aiplayground.data

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONArray
import org.json.JSONObject
import java.util.Locale
import java.util.concurrent.TimeUnit

/** USD→JPY reference rate shown under the USD cost estimate in the token detail. */
data class JpyExchangeRate(val rate: Double, val date: String, val source: String)

/** Mirrors Web's `jpyMoneyText`: 0.01円未満 below one sen, 2 decimals below 1,000円, then whole yen. */
fun jpyAmountText(amount: Double): String = when {
    amount <= 0 -> "0円"
    amount < 0.01 -> "0.01円未満"
    amount < 1000 -> String.format(Locale.JAPAN, "%.2f円", amount)
    else -> String.format(Locale.JAPAN, "%,d円", Math.round(amount))
}

/** Mirrors Web's `costEstimateJpyLabel`, so both clients show the same wording. */
fun jpyCostLabel(cost: CostEstimate, rate: JpyExchangeRate): String {
    if (cost.priced == 0) return "日本円換算: 算出不可"
    val low = jpyAmountText(cost.low * rate.rate)
    val amount = when {
        cost.lowerBound -> if (low.endsWith("未満")) low else "${low}以上"
        cost.high - cost.low > 1e-12 -> "$low〜${jpyAmountText(cost.high * rate.rate)}"
        else -> low
    }
    val unpriced = if (cost.unpriced > 0) "（一部未算入）" else ""
    val date = if (rate.date.isNotEmpty()) "・${rate.date}" else ""
    val perDollar = String.format(Locale.US, "%.2f", rate.rate)
    return "日本円換算: 約$amount$unpriced（参考レート 1ドル=${perDollar}円$date）"
}

/** Mirrors Web's `costItemJpyText`: the per-request share in yen, or blank when it cannot be priced. */
fun jpyItemLabel(item: CostItem, rate: JpyExchangeRate): String {
    if (!item.priced) return ""
    val low = jpyAmountText(item.low * rate.rate)
    return if (item.high - item.low > 1e-12) "約$low〜${jpyAmountText(item.high * rate.rate)}" else "約$low"
}

/**
 * Free, keyless USD→JPY rates: ECB reference rates via Frankfurter, then a public CDN mirror.
 * Only the rate is fetched (no chat, usage or account data) and it is cached on the device.
 */
object JpyExchangeRates {
    private const val PREFS = "exchange_rate"
    private const val CACHE_KEY = "usd_jpy_v1"
    private const val MAX_AGE_MS = 12L * 60 * 60 * 1000
    private const val MIN_RATE = 50.0
    private const val MAX_RATE = 500.0

    private val sources = listOf(
        "ECB" to "https://api.frankfurter.dev/v2/rates?base=USD&quotes=JPY",
        "currency-api" to "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json",
    )

    private val client by lazy {
        OkHttpClient.Builder()
            .connectTimeout(4, TimeUnit.SECONDS)
            .readTimeout(6, TimeUnit.SECONDS)
            .callTimeout(7, TimeUnit.SECONDS)
            .build()
    }

    @Volatile private var memo: JpyExchangeRate? = null
    @Volatile private var failed = false

    /** Returns a fresh rate, a stale cached rate when the network fails, or null when nothing is known. */
    suspend fun load(context: Context): JpyExchangeRate? = withContext(Dispatchers.IO) {
        memo?.let { return@withContext it }
        cached(context, MAX_AGE_MS)?.let {
            memo = it
            return@withContext it
        }
        if (!failed) {
            for ((name, url) in sources) {
                val rate = fetch(name, url) ?: continue
                save(context, rate)
                memo = rate
                return@withContext rate
            }
            failed = true
        }
        cached(context, Long.MAX_VALUE)
    }

    private fun fetch(name: String, url: String): JpyExchangeRate? {
        val body = (try {
            client.newCall(Request.Builder().url(url).header("Accept", "application/json").build())
                .execute().use { response -> if (response.isSuccessful) response.body?.string() else null }
        } catch (_: Exception) {
            null
        }) ?: return null
        val parsed = try {
            if (name == "ECB") {
                val row = JSONArray(body.trim()).optJSONObject(0)
                if (row == null) null else row.optDouble("rate", 0.0) to row.optString("date")
            } else {
                val json = JSONObject(body)
                (json.optJSONObject("usd")?.optDouble("jpy", 0.0) ?: 0.0) to json.optString("date")
            }
        } catch (_: Exception) {
            null
        }
        val (value, date) = parsed ?: return null
        if (!value.isFinite() || value < MIN_RATE || value > MAX_RATE) return null
        return JpyExchangeRate(value, date.take(10), name)
    }

    private fun cached(context: Context, maxAgeMs: Long): JpyExchangeRate? = try {
        val raw = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString(CACHE_KEY, null)
        val json = if (raw == null) null else JSONObject(raw)
        val value = json?.optDouble("rate", 0.0) ?: 0.0
        val fresh = json != null && System.currentTimeMillis() - json.optLong("saved", 0L) <= maxAgeMs
        if (fresh && value.isFinite() && value >= MIN_RATE && value <= MAX_RATE)
            JpyExchangeRate(value, json.optString("date"), json.optString("source")) else null
    } catch (_: Exception) {
        null
    }

    private fun save(context: Context, rate: JpyExchangeRate) {
        try {
            val json = JSONObject().put("rate", rate.rate).put("date", rate.date)
                .put("source", rate.source).put("saved", System.currentTimeMillis())
            context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
                .putString(CACHE_KEY, json.toString()).apply()
        } catch (_: Exception) {
            /* The cache is optional. */
        }
    }
}
