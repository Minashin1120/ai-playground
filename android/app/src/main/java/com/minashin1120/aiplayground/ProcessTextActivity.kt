package com.minashin1120.aiplayground

import android.app.Activity
import android.content.Intent
import android.os.Bundle

/**
 * 「プロンプトとして添付」 in the system text-selection menu (`Intent.ACTION_PROCESS_TEXT`).
 * The selection is appended to the prompt input through the same path as text shared from the
 * share sheet. The activity returns no replacement, so the text in the source app stays as it was.
 */
class ProcessTextActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val text = runCatching {
            intent?.getCharSequenceExtra(Intent.EXTRA_PROCESS_TEXT)?.toString()
        }.getOrNull()?.trim().orEmpty()
        if (text.isNotEmpty()) {
            startActivity(Intent(this, MainActivity::class.java).apply {
                action = Intent.ACTION_SEND
                type = "text/plain"
                putExtra(Intent.EXTRA_TEXT, text)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
            })
        }
        setResult(RESULT_CANCELED)
        finish()
    }
}
