package com.minashin1120.aiplayground

import android.app.Application
import com.minashin1120.aiplayground.data.AccountApi
import com.minashin1120.aiplayground.data.AccountTransferController
import com.minashin1120.aiplayground.data.PlaygroundApi
import kotlinx.coroutines.CoroutineScope

/**
 * Settings Data tab export / import / dedupe (Web account-transfer tab). The controller outlives the
 * settings modal, like the Web tab; after an import only the categories that changed are reloaded.
 */
internal class ChatAccountTransfer(
    private val scope: CoroutineScope,
    private val api: PlaygroundApi,
    private val token: () -> String,
    application: Application,
    private val notify: (String) -> Unit,
    private val reloadThreads: (onDone: () -> Unit) -> Unit,
    private val reloadSkills: (onDone: () -> Unit) -> Unit,
    private val loadStorageUsage: () -> Unit,
    private val loadPreferences: () -> Unit,
) {
    private val resolver = application.contentResolver

    fun api(): AccountApi = AccountApi(api) { token() }

    val controller: AccountTransferController by lazy {
        AccountTransferController(scope, ::api, resolver, notify) { categories ->
            if ("chats" in categories) reloadThreads {}
            if ("skills" in categories) reloadSkills {}
            if ("files" in categories) loadStorageUsage()
            if ("settings" in categories || "api_credentials" in categories) loadPreferences()
        }
    }
}
