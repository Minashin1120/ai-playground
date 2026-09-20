package com.minashin1120.aiplayground

import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import android.app.Application
import com.minashin1120.aiplayground.data.PlaygroundApi
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class AppChangelogUiState(
    val loading: Boolean = false,
    val content: String? = null,
    val errorMessage: String? = null,
)

class AppChangelogViewModel(application: Application) : AndroidViewModel(application) {
    private val api = PlaygroundApi()
    private val mutable = MutableStateFlow(AppChangelogUiState())
    val state = mutable.asStateFlow()
    private var loadJob: Job? = null

    fun load() {
        if (loadJob?.isActive == true) return
        mutable.update { it.copy(loading = true, errorMessage = null) }
        loadJob = viewModelScope.launch {
            try {
                val content = api.getText("/android/release-notes.md")
                mutable.value = AppChangelogUiState(content = content)
            } catch (error: CancellationException) {
                throw error
            } catch (error: Exception) {
                mutable.update {
                    it.copy(
                        loading = false,
                        errorMessage = error.message?.take(300) ?: "更新履歴を取得できませんでした。",
                    )
                }
            } finally {
                loadJob = null
            }
        }
    }
}
