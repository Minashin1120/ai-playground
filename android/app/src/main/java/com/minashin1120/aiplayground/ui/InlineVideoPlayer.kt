package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.layout.*
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.Alignment
import androidx.compose.ui.unit.dp
import com.minashin1120.aiplayground.data.Diagnostics
import kotlinx.coroutines.CancellationException
import java.io.File

/** Generated videos use the same native player as the file viewer, inside the answer bubble. */
@Composable
internal fun InlineVideoPlayer(reference: String, onOpen: (String) -> Unit) {
    val download = LocalAttachmentDownloader.current
    var file by remember(reference) { mutableStateOf<File?>(null) }
    var failed by remember(reference) { mutableStateOf(false) }
    var attempt by remember(reference) { mutableIntStateOf(0) }
    LaunchedEffect(reference, attempt) {
        failed = false
        Diagnostics.log("media.start", "type" to "video")
        try {
            file = download?.invoke(reference)?.first
            failed = file == null
            Diagnostics.log(if (failed) "media.error" else "media.ready", "type" to "video")
        } catch (e: CancellationException) { throw e }
        catch (e: Exception) { failed = true; Diagnostics.failure("media.error", e, "type" to "video") }
    }
    Box(Modifier.fillMaxWidth().height(220.dp), contentAlignment = Alignment.Center) {
        val ready = file
        when {
            ready != null -> VideoPreview(ready, autoPlay = false)
            failed -> Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text("動画を読み込めませんでした")
                TextButton(onClick = { attempt++ }) { Text("再試行") }
                TextButton(onClick = { onOpen(reference) }) { Text("開く") }
            }
            else -> CircularProgressIndicator(Modifier.size(24.dp))
        }
    }
}
