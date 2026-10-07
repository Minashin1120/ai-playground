package com.minashin1120.aiplayground.ui

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import android.view.ViewGroup
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.CameraController
import androidx.camera.view.LifecycleCameraController
import androidx.camera.view.PreviewView
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ZoomState
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.Observer
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.minashin1120.aiplayground.data.normalizeCapturedPhotoOrientation
import java.io.File
import java.util.Locale
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@Composable
internal fun InAppCameraDialog(
    onDismiss: () -> Unit,
    onCaptured: (Uri) -> Unit,
    onError: (String) -> Unit,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val scope = rememberCoroutineScope()
    var permissionGranted by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    var captureInProgress by remember { mutableStateOf(false) }
    val permissionRequest = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        permissionGranted = granted
        if (!granted) {
            onError("カメラの使用が許可されていません。")
            onDismiss()
        }
    }
    LaunchedEffect(Unit) {
        if (!permissionGranted) permissionRequest.launch(Manifest.permission.CAMERA)
    }

    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(usePlatformDefaultWidth = false, decorFitsSystemWindows = false),
    ) {
        if (permissionGranted) {
            val controller = remember(context) { LifecycleCameraController(context).apply {
                setEnabledUseCases(CameraController.IMAGE_CAPTURE)
                isPinchToZoomEnabled = true
            } }
            var zoomState by remember(controller) { mutableStateOf<ZoomState?>(null) }
            DisposableEffect(controller, lifecycleOwner) {
                controller.bindToLifecycle(lifecycleOwner)
                val zoomObserver = Observer<ZoomState> { zoomState = it }
                controller.zoomState.observe(lifecycleOwner, zoomObserver)
                onDispose {
                    controller.zoomState.removeObserver(zoomObserver)
                    controller.unbind()
                }
            }
            Box(Modifier.fillMaxSize().background(Color.Black)) {
                AndroidView(
                    factory = { viewContext -> PreviewView(viewContext).apply {
                        layoutParams = ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
                        scaleType = PreviewView.ScaleType.FIT_CENTER
                        this.controller = controller
                    } },
                    modifier = Modifier.fillMaxSize(),
                    update = { it.controller = controller },
                )
                Text(
                    "閉じる", color = Color.White, modifier = Modifier.align(Alignment.TopStart)
                        .statusBarsPadding().padding(20.dp).clickable(onClick = onDismiss),
                )
                zoomState?.let { state ->
                    val presets = cameraZoomPresets(state.minZoomRatio, state.maxZoomRatio)
                    if (presets.size > 1) {
                        val active = activeCameraZoomPreset(presets, state.zoomRatio)
                        Row(
                            Modifier.align(Alignment.BottomCenter).navigationBarsPadding().padding(bottom = 122.dp)
                                .background(Color.Black.copy(alpha = 0.45f), RoundedCornerShape(50)).padding(4.dp),
                            horizontalArrangement = Arrangement.spacedBy(4.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            presets.forEach { preset ->
                                val selected = preset == active
                                Box(
                                    Modifier.size(40.dp)
                                        .background(if (selected) Color.White.copy(alpha = 0.22f) else Color.Transparent, CircleShape)
                                        .clickable { controller.setZoomRatio(preset) },
                                    contentAlignment = Alignment.Center,
                                ) {
                                    Text(
                                        formatCameraZoomRatio(if (selected) state.zoomRatio else preset),
                                        color = if (selected) Color(0xFFFFD54F) else Color.White,
                                        fontSize = 12.sp,
                                    )
                                }
                            }
                        }
                    }
                }
                Box(
                    Modifier.align(Alignment.BottomCenter).navigationBarsPadding().padding(bottom = 28.dp)
                        .size(78.dp).border(4.dp, Color.White, CircleShape)
                        .clickable(enabled = !captureInProgress) {
                            val directory = File(context.cacheDir, "shared").apply { mkdirs() }
                            val file = File(directory, "camera_${System.currentTimeMillis()}.jpg")
                            captureInProgress = true
                            try {
                                val options = androidx.camera.core.ImageCapture.OutputFileOptions.Builder(file).build()
                                controller.takePicture(options, ContextCompat.getMainExecutor(context), object : ImageCapture.OnImageSavedCallback {
                                    override fun onImageSaved(result: ImageCapture.OutputFileResults) {
                                        scope.launch {
                                            try {
                                                withContext(Dispatchers.IO) { normalizeCapturedPhotoOrientation(file) }
                                                val uri = FileProvider.getUriForFile(context, "${context.packageName}.files", file)
                                                onCaptured(uri)
                                            } catch (exception: CancellationException) {
                                                throw exception
                                            } catch (exception: Exception) {
                                                captureInProgress = false
                                                onError("写真を処理できません。")
                                            }
                                        }
                                    }
                                    override fun onError(exception: ImageCaptureException) {
                                        captureInProgress = false
                                        onError("写真を撮影できません。")
                                    }
                                })
                            } catch (exception: Exception) {
                                captureInProgress = false
                                onError("写真を撮影できません。")
                            }
                        },
                    contentAlignment = Alignment.Center,
                ) {
                    Box(Modifier.size(62.dp).background(Color.White, CircleShape))
                }
            }
        }
    }
}

private val CAMERA_ZOOM_STEPS = listOf(1f, 2f, 5f)

/** Preset zoom buttons: the ultra-wide end when the camera has one, then 1x, 2x and 5x within range. */
internal fun cameraZoomPresets(minZoomRatio: Float, maxZoomRatio: Float): List<Float> {
    val presets = mutableListOf<Float>()
    if (minZoomRatio < 0.95f) presets += minZoomRatio
    CAMERA_ZOOM_STEPS.filterTo(presets) { it >= minZoomRatio - 0.01f && it <= maxZoomRatio + 0.01f }
    return presets
}

/** The preset the current zoom belongs to: the largest one not above the current ratio. */
internal fun activeCameraZoomPreset(presets: List<Float>, zoomRatio: Float): Float =
    presets.lastOrNull { it <= zoomRatio + 0.05f } ?: presets.first()

internal fun formatCameraZoomRatio(zoomRatio: Float): String {
    val tenths = Math.round(zoomRatio * 10f)
    return if (tenths % 10 == 0) "${tenths / 10}×" else String.format(Locale.ROOT, "%.1f×", tenths / 10f)
}
