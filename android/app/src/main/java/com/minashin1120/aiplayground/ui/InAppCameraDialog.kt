package com.minashin1120.aiplayground.ui

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
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
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
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
    onCaptured: (List<Uri>, Boolean) -> Unit,
    onError: (String) -> Unit,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val scope = rememberCoroutineScope()
    var permissionGranted by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    var captureInProgress by remember { mutableStateOf(false) }
    // 画像分割 (ANDROID_ONLY.md): the shots open the split dialog instead of being attached as they are.
    var splitAfterCapture by rememberSaveable { mutableStateOf(false) }
    // Web: shots are collected until 「添付」 (or closing the camera) attaches them all, in the order taken.
    var shots by rememberSaveable { mutableStateOf(arrayListOf<String>()) }
    val attachShots = {
        val uris = shots.map { FileProvider.getUriForFile(context, "${context.packageName}.files", File(it)) }
        shots = arrayListOf()
        onCaptured(uris, splitAfterCapture)
    }
    val close = { if (shots.isNotEmpty()) attachShots() else onDismiss() }
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
        onDismissRequest = close,
        properties = DialogProperties(usePlatformDefaultWidth = false, decorFitsSystemWindows = false),
    ) {
        if (permissionGranted) {
            val controller = remember(context) { LifecycleCameraController(context).apply {
                setEnabledUseCases(CameraController.IMAGE_CAPTURE)
                // The default favours shutter speed (zero-shutter-lag frames, lighter processing); prefer quality.
                imageCaptureMode = ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY
                imageCaptureResolutionSelector = ResolutionSelector.Builder()
                    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                    .setResolutionStrategy(ResolutionStrategy.HIGHEST_AVAILABLE_STRATEGY)
                    .build()
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
                        .statusBarsPadding().padding(20.dp).clickable(onClick = close),
                )
                Text(
                    (if (splitAfterCapture) "✓ " else "") + "画像を分割",
                    color = if (splitAfterCapture) Color(0xFFFFD54F) else Color.White,
                    fontSize = 14.sp,
                    modifier = Modifier.align(Alignment.TopEnd).statusBarsPadding().padding(12.dp)
                        .background(Color.Black.copy(alpha = 0.45f), RoundedCornerShape(50))
                        .clickable { splitAfterCapture = !splitAfterCapture }
                        .padding(horizontal = 14.dp, vertical = 8.dp),
                )
                Column(
                    Modifier.align(Alignment.BottomCenter).fillMaxWidth().navigationBarsPadding().padding(bottom = 28.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(16.dp),
                ) {
                    if (shots.isNotEmpty()) {
                        Text(
                            "${shots.size}枚撮影済み。続けて撮影するか「添付」を押してください。",
                            color = Color.White, fontSize = 12.sp,
                            modifier = Modifier.padding(horizontal = 16.dp)
                                .background(Color.Black.copy(alpha = 0.45f), RoundedCornerShape(8.dp))
                                .padding(horizontal = 10.dp, vertical = 4.dp),
                        )
                        val shotsList = rememberLazyListState()
                        LaunchedEffect(shots.size) { shotsList.animateScrollToItem(shots.size - 1) }
                        LazyRow(
                            state = shotsList,
                            contentPadding = PaddingValues(horizontal = 16.dp),
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            itemsIndexed(shots, key = { _, path -> path }) { index, path ->
                                CapturedShotThumbnail(path, index + 1)
                            }
                        }
                    }
                    zoomState?.let { state ->
                        val presets = cameraZoomPresets(state.minZoomRatio, state.maxZoomRatio)
                        if (presets.size > 1) {
                            val active = activeCameraZoomPreset(presets, state.zoomRatio)
                            Row(
                                Modifier.background(Color.Black.copy(alpha = 0.45f), RoundedCornerShape(50)).padding(4.dp),
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
                    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                        Box(Modifier.weight(1f), contentAlignment = Alignment.Center) {
                            if (shots.isNotEmpty()) {
                                CameraPillButton("撮影分をクリア", enabled = !captureInProgress) {
                                    shots.forEach { runCatching { File(it).delete() } }
                                    shots = arrayListOf()
                                }
                            }
                        }
                        Box(
                            Modifier.size(78.dp).border(4.dp, Color.White, CircleShape)
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
                                                        shots = ArrayList(shots + file.path)
                                                    } catch (exception: CancellationException) {
                                                        throw exception
                                                    } catch (exception: Exception) {
                                                        onError("写真を処理できません。")
                                                    } finally {
                                                        captureInProgress = false
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
                        Box(Modifier.weight(1f), contentAlignment = Alignment.Center) {
                            CameraPillButton("添付 (${shots.size})", enabled = shots.isNotEmpty() && !captureInProgress,
                                color = Color(0xFF2563EB)) { attachShots() }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun CameraPillButton(label: String, enabled: Boolean, color: Color = Color.Black.copy(alpha = 0.45f), onClick: () -> Unit) {
    Text(
        label, color = Color.White.copy(alpha = if (enabled) 1f else 0.5f), fontSize = 14.sp,
        modifier = Modifier.background(color.copy(alpha = if (enabled) color.alpha else color.alpha * 0.5f), RoundedCornerShape(50))
            .clickable(enabled = enabled, onClick = onClick)
            .padding(horizontal = 14.dp, vertical = 8.dp),
    )
}

/** A taken shot waiting to be attached, numbered like the Web camera's preview list. */
@Composable
private fun CapturedShotThumbnail(path: String, number: Int) {
    val bitmap by produceState<ImageBitmap?>(null, path) {
        value = withContext(Dispatchers.IO) { runCatching { decodeShotThumbnail(path) }.getOrNull() }
    }
    val shape = RoundedCornerShape(6.dp)
    Box(Modifier.size(56.dp).clip(shape).background(Color.Black).border(1.dp, Color(0xFF374151), shape)) {
        bitmap?.let { Image(it, null, Modifier.fillMaxSize(), contentScale = ContentScale.Crop) }
        Text(
            "$number", color = Color.White, fontSize = 10.sp,
            modifier = Modifier.align(Alignment.BottomEnd).background(Color.Black.copy(alpha = 0.7f))
                .padding(horizontal = 4.dp, vertical = 1.dp),
        )
    }
}

private fun decodeShotThumbnail(path: String): ImageBitmap? {
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    BitmapFactory.decodeFile(path, bounds)
    var sample = 1
    while (bounds.outWidth / sample > 256 || bounds.outHeight / sample > 256) sample *= 2
    return BitmapFactory.decodeFile(path, BitmapFactory.Options().apply { inSampleSize = sample })?.asImageBitmap()
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
