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
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.minashin1120.aiplayground.data.normalizeCapturedPhotoOrientation
import java.io.File
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
            } }
            DisposableEffect(controller, lifecycleOwner) {
                controller.bindToLifecycle(lifecycleOwner)
                onDispose { controller.unbind() }
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
