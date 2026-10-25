package com.minashin1120.aiplayground.ui

import android.annotation.SuppressLint
import android.content.Context
import android.view.GestureDetector
import android.view.OrientationEventListener
import android.view.ScaleGestureDetector
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageCapture
import androidx.camera.core.Preview
import androidx.camera.core.ZoomState
import androidx.camera.core.UseCase
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.MutableLiveData
import com.minashin1120.aiplayground.data.ActivityLog
import java.util.concurrent.Executor

/** Own the capture use case so its rotation follows the sensor even with screen rotation locked. */
internal class InAppCameraSession(private val context: Context) {
    private val preview = Preview.Builder().setResolutionSelector(ResolutionSelector.Builder()
        .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY).build()).build()
    private val imageCapture: ImageCapture = ImageCapture.Builder()
        .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
        .setResolutionSelector(ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
            .setResolutionStrategy(ResolutionStrategy.HIGHEST_AVAILABLE_STRATEGY).build())
        .build()
    val zoomState = MutableLiveData<ZoomState>()
    private var camera: Camera? = null
    private var desiredRotation = android.view.Surface.ROTATION_0
    private var captureRotationLocked = false
    private val orientationListener = object : OrientationEventListener(context) {
        override fun onOrientationChanged(orientation: Int) {
            if (orientation == ORIENTATION_UNKNOWN) return
            desiredRotation = UseCase.snapToSurfaceRotation(orientation)
            if (!captureRotationLocked) imageCapture.targetRotation = desiredRotation
        }
    }

    @SuppressLint("ClickableViewAccessibility")
    fun attachPreview(view: PreviewView) {
        val displayRotation = view.display?.rotation ?: android.view.Surface.ROTATION_0
        preview.targetRotation = displayRotation
        desiredRotation = displayRotation
        imageCapture.targetRotation = desiredRotation
        preview.setSurfaceProvider(view.surfaceProvider)
        val pinch = ScaleGestureDetector(context, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
            override fun onScale(detector: ScaleGestureDetector): Boolean {
                val state = zoomState.value ?: return false
                setZoomRatio((state.zoomRatio * detector.scaleFactor).coerceIn(state.minZoomRatio, state.maxZoomRatio))
                return true
            }
        })
        val tap = GestureDetector(context, object : GestureDetector.SimpleOnGestureListener() {
            override fun onDown(event: android.view.MotionEvent) = true
            override fun onSingleTapUp(event: android.view.MotionEvent): Boolean {
                val point = view.meteringPointFactory.createPoint(event.x, event.y)
                camera?.cameraControl?.startFocusAndMetering(FocusMeteringAction.Builder(point).build())
                view.performClick()
                return true
            }
        })
        view.setOnTouchListener { _, event ->
            pinch.onTouchEvent(event)
            tap.onTouchEvent(event)
            true
        }
    }

    @SuppressLint("MissingPermission")
    fun bind(owner: LifecycleOwner, onError: () -> Unit): () -> Unit {
        var disposed = false
        var provider: ProcessCameraProvider? = null
        val observer = object : DefaultLifecycleObserver {
            override fun onStart(owner: LifecycleOwner) { orientationListener.enable() }
            override fun onStop(owner: LifecycleOwner) { orientationListener.disable() }
        }
        owner.lifecycle.addObserver(observer)
        ActivityLog.log("state", "camera" to true, "phase" to "start")
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            if (!disposed) {
                try {
                    val ready = future.get()
                    provider = ready
                    camera = ready.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture)
                    camera?.cameraInfo?.zoomState?.observe(owner) { zoomState.value = it }
                    ActivityLog.log("state", "camera" to true, "phase" to "done")
                } catch (exception: Exception) {
                    ActivityLog.log("media.error", "source" to "camera", "type" to "image", "phase" to "error")
                    onError()
                }
            }
        }, ContextCompat.getMainExecutor(context))
        return {
            disposed = true
            owner.lifecycle.removeObserver(observer)
            orientationListener.disable()
            camera?.cameraInfo?.zoomState?.removeObservers(owner)
            provider?.unbind(preview, imageCapture)
            camera = null
            ActivityLog.log("state", "camera" to false, "phase" to "done")
        }
    }

    /** Freeze the shutter orientation until CameraX has finished this capture. */
    fun takePicture(options: ImageCapture.OutputFileOptions, executor: Executor, callback: ImageCapture.OnImageSavedCallback) {
        captureRotationLocked = true
        val unlock = {
            captureRotationLocked = false
            imageCapture.targetRotation = desiredRotation
        }
        try {
            imageCapture.takePicture(options, executor, object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(result: ImageCapture.OutputFileResults) {
                    unlock()
                    callback.onImageSaved(result)
                }
                override fun onError(exception: ImageCaptureException) {
                    unlock()
                    callback.onError(exception)
                }
            })
        } catch (exception: Exception) {
            unlock()
            throw exception
        }
    }

    fun setZoomRatio(ratio: Float) { camera?.cameraControl?.setZoomRatio(ratio) }
}
