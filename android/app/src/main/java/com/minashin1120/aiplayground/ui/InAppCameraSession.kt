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
import java.util.concurrent.TimeUnit
import java.util.concurrent.Executor

internal data class CameraFocusState(
    val request: Int,
    val x: Float,
    val y: Float,
    val finished: Boolean = false,
    val success: Boolean = false,
    val op: String = java.util.UUID.randomUUID().toString(),
)

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
    val focusState = MutableLiveData<CameraFocusState?>()
    private var focusRequest = 0
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
        var multiTouch = false
        val tap = GestureDetector(context, object : GestureDetector.SimpleOnGestureListener() {
            override fun onDown(event: android.view.MotionEvent) = true
            override fun onSingleTapUp(event: android.view.MotionEvent): Boolean {
                if (multiTouch) return false
                focusAt(view, event.x, event.y)
                view.performClick()
                return true
            }
        })
        view.setOnTouchListener { _, event ->
            if (event.actionMasked == android.view.MotionEvent.ACTION_DOWN) multiTouch = false
            if (event.pointerCount > 1) multiTouch = true
            pinch.onTouchEvent(event)
            tap.onTouchEvent(event)
            true
        }
    }

    private fun focusAt(view: PreviewView, x: Float, y: Float) {
        val activeCamera = camera ?: return
        val point = view.meteringPointFactory.createPoint(x, y)
        // FIT_CENTER can leave black bars outside the actual camera image.
        if (point.x !in 0f..1f || point.y !in 0f..1f) return
        cancelFocusLog()
        val state = CameraFocusState(++focusRequest, x, y)
        focusState.value = state
        ActivityLog.log("media.start", "source" to "camera", "type" to "image", "interaction_stage" to "focus", "op" to state.op)
        fun finish(success: Boolean) {
            if (state.request != focusRequest || camera !== activeCamera) return
            focusState.value = state.copy(finished = true, success = success)
            ActivityLog.log(if (success) "media.ready" else "media.error", "source" to "camera",
                "type" to "image", "interaction_stage" to "focus", "op" to state.op)
        }
        try {
            val action = FocusMeteringAction.Builder(point, FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE)
                .setAutoCancelDuration(5, TimeUnit.SECONDS).build()
            if (!activeCamera.cameraInfo.isFocusMeteringSupported(action)) {
                finish(false)
                return
            }
            val future = activeCamera.cameraControl.startFocusAndMetering(action)
            future.addListener({
                try { finish(future.get().isFocusSuccessful) }
                catch (_: Exception) { finish(false) }
            }, ContextCompat.getMainExecutor(context))
        } catch (_: Exception) { finish(false) }
    }

    private fun cancelFocusLog() {
        val state = focusState.value ?: return
        if (!state.finished) ActivityLog.log("media.close", "source" to "camera", "type" to "image",
            "interaction_stage" to "focus", "phase" to "cancelled", "op" to state.op)
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
            cancelFocusLog()
            focusRequest++
            provider?.unbind(preview, imageCapture)
            camera = null
            focusState.value = null
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
