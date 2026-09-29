package com.minashin1120.aiplayground

import android.app.PendingIntent
import android.content.Intent
import android.os.Build
import android.service.quicksettings.TileService

/** Quick Settings entry that keeps the currently visible app behind the split dialog. */
class ImageSplitTileService : TileService() {
    override fun onClick() {
        super.onClick()
        if (isLocked) unlockAndRun(::openSplit) else openSplit()
    }

    private fun openSplit() {
        val intent = Intent(this, ImageSplitOverlayActivity::class.java)
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            val pending = PendingIntent.getActivity(this, 0, intent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
            startActivityAndCollapse(pending)
        } else {
            @Suppress("DEPRECATION")
            startActivityAndCollapse(intent)
        }
    }
}
