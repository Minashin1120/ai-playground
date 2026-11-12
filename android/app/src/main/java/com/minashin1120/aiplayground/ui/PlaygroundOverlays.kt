package com.minashin1120.aiplayground.ui

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import com.minashin1120.aiplayground.data.Attachment
import com.minashin1120.aiplayground.data.ImageSplitOptions
import com.minashin1120.aiplayground.data.Skill
import com.minashin1120.aiplayground.data.ThreadItem

/** A system picker the upload sheet asks for after it has closed (see the launch effect in `PlaygroundScreen`). */
internal enum class AttachmentPicker { Files, Photos, Split }

/**
 * Dialog, sheet and overlay state of the chat screen.
 *
 * These flags used to be separate `remember { mutableStateOf(...) }` locals in `PlaygroundScreen`;
 * keeping them in one snapshot-state holder lets [PlaygroundDialogs] own the dialogs while the
 * screen still drives the back button, the panel log and the composer callbacks.
 */
internal class PlaygroundOverlays {
    var deleting by mutableStateOf<ThreadItem?>(null)
    var modelPicker by mutableStateOf(false)
    var threadSettings by mutableStateOf(false)
    var attachMenu by mutableStateOf(false)
    var libraryOpen by mutableStateOf(false)
    var viewingFile by mutableStateOf<FileViewRequest?>(null)
    var skillEditorOpen by mutableStateOf(false)
    var skillEditing by mutableStateOf<Skill?>(null)
    var skillDeleting by mutableStateOf<Skill?>(null)
    var historyOpen by mutableStateOf(false)
    var branchOpen by mutableStateOf(false)
    var alphaOpen by mutableStateOf(false)
    var legalKind by mutableStateOf<String?>(null)
    var renaming by mutableStateOf<ThreadItem?>(null)
    var settingsOpen by mutableStateOf(false)
    var settingsTab by mutableStateOf("一般")
    var settingsCard by mutableStateOf<String?>(null)
    var compressionOpen by mutableStateOf(false)
    var changelogOpen by mutableStateOf(false)
    var advancedOpen by mutableStateOf(false)
    var realtimeOpen by mutableStateOf(false)
    var lyriaOpen by mutableStateOf(false)
    var awaitingMic by mutableStateOf(false)
    var realtimeOptions by mutableStateOf(RealtimeOptions())
    var startDockAfterMic by mutableStateOf(false)
    var richPasteOpen by mutableStateOf(false)
    var markerTarget by mutableStateOf<Attachment?>(null)
    var markerMaskMode by mutableStateOf(false)
    var visionPicker by mutableStateOf(false)
    var cameraOpen by mutableStateOf(false)
    var pendingAttachmentPicker by mutableStateOf<AttachmentPicker?>(null)
    var pendingSplitSave by mutableStateOf<ImageSplitOptions?>(null)
}
