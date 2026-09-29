package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.IMAGE_SPLIT_MAX_OVERLAP_PCT
import com.minashin1120.aiplayground.data.IMAGE_SPLIT_MAX_PIECES
import com.minashin1120.aiplayground.data.IMAGE_SPLIT_MIN_PIECES
import com.minashin1120.aiplayground.data.ImageSplitOptions
import com.minashin1120.aiplayground.data.ImageSplitRequest
import com.minashin1120.aiplayground.data.SplitPreview
import com.minashin1120.aiplayground.data.chooseSplitGrid
import com.minashin1120.aiplayground.data.clampSplitOverlap
import com.minashin1120.aiplayground.data.clampSplitPieces
import com.minashin1120.aiplayground.data.loadSplitPreview
import com.minashin1120.aiplayground.data.splitFits
import com.minashin1120.aiplayground.data.splitTiles
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

private val SplitMark = Color(0xFFFACC15)

/**
 * 画像分割 (ANDROID_ONLY.md): splits the picked or shared images into overlapping pieces so the model can
 * look at each part in more detail. The pieces are attached, or only saved to the device.
 */
@Composable
internal fun ImageSplitDialog(
    request: ImageSplitRequest,
    onDismiss: () -> Unit,
    onAttach: (ImageSplitOptions) -> Unit,
    onSaveOnly: (ImageSplitOptions) -> Unit,
    attachLabel: String = "分割して添付",
) {
    val web = LocalWebPalette.current
    val context = LocalContext.current
    var piecesText by rememberSaveable { mutableStateOf("4") }
    var overlapText by rememberSaveable { mutableStateOf("10") }
    var includeOriginal by rememberSaveable { mutableStateOf(true) }
    var markOriginal by rememberSaveable { mutableStateOf(true) }
    var markPieces by rememberSaveable { mutableStateOf(true) }
    var index by remember(request.uris) { mutableIntStateOf(0) }
    val uri = request.uris[index.coerceIn(0, request.uris.lastIndex)]
    var preview by remember(uri) { mutableStateOf<SplitPreview?>(null) }
    var previewError by remember(uri) { mutableStateOf<String?>(null) }
    LaunchedEffect(uri) {
        runCatching { withContext(Dispatchers.IO) { loadSplitPreview(context, uri) } }
            .onSuccess { preview = it }
            .onFailure { previewError = it.message ?: "画像を読み込めません。" }
    }
    val pieces = clampSplitPieces(piecesText.toIntOrNull())
    val overlap = clampSplitOverlap(overlapText.toIntOrNull())
    val options = ImageSplitOptions(pieces, overlap, includeOriginal, includeOriginal && markOriginal, markPieces)
    val labelColor = if (web.isLight) web.text else Tw.gray300
    WebOverlayModal({ if (!request.busy) onDismiss() }, grayOverlay(), 4.dp, alignment = Alignment.Center) { _ ->
        val shape = RoundedCornerShape(8.dp)
        Column(
            Modifier.padding(16.dp).widthIn(max = 448.dp).fillMaxWidth().modalPanelTaps().clip(shape)
                .background(web.twBg(Tw.gray800)).border(1.dp, web.twBorder(Tw.gray700), shape).padding(24.dp),
        ) {
            Row(Modifier.padding(bottom = 16.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FaIcon(R.drawable.fa_solid_image, null, size = 18.dp, tint = web.theme300)
                Text("画像を分割", fontSize = 20.sp, fontWeight = FontWeight.Bold, color = if (web.isLight) web.text else Color.White)
            }
            Column(Modifier.weight(1f, fill = false).verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(16.dp)) {
                Text("画像を重なりのある複数の画像に分け、モデルが細部まで読み取れるようにします。各画像には番号と元画像での位置が入り、点線が分割位置です。",
                    fontSize = 12.sp, color = Tw.gray400)
                SplitPreviewBox(preview, previewError, pieces, overlap)
                if (request.uris.size > 1) {
                    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                        PagerText("‹ 前へ", index > 0) { index-- }
                        Text("${index + 1} / ${request.uris.size}（すべて同じ設定で分割します）", fontSize = 12.sp, color = Tw.gray400,
                            modifier = Modifier.weight(1f).padding(horizontal = 8.dp))
                        PagerText("次へ ›", index < request.uris.lastIndex) { index++ }
                    }
                }
                preview?.source?.let { source ->
                    val grid = chooseSplitGrid(pieces, source.width, source.height)
                    if (splitFits(grid, source.width, source.height)) {
                        Text("${grid.columns}列 × ${grid.rows}行・1枚あたり約 ${source.width / grid.columns}×${source.height / grid.rows} px（元画像 ${source.width}×${source.height} px）",
                            fontSize = 12.sp, color = labelColor)
                    } else {
                        Text("この画像は小さすぎるため${pieces}枚に分割できません。分割数を減らしてください。", fontSize = 12.sp, color = Tw.red400)
                    }
                }
                Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                    SplitField("分割数（${IMAGE_SPLIT_MIN_PIECES}〜${IMAGE_SPLIT_MAX_PIECES}枚）", Modifier.weight(1f)) {
                        SettingsTextField(piecesText, { piecesText = it.filter(Char::isDigit).take(2) },
                            Modifier.fillMaxWidth(), number = true, deep = true)
                    }
                    SplitField("重なり（0〜${IMAGE_SPLIT_MAX_OVERLAP_PCT}%）", Modifier.weight(1f)) {
                        SettingsTextField(overlapText, { overlapText = it.filter(Char::isDigit).take(2) },
                            Modifier.fillMaxWidth(), number = true, deep = true)
                    }
                }
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    SettingsCheck("元の画像も含める", includeOriginal, { includeOriginal = it }, boxSize = 16.dp, labelColor = labelColor)
                    SettingsCheck("元の画像に分割位置と番号を描き込む", markOriginal, { markOriginal = it }, enabled = includeOriginal,
                        boxSize = 16.dp, labelColor = labelColor, modifier = Modifier.padding(start = 24.dp))
                    SettingsCheck("分割した画像に番号と分割線を描き込む", markPieces, { markPieces = it }, boxSize = 16.dp, labelColor = labelColor)
                }
                val perImage = pieces + if (includeOriginal) 1 else 0
                Text("作成される画像: ${perImage * request.uris.size}枚", fontSize = 12.sp, color = Tw.gray400)
                request.error?.let { Text(it, fontSize = 12.sp, color = Tw.red400) }
            }
            Row(
                Modifier.fillMaxWidth().padding(top = 24.dp).drawBehind {
                    drawLine(web.twBorder(Tw.gray700), Offset(0f, 0f), Offset(size.width, 0f), 1.dp.toPx())
                }.padding(top = 16.dp).alpha(if (request.busy) 0.6f else 1f),
                horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.End),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                if (request.busy) {
                    Text("処理中...", fontSize = 14.sp, color = Tw.gray400, modifier = Modifier.padding(horizontal = 8.dp, vertical = 8.dp))
                } else {
                    Text("キャンセル", fontSize = 14.sp, color = Tw.gray400,
                        modifier = Modifier.clickable(role = Role.Button, onClick = onDismiss).padding(horizontal = 8.dp, vertical = 8.dp))
                    Text("保存のみ", fontSize = 14.sp, fontWeight = FontWeight.Bold, color = if (web.isLight) web.text else Color.White,
                        modifier = Modifier.clip(RoundedCornerShape(4.dp)).background(web.twBg(Tw.gray700))
                            .clickable(role = Role.Button) { onSaveOnly(options) }.padding(horizontal = 16.dp, vertical = 8.dp))
                    Text(attachLabel, fontSize = 14.sp, fontWeight = FontWeight.Bold, color = Color.White,
                        modifier = Modifier.clip(RoundedCornerShape(4.dp)).background(web.theme.t600)
                            .clickable(role = Role.Button) { onAttach(options) }.padding(horizontal = 16.dp, vertical = 8.dp))
                }
            }
        }
    }
}

/** The image with the overlap bands and cut lines of the current settings. */
@Composable
private fun SplitPreviewBox(preview: SplitPreview?, error: String?, pieces: Int, overlap: Int) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(4.dp)
    val frame = Modifier.fillMaxWidth().clip(shape).background(web.twBg(Tw.gray900)).border(1.dp, web.twBorder(Tw.gray700), shape)
    if (preview == null) {
        Box(frame.height(160.dp), contentAlignment = Alignment.Center) {
            Text(error ?: "読み込み中...", fontSize = 12.sp, color = if (error != null) Tw.red400 else Tw.gray500)
        }
        return
    }
    val source = preview.source
    val bitmap = remember(preview) { preview.bitmap.asImageBitmap() }
    Box(frame.heightIn(max = 260.dp), contentAlignment = Alignment.Center) {
        Box(Modifier.aspectRatio(source.width.toFloat() / source.height, matchHeightConstraintsFirst = source.height > source.width)) {
            Image(bitmap, contentDescription = source.name, contentScale = ContentScale.Fit, modifier = Modifier.fillMaxSize())
            Canvas(Modifier.fillMaxSize()) {
                val grid = chooseSplitGrid(pieces, source.width, source.height)
                if (!splitFits(grid, source.width, source.height)) return@Canvas
                val tiles = splitTiles(source.width, source.height, grid, overlap)
                val sx = size.width / source.width
                val sy = size.height / source.height
                val stroke = 1.5.dp.toPx()
                val first = tiles.first()
                val padX = first.right - first.cellRight
                val padY = first.bottom - first.cellBottom
                tiles.filter { it.row == 0 && it.column > 0 }.forEach { tile ->
                    val x = tile.cellLeft * sx
                    if (padX > 0) drawRect(SplitMark.copy(alpha = 0.22f), Offset(x - padX * sx, 0f), Size(padX * 2 * sx, size.height))
                    drawLine(SplitMark, Offset(x, 0f), Offset(x, size.height), stroke)
                }
                tiles.filter { it.column == 0 && it.row > 0 }.forEach { tile ->
                    val y = tile.cellTop * sy
                    if (padY > 0) drawRect(SplitMark.copy(alpha = 0.22f), Offset(0f, y - padY * sy), Size(size.width, padY * 2 * sy))
                    drawLine(SplitMark, Offset(0f, y), Offset(size.width, y), stroke)
                }
            }
        }
    }
}

@Composable
private fun PagerText(label: String, enabled: Boolean, onClick: () -> Unit) {
    val web = LocalWebPalette.current
    Text(label, fontSize = 12.sp, color = web.twText(Tw.gray300),
        modifier = Modifier.alpha(if (enabled) 1f else 0.4f).clip(RoundedCornerShape(4.dp)).background(web.twBg(Tw.gray700))
            .then(if (enabled) Modifier.clickable(role = Role.Button, onClick = onClick) else Modifier)
            .padding(horizontal = 8.dp, vertical = 4.dp))
}

@Composable
private fun SplitField(label: String, modifier: Modifier, field: @Composable () -> Unit) {
    Column(modifier) {
        Text(label, fontSize = 12.sp, color = Tw.gray400, modifier = Modifier.padding(bottom = 4.dp))
        field()
    }
}
