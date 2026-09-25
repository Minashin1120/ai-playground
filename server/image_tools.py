import struct
import zlib

_IMAGE_MAX_PIXELS = 40_000_000
_IMAGE_THUMBNAIL_TIMEOUT = 10
_IMAGE_THUMBNAIL_MAX_OUTPUT = 2 * 1024 * 1024
_IMAGE_CONVERT_TIMEOUT = 30
_IMAGE_CONVERT_MAX_OUTPUT = 50 * 1024 * 1024
_ISOBMFF_SCAN_LIMIT = 1024 * 1024
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_IMAGE_TOOL_PATHS = {name: shutil.which(name) for name in ("nice", "cwebp", "dwebp", "ffmpeg")}
_CWEBP_INPUT_FORMATS = frozenset({"PNG", "JPEG", "WEBP"})
# ffmpeg must never auto-detect the demuxer: playlist-style formats could open local files or URLs.
_FFMPEG_DEMUXERS = {
    "PNG": "png_pipe",
    "JPEG": "jpeg_pipe",
    "WEBP": "webp_pipe",
    "GIF": "gif",
    "BMP": "bmp_pipe",
    "AVIF": "mov",
}
_JPEG_SOF_MARKERS = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
_AVIF_BRANDS = frozenset({b"avif", b"avis"})
_HEIC_BRANDS = frozenset({b"heic", b"heix", b"heim", b"heis", b"hevc", b"hevx", b"heif"})


def _image_info(image_format, width, height, rgba8=False, animated=False):
    return {
        "format": image_format,
        "width": int(width),
        "height": int(height),
        "rgba8": bool(rgba8),
        "animated": bool(animated),
    }


def _probe_jpeg(data):
    pos = 2
    size = len(data)
    while pos + 4 <= size:
        if data[pos] != 0xFF:
            return None
        marker = data[pos + 1]
        if marker == 0xFF:
            pos += 1
            continue
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:
            pos += 2
            continue
        if marker in (0xD9, 0xDA):
            return None
        segment_length = int.from_bytes(data[pos + 2:pos + 4], "big")
        if segment_length < 2:
            return None
        if marker in _JPEG_SOF_MARKERS:
            if pos + 9 > size:
                return None
            height = int.from_bytes(data[pos + 5:pos + 7], "big")
            width = int.from_bytes(data[pos + 7:pos + 9], "big")
            return _image_info("JPEG", width, height)
        pos += 2 + segment_length
    return None


def _probe_webp(data):
    if len(data) < 30:
        return None
    chunk = data[12:16]
    if chunk == b"VP8X":
        flags = data[20]
        width = 1 + int.from_bytes(data[24:27], "little")
        height = 1 + int.from_bytes(data[27:30], "little")
        return _image_info("WEBP", width, height, animated=bool(flags & 0x02))
    if chunk == b"VP8 ":
        if data[23:26] != b"\x9d\x01\x2a":
            return None
        width = int.from_bytes(data[26:28], "little") & 0x3FFF
        height = int.from_bytes(data[28:30], "little") & 0x3FFF
        return _image_info("WEBP", width, height)
    if chunk == b"VP8L":
        if data[20] != 0x2F:
            return None
        bits = int.from_bytes(data[21:25], "little")
        return _image_info("WEBP", (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1)
    return None


def _probe_isobmff(data):
    ftyp_size = int.from_bytes(data[0:4], "big")
    if ftyp_size < 16 or ftyp_size > min(len(data), 4096):
        return None
    major = data[8:12]
    compatible = {data[pos:pos + 4] for pos in range(16, ftyp_size - 3, 4)}
    if major in _AVIF_BRANDS:
        image_format = "AVIF"
    elif major in _HEIC_BRANDS:
        image_format = "HEIC"
    elif compatible & _AVIF_BRANDS:
        image_format = "AVIF"
    elif compatible & _HEIC_BRANDS:
        image_format = "HEIC"
    else:
        return None
    # Grid images list one `ispe` per tile, so keep the largest extent.
    width = height = 0
    limit = min(len(data), _ISOBMFF_SCAN_LIMIT)
    pos = data.find(b"ispe", 0, limit)
    while pos >= 4 and pos + 16 <= len(data):
        box_width = int.from_bytes(data[pos + 8:pos + 12], "big")
        box_height = int.from_bytes(data[pos + 12:pos + 16], "big")
        if box_width * box_height > width * height:
            width, height = box_width, box_height
        pos = data.find(b"ispe", pos + 4, limit)
    if not width or not height:
        return None
    return _image_info(image_format, width, height)


def _probe_image(data):
    if isinstance(data, (bytearray, memoryview)):
        data = bytes(data)
    if not isinstance(data, bytes) or len(data) < 10:
        return None
    try:
        if data[:8] == _PNG_SIGNATURE:
            if len(data) < 33 or data[12:16] != b"IHDR":
                return None
            width, height = struct.unpack(">II", data[16:24])
            bit_depth, color_type = data[24], data[25]
            return _image_info("PNG", width, height, rgba8=(color_type == 6 and bit_depth == 8))
        if data[:2] == b"\xff\xd8":
            return _probe_jpeg(data)
        if data[:6] in (b"GIF87a", b"GIF89a"):
            width, height = struct.unpack("<HH", data[6:10])
            return _image_info("GIF", width, height)
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return _probe_webp(data)
        if data[:2] == b"BM" and len(data) >= 26:
            dib_size = int.from_bytes(data[14:18], "little")
            if dib_size == 12:
                width, height = struct.unpack("<HH", data[18:22])
            elif dib_size >= 40:
                width, height = struct.unpack("<ii", data[18:26])
            else:
                return None
            return _image_info("BMP", abs(width), abs(height))
        if data[4:8] == b"ftyp":
            return _probe_isobmff(data)
    except (IndexError, struct.error):
        return None
    return None


def _image_pixels_ok(info):
    if not info:
        return False
    width, height = info.get("width") or 0, info.get("height") or 0
    return width > 0 and height > 0 and width * height <= _IMAGE_MAX_PIXELS


def _png_chunks_valid(data):
    pos = len(_PNG_SIGNATURE)
    size = len(data)
    has_idat = False
    while pos + 12 <= size:
        length = int.from_bytes(data[pos:pos + 4], "big")
        chunk_end = pos + 12 + length
        if chunk_end > size:
            return False
        chunk_type = data[pos + 4:pos + 8]
        expected_crc = int.from_bytes(data[chunk_end - 4:chunk_end], "big")
        if zlib.crc32(data[pos + 4:chunk_end - 4]) != expected_crc:
            return False
        if chunk_type == b"IDAT":
            has_idat = True
        elif chunk_type == b"IEND":
            return has_idat
        pos = chunk_end
    return False


def _validate_image_structure(data, info):
    if not _image_pixels_ok(info):
        return False
    image_format = info.get("format")
    if image_format == "PNG":
        return _png_chunks_valid(data)
    if image_format == "JPEG":
        scan_start = data.find(b"\xff\xda")
        return scan_start > 0 and data.rfind(b"\xff\xd9") > scan_start
    if image_format == "GIF":
        return data.rstrip(b"\x00")[-1:] == b";"
    if image_format == "WEBP":
        riff_size = int.from_bytes(data[4:8], "little")
        return 12 <= riff_size and riff_size + 8 <= len(data)
    return True


def _is_png_bytes(data):
    return bool(data) and data[:8] == _PNG_SIGNATURE


def _is_webp_bytes(data):
    return bool(data) and data[:4] == b"RIFF" and data[8:12] == b"WEBP"


def _run_image_command(args, data, timeout, max_output):
    program = args[0] if os.path.isabs(args[0]) else _IMAGE_TOOL_PATHS.get(args[0])
    if not program:
        return None
    command = [program, *args[1:]]
    if _IMAGE_TOOL_PATHS.get("nice"):
        command = [_IMAGE_TOOL_PATHS["nice"], "-n", "10", *command]
    try:
        result = subprocess.run(command, input=data, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Image command %s failed: %s", os.path.basename(program), exc)
        return None
    if result.returncode != 0:
        stderr_tail = result.stderr.decode("utf-8", "replace").strip()[-300:]
        logger.warning("Image command %s exited %s: %s", os.path.basename(program), result.returncode, stderr_tail)
        return None
    if not result.stdout or len(result.stdout) > max_output:
        return None
    return result.stdout


def _run_ffmpeg_image(info, data, output_args, timeout, max_output):
    demuxer = _FFMPEG_DEMUXERS.get(info.get("format"))
    if not demuxer or info.get("animated"):
        return None
    return _run_image_command(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-protocol_whitelist", "pipe", "-f", demuxer, "-i", "pipe:0",
            "-frames:v", "1", *output_args,
        ],
        data,
        timeout,
        max_output,
    )


def _thumbnail_dimensions(width, height, max_side):
    scale = min(1.0, float(max_side) / max(width, height))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _make_thumbnail_webp(data, max_side, quality):
    info = _probe_image(data)
    if not _image_pixels_ok(info) or info.get("animated"):
        return None
    width, height = _thumbnail_dimensions(info["width"], info["height"], max_side)
    if info["format"] in _CWEBP_INPUT_FORMATS:
        thumb = _run_image_command(
            [
                "cwebp", "-quiet", "-q", str(int(quality)), "-m", "4",
                "-resize", str(width), str(height), "-metadata", "none",
                "-o", "-", "--", "-",
            ],
            data,
            _IMAGE_THUMBNAIL_TIMEOUT,
            _IMAGE_THUMBNAIL_MAX_OUTPUT,
        )
        if _is_webp_bytes(thumb):
            return thumb
    thumb = _run_ffmpeg_image(
        info,
        data,
        [
            "-vf", f"scale={width}:{height}:flags=lanczos",
            "-c:v", "libwebp", "-quality", str(int(quality)), "-compression_level", "4",
            "-f", "webp", "pipe:1",
        ],
        _IMAGE_THUMBNAIL_TIMEOUT,
        _IMAGE_THUMBNAIL_MAX_OUTPUT,
    )
    return thumb if _is_webp_bytes(thumb) else None


def _convert_image_to_png(data, rgba=False):
    info = _probe_image(data)
    if not _image_pixels_ok(info) or info.get("animated"):
        return None
    png = None
    if info["format"] == "WEBP" and not rgba:
        png = _run_image_command(
            ["dwebp", "-quiet", "-o", "-", "--", "-"],
            data,
            _IMAGE_CONVERT_TIMEOUT,
            _IMAGE_CONVERT_MAX_OUTPUT,
        )
    if not _is_png_bytes(png):
        png = _run_ffmpeg_image(
            info,
            data,
            [*(["-pix_fmt", "rgba"] if rgba else []), "-c:v", "png", "-f", "image2pipe", "pipe:1"],
            _IMAGE_CONVERT_TIMEOUT,
            _IMAGE_CONVERT_MAX_OUTPUT,
        )
    converted = _probe_image(png) if _is_png_bytes(png) else None
    if not converted or (converted["width"], converted["height"]) != (info["width"], info["height"]):
        return None
    return png


def _png_chunk(chunk_type, payload):
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", zlib.crc32(chunk_type + payload))
    )


def _encode_png(width, height, color_type, bit_depth, rows):
    raw = b"".join(b"\x00" + bytes(row) for row in rows)
    header = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0)
    return (
        _PNG_SIGNATURE
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(raw, 9))
        + _png_chunk(b"IEND", b"")
    )


def _qr_png_bytes(text, box_size=10, border=4):
    import qrcode

    qr = qrcode.QRCode(box_size=box_size, border=border)
    qr.add_data(text)
    qr.make(fit=True)
    matrix = qr.get_matrix()
    size = len(matrix) * box_size
    row_bytes = (size + 7) // 8
    rows = []
    for modules in matrix:
        bits = "".join(("0" if dark else "1") * box_size for dark in modules)
        packed = int(bits.ljust(row_bytes * 8, "1"), 2).to_bytes(row_bytes, "big")
        rows.extend([packed] * box_size)
    return _encode_png(size, size, 0, 1, rows)


def _rasterize_svg_png(svg_bytes, width, height):
    png = _run_image_command(
        [
            sys.executable, "-m", "cairosvg", "-f", "png",
            "--output-width", str(int(width)), "--output-height", str(int(height)),
            "-o", "-", "-",
        ],
        svg_bytes,
        _IMAGE_CONVERT_TIMEOUT,
        _IMAGE_CONVERT_MAX_OUTPUT,
    )
    return png if _is_png_bytes(png) else None
