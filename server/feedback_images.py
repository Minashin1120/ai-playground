# --- Images attached to a feedback ---
# The client sends the feedback first (``POST /api/feedback``), then its images to
# ``POST /api/feedback/<id>/images`` (the same two steps as the device files of a chat copy). They are kept
# in ``feedback/<public_id>/images/`` next to ``feedback.json``; deleting the feedback removes the directory.
# The type is decided from the file header, never from the name or the declared type, and nothing is decoded.
_FEEDBACK_IMAGES_DIR = 'images'
_FEEDBACK_IMAGES_MAX_COUNT = 4
_FEEDBACK_IMAGE_MAX_BYTES = 5 * 1024 * 1024
_FEEDBACK_IMAGES_WINDOW = 3600


def _feedback_image_kind(head):
    if head.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'png'
    if head.startswith(b'\xff\xd8\xff'):
        return 'jpg'
    if head[:4] == b'RIFF' and head[8:12] == b'WEBP':
        return 'webp'
    if head[:6] in (b'GIF87a', b'GIF89a'):
        return 'gif'
    return None


def _feedback_images_path(public_id):
    return os.path.join(_feedback_dir(public_id), _FEEDBACK_IMAGES_DIR)


def _feedback_image_names(public_id):
    try:
        return sorted(name for name in os.listdir(_feedback_images_path(public_id)) if re.fullmatch(r'\d{2}\.(?:png|jpg|webp|gif)', name))
    except OSError:
        return []


@app.route('/api/feedback/<fid>/images', methods=['POST'])
@login_required
def feedback_images(fid):
    """Images added to the user's own feedback ``fid`` right after it was sent."""
    fb = _feedback_by_ref(fid)
    if not fb or fb.user_id != current_user.id or not fb.public_id:
        return jsonify({'error': 'not_found'}), 404
    if not os.path.isdir(_feedback_dir(fb.public_id)):
        return jsonify({'error': 'not_found'}), 404
    if fb.created_at and (datetime.utcnow() - fb.created_at).total_seconds() > _FEEDBACK_IMAGES_WINDOW:
        return jsonify({'error': 'expired'}), 409
    if not rate_limit(f"rl:feedback_images:user:{current_user.id}", 20, 3600):
        return jsonify({'error': 'rate_limit'}), 429
    uploads = [u for u in request.files.getlist('images') if u and u.filename is not None]
    if not uploads:
        return jsonify({'error': 'images_required'}), 400
    existing = _feedback_image_names(fb.public_id)
    if len(existing) + len(uploads) > _FEEDBACK_IMAGES_MAX_COUNT:
        return jsonify({'error': 'too_many_images'}), 400
    accepted = []
    for upload in uploads:
        data = upload.stream.read(_FEEDBACK_IMAGE_MAX_BYTES + 1)
        if len(data) > _FEEDBACK_IMAGE_MAX_BYTES:
            return jsonify({'error': 'image_too_large'}), 413
        kind = _feedback_image_kind(data[:16])
        if kind is None:
            return jsonify({'error': 'unsupported_image'}), 415
        accepted.append((kind, data))
    directory = _feedback_images_path(fb.public_id)
    os.makedirs(directory, mode=0o700, exist_ok=True)
    number = max([int(name[:2]) for name in existing] + [0])
    for kind, data in accepted:
        number += 1
        fd = os.open(os.path.join(directory, f'{number:02d}.{kind}'), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
    names = _feedback_image_names(fb.public_id)
    try:
        _write_feedback_info(fb, images=[
            {'file': name, 'bytes': os.path.getsize(os.path.join(directory, name))} for name in names])
    except Exception as e:
        log_force(f"FEEDBACK-INFO-ERROR: feedback={fb.public_id} err={e}")
    return jsonify({'saved': len(accepted), 'images': len(names)})
