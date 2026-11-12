# --- Background: Python sandbox tool ---

def _sanitize_python_sandbox_output(value):
    """Remove host filesystem paths before Python output reaches the client."""
    text = str(value or "")
    text = re.sub(
        r"(?<![A-Za-z0-9_.-])/(?:home|root)/[^\s'\"`<>]+",
        "[host path redacted]",
        text,
    )
    app_root = os.path.abspath(os.path.dirname(__file__))
    text = text.replace(app_root, "[app path redacted]")
    return text


def safe_execute_python(code, on_image=None, input_images=()):
    """Executes Python code in a restricted environment using bubblewrap."""
    import subprocess
    import tempfile
    import os
    import shutil

    # The sandbox only exposes the system filesystem below.  ``shutil.which``
    # normally resolves to the application's venv, but that host path is not
    # mounted into bubblewrap and therefore cannot be executed there.
    py_path = None
    for candidate in ("/usr/bin/python3", "/bin/python3"):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            py_path = candidate
            break
    if not py_path:
        candidate = shutil.which("python3")
        if candidate:
            resolved = os.path.realpath(candidate)
            if resolved.startswith(("/usr/", "/bin/")) and os.path.isfile(resolved) and os.access(resolved, os.X_OK):
                py_path = resolved
    if not py_path:
        return "Error: python3 not found."

    bwrap = shutil.which("bwrap")
    if not bwrap:
        return "Error: Python execution disabled (sandbox not available)."
    prlimit = shutil.which("prlimit")
    if not prlimit:
        return "Error: Python execution disabled (resource limiter not available)."
    code = str(code or "")
    if len(code.encode("utf-8", errors="ignore")) > 256 * 1024:
        return "Error: Python code exceeds the 256KB limit."

    with tempfile.TemporaryDirectory() as td:
        code_path = os.path.join(td, "code.py")
        with open(code_path, "w") as f:
            f.write(code)
        # Fixed names only; no provider-supplied path can escape the disposable sandbox.
        for image_index, image in enumerate(input_images[:8], 1):
            data = image.get("bytes") or b""
            if len(data) > 8 * 1024 * 1024:
                continue
            try:
                image_data, ext = _prepare_agentic_image_bytes(data, image.get("mime"))
            except ValueError:
                continue
            with open(os.path.join(td, f"input_{image_index}.{ext}"), "wb") as image_file:
                image_file.write(image_data)
        binds = [
            ("--ro-bind", "/usr", "/usr"),
            ("--ro-bind", "/bin", "/bin"),
        ]
        for p in ["/lib", "/lib64"]:
            if os.path.exists(p):
                binds.append(("--ro-bind", p, p))
        cmd = [
            prlimit,
            "--cpu=15",
            "--as=536870912",
            "--fsize=8388608",
            # RLIMIT_NPROC is enforced for the whole service UID, not only
            # processes spawned by this sandbox.  The gunicorn/RQ threads
            # already consume part of that UID-wide budget, so 64 can make
            # bwrap fail before Python starts (EAGAIN) in a healthy service.
            # Keep a bounded sandbox process budget while leaving headroom for
            # the service's resident threads and the namespace helper.
            "--nproc=256",
            "--nofile=64",
            "--",
            bwrap,
            # Never expose the service environment (.env secrets such as the
            # Flask session key and provider API keys) to model-written code.
            "--clearenv",
            "--setenv", "PATH", "/usr/bin:/bin",
            "--setenv", "HOME", "/tmp",  # nosec B108 - private path inside the bwrap mount namespace
            "--setenv", "LANG", "C.UTF-8",
            "--setenv", "PYTHONIOENCODING", "utf-8",
            "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
            "--new-session",
            "--unshare-net",
            "--unshare-uts",
            "--unshare-pid",
            "--unshare-ipc",
            "--die-with-parent",
            "--proc", "/proc",
            "--dev", "/dev",
            "--tmpfs", "/home",
            "--tmpfs", "/var",
            "--dir", "/tmp",  # nosec B108 - private path inside the bwrap mount namespace
            "--chdir", "/tmp",  # nosec B108 - private path inside the bwrap mount namespace
        ]
        for b in binds:
            cmd.extend(list(b))
        cmd.extend(["--bind", td, "/work", "--chdir", "/work", py_path, "/work/code.py"])
        try:
            with tempfile.TemporaryFile(mode="w+b") as output:
                subprocess.run(
                    cmd, stdout=output, stderr=subprocess.STDOUT, timeout=30, check=False,
                    env={"PATH": "/usr/bin:/bin"},
                )
                output.seek(0)
                raw = output.read(2 * 1024 * 1024 + 1)
            truncated = len(raw) > 2 * 1024 * 1024
            out = raw[:2 * 1024 * 1024].decode("utf-8", errors="replace")
            if truncated:
                out += "\n[Output truncated at 2MB]"
            out = _sanitize_python_sandbox_output(out)
            if on_image is not None:
                import stat
                exported = 0
                total = 0
                # Never follow links made by model-written code or read arbitrary host files.
                with os.scandir(td) as entries:
                    for entry in entries:
                        if exported >= 8 or total >= 16 * 1024 * 1024:
                            break
                        if entry.name.startswith("input_") or not entry.name.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".svg")):
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        fd = os.open(entry.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                        with os.fdopen(fd, "rb") as image_file:
                            info = os.fstat(image_file.fileno())
                            if not stat.S_ISREG(info.st_mode) or info.st_size > min(8 * 1024 * 1024, 16 * 1024 * 1024 - total):
                                continue
                            image_bytes = image_file.read(8 * 1024 * 1024 + 1)
                        try:
                            on_image(image_bytes)
                        except ValueError:
                            continue
                        exported += 1
                        total += len(image_bytes)
            return out if out.strip() else "Success (No output)"
        except subprocess.TimeoutExpired:
            return "Error: Execution timed out (30s limit)"
        except Exception as e:
            return _sanitize_python_sandbox_output(f"Error: {e}")


