"""OpenAI code interpreter outputs (thinking with images) for the Responses branch.

Extracted from `server/background_stream_openai.py`.  The helpers used to be
closures inside ``background_stream_openai``: they appended to the branch's
``full_res`` string, which cannot cross a module boundary.  They now live on a
small state holder that receives ``append_content`` / ``append_text`` /
``get_full_res`` callbacks from the branch, so the stream loop only calls
methods.

Module-level helpers are resolved through the `app` namespace because `app.py`
execs every server part into one module.
"""


class OpenAICodeInterpreterOutputs:
    """Collect and publish images/files produced by the hosted Python tool."""

    def __init__(self, client, pub, user_id, user_config,
                 agentic_image_digests, agentic_saved_urls, agentic_filename_url_map,
                 append_content, append_text, get_full_res):
        self.client = client
        self.pub = pub
        self.user_id = user_id
        self.user_config = user_config
        self.agentic_image_digests = agentic_image_digests
        self.agentic_saved_urls = agentic_saved_urls
        self.agentic_filename_url_map = agentic_filename_url_map
        self.append_content = append_content
        self.append_text = append_text
        self.get_full_res = get_full_res
        self.saved_files = {}
        self.image_urls = {}
        self.published_calls = set()
        self.seen_output_urls = set()
        self.container_id = None

    def download_api_bytes(self, url):
        if isinstance(url, str) and url.startswith("data:image/"):
            match = re.fullmatch(r"data:image/(?:png|jpeg|webp|gif);base64,([A-Za-z0-9+/=\r\n]+)", url)
            if not match or len(match.group(1)) > 28 * 1024 * 1024:
                return None
            try:
                data = base64.b64decode(match.group(1).replace("\r", "").replace("\n", ""), validate=True)
            except ValueError:
                log_force("OpenAI code interpreter image decode failed")
                return None
            return data if data and len(data) <= 20 * 1024 * 1024 else None
        parsed = urlparse(str(url or ""))
        if parsed.scheme != "https" or (parsed.hostname or "").lower() != "api.openai.com":
            return None
        if not (parsed.path.startswith("/v1/containers/") or parsed.path.startswith("/v1/files/")):
            return None
        api_key = getattr(self.client, "api_key", None)
        if not api_key:
            return None
        try:
            with httpx.Client(timeout=60, follow_redirects=False) as http:
                resp = http.get(url, headers={"Authorization": f"Bearer {api_key}"})
                if resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get("location") or ""
                    if not location.startswith("https://"):
                        return None
                    resp = http.get(location)
                if resp.status_code != 200:
                    log_force(f"OpenAI code interpreter download failed: status={resp.status_code}")
                    return None
                data = resp.content
        except Exception as download_exc:
            log_force(f"OpenAI code interpreter download failed: {type(download_exc).__name__}")
            return None
        if not data or len(data) > 20 * 1024 * 1024:
            return None
        return data

    def emit_image(self, data, filename):
        if not data:
            return
        try:
            image_data, ext = _prepare_agentic_image_bytes(data)
        except Exception:
            log_force("OpenAI code interpreter output skipped: not an image")
            return
        digest = hashlib.sha256(image_data).hexdigest()
        if digest in self.agentic_image_digests:
            return self.image_urls.get(digest)
        self.agentic_image_digests.add(digest)
        _, url = _save_user_generated_bytes_verified(
            self.user_id, image_data,
            lambda: f"agentic_{int(time.time() * 1000)}_{os.urandom(4).hex()}.{ext}",
            self.user_config.get('enable_e2ee'),
        )
        self.agentic_saved_urls.append(url)
        self.image_urls[digest] = url
        base = os.path.basename(str(filename or "")).lower()
        if base:
            self.agentic_filename_url_map[base] = url
        if not (base and base in self.get_full_res().lower()):
            markdown = f"\n![Agentic View]({url})\n"
            self.append_content(markdown)
        return url

    def save_container_file(self, container_id, file_id, filename):
        if file_id in self.saved_files:
            return self.saved_files[file_id]
        if not (isinstance(container_id, str) and container_id.startswith("cntr_") and re.fullmatch(r"cntr_[A-Za-z0-9]+", container_id)):
            return
        if not (isinstance(file_id, str) and file_id.startswith("cfile_") and re.fullmatch(r"cfile_[A-Za-z0-9]+", file_id)):
            return
        data = self.download_api_bytes(
            f"https://api.openai.com/v1/containers/{container_id}/files/{file_id}/content"
        )
        url = self.emit_image(data, filename)
        if url:
            self.saved_files[file_id] = url
        return url

    def consume_container_citation(self, ann):
        if isinstance(ann, dict):
            ann_type = ann.get("type")
            container_id = ann.get("container_id") or self.container_id
            file_id = ann.get("file_id")
            filename = ann.get("filename")
        else:
            ann_type = getattr(ann, "type", None)
            container_id = getattr(ann, "container_id", None) or self.container_id
            file_id = getattr(ann, "file_id", None)
            filename = getattr(ann, "filename", None)
        if ann_type != "container_file_citation":
            return False
        self.save_container_file(container_id, file_id, filename)
        return True

    def publish_local_python_image(self, data):
        try:
            image_data, ext = _prepare_agentic_image_bytes(data)
        except Exception:
            return
        digest = hashlib.sha256(image_data).hexdigest()
        if digest in self.agentic_image_digests:
            return
        self.agentic_image_digests.add(digest)
        _, url = _save_user_generated_bytes_verified(
            self.user_id, image_data,
            lambda: f"agentic_{int(time.time() * 1000)}_{os.urandom(4).hex()}.{ext}",
            self.user_config.get('enable_e2ee'),
        )
        self.agentic_saved_urls.append(url)
        markdown = f"\n![Agentic View]({url})\n"
        self.append_content(markdown)

    def publish_code_interpreter_call(self, item):
        if isinstance(item, dict):
            code = item.get("code") or ""
            container_id = item.get("container_id")
            outputs = item.get("outputs") or []
            call_id = item.get("id")
        else:
            code = getattr(item, "code", None) or ""
            container_id = getattr(item, "container_id", None)
            outputs = getattr(item, "outputs", None) or []
            call_id = getattr(item, "id", None)
        already = bool(call_id and call_id in self.published_calls)
        if call_id:
            self.published_calls.add(call_id)
        if isinstance(container_id, str) and container_id.startswith("cntr_"):
            self.container_id = container_id
        logs = []
        images = []
        for output in outputs:
            if isinstance(output, dict):
                out_type = output.get("type")
                out_logs = output.get("logs")
                out_url = output.get("url")
                out_file = output.get("file_id")
            else:
                out_type = getattr(output, "type", None)
                out_logs = getattr(output, "logs", None)
                out_url = getattr(output, "url", None)
                out_file = getattr(output, "file_id", None)
            if out_type == "logs" and out_logs and not already:
                logs.append(str(out_logs))
            elif out_type == "image" and isinstance(out_file, str) and out_file.startswith("cfile_"):
                saved = self.save_container_file(container_id or self.container_id, out_file, "output.png")
                if saved:
                    images.append(saved)
            elif out_type == "image" and out_url and out_url not in self.seen_output_urls:
                self.seen_output_urls.add(out_url)
                saved = self.emit_image(self.download_api_bytes(out_url), "output.png")
                if saved:
                    images.append(saved)
        if already:
            return
        if code:
            block = f"\n```python\n{code}\n```\n"
            self.append_content(block)
        output_text = "\n".join(logs)
        if output_text:
            out_md = f"\n**Output:**\n```\n{output_text}\n```\n"
            self.append_content(out_md)
        self.pub("python", {
            "id": call_id or f"py_{int(time.time()*1000)}_{os.urandom(3).hex()}",
            "code": code,
            "output": output_text or "(no output)",
            "images": images,
        })
        self.append_text(f"\n```pyexec\n{json.dumps({'code': code, 'output': output_text, 'images': images})}\n```\n")
