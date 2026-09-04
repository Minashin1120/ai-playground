def _serialize_message_attachment_for_pdf(raw_ref):
    source = "unknown"
    ref = raw_ref
    if isinstance(raw_ref, dict):
        source = _normalize_attachment_source(raw_ref.get("source"))
        ref = raw_ref.get("filepath") or raw_ref.get("path") or raw_ref.get("url") or raw_ref.get("file") or ""
    norm = _normalize_upload_ref(ref)
    if not norm:
        return None
    filename = os.path.basename(norm)
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    is_image = ext in {e.lstrip(".") for e in _IMAGE_THUMB_EXTS}
    preview_endpoint = 'serve_file_thumb' if is_image else 'serve_file'
    return {
        "path": norm,
        "filename": filename,
        "source": source,
        "is_image": is_image,
        "url": url_for('serve_file', filename=norm),
        "preview_url": url_for(preview_endpoint, filename=norm)
    }

def _build_thread_pdf_payload(thread, leaf_id=None):
    messages = Message.query.filter_by(thread_id=thread.id).order_by(Message.timestamp, Message.id).all()
    if not messages:
        return {
            "thread": {
                "id": thread.id,
                "public_id": thread.public_id,
                "title": thread.title or "AI Chat"
            },
            "messages": [],
            "leaf_id": None,
            "generated_at": datetime.utcnow().isoformat() + "Z"
        }

    msg_map = {m.id: m for m in messages}
    leaf = msg_map.get(leaf_id) if leaf_id else None
    if leaf is None:
        leaf = messages[-1]

    path = []
    seen = set()
    curr = leaf
    while curr and curr.id not in seen:
        seen.add(curr.id)
        path.append(curr)
        parent_id = curr.parent_id
        curr = msg_map.get(parent_id) if parent_id else None
    path.reverse()

    serialized = []
    for m in path:
        content = decrypt_val(m.content) if m.is_encrypted else m.content
        thought_raw = decrypt_val(m.thought_data) if (m.is_encrypted and m.thought_data) else m.thought_data
        thought_text = extract_reasoning_text(thought_raw)
        token_in = None
        token_out = None
        token_total = None
        tokens_content = None
        tokens_thought = None
        legacy_token_total = None
        legacy_token_in = None
        legacy_token_out = None
        if (m.tokens_in and m.tokens_in > 0) or (m.tokens_out and m.tokens_out > 0):
            token_in = m.tokens_in if m.tokens_in and m.tokens_in > 0 else None
            token_out = m.tokens_out if m.tokens_out and m.tokens_out > 0 else None
            token_total = sum_token_counts(token_in, token_out)
            stored_tokens_thought = getattr(m, 'tokens_thought', None)
            if stored_tokens_thought is not None and stored_tokens_thought > 0:
                tokens_thought = stored_tokens_thought
        elif m.tokens is not None and m.tokens > 0 and (should_count_tokens_for_display(m.model) or not m.model):
            if m.role == 'user':
                legacy_token_in = m.tokens
            else:
                legacy_token_out = m.tokens
            legacy_token_total = m.tokens
        if token_total is None and should_count_tokens_for_display(m.model):
            details = build_message_token_details(m.role, content, thought_text, m.model, token_in, token_out)
            token_in = details["tokens_in"] if details["tokens_in"] is not None else token_in
            token_out = details["tokens_out"] if details["tokens_out"] is not None else token_out
            token_total = details["tokens_total"] if details["tokens_total"] is not None else token_total
            tokens_content = details["tokens_content"]
            tokens_thought = details["tokens_thought"]
        if token_total is None and legacy_token_total is not None:
            token_in = token_in if token_in is not None else legacy_token_in
            token_out = token_out if token_out is not None else legacy_token_out
            token_total = legacy_token_total

        attachments = []
        for raw_ref in _iter_message_attachment_refs(m.image_url):
            item = _serialize_message_attachment_for_pdf(raw_ref)
            if item:
                attachments.append(item)

        serialized.append({
            "id": m.id,
            "role": m.role,
            "content": content,
            "image_url": m.image_url,
            "attachments": attachments,
            "model": m.model,
            "thought_data": thought_raw,
            "thought_text": thought_text,
            "tokens": token_total,
            "tokens_in": token_in,
            "tokens_out": token_out,
            "tokens_content": tokens_content,
            "tokens_thought": tokens_thought,
            "is_encrypted": bool(m.is_encrypted),
            "quote_text": m.quote_text,
            "parent_id": m.parent_id,
            "timestamp": m.timestamp.isoformat() if m.timestamp else None
        })

    return {
        "thread": {
            "id": thread.id,
            "public_id": thread.public_id,
            "title": thread.title or "AI Chat",
            "last_model": thread.last_model,
            "custom_instruction": thread.custom_instruction,
            "include_global_instruction": thread.include_global_instruction if thread.include_global_instruction is not None else True
        },
        "messages": serialized,
        "leaf_id": leaf.id,
        "generated_at": datetime.utcnow().isoformat() + "Z"
    }

_RICH_PASTE_PDF_FONT_STATE = {
    "ready": False,
    "base": "Helvetica",
    "base_bold": "Helvetica-Bold",
    "mono": "Courier",
}


def _ensure_rich_paste_pdf_fonts():
    if _RICH_PASTE_PDF_FONT_STATE.get("ready"):
        return _RICH_PASTE_PDF_FONT_STATE
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except Exception:
        _RICH_PASTE_PDF_FONT_STATE["ready"] = True
        return _RICH_PASTE_PDF_FONT_STATE

    font_candidates = [
        ("IPAGothic", "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"),
        ("IPAPGothic", "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf"),
        ("NotoSansMono", "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf"),
    ]
    for font_name, font_path in font_candidates:
        if not os.path.exists(font_path):
            continue
        try:
            pdfmetrics.registerFont(TTFont(font_name, font_path))
        except Exception:
            continue

    if "IPAGothic" in getattr(pdfmetrics, "_fonts", {}):
        _RICH_PASTE_PDF_FONT_STATE["base"] = "IPAGothic"
    if "IPAPGothic" in getattr(pdfmetrics, "_fonts", {}):
        _RICH_PASTE_PDF_FONT_STATE["base_bold"] = "IPAPGothic"
    elif _RICH_PASTE_PDF_FONT_STATE["base"] == "IPAGothic":
        _RICH_PASTE_PDF_FONT_STATE["base_bold"] = "IPAGothic"
    if "NotoSansMono" in getattr(pdfmetrics, "_fonts", {}):
        _RICH_PASTE_PDF_FONT_STATE["mono"] = "NotoSansMono"
    _RICH_PASTE_PDF_FONT_STATE["ready"] = True
    return _RICH_PASTE_PDF_FONT_STATE


def _css_color_to_hex(value):
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    lower = raw.lower()
    if lower in {"inherit", "initial", "unset", "transparent", "none", "currentcolor"}:
        return None
    if lower.startswith("#"):
        if len(lower) == 4:
            return "#" + "".join(ch * 2 for ch in lower[1:])
        if len(lower) >= 7:
            return lower[:7]
    m = re.match(r"rgba?\(([^)]+)\)", lower)
    if m:
        parts = [p.strip() for p in m.group(1).split(",")]
        if len(parts) >= 3:
            nums = []
            for part in parts[:3]:
                if part.endswith("%"):
                    try:
                        num = int(round(float(part[:-1]) * 2.55))
                    except Exception:
                        return None
                else:
                    try:
                        num = int(float(part))
                    except Exception:
                        return None
                nums.append(max(0, min(255, num)))
            return "#%02x%02x%02x" % tuple(nums)
    return None


def _parse_inline_style(style_text):
    styles = {}
    if not style_text:
        return styles
    for decl in str(style_text).split(";"):
        if ":" not in decl:
            continue
        prop, value = decl.split(":", 1)
        prop = prop.strip().lower()
        value = value.strip()
        if prop in {"color", "background-color", "font-weight", "font-style", "text-decoration"}:
            styles[prop] = value
    return styles


def _rich_paste_pdf_filename(title):
    slug = re.sub(r"[^0-9A-Za-z\u3040-\u30ff\u4e00-\u9fff]+", "_", str(title or "").strip()).strip("_")
    if not slug:
        slug = "clipboard_rich"
    if len(slug) > 48:
        slug = slug[:48].rstrip("_")
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{slug}_{ts}.pdf"


def _build_rich_paste_pdf_bytes_reportlab(title, content_html, created_at=None):
    from io import BytesIO
    from bs4 import BeautifulSoup, NavigableString, Tag
    from reportlab import rl_config
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import HRFlowable, Image, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle, XPreformatted

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 5000))

    rl_config.defaultPageSize = A4
    fonts = _ensure_rich_paste_pdf_fonts()
    base_font = fonts.get("base", "Helvetica")
    bold_font = fonts.get("base_bold", "Helvetica-Bold")
    mono_font = fonts.get("mono", "Courier")

    def esc(value):
        return html.escape("" if value is None else str(value), quote=False)

    def normalize_text(value, preserve_newlines=False):
        txt = "" if value is None else str(value)
        txt = txt.replace("\u00a0", " ")
        txt = txt.replace("\r\n", "\n").replace("\r", "\n")
        if preserve_newlines:
            return txt.strip("\n")
        txt = re.sub(r"[ \t\f\v]+", " ", txt)
        txt = re.sub(r"\n[ \t]+", "\n", txt)
        txt = re.sub(r"[ \t]+\n", "\n", txt)
        txt = re.sub(r"\n{3,}", "\n\n", txt)
        return txt.strip()

    def get_attr_style(tag):
        return _parse_inline_style(tag.get("style"))

    def apply_inline_styles(text, tag=None):
        markup = text
        style = get_attr_style(tag) if tag is not None and hasattr(tag, "get") else {}
        if tag is not None and getattr(tag, "name", "").lower() == "code":
            markup = f'<font face="{mono_font}" backColor="#eeeeee">{markup}</font>'
        if "font-weight" in style and "bold" in style["font-weight"].lower():
            markup = f"<b>{markup}</b>"
        if "font-style" in style and "italic" in style["font-style"].lower():
            markup = f"<i>{markup}</i>"
        if "text-decoration" in style and "underline" in style["text-decoration"].lower():
            markup = f"<u>{markup}</u>"
        color_hex = _css_color_to_hex(style.get("color"))
        if color_hex:
            markup = f'<font color="{color_hex}">{markup}</font>'
        back_hex = _css_color_to_hex(style.get("background-color"))
        if back_hex:
            markup = f'<font backColor="{back_hex}">{markup}</font>'
        return markup

    def inline_markup(node):
        if node is None:
            return ""
        if isinstance(node, NavigableString):
            if type(node).__name__ in ('Doctype', 'Comment', 'Declaration', 'CData', 'ProcessingInstruction'):
                return ""
            return esc(str(node)).replace("\n", "<br/>")
        if not isinstance(node, Tag):
            return ""
        tag_name = (node.name or "").lower()
        if tag_name in {"script", "style", "noscript", "meta", "link", "head", "title", "base", "canvas", "svg", "object", "embed"}:
            return ""
        if tag_name == "br":
            return "<br/>"
        rendered = "".join(inline_markup(child) for child in node.children)
        rendered = rendered or ""
        if not rendered:
            return ""
        if tag_name in {"strong", "b"}:
            rendered = f"<b>{rendered}</b>"
        elif tag_name in {"em", "i"}:
            rendered = f"<i>{rendered}</i>"
        elif tag_name in {"u"}:
            rendered = f"<u>{rendered}</u>"
        elif tag_name in {"s", "strike", "del"}:
            rendered = f"<strike>{rendered}</strike>"
        elif tag_name == "code":
            rendered = f'<font face="{mono_font}" backColor="#eeeeee">{rendered}</font>'
        elif tag_name == "a":
            href = str(node.get("href") or "").strip()
            parsed_href = urlparse(href) if href else None
            if href and parsed_href and parsed_href.scheme.lower() in {"http", "https", "mailto"}:
                rendered = f'<a href="{esc(href)}">{rendered}</a>'
        rendered = apply_inline_styles(rendered, node)
        return rendered

    def paragraph_style(name, font_size=10.5, leading=15, bold=False, italic=False, color="#111827", space_after=6, left_indent=0, first_line_indent=0):
        return ParagraphStyle(
            name,
            parent=styles["BodyText"],
            fontName=bold_font if bold else base_font,
            fontSize=font_size,
            leading=leading,
            textColor=colors.HexColor(color),
            alignment=TA_LEFT,
            spaceAfter=space_after,
            leftIndent=left_indent,
            firstLineIndent=first_line_indent,
            wordWrap="CJK",
            splitLongWords=1,
        )

    styles = getSampleStyleSheet()
    title_style = paragraph_style("RichPasteTitle", font_size=18, leading=23, bold=True, color="#0f172a", space_after=10)
    meta_style = paragraph_style("RichPasteMeta", font_size=9, leading=12, color="#64748b", space_after=12)
    body_style = paragraph_style("RichPasteBody", font_size=10.5, leading=15, color="#111827", space_after=7)
    heading_styles = {
        1: paragraph_style("RichPasteH1", font_size=16, leading=20, bold=True, color="#0f172a", space_after=8),
        2: paragraph_style("RichPasteH2", font_size=14, leading=18, bold=True, color="#0f172a", space_after=8),
        3: paragraph_style("RichPasteH3", font_size=12.5, leading=16, bold=True, color="#0f172a", space_after=7),
        4: paragraph_style("RichPasteH4", font_size=11.5, leading=15, bold=True, color="#0f172a", space_after=6),
        5: paragraph_style("RichPasteH5", font_size=10.8, leading=14, bold=True, color="#0f172a", space_after=6),
        6: paragraph_style("RichPasteH6", font_size=10.5, leading=14, bold=True, color="#0f172a", space_after=6),
    }
    quote_style = paragraph_style("RichPasteQuote", font_size=10.2, leading=15, color="#334155", space_after=0, left_indent=4)
    code_style = ParagraphStyle(
        "RichPasteCode",
        parent=styles["Code"],
        fontName=mono_font,
        fontSize=9.2,
        leading=12.2,
        textColor=colors.HexColor("#111827"),
        alignment=TA_LEFT,
        spaceAfter=10,
        spaceBefore=10,
        leftIndent=0,
        rightIndent=0,
        wordWrap="CJK",
        splitLongWords=1,
        backColor=colors.HexColor("#f8fafc"),
        borderColor=colors.HexColor("#cbd5e1"),
        borderWidth=0.5,
        borderPadding=6,
        borderRadius=2,
    )
    list_style = paragraph_style("RichPasteList", font_size=10.5, leading=15, color="#111827", space_after=3)
    table_cell_style = paragraph_style("RichPasteTableCell", font_size=9.3, leading=12.5, color="#111827", space_after=0)
    note_style = paragraph_style("RichPasteNote", font_size=9.2, leading=12.2, color="#64748b", space_after=4)

    story = []
    doc_buffer = BytesIO()
    
    # Ensure title and created_at are safe for ReportLab
    safe_title = normalize_text(title) or "Clipboard Export"
    safe_created_at = normalize_text(created_at) or datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

    doc = SimpleDocTemplate(
        doc_buffer,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=str(safe_title),
        author="AI Playground",
    )
    available_width = A4[0] - doc.leftMargin - doc.rightMargin

    def add_paragraph(text, style):
        clean = normalize_text(text)
        if not clean:
            return
        try:
            story.append(Paragraph(clean, style))
        except Exception:
            try:
                # Try escaping the whole thing if markup was invalid
                story.append(Paragraph(esc(clean), style))
            except Exception:
                # Last resort: just add as plain text if still failing
                pass

    def add_blockquote(node):
        text = normalize_text(node.get_text(" ", strip=True))
        if not text:
            return
        try:
            para = Paragraph(esc(text), quote_style)
            # If the blockquote is very long, don't use Table as it may fail page splitting
            if len(text) > 1200:
                story.append(para)
                return
            box = Table([[para]], colWidths=[available_width])
            box.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff9eb")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LINEBEFORE", (0, 0), (0, -1), 4, colors.HexColor("#f59e0b")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#fde68a")),
            ]))
            story.append(box)
        except Exception:
            add_paragraph(esc(text), quote_style)

    def add_hr():
        story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#cbd5e1"), spaceBefore=6, spaceAfter=6))

    def add_image(node):
        src = str(node.get("src") or "").strip()
        if not src:
            return
        image_bytes = None
        try:
            if src.startswith("data:"):
                header, encoded = src.split(",", 1)
                if header.lower().startswith("data:image/") and ";base64" in header.lower() and len(encoded) <= 14_000_000:
                    image_bytes = base64.b64decode(encoded, validate=True)
            elif src.startswith("/files/") and current_user.is_authenticated:
                rel_path = _resolve_user_upload_rel_path(src, current_user.id)
                info = _get_file_disk_info(rel_path) if rel_path else None
                if info and info.get("exists") and int(info.get("size") or 0) <= 10 * 1024 * 1024:
                    image_bytes = _load_user_file_bytes(rel_path, info)
        except Exception:
            image_bytes = None
        if image_bytes and len(image_bytes) > 10 * 1024 * 1024:
            image_bytes = None
        if not image_bytes:
            alt = normalize_text(node.get("alt") or "image")
            if alt:
                story.append(Paragraph(f"[Image: {esc(alt)}]", note_style))
            return
        try:
            image = Image(BytesIO(image_bytes))
            image.hAlign = "CENTER"
            max_width = available_width
            max_height = 150 * mm
            try:
                image._restrictSize(max_width, max_height)
            except Exception:
                pass
            story.append(image)
            caption = normalize_text(node.get("alt") or "")
            if caption:
                story.append(Paragraph(esc(caption), note_style))
        except Exception:
            alt = normalize_text(node.get("alt") or "image")
            if alt:
                story.append(Paragraph(f"[Image: {esc(alt)}]", note_style))

    def list_item_children_text(li_node):
        inline_parts = []
        nested_lists = []
        for child in li_node.children:
            if isinstance(child, Tag) and (child.name or "").lower() in {"ul", "ol"}:
                nested_lists.append(child)
                continue
            inline_parts.append(inline_markup(child))
        return "".join(inline_parts), nested_lists

    def add_list(list_node, level=0, ordered=False):
        items = list_node.find_all("li", recursive=False)
        for idx, li in enumerate(items, start=1):
            item_markup, nested_lists = list_item_children_text(li)
            # Remove any trailing <br/> or whitespace before normalization
            item_markup = re.sub(r'(<br\s*/?>\s*)+$', '', item_markup.strip())
            item_text = normalize_text(item_markup, preserve_newlines=True)
            bullet = f"{idx}." if ordered else "-"
            item_style = paragraph_style(
                f"RichPasteList{level}_{idx}",
                font_size=10.3,
                leading=14.8,
                color="#111827",
                space_after=3,
                left_indent=max(0, (level + 1) * 12),
                first_line_indent=-12,
            )
            if item_text:
                try:
                    story.append(Paragraph(item_text, item_style, bulletText=bullet))
                except Exception:
                    story.append(Paragraph(esc(item_text), item_style, bulletText=bullet))
            elif not nested_lists:
                # Add empty item if no text and no nested lists
                story.append(Paragraph("&nbsp;", item_style, bulletText=bullet))

            for nested in nested_lists:
                add_list(nested, level=level + 1, ordered=(nested.name or "").lower() == "ol")

    def add_table(table_node):
        rows = []
        header_rows = 0
        tbody = table_node.find("tbody")
        tr_nodes = tbody.find_all("tr", recursive=False) if tbody else table_node.find_all("tr", recursive=False)
        if not tr_nodes:
            tr_nodes = table_node.find_all("tr")
        for row_index, tr in enumerate(tr_nodes):
            cells = []
            cell_nodes = tr.find_all(["th", "td"], recursive=False)
            if not cell_nodes:
                continue
            if any((cell.name or "").lower() == "th" for cell in cell_nodes) and header_rows == 0:
                header_rows = 1
            for cell in cell_nodes:
                cell_markup = normalize_text(inline_markup(cell), preserve_newlines=True)
                if not cell_markup:
                    cell_markup = "&nbsp;"
                cell_style = ParagraphStyle(
                    f"RichPasteTableCell{row_index}_{len(cells)}",
                    parent=table_cell_style,
                    wordWrap="CJK",
                    splitLongWords=1,
                )
                try:
                    p = Paragraph(cell_markup, cell_style)
                except Exception:
                    p = Paragraph(esc(cell_markup), cell_style)
                cells.append(p)
            if cells:
                rows.append(cells)
        if not rows:
            return
        
        col_count = max(len(r) for r in rows)
        if col_count == 0:
            return
        for row in rows:
            while len(row) < col_count:
                row.append(Paragraph("&nbsp;", table_cell_style))
        
        # Split very large tables into chunks to avoid layout issues
        CHUNK_SIZE = 50
        for i in range(0, len(rows), CHUNK_SIZE):
            chunk = rows[i : i + CHUNK_SIZE]
            is_first_chunk = (i == 0)
            
            # If not first chunk, we might want to repeat header, but SimpleDocTemplate 
            # might handle it if repeatRows is set. However, splitting manual is safer for build.
            current_header_rows = header_rows if is_first_chunk else 0
            
            table = Table(chunk, repeatRows=current_header_rows, hAlign="LEFT", 
                          colWidths=[available_width / col_count] * col_count,
                          splitByRow=1)
            
            style_cmds = [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
            if current_header_rows:
                style_cmds.extend([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                    ("FONTNAME", (0, 0), (-1, 0), bold_font),
                ])
            table.setStyle(TableStyle(style_cmds))
            story.append(table)
            if i + CHUNK_SIZE < len(rows):
                story.append(Spacer(1, 4))

    def add_pre(node):
        text = normalize_text(node.get_text("\n"), preserve_newlines=True)
        if not text:
            return
        try:
            # XPreformatted inherits from Paragraph and supports splitting across pages.
            # It also interprets XML-like tags, so we MUST escape the content.
            story.append(XPreformatted(esc(text), code_style, dedent=0))
        except Exception:
            try:
                # Fallback to Paragraph with manual line breaks
                story.append(Paragraph(esc(text).replace("\n", "<br/>"), code_style))
            except Exception:
                pass

    def render_node(node):
        if node is None:
            return
        if isinstance(node, NavigableString):
            if type(node).__name__ in ('Doctype', 'Comment', 'Declaration', 'CData', 'ProcessingInstruction'):
                return
            text = normalize_text(str(node))
            if text:
                # NavigableString must be escaped before Paragraph
                add_paragraph(esc(text).replace("\n", "<br/>"), body_style)
            return
        if not isinstance(node, Tag):
            return
        tag_name = (node.name or "").lower()
        if tag_name in {"script", "style", "noscript", "meta", "link", "iframe", "canvas", "svg", "object", "embed", "head", "title"}:
            return
        if tag_name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(tag_name[1])
            add_paragraph(inline_markup(node), heading_styles.get(level, body_style))
            return
        if tag_name == "p":
            add_paragraph(inline_markup(node), body_style)
            return
        if tag_name in {"div", "section", "article", "main", "header", "footer", "aside", "body", "html"}:
            # Optimization: if it contains only inline elements, process as one paragraph
            inline_only = True
            for child in node.children:
                if isinstance(child, Tag):
                    child_tag = (child.name or "").lower()
                    if child_tag not in {"span", "b", "strong", "i", "em", "u", "s", "a", "code", "font", "br"}:
                        inline_only = False
                        break
            if inline_only:
                markup = inline_markup(node)
                if markup:
                    add_paragraph(markup, body_style)
                return
            for child in node.children:
                render_node(child)
            return
        if tag_name == "blockquote":
            add_blockquote(node)
            return
        if tag_name == "pre":
            add_pre(node)
            return
        if tag_name == "hr":
            add_hr()
            return
        if tag_name == "br":
            story.append(Spacer(1, 3))
            return
        if tag_name == "img":
            add_image(node)
            return
        if tag_name == "figure":
            img = node.find("img")
            if img:
                add_image(img)
            caption = node.find("figcaption")
            if caption:
                add_paragraph(inline_markup(caption), note_style)
            return
        if tag_name in {"ul", "ol"}:
            add_list(node, level=0, ordered=(tag_name == "ol"))
            return
        if tag_name == "table":
            add_table(node)
            return
        text_markup = inline_markup(node)
        if text_markup:
            add_paragraph(text_markup, body_style)

    soup = BeautifulSoup(content_html or "", "html.parser")
    container = soup.body or soup
    story.append(Paragraph(esc(str(safe_title)), title_style))
    story.append(Paragraph(f"Created at: {esc(safe_created_at)}", meta_style))
    story.append(Spacer(1, 6))

    try:
        for child in container.children:
            render_node(child)
    except Exception as e:
        logger.exception("Error during rich paste node rendering")
        story.append(Paragraph(f"[Content rendering partially failed: {esc(str(e))}]", note_style))

    if not any(isinstance(item, (Paragraph, Table, XPreformatted)) for item in story if not isinstance(item, Spacer)):
        story.append(Paragraph("内容がありません。", body_style))

    def draw_page(canvas, doc_obj):
        try:
            canvas.saveState()
            canvas.setStrokeColor(colors.HexColor("#dbe3ee"))
            canvas.setLineWidth(0.6)
            canvas.line(doc.leftMargin, doc.height + doc.topMargin + 2, A4[0] - doc.rightMargin, doc.height + doc.topMargin + 2)
            canvas.setFont(base_font, 8.5)
            canvas.setFillColor(colors.HexColor("#64748b"))
            canvas.drawRightString(A4[0] - doc.rightMargin, 10 * mm, f"Page {doc_obj.page}")
            canvas.restoreState()
        except Exception:
            pass

    try:
        doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    except Exception as e:
        logger.exception("ReportLab doc.build failed")
        # Fallback build with simple story if it failed due to complex layout
        doc_buffer.seek(0)
        doc_buffer.truncate()
        fallback_story = [Paragraph(f"PDF生成エラーが発生しました: {esc(str(e))}", body_style)]
        doc.build(fallback_story)

    return doc_buffer.getvalue()


_RICH_PASTE_ALLOWED_TAGS = frozenset({
    "a", "abbr", "address", "article", "b", "blockquote", "br", "caption", "cite", "code",
    "col", "colgroup", "dd", "del", "details", "div", "dl", "dt", "em", "figcaption",
    "figure", "h1", "h2", "h3", "h4", "h5", "h6", "hr", "i", "img", "kbd", "li", "main",
    "mark", "ol", "p", "pre", "q", "s", "samp", "section", "small", "span", "strong",
    "sub", "summary", "sup", "table", "tbody", "td", "th", "thead", "tfoot", "time", "tr",
    "u", "ul", "var",
})
_RICH_PASTE_ALLOWED_ATTRS = frozenset({
    "align", "alt", "cellpadding", "cellspacing", "colspan", "datetime", "dir", "headers",
    "height", "href", "lang", "open", "rel", "reversed", "rowspan", "scope", "src", "start",
    "style", "target", "title", "type", "value", "width",
})
_RICH_PASTE_SAFE_STYLE_PROPS = frozenset({
    "align-items", "align-self", "background", "background-color", "background-image", "border",
    "border-block-color", "border-block-style", "border-block-width", "border-bottom",
    "border-bottom-color", "border-bottom-left-radius", "border-bottom-right-radius",
    "border-bottom-style", "border-bottom-width", "border-collapse", "border-color",
    "border-image", "border-inline-color", "border-inline-style", "border-inline-width",
    "border-left", "border-left-color", "border-left-style", "border-left-width", "border-radius",
    "border-right", "border-right-color", "border-right-style", "border-right-width",
    "border-spacing", "border-style", "border-top", "border-top-color", "border-top-left-radius",
    "border-top-right-radius", "border-top-style", "border-top-width", "border-width",
    "box-shadow", "box-sizing", "break-after", "break-before", "break-inside", "clear",
    "clip-path", "color", "column-gap", "direction", "display", "flex", "flex-basis",
    "flex-direction", "flex-grow", "flex-shrink", "flex-wrap", "float", "font", "font-family",
    "font-feature-settings", "font-kerning", "font-language-override", "font-optical-sizing",
    "font-size", "font-size-adjust", "font-stretch", "font-style", "font-variant",
    "font-variant-caps", "font-variant-ligatures", "font-variation-settings", "font-weight", "gap",
    "grid", "grid-auto-columns", "grid-auto-flow", "grid-auto-rows", "grid-column",
    "grid-column-end", "grid-column-start", "grid-row", "grid-row-end", "grid-row-start",
    "grid-template", "grid-template-areas", "grid-template-columns", "grid-template-rows", "height",
    "hyphens", "justify-content", "justify-items", "justify-self", "letter-spacing", "line-break",
    "line-height", "list-style", "list-style-position", "list-style-type", "margin",
    "margin-block", "margin-block-end", "margin-block-start", "margin-bottom", "margin-inline",
    "margin-inline-end", "margin-inline-start", "margin-left", "margin-right", "margin-top",
    "max-height", "max-width", "min-height", "min-width", "object-fit", "object-position",
    "opacity", "order", "orphans", "outline", "outline-color", "outline-offset", "outline-style",
    "outline-width", "overflow", "overflow-wrap", "overflow-x", "overflow-y", "padding",
    "padding-block", "padding-block-end", "padding-block-start", "padding-bottom",
    "padding-inline", "padding-inline-end", "padding-inline-start", "padding-left",
    "padding-right", "padding-top", "page-break-after", "page-break-before",
    "page-break-inside", "row-gap", "table-layout", "text-align", "text-decoration",
    "text-decoration-color", "text-decoration-line", "text-decoration-style",
    "text-decoration-thickness", "text-indent", "text-overflow", "text-shadow", "text-transform",
    "text-underline-offset", "vertical-align", "visibility", "white-space", "widows", "width",
    "word-break", "word-spacing", "writing-mode", "-webkit-text-stroke",
    "-webkit-text-stroke-color", "-webkit-text-stroke-width",
})
_RICH_PASTE_DROP_TAGS = frozenset({
    "script", "style", "link", "meta", "noscript", "iframe", "canvas", "svg", "object",
    "embed", "base", "form", "input", "button", "select", "textarea",
})
_RICH_PASTE_UNSAFE_STYLE_RE = re.compile(
    r"url\s*\(|expression\s*\(|javascript\s*:|@import|behavior\s*:|-moz-binding|var\s*\(|env\s*\(",
    re.IGNORECASE,
)
_RICH_PASTE_DATA_IMAGE_RE = re.compile(
    r"^data:image/(?:png|jpe?g|gif|webp);base64,[a-z0-9+/=\s]+$",
    re.IGNORECASE,
)


def _sanitize_rich_paste_style(style_text):
    safe = []
    for declaration in str(style_text or "").split(";"):
        if ":" not in declaration:
            continue
        prop, value = declaration.split(":", 1)
        prop = prop.strip().lower()
        value = re.sub(r"\s*!important\s*$", "", value.strip(), flags=re.IGNORECASE)
        if prop not in _RICH_PASTE_SAFE_STYLE_PROPS or not value or len(value) > 1000:
            continue
        if _RICH_PASTE_UNSAFE_STYLE_RE.search(value):
            continue
        if any(ch in value for ch in "{}"):
            continue
        safe.append(f"{prop}: {value}")
    return "; ".join(safe)


def _sanitize_rich_paste_html(content_html):
    from bs4 import BeautifulSoup, Comment, Doctype, Tag

    soup = BeautifulSoup(str(content_html or ""), "html.parser")
    body_text_length = len(re.sub(r"\s+", " ", soup.get_text(" ", strip=True)))
    body_tag_count = len(soup.find_all(True))
    if body_text_length >= 1000 and body_tag_count >= 120:
        candidates = []
        candidates.extend(soup.find_all("article"))
        candidates.extend(soup.find_all("main"))
        candidates.extend(soup.select('[role="main"],[role="article"]'))
        if candidates:
            eligible = [
                node for node in candidates
                if len(re.sub(r"\s+", " ", node.get_text(" ", strip=True))) >= body_text_length * 0.65
            ]
            if eligible:
                best = min(
                    eligible,
                    key=lambda node: (0 if node.find("h1") else 1, len(node.find_all(True))),
                )
                soup = BeautifulSoup(str(best), "html.parser")
    all_tags = list(soup.find_all(True))
    if len(all_tags) > 25_000:
        raise ValueError("rich_paste_html_too_complex")

    for node in list(soup.contents):
        if isinstance(node, (Comment, Doctype)):
            node.extract()
    for node in list(soup.find_all(string=lambda text: isinstance(text, (Comment, Doctype)))):
        node.extract()

    for node in all_tags:
        if not isinstance(node, Tag) or node.parent is None:
            continue
        tag_name = str(node.name or "").lower()
        if tag_name in _RICH_PASTE_DROP_TAGS:
            node.decompose()
            continue
        if tag_name not in _RICH_PASTE_ALLOWED_TAGS:
            node.unwrap()
            continue

        clean_attrs = {}
        for raw_name, raw_value in list(node.attrs.items()):
            attr_name = str(raw_name or "").lower()
            if attr_name not in _RICH_PASTE_ALLOWED_ATTRS:
                continue
            if isinstance(raw_value, (list, tuple)):
                attr_value = " ".join(str(part) for part in raw_value)
            elif raw_value is None:
                attr_value = ""
            else:
                attr_value = str(raw_value)
            if len(attr_value) > 2_100_000:
                continue
            if attr_name == "style":
                attr_value = _sanitize_rich_paste_style(attr_value)
                if not attr_value:
                    continue
            elif attr_name == "href":
                href = attr_value.strip()
                parsed = urlparse(href)
                if parsed.scheme and parsed.scheme.lower() not in {"http", "https", "mailto"}:
                    continue
                attr_value = href
            elif attr_name == "src":
                src = attr_value.strip()
                if tag_name != "img":
                    continue
                if src.startswith("data:"):
                    if len(src) > 2_100_000 or not _RICH_PASTE_DATA_IMAGE_RE.fullmatch(src):
                        continue
                elif not src.startswith("/files/"):
                    continue
                attr_value = src
            elif attr_name in {"width", "height", "colspan", "rowspan", "start", "value"}:
                if not re.fullmatch(r"-?\d+(?:\.\d+)?%?", attr_value.strip()):
                    continue
            clean_attrs[attr_name] = attr_value
        node.attrs = clean_attrs

        if tag_name == "a" and node.get("target") == "_blank":
            node["rel"] = "noopener noreferrer"
        if tag_name == "img" and not node.get("src"):
            alt = str(node.get("alt") or node.get("title") or "image").strip()
            replacement = soup.new_tag("span")
            replacement["style"] = (
                "display: block; color: #64748b; font-style: italic; "
                "border: 1px dashed #cbd5e1; padding: 8px"
            )
            replacement.string = f"[Image: {alt[:300]}]"
            node.replace_with(replacement)

    return str(soup)


def _normalize_rich_paste_print_layout(content_html):
    """Flatten deeply nested screen layouts that can make paged layout non-terminating."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(str(content_html or ""), "html.parser")
    styled_nodes = list(soup.find_all(style=True))
    complex_layout_count = 0
    for node in styled_nodes:
        style = str(node.get("style") or "")
        if re.search(r"(?:^|;)\s*display\s*:\s*(?:inline-)?(?:flex|grid)\b", style, re.IGNORECASE):
            complex_layout_count += 1
    screen_layout_count = 0
    for node in styled_nodes:
        if str(node.name or "").lower() not in {"article", "div", "main", "section"}:
            continue
        style = str(node.get("style") or "")
        padding_values = re.findall(
            r"(?:^|;)\s*padding(?:-left|-right|-inline|-inline-start|-inline-end)?\s*:\s*([^;]+)",
            style,
            flags=re.IGNORECASE,
        )
        large_side_padding = any(
            abs(float(pixel_value)) >= 96
            for padding_value in padding_values
            for pixel_value in re.findall(r"(-?\d+(?:\.\d+)?)px", padding_value, flags=re.IGNORECASE)
        )
        oversized_width = any(
            abs(float(pixel_value)) > 720
            for pixel_value in re.findall(
                r"(?:^|;)\s*(?:width|min-width)\s*:\s*(-?\d+(?:\.\d+)?)px",
                style,
                flags=re.IGNORECASE,
            )
        )
        if large_side_padding or oversized_width:
            screen_layout_count += 1
    if len(soup.find_all(True)) <= 500 and complex_layout_count <= 24 and screen_layout_count == 0:
        return str(soup)

    layout_props = {
        "align-items", "align-self", "column-gap", "flex", "flex-basis", "flex-direction",
        "flex-grow", "flex-shrink", "flex-wrap", "gap", "grid", "grid-auto-columns",
        "grid-auto-flow", "grid-auto-rows", "grid-column", "grid-column-end",
        "grid-column-start", "grid-row", "grid-row-end", "grid-row-start", "grid-template",
        "grid-template-areas", "grid-template-columns", "grid-template-rows", "justify-content",
        "justify-items", "justify-self", "order", "row-gap",
    }
    block_width_tags = {"article", "div", "main", "section"}
    side_padding_props = {
        "padding", "padding-left", "padding-right", "padding-inline", "padding-inline-start", "padding-inline-end",
    }
    for node in styled_nodes:
        declarations = []
        for declaration in str(node.get("style") or "").split(";"):
            if ":" not in declaration:
                continue
            prop, value = declaration.split(":", 1)
            prop = prop.strip().lower()
            value = value.strip()
            if prop in layout_props:
                continue
            if prop in {"height", "max-height", "min-height", "overflow", "overflow-x", "overflow-y"}:
                continue
            if prop in {"width", "min-width"} and str(node.name or "").lower() in block_width_tags:
                continue
            if prop in side_padding_props and str(node.name or "").lower() in block_width_tags:
                pixel_values = [
                    abs(float(match))
                    for match in re.findall(r"(-?\d+(?:\.\d+)?)px", value, flags=re.IGNORECASE)
                ]
                if any(pixel_value >= 96 for pixel_value in pixel_values):
                    value = "0px"
            if prop == "display":
                display_value = value.lower()
                if display_value in {"flex", "grid"}:
                    value = "block"
                elif display_value in {"inline-flex", "inline-grid"}:
                    value = "inline-block"
            declarations.append(f"{prop}: {value}")
        if declarations:
            node["style"] = "; ".join(declarations)
        elif node.has_attr("style"):
            del node["style"]
    return str(soup)


def _parse_rich_paste_css_color(value):
    text_value = str(value or "").strip().lower()
    if not text_value:
        return None
    candidates = re.findall(
        r"(?:rgba?|hsla?|oklab|oklch)\([^)]*\)|#[0-9a-f]{3,8}\b|\b(?:black|white|transparent)\b",
        text_value,
        flags=re.IGNORECASE,
    )
    if not candidates:
        return None
    token = candidates[-1].strip().lower()
    named = {
        "black": (0.0, 0.0, 0.0, 1.0),
        "white": (255.0, 255.0, 255.0, 1.0),
        "transparent": (0.0, 0.0, 0.0, 0.0),
    }
    if token in named:
        return named[token]
    if token.startswith("#"):
        digits = token[1:]
        if len(digits) in {3, 4}:
            digits = "".join(char * 2 for char in digits)
        if len(digits) not in {6, 8}:
            return None
        try:
            red = int(digits[0:2], 16)
            green = int(digits[2:4], 16)
            blue = int(digits[4:6], 16)
            alpha = int(digits[6:8], 16) / 255 if len(digits) == 8 else 1.0
            return float(red), float(green), float(blue), float(alpha)
        except ValueError:
            return None

    function_name, function_body = token.split("(", 1)
    function_body = function_body.rsplit(")", 1)[0].strip()
    alpha = 1.0
    if "/" in function_body:
        function_body, raw_alpha = function_body.rsplit("/", 1)
        raw_alpha = raw_alpha.strip()
        try:
            alpha = float(raw_alpha[:-1]) / 100 if raw_alpha.endswith("%") else float(raw_alpha)
        except ValueError:
            return None
    alpha = max(0.0, min(1.0, alpha))

    def parse_rgb_channel(raw_channel):
        raw_channel = raw_channel.strip()
        if raw_channel.endswith("%"):
            return max(0.0, min(255.0, float(raw_channel[:-1]) * 2.55))
        return max(0.0, min(255.0, float(raw_channel)))

    if function_name in {"rgb", "rgba"}:
        parts = [part for part in re.split(r"\s*,\s*|\s+", function_body.strip()) if part]
        if function_name == "rgba" and len(parts) == 4 and "/" not in token:
            raw_alpha = parts.pop()
            try:
                alpha = float(raw_alpha[:-1]) / 100 if raw_alpha.endswith("%") else float(raw_alpha)
            except ValueError:
                return None
        if len(parts) != 3:
            return None
        try:
            red, green, blue = (parse_rgb_channel(part) for part in parts)
            return red, green, blue, max(0.0, min(1.0, alpha))
        except ValueError:
            return None

    if function_name in {"hsl", "hsla"}:
        parts = [part for part in re.split(r"\s*,\s*|\s+", function_body.strip()) if part]
        if function_name == "hsla" and len(parts) == 4 and "/" not in token:
            raw_alpha = parts.pop()
            try:
                alpha = float(raw_alpha[:-1]) / 100 if raw_alpha.endswith("%") else float(raw_alpha)
            except ValueError:
                return None
        if len(parts) != 3:
            return None
        try:
            hue = float(re.sub(r"(?:deg|rad|turn)$", "", parts[0]))
            if parts[0].endswith("rad"):
                hue = math.degrees(hue)
            elif parts[0].endswith("turn"):
                hue *= 360
            saturation = max(0.0, min(1.0, float(parts[1].rstrip("%")) / 100))
            lightness = max(0.0, min(1.0, float(parts[2].rstrip("%")) / 100))
        except ValueError:
            return None
        chroma = (1 - abs((2 * lightness) - 1)) * saturation
        segment = (hue % 360) / 60
        intermediate = chroma * (1 - abs((segment % 2) - 1))
        if segment < 1:
            red1, green1, blue1 = chroma, intermediate, 0
        elif segment < 2:
            red1, green1, blue1 = intermediate, chroma, 0
        elif segment < 3:
            red1, green1, blue1 = 0, chroma, intermediate
        elif segment < 4:
            red1, green1, blue1 = 0, intermediate, chroma
        elif segment < 5:
            red1, green1, blue1 = intermediate, 0, chroma
        else:
            red1, green1, blue1 = chroma, 0, intermediate
        match = lightness - (chroma / 2)
        return (
            (red1 + match) * 255,
            (green1 + match) * 255,
            (blue1 + match) * 255,
            max(0.0, min(1.0, alpha)),
        )

    if function_name in {"oklab", "oklch"}:
        parts = [part for part in re.split(r"\s+", function_body.strip()) if part]
        if len(parts) != 3:
            return None
        try:
            lightness = float(parts[0].rstrip("%"))
            if parts[0].endswith("%"):
                lightness /= 100
            if function_name == "oklch":
                chroma = float(parts[1])
                hue_text = parts[2]
                hue = float(re.sub(r"(?:deg|rad|turn)$", "", hue_text))
                if hue_text.endswith("rad"):
                    hue = math.degrees(hue)
                elif hue_text.endswith("turn"):
                    hue *= 360
                lab_a = chroma * math.cos(math.radians(hue))
                lab_b = chroma * math.sin(math.radians(hue))
            else:
                lab_a = float(parts[1])
                lab_b = float(parts[2])
        except ValueError:
            return None
        l_value = (lightness + (0.3963377774 * lab_a) + (0.2158037573 * lab_b)) ** 3
        m_value = (lightness - (0.1055613458 * lab_a) - (0.0638541728 * lab_b)) ** 3
        s_value = (lightness - (0.0894841775 * lab_a) - (1.291485548 * lab_b)) ** 3
        red_linear = (4.0767416621 * l_value) - (3.3077115913 * m_value) + (0.2309699292 * s_value)
        green_linear = (-1.2684380046 * l_value) + (2.6097574011 * m_value) - (0.3413193965 * s_value)
        blue_linear = (-0.0041960863 * l_value) - (0.7034186147 * m_value) + (1.707614701 * s_value)

        def linear_to_srgb(channel):
            converted = 12.92 * channel if channel <= 0.0031308 else (1.055 * (channel ** (1 / 2.4))) - 0.055
            return max(0.0, min(255.0, converted * 255))

        return (
            linear_to_srgb(red_linear),
            linear_to_srgb(green_linear),
            linear_to_srgb(blue_linear),
            max(0.0, min(1.0, alpha)),
        )
    return None


def _rich_paste_color_luminance(color):
    if not color:
        return 0.0

    def channel(value):
        normalized = max(0.0, min(255.0, float(value))) / 255
        return normalized / 12.92 if normalized <= 0.04045 else ((normalized + 0.055) / 1.055) ** 2.4

    return (0.2126 * channel(color[0])) + (0.7152 * channel(color[1])) + (0.0722 * channel(color[2]))


def _rich_paste_color_contrast(first, second):
    first_luminance = _rich_paste_color_luminance(first)
    second_luminance = _rich_paste_color_luminance(second)
    return (max(first_luminance, second_luminance) + 0.05) / (
        min(first_luminance, second_luminance) + 0.05
    )


def _rich_paste_color_css(color):
    return f"rgb({round(color[0])}, {round(color[1])}, {round(color[2])})"


def _resolve_rich_paste_theme(content_html, requested_theme=None):
    from bs4 import BeautifulSoup, NavigableString

    requested_background = None
    requested_foreground = None
    if isinstance(requested_theme, dict):
        requested_background = _parse_rich_paste_css_color(requested_theme.get("background"))
        requested_foreground = _parse_rich_paste_css_color(requested_theme.get("foreground"))
        if requested_background and requested_background[3] < 0.9:
            requested_background = None
        if requested_foreground and requested_foreground[3] < 0.5:
            requested_foreground = None

    soup = BeautifulSoup(str(content_html or ""), "html.parser")
    background_candidates = []
    foreground_weights = {}
    inherited_foreground_candidates = []
    foreground_total = 0
    for node in soup.find_all(style=True):
        declarations = {}
        for declaration in str(node.get("style") or "").split(";"):
            if ":" not in declaration:
                continue
            prop, value = declaration.split(":", 1)
            declarations[prop.strip().lower()] = value.strip()

        background_value = declarations.get("background-color") or declarations.get("background")
        background = _parse_rich_paste_css_color(background_value)
        if background and background[3] >= 0.72:
            subtree_weight = len(re.sub(r"\s+", " ", node.get_text(" ", strip=True)))
            background_candidates.append((max(1, subtree_weight), background))

        foreground = _parse_rich_paste_css_color(declarations.get("color"))
        if foreground and foreground[3] >= 0.5:
            subtree_weight = len(re.sub(r"\s+", " ", node.get_text(" ", strip=True)))
            inherited_foreground_candidates.append((max(1, subtree_weight), foreground))
            direct_text = " ".join(
                str(child)
                for child in node.children
                if isinstance(child, NavigableString)
            )
            direct_weight = len(re.sub(r"\s+", " ", direct_text).strip())
            if direct_weight:
                key = tuple(round(part, 3) for part in foreground[:3])
                current = foreground_weights.get(key, {"weight": 0, "color": foreground})
                current["weight"] += direct_weight
                foreground_weights[key] = current
                foreground_total += direct_weight

    foreground_entries = sorted(
        foreground_weights.values(),
        key=lambda entry: entry["weight"],
        reverse=True,
    )
    inherited_foreground_candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    dominant_foreground = (
        foreground_entries[0]["color"]
        if foreground_entries
        else (inherited_foreground_candidates[0][1] if inherited_foreground_candidates else None)
    )
    if foreground_entries:
        light_foreground_weight = sum(
            entry["weight"]
            for entry in foreground_entries
            if _rich_paste_color_luminance(entry["color"]) >= 0.6
        )
    elif inherited_foreground_candidates:
        foreground_total = inherited_foreground_candidates[0][0]
        light_foreground_weight = (
            foreground_total
            if _rich_paste_color_luminance(inherited_foreground_candidates[0][1]) >= 0.6
            else 0
        )
    else:
        light_foreground_weight = 0
    background_candidates.sort(key=lambda candidate: candidate[0], reverse=True)

    background = requested_background
    if not background and background_candidates:
        background = background_candidates[0][1]
    if not background:
        background = (
            (11.0, 11.0, 12.0, 1.0)
            if foreground_total and light_foreground_weight / foreground_total >= 0.55
            else (255.0, 255.0, 255.0, 1.0)
        )
    foreground = requested_foreground
    if not foreground:
        neutral_candidates = [
            (entry["weight"], entry["color"])
            for entry in foreground_entries
            if max(entry["color"][:3]) - min(entry["color"][:3]) <= 64
            and _rich_paste_color_contrast(background, entry["color"]) >= 3
        ]
        neutral_candidates.extend(
            (weight, color)
            for weight, color in inherited_foreground_candidates
            if max(color[:3]) - min(color[:3]) <= 64
            and _rich_paste_color_contrast(background, color) >= 3
        )
        neutral_candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        foreground = neutral_candidates[0][1] if neutral_candidates else dominant_foreground
    dark = _rich_paste_color_luminance(background) < 0.32
    if not foreground or _rich_paste_color_contrast(background, foreground) < 3:
        foreground = (244.0, 244.0, 245.0, 1.0) if dark else (17.0, 24.0, 39.0, 1.0)

    return {
        "mode": "dark" if dark else "light",
        "background": _rich_paste_color_css(background),
        "foreground": _rich_paste_color_css(foreground),
        "muted": "rgb(161, 161, 170)" if dark else "rgb(100, 116, 139)",
        "border": "rgb(63, 63, 70)" if dark else "rgb(203, 213, 225)",
        "surface": "rgb(33, 33, 33)" if dark else "rgb(248, 250, 252)",
        "quote": "rgb(39, 39, 42)" if dark else "rgb(255, 249, 235)",
        "link": "rgb(125, 211, 252)" if dark else "rgb(15, 118, 110)",
    }


def _build_rich_paste_pdf_bytes_weasyprint(title, content_html, created_at=None, theme=None):
    safe_title = html.escape(str(title or "Clipboard Export").strip() or "Clipboard Export")
    safe_created_at = html.escape(
        str(created_at or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")).strip()
    )
    resolved_theme = _resolve_rich_paste_theme(content_html, requested_theme=theme)
    theme_background = resolved_theme["background"]
    theme_foreground = resolved_theme["foreground"]
    theme_muted = resolved_theme["muted"]
    theme_border = resolved_theme["border"]
    theme_surface = resolved_theme["surface"]
    theme_quote = resolved_theme["quote"]
    theme_link = resolved_theme["link"]
    theme_mode = resolved_theme["mode"]
    document_html = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>{safe_title}</title>
<style>
@page {{
  size: A4;
  margin: 16mm 16mm 18mm;
  background: {theme_background};
  @bottom-right {{
    content: "Page " counter(page) " / " counter(pages);
    color: {theme_muted};
    font-size: 8.5pt;
  }}
}}
html {{ color-scheme: {theme_mode}; background: {theme_background}; }}
body {{
  margin: 0;
  background: {theme_background};
  color: {theme_foreground};
  font-family: "IPAPGothic", "IPAGothic", "Droid Sans Fallback", sans-serif;
  font-size: 10.5pt;
  line-height: 1.55;
  overflow-wrap: anywhere;
  word-break: normal;
}}
*, *::before, *::after {{ box-sizing: border-box; }}
.document-title {{
  margin: 0 0 3mm;
  padding: 0 0 3mm;
  border-bottom: 1.5pt solid {theme_border};
  color: {theme_foreground};
  font-size: 18pt;
  line-height: 1.25;
}}
.document-meta {{ margin: 0 0 7mm; color: {theme_muted}; font-size: 8.5pt; }}
.rich-content {{ color: {theme_foreground}; }}
.rich-content h1, .rich-content h2, .rich-content h3,
.rich-content h4, .rich-content h5, .rich-content h6 {{
  line-height: 1.3;
  break-after: avoid;
}}
.rich-content p {{ margin: 0 0 0.85em; }}
.rich-content img {{ display: block; max-width: 100%; height: auto; margin: 0.9em auto; }}
.rich-content table {{
  max-width: 100%;
  margin: 1em 0;
  border-collapse: collapse;
  break-inside: auto;
}}
.rich-content thead {{ display: table-header-group; }}
.rich-content tr {{ break-inside: avoid; }}
.rich-content th, .rich-content td {{
  border: 0.6pt solid {theme_border};
  padding: 5pt 6pt;
  vertical-align: top;
  overflow-wrap: anywhere;
}}
.rich-content th {{ background: {theme_surface}; color: {theme_foreground}; font-weight: 700; }}
.rich-content pre {{
  max-width: 100%;
  margin: 1em 0;
  padding: 9pt;
  border: 0.6pt solid {theme_border};
  border-radius: 3pt;
  background: {theme_surface};
  color: {theme_foreground};
  font-family: "WenQuanYi Zen Hei Mono", "Droid Sans Fallback", monospace;
  font-size: 8.8pt;
  line-height: 1.4;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}}
.rich-content code, .rich-content kbd, .rich-content samp {{
  padding: 0.08em 0.28em;
  border-radius: 2pt;
  background: {theme_surface};
  color: {theme_foreground};
  font-family: "WenQuanYi Zen Hei Mono", "Droid Sans Fallback", monospace;
}}
.rich-content pre code {{ padding: 0; background: transparent; }}
.rich-content blockquote {{
  margin: 1em 0;
  padding: 8pt 10pt;
  border-left: 3pt solid #f59e0b;
  background: {theme_quote};
  color: {theme_foreground};
}}
.rich-content a {{ color: {theme_link}; text-decoration: underline; }}
.rich-content hr {{ margin: 1em 0; border: 0; border-top: 0.7pt solid {theme_border}; }}
.rich-content figure {{ max-width: 100%; margin: 1em 0; }}
.rich-content figcaption {{ color: {theme_muted}; font-size: 9pt; text-align: center; }}
</style>
</head>
<body>
<h1 class="document-title">{safe_title}</h1>
<p class="document-meta">Created at: {safe_created_at}</p>
<main class="rich-content">{content_html}</main>
</body>
</html>"""
    weasyprint_bin = os.path.join(os.path.dirname(sys.executable), "weasyprint")
    if not os.path.isfile(weasyprint_bin):
        raise RuntimeError("weasyprint_command_not_found")
    completed = subprocess.run(
        [
            weasyprint_bin,
            "--quiet",
            "--presentational-hints",
            "--optimize-images",
            "--jpeg-quality", "92",
            "--dpi", "180",
            "--timeout", "3",
            "--allowed-protocols", "data",
            "--no-http-redirects",
            "-", "-",
        ],
        input=document_html.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=75,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"weasyprint_failed: {stderr[-1000:]}")
    pdf_bytes = completed.stdout
    if not isinstance(pdf_bytes, bytes) or not pdf_bytes.startswith(b"%PDF"):
        raise RuntimeError("weasyprint_invalid_pdf")
    return pdf_bytes


def _build_rich_paste_pdf_bytes(title, content_html, created_at=None, theme=None):
    safe_html = _sanitize_rich_paste_html(content_html)
    resolved_theme = _resolve_rich_paste_theme(safe_html, requested_theme=theme)
    print_html = _normalize_rich_paste_print_layout(safe_html)
    try:
        return _build_rich_paste_pdf_bytes_weasyprint(
            title,
            print_html,
            created_at=created_at,
            theme=resolved_theme,
        )
    except Exception:
        logger.exception("WeasyPrint rich paste rendering failed; using ReportLab fallback")
        return _build_rich_paste_pdf_bytes_reportlab(
            title,
            print_html,
            created_at=created_at,
        )


@app.route('/api/rich-paste/pdf', methods=['POST'])
@login_required
def rich_paste_pdf():
    if not getattr(current_user, "is_admin", False):
        return jsonify({'error': '403'}), 403
    if request.content_length and request.content_length > 2 * 1024 * 1024:
        return jsonify({'error': 'payload_too_large'}), 413
    
    log_force(
        "[DEBUG] rich_paste_pdf start "
        f"content_type={request.content_type} "
        f"content_length={request.content_length} "
        f"is_json={request.is_json}"
    )

    d = None
    if request.is_json:
        try:
            d = request.get_json(silent=True)
            if d:
                log_force(f"[DEBUG] rich_paste_pdf get_json success keys={list(d.keys())}")
        except Exception as e:
            log_force(f"[DEBUG] rich_paste_pdf get_json exception: {e}")

    if not isinstance(d, dict) or not d:
        if request.form:
            d = request.form.to_dict(flat=True)
            log_force("[DEBUG] rich_paste_pdf used request.form")
        else:
            try:
                # Crucial: Use cache=True to allow multiple reads if needed
                raw_body = request.get_data(cache=True, as_text=True)
                if raw_body and raw_body.strip():
                    log_force(f"[DEBUG] rich_paste_pdf raw_body_len={len(raw_body)}")
                    try:
                        d = json.loads(raw_body)
                        log_force("[DEBUG] rich_paste_pdf json.loads(raw_body) success")
                    except Exception:
                        # Fallback for some clients that might send raw HTML as body
                        if "<html>" in raw_body.lower() or "<div" in raw_body.lower():
                            d = {"html": raw_body}
                            log_force("[DEBUG] rich_paste_pdf treated raw_body as html")
            except Exception as e:
                log_force(f"[DEBUG] rich_paste_pdf get_data exception: {e}")
                d = {}

    if not isinstance(d, dict):
        d = {}

    content_html = str(d.get('html') or '').strip()
    title = (str(d.get('title') or 'Clipboard Export').strip() or 'Clipboard Export')[:200]
    created_at = str(d.get('created_at') or datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')).strip()[:100]
    requested_theme = d.get('theme') if isinstance(d.get('theme'), dict) else None

    log_force(
        "[DEBUG] rich_paste_pdf payload info "
        f"title_len={len(title)} "
        f"html_len={len(content_html)} "
        f"keys={sorted(list(d.keys())) if d else 'None'}"
    )

    if not content_html:
        log_force("[DEBUG] rich_paste_pdf error: missing_html")
        return jsonify({'error': 'missing_html'}), 400
    if len(content_html) > 2 * 1024 * 1024:
        return jsonify({'error': 'payload_too_large'}), 413

    log_force(f"[DEBUG] rich_paste_pdf starting _build_rich_paste_pdf_bytes for title={title}")
    try:
        pdf_bytes = _build_rich_paste_pdf_bytes(
            title,
            content_html,
            created_at=created_at,
            theme=requested_theme,
        )
        log_force(f"[DEBUG] rich_paste_pdf _build_rich_paste_pdf_bytes finished, size={len(pdf_bytes)}")
    except Exception as e:
        logger.exception("Server-side rich paste PDF generation failed")
        log_force(f"[DEBUG] rich_paste_pdf generation exception: {type(e).__name__}: {e}")
        return jsonify({'error': 'pdf_generation_failed', 'message': str(e)}), 500

    try:
        filename = _rich_paste_pdf_filename(title)
        resp = send_file(
            BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
        resp.headers['X-Rich-Paste-Filename'] = filename
        resp.headers['Cache-Control'] = 'no-store'
        log_force(f"[DEBUG] rich_paste_pdf success filename={filename} bytes={len(pdf_bytes)}")
        return resp
    except Exception as e:
        logger.exception("Server-side rich paste PDF response failed")
        log_force(f"[DEBUG] rich_paste_pdf response exception: {e}")
        return jsonify({'error': 'response_failed', 'message': str(e)}), 500


@app.route('/c/<thread_id>/pdf')
@login_required
def export_thread_pdf(thread_id):
    t = resolve_thread_for_user(thread_id, current_user.id)
    if not t:
        return jsonify({'error': '403'}), 403
    leaf_id = request.args.get('leaf_id', type=int)
    payload = _build_thread_pdf_payload(t, leaf_id=leaf_id)
    return jsonify(payload)

