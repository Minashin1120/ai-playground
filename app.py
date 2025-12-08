import os
import json
import time
import logging
import base64
import mimetypes
import ast
from datetime import datetime
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, redirect, url_for, make_response, flash, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sqlalchemy import or_, text
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types
import pypdf

# Try importing xAI SDK
try:
    from xai_sdk import Client as XAIClient
    from xai_sdk.chat import user as xai_user, file as xai_file, system as xai_system
except ImportError:
    XAIClient = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)

# --- Security Config (V201) ---
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = 86400
# ------------------------------

app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static/uploads')
app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- API Key Logic (V200) ---
def get_key(name):
    if current_user.is_authenticated:
        user_key_field = name.lower() # e.g. openai_api_key
        user_key = getattr(current_user, user_key_field, None)
        if user_key and user_key.strip():
            return user_key.strip()
        
        # Fallback to system key only for admin
        if current_user.username == 'minashin1120':
            sys_key = os.getenv(name)
            if sys_key and "placeholder" not in sys_key:
                return sys_key
    return None

# --- Security Headers ---
@app.after_request
def add_security_headers(response):
    csp = "default-src * 'unsafe-inline' 'unsafe-eval' data: blob:;"
    response.headers['Content-Security-Policy'] = csp
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)
    password_hash = db.Column(db.String(255))
    system_prompt = db.Column(db.Text, default="")
    openai_api_key = db.Column(db.Text, nullable=True)
    gemini_api_key = db.Column(db.Text, nullable=True)
    xai_api_key = db.Column(db.Text, nullable=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Thread(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), default="New Chat")
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)
    messages = db.relationship('Message', backref='thread', cascade="all, delete-orphan", lazy=True)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    thread_id = db.Column(db.Integer, db.ForeignKey('thread.id'), nullable=False)
    role = db.Column(db.String(20))
    content = db.Column(db.Text)
    model = db.Column(db.String(50))
    image_url = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    tokens = db.Column(db.Integer, default=0)
    thought_data = db.Column(db.Text)

@login_manager.user_loader
def load_user(uid): return User.query.get(int(uid))

@app.route('/')
@login_required
def index(): return render_template('chat.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    site_key = os.getenv('TURNSTILE_SITE_KEY')
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user, remember=remember)
            return redirect(url_for('index'))
        else:
            return render_template('login.html', site_key=site_key, error="Invalid username or password")
    return render_template('login.html', site_key=site_key)

@app.route('/register', methods=['POST'])
def register():
    username = request.form.get('username')
    password = request.form.get('password')
    if User.query.filter_by(username=username).first():
        return render_template('login.html', error="Username already exists", mode="register")
    new_user = User(username=username)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()
    login_user(new_user)
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    logout_user()
    resp = make_response(redirect(url_for('login')))
    resp.headers['Cache-Control'] = 'no-cache, no-store'
    return resp

@app.route('/api/settings', methods=['GET', 'POST'])
@login_required
def handle_settings():
    if request.method == 'GET':
        return jsonify({
            'system_prompt': current_user.system_prompt or "",
            'username': current_user.username,
            'openai_key': current_user.openai_api_key or "",
            'gemini_key': current_user.gemini_api_key or "",
            'xai_key': current_user.xai_api_key or ""
        })
    else:
        d = request.json
        # V204: Only admin can change system prompt globally? 
        # Actually V200 made system_prompt per user. So any user can change their OWN system prompt.
        # But if the requirement is strict "Use console", we might limit UI.
        # Here we allow saving, UI will limit visibility if needed.
        
        if 'system_prompt' in d: current_user.system_prompt = d['system_prompt']
        if 'openai_key' in d: current_user.openai_api_key = d['openai_key']
        if 'gemini_key' in d: current_user.gemini_api_key = d['gemini_key']
        if 'xai_key' in d: current_user.xai_api_key = d['xai_key']
        
        if 'new_password' in d and d['new_password']:
            current_user.set_password(d['new_password'])
            
        if 'new_username' in d and d['new_username'] and d['new_username'] != current_user.username:
            if User.query.filter_by(username=d['new_username']).first():
                return jsonify({'error': 'Username taken'}), 400
            current_user.username = d['new_username']

        db.session.commit()
        return jsonify({'status': 'ok'})

@app.route('/api/threads')
@login_required
def get_threads():
    q = request.args.get('q', '').strip()
    query = Thread.query.filter_by(user_id=current_user.id)
    if q: query = query.join(Message).filter(or_(Thread.title.contains(q), Message.content.contains(q))).distinct()
    ts = query.order_by(Thread.updated_at.desc()).limit(50).all()
    return jsonify([{'id': t.id, 'title': t.title} for t in ts])

@app.route('/api/threads', methods=['POST'])
@login_required
def create_thread():
    try:
        t = Thread(user_id=current_user.id)
        db.session.add(t)
        db.session.commit()
        return jsonify({'id': t.id, 'title': t.title})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/threads/<int:tid>/title', methods=['PUT'])
@login_required
def update_title(tid):
    t = Thread.query.get_or_404(tid)
    if t.user_id != current_user.id: return jsonify({'error': '403'}), 403
    t.title = request.json.get('title', 'Untitled')
    db.session.commit()
    return jsonify({'status': 'ok'})

@app.route('/api/threads/<int:tid>')
@login_required
def get_msgs(tid):
    t = Thread.query.get_or_404(tid)
    if t.user_id != current_user.id: return jsonify({'error': '403'}), 403
    ms = Message.query.filter_by(thread_id=tid).order_by(Message.timestamp).all()
    return jsonify([{
        'id': m.id, 'role': m.role, 'content': m.content, 
        'image_url': m.image_url,
        'model': m.model,
        'thought_data': m.thought_data
    } for m in ms])

@app.route('/api/threads/<int:tid>', methods=['DELETE'])
@login_required
def delete_thread(tid):
    t = Thread.query.get_or_404(tid)
    if t.user_id != current_user.id: return jsonify({'error': '403'}), 403
    db.session.delete(t)
    db.session.commit()
    return jsonify({'status': 'deleted'})

@app.route('/api/messages/<int:mid>', methods=['DELETE'])
@login_required
def delete_message(mid):
    msg = Message.query.get_or_404(mid)
    t = Thread.query.get(msg.thread_id)
    if t.user_id != current_user.id: return jsonify({'error': '403'}), 403
    Message.query.filter(Message.thread_id == msg.thread_id, Message.timestamp >= msg.timestamp).delete()
    db.session.commit()
    return jsonify({'status': 'ok'})

@app.route('/api/files')
@login_required
def get_files_lib():
    # Only get files belonging to current user threads
    msgs = Message.query.join(Thread).filter(
        Thread.user_id == current_user.id, 
        Message.image_url != None
    ).order_by(Message.timestamp.desc()).all()
    
    files = []
    seen = set()
    for m in msgs:
        if m.image_url:
            try:
                img_list = json.loads(m.image_url)
                if not isinstance(img_list, list): img_list = [m.image_url]
            except: img_list = [m.image_url]
            
            for path_str in img_list:
                if path_str and path_str not in seen:
                    # V204: Check if file exists in user dir
                    full_path = os.path.join(app.config['UPLOAD_FOLDER'], path_str)
                    if os.path.exists(full_path):
                        seen.add(path_str)
                        ext = os.path.splitext(path_str)[1].lower().replace('.', '')
                        ftype = 'image' if ext in ['png', 'jpg', 'jpeg', 'gif', 'webp'] else 'file'
                        
                        # Use raw filename for display if preferred, or path
                        disp_name = os.path.basename(path_str)
                        
                        files.append({
                            'id': m.id, 
                            'filename': disp_name,
                            'filepath': path_str, # Relative path for deletion/download
                            'url': url_for('static', filename='uploads/' + path_str),
                            'type': ftype, 'ext': ext, 'date': m.timestamp.strftime('%Y-%m-%d %H:%M')
                        })
    return jsonify(files)

@app.route('/api/files/delete', methods=['POST'])
@login_required
def delete_files_batch():
    fnames = request.json.get('filenames', []) # These are now relative paths e.g. "1/abc.webp"
    deleted_count = 0
    for rel_path in fnames:
        # Security check: Ensure path starts with user_id to prevent deleting others' files
        expected_prefix = f"{current_user.id}/"
        if not rel_path.startswith(expected_prefix):
            continue
            
        fp = os.path.join(app.config['UPLOAD_FOLDER'], rel_path)
        if os.path.exists(fp):
            try: os.remove(fp)
            except: pass
        deleted_count += 1
    return jsonify({'status': 'ok', 'count': deleted_count})

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    files = request.files.getlist('file')
    if not files: return jsonify({'error': 'No file'}), 400
    
    # V204: User Specific Directory
    user_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
    if not os.path.exists(user_upload_dir):
        os.makedirs(user_upload_dir, exist_ok=True)
        os.chmod(user_upload_dir, 0o777)

    results = []
    for f in files:
        if not f.filename: continue
        try:
            orig_name = secure_filename(f.filename)
            ext = os.path.splitext(orig_name)[1].lower()
            if not ext: ext = ""
            
            # Generate unique filename
            fname_base = f"{int(time.time())}_{os.urandom(4).hex()}"
            fname = f"{fname_base}{ext}"
            
            save_path = os.path.join(user_upload_dir, fname)
            
            is_image = ext in ['.jpg', '.jpeg', '.png']
            if is_image:
                try:
                    img = Image.open(f).convert('RGB')
                    fname = f"{fname_base}.webp" # Change ext to webp
                    save_path = os.path.join(user_upload_dir, fname)
                    img.save(save_path, 'WEBP', quality=80)
                except:
                    f.seek(0)
                    f.save(save_path)
            else:
                f.save(save_path)
            
            # Store relative path in DB: "user_id/filename"
            db_path = f"{current_user.id}/{fname}"
            results.append(db_path)
            
        except Exception as e: logger.error(f"Upload Error: {e}")
    
    resp = {'filenames': results}
    # Return URL for preview
    if results:
        resp['filename'] = results[0] # This is path
        resp['url'] = url_for('static', filename='uploads/' + results[0])
    return jsonify(resp)

@app.route('/chat_stream', methods=['POST'])
@login_required
def chat_stream():
    d = request.json
    tid = d.get('thread_id')
    model_key = d.get('model')
    msg = d.get('message')
    img_list = d.get('image_urls', []); 
    if not img_list and d.get('image_url'): img_list = [d.get('image_url')]
    img_list = [x for x in img_list if x and x != '[]' and x != 'null']
    enable_search = d.get('enable_search', False)
    enable_thinking = d.get('enable_thinking', False)
    reasoning_effort = d.get('reasoning_effort', 'medium')
    enable_sys_prompt = d.get('enable_system_prompt', False)
    is_gemini = 'gemini' in model_key or 'nano' in model_key
    is_grok = 'grok' in model_key

    req_key = None
    if is_gemini: req_key = get_key('GEMINI_API_KEY')
    elif is_grok: req_key = get_key('XAI_API_KEY')
    else: req_key = get_key('OPENAI_API_KEY')

    if not req_key:
        def err_gen(): yield json.dumps({"type": "error", "data": "API Key is missing in Settings."}) + "\n"
        return Response(stream_with_context(err_gen()), mimetype='application/json')
    
    local_gemini = genai.Client(api_key=req_key, http_options={'api_version': 'v1alpha'}) if is_gemini else None
    local_openai = OpenAI(api_key=req_key) if not is_gemini and not is_grok else None
    local_xai = OpenAI(api_key=req_key, base_url="https://api.x.ai/v1") if is_grok else None

    loaded_files = []
    for rel_path in img_list:
        info = {'name': os.path.basename(rel_path), 'text': None, 'bytes': None, 'mime': None, 'path': None}
        try:
            # V204: Path is relative "uid/file"
            path = os.path.join(app.config['UPLOAD_FOLDER'], rel_path)
            info['path'] = path
            if os.path.exists(path):
                mime, _ = mimetypes.guess_type(path)
                info['mime'] = mime
                fname = os.path.basename(rel_path)
                is_pdf = fname.lower().endswith('.pdf'); is_img = False
                if mime and (mime.startswith('image/') or mime.startswith('video/')): is_img = True
                elif fname.endswith(('.webp','.png','.jpg','.jpeg','.gif','.mp4')): is_img = True
                if is_pdf: info['mime'] = 'application/pdf'
                if not is_img and not is_pdf:
                    try:
                        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                            info['text'] = f.read(); info['mime'] = 'text/plain'
                    except: pass
                if not info['text']:
                    with open(path, 'rb') as f:
                        info['bytes'] = f.read()
                        if not info['mime']: info['mime'] = 'application/octet-stream'
            loaded_files.append(info)
        except Exception as e: logger.error(f"Load File Error: {e}")

    display_content = msg
    for fi in loaded_files:
        if fi['text']: display_content += f"\n\n[File: {fi['name']}]\n```\n{fi['text'][:200]}...\n```"
        elif fi['mime'] == 'application/pdf': display_content += f"\n\n[PDF: {fi['name']}]"
        else: display_content += f"\n\n[Attached: {fi['name']}]"

    db_image_val = json.dumps(img_list) if img_list else None
    db.session.add(Message(thread_id=tid, role='user', content=display_content, model=model_key, image_url=db_image_val))
    t = Thread.query.get(tid)
    if t and Message.query.filter_by(thread_id=tid).count() <= 1: t.title = msg[:30]
    t.updated_at = datetime.utcnow()
    db.session.commit()

    cached_sys_prompt = ""
    if enable_sys_prompt and current_user.is_authenticated:
        cached_sys_prompt = current_user.system_prompt or ""

    def gen():
        full_res, generated_images = "", []
        collected_signatures = {} 
        thought_accumulated = ""
        all_msgs = Message.query.filter_by(thread_id=tid).order_by(Message.timestamp).all()
        history = all_msgs[:-1] if len(all_msgs) > 0 else []
        sys_instr = cached_sys_prompt if cached_sys_prompt else None

        if is_gemini:
            real_model = model_key
            if "nano-banana-pro" in model_key: real_model = "gemini-3-pro-image-preview"
            elif "nano-banana" in model_key: real_model = "gemini-2.5-flash-image"
            elif "3.0" in model_key: real_model = "gemini-3-pro-preview"
            elif "2.5" in model_key: real_model = "gemini-2.5-flash"
            config_params = {'temperature': 0.7}
            if "nano" not in model_key:
                if enable_thinking:
                    if "3.0" in model_key: config_params['thinking_config'] = types.ThinkingConfig(include_thoughts=True, thinking_level="high")
                    elif "2.5" in model_key: config_params['thinking_config'] = types.ThinkingConfig(include_thoughts=True, thinking_budget=1024)
                if enable_search: config_params['tools'] = [types.Tool(google_search=types.GoogleSearch())]
                if sys_instr: config_params['system_instruction'] = sys_instr
            else: config_params['tools'] = None
            config = types.GenerateContentConfig(**config_params)
            contents = []
            for m in history:
                role = 'model' if m.role == 'assistant' else 'user'
                parts = []
                thought_text = None; thought_sig = None
                if m.role == 'assistant' and m.thought_data:
                    try:
                        td = json.loads(m.thought_data)
                        if isinstance(td, dict):
                            thought_text = td.get('text'); sigs = td.get('signatures')
                            if isinstance(sigs, dict):
                                thought_sig = sigs.get('signature')
                                if thought_sig and isinstance(thought_sig, str):
                                    try: thought_sig = base64.b64decode(thought_sig)
                                    except: pass
                    except: pass
                if thought_text and thought_sig:
                    parts.append(types.Part(text=m.content, thought=thought_text, thought_signature=thought_sig))
                else: parts.append(types.Part(text=m.content))
                if m.image_url:
                    try:
                        h_imgs = json.loads(m.image_url)
                        if not isinstance(h_imgs, list): h_imgs = [m.image_url]
                        for rel_path in h_imgs:
                            path = os.path.join(app.config['UPLOAD_FOLDER'], rel_path)
                            if path.endswith(('.webp', '.png', '.jpg')) and os.path.exists(path):
                                with open(path, 'rb') as f:
                                    parts.append(types.Part.from_bytes(data=f.read(), mime_type='image/webp'))
                    except: pass
                contents.append(types.Content(role=role, parts=parts))
            curr_parts = [types.Part(text=msg)]
            for fi in loaded_files:
                if fi['text']: curr_parts.append(types.Part(text=f"\n\nFile: {fi['name']}\nContent:\n{fi['text']}"))
                elif fi['bytes']:
                    m_type = fi['mime']; 
                    if not m_type or m_type == 'application/octet-stream': m_type = 'image/webp'
                    curr_parts.append(types.Part.from_bytes(data=fi['bytes'], mime_type=m_type))
            contents.append(types.Content(role='user', parts=curr_parts))
            try:
                stream = local_gemini.models.generate_content_stream(model=real_model, contents=contents, config=config)
                for chunk in stream:
                    if hasattr(chunk, 'candidates') and chunk.candidates:
                        for part in chunk.candidates[0].content.parts:
                            if hasattr(part, 'thought_signature') and part.thought_signature:
                                try: collected_signatures['signature'] = base64.b64encode(part.thought_signature).decode('utf-8')
                                except: pass
                            thought_text_part = None
                            if hasattr(part, 'thought') and part.thought:
                                if isinstance(part.thought, str): thought_text_part = part.thought
                                else: thought_text_part = part.text
                            if thought_text_part:
                                thought_accumulated += thought_text_part
                                yield json.dumps({"type": "thought", "data": thought_text_part}) + "\n"
                            elif part.text:
                                full_res += part.text
                                yield json.dumps({"type": "content", "data": part.text}) + "\n"
                            if hasattr(part, 'inline_data') and part.inline_data:
                                try:
                                    # Save to user dir
                                    user_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
                                    if not os.path.exists(user_upload_dir): os.makedirs(user_upload_dir, exist_ok=True)
                                    
                                    fname_base = f"gen_{int(time.time())}_{len(generated_images)}.png"
                                    save_path = os.path.join(user_upload_dir, fname_base)
                                    
                                    Image.open(BytesIO(part.inline_data.data)).save(save_path)
                                    
                                    db_path = f"{current_user.id}/{fname_base}"
                                    generated_images.append(db_path)
                                    
                                    yield json.dumps({"type": "content", "data": f"\n\n![Img](/static/uploads/{db_path})\n"}) + "\n"
                                except: pass
            except Exception as e:
                logger.error(f"Gemini Error: {e}")
                yield json.dumps({"type": "error", "data": str(e)}) + "\n"
        else:
            client = local_xai if is_grok else local_openai
            is_pdf_handled = False
            target_pdf = next((f for f in loaded_files if f['mime'] == 'application/pdf' and f['path']), None)
            if target_pdf:
                if is_grok and XAIClient:
                    yield json.dumps({"type": "tool_status", "data": "Processing PDF with xAI SDK..."}) + "\n"
                    try:
                        xai_sdk_client = XAIClient(api_key=req_key)
                        uploaded_file = xai_sdk_client.files.upload(target_pdf['path'])
                        chat = xai_sdk_client.chat.create(model=model_key)
                        if sys_instr: chat.append(xai_system(sys_instr))
                        chat.append(xai_user(msg, xai_file(uploaded_file.id)))
                        response = chat.sample()
                        full_res = response.content
                        yield json.dumps({"type": "content", "data": full_res}) + "\n"
                        is_pdf_handled = True
                        try: xai_sdk_client.files.delete(uploaded_file.id)
                        except: pass
                    except Exception as e: yield json.dumps({"type": "tool_status", "data": "Grok PDF failed. Fallback..."}) + "\n"
                elif not is_grok:
                    yield json.dumps({"type": "tool_status", "data": "Analyzing PDF with Assistants..."}) + "\n"
                    try:
                        uf = client.files.create(file=open(target_pdf['path'], "rb"), purpose="assistants")
                        vs = client.beta.vector_stores.create(name=f"VS_{int(time.time())}")
                        client.beta.vector_stores.files.create(vector_store_id=vs.id, file_id=uf.id)
                        asst = client.beta.assistants.create(name="PDF Helper", instructions=sys_instr or "Helpful", model=model_key, tools=[{"type": "file_search"}], tool_resources={"file_search": {"vector_store_ids": [vs.id]}})
                        th = client.beta.threads.create(messages=[{"role": "user", "content": msg, "attachments": [{"file_id": uf.id, "tools": [{"type": "file_search"}]}]}])
                        with client.beta.threads.runs.stream(thread_id=th.id, assistant_id=asst.id) as stream:
                            for event in stream:
                                if event.event == 'thread.message.delta':
                                    c = event.data.delta.content[0].text.value
                                    if c: full_res += c; yield json.dumps({"type": "content", "data": c}) + "\n"
                        is_pdf_handled = True
                        try: client.files.delete(uf.id); client.beta.vector_stores.delete(vs.id); client.beta.assistants.delete(asst.id)
                        except: pass
                    except Exception as e: yield json.dumps({"type": "tool_status", "data": "Assistants failed. Fallback..."}) + "\n"
            if not is_pdf_handled:
                msgs = []
                if sys_instr: msgs.append({"role": "system", "content": sys_instr})
                for m in history: msgs.append({"role": m.role, "content": m.content})
                content_list = []
                user_text = msg
                for fi in loaded_files:
                    if fi['text']: user_text += f"\n\n[File: {fi['name']}]\n```\n{fi['text']}\n```"
                    elif fi['mime'] == 'application/pdf' and fi['path']:
                        try:
                            reader = pypdf.PdfReader(fi['path']); pdf_txt = ""
                            for page in reader.pages: pdf_txt += page.extract_text() + "\n"
                            user_text += f"\n\n[PDF Content: {fi['name']}]\n```\n{pdf_txt[:50000]}\n```"
                        except: pass
                content_list.append({"type": "text", "text": user_text})
                for fi in loaded_files:
                    if fi['bytes'] and fi['mime'] and fi['mime'].startswith('image/'):
                        try:
                            b64 = base64.b64encode(fi['bytes']).decode('utf-8')
                            content_list.append({"type": "image_url", "image_url": {"url": f"data:{fi['mime']};base64,{b64}"}})
                        except: pass
                msgs.append({"role": "user", "content": content_list})
                kwargs = {"model": model_key, "messages": msgs, "stream": True}
                enable_web = enable_search
                if not is_grok and reasoning_effort != 'low': kwargs['reasoning_effort'] = reasoning_effort
                try:
                    stream = None
                    if enable_web:
                        yield json.dumps({"type": "tool_status", "data": f"Searching..."}) + "\n"
                        if is_grok:
                            kwargs["extra_body"] = {"search_parameters": {"mode": "on"}}
                            stream = client.chat.completions.create(**kwargs)
                        else: stream = client.chat.completions.create(**kwargs)
                    else: stream = client.chat.completions.create(**kwargs)
                    has_status = False
                    for chunk in stream:
                        c, t = None, None
                        d = chunk.model_dump() if hasattr(chunk, 'model_dump') else (chunk.to_dict() if hasattr(chunk, 'to_dict') else chunk.__dict__)
                        if 'choices' in d and len(d['choices']) > 0:
                            delta = d['choices'][0].get('delta', {}); c = delta.get('content'); t = delta.get('reasoning_content')
                        elif 'delta' in d:
                            if isinstance(d['delta'], str): c = d['delta']
                            elif isinstance(d['delta'], dict): c = d['delta'].get('content')
                        elif 'output_delta' in d: c = d['output_delta'].get('content')
                        if isinstance(c, list): c = "".join([str(x) for x in c])
                        if c and isinstance(c, str) and c.strip().startswith("{") and "'annotations':" in c:
                             try: c = ast.literal_eval(c).get('text', c)
                             except: pass
                        if enable_web and not has_status and (c or t):
                            has_status = True; yield json.dumps({"type": "tool_status", "data": "<i class='fas fa-check-circle text-green-400'></i> Search Complete"}) + "\n"
                        if t: thought_accumulated += t; yield json.dumps({"type": "thought", "data": t}) + "\n"
                        if c: full_res += c; yield json.dumps({"type": "content", "data": c}) + "\n"
                except Exception as e:
                    logger.error(f"GPT/Grok Error: {e}")
                    yield json.dumps({"type": "error", "data": str(e)}) + "\n"

        if full_res or generated_images:
            t_data = json.dumps({'text': thought_accumulated, 'signatures': collected_signatures}) if (thought_accumulated or collected_signatures) else None
            with app.app_context():
                db.session.add(Message(thread_id=tid, role='assistant', content=full_res, model=model_key, image_url=db_image_val, thought_data=t_data))
                db.session.commit()

    return Response(stream_with_context(gen()), mimetype='application/json')

if __name__ == '__main__': app.run(debug=True)
